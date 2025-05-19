'''
Fine-tuning script for GeoLDM on lipid data with transfection_score conditioning.

'''

import sys
import os
# Add the parent directory (project root) to sys.path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from rdkit import Chem


import copy
import GeoLDM.utils as utils
import argparse
import wandb
from os.path import join
from GeoLDM.core.models import get_optim, get_latent_diffusion 
from GeoLDM.equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle
from GeoLDM.core.utils import prepare_context 

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import logging
import random
from torch.utils.data import DataLoader

import GeoLDM.lipid_dataset as lipid_dataset
from GeoLDM.configs.datasets_config import get_dataset_info

from GeoLDM.core import losses as qm9_losses
from tqdm import tqdm
from collections import Counter
from datetime import datetime
from torch.distributions import Categorical
import torch.nn as nn

from GeoLDM.core.bond_analyze import get_bond_order # allowed_bonds unused
from rdkit.Chem import ChemicalFeatures 
import json
import traceback # Used in train_epoch
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
from SA_Score import sascorer


def get_layer_by_path(model, path):
    """Get a layer by its path in the model."""
    parts = path.split('.')
    current = model
    for part in parts:
        if part in current._modules:
            current = current._modules[part]
        else:
            return None
    return current



# --- Stage 2 Dataset Statistics Calculation ---
def calculate_dataset_statistics_stage2(data_path, device):
    logging.info(f"Calculating Stage 2 dataset statistics from: {data_path}")
    try:
        with open(data_path, "rb") as f:
            raw_data_list = pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading data for statistics calculation: {e}")
        return None, None, None, None

    all_positions = []
    all_h_cat = []
    num_valid_samples = 0
    num_skipped_samples = 0

    for data_dict in tqdm(raw_data_list, desc="Processing samples for Stage 2 stats"):
        try:
            # Convert numpy arrays to PyTorch tensors
            positions = torch.from_numpy(data_dict["positions"]).to(torch.float64)
            one_hot = torch.from_numpy(data_dict["one_hot"]).to(torch.float64)
            atom_mask = torch.from_numpy(data_dict["atom_mask"]).bool()

            # Validate data before processing
            if not torch.isfinite(positions).all() or not torch.isfinite(one_hot).all():
                logging.warning("Skipping sample with NaN/inf in positions or one_hot during stats calculation.")
                num_skipped_samples += 1
                continue
            
            # Mask out padded atoms before collecting for stats
            # atom_mask is (num_nodes, 1), positions is (num_nodes, 3), one_hot is (num_nodes, num_features)
            num_atoms = atom_mask.sum()
            if num_atoms == 0:
                logging.warning("Skipping sample with no atoms after masking.")
                num_skipped_samples += 1
                continue

            masked_positions = positions[atom_mask.squeeze(-1)]
            masked_h_cat = one_hot[atom_mask.squeeze(-1)]
            
            # Additional validation for masked data
            if masked_positions.shape[0] == 0 or masked_h_cat.shape[0] == 0:
                logging.warning("Skipping sample with empty masked data.")
                num_skipped_samples += 1
                continue

            # Check for extreme values that might indicate data issues
            if torch.abs(masked_positions).max() > 1000:  # Arbitrary large value
                logging.warning("Skipping sample with extreme position values.")
                num_skipped_samples += 1
                continue
            
            all_positions.append(masked_positions)
            all_h_cat.append(masked_h_cat)
            num_valid_samples += 1

        except KeyError as e:
            logging.warning(f"Skipping sample due to missing key {e} during stats calculation.")
            num_skipped_samples += 1
            continue
        except Exception as ex:
            logging.warning(f"Skipping sample due to error: {ex} during stats calculation.")
            num_skipped_samples += 1
            continue
            
    if not all_positions or not all_h_cat:
        logging.error("No valid data found to calculate Stage 2 statistics.")
        return None, None, None, None

    logging.info(f"Calculating stats from {num_valid_samples} valid samples (skipped {num_skipped_samples} samples).")

    # Concatenate all valid masked data
    all_positions_tensor = torch.cat(all_positions, dim=0)
    all_h_cat_tensor = torch.cat(all_h_cat, dim=0)

    # Calculate mean and std for positions (x) with robust handling
    dataset_x_mean = all_positions_tensor.mean(dim=0)
    dataset_x_std = all_positions_tensor.std(dim=0)
    
    # Use a more robust epsilon for std normalization
    min_std = 1e-4  # Increased from 1e-6
    dataset_x_std = torch.clamp(dataset_x_std, min=min_std)
    
    # Log statistics for debugging
    logging.info(f"Position statistics:")
    logging.info(f"  Mean range: [{dataset_x_mean.min():.3f}, {dataset_x_mean.max():.3f}]")
    logging.info(f"  Std range: [{dataset_x_std.min():.3f}, {dataset_x_std.max():.3f}]")

    # Calculate mean and std for one-hot categorical features (h_cat)
    dataset_h_cat_mean = all_h_cat_tensor.mean(dim=0)
    dataset_h_cat_std = all_h_cat_tensor.std(dim=0)
    dataset_h_cat_std = torch.clamp(dataset_h_cat_std, min=min_std)
    
    # Log statistics for debugging
    logging.info(f"One-hot statistics:")
    logging.info(f"  Mean range: [{dataset_h_cat_mean.min():.3f}, {dataset_h_cat_mean.max():.3f}]")
    logging.info(f"  Std range: [{dataset_h_cat_std.min():.3f}, {dataset_h_cat_std.max():.3f}]")

    # Convert to float32 for model compatibility
    return (dataset_x_mean.float(), dataset_x_std.float(), 
            dataset_h_cat_mean.float(), dataset_h_cat_std.float())
# --- End Stage 2 Dataset Statistics Calculation ---

# Define arguments specific to fine-tuning or conditioning
parser = argparse.ArgumentParser(description='GeoLDM Lipid Fine-tuning (Stage 2 Multi-Objective)') # Changed description
parser.add_argument('--exp_name', type=str, default='geoldm_lipid_finetune_multiobj') # More specific default
parser.add_argument('--output_dir', type=str, default='outputs/stage2_multiobj_finetune', help='Directory to save outputs and checkpoints') # More specific default

# --- Data Arguments --- 
parser.add_argument('--lipid_data_path', type=str, required=True, help='Path to processed_train_lipids.pkl for Stage 2') # Clarified help
parser.add_argument('--lipid_stats_path', type=str, required=True, help='Path to lipid_stats.pkl (for property normalization)') # Clarified help
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
parser.add_argument('--dataset', type=str, default='geom', help='Base dataset type (used for dataset_info from Stage 1)') # Clarified help
parser.add_argument('--remove_h', action='store_true', help='Use dataset config without hydrogens (ensure model matches Stage 1 config)') # Clarified help

# --- Model Loading Arguments ---
parser.add_argument('--pretrained_path', type=str, required=True, help='Path to Stage 1 checkpoint directory (containing model .npy and args.pickle)') # Clarified help
parser.add_argument('--model_name', type=str, default='stage1_adapted_ema_best.npy', help='Name of the Stage 1 model state_dict file to load') # Changed default

# --- Fine-tuning Arguments --- 
parser.add_argument('--n_epochs', type=int, default=100, help='Number of fine-tuning epochs for Stage 2')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for Stage 2 (smaller for better diversity)')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for Stage 2 fine-tuning (lower for more stable training)')
parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate, 0 to disable (should match Stage 1 if continuing EMA)')
parser.add_argument('--test_epochs', type=int, default=5, help='Run validation every N epochs')
parser.add_argument('--save_model', type=eval, default=True, help='Save checkpoints')
parser.add_argument('--clip_grad', type=eval, default=True, help='Clip gradients during training')
parser.add_argument('--clip_value', type=float, default=0.5, help='Value for gradient clipping (lower for more stable training)')
parser.add_argument('--n_report_steps', type=int, default=50, help='Log training progress every N steps')
parser.add_argument('--ode_regularization', type=float, default=1e-3, help='ODE regularization weight')

# --- Conditioning Arguments --- 
parser.add_argument('--conditioning', nargs='+', default=['transfection_score'], help='Conditioning property key(s) from lipid_dataset') # Clarified help
parser.add_argument('--cfg_prob', type=float, default=0.1, help='Probability of using unconditional context during training for CFG (0.0 to disable)')

# --- Biodegradability related arguments ---
parser.add_argument('--biodegradability_loss_weight', type=float, default=0.05, help='Weight for biodegradability loss term')
# Optional: parser.add_argument('--target_min_biodeg_score', type=float, default=1.0)

# --- Toxicity related arguments ---
parser.add_argument('--toxicity_loss_weight', type=float, default=0.1, help='Weight for toxicity loss term')
# Optional: parser.add_argument('--target_max_toxicity_penalty', type=float, default=1.0)

# --- Synthesizability related arguments ---
parser.add_argument('--synthesizability_loss_weight', type=float, default=0.05, help='Weight for synthesizability loss term')
parser.add_argument('--target_max_sa_score', type=float, default=5.0, help='Target max SAscore (lower is better)')

# --- Other auxiliary loss arguments ---
parser.add_argument('--validity_loss_weight', type=float, default=0.1, help='Weight for validity loss term') # Renamed from validity_weight
parser.add_argument('--novelty_loss_weight', type=float, default=0.1, help='Weight for novelty loss term') # Renamed from novelty_weight
parser.add_argument('--aux_loss_frequency', type=int, default=5, help='Frequency (in batches) to calculate auxiliary losses')

# --- Diffusion Arguments (will be loaded from Stage 1 args.pickle) --- 
# Note: These are intentionally left as defaults - actual values will be loaded from Stage 1 args.pickle
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2')
parser.add_argument('--diffusion_loss_type', type=str, default='l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

# --- Model Architecture Arguments (will be loaded from Stage 1 args.pickle) --- 
# Note: These are intentionally left as defaults - actual values will be loaded from Stage 1 args.pickle
parser.add_argument('--model', type=str, default='egnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion', help='Type of probabilistic model (diffusion)')
parser.add_argument('--n_layers', type=int, default=4) # Crucial for GCL unfreezing logic
parser.add_argument('--nf', type=int, default=256)
parser.add_argument('--latent_nf', type=int, default=3)
parser.add_argument('--tanh', type=eval, default=True)
parser.add_argument('--attention', type=eval, default=True)
parser.add_argument('--norm_constant', type=float, default=1)
parser.add_argument('--inv_sublayers', type=int, default=1)
parser.add_argument('--sin_embedding', type=eval, default=False)
parser.add_argument('--include_charges', type=bool, default=True,
                  help="Include charges in the model")
parser.add_argument('--normalization_factor', type=float, default=1.0)
parser.add_argument('--aggregation_method', type=str, default='sum')
parser.add_argument('--kl_weight', type=float)
parser.add_argument('--train_diffusion', action='store_true')
parser.add_argument('--trainable_ae', action='store_true')
parser.add_argument('--normalize_factors', type=eval, help='normalize factors for [x, categorical, integer] - will be set to [1,1,1] for external norm')

# --- Other Arguments --- 
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
parser.add_argument('--wandb_usr', type=str, default=None, help='WandB username')
parser.add_argument('--wandb_project_name', type=str, default='e3_diffusion_lipid_finetune_stage2', help='WandB project name for Stage 2') # New arg for specific project name
parser.add_argument('--online', type=bool, default=True, help='WandB online mode')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

# --- Add missing arguments to the global parser that are used in main() --- 
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help='Device to use for training (e.g., cpu, cuda, cuda:0)')

# --- Training loop arguments (from parse_args function, critical for main/train_epoch) ---
parser.add_argument('--validation_interval', type=int, default=5,
                    help='Run validation every N epochs')
parser.add_argument('--save_interval', type=int, default=10,
                    help='Save checkpoint every N epochs')
parser.add_argument('--log_interval', type=int, default=10, 
                    help='Log training progress every N batches')

# --- Anti-overfitting Arguments ---
parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'reduce_on_plateau', 'none'],
                    help='Learning rate scheduler type. Cosine annealing is recommended for molecular fine-tuning.')
parser.add_argument('--lr_patience', type=int, default=5,
                    help='Patience for ReduceLROnPlateau scheduler. Number of epochs to wait before reducing LR.')
parser.add_argument('--lr_factor', type=float, default=0.5,
                    help='Factor to reduce learning rate by when using ReduceLROnPlateau.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 regularization) coefficient (higher for better generalization)')
parser.add_argument('--early_stopping_patience', type=int, default=10,
                    help='Number of epochs to wait before early stopping if validation NLL does not improve.')
parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4,
                    help='Minimum change in validation NLL to be considered as improvement for early stopping.')

# --- Update the argument defaults for better diversity control
parser.add_argument('--diversity_beta', type=float, default=0.3,
                    help='Weight for diversity penalty in loss function (higher = more diverse from training data)')
parser.add_argument('--internal_diversity_weight', type=float, default=0.2,
                    help='Weight for internal diversity loss (higher = more diverse within batches)')
parser.add_argument('--diversity_threshold', type=float, default=0.6,
                    help='Maximum allowed similarity between generated and training molecules (lower = more novel)')

# --- Add new arguments for better generation control
parser.add_argument('--validity_weight', type=float, default=0.1,
                    help='Weight for validity loss term (higher = more emphasis on valid molecules)')
parser.add_argument('--novelty_weight', type=float, default=0.15,
                    help='Weight for novelty loss term (higher = more emphasis on novel structures)')

# --- Add new arguments for augmentation
parser.add_argument('--augmentation_prob', type=float, default=0.2,
                    help='Probability of applying data augmentation (conservative default)')
parser.add_argument('--max_bond_length', type=float, default=1.8,
                    help='Maximum allowed bond length in Angstroms (conservative default)')
parser.add_argument('--min_bond_length', type=float, default=0.8,
                    help='Minimum allowed bond length in Angstroms (conservative default)')
parser.add_argument('--min_atom_distance', type=float, default=1.8,
                    help='Minimum allowed distance between non-bonded atoms (conservative default)')

# --- Argument Parsing and Setup ---
args = parser.parse_args()

# Setup Logging
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
logging.info("GeoLDM Stage 2 Fine-tuning Script Initialized")

# --- Device, Seed, Output Dirs ---
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
torch.manual_seed(args.seed); np.random.seed(args.seed)
if args.cuda: torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # Ensure reproducibility
args.output_dir = join(args.output_dir, args.exp_name); utils.create_folders(args)


# --- Get Dataset Info (Consistent with Stage 1) ---
dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h) # From original script
logging.info(f"Using dataset info for: {args.dataset} (remove_h={args.remove_h}) Stage 2")

# --- Wandb Setup ---
# Initialize wandb if not disabled
if not args.no_wandb:
    # Create unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"{args.exp_name}_{timestamp}"
    
    wandb.init(
        project=args.wandb_project_name,
        config=vars(args),
        name=unique_run_name,
        entity=args.wandb_usr,
        mode="online" if args.online else "offline"
    )
    logging.info(f"Initialized wandb with project: {args.wandb_project_name}, run: {unique_run_name}")

logging.info("Effective Arguments for Stage 2 Fine-tuning:")
for k, v in vars(args).items(): logging.info(f"  {k}: {v}")

# --- Load Stage 2 Data ---
print("get_dataloaders function:", lipid_dataset.get_dataloaders)
dataloaders = lipid_dataset.get_dataloaders(
    args.lipid_data_path,           # positional
    args.batch_size,                # positional
    num_workers=args.num_workers,   # keyword
    seed=args.seed,
    val_split_ratio=0.1,
    lipid_stats_path=args.lipid_stats_path,
    is_stage2_data=True
)
if not dataloaders['train'] or not dataloaders['val']:
    logging.error("Failed to load Stage 2 train/val dataloaders. Exiting."); exit(1)

# --- Stage 2 Data Normalization Setup ---
logging.info("Setting up Stage 2 data normalization.")
# Force disable model's internal normalization if we are doing external dataset-level normalization
args.normalize_factors = [1.0, 1.0, 1.0] 
logging.info(f"args.normalize_factors set to: {args.normalize_factors}")

# Calculate and store Stage 2 dataset-specific normalization statistics
# These will be used in the training/validation loops
# We use the training data specified by --lipid_data_path for these stats
args.stage2_dataset_x_mean, args.stage2_dataset_x_std, \
args.stage2_dataset_h_cat_mean, args.stage2_dataset_h_cat_std = calculate_dataset_statistics_stage2(args.lipid_data_path, device="cpu") # Calculate on CPU

if args.stage2_dataset_x_mean is None:
    logging.error("Failed to calculate Stage 2 dataset statistics. Exiting.")
    sys.exit(1)

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
args.checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
os.makedirs(args.checkpoints_dir, exist_ok=True)

# Save arguments after potentially adding new ones like dataset stats
args_path = join(args.output_dir, "args.pickle")
with open(args_path, "wb") as f:
    pickle.dump(args, f)
logging.info(f"Full arguments saved to {args_path}")

# --- Prepare Conditioning Norms (Property Norms) ---
logging.info(f"Loading property norms from {args.lipid_stats_path}")
try:
    with open(args.lipid_stats_path, "rb") as f:
        lipid_stats = pickle.load(f)
        logging.info(f"Loaded lipid stats keys: {list(lipid_stats.keys())}")
        
        # Extract property norms from the stats dictionary
        property_norms = {}
        for prop in args.conditioning:
            # For this dataset, we have a single property (transfection_score)
            # with direct mean and std values
            property_norms[prop] = {
                'mean': float(lipid_stats['mean']),
                'std': float(lipid_stats['std'])  
            }
        
        logging.info(f"Loaded property norms: {property_norms}")
except Exception as e:
    logging.error(f"Failed to load property norms from {args.lipid_stats_path}: {e}")
    logging.error(f"Full lipid stats content: {lipid_stats if 'lipid_stats' in locals() else 'Not loaded'}")
    sys.exit(1)


args.property_norms_for_lsuv = property_norms

# Set context_node_nf for conditioning
args.context_node_nf = 1  # One feature for transfection_score
logging.info(f"Set context_node_nf to {args.context_node_nf} for conditioning")

# --- EMA Class Definition ---
class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone().detach()

    def update_model_average(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone().detach()

    def apply_shadow(self):
        for name, param in self.shadow.items():
            self.original[name] = param.clone().detach()
            param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.original:
                param.data = self.original[name]

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

class LipidDataAugmenter:
    """Class to handle data augmentation for lipid molecules."""

    @staticmethod
    def augment(data_dict, dataset_info): # dataset_info might not be strictly needed if augmentations are purely geometric
        """Apply data augmentation to molecule data."""
        try:
            augmented_data = copy.deepcopy(data_dict)
            positions = augmented_data["positions"].cpu().numpy()

            if random.random() < 0.5: # Example: 50% chance to apply rotation
                positions = LipidDataAugmenter._apply_rotation(positions)

            positions = LipidDataAugmenter._apply_perturbation(positions) # Always apply small perturbation

            augmented_data["positions"] = torch.from_numpy(positions).to(data_dict["positions"].device)
            return augmented_data
        except Exception as e:
            logging.warning(f"Error in LipidDataAugmenter.augment: {e}")
            return data_dict # Return original data if augmentation fails

    @staticmethod
    def _apply_rotation(positions):
        """Apply random 3D rotation to positions."""
        # Generate a random rotation matrix (e.g., using Euler angles or random axis-angle)
        # For simplicity, a random rotation around Z-axis, then Y, then X
        theta_z = random.uniform(0, 2 * np.pi)
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                       [np.sin(theta_z), np.cos(theta_z), 0],
                       [0, 0, 1]])
        theta_y = random.uniform(0, 2 * np.pi)
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                       [0, 1, 0],
                       [-np.sin(theta_y), 0, np.cos(theta_y)]])
        theta_x = random.uniform(0, 2 * np.pi)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta_x), -np.sin(theta_x)],
                       [0, np.sin(theta_x), np.cos(theta_x)]])
        
        rotation_matrix = Rz @ Ry @ Rx # Combined rotation
        return np.dot(positions, rotation_matrix.T) # Apply rotation (transpose for row vectors)

    @staticmethod
    def _apply_perturbation(positions, noise_scale=0.05): # Reduced default noise_scale
        """Apply small random perturbations to positions."""
        noise = np.random.normal(0, noise_scale, positions.shape)
        return positions + noise


# --- Create Model (Load from Stage 1) ---
logging.info(f"Instantiating model (train_diffusion={args.train_diffusion}, trainable_ae={args.trainable_ae})...")
model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
logging.info("Model structure instantiated.")

# Update nodes_dist with Stage 2 dataset node counts
logging.info("Updating nodes distribution with Stage 2 dataset node counts...")
stage2_node_counts = set()
for data in dataloaders['train'].dataset.data['num_atoms']:
    stage2_node_counts.add(data.item())
for data in dataloaders['val'].dataset.data['num_atoms']:
    stage2_node_counts.add(data.item())

logging.info(f"Found node counts in Stage 2 dataset: {sorted(list(stage2_node_counts))}")

# Update nodes_dist with new counts
for n_nodes in stage2_node_counts:
    if n_nodes not in nodes_dist.keys:
        nodes_dist.keys[n_nodes] = len(nodes_dist.keys)
        nodes_dist.prob = torch.cat([nodes_dist.prob, torch.tensor([0.0])])  # Add zero probability for new count
        nodes_dist.prob = nodes_dist.prob / nodes_dist.prob.sum()  # Renormalize
        # Update the Categorical distribution
        nodes_dist.m = Categorical(nodes_dist.prob)

logging.info(f"Updated nodes distribution with counts: {sorted(list(nodes_dist.keys.keys()))}")

# --- Load Pretrained Weights (from Stage 1) ---
if args.pretrained_path is not None:
    model_path = os.path.join(args.pretrained_path, args.model_name)
    if os.path.exists(model_path):
        try:
            logging.info(f"Loading Stage 1 model weights from: {model_path}")
            model_state_dict = torch.load(model_path, map_location=device)
            
            # Handle potential DataParallel prefix if saved with it
            if any(key.startswith("module.") for key in model_state_dict.keys()):
                logging.info("  Detected 'module.' prefix in checkpoint keys. Adjusting for non-DataParallel model.")
                model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

            # Load with strict=False to allow for fine-tuning specific parts if needed,
            # though for full diffusion model training, this is less critical.
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                logging.warning(f"  Missing keys during Stage 1 weight loading: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"  Unexpected keys during Stage 1 weight loading: {unexpected_keys}")
            logging.info("  Successfully loaded Stage 1 model weights.")
        except Exception as e:
            logging.error(f"  Error loading Stage 1 model weights from {model_path}: {e}", exc_info=True)
            logging.warning("  Proceeding with randomly initialized model due to loading error.")
    else:
        logging.warning(f"Stage 1 model path not found: {model_path}. Proceeding with randomly initialized model.")
else:
    logging.warning("No pretrained_path provided for Stage 1 model. Proceeding with randomly initialized model.")


# --- Configure Layers for Stage 2 ---
logging.info("Configuring model layers for Stage 2 fine-tuning...")
# VAE freezing is controlled by args.trainable_ae
if hasattr(model, 'vae'):
    if not args.trainable_ae:
        logging.info("  Freezing VAE parameters as per args.trainable_ae=False.")
        for param in model.vae.parameters():
            param.requires_grad = False
    else:
        logging.info("  VAE parameters will be trainable as per args.trainable_ae=True.")
        for param in model.vae.parameters():
            param.requires_grad = True
else:
    logging.info("  model.vae not found, skipping VAE configuration.")

# Diffusion model (dynamics) parameters are intended to be trained.
if hasattr(model, 'dynamics'):
    logging.info("  Setting all model.dynamics parameters to be trainable for Stage 2.")
    for name,param in model.dynamics.named_parameters():
        param.requires_grad = True
else:
    logging.warning("model.dynamics not found. Cannot configure dynamics layers for training.")


logging.info("Layers configured. Proceeding with existing weights.")

# --- EMA Model Setup (if used) ---
ema = None
if args.ema_decay > 0:
    ema = EMA(args.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    logging.info(f"EMA initialized with decay {args.ema_decay}.")

# --- Grad Norm Queue for Logging ---
gradnorm_queue = utils.Queue()
gradnorm_queue.add(100) 

# --- train_epoch definition ---
def train_epoch(model, optimizer, dataloader, device, epoch, args, 
                ema_object, 
                validator_instance, 
                dataset_info_for_train,
                lipid_property_assessor_instance
                ):
    model.train()
    total_loss, total_nll, total_ode_reg = 0, 0, 0
    total_validity_loss, total_novelty_loss = 0,0 
    total_biodegradability_loss, total_toxicity_loss, total_synthesizability_loss = 0,0,0
    processed_batches = 0
    num_aux_loss_calcs = 0 
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if dataset_info_for_train is None:
        logging.error("CRITICAL: dataset_info_for_train not provided to train_epoch.")
        try:
            dataset_info_for_train = get_dataset_info(args.dataset, remove_h=args.remove_h)
            logging.info("train_epoch: Used fallback to get_dataset_info.")
        except Exception as e_di:
            logging.error(f"train_epoch: Fallback get_dataset_info failed: {e_di}. Critical error.")
            return {'avg_loss': float('inf')} # Return early with error metric

    for batch_idx, batch_data in enumerate(dataloader):
        try:
            if random.random() < args.augmentation_prob and hasattr(LipidDataAugmenter, 'augment'):
                batch_data = LipidDataAugmenter.augment(batch_data, dataset_info_for_train) 

            positions = batch_data['positions'].to(device, dtype=torch.float32)
            one_hot = batch_data['one_hot'].to(device, dtype=torch.float32) # Original shape (B, N, 16)
            charges = batch_data['charges'].to(device, dtype=torch.float32)
            atom_mask = batch_data['atom_mask'].to(device, dtype=torch.float32)
            edge_mask = batch_data['edge_mask'].to(device, dtype=torch.float32)




            if hasattr(args, 'stage2_dataset_x_mean') and args.stage2_dataset_x_mean is not None:
                x_mean = args.stage2_dataset_x_mean.to(device).unsqueeze(0).unsqueeze(0) 
                x_std = args.stage2_dataset_x_std.to(device).unsqueeze(0).unsqueeze(0)
                h_cat_mean = args.stage2_dataset_h_cat_mean.to(device).unsqueeze(0).unsqueeze(0)
                h_cat_std = args.stage2_dataset_h_cat_std.to(device).unsqueeze(0).unsqueeze(0)

                # Pad one_hot from 16 to 18 features 
                # The loaded model's egnn.embedding layer expects 18 input features for atom types (h['categorical']).
                # Our preprocessed one_hot encoding has 16 features.
                if one_hot.shape[-1] == 16:
                    padding_size = 18 - one_hot.shape[-1] # Should be 2
                    if padding_size > 0:
                        # Pad one_hot
                        padding_shape = list(one_hot.shape[:-1]) + [padding_size] # (B, N, padding_size)
                        padding = torch.zeros(padding_shape, device=device, dtype=one_hot.dtype)
                        one_hot = torch.cat([one_hot, padding], dim=-1) # Shape (B, N, 18)
                        
                        # Pad h_cat_mean (with zeros)
                        mean_padding_shape = list(h_cat_mean.shape[:-1]) + [padding_size]
                        mean_padding = torch.zeros(mean_padding_shape, device=device, dtype=h_cat_mean.dtype)
                        h_cat_mean = torch.cat([h_cat_mean, mean_padding], dim=-1)
                        
                        # Pad h_cat_std (with ones to avoid division by zero)
                        std_padding_shape = list(h_cat_std.shape[:-1]) + [padding_size]
                        std_padding = torch.ones(std_padding_shape, device=device, dtype=h_cat_std.dtype)
                        h_cat_std = torch.cat([h_cat_std, std_padding], dim=-1)

                positions = (positions - x_mean) / (x_std + 1e-6)
                one_hot = (one_hot - h_cat_mean) / (h_cat_std + 1e-6)
            else:
                logging.warning("Stage 2 dataset statistics not found for training normalization.")

            positions = positions * atom_mask
            one_hot = one_hot * atom_mask
            charges = charges * atom_mask
            x_centered = diffusion_utils.remove_mean_with_mask(positions, atom_mask)
            h_dict = {'categorical': one_hot, 'integer': charges}
            
            optimizer.zero_grad()
            context = None
            if args.conditioning and hasattr(args, 'property_norms') and args.property_norms is not None: 
                 context = prepare_context(args.conditioning, batch_data, args.property_norms, device)
            
            loss_output = model(x=x_centered, h=h_dict, node_mask=atom_mask, edge_mask=edge_mask, context=context)
            nll_loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
            if nll_loss.ndim > 0: nll_loss = nll_loss.mean()

            ode_reg_term = torch.tensor(0.0, device=device)
            if args.ode_regularization > 0 and hasattr(model, 'dynamics') and \
               hasattr(model.dynamics, 'dx_mean') and model.dynamics.dx_mean is not None and \
               hasattr(model.dynamics, 'dx_T') and model.dynamics.dx_T is not None:
                try:
                    if isinstance(model.dynamics.dx_mean, torch.Tensor) and isinstance(model.dynamics.dx_T, torch.Tensor):
                        ode_reg_term = (model.dynamics.dx_mean - model.dynamics.dx_T).pow(2).sum(dim=-1).mean() * args.ode_regularization
                except Exception as e_ode: 
                    logging.debug(f"Epoch {epoch}, Batch {batch_idx}: Error ODE reg: {e_ode}")
            
            current_loss = nll_loss + ode_reg_term
            validity_loss_term, novelty_loss_term, biodegradability_penalty_term, toxicity_penalty_term, synthesizability_penalty_term = (torch.tensor(0.0, device=device) for _ in range(5))

            if batch_idx > 0 and batch_idx % args.aux_loss_frequency == 0 and dataset_info_for_train is not None and lipid_property_assessor_instance is not None:
                num_aux_loss_calcs += 1
                with torch.no_grad():
                    num_samples_for_aux_cfg = min(args.batch_size // 2, 8) or 1
                    current_n_nodes = x_centered.size(1)
                    num_samples_for_aux_effective = min(atom_mask.size(0), num_samples_for_aux_cfg)
                    
                    processed_sampled_x_denorm, processed_sampled_h_cat_denorm = None, None
                    actual_num_samples_generated = 0

                    if num_samples_for_aux_effective > 0:
                        current_node_mask_for_sampling = atom_mask[:num_samples_for_aux_effective]
                        context_for_sampling = None
                        if args.conditioning and 'transfection_score' in args.conditioning and context is not None:
                            context_for_sampling = context[:num_samples_for_aux_effective]
                        
                        try:
                            sampled_x_aux, sampled_h_aux = model.sample(
                                num_samples=num_samples_for_aux_effective, n_nodes=current_n_nodes, 
                                node_mask=current_node_mask_for_sampling, context=context_for_sampling,
                                device=device, dataset_info=dataset_info_for_train
                            )
                            if hasattr(args, 'stage2_dataset_x_mean') and args.stage2_dataset_x_mean is not None:
                                x_m, x_s = args.stage2_dataset_x_mean.to(device).unsqueeze(0).unsqueeze(0), args.stage2_dataset_x_std.to(device).unsqueeze(0).unsqueeze(0)
                                h_m, h_s = args.stage2_dataset_h_cat_mean.to(device).unsqueeze(0).unsqueeze(0), args.stage2_dataset_h_cat_std.to(device).unsqueeze(0).unsqueeze(0)
                                processed_sampled_x_denorm = sampled_x_aux * (x_s + 1e-6) + x_m
                                processed_sampled_h_cat_denorm = sampled_h_aux['categorical'] * (h_s + 1e-6) + h_m
                            else:
                                processed_sampled_x_denorm, processed_sampled_h_cat_denorm = sampled_x_aux, sampled_h_aux['categorical']
                            actual_num_samples_generated = num_samples_for_aux_effective
                        except Exception as e_sampling:
                            logging.error(f"Error sampling/denorm in aux: {e_sampling}", exc_info=True)
                
                if actual_num_samples_generated > 0 and processed_sampled_x_denorm is not None:
                    generated_rdkit_molecules, bio_scores, tox_penalties, sa_scores = [], [], [], []
                    x_cpu, h_cpu = processed_sampled_x_denorm.cpu().numpy(), processed_sampled_h_cat_denorm.cpu().numpy()

                    for i in range(actual_num_samples_generated):
                        if 'atom_decoder' not in dataset_info_for_train: break
                        atom_indices = h_cpu[i].argmax(axis=-1)
                        atom_types = [dataset_info_for_train['atom_decoder'][j] for j in atom_indices]
                        mol = build_molecule(x_cpu[i], atom_types, dataset_info_for_train)
                        if mol:
                            generated_rdkit_molecules.append(mol)
                            bio_scores.append(torch.tensor(float(lipid_property_assessor_instance.assess_biodegradability_score(mol)), device=device))
                            tox_penalties.append(torch.tensor(float(lipid_property_assessor_instance.assess_toxicity_score(mol)), device=device))
                            try: sa_scores.append(torch.tensor(float(SAscore.calculatescore(mol)), device=device))
                            except: sa_scores.append(torch.tensor(10.0, device=device))
                        else:
                            bio_scores.append(torch.tensor(0.0, device=device)); tox_penalties.append(torch.tensor(10.0, device=device)); sa_scores.append(torch.tensor(10.0, device=device))
                    
                    if bio_scores and args.biodegradability_loss_weight > 0: 
                        biodegradability_penalty_term = (-torch.stack(bio_scores).mean()) * args.biodegradability_loss_weight
                    if tox_penalties and args.toxicity_loss_weight > 0: 
                        toxicity_penalty_term = torch.stack(tox_penalties).mean() * args.toxicity_loss_weight
                    if sa_scores and args.synthesizability_loss_weight > 0: 
                        synthesizability_penalty_term = torch.relu(torch.stack(sa_scores) - args.target_max_sa_score).mean() * args.synthesizability_loss_weight
                    if generated_rdkit_molecules and validator_instance:
                        if args.validity_loss_weight > 0:
                            inv_count = sum(1 for m in generated_rdkit_molecules if not validator_instance.is_valid(m))
                            validity_loss_term = torch.tensor(inv_count / len(generated_rdkit_molecules), device=device) * args.validity_loss_weight
                        if args.novelty_loss_weight > 0 and len(generated_rdkit_molecules) > 1:
                            div_score = LipidDiversityCalculator.compute_internal_diversity(generated_rdkit_molecules)
                            novelty_loss_term = (1.0 - torch.tensor(div_score, device=device)) * args.novelty_loss_weight
            
            current_loss += validity_loss_term + 0.5 * novelty_loss_term + 0.8 *biodegradability_penalty_term + toxicity_penalty_term + synthesizability_penalty_term
            
            if torch.isnan(current_loss) or torch.isinf(current_loss):
                logging.warning(f"NaN/Inf loss in batch {batch_idx}. Reverting to base loss.")
                current_loss = nll_loss + ode_reg_term
                validity_loss_term, novelty_loss_term, biodegradability_penalty_term, toxicity_penalty_term, synthesizability_penalty_term = (torch.tensor(0.0, device=device) for _ in range(5))
            else:
                total_validity_loss += validity_loss_term.item(); total_novelty_loss += novelty_loss_term.item()
                total_biodegradability_loss += biodegradability_penalty_term.item(); total_toxicity_loss += toxicity_penalty_term.item()
                total_synthesizability_loss += synthesizability_penalty_term.item()

            if torch.isnan(current_loss) or torch.isinf(current_loss):
                logging.error(f"NaN/Inf loss before backward in batch {batch_idx}. Skipping update.")
                optimizer.zero_grad(); continue
            
            current_loss.backward()
            grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value).item() if args.clip_grad else 0.0
            optimizer.step()
            if ema_object: 
                ema_object.update_model_average(model)
            
            total_loss += current_loss.item(); total_nll += nll_loss.item(); total_ode_reg += ode_reg_term.item()
            processed_batches += 1
            
            if batch_idx % args.log_interval == 0:
                log_msg = (f'Train Ep {epoch} [{batch_idx}/{len(dataloader)}] Loss: {current_loss.item():.4f} '
                           f'NLL: {nll_loss.item():.4f} ODE: {ode_reg_term.item():.4f} ValL: {validity_loss_term.item():.4f} '
                           f'NovL: {novelty_loss_term.item():.4f} BioL: {biodegradability_penalty_term.item():.4f} '
                           f'ToxL: {toxicity_penalty_term.item():.4f} SyntL: {synthesizability_penalty_term.item():.4f}')
                logging.info(log_msg)
                if not args.no_wandb and wandb.run:
                    wandb.log({'train_batch_loss': current_loss.item(), 'train_batch_nll': nll_loss.item(),
                               'train_batch_ode_reg': ode_reg_term.item(), 'grad_norm': grad_norm_val,
                               'train_batch_validity_loss': validity_loss_term.item(), 'train_batch_novelty_loss': novelty_loss_term.item(),
                               'train_batch_biodegradability_loss': biodegradability_penalty_term.item(), 
                               'train_batch_toxicity_loss': toxicity_penalty_term.item(),
                               'train_batch_synthesizability_loss': synthesizability_penalty_term.item()}, commit=True)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e): 
                logging.error(f"OOM in train batch {batch_idx}. Skipping.")
            else: 
                logging.error(f"Runtime error in train batch {batch_idx}: {e}", exc_info=True)
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
            if not args.no_wandb and wandb.run: 
                wandb.log({"training_error_batch": str(e)})
            optimizer.zero_grad()
            continue
        except Exception as e:
            logging.error(f"Error in train batch {batch_idx}: {e}", exc_info=True)
            if not args.no_wandb and wandb.run: 
                wandb.log({"training_error_batch": str(e)})
            optimizer.zero_grad()
            continue

    avg_loss = total_loss / processed_batches if processed_batches > 0 else float('inf')
    avg_nll = total_nll / processed_batches if processed_batches > 0 else float('inf')
    avg_ode_reg = total_ode_reg / processed_batches if processed_batches > 0 else float('inf')
    avg_validity_loss = total_validity_loss / num_aux_loss_calcs if num_aux_loss_calcs > 0 else 0
    avg_novelty_loss = total_novelty_loss / num_aux_loss_calcs if num_aux_loss_calcs > 0 else 0
    avg_biodegradability_loss = total_biodegradability_loss / num_aux_loss_calcs if num_aux_loss_calcs > 0 else 0
    avg_toxicity_loss = total_toxicity_loss / num_aux_loss_calcs if num_aux_loss_calcs > 0 else 0
    avg_synthesizability_loss = total_synthesizability_loss / num_aux_loss_calcs if num_aux_loss_calcs > 0 else 0

    epoch_metrics = {'avg_loss': avg_loss, 'avg_nll': avg_nll, 'avg_ode_reg': avg_ode_reg,
                     'avg_validity_loss': avg_validity_loss, 'avg_novelty_loss': avg_novelty_loss,
                     'avg_biodegradability_loss': avg_biodegradability_loss, 
                     'avg_toxicity_loss': avg_toxicity_loss, 'avg_synthesizability_loss': avg_synthesizability_loss}
    logging.info(f'====> Epoch: {epoch} Avg Loss: {avg_loss:.4f} (NLL: {avg_nll:.4f} ODE: {avg_ode_reg:.4f})')
    logging.info(f'====> Aux Losses (Avg): Val: {avg_validity_loss:.4f}, Nov: {avg_novelty_loss:.4f}, Bio: {avg_biodegradability_loss:.4f}, Tox: {avg_toxicity_loss:.4f}, Synt: {avg_synthesizability_loss:.4f}')
    return epoch_metrics

# --- validate definition ---
def validate(model, dataloader, device, dataset_info_val, args, validator_instance, epoch_num):
    model.eval()
    total_val_loss = 0
    valid_molecules_generated_overall = []
    num_valid_by_validator = 0
    batch_metrics_list = []
    processed_batches = 0

    if dataset_info_val is None:
        logging.error("CRITICAL: dataset_info_val not for validate function.")
        try:
            dataset_info_val = get_dataset_info(args.dataset, remove_h=args.remove_h)
            logging.info("validate: Used fallback to get_dataset_info.")
        except Exception as e_di_val:
            logging.error(f"validate: Fallback get_dataset_info failed: {e_di_val}.")
            # Cannot generate molecules for metrics if this fails and is needed.
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            try:
                positions = batch_data['positions'].to(device, dtype=torch.float32)
                one_hot = batch_data['one_hot'].to(device, dtype=torch.float32) # Original shape (B, N, 16)
                charges = batch_data['charges'].to(device, dtype=torch.float32)
                atom_mask = batch_data['atom_mask'].to(device, dtype=torch.float32)
                edge_mask = batch_data['edge_mask'].to(device, dtype=torch.float32)

                if hasattr(args, 'stage2_dataset_x_mean') and args.stage2_dataset_x_mean is not None:
                    x_mean, x_std = args.stage2_dataset_x_mean.to(device).unsqueeze(0).unsqueeze(0), args.stage2_dataset_x_std.to(device).unsqueeze(0).unsqueeze(0)
                    h_cat_mean, h_cat_std = args.stage2_dataset_h_cat_mean.to(device).unsqueeze(0).unsqueeze(0), args.stage2_dataset_h_cat_std.to(device).unsqueeze(0).unsqueeze(0)
                    
                    # === PATCH: Pad h_cat_mean and h_cat_std if one_hot was padded ===
                    if one_hot.shape[-1] == 18 and h_cat_mean.shape[-1] == 16:
                        padding_size_stats = 18 - h_cat_mean.shape[-1] # Should be 2
                        
                        # Padding for h_cat_mean (zeros)
                        mean_padding_shape = list(h_cat_mean.shape[:-1]) + [padding_size_stats]
                        mean_padding = torch.zeros(mean_padding_shape, device=device, dtype=h_cat_mean.dtype)
                        h_cat_mean = torch.cat([h_cat_mean, mean_padding], dim=-1)
                        
                        # Padding for h_cat_std (ones)
                        std_padding_shape = list(h_cat_std.shape[:-1]) + [padding_size_stats]
                        std_padding = torch.ones(std_padding_shape, device=device, dtype=h_cat_std.dtype)
                        h_cat_std = torch.cat([h_cat_std, std_padding], dim=-1)
                    # === END PATCH ===

                    positions = (positions - x_mean) / (x_std + 1e-6)
                    one_hot = (one_hot - h_cat_mean) / (h_cat_std + 1e-6)
                else:
                    logging.warning("Stage 2 dataset statistics not found for training normalization.")

                positions = positions * atom_mask; one_hot = one_hot * atom_mask; charges = charges * atom_mask
                x_centered = diffusion_utils.remove_mean_with_mask(positions, atom_mask)
                h_dict = {'categorical': one_hot, 'integer': charges}
                context = prepare_context(args.conditioning, batch_data, args.property_norms, device) if args.conditioning and hasattr(args, 'property_norms') and args.property_norms is not None else None
                
                loss_output = model(x=x_centered, h=h_dict, node_mask=atom_mask, edge_mask=edge_mask, context=context)
                val_batch_loss = (loss_output[0] if isinstance(loss_output, tuple) else loss_output).mean()
                total_val_loss += val_batch_loss.item()
                batch_metrics_list.append({'loss': val_batch_loss.item()})
                processed_batches += 1
                
                if dataset_info_val and 'atom_decoder' in dataset_info_val and batch_idx % 5 == 0:
                    num_mols_to_sample_val = min(5, positions.size(0))
                    if num_mols_to_sample_val == 0: 
                        continue

                    node_mask_sampling = atom_mask[:num_mols_to_sample_val]
                    context_sampling = context[:num_mols_to_sample_val] if context is not None and context.size(0) >= num_mols_to_sample_val else None
                    
                    sampled_x, sampled_h = model.sample(
                        num_samples=num_mols_to_sample_val, n_nodes=positions.size(1),
                        node_mask=node_mask_sampling, context=context_sampling,
                        device=device, dataset_info=dataset_info_val
                    )
                    if hasattr(args, 'stage2_dataset_x_mean') and args.stage2_dataset_x_mean is not None:
                        x_m, x_s = args.stage2_dataset_x_mean.to(device).unsqueeze(0).unsqueeze(0), args.stage2_dataset_x_std.to(device).unsqueeze(0).unsqueeze(0)
                        h_m, h_s = args.stage2_dataset_h_cat_mean.to(device).unsqueeze(0).unsqueeze(0), args.stage2_dataset_h_cat_std.to(device).unsqueeze(0).unsqueeze(0)
                        sampled_x_denorm, sampled_h_cat_denorm = sampled_x * (x_s + 1e-6) + x_m, sampled_h['categorical'] * (h_s + 1e-6) + h_m
                    else:
                        sampled_x_denorm, sampled_h_cat_denorm = sampled_x, sampled_h['categorical']
                    
                    x_cpu_val, h_cpu_val = sampled_x_denorm.cpu().numpy(), sampled_h_cat_denorm.cpu().numpy()
                    for i in range(sampled_x_denorm.size(0)):
                        try:
                            atom_indices = h_cpu_val[i].argmax(dim=-1)
                            atom_types = [dataset_info_val['atom_decoder'][j] for j in atom_indices]
                            mol = build_molecule(x_cpu_val[i], atom_types, dataset_info_val)
                            if mol:
                                valid_molecules_generated_overall.append(mol)
                                if validator_instance and validator_instance.is_valid(mol): num_valid_by_validator +=1
                        except Exception as e_mol_build: 
                            logging.warning(f"Error building/validating in val: {e_mol_build}", exc_info=False)
            except Exception as e_val_b: 
                logging.warning(f"Error in val batch {batch_idx} ep {epoch_num}: {e_val_b}", exc_info=True)
                continue
    
    avg_val_loss = total_val_loss / processed_batches if processed_batches > 0 else float('inf')
    diversity_score = LipidDiversityCalculator.compute_internal_diversity(valid_molecules_generated_overall)
    val_metrics_summary = {
        'val_loss': avg_val_loss, 'num_molecules_sampled_for_metrics': len(valid_molecules_generated_overall),
        'num_valid_molecules_by_validator': num_valid_by_validator, 'diversity_of_sampled_molecules': diversity_score,
        'min_batch_loss': min(m['loss'] for m in batch_metrics_list) if batch_metrics_list else 0.0,
        'max_batch_loss': max(m['loss'] for m in batch_metrics_list) if batch_metrics_list else 0.0,
        'std_batch_loss': np.std([m['loss'] for m in batch_metrics_list]).item() if batch_metrics_list and len(batch_metrics_list) > 1 else 0.0,
        'num_val_batches': processed_batches}
    logging.info(f"Validation Summary Epoch {epoch_num}: {json.dumps(val_metrics_summary, indent=2)}")
    return val_metrics_summary

# --- save_checkpoint definition ---
def save_checkpoint(model, optimizer, epoch, val_metrics, args, is_final=False):
    checkpoint_dir = args.checkpoints_dir 
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = f"final_checkpoint_epoch_{epoch + 1}.pt" if is_final else f"checkpoint_epoch_{epoch + 1}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics, 
        "args": {k: v for k, v in vars(args).items() if k not in ['dataset_info_fallback']} # Exclude bulky objects if not needed
    }
    try:
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e_save:
        logging.error(f"Error saving checkpoint {checkpoint_path}: {e_save}")


# --- Training & Validation 
def main():
    global args, model, optim, ema, dataloaders, device, dataset_info 

    # Ensure dataset_info is loaded and available 
    if 'dataset_info' not in globals() or dataset_info is None:
        logging.warning("Global 'dataset_info' not found. Attempting to load.")
        if hasattr(args, 'dataset_info_fallback') and args.dataset_info_fallback is not None:
            dataset_info = args.dataset_info_fallback
        else:
            try:
                dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
                args.dataset_info_fallback = dataset_info 
            except Exception as e_di_main:
                logging.error(f"Critical: Failed to load dataset_info in main: {e_di_main}. Exiting.")
                sys.exit(1)
    if dataset_info is None: 
        logging.error("Critical: dataset_info is None in main. Exiting.")
        sys.exit(1)

    optim = get_optim(args, model) # Initialize optimizer
    logging.info(f"Optimizer initialized with learning rate: {args.lr} and weight decay: {args.weight_decay}")

    lipid_validator = LipidValidator() 
    lipid_assessor = LipidAssessor()
    early_stopping = EarlyStoppingManager(args)
    
    current_optimizer = optim # Use the globally defined optimizer
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(current_optimizer, T_max=args.n_epochs)
    elif args.lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(current_optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    metrics_file = os.path.join(args.output_dir, "training_metrics.json")
    metrics_history = {'train_metrics': [], 'val_metrics': [], 'epochs': []}
    best_val_nll = float('inf')

    logging.info("Starting Stage 2 Fine-tuning from main function...")
    for epoch in range(args.n_epochs):
        train_metrics = train_epoch(
            model=model, optimizer=current_optimizer, dataloader=dataloaders['train'], device=device, 
            epoch=epoch, args=args, ema_object=ema, validator_instance=lipid_validator, 
            dataset_info_for_train=dataset_info, lipid_property_assessor_instance=lipid_assessor
        )
        
        if (epoch + 1) % args.validation_interval == 0:
            val_metrics = validate(
                model=model, dataloader=dataloaders['val'], device=device, dataset_info_val=dataset_info, 
                args=args, validator_instance=lipid_validator, epoch_num=epoch + 1
            )
            
            if args.lr_scheduler == 'reduce_on_plateau' and scheduler: scheduler.step(val_metrics['val_loss'])
            
            current_lr = current_optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch + 1}/{args.n_epochs} - Train Loss: {train_metrics['avg_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Diversity (Val): {val_metrics.get('diversity_of_sampled_molecules', 0.0):.4f} - "
                f"Valid Mols (Val): {val_metrics.get('num_valid_molecules_by_validator', 0)} - LR: {current_lr:.6f}"
            )
            
            metrics_history['epochs'].append(epoch + 1); metrics_history['train_metrics'].append(train_metrics); metrics_history['val_metrics'].append(val_metrics)
            with open(metrics_file, 'w') as f_metrics: json.dump(metrics_history, f_metrics, indent=2)
            
            if early_stopping.should_stop(epoch, val_metrics['val_loss']):
                logging.info(f"Early stopping at epoch {epoch + 1}. Best val loss: {early_stopping.best_val_loss:.4f}")
                save_checkpoint(model, current_optimizer, epoch, val_metrics, args, is_final=True); break
            
            if val_metrics['val_loss'] < best_val_nll:
                best_val_nll = val_metrics['val_loss']
                logging.info(f"New best val loss: {best_val_nll:.4f}")
                save_checkpoint(model, current_optimizer, epoch, val_metrics, args)
            
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(model, current_optimizer, epoch, val_metrics, args)
        else:
            current_lr = current_optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{args.n_epochs} - Train Loss: {train_metrics['avg_loss']:.4f} - LR: {current_lr:.6f}")
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['train_metrics'].append(train_metrics)
            metrics_history['val_metrics'].append(None)
            with open(metrics_file, 'w') as f_metrics: 
                json.dump(metrics_history, f_metrics, indent=2)
        
        if args.lr_scheduler == 'cosine' and scheduler: 
            scheduler.step()

    logging.info("Training finished. Saving final model.")
    final_val_metrics = metrics_history['val_metrics'][-1] if metrics_history['val_metrics'] and metrics_history['val_metrics'][-1] is not None else {'val_loss': float('inf'), 'comment': 'No validation data'}

    if not (early_stopping.counter >= early_stopping.patience and epoch < args.n_epochs -1 ):
        save_checkpoint(model, current_optimizer, args.n_epochs -1, final_val_metrics, args, is_final=True)
        logging.info("Final model checkpoint saved after completing all epochs.")
    else:
        logging.info("Final model checkpoint was already saved due to early stopping.")



class LipidAssessor:
    """Assesses lipid properties like biodegradability and toxicity based on structural features."""

    # SMARTS patterns for biodegradable functional groups
    # Higher score is better for biodegradability
    BIODEGRADABLE_PATTERNS = {
        "Ester": ("[C;H0;D3](=[O;D1])[O;D2][C;D4]", 1.0),
        "Amide": ("[C;H0;D3](=[O;D1])[N;D3]", 1.0),
        "CarboxylicAcid": ("[C;H0;D3](=[O;D1])[O;D1][H]", 0.5),
        "Urethane": ("[N;D3][C;H0;D3](=[O;D1])[O;D2][C;D4]", 0.8),
        "Anhydride": ("[C;H0;D3](=[O;D1])[O;D2][C;H0;D3](=[O;D1])", 1.2),
        "PhosphateEster": ("[P;D4](=[O;D1])([O;D2])([O;D2])[O-]", 0.7),
        "Ether_Activated": ("[C;D4]-[O;D2]-[C;H0;D3](=[O;D1])", 0.5)
    }

    # SMARTS patterns for common toxicophores / structural alerts
    # Higher score means more toxic (penalty)
    TOXICITY_PATTERNS = {
        "MichaelAcceptor_Keto": ("[C;H0;D3](=[O;D1])[C;H0;D2]=[C;H0;D2]", 1.5),
        "MichaelAcceptor_Nitro": ("[N;+;H0;D3]([=O;D1])([=O;D1])[C;D2]=[C;D2]", 1.5),
        "Epoxide": ("[O;D2;r3]1[C;r3][C;r3]1", 2.0),
        "Aziridine": ("[N;D3;r3]1[C;r3][C;r3]1", 2.0),
        "Nitro_Aromatic": ("[a;D2][N+](=[O;D1])[O-]", 1.0),
        "Nitro_Aliphatic": ("[C;D4][N+](=[O;D1])[O-]", 1.0),
        "AromaticAmine": ("[a;D2][N;D3;H1,H2]", 0.8),
        "Hydrazine": ("[N;D3;H1,H2][N;D3;H1,H2]", 1.5),
        "AlkylHalide_Reactive": ("[C;D4][Cl,Br,I]", 0.5),
        "AlphaHaloKetone": ("[C;H0;D3](=[O;D1])[C;D4;H1,H2][F,Cl,Br,I]", 1.8),
        "Aldehyde": ("[C;H1;D2](=[O;D1])", 0.7),
        "Peroxide": ("[O;D2][O;D2]", 2.0),
        "Diazonium": ("[C;D4]-[N+]#[N;D1]", 3.0),
        "Isocyanate": ("[C;D4]-[N;D2]=[C;D1]=[O;D1]", 2.5),
        "Thiol": ("[C;D4][S;D1;H1]", 0.5),
        "PAINS_A": ("[O;D1]=[C;D3]([N;D3])[C;D2]=[C;D2]", 0.5),
        "PAINS_B": ("[c;D2]1[c;D2][c;D2][c;D2][c;D2][c;D2]1[S;D4](=[O;D1])(=[O;D1])[N;D3]", 0.5)
    }

    def __init__(self):
        self.compiled_biodegradable_patterns = []
        for name, (smarts, score) in self.BIODEGRADABLE_PATTERNS.items():
            mol_pattern = Chem.MolFromSmarts(smarts)
            if mol_pattern:
                self.compiled_biodegradable_patterns.append((name, mol_pattern, score))
            else:
                logging.warning(f"LipidAssessor: Invalid SMARTS for biodegradable pattern '{name}': {smarts}")

        self.compiled_toxicity_patterns = []
        for name, (smarts, score) in self.TOXICITY_PATTERNS.items():
            mol_pattern = Chem.MolFromSmarts(smarts)
            if mol_pattern:
                self.compiled_toxicity_patterns.append((name, mol_pattern, score))
            else:
                logging.warning(f"LipidAssessor: Invalid SMARTS for toxicity pattern '{name}': {smarts}")
        
        logging.info(f"LipidAssessor initialized with {len(self.compiled_biodegradable_patterns)} biodegradable and {len(self.compiled_toxicity_patterns)} toxicity patterns.")

    def assess_biodegradability_score(self, mol):
        """Calculates a biodegradability score. Higher is better."""
        score = 0.0
        if not mol:
            return score

        try:
            # Basic check for very large or very small molecules, adjust as needed
            mw = Descriptors.ExactMolWt(mol)
            if not (100 < mw < 1500): # Penalize if outside a reasonable range for lipids
                score -= 1.0

            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if num_heavy_atoms == 0:
                return 0.0

            # Score based on presence of biodegradable groups
            for name, pattern, pattern_score in self.compiled_biodegradable_patterns:
                if mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    score += len(matches) * pattern_score
                    logging.debug(f"Biodegradability: Found {len(matches)} of '{name}' (score: +{len(matches) * pattern_score})")

            # Consider LogP as a factor (lipids are lipophilic, but excessively high LogP can hinder biodegradation)
            logP = Descriptors.MolLogP(mol)
            if logP > 8:
                score -= (logP - 8) * 0.5 # Penalty for very high LogP
            elif logP < 1:
                score -= (1 - logP) * 0.5 # Penalty for being too hydrophilic for a typical lipid

            # Normalize score by number of heavy atoms to avoid bias towards larger molecules simply having more groups
            if num_heavy_atoms > 0:
                 normalized_score = score / (num_heavy_atoms / 10.0) # Arbitrary scaling factor
            else:
                 normalized_score = 0.0
            
            return max(0.0, normalized_score) # Ensure score is not negative

        except Exception as e:
            logging.warning(f"Error in assess_biodegradability_score: {e}")
            return 0.0

    def assess_toxicity_score(self, mol):
        """Calculates a toxicity penalty score. Higher means more potential toxicity."""
        penalty = 0.0
        if not mol:
            return penalty
        
        try:
            # Score based on presence of toxicophores
            for name, pattern, pattern_penalty in self.compiled_toxicity_patterns:
                if mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    penalty += len(matches) * pattern_penalty
                    logging.debug(f"Toxicity: Found {len(matches)} of '{name}' (penalty: +{len(matches) * pattern_penalty})")
            
            # Consider extreme charge states as a potential issue
            formal_charge = Chem.GetFormalCharge(mol)
            if abs(formal_charge) > 2:
                penalty += abs(formal_charge) - 2

            return penalty
        except Exception as e:
            logging.warning(f"Error in assess_toxicity_score: {e}")
            return 0.0 # Return a neutral score in case of error

class LipidValidator:
    """Class to handle validation of ionizable lipids with a single, comprehensive set of constraints."""

    def __init__(self): # Removed training_stage and max_stages
        logging.info("LipidValidator initialized with a fixed, comprehensive set of validation rules.")
        # Define SMARTS patterns directly in the class
        # More general ester patterns
        self.ESTER_PATTERNS = [
            Chem.MolFromSmarts("[C;H0;D3](=[O;D1])[O;D2][C;D4]"),  # Ester: C(=O)O-C
            Chem.MolFromSmarts("[C;D4][C;H0;D3](=[O;D1])[O;D2][C;D4]") # Ester: C-C(=O)O-C
        ]
        # Common linker types
        self.LINKER_PATTERNS = [
            Chem.MolFromSmarts("[*]-[C;H0;D3](=[O;D1])-[*]"),  # Carbonyl linker
            Chem.MolFromSmarts("[*]-[O;D2]-[*]"),      # Ether linker
            Chem.MolFromSmarts("[*]-[S;D2]-[*]"),      # Thioether linker
            Chem.MolFromSmarts("[*]-[N;D3]-[*]")       # Amine linker
        ]
        # Simplified backbone structure for lipids
        self.BACKBONE_PATTERNS = [
            # Glycerol-like (central C with 2+ O-linked chains) - simplified
            Chem.MolFromSmarts("[C;D4](-[C;D4])(-[O;D2][C;D4])-[O;D2][C;D4]"), # C(C)(OC)OC
            # Two long alkyl chains connected by some linker (very general)
            Chem.MolFromSmarts("[C;D4]([C;D4])([C;D4])-[C;D4]~[*]~[C;D4]([C;D4])([C;D4])-[C;D4]") # Simplified long chain check
        ]
        # Common ionizable groups for lipids
        self.IONIZABLE_PATTERNS = [
            Chem.MolFromSmarts("[N;+;D3;H1,H2,H3]"),          # Protonated amine (general)
            Chem.MolFromSmarts("[N;D3]([C;D4])([C;D4])[C;D4]"),        # Tertiary amine (neutral form)
            Chem.MolFromSmarts("[C;D4]-[N;D3](-[C;D4])-[C;D4]"),   # Tertiary amine
            Chem.MolFromSmarts("[C;D4]=[N;D2][C;D4]"),                  # Imine
            Chem.MolFromSmarts("[C;D4]-[N;D2]=[C;D4]"),              # Imine variant
            Chem.MolFromSmarts("[N;D3]1[C;D4][C;D4][N;D3][C;D4][C;D4]1"),                 # Piperazine core
            Chem.MolFromSmarts("[c;D2]1[c;D2][n;D2][n;D2][c;D2]1"),               # Imidazole
        ]

    def is_valid(self, mol):
        """Validate if a molecule is a valid ionizable lipid using all constraints."""
        if mol is None:
            return False
        
        # Apply all validation checks
        return (self._is_chemically_valid(mol) and
                self._has_basic_lipid_structure(mol) and
                self._has_ionizable_group(mol) and
                self._meets_size_requirements(mol))

    def _is_chemically_valid(self, mol):
        """Basic chemical validity check."""
        try:
            return (mol is not None and
                    len(Chem.GetMolFrags(mol)) == 1 and
                    all(atom.GetDegree() <= 4 for atom in mol.GetAtoms()))
        except Exception as e:
            logging.warning(f"Error in chemical validity check: {e}")
            return False

    def _has_basic_lipid_structure(self, mol):
        """Check for basic lipid-like structure without strict patterns."""
        try:
            # More lenient backbone check
            return any(mol.HasSubstructMatch(pattern) # Removed Chem.MolFromSmarts as patterns are pre-compiled
                       for pattern in self.BACKBONE_PATTERNS)
        except Exception as e:
            logging.warning(f"Error in basic structure check: {e}")
            return False

    def _has_ionizable_group(self, mol):
        """Check for ionizable groups."""
        try:
            # Full ionizable group check
            return any(mol.HasSubstructMatch(pattern) # Removed Chem.MolFromSmarts
                       for pattern in self.IONIZABLE_PATTERNS)
        except Exception as e:
            logging.warning(f"Error in ionizable group check: {e}")
            return False

    def _meets_size_requirements(self, mol):
        """Check size requirements."""
        try:
            mw = Descriptors.ExactMolWt(mol)
            num_rotatable = Descriptors.NumRotatableBonds(mol)
            # Final size requirements
            return (300 <= mw <= 1000) and (num_rotatable >= 5)
        except Exception as e:
            logging.warning(f"Error in size requirement check: {e}")
            return False

# Removed LipidSimilarityCalculator and LipidDataAugmenter as they were outside user's immediate request for this edit.
# If they were used by the simplified LipidValidator, they would be kept.

# Helper Classes (continued)

class LipidDiversityCalculator:
    """Calculates internal diversity of generated molecules."""

    @staticmethod
    def compute_internal_diversity(molecules):
        """Compute the internal diversity of a list of molecules."""
        if not molecules:
            return 0.0
        # Convert molecules to Morgan fingerprints
        fps = [AllChem.GetMorganFingerprint(mol, 2) for mol in molecules]
        # Compute Tanimoto similarity matrix
        similarity_matrix = DataStructs.BulkTanimotoSimilarity(fps, fps)
        # Compute average similarity
        avg_similarity = similarity_matrix.mean()
        return 1.0 - avg_similarity

# --- Molecule Building and Validation Functions ---

def calculate_angle(v1, v2, v3):
    """Calculate the angle between three points (vectors v2->v1 and v2->v3)."""
    v1 = np.array(v1); v2 = np.array(v2); v3 = np.array(v3)
    vec1 = v1 - v2
    vec2 = v3 - v2
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # Ensure valid range for arccos
    return np.arccos(cos_angle)

def is_reasonable_angle(angle_rad, min_deg=60.0, max_deg=180.0):
    """Check if a bond angle (in radians) is reasonable."""
    angle_deg = np.degrees(angle_rad)
    return min_deg <= angle_deg <= max_deg

def has_reasonable_geometry(mol):
    """Check if the molecule has reasonable bond lengths and angles."""
    try:
        if mol.GetNumConformers() == 0: 
            return True # Cannot check if no conformer
        conf = mol.GetConformer()

        # Check bond lengths against typical ranges (very rough check)
        for bond in mol.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            p1, p2 = conf.GetAtomPosition(begin_idx), conf.GetAtomPosition(end_idx)
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            
            # A very basic check, could be refined with covalent radii sums
            atom1_sym = mol.GetAtomWithIdx(begin_idx).GetSymbol()
            atom2_sym = mol.GetAtomWithIdx(end_idx).GetSymbol()
            
            if not (0.7 < dist < 2.0): # General range for typical covalent bonds
                pass


        # Check angles
        for atom in mol.GetAtoms():
            neighbors = atom.GetNeighbors()
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        angle = calculate_angle(
                            conf.GetAtomPosition(neighbors[i].GetIdx()),
                            conf.GetAtomPosition(atom.GetIdx()),
                            conf.GetAtomPosition(neighbors[j].GetIdx())
                        )
                        if not is_reasonable_angle(angle):
                            pass
        return True
    except Exception as e:
        logging.warning(f"Error in has_reasonable_geometry: {e}")
        return False # Treat errors as unreasonable

def is_valid_chirality(mol):
    """Check if the molecule has valid chirality centers."""
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False) # Only check assigned
        return True # Simplified: if RDKit can assign it without error, assume valid for now
    except Exception as e:
        logging.warning(f"Error in is_valid_chirality: {e}")
        return False


def has_reasonable_conformation(mol):
    """Check if the molecule has a reasonable 3D conformation using MMFF94s if possible."""
    try:
        if mol.GetNumConformers() == 0:
            # Try to generate a conformer if none exists
            conf_id = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if conf_id < 0: return False # Embedding failed
        

        # After potential optimization, check geometry
        if not has_reasonable_geometry(mol):
            return False
            
        return True
    except Exception as e:
        return False # Treat errors as unreasonable

def is_valid_molecule(mol):
    """Check if an RDKit molecule is chemically and somewhat biologically valid (more checks)."""
    if mol is None: return False
    try:
        # Basic chemical validity
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception as e_sanitize:
            return False

        # Check for number of fragments
        if len(Chem.GetMolFrags(mol)) > 1:
            return False
            
        # Check atom valences 
        for atom in mol.GetAtoms():
            try:
                atom.UpdatePropertyCache(strict=True) # Recalculate valency info
            except Exception: # Raised if valence is invalid
                return False
        
        # Check for reasonable 3D conformation and geometry
        if not has_reasonable_conformation(mol): # This includes geometry checks
            # logging.debug("Molecule has unreasonable conformation or geometry.")
            return False

        # Check for valid chirality assignment (basic check)
        if not is_valid_chirality(mol):
            # logging.debug("Molecule has invalid chirality.")
            return False


        return True
    except Exception as e:
        # logging.warning(f"Unexpected error in is_valid_molecule: {e}")
        return False

def build_molecule(positions, atom_symbols, dataset_info):
    """Build an RDKit molecule from positions and atom types using a more robust approach."""
    try:
        mol = Chem.RWMol()
        atom_map = {} # To map original index to RDKit atom index if needed later

        for i, atom_symbol in enumerate(atom_symbols):
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(atom_symbol)
            if atomic_num == 0:
                logging.warning(f"Invalid atom type '{atom_symbol}' at index {i}. Skipping molecule.")
                return None
            atom = Chem.Atom(atomic_num)
            rdkit_idx = mol.AddAtom(atom)
            atom_map[i] = rdkit_idx

        # Create a conformer and add atom positions
        conf = Chem.Conformer(len(atom_symbols))
        for i in range(len(atom_symbols)):
            conf.SetAtomPosition(i, tuple(positions[i]))
        mol.AddConformer(conf)

        # Add bonds based on distance and get_bond_order
        
        adj = np.zeros((len(atom_symbols), len(atom_symbols)), dtype=int)
        for i in range(len(atom_symbols)):
            for j in range(i + 1, len(atom_symbols)):
                dist = np.linalg.norm(positions[i] - positions[j])
                try:
                    # Ensure atom symbols are strings
                    sym_i = str(atom_symbols[i])
                    sym_j = str(atom_symbols[j])
                    bond_order = get_bond_order(sym_i, sym_j, dist) # From GeoLDM.core.bond_analyze
                    if bond_order > 0:
                        # Map original indices to RDKit atom indices if necessary
                        mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.values[bond_order])
                except Exception as e_bond:
                    logging.debug(f"Could not determine bond order for atoms {i}-{j} ({sym_i}-{sym_j}, dist {dist:.2f}): {e_bond}")


        # Convert RWMol to Mol
        final_mol = mol.GetMol()
        if final_mol is None:
            logging.warning("Failed to convert RWMol to Mol.")
            return None

        # Sanitize the molecule 
        try:
            Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception as e:
            
            # If sanitization fails, often the molecule is problematic.
            return None # Or return the unsanitized mol for debugging, but it's risky
        # Perform final validation checks
        if not is_valid_molecule(final_mol): # Using the enhanced is_valid_molecule
            # logging.debug("Built molecule failed extended validation (is_valid_molecule).")
            return None
            
        return final_mol

    except Exception as e:
        logging.warning(f"Error building molecule: {e}", exc_info=False) 
        return None

class EarlyStoppingManager:
    """Manages early stopping logic."""
    def __init__(self, args):
        self.patience = args.early_stopping_patience
        self.min_delta = args.early_stopping_min_delta
        self.best_val_loss = float('inf')
        self.counter = 0
        logging.info(f"EarlyStoppingManager initialized with patience: {self.patience}, min_delta: {self.min_delta}")

    def should_stop(self, current_epoch, val_loss):
        """Check if training should stop based on validation loss."""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            logging.info(f"EarlyStopping: New best val_loss {self.best_val_loss:.4f} at epoch {current_epoch + 1}.")
            return False
        else:
            self.counter += 1
            logging.info(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs. Best: {self.best_val_loss:.4f}, Current: {val_loss:.4f}")
            if self.counter >= self.patience:
                logging.info(f"Early stopping triggered at epoch {current_epoch + 1}.")
                return True
        return False


if __name__ == "__main__":
    main()


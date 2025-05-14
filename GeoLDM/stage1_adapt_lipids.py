'''
Stage 1 Adaptation Script: Fine-tuning GeoLDM on unlabeled lipid data.

Adapts a pre-trained GeoLDM model (e.g., from GEOM-Drugs) to the general
chemical space of a target lipid dataset provided via an SDF file (preprocessed).
This script performs UNCONDITIONAL training.

Based on finetune_lipids.py
'''

from rdkit import Chem
import logging
import sys


# Local imports
import GeoLDM.lipid_dataset as lipid_dataset
from GeoLDM.configs.datasets_config import get_dataset_info


# Original GeoLDM imports
import copy
import GeoLDM.utils as utils
import argparse
import wandb
import os
from os.path import join
from GeoLDM.core.models import get_optim, get_latent_diffusion
from GeoLDM.equivariant_diffusion import en_diffusion
from GeoLDM.equivariant_diffusion import utils as diffusion_utils
from GeoLDM.core import losses as qm9_losses
import torch
import time
import pickle
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
from collections import Counter
import math # Added for math.sqrt

# --- LSUV Global and Helper Functions (NEW - PLACED AT TOP) ---
captured_output_lsuv = None

def hook_fn_lsuv(module, input, output):
    global captured_output_lsuv
    captured_output_lsuv = output

def get_layer_by_path(model, path_str):
    obj = model
    for part in path_str.split('.'):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif part.isdigit() and isinstance(obj, (torch.nn.ModuleList, torch.nn.Sequential)):
            try:
                obj = obj[int(part)]
            except IndexError:
                logging.error(f"IndexError getting layer part {part} in {path_str}")
                return None
        else:
            logging.error(f"Could not get layer part {part} in {path_str}")
            return None
    return obj

def lsuv_initialize_modules(model, module_paths, sample_batch_data, args_obj, device, dtype,
                           target_std=0.05, target_mean=0.0, iterations=10, tolerance=0.01,
                           is_stage2_run=False):
    logging.info(f"Starting LSUV init for {len(module_paths)} modules (is_stage2_run: {is_stage2_run}), target_std={target_std}, target_mean={target_mean}")
    global captured_output_lsuv

    # Get all suitable modules for LSUV initialization
    modules = get_lsuv_modules(model, module_paths)
    logging.info(f"Found {len(modules)} suitable modules for LSUV initialization")

    # Initialize max deviation variables
    max_mean_deviation_from_target = 0.0
    max_std_deviation_from_target = 0.0

    # --- Prepare sample batch data (x, h) ---
    x_sample_raw = sample_batch_data['positions'].to(device, dtype)
    one_hot_sample_raw = sample_batch_data['one_hot'].to(device, dtype)
    charges_sample_raw = sample_batch_data['charges'].to(device, dtype)
    node_mask_sample = sample_batch_data['atom_mask'].to(device, dtype)
    edge_mask_sample = sample_batch_data['edge_mask'].to(device, dtype)

    if node_mask_sample.ndim == 2:
        node_mask_sample = node_mask_sample.unsqueeze(-1)

    x_mean_attr = 'stage2_dataset_x_mean' if is_stage2_run else 'dataset_x_mean'
    x_std_attr = 'stage2_dataset_x_std' if is_stage2_run else 'dataset_x_std'
    h_cat_mean_attr = 'stage2_dataset_h_cat_mean' if is_stage2_run else 'dataset_h_cat_mean'
    h_cat_std_attr = 'stage2_dataset_h_cat_std' if is_stage2_run else 'dataset_h_cat_std'

    # Initialize normalized tensors
    x_sample_normalized = x_sample_raw
    one_hot_sample_normalized = one_hot_sample_raw

    # Apply normalization if stats are available
    if hasattr(args_obj, x_mean_attr) and getattr(args_obj, x_mean_attr) is not None:
        dataset_x_mean_dev = getattr(args_obj, x_mean_attr).to(device)
        dataset_x_std_dev = getattr(args_obj, x_std_attr).to(device)
        x_sample_normalized = (x_sample_raw - dataset_x_mean_dev.unsqueeze(0).unsqueeze(0)) / (dataset_x_std_dev.unsqueeze(0).unsqueeze(0) + 1e-6)
    else:
        logging.warning(f"LSUV: Dataset x mean/std not found in args_obj ({x_mean_attr}). Using raw positions.")

    if hasattr(args_obj, h_cat_mean_attr) and getattr(args_obj, h_cat_mean_attr) is not None:
        dataset_h_cat_mean_dev = getattr(args_obj, h_cat_mean_attr).to(device)
        dataset_h_cat_std_dev = getattr(args_obj, h_cat_std_attr).to(device)
        one_hot_sample_normalized = (one_hot_sample_raw - dataset_h_cat_mean_dev.unsqueeze(0).unsqueeze(0)) / (dataset_h_cat_std_dev.unsqueeze(0).unsqueeze(0) + 1e-6)
    else:
        logging.warning(f"LSUV: Dataset h_cat mean/std not found in args_obj ({h_cat_mean_attr}). Using raw one_hot features.")

    x_sample_for_model = diffusion_utils.remove_mean_with_mask(x_sample_normalized, node_mask_sample)
    one_hot_sample_for_model = one_hot_sample_normalized * node_mask_sample
    charges_sample_for_model = charges_sample_raw * node_mask_sample
    h_sample_for_model = {'categorical': one_hot_sample_for_model, 'integer': charges_sample_for_model}

    # Ensure model's gamma tensor is on the correct device
    if hasattr(model, 'gamma'):
        model.gamma = model.gamma.to(device)
    if hasattr(model, 'dynamics') and hasattr(model.dynamics, 'gamma'):
        model.dynamics.gamma = model.dynamics.gamma.to(device)

    original_training_state = model.training
    model.eval()

    for iteration in range(iterations):
        logging.info(f"  LSUV Iteration {iteration + 1}/{iterations}")
        iteration_max_std_dev = 0.0
        iteration_max_mean_dev = 0.0
        
        for path, module in modules:
            if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                continue

            handle = module.register_forward_hook(hook_fn_lsuv)
            with torch.no_grad():
                _ = model(x_sample_for_model, h_sample_for_model, node_mask_sample, edge_mask_sample, context=None)
            handle.remove()

            if captured_output_lsuv is not None:
                adjusted_activations = captured_output_lsuv.detach()
                captured_output_lsuv = None
                
                # Apply masking to adjusted activations if they have the same shape as node_mask
                if adjusted_activations.shape[:2] == node_mask_sample.shape[:2]:
                    adjusted_activations = adjusted_activations * node_mask_sample
                
                finite_adjusted = adjusted_activations[torch.isfinite(adjusted_activations)]
                if finite_adjusted.numel() > 0:
                    mean_after_adjust = finite_adjusted.mean()
                    std_after_adjust = finite_adjusted.std()
                    iteration_max_mean_dev = max(iteration_max_mean_dev, abs(mean_after_adjust.item() - target_mean))
                    iteration_max_std_dev = max(iteration_max_std_dev, abs(std_after_adjust.item() - target_std))

        max_mean_deviation_from_target = iteration_max_mean_dev
        max_std_deviation_from_target = iteration_max_std_dev
        
        logging.info(f"  LSUV Iteration {iteration + 1} ended. Max mean_dev ~ {max_mean_deviation_from_target:.4f}, Max std_dev_from_target={max_std_deviation_from_target:.4f}")
        if max_std_deviation_from_target < tolerance and max_mean_deviation_from_target < tolerance:
            logging.info(f"  LSUV converged after {iteration + 1} iterations.")
            break
            
    model.train(original_training_state)
    logging.info("LSUV-like initialization finished.")
# --- End of LSUV Global and Helper Functions ---

# --- Argument Parsing Setup ---
# Define parser first
parser = argparse.ArgumentParser(description='GeoLDM Lipid Adaptation (Stage 1 - Unlabeled)')
parser.add_argument('--log_file', type=str, default=None, help='Path to save log file. If None, logs to console.')
parser.add_argument('--exp_name', type=str, default='geoldm_lipid_adapt_stage1', help='Experiment name for logging and output')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs and checkpoints')

# --- Data Arguments --- 
parser.add_argument('--unlabeled_data_path', type=str, required=True, help='Path to processed UNLABELED lipid data (e.g., processed_unlabeled_lipids.pkl from SDF)')
# Removed: --lipid_stats_path
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
parser.add_argument('--dataset', type=str, default='geom', help='Base dataset type (used for dataset_info, should match pretrained model)')
parser.add_argument('--remove_h', action='store_true', help='Use dataset config without hydrogens (ensure model matches)') # Keep consistency with pretrained
parser.add_argument('--val_split_ratio', type=float, default=0.01, help='Fraction of unlabeled data for validation (can be small/zero for adaptation)')

# --- Model Loading Arguments --- 
parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the folder containing pretrained GEOM model (.npy, .pickle files) to adapt')
parser.add_argument('--model_name', type=str, default='generative_model_ema.npy', help='Name of the model state_dict file to load')

# --- Adaptation Arguments (similar to Fine-tuning) --- 
parser.add_argument('--n_epochs', type=int, default=50, help='Number of adaptation epochs (adjust as needed)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (adjust based on GPU memory)')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for adaptation (might need lower than Stage 2)')
parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate, 0 to disable')
parser.add_argument('--test_epochs', type=int, default=5, help='Run validation every N epochs')
parser.add_argument('--save_model', type=eval, default=True, help='Save checkpoints')
parser.add_argument('--clip_grad', type=eval, default=True, help='Clip gradients during training')
parser.add_argument('--clip_value', type=float, default=1.0, help='Value for gradient clipping')
parser.add_argument('--n_report_steps', type=int, default=50, help='Log training progress every N steps')
parser.add_argument('--ode_regularization', type=float, default=1e-3, help='ODE regularization weight')

# --- Conditioning Arguments --- 
# REMOVED: --conditioning argument
# REMOVED: --cfg_prob argument

# --- Diffusion Arguments (load from pickle) --- 
# Defaults are provided but should be overridden by loaded args.pickle
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2')
parser.add_argument('--diffusion_loss_type', type=str, default='l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

# --- Model Architecture Arguments (load from pickle) --- 
parser.add_argument('--model', type=str, default='egnn_dynamics')
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--nf', type=int, default=256)
parser.add_argument('--latent_nf', type=int, default=2)
parser.add_argument('--tanh', type=eval, default=True)
parser.add_argument('--attention', type=eval, default=True)
parser.add_argument('--norm_constant', type=float, default=1)
parser.add_argument('--inv_sublayers', type=int, default=1)
parser.add_argument('--sin_embedding', type=eval, default=False)
parser.add_argument('--include_charges', type=eval, default=True) # Keep consistency with pretrained
parser.add_argument('--normalization_factor', type=float, default=1.0)
parser.add_argument('--aggregation_method', type=str, default='sum')
parser.add_argument('--kl_weight', type=float, help='Weight for KL divergence term in VAE loss (if VAE is trained).')
parser.add_argument('--train_diffusion', type=eval, help='Whether to train the diffusion model component.')

# Boolean flags for trainable_ae - will be reconciled with pickle default later
# Use a temporary destination to detect if user explicitly used the flag
parser.add_argument('--trainable_ae_cmd', dest='trainable_ae_cmd', action='store_true', default=None, help='Command-line: Set VAE to be trainable. If both this and --no-trainable_ae_cmd are given, the last one wins.')
parser.add_argument('--no-trainable_ae_cmd', dest='trainable_ae_cmd', action='store_false', help='Command-line: Set VAE to be NOT trainable (frozen).')

# Boolean flags for condition_time - will be reconciled with pickle default later
parser.add_argument('--condition_time_cmd', dest='condition_time_cmd', action='store_true', default=None, help='Command-line: Set dynamics model to be conditioned on time. If both this and --no-condition_time_cmd are given, the last one wins.')
parser.add_argument('--no-condition_time_cmd', dest='condition_time_cmd', action='store_false', help='Command-line: Set dynamics model to be NOT conditioned on time.')

parser.add_argument('--normalize_factors', type=eval, help='Tuple of normalization factors for x, h_cat, h_int.')

# --- Other Arguments --- 
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
parser.add_argument('--wandb_usr', type=str, default=None, help='WandB username')
parser.add_argument('--wandb_project', type=str, default='e3_diffusion_lipid_adapt_stage1', help='WandB project name')
parser.add_argument('--online', type=bool, default=True, help='WandB online mode')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

# --- Load Pretrained Args (BEFORE parsing) ---
# Need to manually parse just the pretrained_path argument first
temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument('--pretrained_path', type=str, required=True)
known_args, _ = temp_parser.parse_known_args()
pretrained_args_path = join(known_args.pretrained_path, 'args.pickle')
loaded_pretrained_args = None
try:
    with open(pretrained_args_path, 'rb') as f:
        loaded_pretrained_args = pickle.load(f)
    logging.info(f"Loaded arguments from pretrained model: {pretrained_args_path}")
except Exception as e:
    logging.error(f"Error: Could not load required pretrained args.pickle from {pretrained_args_path}. Error: {e}. Exiting.")
    exit(1)

# Store the trainable_ae value from pickle if it exists
pickle_trainable_ae = getattr(loaded_pretrained_args, 'trainable_ae', True) # Default to True if not in pickle for some reason

# Store the condition_time value from pickle if it exists
pickle_condition_time = getattr(loaded_pretrained_args, 'condition_time', True) # Default to True (common for these models)

# --- Set Defaults from Loaded Args ---
# Critical keys that define the model architecture and diffusion process
# These should NOT be overridden by command-line arguments easily
critical_keys_to_set_defaults = [
    'model', 'n_layers', 'nf', 'latent_nf', 'tanh', 'attention',
    'norm_constant', 'inv_sublayers', 'sin_embedding', 'include_charges',
    'normalization_factor', 'aggregation_method', 'dataset', 'remove_h',
    'diffusion_steps', 'diffusion_noise_schedule', 'diffusion_loss_type',
    'diffusion_noise_precision', 'normalize_factors',
    'kl_weight', 'train_diffusion', 'condition_time'
]
loaded_defaults = {}
for key in critical_keys_to_set_defaults:
    if hasattr(loaded_pretrained_args, key):
        loaded_defaults[key] = getattr(loaded_pretrained_args, key)
    else:
        logging.warning(f"Warning: Critical key '{key}' not found in pretrained args. Using script default.")

logging.info("Updating parser defaults based on pretrained args (for non-overridden critical keys):")
for key, value in loaded_defaults.items():
    logging.info(f"  Setting default for '{key}': {value}")
parser.set_defaults(**loaded_defaults)

# --- Parse ALL Arguments ---
# Now parse, command-line args will override non-critical defaults set above
args = parser.parse_args()

# --- Reconcile trainable_ae ---
# Command line (--trainable_ae_cmd or --no-trainable_ae_cmd) takes precedence.
# If neither was used, fall back to the value from the pickle file.
if args.trainable_ae_cmd is not None:
    args.trainable_ae = args.trainable_ae_cmd  # Command-line flag was used
    logging.info(f"INFO: 'trainable_ae' explicitly set to {args.trainable_ae} by command-line argument.")
else:
    args.trainable_ae = pickle_trainable_ae  # No command-line flag, use pickle default
    logging.info(f"INFO: 'trainable_ae' set to {args.trainable_ae} from pickle default (no command-line override for trainable_ae).")
delattr(args, 'trainable_ae_cmd') # Clean up the temporary command-line argument storage

# --- Reconcile condition_time ---
if args.condition_time_cmd is not None:
    args.condition_time = args.condition_time_cmd
    logging.info(f"INFO: 'condition_time' explicitly set to {args.condition_time} by command-line argument.")
else:
    args.condition_time = pickle_condition_time
    logging.info(f"INFO: 'condition_time' set to {args.condition_time} from pickle default (no command-line override for condition_time).")
delattr(args, 'condition_time_cmd')

# Override latent_nf to 3 if loading a pretrained model,
# as the original GeoLDM drugs_latent2 checkpoint requires latent_nf=3 for its VAE.
if args.pretrained_path: 
    logging.info(f"INFO: Pretrained model path specified: {args.pretrained_path}")
    # Check if latent_nf needs to be set or overridden
    # The default from the pickle might be 2, but the actual model weights correspond to 3.
    if getattr(args, 'latent_nf', None) != 3:
        logging.info(f"INFO: Overriding args.latent_nf from {getattr(args, 'latent_nf', 'Not set')} to 3 for checkpoint compatibility.")
        args.latent_nf = 3
    else:
        logging.info(f"INFO: args.latent_nf is already 3. No override needed.")
else:
    # This block might not be strictly necessary if latent_nf default is sensible without a pretrained model
    logging.info("INFO: No pretrained_path specified, not overriding latent_nf (using parsed/default value).")

# Set probabilistic_model, as it's expected by the loss function
# and other parts of the original GeoLDM framework.
# For EnLatentDiffusion, this should be 'diffusion'.
if not hasattr(args, 'probabilistic_model'):
    logging.info("INFO: Setting args.probabilistic_model = 'diffusion'")
    args.probabilistic_model = 'diffusion'
elif args.probabilistic_model != 'diffusion':
    logging.warning(f"WARNING: args.probabilistic_model was {args.probabilistic_model}. Forcing to 'diffusion' for EnLatentDiffusion.")
    args.probabilistic_model = 'diffusion'

# --- Setup Device, Seed, and Output Dirs --- 
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

args.output_dir = join(args.output_dir, args.exp_name)
utils.create_folders(args)

# --- Setup Logging (after parsing args and creating output_dir) ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
root_logger = logging.getLogger() # Get the root logger

# Set level for the root logger itself
# All handlers will inherit this level unless they set their own lower level
root_logger.setLevel(logging.INFO) # Or logging.DEBUG for more verbosity

# Remove any pre-existing handlers from the root logger
# This is important if the script might be run multiple times in the same session (e.g., a notebook)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close() # Close the handler before removing

# Console handler (always logs to console)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# File handler (only if --log_file is specified)
if args.log_file:
    try:
        log_file_path = Path(args.log_file)
        # Ensure the directory for the log file exists
        log_file_dir = log_file_path.parent
        if not log_file_dir.exists():
            log_file_dir.mkdir(parents=True, exist_ok=True)
            # print(f"DEBUG: Created directory for log file: {log_file_dir}") # Temporary debug
        
        file_handler = logging.FileHandler(args.log_file, mode='a') # 'a' for append
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to file: {args.log_file}") # This message will go to console and file
    except Exception as e:
        logging.error(f"Error setting up file logger at {args.log_file}: {e}")
        # If file logging fails, script continues with console logging
else:
    logging.info("Logging to console only (no --log_file specified).")

# --- Get Dataset Info (AFTER final args are set) ---
# Ensure remove_h used here corresponds to loaded include_charges
# Typically, if include_charges is True, remove_h should be False, and vice-versa
# We rely on the pretrained_args to have consistent values for these.
dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
logging.info(f"Using dataset info for: {args.dataset} (remove_h={args.remove_h}, loaded include_charges={args.include_charges})")

# --- Wandb Setup --- 
if args.no_wandb:
    mode = 'disabled'
    logging.info("WandB logging is disabled.")
else:
    mode = 'online' if args.online else 'offline'
    if not args.wandb_usr:
        args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
if not args.no_wandb:
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': args.wandb_project, 'config': args,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

logging.info("Effective Arguments for Stage 1 Adaptation:")
for k, v in vars(args).items():
    logging.info(f"  {k}: {v}")

# --- Load Data --- 
# Load the UNLABELED data using the specific path argument
# Use val_split_ratio from args to control validation set size (can be 0)
dataloaders = lipid_dataset.get_dataloaders(
    data_path=args.unlabeled_data_path,
    batch_size=args.batch_size, # Batch size here is for training, for histogram can iterate dataset directly
    num_workers=args.num_workers,
    seed=args.seed,
    val_split_ratio=args.val_split_ratio # Control split here
)

if dataloaders is None:
    logging.error("Error: Failed to create dataloaders. Exiting.")
    exit(1)
if not dataloaders['train']:
    logging.error("Error: Training dataloader is empty. Check data path and preprocessing. Exiting.")
    exit(1)
if not dataloaders['val'] and args.val_split_ratio > 0:
     logging.warning("Warning: Validation dataloader is empty, but validation split ratio > 0.")

# --- Calculate n_nodes histogram from the actual training data ---
logging.info("Calculating n_nodes histogram from the loaded unlabeled training data...")
n_nodes_counts = Counter()
try:
    train_dataset = dataloaders['train'].dataset
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        if 'positions' in sample and hasattr(sample['positions'], 'shape'):
            num_nodes = sample['positions'].shape[0]
            n_nodes_counts[num_nodes] += 1
        elif 'num_atoms' in sample: # Fallback if 'num_atoms' field exists
            num_nodes = sample['num_atoms']
            if isinstance(num_nodes, torch.Tensor):
                num_nodes = num_nodes.item()
            n_nodes_counts[num_nodes] +=1
        else:
            logging.warning(f"Could not determine num_nodes for sample {i} in train_dataset. Skipping for histogram.")

    if not n_nodes_counts:
        logging.error("Failed to compute n_nodes histogram. It's empty. Check data loading and sample structure. Exiting.")
        exit(1)

    dataset_info['n_nodes'] = dict(n_nodes_counts) # Update the existing dataset_info
    logging.info(f"Successfully updated n_nodes histogram from unlabeled data. Example counts (first 5): {list(n_nodes_counts.items())[:5]}")
    logging.info(f"Min nodes: {min(n_nodes_counts.keys()) if n_nodes_counts else 'N/A'}, Max nodes: {max(n_nodes_counts.keys()) if n_nodes_counts else 'N/A'}, Unique sizes: {len(n_nodes_counts)}")
except Exception as e_hist:
    logging.error(f"Error calculating n_nodes histogram from unlabeled data: {e_hist}. Exiting.", exc_info=True)
    exit(1)
# --- End of n_nodes histogram calculation ---

# --- Compute Dataset Statistics for Normalization (NEW) ---
logging.info("Calculating dataset statistics for normalization...")
all_positions_list = []
all_one_hot_list = []
# Iterate through the dataset directly to avoid shuffling and batching artifacts for stats
temp_dataset_for_stats = dataloaders['train'].dataset
if not temp_dataset_for_stats or len(temp_dataset_for_stats) == 0: # Check if dataset has items
    logging.error("Training dataset is empty or has no length, cannot compute normalization stats. Exiting.")
    exit(1)

for i in range(len(temp_dataset_for_stats)):
    sample_data = temp_dataset_for_stats[i] # This is a dict of numpy arrays
    
    if not all(k in sample_data for k in ['positions', 'one_hot', 'atom_mask']):
        logging.warning(f"Skipping sample {i} for stats calculation due to missing keys.")
        continue
        
    pos = sample_data['positions']      # (N, 3)
    one_hot = sample_data['one_hot']    # (N, num_feat_one_hot)
    atom_mask = sample_data['atom_mask']# (N, 1)
    
    valid_indices = atom_mask.squeeze().astype(bool)
    
    if np.any(valid_indices): # Ensure there are valid atoms before trying to index
        masked_pos = pos[valid_indices]
        masked_one_hot = one_hot[valid_indices]
    
        # NEW: Check for NaNs/infs in masked_pos before appending
        if masked_pos.size > 0:
            if np.isnan(masked_pos).any() or np.isinf(masked_pos).any():
                logging.warning(f"Sample {i} position data (after masking) contains NaN/inf. Skipping for position stats.")
            else:
                all_positions_list.append(masked_pos)
        
        # NEW: Check for NaNs/infs in masked_one_hot before appending
        if masked_one_hot.size > 0:
            if np.isnan(masked_one_hot).any() or np.isinf(masked_one_hot).any():
                logging.warning(f"Sample {i} one_hot data (after masking) contains NaN/inf. Skipping for one_hot stats.")
            else:
                all_one_hot_list.append(masked_one_hot)
    else:
        logging.warning(f"Sample {i} has no valid atoms based on atom_mask. Skipping for stats.")


if not all_positions_list or not all_one_hot_list:
    logging.error("No valid data found after masking (or all samples had no valid atoms) to compute normalization stats. Check dataset. Exiting")
    exit(1)

# Concatenate and convert to tensors
all_positions_np = np.concatenate(all_positions_list, axis=0)
all_one_hot_np = np.concatenate(all_one_hot_list, axis=0)

all_positions_tensor = torch.from_numpy(all_positions_np).float()
all_one_hot_tensor = torch.from_numpy(all_one_hot_np).float()

# Calculate stats for positions
args.dataset_x_mean = torch.mean(all_positions_tensor, dim=0) 
args.dataset_x_std = torch.std(all_positions_tensor, dim=0)
args.dataset_x_std[args.dataset_x_std < 1e-6] = 1.0 # Avoid division by zero

# Calculate stats for one-hot features
args.dataset_h_cat_mean = torch.mean(all_one_hot_tensor, dim=0)
args.dataset_h_cat_std = torch.std(all_one_hot_tensor, dim=0)
args.dataset_h_cat_std[args.dataset_h_cat_std < 1e-6] = 1.0

logging.info(f"  Dataset x_mean: {args.dataset_x_mean.tolist()}")
logging.info(f"  Dataset x_std: {args.dataset_x_std.tolist()}")
logging.info(f"  Dataset h_cat_mean: {args.dataset_h_cat_mean.tolist()}")
logging.info(f"  Dataset h_cat_std: {args.dataset_h_cat_std.tolist()}")

original_normalize_factors = args.normalize_factors
args.normalize_factors = [1.0, 1.0, 1.0] # Make internal normalization an identity op
logging.info(f"Original args.normalize_factors were {original_normalize_factors}. Set to {args.normalize_factors} for external normalization.")
# --- End of Dataset Statistics for Normalization ---


# --- Set Unconditional Context ---
args.context_node_nf = 0 # Unconditional for Stage 1
logging.info(f"Running Stage 1 Adaptation (Unconditional): context_node_nf = {args.context_node_nf}")


# --- Create Model --- 
# args object should now have the correct architecture parameters loaded as defaults
logging.info(f"Instantiating model (train_diffusion={args.train_diffusion})...")
# dataset_info now contains the correct n_nodes histogram for the unlabeled data
model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
logging.info("Model instantiated.")

# --- Custom Initialization for Problematic Layers (called BEFORE pretrained loading) ---
def initialize_problematic_layers(model_to_init, args_for_init):
    # This function should use normal_(mean=0.0, std=0.01) as per previous step
    logging.info("Applying custom CONSERVATIVE initialization (normal std=0.01) to specific layers with changed sizes (BEFORE LSUV)...")
    try:
        # 1. dynamics.egnn.embedding
        if hasattr(model_to_init, 'dynamics') and hasattr(model_to_init.dynamics, 'egnn') and hasattr(model_to_init.dynamics.egnn, 'embedding'):
            layer = model_to_init.dynamics.egnn.embedding
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01) 
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                logging.info("  Initialized model.dynamics.egnn.embedding with normal_(std=0.01)")
        # ... (repeat for other 3 layers: dynamics.egnn.embedding_out, vae.encoder.final_mlp[2], vae.decoder.egnn.embedding) ...
        # Ensure all 4 layers from previous edits are covered here with normal_(0,0.01)
        # 2. dynamics.egnn.embedding_out
        if hasattr(model_to_init, 'dynamics') and hasattr(model_to_init.dynamics, 'egnn') and hasattr(model_to_init.dynamics.egnn, 'embedding_out'):
            layer = model_to_init.dynamics.egnn.embedding_out
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                logging.info("  Initialized model.dynamics.egnn.embedding_out with normal_(std=0.01)")

        # 3. vae.encoder.final_mlp[2]
        # Note: Accessing final_mlp[2] needs care if model structure changes.
        # This path should be verified.
        final_mlp_module = get_layer_by_path(model_to_init, "vae.encoder.final_mlp")
        if final_mlp_module and isinstance(final_mlp_module, torch.nn.Sequential) and len(final_mlp_module) > 2:
            layer = final_mlp_module[2]
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                logging.info("  Initialized model.vae.encoder.final_mlp[2] with normal_(std=0.01)")
            else:
                logging.warning("  model.vae.encoder.final_mlp[2] found but not nn.Linear")
        else:
            logging.warning("  Could not find or access model.vae.encoder.final_mlp[2] for conservative init.")
            
        # 4. vae.decoder.egnn.embedding
        if hasattr(model_to_init, 'vae') and hasattr(model_to_init.vae, 'decoder') and hasattr(model_to_init.vae.decoder, 'egnn') and hasattr(model_to_init.vae.decoder.egnn, 'embedding'):
            layer = model_to_init.vae.decoder.egnn.embedding
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                logging.info("  Initialized model.vae.decoder.egnn.embedding with normal_(std=0.01)")
        
        logging.info("Conservative custom initialization attempt finished.")
    except Exception as e:
        logging.error(f"Error during conservative custom initialization: {e}", exc_info=True)
# --- End of Custom Conservative Initialization ---


# --- Load Pretrained Weights & Identify Mismatched for LSUV ---
model_path = join(args.pretrained_path, args.model_name)
mismatched_module_paths_for_lsuv = set() # Use a set to store unique module paths

logging.info(f"Attempting to load pretrained weights from {model_path}...")
try:
    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith('module.') for key in state_dict.keys()): # Check if any key starts with 'module.'
        logging.info("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    
    model_state_dict_ref = model.state_dict() # Reference current model state
    loaded_keys = set()
    skipped_keys_size_mismatch = []
    skipped_keys_other = [] # Keys in checkpoint not in model, or other load errors

    final_state_dict_to_load = {} # Build a state_dict of only matching parameters

    for k_checkpoint, v_checkpoint in state_dict.items():
        if k_checkpoint in model_state_dict_ref:
            v_model = model_state_dict_ref[k_checkpoint]
            if v_model.shape == v_checkpoint.shape:
                final_state_dict_to_load[k_checkpoint] = v_checkpoint
                loaded_keys.add(k_checkpoint)
            else:
                logging.warning(f"  Skipping key '{k_checkpoint}' due to size mismatch: Model shape {v_model.shape}, Checkpoint shape {v_checkpoint.shape}")
                skipped_keys_size_mismatch.append(k_checkpoint)
                # Extract module path for LSUV: e.g., 'dynamics.egnn.embedding' from 'dynamics.egnn.embedding.weight'
                module_path = '.'.join(k_checkpoint.split('.')[:-1]) # Remove .weight or .bias
                mismatched_module_paths_for_lsuv.add(module_path)
        else:
            logging.warning(f"  Skipping key '{k_checkpoint}' from checkpoint as it's not in the current model architecture.")
            skipped_keys_other.append(k_checkpoint)

    # Load the carefully constructed state dict
    # Using strict=False here is a final fallback, but we've already done the filtering.
    # More robust is to load into a fresh copy of the model or model.load_state_dict(final_state_dict_to_load, strict=False)
    # For simplicity, we'll assume model.load_state_dict can handle missing keys gracefully if strict=False
    # and we only provide matching ones.
    # However, the primary goal is to apply `final_state_dict_to_load`.
    # Let's update the model's state dict directly for the keys we want to load.
    
    current_model_sd = model.state_dict()
    current_model_sd.update(final_state_dict_to_load) # Update with loaded weights
    model.load_state_dict(current_model_sd) # Load the updated state dict

    logging.info(f"Successfully attempted load of pretrained weights.")
    logging.info(f"  Loaded {len(loaded_keys)} matching parameters.")
    if skipped_keys_size_mismatch:
        logging.warning(f"  Skipped {len(skipped_keys_size_mismatch)} keys from {len(mismatched_module_paths_for_lsuv)} unique modules due to size mismatches: {skipped_keys_size_mismatch}")
    if skipped_keys_other:
        logging.warning(f"  Skipped {len(skipped_keys_other)} other keys (e.g., not in model): {skipped_keys_other}")

except FileNotFoundError:
    logging.error(f"Error: Pretrained model file not found at {model_path}. Exiting.")
    exit(1)
except Exception as e_load:
    logging.error(f"Error loading pretrained weights: {e_load}", exc_info=True)
    # Decide if to exit or continue without pretrained weights / LSUV
    # For now, let's assume if loading fails, LSUV on mismatched might not make sense or use all layers
    mismatched_module_paths_for_lsuv.clear() # Don't do LSUV if loading failed badly

model = model.to(device)
model_dp = model # No DataParallel wrapper

# --- FREEZE DYNAMICS MODEL & SELECTIVELY UNFREEZE MISMATCHED LAYERS (NEW) ---
if hasattr(model, 'dynamics') and model.dynamics is not None:
    logging.info("Freezing all parameters in model.dynamics...")
    for param in model.dynamics.parameters():
        param.requires_grad = False
    
    layers_to_unfreeze_in_dynamics = []
    # Unfreeze model.dynamics.egnn.embedding if it exists
    dyn_embedding_layer = get_layer_by_path(model, "dynamics.egnn.embedding")
    if dyn_embedding_layer and isinstance(dyn_embedding_layer, torch.nn.Module):
        logging.info("  Unfreezing model.dynamics.egnn.embedding parameters.")
        for param in dyn_embedding_layer.parameters():
            param.requires_grad = True
        layers_to_unfreeze_in_dynamics.append("dynamics.egnn.embedding")
    else:
        logging.warning("  Could not find model.dynamics.egnn.embedding to unfreeze.")

    # Unfreeze the LAST GCL layer in model.dynamics.egnn.gcl_layers (NEW)
    if hasattr(args, 'n_layers') and args.n_layers > 0:
        last_gcl_layer_idx = args.n_layers - 1
        last_gcl_layer_path = f"dynamics.egnn.gcl_layers.{last_gcl_layer_idx}"
        dyn_last_gcl_layer = get_layer_by_path(model, last_gcl_layer_path)
        if dyn_last_gcl_layer and isinstance(dyn_last_gcl_layer, torch.nn.Module):
            logging.info(f"  Unfreezing {last_gcl_layer_path} parameters.")
            for param in dyn_last_gcl_layer.parameters():
                param.requires_grad = True
            layers_to_unfreeze_in_dynamics.append(last_gcl_layer_path)
        else:
            logging.warning(f"  Could not find {last_gcl_layer_path} to unfreeze.")
    else:
        logging.warning("  args.n_layers not available or invalid, cannot unfreeze last GCL layer.")

    # Unfreeze model.dynamics.egnn.embedding_out if it exists
    dyn_embedding_out_layer = get_layer_by_path(model, "dynamics.egnn.embedding_out")
    if dyn_embedding_out_layer and isinstance(dyn_embedding_out_layer, torch.nn.Module):
        logging.info("  Unfreezing model.dynamics.egnn.embedding_out parameters.")
        for param in dyn_embedding_out_layer.parameters():
            param.requires_grad = True
        layers_to_unfreeze_in_dynamics.append("dynamics.egnn.embedding_out")
    else:
        logging.warning("  Could not find model.dynamics.egnn.embedding_out to unfreeze.")
    logging.info("Selective freezing/unfreezing of dynamics model complete.")
else:
    logging.warning("model.dynamics not found, skipping selective freezing.")

# Note: VAE freezing is handled by args.trainable_ae = False, which should prevent its params from being passed to optimizer
# or by ensuring get_optim filters by requires_grad.

# Update mismatched_module_paths_for_lsuv to only include the ones we actually unfroze and want to LSUV
# This assumes LSUV should only run on trainable layers we are trying to adapt.
# The original mismatched_module_paths_for_lsuv included VAE layers if they were mismatched.
# For VAE layers, if trainable_ae is False, LSUV on them is moot as they won't train.
# So, we refine mismatched_module_paths_for_lsuv for clarity and intent.

final_mismatched_paths_for_lsuv = []
if "dynamics.egnn.embedding" in layers_to_unfreeze_in_dynamics and "dynamics.egnn.embedding" in mismatched_module_paths_for_lsuv:
    final_mismatched_paths_for_lsuv.append("dynamics.egnn.embedding")
if "dynamics.egnn.embedding_out" in layers_to_unfreeze_in_dynamics and "dynamics.egnn.embedding_out" in mismatched_module_paths_for_lsuv:
    final_mismatched_paths_for_lsuv.append("dynamics.egnn.embedding_out")

# If VAE is trainable (args.trainable_ae is True) and has mismatched layers identified for LSUV:
if args.trainable_ae: # This should be false based on current logs, but good practice for robustness
    if "vae.encoder.final_mlp.2" in mismatched_module_paths_for_lsuv:
        final_mismatched_paths_for_lsuv.append("vae.encoder.final_mlp.2")
    if "vae.decoder.egnn.embedding" in mismatched_module_paths_for_lsuv:
        final_mismatched_paths_for_lsuv.append("vae.decoder.egnn.embedding")

# Add the last GCL layer to LSUV targets if it was unfrozen and originally mismatched (NEW)
if hasattr(args, 'n_layers') and args.n_layers > 0:
    last_gcl_layer_idx = args.n_layers - 1
    last_gcl_layer_path = f"dynamics.egnn.gcl_layers.{last_gcl_layer_idx}"
    if last_gcl_layer_path in layers_to_unfreeze_in_dynamics and last_gcl_layer_path in mismatched_module_paths_for_lsuv:
        if last_gcl_layer_path not in final_mismatched_paths_for_lsuv: # Avoid duplicates
             final_mismatched_paths_for_lsuv.append(last_gcl_layer_path)

logging.info(f"Final module paths targeted for LSUV (must be trainable and originally mismatched): {final_mismatched_paths_for_lsuv}")

# --- LSUV-like Initialization for Mismatched Layers (NEW) ---
if final_mismatched_paths_for_lsuv: # Use the filtered list
    logging.info("Preparing for LSUV-like initialization of specified trainable mismatched modules...")
    # Ensure dataloaders are available
    if 'train' not in dataloaders or not dataloaders['train']:
        logging.error("Training dataloader not available for LSUV. Skipping LSUV.")
    else:
        try:
            sample_batch_for_lsuv = next(iter(dataloaders['train']))
            lsuv_initialize_modules(
                model, 
                list(final_mismatched_paths_for_lsuv),  # Use the filtered list
                sample_batch_for_lsuv, 
                args, # Pass the main args object
                device, 
                dtype, 
                target_std=0.05,
                target_mean=0.0,
                iterations=10,
                tolerance=0.01
            )
        except Exception as e_lsuv:
            logging.error(f"Error during LSUV-like initialization: {e_lsuv}", exc_info=True)
            logging.error("Continuing without LSUV adjustments due to error.")
else:
    logging.info("No trainable mismatched modules identified for LSUV-like init. Skipping LSUV.")
# --- End of LSUV ---


# --- Setup Optimizer --- 
optim = get_optim(args, model) # get_optim should filter for param.requires_grad == True
logging.info(f"Optimizer created with LR: {args.lr}")
# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters: {trainable_params}")

gradnorm_queue = utils.Queue()
gradnorm_queue.add(100) # Initialize with a more conservative value

# --- Setup EMA --- 
if args.ema_decay > 0:
    ema = diffusion_utils.EMA(args.ema_decay)
    model_ema = copy.deepcopy(model)
    logging.info(f"EMA enabled with decay: {args.ema_decay}")
else:
    ema = None
    model_ema = model
    logging.info("EMA disabled.")

# --- Training Loop --- 
best_val_loss = float('inf')
logging.info("Starting Stage 1 Adaptation training...")
for epoch in range(args.n_epochs):
    model.train()
    model_dp.train()
    start_time = time.time()
    epoch_loss = 0.0
    epoch_nll = 0.0
    epoch_reg = 0.0
    n_batches = 0

    pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.n_epochs} (Stage 1)")
    for i, data in enumerate(pbar):
        # --- CRITICAL NaN CHECK FOR RAW INPUT (NEW) ---
        if torch.isnan(data['positions']).any():
            logging.error(f"CRITICAL NaN DETECTED IN RAW data['positions'] at Epoch {epoch+1}, Batch {i}. Skipping batch.")
            # TODO: Consider saving problematic indices/data for offline analysis
            # problematic_indices = data.get('idx', 'Unknown indices') 
            # logging.error(f"Problematic indices (if available): {problematic_indices}")
            # You might want to exit or implement more robust error handling/data skipping here.
            continue # Skip this batch
        # --- END OF CRITICAL NaN CHECK ---

        # --- Data Preparation --- 
        x = data['positions'].to(device, dtype)
        
        # Check for NaNs in input positions (this check is now somewhat redundant due to the one above, but kept for safety)
        if torch.isnan(x).any():
            logging.error(f"ERROR: NaN detected in input x (positions) at Epoch {epoch+1}, Batch {i}. Skipping batch.")
            # For more detailed debugging, you could print the problematic sample(s) or raise an error.
            # Example: 
            # for sample_idx in range(x.shape[0]):
            #     if torch.isnan(x[sample_idx]).any():
            #         print(f"  NaN found in sample {sample_idx} of batch {i}")
            #         # import pdb; pdb.set_trace() # For interactive debugging
            continue # Skip this batch if NaNs are found

        # Get original atom_mask shape from dataloader
        original_atom_mask_shape = data['atom_mask'].shape
        node_mask = data['atom_mask'].to(device, dtype)
        # Ensure node_mask becomes (B, N, 1)
        if node_mask.ndim == 2: # (B, N)
            node_mask = node_mask.unsqueeze(-1) # Becomes (B, N, 1)
        elif node_mask.ndim == 3 and node_mask.shape[-1] == 1: # Already (B, N, 1)
            pass # Correct shape
        elif node_mask.ndim == 4 and node_mask.shape[-1] == 1 and node_mask.shape[-2] == 1: # (B, N, 1, 1)
            node_mask = node_mask.squeeze(-1) # Becomes (B, N, 1)
        else:
            # If it's some other shape, this might be an issue from dataloader or preprocessing
            logging.warning(f"WARNING: Unexpected original_atom_mask_shape: {original_atom_mask_shape}, current node_mask shape: {node_mask.shape}. Attempting to force to (B,N,1) if possible.")
            # This part might need more robust handling depending on what shapes appear
            if node_mask.ndim > 2 and node_mask.shape[-1] !=1 :
                 node_mask = node_mask.unsqueeze(-1) # Try to add a final dim if not present
            while node_mask.ndim > 3 or (node_mask.ndim == 3 and node_mask.shape[-1] != 1):
                if node_mask.ndim > 2 and node_mask.shape[-1] == 1 and node_mask.shape[-2] == 1: # e.g. (B,N,1,1)
                    node_mask = node_mask.squeeze(-1)
                elif node_mask.ndim > 2 and node_mask.shape[-1] !=1: # e.g. (B,N,D) where D!=1
                    logging.error("ERROR: Cannot easily convert node_mask to (B,N,1)") # Should not happen
                    break 
                else: # Fallback, just try to squeeze last if it's 1, or break
                    if node_mask.shape[-1] == 1 and node_mask.ndim >3:
                        node_mask = node_mask.squeeze(-1)
                    else:
                        break # Avoid infinite loop
            if node_mask.ndim !=3 or node_mask.shape[-1] !=1:
                 logging.warning(f"ALERT: Could not reliably shape node_mask to (B,N,1). Final shape: {node_mask.shape}")

        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        # --- Apply Pre-calculated Dataset-level Normalization (NEW) ---
        # Ensure stats are on the same device as the data
        dataset_x_mean_dev = args.dataset_x_mean.to(device)
        dataset_x_std_dev = args.dataset_x_std.to(device)
        dataset_h_cat_mean_dev = args.dataset_h_cat_mean.to(device)
        dataset_h_cat_std_dev = args.dataset_h_cat_std.to(device)

        # Normalize positions: (x - mean) / std. Unsqueeze to allow broadcasting.
        x_normalized = (x - dataset_x_mean_dev.unsqueeze(0).unsqueeze(0)) / dataset_x_std_dev.unsqueeze(0).unsqueeze(0)
        
        # Normalize one-hot features: (h_cat - mean) / std. Unsqueeze to allow broadcasting.
        one_hot_normalized = (one_hot - dataset_h_cat_mean_dev.unsqueeze(0).unsqueeze(0)) / dataset_h_cat_std_dev.unsqueeze(0).unsqueeze(0)

        # NEW: Ensure masked parts are zero after normalization
        x_normalized = x_normalized * node_mask
        one_hot_normalized = one_hot_normalized * node_mask
        # Charges are not included/normalized based on args.include_charges = False
        # --- End of New Normalization ---

        # DEBUG: Print shapes just before the operation causing the error
        logging.debug(f"Epoch {epoch+1}, Batch {i}: original_atom_mask_shape = {original_atom_mask_shape}, x.shape = {x.shape}, final node_mask.shape = {node_mask.shape}")
        if x.shape[1] != node_mask.shape[1]:
            logging.warning(f"ALERT: Mismatch in n_nodes between x ({x.shape[1]}) and node_mask ({node_mask.shape[1]}) in batch {i}!")

        # The remove_mean_with_mask is usually applied by the model. 
        # If we normalize x to have zero mean here, this call might become redundant or operate on already zero-mean data.
        # Original model structure applies it to z (latent) or x input to diffusion.
        # For now, we pass the normalized x and h to the loss function, which then feeds them to the model.
        # The model's internal normalize() is now an identity op.
        # The model's internal remove_mean_with_mask will still operate on its input (e.g. latents, or normalized x if VAE is bypassed).
        
        # Prepare x and h for the model/loss function
        x_for_model = diffusion_utils.remove_mean_with_mask(x_normalized, node_mask) # Apply per-sample centering
        
        one_hot_for_model = one_hot_normalized * node_mask # Mask one-hot features
        charges_for_model = charges * node_mask # Mask charges

        h_for_model = {'categorical': one_hot_for_model, 'integer': charges_for_model}

        # --- Context Preparation --- 
        context = None # No context for unconditional adaptation

        # --- Loss Calculation --- 
        optim.zero_grad()
        try:
            # Call loss function with context=None, using externally normalized x and h
            nll, reg_term, mean_abs_z = qm9_losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                      x_for_model, h_for_model, node_mask, edge_mask, context=None)
            loss = nll + args.ode_regularization * reg_term

            if torch.isnan(loss):
                logging.warning(f"Warning: NaN loss encountered in batch {i}. Skipping.")
                continue

            # --- Backpropagation & Optimization --- 
            loss.backward()

            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            else:
                grad_norm = -1

            optim.step()

            if ema is not None:
                ema.update_model_average(model_ema, model)

            # Logging
            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_reg += reg_term.item()
            n_batches += 1

            if i % args.n_report_steps == 0:
                pbar.set_postfix({'Loss': loss.item(), 'NLL': nll.item(), 'Reg': reg_term.item(), 'GradNorm': f"{grad_norm:.1f}"})
                if not args.no_wandb:
                    wandb.log({'batch_loss': loss.item(), 'batch_nll': nll.item(), 'batch_reg': reg_term.item(), 'grad_norm': grad_norm}, commit=False)

        except Exception as e:
            logging.error(f"Error during training batch {i}: {e}", exc_info=True)
            continue # Corrected indentation: should be aligned with logging.error within the except block

    # --- End of Epoch --- 
    epoch_duration = time.time() - start_time
    avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
    avg_epoch_nll = epoch_nll / n_batches if n_batches > 0 else 0.0
    avg_epoch_reg = epoch_reg / n_batches if n_batches > 0 else 0.0
    logging.info(f"Stage 1 Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f} (NLL: {avg_epoch_nll:.4f}, Reg: {avg_epoch_reg:.4f}), Duration: {epoch_duration:.2f}s")
    log_dict = {'epoch_loss': avg_epoch_loss, 'epoch_nll': avg_epoch_nll, 'epoch_reg': avg_epoch_reg, 'epoch': epoch}

    # --- Validation --- 
    # Optional: run validation even with small/no val set just to check for errors
    if epoch % args.test_epochs == 0 and dataloaders['val'] and len(dataloaders['val'].dataset) > 0:
        model.eval()
        model_eval = model_ema if ema is not None else model
        model_eval_dp = model_eval

        val_nll = 0.0
        n_val_samples = 0
        with torch.no_grad():
            pbar_val = tqdm(dataloaders['val'], desc=f"Validation Epoch {epoch+1} (Stage 1)")
            for data in pbar_val:
                # --- Validation Data Prep --- 
                x_raw_val = data['positions'].to(device, dtype)
                one_hot_raw_val = data['one_hot'].to(device, dtype)
                charges_val_raw = data['charges'].to(device, dtype) # Use a distinct name for raw charges
                
                batch_size = x_raw_val.size(0)
                # Use distinct variable name for validation node_mask to avoid confusion with training loop
                node_mask_val = data['atom_mask'].to(device, dtype)
                if node_mask_val.ndim == 2: # Ensure (B,N)
                     node_mask_val = node_mask_val # Keep as (B,N) for unsqueeze later if needed by specific functions
                # Reshape to (B, N, 1) for broadcasting and functions expecting it like remove_mean_with_mask
                node_mask_for_ops_val = node_mask_val.unsqueeze(-1) if node_mask_val.ndim == 2 else node_mask_val 
                if node_mask_for_ops_val.ndim != 3 or node_mask_for_ops_val.shape[-1] !=1:
                    # Attempt to fix or log error if shape is still not (B,N,1)
                    logging.warning(f"Valid. node_mask_for_ops_val unexpected shape: {node_mask_for_ops_val.shape}. Re-checking original atom_mask shape: {data['atom_mask'].shape}")
                    # Add robust reshaping if necessary, similar to training loop if problems persist
                    # For now, assume it will be (B,N,1) after unsqueeze if original is (B,N)
                
                edge_mask_val = data['edge_mask'].to(device, dtype)

                # Apply dataset-level normalization to validation data too
                dataset_x_mean_dev = args.dataset_x_mean.to(device)
                dataset_x_std_dev = args.dataset_x_std.to(device)
                dataset_h_cat_mean_dev = args.dataset_h_cat_mean.to(device)
                dataset_h_cat_std_dev = args.dataset_h_cat_std.to(device)

                x_val_normalized = (x_raw_val - dataset_x_mean_dev.unsqueeze(0).unsqueeze(0)) / (dataset_x_std_dev.unsqueeze(0).unsqueeze(0) + 1e-6)
                one_hot_val_normalized = (one_hot_raw_val - dataset_h_cat_mean_dev.unsqueeze(0).unsqueeze(0)) / (dataset_h_cat_std_dev.unsqueeze(0).unsqueeze(0) + 1e-6)
                
                # --- Explicitly mask x_val_normalized BEFORE remove_mean_with_mask (NEW) ---
                x_val_normalized = x_val_normalized * node_mask_for_ops_val
                # --- End of NEW ---

                # The problematic line `x_val_normalized = x_val_normalized * node_mask` should be removed.
                # Correct processing for x and h for the model:
                x_val_for_model = diffusion_utils.remove_mean_with_mask(x_val_normalized, node_mask_for_ops_val) 

                one_hot_val_for_model = one_hot_val_normalized * node_mask_for_ops_val 
                charges_val_for_model = charges_val_raw * node_mask_for_ops_val # Mask raw charges
                
                h_val_for_model = {'categorical': one_hot_val_for_model, 'integer': charges_val_for_model}
                
                # --- Validation Context --- 
                context = None # No context for unconditional validation
                
                # --- Validation Loss Calculation --- 
                try:
                     # Call loss function with context=None
                    nll, _, _ = qm9_losses.compute_loss_and_nll(args, model_eval_dp, nodes_dist, 
                                                                 x_val_for_model, h_val_for_model,
                                                                 node_mask_for_ops_val, # Use the correctly shaped mask
                                                                 edge_mask_val, context=None)
                    if not torch.isnan(nll):
                        val_nll += nll.item() * batch_size
                        n_val_samples += batch_size
                    else:
                        logging.warning("Warning: NaN NLL encountered during validation.")
                except Exception as e:
                    logging.error(f"Error during validation batch: {e}", exc_info=True)
                    # Continue validation

        avg_val_nll = val_nll / n_val_samples if n_val_samples > 0 else float('inf')
        logging.info(f"Stage 1 Validation Epoch {epoch+1} Avg NLL: {avg_val_nll:.4f}")
        log_dict['val_nll'] = avg_val_nll

        # --- Save Checkpoint based on Validation NLL --- 
        if avg_val_nll < best_val_loss: # Still useful to save best model based on NLL on unlabeled data
            best_val_loss = avg_val_nll
            logging.info(f"New best validation NLL: {best_val_loss:.4f}. Saving checkpoint...")
            if args.save_model:
                chkpt_dir = join(args.output_dir, 'checkpoints')
                os.makedirs(chkpt_dir, exist_ok=True)
                # Save adapted model checkpoints (use specific names)
                utils.save_model(model, join(chkpt_dir, 'stage1_adapted_last.npy')) 
                model_ema_save = model_ema if ema is not None else model
                utils.save_model(model_ema_save, join(chkpt_dir, 'stage1_adapted_ema_best.npy'))
                utils.save_model(optim, join(chkpt_dir, 'stage1_optim_best.npy'))
                args.current_epoch = epoch + 1
                with open(join(chkpt_dir, 'stage1_args_best.pickle'), 'wb') as f:
                    pickle.dump(args, f)
                logging.info(f"Stage 1 Checkpoint saved to {chkpt_dir}")
    else:
         # If not validating, still save periodically or at the end?
         # For now, only saving on validation improvement.
         pass

    # Log epoch metrics to wandb
    if not args.no_wandb:
        wandb.log(log_dict, commit=True)


logging.info("Stage 1 Adaptation finished.")
# Save final model state if needed
if args.save_model:
    chkpt_dir = join(args.output_dir, 'checkpoints')
    os.makedirs(chkpt_dir, exist_ok=True)
    final_model_name = 'stage1_adapted_ema_final.npy' if ema is not None else 'stage1_adapted_final.npy'
    final_model_to_save = model_ema if ema is not None else model
    utils.save_model(final_model_to_save, join(chkpt_dir, final_model_name))
    utils.save_model(optim, join(chkpt_dir, 'stage1_optim_final.npy'))
    args.current_epoch = args.n_epochs
    with open(join(chkpt_dir, 'stage1_args_final.pickle'), 'wb') as f:
        pickle.dump(args, f)
    logging.info(f"Final Stage 1 model state saved to {chkpt_dir}")


if not args.no_wandb:
    wandb.finish() 
'''
Stage 1 Adaptation Script: Fine-tuning GeoLDM on Unlabeled Lipid Data.

This script adapts a pre-trained GeoLDM model (e.g., from GEOM-Drugs) to the
chemical space of a target lipid dataset. The lipid data is provided as a
preprocessed PKL file containing molecular conformers and features. This stage
performs unconditional training on the unlabeled lipid data, primarily focusing
on adapting the diffusion model's dynamics.

Key Operations:
1.  Loads command-line arguments and arguments from a pre-trained model.
2.  Initializes logging (console and optional file) and Weights & Biases (WandB).
3.  Loads the unlabeled lipid dataset and computes necessary statistics (n_nodes histogram,
    feature normalization).
4.  Builds the generative model (Latent Diffusion Model with a VAE).
    - Initializes potentially problematic layers conservatively.
    - Loads pre-trained weights, skipping mismatched layers.
    - Configures layer freezing: VAE is frozen, dynamics model is fully unfrozen.
    - LSUV initialization is SKIPPED in this stage.
5.  Sets up the optimizer and Exponential Moving Average (EMA).
6.  Runs the training loop:
    - Iterates through epochs and batches.
    - Computes loss (NLL + ODE regularization).
    - Performs backpropagation and optimizer steps.
    - Optionally applies gradient clipping and EMA updates.
    - Logs training metrics to console and WandB.
7.  Periodically validates the model on a validation split and saves checkpoints
    (best and final).
'''

import sys
import os

# Add the parent directory (project root) to sys.path to allow absolute imports
# This ensures that GeoLDM modules can be imported correctly.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import logging
import argparse
import pickle
import time
from os.path import join
from pathlib import Path
from collections import Counter
import copy

import torch
import numpy as np
import wandb
from tqdm import tqdm

# Local GeoLDM imports
from GeoLDM import utils as project_utils # Renamed to avoid conflict with diffusion_utils
from GeoLDM import lipid_dataset
from GeoLDM.configs.datasets_config import get_dataset_info
from GeoLDM.core.models import get_optim, get_latent_diffusion
from GeoLDM.equivariant_diffusion import utils as diffusion_utils
from GeoLDM.core import losses as qm9_losses

# --- Constants ---
DEFAULT_WANDB_PROJECT = "e3_diffusion_lipid_adapt_stage1"
DEFAULT_EXP_NAME = "geoldm_lipid_adapt_stage1"
PRETRAINED_ARGS_FILENAME = "args.pickle"
DEFAULT_MODEL_FILENAME = "generative_model_ema.npy"


# --- Utility Functions ---
def _get_module_by_path(model_obj: torch.nn.Module, path_str: str) -> torch.nn.Module | None:
    """
    Retrieve a PyTorch module from a model using a dot-separated path string.

    Args:
        model_obj: The parent PyTorch model.
        path_str: Dot-separated path to the target module (e.g., "dynamics.egnn.embedding").

    Returns:
        The PyTorch module if found, otherwise None.
    """
    obj = model_obj
    for part in path_str.split('.'):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif part.isdigit() and isinstance(obj, (torch.nn.ModuleList, torch.nn.Sequential)):
            try:
                obj = obj[int(part)]
            except IndexError:
                logging.error(f"IndexError getting module part '{part}' in path '{path_str}'")
                return None
        else:
            logging.warning(f"Could not get module part '{part}' in path '{path_str}' from object of type {type(obj)}")
            return None
    return obj


# --- Argument Parsing ---
def _add_general_args(parser: argparse.ArgumentParser):
    '''Adds general arguments for experiment setup and logging.'''
    parser.add_argument('--log_file', type=str, default=None,
                        help="Path to save log file. If None, logs to console.")
    parser.add_argument('--exp_name', type=str, default=DEFAULT_EXP_NAME,
                        help="Experiment name for logging and output directories.")
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help="Base directory to save outputs and checkpoints.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--no_cuda', action='store_true', default=False, help="Disable CUDA, run on CPU.")


def _add_data_args(parser: argparse.ArgumentParser):
    '''Adds arguments related to data loading and preprocessing.'''
    parser.add_argument('--unlabeled_data_path', type=str, required=True,
                        help="Path to processed UNLABELED lipid data (e.g., processed_unlabeled_lipids.pkl).")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument('--val_split_ratio', type=float, default=0.01,
                        help="Fraction of unlabeled data to use for validation.")


def _add_model_loading_args(parser: argparse.ArgumentParser):
    '''Adds arguments for loading the pretrained model.'''
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help=f"Path to the folder containing pretrained model state and '{PRETRAINED_ARGS_FILENAME}'.")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_FILENAME,
                        help="Name of the model state_dict file to load from pretrained_path.")


def _add_training_args(parser: argparse.ArgumentParser):
    '''Adds arguments specific to the training/adaptation process.'''
    parser.add_argument('--n_epochs', type=int, default=50, help="Number of adaptation epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help="Exponential Moving Average decay rate (0 to disable EMA).")
    parser.add_argument('--test_epochs', type=int, default=5, help="Run validation every N epochs.")
    parser.add_argument('--save_model', type=eval, default=True, choices=[True, False],
                        help="Whether to save model checkpoints.")
    parser.add_argument('--clip_grad', type=eval, default=True, choices=[True, False],
                        help="Whether to enable gradient clipping.")
    parser.add_argument('--clip_value', type=float, default=1.0, help="Value for gradient clipping if enabled.")
    parser.add_argument('--n_report_steps', type=int, default=50,
                        help="Log training progress every N steps within an epoch.")
    parser.add_argument('--ode_regularization', type=float, default=1e-3, help="Weight for ODE regularization term.")


def _add_wandb_args(parser: argparse.ArgumentParser):
    '''Adds arguments for Weights & Biases integration.'''
    parser.add_argument('--no_wandb', action='store_true', help="Disable WandB logging.")
    parser.add_argument('--wandb_usr', type=str, default=None, help="WandB username or entity.")
    parser.add_argument('--wandb_project', type=str, default=DEFAULT_WANDB_PROJECT,
                        help="WandB project name.")
    parser.add_argument('--online', type=eval, default=True, choices=[True, False],
                        help="WandB online mode (True) or offline mode (False).")


def _add_configurable_model_args(parser: argparse.ArgumentParser):
    '''
    Adds model architecture and diffusion process arguments that can be
    loaded from a pretrained model's args.pickle or overridden by command line.
    Defaults are provided here.
    '''
    # Base dataset configuration
    parser.add_argument('--dataset', type=str, default='geom',
                        help="Base dataset type used for pretraining (e.g., 'geom', 'qm9'). Affects atom decoders.")
    parser.add_argument('--remove_h', action='store_true', help="If true, uses dataset config without hydrogens.")

    # Core Model Architecture
    parser.add_argument('--model', type=str, default='egnn_dynamics', help="Core model type (e.g., 'egnn_dynamics').")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of EGNN layers in the dynamics model.")
    parser.add_argument('--nf', type=int, default=256, help="Number of features for hidden layers.")
    parser.add_argument('--latent_nf', type=int, default=2,
                        help="Number of latent features for the VAE. Critical for checkpoint compatibility.")
    parser.add_argument('--tanh', type=eval, default=True, choices=[True, False], help="Whether to use tanh activation in EGNN.")
    parser.add_argument('--attention', type=eval, default=True, choices=[True, False], help="Whether to use attention in EGNN.")
    parser.add_argument('--norm_constant', type=float, default=1.0, help="Normalization constant for EGNN.")
    parser.add_argument('--inv_sublayers', type=int, default=1, help="Number of invariant sublayers in EGNN.")
    parser.add_argument('--sin_embedding', type=eval, default=False, choices=[True, False],
                        help="Whether to use sinusoidal embeddings.")
    parser.add_argument('--include_charges', type=eval, default=True, choices=[True, False],
                        help="Whether to include atomic charges as a feature.")
    parser.add_argument('--normalization_factor', type=float, default=1.0, help="General normalization factor.")
    parser.add_argument('--aggregation_method', type=str, default='sum', help="Aggregation method in EGNN (e.g., 'sum').")

    # Diffusion Process
    parser.add_argument('--diffusion_steps', type=int, default=1000, help="Number of diffusion timesteps.")
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help="Noise schedule for diffusion (e.g., 'polynomial_2', 'cosine').")
    parser.add_argument('--diffusion_loss_type', type=str, default='l2', help="Loss type for diffusion (e.g., 'l2', 'vlb').")
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5, help="Precision for noise schedule.")

    # VAE / Latent Diffusion Specific
    parser.add_argument('--kl_weight', type=float, default=0.01, help="Weight for KL divergence term in VAE loss.")
    # For Stage 1, train_diffusion is effectively True for the dynamics model.
    # trainable_ae and condition_time are handled via command-line overrides below.
    parser.add_argument('--train_diffusion', type=eval, default=True, choices=[True, False],
                        help="Whether the diffusion process itself is trained (should be True for adaptation).")
    parser.add_argument('--normalize_factors', type=lambda s: [float(item) for item in s.split(',')],
                        default="1.0,1.0,1.0", help="Comma-separated list of normalization factors (e.g., '1,4,10').")

    # Command-line flags for specific overrides of boolean args from pickle
    # VAE trainability is False by default for Stage 1 unless overridden by pickle and not by command line.
    parser.add_argument('--trainable_ae_cmd', dest='trainable_ae_cmd', action='store_true', default=None,
                        help="Command-line override to set trainable_ae to True.")
    parser.add_argument('--no-trainable_ae_cmd', dest='trainable_ae_cmd', action='store_false',
                        help="Command-line override to set trainable_ae to False.")
    # Dynamics model conditioning on time (typically False for unconditional Stage 1 from pretrained GeoLDM)
    parser.add_argument('--condition_time_cmd', dest='condition_time_cmd', action='store_true', default=None,
                        help="Command-line override to set condition_time to True.")
    parser.add_argument('--no-condition_time_cmd', dest='condition_time_cmd', action='store_false',
                        help="Command-line override to set condition_time to False.")


def _create_arg_parser() -> argparse.ArgumentParser:
    '''Creates and returns the ArgumentParser for the script, adding all argument groups.'''
    parser = argparse.ArgumentParser(description="GeoLDM Lipid Adaptation (Stage 1 - Unlabeled Data)")
    _add_general_args(parser)
    _add_data_args(parser)
    _add_model_loading_args(parser)
    _add_training_args(parser)
    _add_wandb_args(parser)
    _add_configurable_model_args(parser)
    return parser


def _finalize_args(args: argparse.Namespace, loaded_pretrained_args: argparse.Namespace | None):
    '''
    Finalizes arguments after loading from pretrained model and command line.
    Handles overrides and sets fixed configurations for Stage 1.
    '''
    # Reconcile trainable_ae: Command line > Pickle > Default (False for Stage 1)
    pickle_trainable_ae = getattr(loaded_pretrained_args, 'trainable_ae', False)
    if args.trainable_ae_cmd is not None:
        args.trainable_ae = args.trainable_ae_cmd
        logging.info(f"'trainable_ae' set to {args.trainable_ae} by command-line argument.")
    else:
        args.trainable_ae = pickle_trainable_ae
        logging.info(f"'trainable_ae' set to {args.trainable_ae} from pickle/default (no command-line override).")
    # Ensure it's False for Stage 1 as intended by current adaptation strategy
    if args.trainable_ae:
        logging.warning("Stage 1 typically keeps VAE frozen. Overriding args.trainable_ae to False.")
        args.trainable_ae = False

    # Reconcile condition_time: Command line > Pickle > Default (False for unconditional Stage 1)
    pickle_condition_time = getattr(loaded_pretrained_args, 'condition_time', False)
    if args.condition_time_cmd is not None:
        args.condition_time = args.condition_time_cmd
        logging.info(f"'condition_time' set to {args.condition_time} by command-line argument.")
    else:
        args.condition_time = pickle_condition_time
        logging.info(f"'condition_time' set to {args.condition_time} from pickle/default (no command-line override).")
    # Ensure it's False for Stage 1 (unconditional generation)
    if args.condition_time:
        logging.warning("Stage 1 is for unconditional generation. Overriding args.condition_time to False.")
        args.condition_time = False
    
    # Clean up temporary command-line override attributes
    delattr(args, 'trainable_ae_cmd')
    delattr(args, 'condition_time_cmd')

    # Override latent_nf to 2, based on GeoLDM README for pretrained model compatibility.
    desired_latent_nf = 2
    current_latent_nf = getattr(args, 'latent_nf', 'Not set')
    if current_latent_nf != desired_latent_nf:
        logging.info(f"Overriding args.latent_nf from {current_latent_nf} to {desired_latent_nf} "
                     "for GeoLDM pretrained model compatibility (README states latent_nf=2).")
    args.latent_nf = desired_latent_nf

    # Ensure probabilistic_model is set for GeoLDM framework (should be diffusion)
    if not hasattr(args, 'probabilistic_model') or args.probabilistic_model != 'diffusion':
        logging.info("Setting/forcing args.probabilistic_model = 'diffusion' for compatibility.")
    args.probabilistic_model = 'diffusion'
    
    # Set context_node_nf to 0 for unconditional Stage 1
    if getattr(args, 'context_node_nf', -1) != 0: # Use -1 as a sentinel if not present
        logging.info("Setting args.context_node_nf to 0 for unconditional Stage 1 adaptation.")
    args.context_node_nf = 0

    # External normalization is used, so model's internal normalization factors should be identity.
    identity_norm_factors = [1.0, 1.0, 1.0]
    if getattr(args, 'normalize_factors', None) != identity_norm_factors:
        logging.info(f"Setting args.normalize_factors to {identity_norm_factors} for external dataset normalization.")
    args.normalize_factors = identity_norm_factors


def _parse_and_prepare_args() -> tuple[argparse.Namespace, torch.device]:
    '''
    Parses command-line arguments, loads arguments from the pretrained model,
    reconciles them, and sets up essential configurations.

    Returns:
        A tuple containing the finalized arguments (Namespace) and the device (torch.device).
    '''
    parser = _create_arg_parser()
    # Initial parse to get pretrained_path for loading original args
    temp_args, _ = parser.parse_known_args()

    loaded_pretrained_args = None
    if not temp_args.pretrained_path:
        logging.error("--pretrained_path is a required argument. Exiting.")
        sys.exit(1)

    pretrained_args_path = join(temp_args.pretrained_path, PRETRAINED_ARGS_FILENAME)
    try:
        with open(pretrained_args_path, 'rb') as f:
            loaded_pretrained_args = pickle.load(f)
        logging.info(f"Successfully loaded arguments from pretrained model: {pretrained_args_path}")
    except FileNotFoundError:
        logging.error(f"Pretrained arguments file not found at {pretrained_args_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading pretrained args from {pretrained_args_path}: {e}. Exiting.")
        sys.exit(1)

    # Set defaults in the parser from the loaded pretrained_args.
    # Command-line arguments will override these if provided.
    default_overrides = {
        k: v for k, v in vars(loaded_pretrained_args).items()
        if hasattr(parser.get_default(k) if parser.get_default(k) is not None else None, '__dict__') # Check if it's a namespace
           or parser.get_default(k) != v # Only override if different, or for complex types
    }
    # Filter out keys that are specific parser actions or not meant to be set this way
    critical_keys_to_set_defaults = [
        'model', 'n_layers', 'nf', 'latent_nf', 'tanh', 'attention',
        'norm_constant', 'inv_sublayers', 'sin_embedding', 'include_charges',
        'normalization_factor', 'aggregation_method', 'dataset', 'remove_h',
        'diffusion_steps', 'diffusion_noise_schedule', 'diffusion_loss_type',
        'diffusion_noise_precision', 'normalize_factors',
        'kl_weight', 'train_diffusion', 'condition_time'
    ]
    loaded_defaults_for_parser = {}
    for key in critical_keys_to_set_defaults:
        if hasattr(loaded_pretrained_args, key):
            loaded_defaults_for_parser[key] = getattr(loaded_pretrained_args, key)
        else:
            logging.warning(f"Key '{key}' not found in pretrained args. Script/argparse default will be used.")
    parser.set_defaults(**loaded_defaults_for_parser)

    # Final parse of all arguments
    args = parser.parse_args()

    # Finalize arguments (handle overrides, set Stage 1 specifics)
    _finalize_args(args, loaded_pretrained_args)
    
    # Setup device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    return args, device


# --- Setup Functions ---
def _setup_environment(args: argparse.Namespace, device: torch.device):
    '''Sets up random seeds and creates output directories.'''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    logging.info(f"Using device: {device}")
    logging.info(f"Set random seed: {args.seed}")

    # Create output directories
    args.output_dir = join(args.output_dir, args.exp_name) # Final experiment output path
    args.checkpoints_dir = join(args.output_dir, 'checkpoints') # Specific checkpoints path
    # project_utils.create_folders(args) # Original call had issues with Namespace
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Checkpoints directory: {args.checkpoints_dir}")


def _setup_logging(log_file_path: str | None):
    '''Initializes logging to console and optionally to a file.'''
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if log_file_path:
        try:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode='a')
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            logging.error(f"Failed to set up file logger at {log_file_path}: {e}")
    else:
        logging.info("Logging to console only (no --log_file specified).")


def _setup_wandb(args: argparse.Namespace):
    '''Initializes Weights & Biases if enabled.'''
    if args.no_wandb:
        logging.info("WandB logging is disabled.")
        return

    mode = 'online' if args.online else 'offline'
    wandb_entity = args.wandb_usr if args.wandb_usr else project_utils.get_wandb_username(args.wandb_usr)
    
    try:
        wandb.init(
            entity=wandb_entity,
            name=args.exp_name,
            project=args.wandb_project,
            config=vars(args), # Pass all args as config
            settings=wandb.Settings(_disable_stats=True),
            reinit=True,
            mode=mode
        )
        # wandb.save('*.txt') # Example: save text files in the run directory
        logging.info(f"WandB initialized: project='{args.wandb_project}', name='{args.exp_name}', mode='{mode}'.")
    except Exception as e:
        logging.error(f"Error initializing WandB: {e}. Continuing without WandB.")
        args.no_wandb = True # Disable further WandB calls

    logging.info("Final Effective Arguments for Stage 1 Adaptation:")
    for k, v in sorted(vars(args).items()):
        logging.info(f"  {k}: {v}")


# --- Data Handling ---
def _compute_dataset_statistics(args: argparse.Namespace, train_dataset: torch.utils.data.Dataset, device: torch.device):
    '''
    Computes and stores dataset normalization statistics (mean/std for positions and one-hot features)
    in the args object.
    '''
    logging.info("Calculating dataset normalization statistics for positions and one-hot features...")
    all_positions_list, all_one_hot_list = [], []

    for i in tqdm(range(len(train_dataset)), desc="Processing samples for stats"):
        sample_data = train_dataset[i]
        if not all(k in sample_data for k in ['positions', 'one_hot', 'atom_mask']):
            logging.warning(f"Sample {i} is missing required keys ('positions', 'one_hot', or 'atom_mask'). Skipping for stats.")
            continue

        pos_tensor = sample_data['positions'].to(device)      # Shape: (1, N, 3)
        one_hot_tensor = sample_data['one_hot'].to(device)  # Shape: (1, N, num_atom_types)
        atom_mask_tensor = sample_data['atom_mask'].to(device) # Shape: (1, N, 1)

        # Squeeze out the batch dimension (1) and the feature dimension (1) for the mask
        valid_indices = atom_mask_tensor.squeeze(0).squeeze(-1).bool() # Shape: (N)

        if torch.any(valid_indices):
            # Squeeze out batch dim from positions and one_hot before masking
            pos_tensor_squeezed = pos_tensor.squeeze(0)          # Shape: (N, 3)
            one_hot_tensor_squeezed = one_hot_tensor.squeeze(0)  # Shape: (N, num_atom_types)

            masked_pos_tensor = pos_tensor_squeezed[valid_indices]
            masked_one_hot_tensor = one_hot_tensor_squeezed[valid_indices]

            if masked_pos_tensor.numel() > 0:
                all_positions_list.append(masked_pos_tensor.cpu().numpy())
            if masked_one_hot_tensor.numel() > 0:
                all_one_hot_list.append(masked_one_hot_tensor.cpu().numpy())
        else:
            logging.warning(f"Sample {i} has no valid atoms per atom_mask. Skipping for stats.")

    if not all_positions_list or not all_one_hot_list:
        logging.error("No valid data found after masking for normalization statistics. Exiting.")
        sys.exit(1)

    all_positions_np = np.concatenate(all_positions_list, axis=0)
    all_one_hot_np = np.concatenate(all_one_hot_list, axis=0)

    # Filter out NaNs/Infs that might have slipped through (e.g., from RDKit errors)
    all_positions_np = all_positions_np[~np.isnan(all_positions_np).any(axis=1)]
    all_positions_np = all_positions_np[~np.isinf(all_positions_np).any(axis=1)]
    all_one_hot_np = all_one_hot_np[~np.isnan(all_one_hot_np).any(axis=1)]
    all_one_hot_np = all_one_hot_np[~np.isinf(all_one_hot_np).any(axis=1)]
    
    if all_positions_np.shape[0] == 0 or all_one_hot_np.shape[0] == 0:
        logging.error("No valid data after NaN/Inf filtering for normalization stats. Exiting.")
        sys.exit(1)

    args.dataset_x_mean = torch.from_numpy(np.mean(all_positions_np, axis=0)).float()
    args.dataset_x_std = torch.from_numpy(np.std(all_positions_np, axis=0)).float()
    args.dataset_x_std[args.dataset_x_std < 1e-6] = 1.0 # Prevent division by zero

    args.dataset_h_cat_mean = torch.from_numpy(np.mean(all_one_hot_np, axis=0)).float()
    args.dataset_h_cat_std = torch.from_numpy(np.std(all_one_hot_np, axis=0)).float()
    args.dataset_h_cat_std[args.dataset_h_cat_std < 1e-6] = 1.0

    logging.info(f"  Dataset x_mean (coords): {args.dataset_x_mean.tolist()}")
    logging.info(f"  Dataset x_std (coords): {args.dataset_x_std.tolist()}")
    logging.info(f"  Dataset h_cat_mean (features): {args.dataset_h_cat_mean.tolist()}")
    logging.info(f"  Dataset h_cat_std (features): {args.dataset_h_cat_std.tolist()}")


def _load_and_prepare_data(args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    '''
    Loads dataset_info, creates dataloaders, calculates node histogram,
    and computes dataset normalization statistics.

    Returns:
        A tuple containing dataloaders (dict) and dataset_info (dict).
    '''
    dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
    logging.info(f"Initial dataset_info for '{args.dataset}' (remove_h={args.remove_h}, include_charges by arg: {args.include_charges})")

    dataloaders = lipid_dataset.get_dataloaders(
        data_path=args.unlabeled_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio
    )
    if not dataloaders or not dataloaders.get('train'):
        logging.error("Failed to create dataloaders or training dataloader is missing. Exiting.")
        sys.exit(1)

    train_dataset = dataloaders['train'].dataset
    if not train_dataset or len(train_dataset) == 0:
        logging.error("Training dataset is empty. Cannot compute statistics. Exiting.")
        sys.exit(1)

    logging.info("Calculating n_nodes histogram from unlabeled training data...")
    n_nodes_counts = Counter()
    for i in tqdm(range(len(train_dataset)), desc="Counting nodes per sample"):
        sample = train_dataset[i]
        num_nodes = sample.get('positions', torch.empty(0, 0, 0)).shape[1] # (1,N,3) -> N
        if num_nodes > 0:
            n_nodes_counts[num_nodes] += 1
        elif 'num_atoms' in sample: # Fallback
            num_atoms_val = sample['num_atoms']
            num_nodes = int(num_atoms_val.item()) if isinstance(num_atoms_val, torch.Tensor) else int(num_atoms_val)
            if num_nodes > 0: n_nodes_counts[num_nodes] += 1
            else: logging.warning(f"Sample {i} has 0 atoms based on 'num_atoms' fallback.")
        else:
            logging.warning(f"Could not determine num_nodes for sample {i}. Skipping for histogram.")

    if not n_nodes_counts:
        logging.error("n_nodes histogram is empty after processing dataset. Check data. Exiting.")
        sys.exit(1)
    dataset_info['n_nodes'] = dict(n_nodes_counts)
    logging.info(f"Updated n_nodes histogram. Min: {min(n_nodes_counts.keys()) if n_nodes_counts else 'N/A'}, "
                 f"Max: {max(n_nodes_counts.keys()) if n_nodes_counts else 'N/A'}, "
                 f"Unique sizes: {len(n_nodes_counts)}")

    _compute_dataset_statistics(args, train_dataset, device)
    
    return dataloaders, dataset_info


# --- Model Building and Initialization ---
def _initialize_problematic_layers_conservatively(model: torch.nn.Module):
    '''
    Applies conservative normal initialization (small std) to specific layers
    that might be prone to size mismatches or require careful re-initialization
    if not fully loaded from a checkpoint.
    '''
    logging.info("Applying conservative initialization (normal std=0.01) to specific layers...")
    layers_to_init = {
        "dynamics.egnn.embedding": None,
        "dynamics.egnn.embedding_out": None,
        "vae.encoder.final_mlp.2": None, # Path for a specific Linear layer in a Sequential final_mlp
        "vae.decoder.egnn.embedding": None
    }
    
    for path_str in layers_to_init.keys():
        layer = _get_module_by_path(model, path_str) # Renamed call
        if layer and isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01) 


def _load_pretrained_weights(model: torch.nn.Module, pretrained_dir: str, model_filename: str, device: torch.device):
    '''
    Loads pretrained weights from a checkpoint file into the model.
    Handles potential "module." prefix from DataParallel training and logs mismatches.
    '''
    model_path_full = join(pretrained_dir, model_filename)
    logging.info(f"Attempting to load pretrained weights from: {model_path_full}")
    try:
        state_dict_checkpoint = torch.load(model_path_full, map_location=device)
        
        # Remove "module." prefix if present (from DataParallel/DistributedDataParallel)
        if any(key.startswith('module.') for key in state_dict_checkpoint.keys()):
            logging.info("Removing 'module.' prefix from checkpoint state_dict keys.")
            state_dict_checkpoint = {k[len('module.'):]: v for k, v in state_dict_checkpoint.items()}
        
        current_model_sd = model.state_dict()
        final_sd_to_load = {}
        skipped_mismatch_details = []
        mismatched_module_roots = set()

        for k_ckpt, v_ckpt in state_dict_checkpoint.items():
            if k_ckpt in current_model_sd:
                v_model = current_model_sd[k_ckpt]
                if v_model.shape == v_ckpt.shape:
                    final_sd_to_load[k_ckpt] = v_ckpt
                else:
                    detail = (f"'{k_ckpt}': Model shape {v_model.shape}, Ckpt shape {v_ckpt.shape}")
                    skipped_mismatch_details.append(detail)
                    # Get the root module path (e.g., "vae.encoder.final_mlp.2")
                    mismatched_module_roots.add('.'.join(k_ckpt.split('.')[:-1])) 
            else:
                logging.warning(f"  Key '{k_ckpt}' from checkpoint not found in current model. Skipping.")
        
        # Load with strict=False to allow partial loading
        missing_keys, unexpected_keys = model.load_state_dict(final_sd_to_load, strict=False)
        
        logging.info(f"Successfully loaded {len(final_sd_to_load)} parameters from checkpoint.")
        if skipped_mismatch_details:
            logging.warning(f"  Skipped {len(skipped_mismatch_details)} parameter tensors from "
                            f"{len(mismatched_module_roots)} unique modules due to SHAPE MISMATCHES:")
            for detail_msg in skipped_mismatch_details:
                logging.warning(f"    - {detail_msg}")
        if missing_keys:
            logging.warning(f"  {len(missing_keys)} keys were MISSING in the checkpoint but expected by the model (will keep current model init):")
            # for key in missing_keys: logging.warning(f"    - {key}") # Can be verbose
        if unexpected_keys:
            logging.warning(f"  {len(unexpected_keys)} keys were UNEXPECTED (in checkpoint but not in model):")
            # for key in unexpected_keys: logging.warning(f"    - {key}") # Can be verbose

    except FileNotFoundError:
        logging.error(f"Pretrained model file not found at {model_path_full}. Critical error. Exiting.")
        sys.exit(1)
    except Exception as e_load:
        logging.error(f"Critical error loading pretrained weights: {e_load}", exc_info=True)
        sys.exit(1)


def _configure_model_freezing_for_stage1(model: torch.nn.Module, args: argparse.Namespace):
    '''
    Configures model parameter freezing for Stage 1 adaptation.
    - VAE (first_stage_model) is always frozen (args.trainable_ae is forced to False for Stage 1).
    - Dynamics model (diffusion model) is fully unfrozen.
    '''
    if hasattr(model, 'vae') and model.vae is not None:
        if not args.trainable_ae: # Should be True due to _finalize_args forcing it
            logging.info("Freezing all parameters in model.vae (Autoencoder) as args.trainable_ae is False for Stage 1.")
            for param in model.vae.parameters():
                param.requires_grad = False
        else: # This case should not be hit if _finalize_args works as intended for Stage 1
            logging.warning("model.vae parameters will be trainable as args.trainable_ae is True (unusual for Stage 1).")
            for param in model.vae.parameters():
                param.requires_grad = True 
    else:
        logging.warning("model.vae not found, skipping VAE freezing configuration.")

    if hasattr(model, 'dynamics') and model.dynamics is not None:
        logging.info("Unfreezing all parameters in model.dynamics for Stage 1 adaptation.")
        unfrozen_dynamics_count = 0
        for param in model.dynamics.parameters():
            param.requires_grad = True
            unfrozen_dynamics_count += param.numel()
        logging.info(f"Successfully unfroze all {unfrozen_dynamics_count} parameters in model.dynamics.")
    else:
        logging.warning("model.dynamics not found, skipping dynamics unfreezing configuration.")


def _build_model_and_optimizer_ema(args: argparse.Namespace, device: torch.device,
                                   dataset_info: dict, dataloaders: dict
                                   ) -> tuple[torch.nn.Module, any, torch.optim.Optimizer, diffusion_utils.EMA | None, torch.nn.Module | None]:
    '''
    Initializes the model, loads pretrained weights, configures layer freezing,
    and creates the optimizer and EMA objects.
    LSUV initialization is skipped for Stage 1.
    '''
    logging.info(f"Instantiating model (train_diffusion={args.train_diffusion}, trainable_ae={args.trainable_ae})...")
    # Dataloader_train is passed for nodes_dist and potentially prop_dist if used by model.
    model, nodes_dist, _ = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
    logging.info("Model components instantiated.")

    # Fallback conservative initialization for layers that might not load from checkpoint.
    _initialize_problematic_layers_conservatively(model)
    
    # Load pretrained weights for the entire model.
    _load_pretrained_weights(model, args.pretrained_path, args.model_name, device)
    model.to(device) # Ensure model is on the correct device after loading.

    # Configure freezing: VAE frozen, Dynamics unfrozen for Stage 1.
    _configure_model_freezing_for_stage1(model, args)

    logging.info("Skipping LSUV initialization for Stage 1 as dynamics model is fully unfrozen "
                 "from pretrained weights and VAE is fixed.")

    optimizer = get_optim(args, model) # get_optim filters for param.requires_grad == True
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Optimizer created with LR: {args.lr}. Trainable parameters: {trainable_params_count}")

    ema_updater, model_ema_obj = None, None
    if args.ema_decay > 0:
        ema_updater = diffusion_utils.EMA(args.ema_decay)
        model_ema_obj = copy.deepcopy(model)
        model_ema_obj.to(device) # Ensure EMA model is also on device
        logging.info(f"EMA enabled with decay: {args.ema_decay}")
    else:
        logging.info("EMA disabled (ema_decay <= 0).")

    return model, nodes_dist, optimizer, ema_updater, model_ema_obj


# --- Training and Validation ---
def _prepare_batch_for_model(data_batch: dict, args: argparse.Namespace, device: torch.device
                             ) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    '''
    Prepares a raw data batch for model input by moving to device, normalizing,
    and applying mean centering for coordinates.
    '''
    x_raw = data_batch['positions'].to(device, torch.float32)
    node_mask_raw = data_batch['atom_mask'].to(device, torch.float32)
    if node_mask_raw.ndim == 2:  # Ensure (B, N, 1)
        node_mask_raw = node_mask_raw.unsqueeze(-1)
    
    edge_mask_batch = data_batch['edge_mask'].to(device, torch.float32)
    one_hot_raw = data_batch['one_hot'].to(device, torch.float32)
    charges_raw = data_batch['charges'].to(device, torch.float32)

    # Apply pre-calculated dataset-level normalization
    # Move mean/std to device and unsqueeze for broadcasting: (1, 1, D_coord/D_feat)
    dataset_x_mean_dev = args.dataset_x_mean.to(device).unsqueeze(0).unsqueeze(0)
    dataset_x_std_dev = args.dataset_x_std.to(device).unsqueeze(0).unsqueeze(0)
    dataset_h_cat_mean_dev = args.dataset_h_cat_mean.to(device).unsqueeze(0).unsqueeze(0)
    dataset_h_cat_std_dev = args.dataset_h_cat_std.to(device).unsqueeze(0).unsqueeze(0)

    x_norm = (x_raw - dataset_x_mean_dev) / (dataset_x_std_dev + 1e-6)
    one_hot_norm = (one_hot_raw - dataset_h_cat_mean_dev) / (dataset_h_cat_std_dev + 1e-6)
    
    # Mask features after normalization
    x_norm = x_norm * node_mask_raw
    one_hot_norm = one_hot_norm * node_mask_raw
    charges_masked = charges_raw * node_mask_raw # Charges are typically not normalized with mean/std

    # Center coordinates
    x_model_input = diffusion_utils.remove_mean_with_mask(x_norm, node_mask_raw)
    h_model_input = {'categorical': one_hot_norm, 'integer': charges_masked}
    
    return x_model_input, h_model_input, node_mask_raw, edge_mask_batch


def _train_one_epoch(epoch_num: int, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                     optimizer: torch.optim.Optimizer, ema_updater: diffusion_utils.EMA | None,
                     model_ema: torch.nn.Module | None, args: argparse.Namespace, device: torch.device,
                     nodes_dist: any, grad_norm_queue: project_utils.Queue
                     ) -> tuple[float, float, float]:
    '''Trains the model for one epoch.'''
    model.train()
    total_loss, total_nll, total_reg = 0.0, 0.0, 0.0
    batches_processed = 0

    pbar_desc = f"Epoch {epoch_num}/{args.n_epochs} (Stage 1 Train)"
    pbar = tqdm(dataloader, desc=pbar_desc, unit="batch")

    for i, data_batch in enumerate(pbar):
        if torch.isnan(data_batch['positions']).any():
            logging.error(f"NaN detected in raw input positions at Epoch {epoch_num}, Batch {i}. Skipping batch.")
            continue

        x_input, h_input, node_mask, edge_mask = _prepare_batch_for_model(data_batch, args, device)
        
        optimizer.zero_grad()
        try:
            nll, reg_term, _ = qm9_losses.compute_loss_and_nll(
                args, model, nodes_dist,
                x_input, h_input, node_mask, edge_mask, context=None # Stage 1 is unconditional
            )
            loss = nll + args.ode_regularization * reg_term

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected at training batch {i}, epoch {epoch_num}. Skipping batch update.")
                continue
            
            loss.backward()
            
            grad_norm = -1.0
            if args.clip_grad:
                grad_norm = project_utils.gradient_clipping(model, grad_norm_queue)
            
            optimizer.step()

            if ema_updater is not None and model_ema is not None:
                ema_updater.update_model_average(model_ema, model)

            total_loss += loss.item()
            total_nll += nll.item()
            total_reg += reg_term.item()
            batches_processed += 1

            if i % args.n_report_steps == 0 or i == len(dataloader) - 1:
                pbar.set_postfix({
                    'Loss': loss.item(), 'NLL': nll.item(), 
                    'Reg': reg_term.item(), 'GradNorm': f"{grad_norm:.2f}"
                })
                if not args.no_wandb:
                    wandb.log({
                        'train_batch_loss_s1': loss.item(), 
                        'train_batch_nll_s1': nll.item(),
                        'train_batch_reg_s1': reg_term.item(), 
                        'train_batch_grad_norm_s1': grad_norm
                    }, commit=False) # Commit at epoch end
        
        except Exception as e:
            logging.error(f"Error during training batch {i} in epoch {epoch_num}: {e}", exc_info=True)
            continue # Continue to the next batch

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    avg_nll = total_nll / batches_processed if batches_processed > 0 else 0.0
    avg_reg = total_reg / batches_processed if batches_processed > 0 else 0.0
    return avg_loss, avg_nll, avg_reg


@torch.no_grad()
def _validate_one_epoch(model_to_eval: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                        args: argparse.Namespace, device: torch.device, nodes_dist: any
                        ) -> float:
    '''Validates the model for one epoch, returning average NLL.'''
    model_to_eval.eval()
    total_val_nll = 0.0
    samples_processed = 0

    pbar_val = tqdm(dataloader, desc="Validation (Stage 1)", unit="batch")
    for data_batch in pbar_val:
        if torch.isnan(data_batch['positions']).any():
            logging.warning("NaN detected in raw validation input positions. Skipping batch.")
            continue

        x_input, h_input, node_mask, edge_mask = _prepare_batch_for_model(data_batch, args, device)
        
        try:
            nll, _, _ = qm9_losses.compute_loss_and_nll(
                args, model_to_eval, nodes_dist,
                x_input, h_input, node_mask, edge_mask, context=None
            )
            if not (torch.isnan(nll) or torch.isinf(nll)):
                total_val_nll += nll.item() * x_input.size(0) # NLL is per sample, sum then average
                samples_processed += x_input.size(0)
            else:
                logging.warning("NaN/Inf NLL during validation batch.")
        except Exception as e:
            logging.error(f"Error during validation batch: {e}", exc_info=True)
            continue
                
    avg_val_nll = total_val_nll / samples_processed if samples_processed > 0 else float('inf')
    return avg_val_nll


def _save_checkpoint(model_to_save: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     args_config: argparse.Namespace, epoch_idx: int, is_best: bool = False, is_final: bool = False):
    '''Saves a model checkpoint, optimizer state, and arguments.'''
    if not args_config.save_model:
        return

    suffix = "best" if is_best else ("final" if is_final else f"epoch_{epoch_idx+1}")
    
    # Save model weights
    project_utils.save_model(model_to_save, join(args_config.checkpoints_dir, f'stage1_adapted_model_{suffix}.npy'))
    
    # For best and final checkpoints, also save optimizer and args
    if is_best or is_final:
        project_utils.save_model(optimizer, join(args_config.checkpoints_dir, f'stage1_optim_{suffix}.npy'))
        
        # Save args, adding current epoch
        args_to_save = copy.deepcopy(args_config)
        args_to_save.completed_epoch_s1 = epoch_idx + 1 # 1-indexed completed epoch
        args_path = join(args_config.checkpoints_dir, f'stage1_args_{suffix}.pickle')
        with open(args_path, 'wb') as f:
            pickle.dump(args_to_save, f)
        logging.info(f"Args for Stage 1 checkpoint '{suffix}' saved to {args_path}")

    logging.info(f"Stage 1 Checkpoint '{suffix}' (Epoch {epoch_idx+1}) saved to {args_config.checkpoints_dir}")


def run_training_loop(args: argparse.Namespace, model: torch.nn.Module, model_ema: torch.nn.Module | None,
                      dataloaders: dict, nodes_dist: any, optimizer: torch.optim.Optimizer,
                      ema_updater: diffusion_utils.EMA | None, device: torch.device):
    '''Main training and validation loop for Stage 1 adaptation.'''
    best_val_nll = float('inf')
    grad_norm_queue = project_utils.Queue() # For gradient norm logging
    grad_norm_queue.add(100.0) # Initialize with a relatively high value

    logging.info("Starting Stage 1 Adaptation training loop...")
    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()
        
        avg_loss, avg_nll, avg_reg = _train_one_epoch(
            epoch + 1, model, dataloaders['train'], optimizer, ema_updater, model_ema,
            args, device, nodes_dist, grad_norm_queue
        )
        epoch_duration = time.time() - epoch_start_time
        
        logging.info(
            f"Stage 1 Epoch {epoch+1}/{args.n_epochs} Summary: "
            f"Avg Loss: {avg_loss:.4f} (NLL: {avg_nll:.4f}, Reg: {avg_reg:.4f}), "
            f"Duration: {epoch_duration:.2f}s"
        )
        
        wandb_log_data = {
            'epoch_s1': epoch + 1,
            'train_epoch_loss_s1': avg_loss,
            'train_epoch_nll_s1': avg_nll,
            'train_epoch_reg_s1': avg_reg,
            'epoch_duration_s1_sec': epoch_duration
        }

        # Validation
        perform_validation = (epoch + 1) % args.test_epochs == 0
        val_dl = dataloaders.get('val')
        
        if perform_validation and val_dl and len(val_dl.dataset) > 0:
            model_for_eval = model_ema if ema_updater is not None else model
            avg_val_nll = _validate_one_epoch(model_for_eval, val_dl, args, device, nodes_dist)
            logging.info(f"  Validation NLL (Epoch {epoch+1}): {avg_val_nll:.4f}")
            wandb_log_data['val_epoch_nll_s1'] = avg_val_nll

            if avg_val_nll < best_val_nll:
                best_val_nll = avg_val_nll
                logging.info(f"    New best validation NLL for Stage 1: {best_val_nll:.4f}. Saving checkpoint.")
                _save_checkpoint(model_for_eval, optimizer, args, epoch, is_best=True)
        elif perform_validation:
            logging.info(f"Skipping validation for epoch {epoch+1}: No validation data or dataloader is empty.")
        
        # Save last model checkpoint periodically (optional, can be frequent)
        # if (epoch + 1) % args.save_every_n_epochs == 0: # Example: save every 10 epochs
        # _save_checkpoint(model_ema if ema_updater else model, optimizer, args, epoch, is_best=False)

        if not args.no_wandb:
            wandb.log(wandb_log_data, commit=True) # Commit all epoch metrics at once
            
    logging.info("Stage 1 Adaptation training finished.")
    # Save the final model
    final_model_to_save = model_ema if ema_updater is not None else model
    _save_checkpoint(final_model_to_save, optimizer, args, args.n_epochs - 1, is_final=True)


# --- Main Execution ---
def main():
    '''Main execution function for Stage 1 adaptation.'''
    # --- Setup ---
    args, device = _parse_and_prepare_args()
    _setup_logging(args.log_file)
    _setup_environment(args, device) # Seed, directories
    _setup_wandb(args) # Must be after logging and args are finalized

    # --- Data Loading and Preparation ---
    dataloaders, dataset_info = _load_and_prepare_data(args, device)
    
    # --- Model, Optimizer, EMA Initialization ---
    model, nodes_dist, optimizer, ema_updater, model_ema = \
        _build_model_and_optimizer_ema(args, device, dataset_info, dataloaders)
    
    # --- Training ---
    run_training_loop(args, model, model_ema, dataloaders, nodes_dist,
                      optimizer, ema_updater, device)

    # --- Finalization ---
    if not args.no_wandb:
        wandb.finish()
    logging.info("Script execution completed successfully.")


if __name__ == "__main__":
    main() 
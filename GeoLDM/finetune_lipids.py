'''
Fine-tuning script for GeoLDM on lipid data with transfection_score conditioning.

Based on main_geom_drugs.py and train_test.py from the original GeoLDM codebase.
'''

import sys
import os
# Add the parent directory (project root) to sys.path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
from GeoLDM.configs.datasets_config import geom_with_h
import copy
import GeoLDM.utils as utils
import argparse
import wandb
from os.path import join
from GeoLDM.core.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from GeoLDM.equivariant_diffusion import en_diffusion
from GeoLDM.equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle
from GeoLDM.core.utils import prepare_context, compute_mean_mad
from GeoLDM.train_test import train_epoch, test, save_and_sample_chain, analyze_and_save
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import logging
import random
from torch.utils.data import DataLoader

# Local imports
import GeoLDM.lipid_dataset as lipid_dataset
from GeoLDM.configs.datasets_config import get_dataset_info

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
from collections import Counter # For dataset stats calculation
from datetime import datetime
from torch.distributions import Categorical
import torch.nn as nn
from GeoLDM.egnn.egnn_new import EquivariantBlock, GCL
from GeoLDM.core.bond_analyze import get_bond_order, allowed_bonds
from rdkit.Chem import Descriptors
from rdkit.Chem import ChemicalFeatures
import json
from torch.utils.data import DataLoader
from GeoLDM.core.rdkit_functions import build_molecule
import traceback # Add this import at the top of the file with other imports

# --- LSUV Global and Helper Functions (Copied and adapted for Stage 2) ---
captured_output_lsuv = None

def hook_fn_lsuv(module, input, output):
    global captured_output_lsuv
    captured_output_lsuv = output

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

def is_suitable_for_lsuv(layer):
    """Check if a layer is suitable for LSUV initialization."""
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        return True
    if isinstance(layer, nn.Sequential):
        # Check if any submodule is suitable
        return any(is_suitable_for_lsuv(submodule) for submodule in layer)
    return False

def get_lsuv_modules(model, paths):
    """Get modules suitable for LSUV initialization."""
    modules = []
    for path in paths:
        layer = get_layer_by_path(model, path)
        if layer is not None:
            if isinstance(layer, EquivariantBlock):
                # For EquivariantBlock, we need to initialize its internal GCL layers
                for i in range(layer.n_layers):
                    gcl_path = f"{path}.gcl_{i}"
                    gcl_layer = get_layer_by_path(model, gcl_path)
                    if gcl_layer is not None:
                        # Add the GCL layer's internal components
                        if hasattr(gcl_layer, 'edge_mlp'):
                            modules.append((f"{gcl_path}.edge_mlp", gcl_layer.edge_mlp))
                        if hasattr(gcl_layer, 'node_mlp'):
                            modules.append((f"{gcl_path}.node_mlp", gcl_layer.node_mlp))
                        if hasattr(gcl_layer, 'att_mlp') and gcl_layer.attention:
                            modules.append((f"{gcl_path}.att_mlp", gcl_layer.att_mlp))
                # Add the equivariant update layer
                if hasattr(layer, 'gcl_equiv'):
                    modules.append((f"{path}.gcl_equiv", layer.gcl_equiv))
            elif isinstance(layer, nn.Sequential):
                # Handle Sequential layers by adding their submodules
                for i, submodule in enumerate(layer):
                    if is_suitable_for_lsuv(submodule):
                        modules.append((f"{path}.{i}", submodule))
            elif is_suitable_for_lsuv(layer):
                modules.append((path, layer))
    return modules

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
    
    # Prepare context if needed
    property_context_sample_for_lsuv = None
    if is_stage2_run and args_obj.conditioning and hasattr(args_obj, 'property_norms_for_lsuv'):
        try:
            # Ensure the property tensor is 1D (global property)
            if 'transfection_score' in sample_batch_data:
                transfection_score = sample_batch_data['transfection_score']
                # Convert to tensor if not already
                if not isinstance(transfection_score, torch.Tensor):
                    transfection_score = torch.tensor(transfection_score, device=device, dtype=dtype)
                else:
                    transfection_score = transfection_score.to(device, dtype=dtype)
                
                # Ensure proper shape [batch_size]
                if transfection_score.dim() == 0:
                    transfection_score = transfection_score.view(1)
                elif transfection_score.dim() > 1:
                    transfection_score = transfection_score.squeeze(-1)
                
                # Update the sample batch data with properly formatted tensor
                sample_batch_data['transfection_score'] = transfection_score
                logging.info(f"Transfection score shape after formatting: {transfection_score.shape}")
                    
            property_context_sample_for_lsuv = prepare_context(args_obj.conditioning, sample_batch_data, args_obj.property_norms_for_lsuv)
            if property_context_sample_for_lsuv is not None:
                property_context_sample_for_lsuv = property_context_sample_for_lsuv.to(device, dtype)
                logging.info(f"Successfully prepared context for LSUV with shape: {property_context_sample_for_lsuv.shape}")
            else:
                logging.warning("Context preparation returned None, proceeding without context")
        except Exception as e:
            logging.error(f"Error preparing context for LSUV: {e}")
            logging.warning("Proceeding with LSUV without context")
            property_context_sample_for_lsuv = None

    x_mean_attr = 'stage2_dataset_x_mean' if is_stage2_run else 'dataset_x_mean'
    x_std_attr = 'stage2_dataset_x_std' if is_stage2_run else 'dataset_x_std'
    h_cat_mean_attr = 'stage2_dataset_h_cat_mean' if is_stage2_run else 'dataset_h_cat_mean'
    h_cat_std_attr = 'stage2_dataset_h_cat_std' if is_stage2_run else 'dataset_h_cat_std'

    # Initialize normalized tensors
    x_sample_normalized = x_sample_raw
    one_hot_sample_normalized = one_hot_sample_raw

    # Normalize positions with more robust handling
    if hasattr(args_obj, x_mean_attr) and getattr(args_obj, x_mean_attr) is not None:
        dataset_x_mean_dev = getattr(args_obj, x_mean_attr).to(device)
        dataset_x_std_dev = getattr(args_obj, x_std_attr).to(device)
        
        # Add small epsilon to prevent division by zero
        x_std_safe = dataset_x_std_dev.unsqueeze(0).unsqueeze(0) + 1e-6
        
        # Normalize positions
        x_sample_normalized = (x_sample_raw - dataset_x_mean_dev.unsqueeze(0).unsqueeze(0)) / x_std_safe
        
        # Apply masking before mean removal
        x_sample_normalized = x_sample_normalized * node_mask_sample
        
        # Remove mean with masking
        x_sample_for_model = diffusion_utils.remove_mean_with_mask(x_sample_normalized, node_mask_sample)
        
        # Check for NaN/Inf values
        if torch.isnan(x_sample_for_model).any() or torch.isinf(x_sample_for_model).any():
            logging.warning("NaN/Inf detected in normalized positions. Using raw positions instead.")
            x_sample_for_model = x_sample_raw * node_mask_sample
    else:
        logging.warning(f"LSUV: Dataset x mean/std ({x_mean_attr}) not found. Using raw positions.")
        x_sample_for_model = x_sample_raw * node_mask_sample

    # Normalize one-hot features
    if hasattr(args_obj, h_cat_mean_attr) and getattr(args_obj, h_cat_mean_attr) is not None:
        dataset_h_cat_mean_dev = getattr(args_obj, h_cat_mean_attr).to(device)
        dataset_h_cat_std_dev = getattr(args_obj, h_cat_std_attr).to(device)
        one_hot_sample_normalized = (one_hot_sample_raw - dataset_h_cat_mean_dev.unsqueeze(0).unsqueeze(0)) / (dataset_h_cat_std_dev.unsqueeze(0).unsqueeze(0) + 1e-6)
    else:
        logging.warning(f"LSUV: Dataset h_cat mean/std ({h_cat_mean_attr}) not found. Using raw one_hot.")
        one_hot_sample_normalized = one_hot_sample_raw

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
            if not is_suitable_for_lsuv(module):
                logging.warning(f"    LSUV: Skipping {path}, not suitable or no weight.")
                continue

            handle = module.register_forward_hook(hook_fn_lsuv)
            with torch.no_grad():
                try:
                    _ = model(x_sample_for_model, h_sample_for_model, node_mask_sample, edge_mask_sample, context=property_context_sample_for_lsuv)
                except Exception as e:
                    logging.error(f"Error during forward pass for {path}: {e}")
                    handle.remove()
                    continue
            handle.remove()
            
            if captured_output_lsuv is None:
                logging.warning(f"    LSUV: No output for {path}.")
                continue
            
            current_activations = captured_output_lsuv.detach()
            captured_output_lsuv = None
            
            # Apply masking to activations if they have the same shape as node_mask
            if current_activations.shape[:2] == node_mask_sample.shape[:2]:
                current_activations = current_activations * node_mask_sample
            
            finite_activations = current_activations[torch.isfinite(current_activations)]
            if finite_activations.numel() == 0:
                logging.warning(f"    LSUV: No finite output for {path}.")
                continue
                
            act_mean = finite_activations.mean()
            act_std = finite_activations.std()
            logging.info(f"    LSUV Layer {path}: current mean={act_mean.item():.4f}, std={act_std.item():.4f}")

            with torch.no_grad():
                # More conservative scaling approach
                if hasattr(module, 'bias') and module.bias is not None:
                    # Scale bias adjustment by a factor to prevent large jumps
                    bias_adjustment = (act_mean - target_mean) * 0.5
                    module.bias.data -= bias_adjustment
                
                if act_std > 1e-5:
                    # Scale weight adjustment by a factor to prevent large jumps
                    scale_factor = (target_std / act_std) * 0.5
                    if hasattr(module, 'weight'):
                        module.weight.data *= scale_factor
                    elif isinstance(module, nn.Sequential):
                        # For Sequential layers, scale each submodule's weights
                        for submodule in module:
                            if hasattr(submodule, 'weight'):
                                submodule.weight.data *= scale_factor
                elif target_std > 1e-5:
                    logging.warning(f"    LSUV {path}: current std ({act_std.item()}) too small, target_std={target_std}. Scaling risky.")
            
            # Recompute statistics after adjustment
            handle = module.register_forward_hook(hook_fn_lsuv)
            with torch.no_grad():
                try:
                    _ = model(x_sample_for_model, h_sample_for_model, node_mask_sample, edge_mask_sample, context=property_context_sample_for_lsuv)
                except Exception as e:
                    logging.error(f"Error during forward pass for {path} after adjustment: {e}")
                    handle.remove()
                    continue
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
parser = argparse.ArgumentParser(description='GeoLDM Lipid Fine-tuning (Stage 2)') # Changed description
parser.add_argument('--exp_name', type=str, default='geoldm_lipid_finetune_stage2') # More specific default
parser.add_argument('--output_dir', type=str, default='outputs/stage2_finetune', help='Directory to save outputs and checkpoints') # More specific default

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
parser.add_argument('--n_epochs', type=int, default=200, help='Number of fine-tuning epochs for Stage 2')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for Stage 2 (smaller for better diversity)')
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

# --- Stage arguments (from parse_args function, critical for StageManager) ---
parser.add_argument("--stage0_epochs", type=int, default=20,
                    help="Percentage of epochs for stage 0 (basic validity)")
parser.add_argument("--stage1_epochs", type=int, default=20,
                    help="Percentage of epochs for stage 1 (basic structure)")
parser.add_argument("--stage2_epochs", type=int, default=20,
                    help="Percentage of epochs for stage 2 (charge patterns)")
parser.add_argument("--stage3_epochs", type=int, default=40,
                    help="Percentage of epochs for stage 3 (refinement)")

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

# --- Load Stage 1 Args (Pretrained Args) ---
# This part remains largely the same, ensuring critical architectural params are loaded.
# ... (Existing pretrained_args loading logic) ...
# CRITICAL: Ensure args.normalize_factors is set to [1.0, 1.0, 1.0] for Stage 2 if we do external normalization
# This might be loaded from Stage 1 args, or we might need to override it.
# Let's assume Stage 1 saved it as [1,1,1] if we used external norm there.
# If not, we must set it here.
# For safety, explicitly set it if external normalization is done for Stage 2 data.
# args.normalize_factors = [1.0, 1.0, 1.0] # To be set AFTER loading Stage 1 args if needed

# --- Get Dataset Info (Consistent with Stage 1) ---
# ... (Existing dataset_info loading, ensure remove_h is consistent) ...
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
# These will be used for LSUV and in the training/validation loops
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
                'std': float(lipid_stats['std'])  # Changed 'mad' to 'std'
            }
        
        logging.info(f"Loaded property norms: {property_norms}")
except Exception as e:
    logging.error(f"Failed to load property norms from {args.lipid_stats_path}: {e}")
    logging.error(f"Full lipid stats content: {lipid_stats if 'lipid_stats' in locals() else 'Not loaded'}")
    sys.exit(1)

# Store for LSUV if needed:
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
# ... (existing model weight loading, strict=False) ...
# Collect mismatched paths again, though for Stage 2, we control unfreezing more directly.
# The LSUV below will target the layers we explicitly unfreeze.
mismatched_during_stage1_load = set() # Placeholder, actual mismatches happened in Stage 1.
                                   # For Stage 2, we care about what WE decide to fine-tune.

# --- Freeze/Unfreeze Layers for Stage 2 ---
logging.info("Configuring model layers for Stage 2 fine-tuning...")
# Default: VAE frozen (args.trainable_ae=False from Stage 1), Dynamics potentially trainable
if not args.trainable_ae and hasattr(model, 'vae'):
    logging.info("  Freezing VAE parameters as per args.trainable_ae=False.")
    for param in model.vae.parameters():
        param.requires_grad = False

trainable_dynamics_paths = []
if hasattr(model, 'dynamics'):
    logging.info("  Freezing all model.dynamics parameters initially for Stage 2.")
    for param in model.dynamics.parameters():
        param.requires_grad = False

    # Unfreeze layers trained in Stage 1 + one more GCL
    paths_to_unfreeze = [
        "dynamics.egnn.embedding",
        "dynamics.egnn.embedding_out"
    ]

    # Try to find and unfreeze GCL layers
    for i in range(args.n_layers):
        path_str = f"dynamics.egnn.e_block_{i}"
        layer = get_layer_by_path(model, path_str)
        if layer and isinstance(layer, torch.nn.Module):
            logging.info(f"  Unfreezing {path_str} for Stage 2 fine-tuning.")
            for param in layer.parameters():
                param.requires_grad = True
            trainable_dynamics_paths.append(path_str)
        else:
            logging.warning(f"  Could not find/unfreeze {path_str} for Stage 2.")

    # Unfreeze embedding layers
    for path_str in paths_to_unfreeze:
        layer = get_layer_by_path(model, path_str)
        if layer and isinstance(layer, torch.nn.Module):
            logging.info(f"  Unfreezing {path_str} for Stage 2 fine-tuning.")
            for param in layer.parameters():
                param.requires_grad = True
            trainable_dynamics_paths.append(path_str)
        else:
            logging.warning(f"  Could not find/unfreeze {path_str} for Stage 2.")

else:
    logging.warning("model.dynamics not found. Cannot apply Stage 2 freezing strategy.")

# --- LSUV for Unfrozen/Trainable Dynamics Layers ---
if trainable_dynamics_paths:
    logging.info("Preparing for LSUV on selected trainable dynamics modules for Stage 2...")
    if 'train' not in dataloaders or not dataloaders['train']:
        logging.error("S2 LSUV: Training dataloader not available. Skipping LSUV.")
    else:
        try:
            sample_batch_for_lsuv_s2 = next(iter(dataloaders['train']))
            lsuv_initialize_modules(
                model, trainable_dynamics_paths, sample_batch_for_lsuv_s2, args, device, dtype,
                target_std=0.05, target_mean=0.0, iterations=10, tolerance=0.01, is_stage2_run=True
            )
        except Exception as e_lsuv_s2:
            logging.error(f"Error during Stage 2 LSUV: {e_lsuv_s2}", exc_info=True)
            logging.error("Continuing Stage 2 without LSUV adjustments due to error.")
else:
    logging.info("No specific dynamics modules targeted for LSUV in Stage 2. Skipping.")

# --- Optimizer and Trainable Param Count ---
optim = get_optim(args, model, weight_decay=args.weight_decay)
logging.info(f"Optimizer created with LR: {args.lr} for Stage 2.")
trainable_params_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters for Stage 2: {trainable_params_s2}")

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
gradnorm_queue.add(1000) # Initialize with a relatively high value

# --- Training & Validation --- 
logging.info("Starting Stage 2 Fine-tuning...")
best_val_nll = float('inf')
best_epoch = 0

# Setup learning rate scheduler
if args.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    logging.info(f"Initialized cosine annealing scheduler with T_max={args.n_epochs}")
elif args.lr_scheduler == 'reduce_on_plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True
    )
    logging.info(f"Initialized reduce on plateau scheduler with factor={args.lr_factor}, patience={args.lr_patience}")
else:
    scheduler = None
    logging.info("No learning rate scheduler will be used")

# --- Diversity Functions ---
def is_valid_molecule(mol):
    """Check if a molecule is chemically and biologically valid."""
    try:
        # Basic chemical validity checks
        if len(Chem.GetMolFrags(mol)) > 1:
            return False
            
        # Check for atoms with too many bonds
        for atom in mol.GetAtoms():
            if atom.GetDegree() > 4:  # Maximum 4 bonds per atom
                return False
                
        # Check for rings that are too small
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) < 3:  # No 3-membered rings
                return False
                
        # Check for strained bonds
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                if bond.GetBeginAtom().GetDegree() > 4 or bond.GetEndAtom().GetDegree() > 4:
                    return False
        
        # Check for valid chirality
        if not is_valid_chirality(mol):
            return False
            
        # Check for valid tautomers
        if not is_valid_tautomer(mol):
            return False
            
        # Check for reasonable conformational energy
        if not has_reasonable_conformation(mol):
            return False
            
        return True
        
    except Exception as e:
        logging.warning(f"Error in is_valid_molecule: {e}")
        return False

def is_valid_chirality(mol):
    """Check if the molecule has valid chirality centers."""
    try:
        # Assign stereochemistry
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        
        # Get chiral centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        
        for atom_idx, chirality in chiral_centers:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Skip if not a chiral center
            if chirality == '?':
                continue
                
            # Check if the chiral center is reasonable
            if not is_reasonable_chiral_center(atom):
                return False
                
        return True
        
    except Exception as e:
        logging.warning(f"Error in is_valid_chirality: {e}")
        return False

def is_reasonable_chiral_center(atom):
    """Check if a chiral center is reasonable based on its environment."""
    try:
        # Get neighboring atoms
        neighbors = atom.GetNeighbors()
        
        # Must have exactly 4 different substituents
        if len(neighbors) != 4:
            return False
            
        # Check for common chiral center patterns
        substituents = set()
        for neighbor in neighbors:
            # Get the atom type and number of hydrogens
            atom_type = neighbor.GetAtomicNum()
            num_h = neighbor.GetTotalNumHs()
            substituents.add((atom_type, num_h))
            
        # Must have 4 different substituents
        if len(substituents) != 4:
            return False
            
        return True
        
    except Exception as e:
        logging.warning(f"Error in is_reasonable_chiral_center: {e}")
        return False

def is_valid_tautomer(mol):
    """Check if the molecule has valid tautomeric forms."""
    try:
        # Simplified tautomer check without MolStandardize
        # Check for common tautomeric patterns using SMARTS
        patterns = [
            # Keto-enol tautomerism
            "[C:1](=[O:2])[C:3][H:4]",
            # Imine-enamine tautomerism
            "[C:1](=[N:2])[C:3][H:4]",
            # Amide tautomerism
            "[C:1](=[O:2])[N:3][H:4]"
        ]
        
        for pattern in patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
                
        return False
        
    except Exception as e:
        logging.warning(f"Error in is_valid_tautomer: {e}")
        return False

def has_reasonable_conformation(mol):
    """Check if the molecule has a reasonable 3D conformation."""
    try:
        # Generate 3D coordinates if not present
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useSmallRingTorsions = True
            params.useMacrocycleTorsions = True
            params.useBasicKnowledge = True
            AllChem.EmbedMolecule(mol, params=params)
        
        # Optimize with MMFF94s
        try:
            energy = AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s')
            if energy is None or energy > 100:  # Energy threshold in kcal/mol
                return False
        except:
            # Fall back to UFF if MMFF fails
            try:
                energy = AllChem.UFFOptimizeMolecule(mol)
                if energy is None or energy > 100:
                    return False
            except:
                return False
        
        # Check for reasonable bond lengths and angles
        if not has_reasonable_geometry(mol):
            return False
            
        return True
        
    except Exception as e:
        logging.warning(f"Error in has_reasonable_conformation: {e}")
        return False

def has_reasonable_geometry(mol):
    """Check if the molecule has reasonable bond lengths and angles."""
    try:
        conf = mol.GetConformer()
        
        # Check bond lengths
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            begin_pos = conf.GetAtomPosition(begin_idx)
            end_pos = conf.GetAtomPosition(end_idx)
            
            bond_length = np.linalg.norm(np.array(begin_pos) - np.array(end_pos))
            
            # Use bond_analyze.get_bond_order for more accurate bond validation
            atom1 = mol.GetAtomWithIdx(begin_idx).GetSymbol()
            atom2 = mol.GetAtomWithIdx(end_idx).GetSymbol()
            expected_order = get_bond_order(atom1, atom2, bond_length)
            
            if expected_order == 0 and bond.GetBondType() != Chem.BondType.ZERO:
                return False
        
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
                            return False
        
        return True
        
    except Exception as e:
        logging.warning(f"Error in has_reasonable_geometry: {e}")
        return False

def is_reasonable_angle(angle):
    """Check if a bond angle is reasonable."""
    # Convert to degrees
    angle_deg = np.degrees(angle)
    
    # Typical bond angles in degrees
    min_angle = 60.0  # Minimum reasonable angle
    max_angle = 180.0  # Maximum reasonable angle
    
    return min_angle <= angle_deg <= max_angle

def calculate_angle(v1, v2, v3):
    """Calculate the angle between three points."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    
    v1v2 = v1 - v2
    v3v2 = v3 - v2
    
    cos_angle = np.dot(v1v2, v3v2) / (np.linalg.norm(v1v2) * np.linalg.norm(v3v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure within valid range
    
    return np.arccos(cos_angle)

def build_molecule(positions, atom_symbols, dataset_info):
    """Build an RDKit molecule from positions and atom types using a more biologically sound approach."""
    try:
        # Create molecule
        mol = Chem.RWMol()
        
        # Add atoms
        for atom_symbol in atom_symbols:
            # Get atomic number from the atom symbol
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(atom_symbol)
            if atomic_num == 0:  # Invalid atom type
                logging.warning(f"Invalid atom type {atom_symbol} found. Skipping molecule.")
                return None
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)
        
        # Create distance matrix
        n_atoms = len(positions)
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        
        # Get bond information from distance matrix using bond_analyze
        bonds = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                atom1 = atom_symbols[i]
                atom2 = atom_symbols[j]
                bond_order = get_bond_order(atom1, atom2, dist_matrix[i, j])
                if bond_order > 0:
                    bonds.append((i, j, bond_order))
        
        # Sort bonds by distance
        bonds.sort(key=lambda x: dist_matrix[x[0], x[1]])
        
        # Add bonds in order of increasing distance
        for i, j, order in bonds:
            # Skip if either atom already has too many bonds
            if (mol.GetAtomWithIdx(i).GetDegree() >= 4 or 
                mol.GetAtomWithIdx(j).GetDegree() >= 4):
                continue
                
            # Try to add bond
            mol.AddBond(i, j, Chem.BondType.values[order])
            
            # Check if molecule is still valid
            if not is_valid_molecule(mol):
                mol.RemoveBond(i, j)
        
        # Convert to regular molecule
        mol = mol.GetMol()
        
        # Sanitize the molecule before adding hydrogens
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logging.warning(f"Sanitization failed: {e}")
            return None
            
        # Add explicit hydrogens
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception as e:
            logging.warning(f"Failed to add hydrogens: {e}")
            return None
        
        # Generate 3D coordinates using ETKDG
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.useBasicKnowledge = True
        
        # Try to generate 3D coordinates
        conf_id = AllChem.EmbedMolecule(mol, params=params)
        if conf_id < 0:
            return None
            
        # Optimize the molecule with MMFF94s
        try:
            energy = AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s')
            if energy is None or energy > 100:  # Energy threshold in kcal/mol
                return None
        except:
            try:
                energy = AllChem.UFFOptimizeMolecule(mol)
                if energy is None or energy > 100:
                    return None
            except:
                return None
        
        # Check for reasonable geometry
        if not has_reasonable_geometry(mol):
            return None
            
        # Check for valid chirality
        if not is_valid_chirality(mol):
            return None
            
        # Check for valid tautomers
        if not is_valid_tautomer(mol):
            return None
            
        return mol
        
    except Exception as e:
        logging.warning(f"Error building molecule: {e}")
        return None

class LipidValidator:
    """Class to handle validation of ionizable lipids with progressive constraints."""
    
    def __init__(self, training_stage=0, max_stages=3):
        self.training_stage = training_stage
        self.max_stages = max_stages
        
    def is_valid(self, mol):
        """Validate if a molecule is a valid ionizable lipid with progressive constraints."""
        if mol is None:
            return False
            
        # Stage 0: Basic chemical validity only
        if self.training_stage == 0:
            return self._is_chemically_valid(mol)
            
        # Stage 1: Add basic lipid structure
        elif self.training_stage == 1:
            return (self._is_chemically_valid(mol) and 
                   self._has_basic_lipid_structure(mol))
            
        # Stage 2: Add ionizable group requirement
        elif self.training_stage == 2:
            return (self._is_chemically_valid(mol) and 
                   self._has_basic_lipid_structure(mol) and
                   self._has_ionizable_group(mol))
            
        # Final stage: All constraints
        else:
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
            return any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                      for pattern in self.BACKBONE_PATTERNS)
        except Exception as e:
            logging.warning(f"Error in basic structure check: {e}")
            return False
    
    def _has_ionizable_group(self, mol):
        """Check for ionizable groups with progressive strictness."""
        try:
            if self.training_stage == 2:
                # More lenient ionizable group check
                return any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                          for pattern in self.IONIZABLE_PATTERNS[:4])  # Start with basic patterns
            else:
                # Full ionizable group check
                return any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                          for pattern in self.IONIZABLE_PATTERNS)
        except Exception as e:
            logging.warning(f"Error in ionizable group check: {e}")
            return False
    
    def _meets_size_requirements(self, mol):
        """Check size requirements with progressive strictness."""
        try:
            mw = Descriptors.ExactMolWt(mol)
            num_rotatable = Descriptors.NumRotatableBonds(mol)
            
            if self.training_stage == 2:
                # More lenient size requirements
                return (200 <= mw <= 1200) and (num_rotatable >= 3)
            else:
                # Final size requirements
                return (300 <= mw <= 1000) and (num_rotatable >= 5)
        except Exception as e:
            logging.warning(f"Error in size requirement check: {e}")
            return False
    
    def advance_stage(self):
        """Advance to the next training stage."""
        if self.training_stage < self.max_stages:
            self.training_stage += 1
            logging.info(f"Advanced to training stage {self.training_stage}")

class LipidSimilarityCalculator:
    """Class to handle lipid similarity calculations."""
    
    @staticmethod
    def compute_similarity(mol1, mol2):
        """Compute overall similarity between two lipids."""
        try:
            # 2D similarity (Morgan fingerprints)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
            tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            # 3D similarity using 3D descriptors
            shape_sim = LipidSimilarityCalculator.compute_shape_similarity(mol1, mol2)
            
            # Ionizable group similarity
            ion_sim = LipidSimilarityCalculator.compute_ionizable_group_similarity(mol1, mol2)
            
            # Combine similarities with weights
            return (0.4 * tanimoto_sim +  # 2D structure
                    0.3 * shape_sim +     # 3D shape
                    0.3 * ion_sim)        # Ionizable group similarity
                    
        except Exception as e:
            logging.warning(f"Error in compute_similarity: {e}")
            return 0.0
    
    @staticmethod
    def compute_shape_similarity(mol1, mol2):
        """Compute 3D shape similarity using standard RDKit descriptors."""
        try:
            # Generate 3D coordinates if not present
            for mol in [mol1, mol2]:
                if mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
            
            # Calculate 3D descriptors
            desc1 = {
                'asphericity': Descriptors.Asphericity(mol1),
                'eccentricity': Descriptors.Eccentricity(mol1),
                'inertial_shape_factor': Descriptors.InertialShapeFactor(mol1),
                'radius_of_gyration': Descriptors.RadiusOfGyration(mol1),
                'molecular_volume': Descriptors.MolVolume(mol1)
            }
            
            desc2 = {
                'asphericity': Descriptors.Asphericity(mol2),
                'eccentricity': Descriptors.Eccentricity(mol2),
                'inertial_shape_factor': Descriptors.InertialShapeFactor(mol2),
                'radius_of_gyration': Descriptors.RadiusOfGyration(mol2),
                'molecular_volume': Descriptors.MolVolume(mol2)
            }
            
            # Normalize descriptors
            for key in desc1:
                max_val = max(abs(desc1[key]), abs(desc2[key]))
                if max_val > 0:
                    desc1[key] /= max_val
                    desc2[key] /= max_val
            
            # Compute Euclidean distance between normalized descriptors
            dist = sum((desc1[key] - desc2[key])**2 for key in desc1)**0.5
            
            # Convert distance to similarity (1 / (1 + distance))
            return 1.0 / (1.0 + dist)
            
        except Exception as e:
            logging.warning(f"Error in compute_shape_similarity: {e}")
            return 0.0
    
    @staticmethod
    def compute_ionizable_group_similarity(mol1, mol2):
        """Compute similarity based on ionizable group patterns."""
        try:
            # Create feature vectors
            vec1 = np.zeros(len(LipidValidator.IONIZABLE_PATTERNS))
            vec2 = np.zeros(len(LipidValidator.IONIZABLE_PATTERNS))
            
            for i, pattern in enumerate(LipidValidator.IONIZABLE_PATTERNS):
                vec1[i] = len(mol1.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
                vec2[i] = len(mol2.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
            
            # Compute Tanimoto similarity
            intersection = np.sum(np.minimum(vec1, vec2))
            union = np.sum(np.maximum(vec1, vec2))
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"Error in compute_ionizable_group_similarity: {e}")
            return 0.0

def compute_diversity_penalty(model, batch_size, n_nodes, edge_mask, device, dataset_info):
    """Compute diversity penalty for generated ionizable lipids."""
    try:
        # Prepare input tensors
        node_mask = torch.ones(batch_size, n_nodes, 1, device=device)
        if edge_mask is None:
            edge_mask = torch.ones(batch_size, n_nodes, n_nodes, device=device)
        elif edge_mask.dim() == 2:
            edge_mask = edge_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Sample and decode molecules
        z = model.sample_combined_position_feature_noise(
            n_samples=batch_size,
            n_nodes=n_nodes,
            node_mask=node_mask
        )
        
        if z.dim() == 2 and z.shape[0] == batch_size * n_nodes:
            z = z.view(batch_size, n_nodes, -1)
        
        if z.shape[-1] != model.vae.encoder.out_node_nf:
            z = model.vae.encoder.embedding_out(z)
        
        h, x = model.vae.decode(z, node_mask=node_mask, edge_mask=edge_mask)
        
        if x.dim() == 2:
            if x.shape[0] == batch_size * n_nodes:
                x = x.view(batch_size, n_nodes, 3)
            else:
                inferred_nodes = x.shape[0] // batch_size
                if inferred_nodes > 0:
                    x = x.view(batch_size, inferred_nodes, 3)
                else:
                    return torch.tensor(0.0, device=device)
        
        # Group valid lipids by properties
        molecules_by_group = {}
        for i in range(batch_size):
            try:
                atom_types = [dataset_info['atom_decoder'][j] for j in h['categorical'][i].argmax(dim=-1).cpu().numpy()]
                mol = build_molecule(x[i].cpu().numpy(), atom_types, dataset_info)
                
                if mol is not None and LipidValidator.is_valid(mol):
                    # Calculate lipid properties for grouping
                    mw = Descriptors.ExactMolWt(mol)
                    charge = Chem.GetFormalCharge(mol)
                    num_ionizable = LipidValidator.count_ionizable_groups(mol)
                    num_rotatable = Descriptors.NumRotatableBonds(mol)
                    
                    group_key = (
                        round(mw / 100) * 100,
                        charge,
                        num_ionizable,
                        round(num_rotatable / 2) * 2
                    )
                    
                    if group_key not in molecules_by_group:
                        molecules_by_group[group_key] = []
                    molecules_by_group[group_key].append(mol)
                    
            except Exception as e:
                logging.warning(f"Error processing molecule {i}: {e}")
                continue
        
        # Compute similarities within each group
        similarities = []
        for molecules in molecules_by_group.values():
            if len(molecules) < 2:
                continue
            
            for i in range(len(molecules)):
                for j in range(i + 1, len(molecules)):
                    try:
                        sim = LipidSimilarityCalculator.compute_similarity(molecules[i], molecules[j])
                        similarities.append(sim)
                    except Exception as e:
                        logging.warning(f"Error computing similarity: {e}")
                        continue
        
        # Return average similarity as penalty
        avg_sim = np.mean(similarities) if similarities else 0.0
        return torch.tensor(avg_sim, device=device)
        
    except Exception as e:
        logging.warning(f"Error in compute_diversity_penalty: {e}")
        return torch.tensor(0.0, device=device)

class LipidDataAugmenter:
    """Class to handle data augmentation for lipid molecules."""
    
    @staticmethod
    def augment(data_dict, dataset_info):
        """Apply data augmentation to molecule data."""
        try:
            # Create a copy of the data to modify
            augmented_data = copy.deepcopy(data_dict)
            
            # Get the positions and convert to numpy for easier manipulation
            positions = augmented_data["positions"].cpu().numpy()  # Move to CPU first
            
            # Apply random rotation
            if random.random() < 0.5:
                positions = LipidDataAugmenter._apply_rotation(positions)
            
            # Apply small random perturbations
            positions = LipidDataAugmenter._apply_perturbation(positions)
            
            # Convert back to tensor and move to the same device as input
            augmented_data["positions"] = torch.from_numpy(positions).to(data_dict["positions"].device)
            
            return augmented_data
            
        except Exception as e:
            logging.warning(f"Error in augment: {e}")
            return data_dict
    
    @staticmethod
    def _apply_rotation(positions):
        """Apply random rotation to positions."""
        theta = random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return np.dot(positions, rotation_matrix)
    
    @staticmethod
    def _apply_perturbation(positions, noise_scale=0.1):
        """Apply small random perturbations to positions."""
        noise = np.random.normal(0, noise_scale, positions.shape)
        return positions + noise

class LipidDiversityCalculator:
    """Class to handle diversity calculations for lipid molecules."""
    
    @staticmethod
    def compute_internal_diversity(molecules):
        """Compute internal diversity of a set of molecules."""
        try:
            if not molecules or len(molecules) < 2:
                return 0.0
            
            # Compute similarity matrix
            similarity_matrix = np.zeros((len(molecules), len(molecules)))
            
            for i, mol1 in enumerate(molecules):
                if mol1 is None:
                    continue
                for j, mol2 in enumerate(molecules):
                    if i != j and mol2 is not None:
                        try:
                            similarity_matrix[i, j] = LipidSimilarityCalculator.compute_similarity(mol1, mol2)
                        except Exception as e:
                            logging.warning(f"Error computing similarity: {e}")
                            continue
            
            # Compute average similarity (excluding self-similarities)
            mask = ~np.eye(len(molecules), dtype=bool)
            valid_similarities = similarity_matrix[mask]
            if len(valid_similarities) == 0:
                return 0.0
            
            avg_similarity = np.mean(valid_similarities)
            return 1.0 - avg_similarity  # Return diversity (1 - similarity)
            
        except Exception as e:
            logging.warning(f"Error in compute_internal_diversity: {e}")
            return 0.0

def train_epoch(model, optimizer, dataloader, device, epoch, args):
    model.train()
    total_loss = 0
    total_samples = 0
    
    # Set memory management parameters for CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # Move data to device and ensure float32
            batch = {k: v.to(device).float() for k, v in batch.items()}
            
            # Prepare 'node_mask' for the model and other uses, from 'atom_mask'
            if 'atom_mask' in batch: # Correctly check for 'atom_mask' from dataloader
                processed_mask = batch['atom_mask'].float() # Use 'atom_mask' as source
                if processed_mask.dim() == 2:
                    processed_mask = processed_mask.unsqueeze(-1)
                batch['node_mask'] = (processed_mask > 0.5).float() # Store processed mask as 'node_mask'
            # else: if 'atom_mask' is not in batch, 'node_mask' will not be added to batch here
                
            if 'edge_mask' in batch:
                edge_mask = batch['edge_mask'].float()
                if edge_mask.dim() == 2:
                    edge_mask = edge_mask.unsqueeze(0)
                edge_mask = (edge_mask > 0.5).float()
                batch['edge_mask'] = edge_mask
            
            # Simple mean centering using the original function
            if 'positions' in batch:
                x = batch['positions']
                # Use 'node_mask' that was (potentially) prepared in the block above
                node_mask_for_centering = batch.get('node_mask', None) 
                
                if node_mask_for_centering is not None: # Check if 'node_mask' was successfully prepared
                    x_centered = diffusion_utils.remove_mean_with_mask(x, node_mask_for_centering)
                    batch['positions'] = x_centered
                elif x is not None: # 'atom_mask' (and thus 'node_mask') was not in the batch
                    logging.warning("Mean centering skipped: 'atom_mask' (and thus 'node_mask') not found in batch. This could indicate a data loading issue.")
            
            # Forward pass
            optimizer.zero_grad()

            # Prepare arguments for model.forward()
            x_in = batch.get('positions')
            h_cat_in = batch.get('one_hot')
            h_charges_in = batch.get('charges') # This should be (batch_size, n_nodes, 1) or similar
            node_mask_in = batch.get('node_mask') # Already processed
            edge_mask_in = batch.get('edge_mask') # Already processed

            if x_in is None or h_cat_in is None or h_charges_in is None:
                logging.error(f"Missing critical data for model input in batch {batch_idx}. Skipping batch. x: {x_in is not None}, h_cat: {h_cat_in is not None}, h_charges: {h_charges_in is not None}")
                continue

            # Construct the 'h' dictionary for the model
            # Ensure h_charges_in has the correct dimensions if it's not empty
            # The model's internal processing of 'h' (e.g. in EnVariationalDiffusion.normalize)
            # expects h['integer'] to be meaningful if args.include_charges is True.
            # If charges are not used by model config, this might still be passed but ignored or should be None.
            # Based on GeoLDM/core/models.py, in_node_nf includes charges if args.include_charges.
            # The 'charges' feature in preprocess_lipids.py is (num_atoms, 0) if not used,
            # or (num_atoms, 1) if used (e.g. formal charges).
            # For EnLatentDiffusion.forward, h['integer'] is concatenated with h['categorical'] internally after VAE.
            # So, it should be shaped (batch_size, n_nodes, num_charge_features).
            # If charges are a single value per node, it should be (batch_size, n_nodes, 1).
            # If `args.include_charges` is false for the model, h_charges_in might be (B,N,0)
            # and might need to be handled or ensured it's what the model expects.
            # Let's assume batch['charges'] is correctly prepared by the dataloader to be (B,N,C_int)
            h_in = {'categorical': h_cat_in, 'integer': h_charges_in}

            # Prepare context if conditioning
            context_in = None
            if args.conditioning:
                # Ensure property_norms is accessible; it's loaded in main() of finetune_lipids.py
                # and should be passed or made accessible to train_epoch if not part of args
                # For now, assume args contains the necessary norms or they are globally accessible in a way
                # that prepare_context can use them.
                # We know `property_norms` is calculated in `main` and then `args.property_norms_for_lsuv` is set.
                # We should use a consistent name or pass it correctly.
                # Let's assume property_norms are in args.property_norms_for_lsuv for consistency with LSUV part.
                current_property_norms = getattr(args, 'property_norms_for_lsuv', None)
                if current_property_norms:
                    context_in = prepare_context(args.conditioning, batch, current_property_norms) # batch already on device
                else:
                    logging.warning(f"Conditioning enabled, but property_norms not found in args.property_norms_for_lsuv. Proceeding without context for batch {batch_idx}.")
            
            output = model(x=x_in, h=h_in, node_mask=node_mask_in, edge_mask=edge_mask_in, context=context_in)
            
            # Calculate loss
            loss = output # Changed from: output['loss']
            if loss.ndim > 0: # Ensure loss is scalar
                loss = loss.mean()
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_samples += batch['positions'].size(0)
            
            # Log batch metrics
            if batch_idx % args.log_interval == 0:
                logging.info(f'Train Epoch: {epoch} [{batch_idx}/{len(dataloader)} '
                           f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                           f'Loss: {loss.item():.6f}')
                
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            traceback.print_exc() # Add this line to print the full traceback
            continue
    
    # Calculate epoch metrics
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        logging.info(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
        return avg_loss
    else:
        logging.warning("No valid samples processed in epoch")
        return float('inf')

def validate(model, dataloader, device, dataset_info, args):
    """Validate the model."""
    model.eval()
    val_loss = 0
    val_steps = 0
    valid_molecules = []
    batch_metrics = []  # Store metrics for each batch
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            try:
                # Move data to device
                data = {k: v.to(device) for k, v in data.items()}
                
                # Prepare data for model
                x = data["positions"]
                h = {
                    "categorical": data["one_hot"],
                    "integer": data["charges"] if "charges" in data else None
                }
                node_mask = data["atom_mask"]
                edge_mask = data["edge_mask"] if "edge_mask" in data else None
                
                # Forward pass
                loss = model(x, h, node_mask, edge_mask)
                
                # Update statistics
                val_loss += loss.item()
                val_steps += 1
                
                # Store batch metrics
                batch_metrics.append({
                    'batch_idx': batch_idx,
                    'loss': loss.item()
                })
                
                # Generate molecules for diversity calculation
                if batch_idx % args.validation_interval == 0:
                    for i in range(min(10, x.shape[0])):  # Process up to 10 molecules per batch
                        try:
                            atom_types = [dataset_info['atom_decoder'][j] for j in h['categorical'][i].argmax(dim=-1).cpu().numpy()]
                            mol = build_molecule(x[i].cpu().numpy(), atom_types, dataset_info)
                            if mol is not None and LipidValidator.is_valid(mol):
                                valid_molecules.append(mol)
                        except Exception as e:
                            logging.warning(f"Error building molecule {i} in batch {batch_idx}: {e}")
                            continue
                
            except Exception as e:
                logging.warning(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    # Calculate validation metrics
    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
    diversity = LipidDiversityCalculator.compute_internal_diversity(valid_molecules)
    
    # Calculate additional validation metrics
    val_metrics = {
        'val_loss': avg_val_loss,
        'num_valid_molecules': len(valid_molecules),
        'diversity': diversity,
        'min_loss': min([m['loss'] for m in batch_metrics]) if batch_metrics else 0.0,
        'max_loss': max([m['loss'] for m in batch_metrics]) if batch_metrics else 0.0,
        'std_loss': np.std([m['loss'] for m in batch_metrics]) if batch_metrics else 0.0,
        'num_batches': val_steps
    }
    
    # Log validation summary in a structured format
    logging.info(f"Validation Summary: {json.dumps(val_metrics, indent=2)}")
    
    return val_metrics

def initialize_model(dataset_info, device, pretrained_path, args):
    """Initialize the model with pre-trained weights."""
    try:
        # Create model using get_latent_diffusion
        model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, None)
        model = model.to(device)
        
        # Load pre-trained weights if provided
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logging.info(f"Loaded pre-trained weights from {pretrained_path}")
        
        return model
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        raise

class EarlyStoppingManager:
    """Manages early stopping with stage awareness."""
    
    def __init__(self, args, stage_manager):
        self.patience = args.early_stopping_patience
        self.min_delta = args.early_stopping_min_delta
        self.stage_manager = stage_manager
        self.best_val_loss = float('inf')
        self.counter = 0
        self.current_stage = 0
        self.stage_patience = {
            0: int(args.early_stopping_patience * 1.5),  # More patience in early stages
            1: int(args.early_stopping_patience * 1.2),
            2: args.early_stopping_patience,
            3: args.early_stopping_patience
        }
        logging.info(f"Early stopping initialized with stage-specific patience: {self.stage_patience}")
    
    def should_stop(self, current_epoch, val_loss, current_stage):
        """Check if training should stop, considering the current stage."""
        # Reset state if entering a new stage
        if current_stage != self.current_stage:
            logging.info(f"Resetting early stopping state for stage {current_stage}")
            self.best_val_loss = float('inf')
            self.counter = 0
            self.current_stage = current_stage
        
        # Get patience for current stage
        current_patience = self.stage_patience[current_stage]
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= current_patience:
                logging.info(f"Early stopping triggered at epoch {current_epoch} in stage {current_stage}")
                logging.info(f"Best validation loss for stage {current_stage}: {self.best_val_loss:.4f}")
                return True
        return False

class StageManager:
    """Manages the progression through different training stages."""
    
    def __init__(self, args):
        self.args = args
        self.total_epochs = args.n_epochs
        
        # Calculate stage boundaries based on percentages
        total_stage_percentage = (args.stage0_epochs + args.stage1_epochs + 
                                args.stage2_epochs + args.stage3_epochs)
        
        # Normalize stage percentages
        self.stage0_end = int((args.stage0_epochs / total_stage_percentage) * self.total_epochs)
        self.stage1_end = self.stage0_end + int((args.stage1_epochs / total_stage_percentage) * self.total_epochs)
        self.stage2_end = self.stage1_end + int((args.stage2_epochs / total_stage_percentage) * self.total_epochs)
        self.stage3_end = self.stage2_end + int((args.stage3_epochs / total_stage_percentage) * self.total_epochs)
        
        # Initialize validator
        self.validator = LipidValidator(training_stage=0)
        
        logging.info(f"Stage boundaries (epochs):")
        logging.info(f"  Stage 0: 0-{self.stage0_end}")
        logging.info(f"  Stage 1: {self.stage0_end}-{self.stage1_end}")
        logging.info(f"  Stage 2: {self.stage1_end}-{self.stage2_end}")
        logging.info(f"  Stage 3: {self.stage2_end}-{self.stage3_end}")
        logging.info(f"  Total epochs: {self.total_epochs}")
    
    def get_current_stage(self, epoch):
        """Determine the current training stage based on epoch number."""
        if epoch < self.stage0_end:
            stage = 0
        elif epoch < self.stage1_end:
            stage = 1
        elif epoch < self.stage2_end:
            stage = 2
        else:
            stage = 3
        
        # Update validator stage if needed
        if self.validator.training_stage != stage:
            self.validator.training_stage = stage
            logging.info(f"Advancing to training stage {stage} at epoch {epoch}")
            
            # Log stage-specific information
            if stage == 0:
                logging.info("Stage 0: Basic chemical validity only")
            elif stage == 1:
                logging.info("Stage 1: Adding basic lipid structure requirements")
            elif stage == 2:
                logging.info("Stage 2: Introducing ionizable group requirements")
            else:
                logging.info("Final Stage: Enforcing all ionizable lipid requirements")
        
        return stage

def load_dataset_info():
    """Load dataset information."""
    try:
        # Get dataset info from configs
        from GeoLDM.configs.datasets_config import get_dataset_info
        dataset_info = get_dataset_info("geom", remove_h=False)  # Always use with hydrogens for lipids
        return dataset_info
    except Exception as e:
        logging.error(f"Error loading dataset info: {e}")
        raise

def save_checkpoint(model, optimizer, epoch, stage, val_metrics, args, is_final=False):
    """Save model checkpoint with stage information."""
    checkpoint = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "args": args
    }
    
    # Save to stage-specific directory
    stage_dir = os.path.join(args.output_dir, f"stage_{stage}")
    os.makedirs(stage_dir, exist_ok=True)
    
    # Use different filename for final checkpoint
    if is_final:
        checkpoint_path = os.path.join(stage_dir, f"final_checkpoint_epoch_{epoch + 1}.pt")
    else:
        checkpoint_path = os.path.join(stage_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    """Main training function."""
    # args = parse_args() # Removed this line to use the global args
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create metrics file
    metrics_file = os.path.join(args.output_dir, "training_metrics.json")
    metrics_history = {
        'train_metrics': [],
        'val_metrics': [],
        'epochs': []
    }
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Load dataset info (centralized)
    dataset_info = load_dataset_info()
    logging.info(f"Loaded dataset info for: {args.dataset} (remove_h={args.remove_h})")
    
    # Load dataset and create dataloaders using lipid_dataset.get_dataloaders
    dataloaders = lipid_dataset.get_dataloaders(
        args.lipid_data_path,           # positional
        args.batch_size,                # positional
        num_workers=args.num_workers,   # keyword
        seed=args.seed,
        val_split_ratio=0.1,
        lipid_stats_path=args.lipid_stats_path,
        is_stage2_data=True
    )
    
    # Initialize model using get_latent_diffusion
    model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    logging.info(f"Optimizer created with LR: {args.lr}, weight_decay: {args.weight_decay}")
    
    # Initialize scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        logging.info(f"Initialized cosine annealing scheduler with T_max={args.n_epochs}")
    elif args.lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True
        )
        logging.info(f"Initialized reduce on plateau scheduler with factor={args.lr_factor}, patience={args.lr_patience}")
    else:
        scheduler = None
        logging.info("No learning rate scheduler will be used")
    
    # Initialize stage manager
    stage_manager = StageManager(args)
    
    # Initialize early stopping
    early_stopping = EarlyStoppingManager(args, stage_manager)
    
    # Training loop
    best_val_loss = float("inf")
    current_stage = 0
    
    for epoch in range(args.n_epochs):
        # Update stage if needed
        current_stage = stage_manager.get_current_stage(epoch)
        
        # Train for one epoch
        train_metrics = train_epoch(model, optimizer, dataloaders['train'], device, epoch, args)
        
        # Validate
        if (epoch + 1) % args.validation_interval == 0:
            val_metrics = validate(model, dataloaders['val'], device, dataset_info, args)
            
            # Update scheduler if using reduce on plateau
            if args.lr_scheduler == 'reduce_on_plateau' and scheduler is not None:
                scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch + 1}/{args.n_epochs} - "
                f"Train Loss: {train_metrics:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Diversity: {val_metrics['diversity']:.4f} - "
                f"Valid Molecules: {val_metrics['num_valid_molecules']} - "
                f"LR: {current_lr:.6f} - "
                f"Stage: {current_stage}"
            )
            
            # Save metrics to history
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['train_metrics'].append(train_metrics)
            metrics_history['val_metrics'].append(val_metrics)
            
            # Save metrics to file
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)
            
            # Check early stopping with stage awareness
            if early_stopping.should_stop(epoch, val_metrics['val_loss'], current_stage):
                logging.info(f"Early stopping triggered at epoch {epoch + 1} in stage {current_stage}")
                logging.info(f"Best validation loss: {best_val_loss:.4f}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    stage=current_stage,
                    val_metrics=val_metrics,
                    args=args,
                    is_final=True
                )
                break
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                logging.info(f"New best validation loss: {best_val_loss:.4f}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    stage=current_stage,
                    val_metrics=val_metrics,
                    args=args
                )
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    stage=current_stage,
                    val_metrics=val_metrics,
                    args=args
                )
        else:
            # Save training metrics even when not validating
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['train_metrics'].append(train_metrics)
            metrics_history['val_metrics'].append(None)  # No validation this epoch
            
            # Save metrics to file
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune GeoLDM for ionizable lipids")
    
    # Data arguments
    parser.add_argument('--lipid_data_path', type=str, required=True,
                      help='Path to processed_train_lipids.pkl for Stage 2')
    parser.add_argument('--lipid_stats_path', type=str, required=True,
                      help='Path to lipid_stats.pkl')
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers for DataLoader")
    parser.add_argument("--dataset", type=str, default="geom",
                      help="Base dataset type (used for dataset_info from Stage 1)")
    parser.add_argument("--remove_h", action="store_true",
                      help="Use dataset config without hydrogens (ensure model matches Stage 1 config)")
    
    # Training arguments
    parser.add_argument("--n_epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                      help="Weight decay for optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use for training")
    parser.add_argument("--cuda", action="store_true", default=torch.cuda.is_available(),
                      help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    # Training loop arguments
    parser.add_argument("--validation_interval", type=int, default=5,
                      help="Run validation every N epochs")
    parser.add_argument("--save_interval", type=int, default=10,
                      help="Save checkpoint every N epochs")
    parser.add_argument("--log_interval", type=int, default=10,  # Changed from 50 to 10
                      help="Log training progress every N batches")
    parser.add_argument("--augmentation_prob", type=float, default=0.2,
                      help="Probability of applying data augmentation")
    
    # Learning rate scheduler arguments
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                      choices=["cosine", "reduce_on_plateau", "none"],
                      help="Learning rate scheduler type")
    parser.add_argument("--lr_patience", type=int, default=5,
                      help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                      help="Factor to reduce learning rate by when using ReduceLROnPlateau")
    
    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                      help="Number of epochs to wait before early stopping if validation NLL does not improve")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4,
                      help="Minimum change in validation NLL to be considered as improvement for early stopping")
    
    # Model arguments
    parser.add_argument("--pretrained_path", type=str, required=True,
                      help="Path to pre-trained model weights")
    parser.add_argument("--model_name", type=str, default=None,
                      help="Name of the model file to load")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save model checkpoints and logs")
    parser.add_argument("--exp_name", type=str, default=None,
                      help="Name of the experiment")
    
    # Latent Diffusion args
    parser.add_argument("--train_diffusion", action="store_true", 
                      help="Train second stage LatentDiffusionModel model")
    parser.add_argument("--ae_path", type=str, default=None,
                      help="Specify first stage model path")
    parser.add_argument("--trainable_ae", action="store_true",
                      help="Train first stage AutoEncoder model")
    parser.add_argument("--latent_nf", type=int, default=4,
                      help="Number of latent features")
    parser.add_argument("--kl_weight", type=float, default=0.01,
                      help="Weight of KL term in ELBO")
    parser.add_argument("--model", type=str, default="egnn_dynamics",
                      help="Model type: egnn_dynamics | schnet | simple_dynamics | kernel_dynamics | gnn_dynamics")
    parser.add_argument("--probabilistic_model", type=str, default="diffusion",
                      help="Probabilistic model type")
    parser.add_argument("--diffusion_steps", type=int, default=500,
                      help="Number of diffusion steps")
    parser.add_argument("--diffusion_noise_schedule", type=str, default="polynomial_2",
                      help="Noise schedule for diffusion")
    parser.add_argument("--diffusion_noise_precision", type=float, default=1e-5,
                      help="Noise precision for diffusion")
    parser.add_argument("--diffusion_loss_type", type=str, default="l2",
                      help="Loss type for diffusion")
    parser.add_argument("--normalize_factors", type=float, nargs="+", default=[1.0, 1.0, 1.0],
                      help="Normalization factors")
    parser.add_argument("--include_charges", type=bool, default=True,
                      help="Include charges in the model")
    parser.add_argument("--context_node_nf", type=int, default=0,
                      help="Number of context node features")
    parser.add_argument("--nf", type=int, default=256,
                      help="Number of hidden features")
    parser.add_argument("--n_layers", type=int, default=4,
                      help="Number of layers")
    parser.add_argument("--attention", action="store_true",
                      help="Use attention in the model")
    parser.add_argument("--tanh", action="store_true",
                      help="Use tanh activation")
    parser.add_argument("--norm_constant", type=float, default=1.0,
                      help="Normalization constant")
    parser.add_argument("--inv_sublayers", type=int, default=1,
                      help="Number of invariant sublayers")
    parser.add_argument("--sin_embedding", action="store_true",
                      help="Use sinusoidal embedding")
    parser.add_argument("--normalization_factor", type=float, default=1.0,
                      help="Normalization factor")
    parser.add_argument("--aggregation_method", type=str, default="sum",
                      help="Aggregation method")
    
    # Stage arguments (as percentages of total epochs)
    parser.add_argument("--stage0_epochs", type=int, default=20,
                      help="Percentage of epochs for stage 0 (basic validity)")
    parser.add_argument("--stage1_epochs", type=int, default=20,
                      help="Percentage of epochs for stage 1 (basic structure)")
    parser.add_argument("--stage2_epochs", type=int, default=20,
                      help="Percentage of epochs for stage 2 (charge patterns)")
    parser.add_argument("--stage3_epochs", type=int, default=40,
                      help="Percentage of epochs for stage 3 (refinement)")
    
    # Conditioning arguments
    parser.add_argument("--conditioning", type=str, nargs="+", default=[],
                      help="List of properties to condition on")
    
    return parser.parse_args()

# Remove the duplicate load_dataset_info function at the end of the file
if __name__ == "__main__":
    main()

# Log the anti-overfitting settings
logging.info("Anti-overfitting settings:")
logging.info(f"  Learning rate scheduler: {args.lr_scheduler}")
logging.info(f"  Weight decay: {args.weight_decay}")
logging.info(f"  Early stopping patience: {args.early_stopping_patience}")
logging.info(f"  Early stopping min delta: {args.early_stopping_min_delta}")
if args.lr_scheduler == 'reduce_on_plateau':
    logging.info(f"  LR reduction patience: {args.lr_patience}")
    logging.info(f"  LR reduction factor: {args.lr_factor}")

logging.getLogger().setLevel(logging.DEBUG)
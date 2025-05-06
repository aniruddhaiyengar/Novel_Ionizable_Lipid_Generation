'''
Fine-tuning script for GeoLDM on lipid data with transfection_score conditioning.

Based on main_geom_drugs.py and train_test.py from the original GeoLDM codebase.
'''

from rdkit import Chem


# Local imports
from . import lipid_dataset
from .configs.datasets_config import get_dataset_info
from .core.utils import prepare_context

import copy
from . import utils
import argparse
import wandb
import os
from os.path import join
from .core.models import get_optim, get_latent_diffusion
from .equivariant_diffusion import en_diffusion
from .equivariant_diffusion import utils as diffusion_utils
from .core import losses as qm9_losses
import torch
import time
import pickle
from tqdm import tqdm
import numpy as np

# Define arguments specific to fine-tuning or conditioning
parser = argparse.ArgumentParser(description='GeoLDM Lipid Fine-tuning')
parser.add_argument('--exp_name', type=str, default='geoldm_lipid_finetune')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs and checkpoints')

# --- Data Arguments --- 
parser.add_argument('--lipid_data_path', type=str, required=True, help='Path to processed_lipid_data.pkl')
parser.add_argument('--lipid_stats_path', type=str, required=True, help='Path to lipid_stats.pkl')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
parser.add_argument('--dataset', type=str, default='geom', help='Base dataset type (used for dataset_info)') # Added dataset arg
parser.add_argument('--remove_h', action='store_true', help='Use dataset config without hydrogens (ensure model matches)') # Added remove_h arg

# --- Model Loading Arguments --- 
parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the folder containing pretrained GEOM model (.npy, .pickle files)')
parser.add_argument('--model_name', type=str, default='generative_model_ema.npy', help='Name of the model state_dict file to load (e.g., generative_model_ema.npy or generative_model.npy)')

# --- Fine-tuning Arguments --- 
parser.add_argument('--n_epochs', type=int, default=100, help='Number of fine-tuning epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (adjust based on GPU memory)')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for fine-tuning')
parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate, 0 to disable')
parser.add_argument('--test_epochs', type=int, default=5, help='Run validation every N epochs')
parser.add_argument('--save_model', type=eval, default=True, help='Save checkpoints')
parser.add_argument('--clip_grad', type=eval, default=True, help='Clip gradients during training')
parser.add_argument('--clip_value', type=float, default=1.0, help='Value for gradient clipping')
parser.add_argument('--n_report_steps', type=int, default=50, help='Log training progress every N steps') # From train_test
parser.add_argument('--ode_regularization', type=float, default=1e-3, help='ODE regularization weight') # From train_test

# --- Conditioning Arguments --- 
parser.add_argument('--conditioning', nargs='+', default=['transfection_score'], help='Conditioning property key(s)')
parser.add_argument('--cfg_prob', type=float, default=0.1, help='Probability of using unconditional context during training for CFG (0.0 to disable)')

# --- Diffusion Arguments (load from pickle) --- 
# Defaults are provided but should be overridden by loaded args.pickle
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2')
parser.add_argument('--diffusion_loss_type', type=str, default='l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5) # From README example

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
parser.add_argument('--include_charges', type=eval, default=True) # Default True based on geom_with_h
parser.add_argument('--normalization_factor', type=float, default=1.0)
parser.add_argument('--aggregation_method', type=str, default='sum')
parser.add_argument('--kl_weight', type=float, default=0.01)
parser.add_argument('--train_diffusion', action='store_true')
parser.add_argument('--trainable_ae', action='store_true')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10], help='normalize factors for [x, categorical, integer]') # Added from README

# --- Other Arguments --- 
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
parser.add_argument('--wandb_usr', type=str, default=None, help='WandB username')
parser.add_argument('--online', type=bool, default=True, help='WandB online mode')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

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

# --- Load Pretrained Args --- 
pretrained_args_path = join(args.pretrained_path, 'args.pickle')
try:
    with open(pretrained_args_path, 'rb') as f:
        pretrained_args = pickle.load(f)
    print(f"Loaded arguments from pretrained model: {pretrained_args_path}")
    arg_dict = vars(args)
    pretrained_arg_dict = vars(pretrained_args)
    keys_to_load = [
        'model', 'n_layers', 'nf', 'latent_nf', 'tanh', 'attention',
        'norm_constant', 'inv_sublayers', 'sin_embedding', 'include_charges',
        'normalization_factor', 'aggregation_method',
        'diffusion_steps', 'diffusion_noise_schedule', 'diffusion_loss_type',
        'diffusion_noise_precision', 'normalize_factors', 
        'kl_weight', 'train_diffusion', 'trainable_ae'
    ]
    for key in keys_to_load:
        if key in pretrained_arg_dict:
            arg_dict[key] = pretrained_arg_dict[key]
        else:
             print(f"  Warning: Key '{key}' not found in pretrained args.pickle.")

except FileNotFoundError:
    print(f"Warning: Pretrained args.pickle not found at {pretrained_args_path}.")
except Exception as e:
    print(f"Error loading pretrained args.pickle: {e}.")

# --- Get Dataset Info --- 
# Use the function from datasets_config.py
# Ensure --remove_h matches the intended model state (geom_with_h is default)
args.remove_h = False # Explicitly set based on using geom_with_h
if args.dataset != 'geom':
     print(f"Warning: Dataset argument is {args.dataset}, but fine-tuning GEOM model. Using GEOM dataset info.")
     args.dataset = 'geom'
dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
print(f"Using dataset info for: {args.dataset} (remove_h={args.remove_h})")

# --- Wandb Setup --- 
# (Same as before)
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
    if not args.wandb_usr:
        args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
if not args.no_wandb:
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion_lipid_finetune', 'config': args,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

print("Effective Arguments:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

# --- Load Data --- 
dataloaders = lipid_dataset.get_dataloaders(
    args.lipid_data_path,
    args.lipid_stats_path,
    args.batch_size,
    num_workers=args.num_workers,
    seed=args.seed
)

# --- Prepare Conditioning Norms --- 
try:
    with open(args.lipid_stats_path, 'rb') as f:
        stats = pickle.load(f)
        mean_score = stats['mean']
        std_score = stats['std']
except Exception as e:
    print(f"Error loading lipid stats file: {e}. Cannot proceed.")
    raise e

property_norms = {
    'transfection_score': {
        'mean': torch.tensor(mean_score, device=device, dtype=dtype),
        'mad': torch.tensor(std_score if std_score > 1e-6 else 1.0, device=device, dtype=dtype) # Use std, prevent 0
    }
}
print(f"Using property norms: {property_norms}")

# --- Determine Context Node Features --- 
context_node_nf = 0
if args.conditioning:
    try:
        dummy_batch = next(iter(dataloaders['train']))
        for key in dummy_batch:
             if isinstance(dummy_batch[key], torch.Tensor):
                 dummy_batch[key] = dummy_batch[key].to(device)
        dummy_context = prepare_context(args.conditioning, dummy_batch, property_norms, force_unconditional=False)
        if dummy_context is not None:
            context_node_nf = dummy_context.size(-1)
        else:
             print("Warning: prepare_context returned None for dummy batch.")
    except Exception as e:
        print(f"Error determining context_node_nf: {e}. Assuming 1 for scalar.")
        context_node_nf = 1

args.context_node_nf = context_node_nf
print(f"Determined context_node_nf: {context_node_nf}")

# --- Create Model --- 
# Ensure necessary args for model creation are present
if not hasattr(args, 'train_diffusion'): args.train_diffusion = False
if not hasattr(args, 'trainable_ae'): args.trainable_ae = False

print(f"Instantiating model (train_diffusion={args.train_diffusion})...")
# Assume we are loading a Latent Diffusion model checkpoint
# The original code uses get_latent_diffusion if train_diffusion is True in args
# Ensure args.train_diffusion is set correctly (likely True from loaded args)
if not args.train_diffusion:
     print("Warning: args.train_diffusion is False, but attempting to load/fine-tune diffusion model. Ensure this is intended.")
# Use get_latent_diffusion as it returns the full model needed
model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])

print("Model instantiated.")

# --- Load Pretrained Weights --- 
model_path = join(args.pretrained_path, args.model_name)
try:
    state_dict = torch.load(model_path, map_location=device)
    if all(key.startswith('module.') for key in state_dict.keys()):
        print("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {model_path}")
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
except FileNotFoundError:
    print(f"Error: Pretrained model file not found at {model_path}. Exiting.")
    exit(1)
except Exception as e:
    print(f"Error loading pretrained weights: {e}. Exiting.")
    exit(1)

model = model.to(device)
# DataParallel wrapper (optional, consider DistributedDataParallel for multi-GPU)
# model_dp = torch.nn.DataParallel(model) # Apply wrapper if using multiple GPUs
model_dp = model # No wrapper for single GPU

# --- Setup Optimizer --- 
optim = get_optim(args, model) # Gets optimizer based on args.lr
print(f"Optimizer created with LR: {args.lr}")
# gradnorm_queue needed for original gradient clipping util
gradnorm_queue = utils.Queue() 
gradnorm_queue.add(3000) # Initialize large value

# --- Setup EMA --- 
if args.ema_decay > 0:
    ema = diffusion_utils.EMA(args.ema_decay)
    model_ema = copy.deepcopy(model)
    print(f"EMA enabled with decay: {args.ema_decay}")
else:
    ema = None
    model_ema = model
    print("EMA disabled.")

# --- Training Loop --- 
best_val_loss = float('inf')
print("Starting fine-tuning...")
for epoch in range(args.n_epochs):
    model.train() # Ensure model is in train mode
    model_dp.train() # Ensure wrapper is in train mode if used
    start_time = time.time()
    epoch_loss = 0.0
    epoch_nll = 0.0
    epoch_reg = 0.0
    n_batches = 0

    pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.n_epochs}")
    for i, data in enumerate(pbar):
        # --- Data Preparation --- 
        # (Adapted from train_test.train_epoch)
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype) # Assumes collate_fn provides edge_mask
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype) # Include charges based on dataset

        # Center positions
        x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        # --- Prepare Features Dictionary `h` --- 
        # Match the structure expected by losses.compute_loss_and_nll
        h = {'categorical': one_hot, 'integer': charges}

        # --- CFG Context Preparation --- 
        context = None
        if args.conditioning:
            use_uncond = torch.rand(1).item() < args.cfg_prob
            context = prepare_context(args.conditioning, data, property_norms, force_unconditional=use_uncond)
            if context is not None:
                 # Ensure context is on the correct device
                 context = context.to(device, dtype)
                 diffusion_utils.assert_correctly_masked(context, node_mask) # From train_test

        # --- Loss Calculation --- 
        optim.zero_grad()
        try:
            # Use the loss function from the original codebase
            nll, reg_term, mean_abs_z = qm9_losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                      x, h, node_mask, edge_mask, context)
            loss = nll + args.ode_regularization * reg_term

            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in batch {i}. Skipping.")
                continue

            # --- Backpropagation & Optimization --- 
            loss.backward()

            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            else:
                grad_norm = -1 # Placeholder if not clipped

            optim.step()

            # Update EMA
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
            print(f"Error during training batch {i}: {e}")
            import traceback
            traceback.print_exc()
            continue # Continue to next batch

    # --- End of Epoch --- 
    epoch_duration = time.time() - start_time
    avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
    avg_epoch_nll = epoch_nll / n_batches if n_batches > 0 else 0.0
    avg_epoch_reg = epoch_reg / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f} (NLL: {avg_epoch_nll:.4f}, Reg: {avg_epoch_reg:.4f}), Duration: {epoch_duration:.2f}s")
    if not args.no_wandb:
        wandb.log({'epoch_loss': avg_epoch_loss, 'epoch_nll': avg_epoch_nll, 'epoch_reg': avg_epoch_reg, 'epoch': epoch}, commit=True)

    # --- Validation --- 
    if epoch % args.test_epochs == 0:
        model.eval() # Set model to evaluation mode
        if ema is not None:
             model_eval = model_ema
        else:
             model_eval = model
        model_eval_dp = model_eval # Assign dp wrapper if used

        val_nll = 0.0
        n_val_samples = 0
        with torch.no_grad():
             pbar_val = tqdm(dataloaders['val'], desc=f"Validation Epoch {epoch+1}")
             for data in pbar_val:
                # --- Validation Data Prep --- 
                x = data['positions'].to(device, dtype)
                batch_size = x.size(0)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = data['charges'].to(device, dtype)
                x = diffusion_utils.remove_mean_with_mask(x, node_mask)
                h = {'categorical': one_hot, 'integer': charges}
                
                # --- Validation Context (Always Conditional) --- 
                context = None
                if args.conditioning:
                    context = prepare_context(args.conditioning, data, property_norms, force_unconditional=False)
                    if context is not None:
                         context = context.to(device, dtype)
                         diffusion_utils.assert_correctly_masked(context, node_mask)
                
                # --- Validation Loss Calculation --- 
                try:
                     # Use the same loss function as training
                     nll, _, _ = qm9_losses.compute_loss_and_nll(args, model_eval_dp, nodes_dist, x, h,
                                                                 node_mask, edge_mask, context)
                     if not torch.isnan(nll):
                          val_nll += nll.item() * batch_size # Accumulate total NLL
                          n_val_samples += batch_size
                     else:
                          print("Warning: NaN NLL encountered during validation.")
                except Exception as e:
                    print(f"Error during validation batch: {e}")
                    # Continue validation

        avg_val_nll = val_nll / n_val_samples if n_val_samples > 0 else float('inf')
        print(f"Validation Epoch {epoch+1} Avg NLL: {avg_val_nll:.4f}")
        if not args.no_wandb:
            # Log validation NLL as the primary metric
            wandb.log({'val_nll': avg_val_nll, 'epoch': epoch}, commit=True)

        # --- Save Checkpoint based on Validation NLL --- 
        if avg_val_nll < best_val_loss: # Compare with best NLL
            best_val_loss = avg_val_nll
            print(f"New best validation NLL: {best_val_loss:.4f}. Saving checkpoint...")
            if args.save_model:
                chkpt_dir = join(args.output_dir, 'checkpoints')
                os.makedirs(chkpt_dir, exist_ok=True)
                # Save main model state
                utils.save_model(model, join(chkpt_dir, 'generative_model_last.npy')) 
                # Save EMA model state (best one)
                model_ema_save = model_ema if ema is not None else model
                utils.save_model(model_ema_save, join(chkpt_dir, 'generative_model_ema_best.npy'))
                # Save optimizer state
                utils.save_model(optim, join(chkpt_dir, 'optim_best.npy'))
                # Save arguments used for this run
                args.current_epoch = epoch + 1
                with open(join(chkpt_dir, 'args_best.pickle'), 'wb') as f:
                    pickle.dump(args, f)
                print(f"Checkpoint saved to {chkpt_dir}")

print("Fine-tuning finished.")
if not args.no_wandb:
    wandb.finish() 
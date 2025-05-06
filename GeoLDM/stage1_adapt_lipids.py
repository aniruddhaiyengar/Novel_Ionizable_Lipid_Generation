'''
Stage 1 Adaptation Script: Fine-tuning GeoLDM on unlabeled lipid data.

Adapts a pre-trained GeoLDM model (e.g., from GEOM-Drugs) to the general
chemical space of a target lipid dataset provided via an SDF file (preprocessed).
This script performs UNCONDITIONAL training.

Based on finetune_lipids.py
'''

# Rdkit import should be first, do not move it
from rdkit import Chem


# Local imports
import lipid_dataset # Import our dataset module
from configs.datasets_config import get_dataset_info # Use the function to get dataset info
# Removed: from qm9.utils import prepare_context

# Original GeoLDM imports
import copy
import utils # Original GeoLDM utils
import argparse
import wandb
import os
from os.path import join
from core.models import get_optim, get_latent_diffusion # Import model getter
from equivariant_diffusion import en_diffusion # For type checking
from equivariant_diffusion import utils as diffusion_utils
from core import losses as qm9_losses # Import the loss computation (keeping alias qm9_losses for stability)
import torch
import time
import pickle
from tqdm import tqdm
import numpy as np # Added for np.mean

# Define arguments specific to Stage 1 Adaptation
parser = argparse.ArgumentParser(description='GeoLDM Lipid Adaptation (Stage 1 - Unlabeled)')
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
parser.add_argument('--kl_weight', type=float, default=0.01)
parser.add_argument('--train_diffusion', action='store_true')
parser.add_argument('--trainable_ae', action='store_true')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10], help='normalize factors for [x, categorical, integer]')

# --- Other Arguments --- 
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
parser.add_argument('--wandb_usr', type=str, default=None, help='WandB username')
parser.add_argument('--wandb_project', type=str, default='e3_diffusion_lipid_adapt_stage1', help='WandB project name')
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
    # Load model architecture and diffusion parameters from pretrained model
    for key in keys_to_load:
        if key in pretrained_arg_dict:
            # Only set if not already provided as a command-line argument for this script
            # (Allows overriding some pretrained args if needed, though usually not for arch/diffusion)
            if key not in arg_dict or arg_dict[key] is None:
                 arg_dict[key] = pretrained_arg_dict[key]
        else:
             print(f"  Warning: Key '{key}' not found in pretrained args.pickle.")

    # Set context_node_nf to 0 as we are not using conditioning here
    arg_dict['context_node_nf'] = 0

except FileNotFoundError:
    print(f"Warning: Pretrained args.pickle not found at {pretrained_args_path}. Using defaults for model/diffusion params.")
    # Set context_node_nf to 0 if args file is missing
    args.context_node_nf = 0
except Exception as e:
    print(f"Error loading pretrained args.pickle: {e}. Using defaults.")
    args.context_node_nf = 0


# --- Get Dataset Info --- 
# Use the function from datasets_config.py - Should match the original pretraining dataset (e.g., geom)
args.remove_h = False # Set based on typical GeoLDM pretraining (e.g., geom_with_h)
if args.dataset != 'geom':
     print(f"Warning: Dataset argument is {args.dataset}, but adapting a GEOM model. Using GEOM dataset info.")
     args.dataset = 'geom'
dataset_info = get_dataset_info(dataset_name=args.dataset, remove_h=args.remove_h)
print(f"Using dataset info for: {args.dataset} (remove_h={args.remove_h})")

# --- Wandb Setup --- 
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
    if not args.wandb_usr:
        args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
if not args.no_wandb:
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': args.wandb_project, 'config': args,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

print("Effective Arguments for Stage 1 Adaptation:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

# --- Load Data --- 
# Load the UNLABELED data using the specific path argument
# Use val_split_ratio from args to control validation set size (can be 0)
dataloaders = lipid_dataset.get_dataloaders(
    data_path=args.unlabeled_data_path,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    seed=args.seed,
    val_split_ratio=args.val_split_ratio # Control split here
)

if dataloaders is None:
    print("Error: Failed to create dataloaders. Exiting.")
    exit(1)
if not dataloaders['train']:
    print("Error: Training dataloader is empty. Check data path and preprocessing. Exiting.")
    exit(1)
if not dataloaders['val'] and args.val_split_ratio > 0:
     print("Warning: Validation dataloader is empty, but validation split ratio > 0.")

# --- Prepare Conditioning Norms --- 
# REMOVED: No conditioning norms needed for Stage 1

# --- Determine Context Node Features --- 
# REMOVED: Set context_node_nf=0 when loading args
args.context_node_nf = 0 # Explicitly ensure it's 0
print(f"Running Stage 1 Adaptation (Unconditional): context_node_nf = {args.context_node_nf}")


# --- Create Model --- 
# Ensure necessary args for model creation are present (loaded from pickle)
if not hasattr(args, 'train_diffusion'): args.train_diffusion = False # Default if not in pickle
if not hasattr(args, 'trainable_ae'): args.trainable_ae = False   # Default if not in pickle

print(f"Instantiating model (train_diffusion={args.train_diffusion})...")
# Use get_latent_diffusion, as it handles both AE and Diffusion components
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
    print(f"Error: Pretrained model file not found at {model_path}. Starting from scratch (if model init allows).")
    # Depending on model init, this might fail. Add warning.
    print("Warning: Training may fail if model requires pretrained weights.")
except Exception as e:
    print(f"Error loading pretrained weights: {e}. Exiting.")
    exit(1)

model = model.to(device)
model_dp = model # No DataParallel wrapper for simplicity, assume single GPU

# --- Setup Optimizer --- 
optim = get_optim(args, model) # Gets optimizer based on args.lr
print(f"Optimizer created with LR: {args.lr}")
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
print("Starting Stage 1 Adaptation training...")
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
        # --- Data Preparation --- 
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        h = {'categorical': one_hot, 'integer': charges}

        # --- Context Preparation --- 
        context = None # No context for unconditional adaptation

        # --- Loss Calculation --- 
        optim.zero_grad()
        try:
            # Call loss function with context=None
            nll, reg_term, mean_abs_z = qm9_losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                      x, h, node_mask, edge_mask, context=None)
            loss = nll + args.ode_regularization * reg_term

            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in batch {i}. Skipping.")
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
            print(f"Error during training batch {i}: {e}")
            import traceback
            traceback.print_exc()
            continue # Continue to next batch

    # --- End of Epoch --- 
    epoch_duration = time.time() - start_time
    avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
    avg_epoch_nll = epoch_nll / n_batches if n_batches > 0 else 0.0
    avg_epoch_reg = epoch_reg / n_batches if n_batches > 0 else 0.0
    print(f"Stage 1 Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f} (NLL: {avg_epoch_nll:.4f}, Reg: {avg_epoch_reg:.4f}), Duration: {epoch_duration:.2f}s")
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
                x = data['positions'].to(device, dtype)
                batch_size = x.size(0)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = data['charges'].to(device, dtype)
                x = diffusion_utils.remove_mean_with_mask(x, node_mask)
                h = {'categorical': one_hot, 'integer': charges}
                
                # --- Validation Context --- 
                context = None # No context for unconditional validation
                
                # --- Validation Loss Calculation --- 
                try:
                     # Call loss function with context=None
                     nll, _, _ = qm9_losses.compute_loss_and_nll(args, model_eval_dp, nodes_dist, x, h,
                                                                 node_mask, edge_mask, context=None)
                     if not torch.isnan(nll):
                          val_nll += nll.item() * batch_size
                          n_val_samples += batch_size
                     else:
                          print("Warning: NaN NLL encountered during validation.")
                except Exception as e:
                    print(f"Error during validation batch: {e}")
                    # Continue validation

        avg_val_nll = val_nll / n_val_samples if n_val_samples > 0 else float('inf')
        print(f"Stage 1 Validation Epoch {epoch+1} Avg NLL: {avg_val_nll:.4f}")
        log_dict['val_nll'] = avg_val_nll

        # --- Save Checkpoint based on Validation NLL --- 
        if avg_val_nll < best_val_loss: # Still useful to save best model based on NLL on unlabeled data
            best_val_loss = avg_val_nll
            print(f"New best validation NLL: {best_val_loss:.4f}. Saving checkpoint...")
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
                print(f"Stage 1 Checkpoint saved to {chkpt_dir}")
    else:
         # If not validating, still save periodically or at the end?
         # For now, only saving on validation improvement.
         pass

    # Log epoch metrics to wandb
    if not args.no_wandb:
        wandb.log(log_dict, commit=True)


print("Stage 1 Adaptation finished.")
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
    print(f"Final Stage 1 model state saved to {chkpt_dir}")


if not args.no_wandb:
    wandb.finish() 
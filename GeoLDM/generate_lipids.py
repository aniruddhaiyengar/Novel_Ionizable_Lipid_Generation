from rdkit import Chem
from rdkit.Chem import AllChem


import sys
import os
# Add the parent directory (project root) to sys.path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import pickle
import numpy as np
import argparse
from os.path import join
from tqdm import tqdm

# GeoLDM imports
import GeoLDM.utils as utils
from GeoLDM.configs.datasets_config import get_dataset_info
from GeoLDM.core.models import get_latent_diffusion
from GeoLDM.core import visualizer as core_visualizer
from GeoLDM.equivariant_diffusion.en_diffusion import sample_normal


def main():
    parser = argparse.ArgumentParser(description='Conditional Molecule Generation with GeoLDM using CFG')
    # --- Input Model and Data --- 
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory containing the fine-tuned (Stage 2) model checkpoints and args.pickle')
    parser.add_argument('--model_name', type=str, default='generative_model_ema_best.npy',
                        help='Name of the fine-tuned model state_dict file (usually EMA)')
    parser.add_argument('--stats_path', type=str, default="data/lipid_stats.pkl",
                        help='Path to the dataset statistics file (lipid_stats.pkl) used for Stage 2.')

    # --- Generation Parameters --- 
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Total number of molecules to generate.')
    parser.add_argument('--target_score_std', type=float, default=2.0,
                        help='Target score in terms of standard deviations above the mean.')
    parser.add_argument('--guidance_scale', '--w', type=float, default=5.0,
                        help='Guidance scale (w) for Classifier-Free Guidance.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for generation (adjust based on GPU memory).')
    parser.add_argument('--num_atoms', type=int, default=None,
                        help='If set, generate molecules with exactly this number of atoms. Otherwise, sample from training distribution.')
    parser.add_argument('--diffusion_steps', type=int, default=None,
                        help='Number of diffusion steps. If None, uses value from loaded args.')

    # --- Output --- 
    parser.add_argument('--output_path', type=str, default="generated_lipids_cfg.sdf",
                        help='Path to save the generated molecules (SDF format).')

    # --- Other --- 
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (\'cuda\' or \'cpu\')')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    eval_args = parser.parse_args()

    # --- Load Fine-tuned Model Arguments --- 
    args_path = join(eval_args.model_path, 'args_best.pickle') # Assume using best model args
    try:
        with open(args_path, 'rb') as f:
            args = pickle.load(f)
        print(f"Loaded arguments from {args_path}")
    except FileNotFoundError:
        print(f"Error: args_best.pickle not found in {eval_args.model_path}. Cannot proceed.")
        return
    except Exception as e:
        print(f"Error loading args_best.pickle: {e}")
        return

    # --- Setup Device and Seed --- 
    device = torch.device(eval_args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(eval_args.seed)
    np.random.seed(eval_args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(eval_args.seed)

    # --- Load Dataset Info --- 
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # --- Load Statistics --- 
    try:
        with open(eval_args.stats_path, 'rb') as f:
            stats = pickle.load(f)
            mean_score = stats['mean']
            std_score = stats['std']
            print(f"Loaded stats from {eval_args.stats_path}: mean={mean_score:.4f}, std={std_score:.4f}")
    except FileNotFoundError:
        print(f"Error: Statistics file not found at {eval_args.stats_path}. Cannot proceed with conditioning.")
        return
    except Exception as e:
        print(f"Error loading stats file: {e}")
        return

    # --- Load Model --- 
    if not hasattr(args, 'context_node_nf'):
        print("Warning: context_node_nf not found in loaded args. Assuming 1 for scalar conditioning.")
        args.context_node_nf = 1 
    
    if eval_args.diffusion_steps is not None:
        args.diffusion_steps = eval_args.diffusion_steps
        print(f"Using overridden diffusion steps: {args.diffusion_steps}")
    elif not hasattr(args, 'diffusion_steps'):
        print("Error: diffusion_steps not found in loaded args and not provided. Cannot proceed.")
        return
        
    model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloader=None) 

    model_weights_path = join(eval_args.model_path, eval_args.model_name)
    try:
        state_dict = torch.load(model_weights_path, map_location=device)
        if all(key.startswith('module.') for key in state_dict.keys()):
            print("Removing 'module.' prefix from state_dict keys.")
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded fine-tuned weights from {model_weights_path}")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_weights_path}. Cannot proceed.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)
    model.eval()
    
    # Extract necessary components from the loaded model
    # Assumes model is EnLatentDiffusion or EnVariationalDiffusion
    try:
        dynamics = model.dynamics
        vae = model.vae # Needed for decoding if EnLatentDiffusion
        gamma = model.gamma
        timesteps = model.timesteps
        include_charges = model.include_charges
        print("Extracted dynamics, vae, gamma schedule from model.")
    except AttributeError as e:
        print(f"Error accessing model components (e.g., dynamics, vae, gamma): {e}")
        print("Ensure the loaded model is EnLatentDiffusion or EnVariationalDiffusion type.")
        return

    # --- Prepare Conditional Context --- 
    target_raw_score = mean_score + eval_args.target_score_std * std_score
    print(f"Targeting raw score: {target_raw_score:.4f} ({eval_args.target_score_std:.2f} std dev above mean)")
    target_normalized_score = eval_args.target_score_std 
    
    # Null context for unconditional prediction in CFG
    null_context = torch.zeros(eval_args.batch_size, 1, args.context_node_nf).to(device, dtype)
    # Target context (will resize in loop if last batch is smaller)
    target_context = torch.ones(eval_args.batch_size, 1, args.context_node_nf).to(device, dtype) * target_normalized_score

    # --- Generation Loop --- 
    print(f"Starting conditional generation for {eval_args.n_samples} molecules with CFG (w={eval_args.guidance_scale:.1f})...")
    generated_count = 0
    
    output_dir = os.path.dirname(eval_args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sdf_writer = Chem.SDWriter(eval_args.output_path)
    w = eval_args.guidance_scale # Guidance scale

    with tqdm(total=eval_args.n_samples, desc="Generating Molecules") as pbar:
        while generated_count < eval_args.n_samples:
            current_batch_size = min(eval_args.batch_size, eval_args.n_samples - generated_count)
            
            # Adjust context tensor size for the last batch if smaller
            if target_context.size(0) != current_batch_size:
                target_context = torch.ones(current_batch_size, 1, args.context_node_nf).to(device, dtype) * target_normalized_score
                null_context = torch.zeros(current_batch_size, 1, args.context_node_nf).to(device, dtype)

            # Determine number of atoms for this batch
            if eval_args.num_atoms is not None:
                num_atoms_batch = torch.tensor([eval_args.num_atoms] * current_batch_size).to(device)
            elif nodes_dist is not None:
                num_atoms_batch = nodes_dist.sample(current_batch_size).to(device)
            else:
                print("Error: Cannot determine number of atoms. Provide --num_atoms or ensure nodes_dist is loaded.")
                break
            
            max_n_nodes = int(num_atoms_batch.max().item())
            # Create node mask based on sampled number of atoms for the batch
            node_mask = torch.zeros(current_batch_size, max_n_nodes, 1, device=device, dtype=dtype)
            for i in range(current_batch_size):
                node_mask[i, :num_atoms_batch[i], :] = 1

            # --- Manual CFG Sampling Loop --- 
            # 1. Sample initial noise z_T
            # Note: sample_combined_position_feature_noise handles both positions and features based on model type
            z = model.sample_combined_position_feature_noise(n_samples=current_batch_size, n_nodes=max_n_nodes, node_mask=node_mask)

            # 2. Iterative denoising loop
            for i in reversed(range(timesteps)):
                t_int = torch.full((current_batch_size,), i, device=device, dtype=torch.long)
                s_int = t_int - 1
                
                t = t_int / timesteps
                s = s_int / timesteps
                
                # Fetch gamma values from schedule
                gamma_t = gamma(t)
                gamma_s = gamma(s if s_int >= 0 else torch.zeros_like(s))
                
                # Calculate diffusion coefficients (using model's internal functions)
                sigma2_t_given_s, alpha_t_given_s, _ = model.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z) # broadcasting z just for device/dtype
                sigma_t = model.sigma(gamma_t, z)
                sigma_s = model.sigma(gamma_s, z)
                alpha_t = model.alpha(gamma_t, z)
                alpha_s = model.alpha(gamma_s, z) # Calculate alpha_s

                # Prepare input for dynamics model (time embedding)
                t_time = t.unsqueeze(-1).float() # Add feature dim
                t_batched = t_time.repeat(1, max_n_nodes).unsqueeze(-1) # Repeat per node

                # Predict noise for conditional and unconditional cases
                # Assumes model.dynamics takes (z, t_emb, node_mask, edge_mask=None, context=context)
                # Note: Edge mask might not be needed/used by the dynamics model itself, check EGNN_dynamics_QM9 if issues arise
                pred_eps_cond = model.phi(z, t_batched, node_mask, edge_mask=None, context=target_context)
                pred_eps_uncond = model.phi(z, t_batched, node_mask, edge_mask=None, context=null_context)
                
                # Classifier-Free Guidance prediction
                eps_theta = (1 + w) * pred_eps_cond - w * pred_eps_uncond

                # DDPM Reverse Process Step (as in en_diffusion.py sample_p_zs_given_zt)
                # Constants for the mean calculation
                c1 = alpha_t_given_s / alpha_t
                c2 = sigma2_t_given_s * alpha_s / (alpha_t * sigma_t) # Use alpha_s here
                # Mean of p(z_s | z_t)               
                mu = c1 * z - c2 * eps_theta 
                # Variance of p(z_s | z_t)
                sigma_tilde_t = sigma2_t_given_s * (sigma_s ** 2) / (sigma_t ** 2) 
                # Final sample z_s
                z_s_noise = sample_normal(mu=torch.zeros_like(mu), sigma=torch.ones_like(mu), node_mask=node_mask) 
                z = mu + torch.sqrt(sigma_tilde_t) * z_s_noise

                # Apply mask??? - Check if sample_normal already does this
                z = z * node_mask
                
            # --- End of Denoising Loop --- 
            z_0 = z # Final latent variable
            
            # 3. Decode z_0 using VAE decoder
            try:
                # Pass necessary arguments to decode, context might be needed by decoder too
                h_final, x_final = vae.decode(z_0, node_mask, edge_mask=None, context=target_context) 
            except Exception as e:
                print(f"\nError during decoding batch: {e}")
                import traceback
                traceback.print_exc()
                # Skip batch on error
                generated_count += current_batch_size 
                pbar.update(current_batch_size)
                continue
                
            # --- Convert to RDKit Mol and Save --- 
            atom_types = h_final['categorical'].argmax(dim=-1).cpu().numpy() # Get highest prob atom type
            positions = x_final.cpu().numpy()
            
            mols_batch_indices = np.arange(current_batch_size)
            for idx in mols_batch_indices:
                num_atoms_mol = int(num_atoms_batch[idx].item()) 
                atom_symbols = [dataset_info['atom_decoder'][i] for i in atom_types[idx, :num_atoms_mol]]
                mol_positions = positions[idx, :num_atoms_mol]
                
                try:
                    mol = build_molecule(mol_positions, atom_symbols, dataset_info)
                    if mol is not None:
                        mol.SetDoubleProp("target_score_std", target_normalized_score)
                        mol.SetDoubleProp("guidance_scale", w)
                        mol.SetProp("_Name", f"Generated_CFG_{generated_count + idx + 1}")
                        sdf_writer.write(mol)
                    else:
                        print(f"\nWarning: Failed to build valid RDKit mol for sample {generated_count + idx + 1}")
                except Exception as build_e:
                     print(f"\nError building RDKit mol for sample {generated_count + idx + 1}: {build_e}")
            
            generated_count += current_batch_size
            pbar.update(current_batch_size)

    sdf_writer.close()
    print(f"\nFinished generation. Saved {generated_count} molecules (attempted) to {eval_args.output_path}")

def build_molecule(positions, atom_symbols, dataset_info):
    """ Helper function to build RDKit mol from positions and symbols """
    try:
        mol = Chem.RWMol()
        for atom_symbol in atom_symbols:
            atomic_num = dataset_info['atom_encoder'][atom_symbol]
            mol.AddAtom(Chem.Atom(atomic_num))
        
        conf = Chem.Conformer(len(atom_symbols))
        for i in range(len(atom_symbols)):
            conf.SetAtomPosition(i, positions[i].tolist())
        mol.AddConformer(conf)

        # Infer bonds (simple distance-based approach, might need refinement)
        AllChem.ConnectMol(mol)
        Chem.SanitizeMol(mol)
        return mol.GetMol()
    except Exception as e:
        return None # Return None if building/sanitizing fails

if __name__ == "__main__":
    main() 
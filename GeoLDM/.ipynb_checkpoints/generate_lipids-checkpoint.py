from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


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
import torch.nn as nn
import torch.nn.functional as F
import logging

# GeoLDM imports
import GeoLDM.utils as utils
from GeoLDM.configs.datasets_config import get_dataset_info
from GeoLDM.core.models import get_latent_diffusion
from GeoLDM.core import visualizer as core_visualizer
from GeoLDM.core.rdkit_functions import build_molecule


def calculate_pka(mol):
    """Calculate the pKa of ionizable groups in the molecule."""
    try:
        # Get the most basic nitrogen (likely the ionizable center)
        pka = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                # Check if it's a tertiary amine (common in ionizable lipids)
                if atom.GetDegree() == 3 and atom.GetFormalCharge() == 0:
                    # Estimate pKa based on local environment
                    # This is a simplified model - in practice, you'd want a more sophisticated approach
                    neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
                    if all(n == 6 for n in neighbors):  # All carbon neighbors
                        pka = 8.0  # Typical pKa for tertiary amines
                    elif any(n == 8 for n in neighbors):  # Oxygen neighbors
                        pka = 7.0  # Slightly lower pKa due to electron withdrawal
                    break
        return pka
    except:
        return None

def calculate_logp(mol):
    """Calculate the octanol-water partition coefficient (logP)."""
    try:
        return Chem.Crippen.MolLogP(mol)
    except:
        return None

def calculate_molecular_weight(mol):
    """Calculate the molecular weight in Daltons."""
    try:
        return Chem.Descriptors.ExactMolWt(mol)
    except:
        return None

def is_valid_ionizable_lipid(mol, min_pka=6.0, max_pka=8.5, min_logp=2.0, max_logp=8.0, 
                           min_mw=400, max_mw=800):
    """Check if a molecule meets the criteria for a good ionizable lipid."""
    if mol is None:
        return False, "Invalid molecule"
        
    # Check pKa
    pka = calculate_pka(mol)
    if pka is None:
        return False, "No suitable ionizable group found"
    if not (min_pka <= pka <= max_pka):
        return False, f"pKa {pka:.2f} outside acceptable range [{min_pka}, {max_pka}]"
        
    # Check logP
    logp = calculate_logp(mol)
    if logp is None:
        return False, "Could not calculate logP"
    if not (min_logp <= logp <= max_logp):
        return False, f"logP {logp:.2f} outside acceptable range [{min_logp}, {max_logp}]"
        
    # Check molecular weight
    mw = calculate_molecular_weight(mol)
    if mw is None:
        return False, "Could not calculate molecular weight"
    if not (min_mw <= mw <= max_mw):
        return False, f"Molecular weight {mw:.1f} outside acceptable range [{min_mw}, {max_mw}]"
        
    return True, "Valid ionizable lipid"

def main():
    parser = argparse.ArgumentParser(description='Generate Valid Ionizable Lipids with GeoLDM')
    # --- Input Model and Data --- 
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory containing the fine-tuned (Stage 2) model checkpoints and args.pickle')
    parser.add_argument('--model_name', type=str, default='generative_model_ema_best.npy',
                        help='Name of the fine-tuned model state_dict file (usually EMA)')
    parser.add_argument('--stats_path', type=str, default="data/lipid_stats.pkl",
                        help='Path to the dataset statistics file (lipid_stats.pkl) used for Stage 2.')

    # --- Generation Parameters --- 
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of valid molecules to generate')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for generation')
    parser.add_argument('--output_path', type=str, default="generated_lipids.sdf",
                        help='Path to save the generated molecules (SDF format)')

    # --- Other --- 
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (\'cuda\' or \'cpu\')')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Add new arguments for ionizable lipid filtering
    parser.add_argument('--min_pka', type=float, default=6.0,
                        help='Minimum acceptable pKa for ionizable lipids')
    parser.add_argument('--max_pka', type=float, default=8.5,
                        help='Maximum acceptable pKa for ionizable lipids')
    parser.add_argument('--min_logp', type=float, default=2.0,
                        help='Minimum acceptable logP for ionizable lipids')
    parser.add_argument('--max_logp', type=float, default=8.0,
                        help='Maximum acceptable logP for ionizable lipids')
    parser.add_argument('--min_mw', type=float, default=400,
                        help='Minimum acceptable molecular weight for ionizable lipids')
    parser.add_argument('--max_mw', type=float, default=800,
                        help='Maximum acceptable molecular weight for ionizable lipids')

    # --- CFG Control for Generation ---
    parser.add_argument('--guidance_target_value', type=float, default=None,
                        help='Target value for the conditioning property (e.g., desired transfection score) for CFG. Overrides value from loaded args.')
    parser.add_argument('--print_n_smiles', type=int, default=10,
                        help='Number of SMILES strings to print at the end (0 to disable).')

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
    
    # --- Setup for Classifier-Free Guidance (CFG) ---
    # Guidance scale (w)
    w = args.guidance_scale if hasattr(args, 'guidance_scale') else 0.0 # Default to 0.0 if not in args
    if w > 0 and not args.conditioning:
        print("Warning: Guidance scale w > 0 but no conditioning properties specified in loaded args. CFG might not work as expected.")
    
    # Prepare normalization dictionary for prepare_context
    # Ensure args.conditioning is not empty and is a list
    property_norms_for_context = {}
    guidance_target_for_generation = eval_args.guidance_target_value # Prioritize CLI arg

    if guidance_target_for_generation is None and hasattr(args, 'guidance_target_value'):
        guidance_target_for_generation = args.guidance_target_value # Fallback to loaded args
        print(f"Using guidance_target_value from loaded args: {guidance_target_for_generation}")
    elif eval_args.guidance_target_value is not None:
        print(f"Using guidance_target_value from command line: {guidance_target_for_generation}")

    if w > 0 and guidance_target_for_generation is None:
        print("ERROR: CFG is enabled (guidance_scale > 0) but no guidance_target_value is available. ")
        print("       Please provide --guidance_target_value or ensure it's in the loaded args_best.pickle.")
        return

    if args.conditioning and isinstance(args.conditioning, list) and len(args.conditioning) > 0:
        # Assuming the first conditioning property is the one we target (e.g., 'transfection_score')
        # and its stats are mean_score, std_score directly
        cond_prop_name = args.conditioning[0]
        property_norms_for_context = {
            cond_prop_name: {'mean': mean_score, 'std': std_score}
        }
    elif w > 0:
        print(f"Warning: CFG guidance_scale is {w} but args.conditioning is empty or invalid. Cannot prepare target_context effectively.")

    # Extract necessary components from the loaded model
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

    # --- Generation Loop --- 
    print(f"Starting generation of {eval_args.n_samples} valid ionizable lipids...")
    generated_count = 0
    valid_ionizable_lipids = []
    valid_smiles_strings = [] # Initialize list to store SMILES

    output_dir = os.path.dirname(eval_args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with tqdm(total=eval_args.n_samples, desc="Generating Molecules") as pbar:
        while len(valid_ionizable_lipids) < eval_args.n_samples:
            current_batch_size = min(eval_args.batch_size, eval_args.n_samples - len(valid_ionizable_lipids))
            
            # Determine number of atoms for this batch
            if nodes_dist is not None:
                num_atoms_batch = nodes_dist.sample(current_batch_size).to(device)
            else:
                print("Error: Cannot determine number of atoms. Provide --num_atoms or ensure nodes_dist is loaded.")
                break
            
            max_n_nodes = int(num_atoms_batch.max().item())
            # Create node mask based on sampled number of atoms for the batch
            node_mask = torch.zeros(current_batch_size, max_n_nodes, 1, device=device, dtype=dtype)
            for i in range(current_batch_size):
                node_mask[i, :num_atoms_batch[i], :] = 1

            # --- Prepare Context for CFG ---
            target_context = None
            if args.conditioning and isinstance(args.conditioning, list) and len(args.conditioning) > 0 and guidance_target_for_generation is not None:
                cond_prop_name = args.conditioning[0]
                # Create a "minibatch-like" structure for prepare_context
                target_values_tensor = torch.full((current_batch_size,), guidance_target_for_generation, device=device, dtype=dtype)
                
                # Ensure target_values_tensor is correctly shaped if context_node_nf > 1 later
                # For now, assuming scalar property, so (batch_size,) is fine, prepare_context will handle unsqueezing.
                
                target_properties_batch_for_context = {
                    cond_prop_name: target_values_tensor
                }
                target_context = utils.prepare_context(
                    conditioning=args.conditioning,
                    minibatch=target_properties_batch_for_context,
                    norms=property_norms_for_context
                )
                if target_context is not None:
                    print(f"DEBUG: Generated target_context shape: {target_context.shape}")
            
            # null_context for unconditional part of CFG (model's phi should handle context=None)
            null_context = None

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
                z_s_noise = model.vae.sample_normal(n_samples=current_batch_size, n_nodes=max_n_nodes) 
                z = mu + torch.sqrt(sigma_tilde_t) * z_s_noise

                # Apply mask??? - Check if sample_normal already does this
                z = z * node_mask
                
            # --- End of Denoising Loop --- 
            z_0 = z # Final latent variable
            
            # 3. Decode z_0 using VAE decoder
            try:
                # Pass necessary arguments to decode, context might be needed by decoder too
                h_final, x_final = vae.decode(z_0, node_mask, edge_mask=None, context=target_context) # Pass target_context
            except Exception as e:
                print(f"\nError during decoding batch: {e}")
                import traceback
                traceback.print_exc()
                # Skip batch on error
                generated_count += current_batch_size 
                pbar.update(current_batch_size)
                continue
                
            # --- Convert to RDKit Mol and Save --- 
            atom_types = h_final['categorical'].argmax(dim=-1).cpu().numpy()
            positions = x_final.cpu().numpy()
            
            mols_batch_indices = np.arange(current_batch_size)
            for idx in mols_batch_indices:
                num_atoms_mol = int(num_atoms_batch[idx].item()) 
                atom_symbols = [dataset_info['atom_decoder'][i] for i in atom_types[idx, :num_atoms_mol]]
                mol_positions = positions[idx, :num_atoms_mol]
                
                try:
                    mol = build_molecule(mol_positions, atom_symbols, dataset_info)
                    if mol is not None:
                        # Check if it's a valid ionizable lipid
                        is_valid, reason = is_valid_ionizable_lipid(
                            mol,
                            min_pka=eval_args.min_pka,
                            max_pka=eval_args.max_pka,
                            min_logp=eval_args.min_logp,
                            max_logp=eval_args.max_logp,
                            min_mw=eval_args.min_mw,
                            max_mw=eval_args.max_mw
                        )
                        
                        if is_valid:
                            # Add calculated properties
                            pka = calculate_pka(mol)
                            logp = calculate_logp(mol)
                            mw = calculate_molecular_weight(mol)
                            
                            mol.SetDoubleProp("pKa", pka)
                            mol.SetDoubleProp("logP", logp)
                            mol.SetDoubleProp("MW", mw)
                            mol.SetProp("_Name", f"Generated_{len(valid_ionizable_lipids) + 1}")
                            
                            valid_ionizable_lipids.append(mol)
                            valid_smiles_strings.append(Chem.MolToSmiles(mol)) # Store SMILES string
                            pbar.update(1)
                        else:
                            logging.debug(f"Rejected molecule: {reason}")
                    else:
                        logging.debug("Failed to build valid RDKit mol")
                except Exception as build_e:
                    logging.warning(f"Error building RDKit mol: {build_e}")
            
            # Update progress bar
            pbar.set_postfix({
                'valid': len(valid_ionizable_lipids),
                'total_target': eval_args.n_samples
            })

    # Save all valid molecules
    with Chem.SDWriter(eval_args.output_path) as writer:
        for mol in valid_ionizable_lipids:
            writer.write(mol)
    
    print(f"\nGeneration complete. Generated {len(valid_ionizable_lipids)} valid ionizable lipids.")
    print(f"Output saved to {eval_args.output_path}")
    
    # Print summary statistics
    pka_values = [mol.GetDoubleProp("pKa") for mol in valid_ionizable_lipids]
    logp_values = [mol.GetDoubleProp("logP") for mol in valid_ionizable_lipids]
    mw_values = [mol.GetDoubleProp("MW") for mol in valid_ionizable_lipids]
    
    print("\nProperty Statistics:")
    print(f"pKa: mean={np.mean(pka_values):.2f}, std={np.std(pka_values):.2f}, range=[{min(pka_values):.2f}, {max(pka_values):.2f}]")
    print(f"logP: mean={np.mean(logp_values):.2f}, std={np.std(logp_values):.2f}, range=[{min(logp_values):.2f}, {max(logp_values):.2f}]")
    print(f"MW: mean={np.mean(mw_values):.1f}, std={np.std(mw_values):.1f}, range=[{min(mw_values):.1f}, {max(mw_values):.1f}]")

    # --- Print SMILES --- 
    if eval_args.print_n_smiles > 0 and valid_smiles_strings:
        print(f"\n--- First {min(eval_args.print_n_smiles, len(valid_smiles_strings))} Generated SMILES ---")
        for i, smiles in enumerate(valid_smiles_strings[:eval_args.print_n_smiles]):
            print(f"{i+1}: {smiles}")
    elif eval_args.print_n_smiles > 0:
        print("\nNo valid SMILES strings were generated to print.")

def compute_similarity(mol1, mol2):
    """Compute Tanimoto similarity between two molecules."""
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0  # Return 0 similarity if computation fails

def filter_similar_molecules(molecules, similarity_threshold=0.7):
    """Filter out molecules that are too similar to already accepted ones."""
    if not molecules:
        return []
    
    filtered = [molecules[0]]
    for mol in molecules[1:]:
        # Check similarity against all accepted molecules
        max_similarity = max(compute_similarity(mol, accepted) for accepted in filtered)
        if max_similarity < similarity_threshold:
            filtered.append(mol)
    return filtered

if __name__ == "__main__":
    main() 
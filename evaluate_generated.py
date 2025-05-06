from rdkit import Chem
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') # Disable RDKit logging

import torch
import numpy as np
import argparse
import os
import sys
from scipy.spatial import distance_matrix
from tqdm import tqdm
import pickle   

try:
    # Need to add TransLNP parent directory to sys.path if running script directly
    # and TransLNP is a subdir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) 
    translnp_dir = os.path.join(parent_dir, 'TransLNP')
    if translnp_dir not in sys.path:
        sys.path.insert(0, translnp_dir) # Add TransLNP parent dir

    from models.unimol import UniMolModel # Adjusted import path
    from unicore.data import Dictionary
    # Import padding utils if needed (or replicate logic)
    from utils import pad_1d_tokens, pad_2d, pad_coords 
except ImportError as e:
    print(f"Error importing TransLNP modules: {e}")
    print("Please ensure the TransLNP directory is in your PYTHONPATH or the script is run")
    print("from the parent directory containing TransLNP.")
    exit()
# --- End TransLNP imports ---

def coords2unimol(atoms, coordinates, dictionary, max_atoms=256):
    """
    Converts atoms and coordinates to the UniMol input format dictionary.
    Based on TransLNP/data/conformer.py:coords2unimol

    Args:
        atoms (list): List of atom symbols (e.g., ['C', 'O', 'H']).
        coordinates (np.ndarray): Numpy array of coordinates (N, 3).
        dictionary (Dictionary): UniMol dictionary object.
        max_atoms (int): Maximum number of atoms allowed (truncates if exceeded).

    Returns:
        dict: Dictionary with 'src_tokens', 'src_distance', 'src_coord', 'src_edge_type'.
              Returns None if an atom is not in the dictionary.
    """
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    
    # --- Cropping ---
    if len(atoms) > max_atoms:
        # Simple truncation for now, could be random sampling
        atoms = atoms[:max_atoms]
        coordinates = coordinates[:max_atoms]

    # --- Create src_tokens ---
    try:
        atom_indices = [dictionary.index(atom) for atom in atoms]
    except KeyError as e:
        print(f"Warning: Atom symbol {e} not found in dictionary. Skipping molecule.")
        return None
        
    src_tokens = np.array([dictionary.bos()] + atom_indices + [dictionary.eos()], dtype=int)
    
    # --- Create src_coord ---
    # Center coordinates (excluding BOS/EOS)
    centered_coords = coordinates - coordinates.mean(axis=0)
    # Pad with zeros for BOS/EOS
    src_coord = np.concatenate([np.zeros((1, 3)), centered_coords, np.zeros((1, 3))], axis=0).astype(np.float32)

    # --- Create src_distance ---
    # Calculate pairwise distances on padded coordinates
    src_distance = distance_matrix(src_coord, src_coord).astype(np.float32)

    # --- Create src_edge_type ---
    n_dict = len(dictionary)
    src_edge_type = src_tokens.reshape(-1, 1) * n_dict + src_tokens.reshape(1, -1)
    src_edge_type = src_edge_type.astype(int)

    return {
        'src_tokens': src_tokens,
        'src_distance': src_distance,
        'src_coord': src_coord,
        'src_edge_type': src_edge_type,
    }

def batch_collate_fn_unimol(batch_list, dictionary):
    """
    Collates a list of unimol_input dictionaries into a batch suitable for the model.
    Uses padding utilities similar to UniMolModel.batch_collate_fn.

    Args:
        batch_list (list): List of dictionaries from coords2unimol.
        dictionary (Dictionary): UniMol dictionary object (for padding_idx).

    Returns:
        dict: Dictionary of batched and padded tensors ('src_tokens', 'src_distance', 'src_coord', 'src_edge_type').
    """
    collated_batch = {}
    pad_idx = dictionary.pad()
    
    # Convert numpy arrays to tensors first
    tensor_batch = []
    for item in batch_list:
        tensor_item = {k: torch.from_numpy(v) for k, v in item.items()}
        tensor_batch.append(tensor_item)

    # Pad each component
    collated_batch['src_tokens'] = pad_1d_tokens([s['src_tokens'] for s in tensor_batch], pad_idx=pad_idx)
    collated_batch['src_coord'] = pad_coords([s['src_coord'] for s in tensor_batch], pad_idx=0.0)
    collated_batch['src_distance'] = pad_2d([s['src_distance'] for s in tensor_batch], pad_idx=0.0)
    collated_batch['src_edge_type'] = pad_2d([s['src_edge_type'] for s in tensor_batch], pad_idx=pad_idx) # Pad edge type with pad_idx

    return collated_batch

def load_translnp_model(model_path, dict_path, device):
    """ Loads the UniMol model and dictionary. """
    print(f"Loading dictionary from: {dict_path}")
    try:
        dictionary = Dictionary.load(dict_path)
        # Manually add mask if not present (based on conformer.py)
        if "[MASK]" not in dictionary:
             dictionary.add_symbol("[MASK]", is_special=True)
        print(f"Dictionary loaded: {len(dictionary)} symbols.")
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return None, None

    print(f"Loading UniMol model weights from: {model_path}")
    try:
        # Initialize model - Assuming output_dim=1 for regression, data_type='molecule'
        # remove_hs=False based on user-provided paths (mol_pre_all_h...)
        model = UniMolModel(output_dim=1, data_type='molecule', remove_hs=False).to(device)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle potential wrapping ('model' key) and module prefix
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if all(key.startswith('module.') for key in state_dict.keys()):
            print("Removing 'module.' prefix from state_dict keys.")
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
        # Load into model - allow non-strict loading initially
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
             print(f"Warning: Unexpected keys found in state_dict: {unexpected_keys}")
        if missing_keys:
             print(f"Warning: Missing keys in state_dict: {missing_keys}")
             
        # Try strict loading if no missing/unexpected keys were found
        if not missing_keys and not unexpected_keys:
             model.load_state_dict(state_dict, strict=True)
             print("Strict state_dict loading successful.")
        else:
             print("Loaded state_dict with non-strict matching.")
             
        model.eval() # Set to evaluation mode
        print("UniMol model loaded successfully.")
        return model, dictionary
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Evaluate Generated Lipids with TransLNP')
    # --- Input/Output ---
    parser.add_argument('--input_sdf', type=str, required=True,
                        help='Path to the SDF file containing generated molecules.')
    parser.add_argument('--output_file', type=str, default="evaluated_lipids.sdf",
                        help='Path to save the filtered and scored molecules (SDF format). Set to "none" to disable saving.')
    parser.add_argument('--translnp_model', type=str, default="TransLNP/weights/mol_pre_all_h_220816.pt",
                        help='Path to the pre-trained TransLNP model (.pt file).')
    parser.add_argument('--translnp_dict', type=str, default="TransLNP/weights/mol.dict.txt",
                        help='Path to the TransLNP dictionary file (mol.dict.txt).')

    # --- Filtering ---
    parser.add_argument('--filter_smarts', type=str, default="[NX3;H0;!$(*=[!#6]);!$(*[!#6]=[!#6])]", # Tertiary amine (not connected to double bonds)
                        help='SMARTS pattern for required structural features (e.g., ionizable group). Set to "none" to disable.')
    parser.add_argument('--min_atoms', type=int, default=10, help='Minimum number of atoms (after potential H removal by UniMol).')
    parser.add_argument('--max_atoms', type=int, default=256, help='Maximum number of atoms for UniMol input.')


    # --- Prediction & Ranking ---
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for TransLNP prediction.')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top molecules to print.')

    # --- Other ---
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (\'cuda\' or \'cpu\')')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for numpy (e.g., for potential sampling if max_atoms is hit)')

    args = parser.parse_args()

    # --- Setup ---
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load TransLNP Model and Dictionary ---
    model, dictionary = load_translnp_model(args.translnp_model, args.translnp_dict, device)
    if model is None or dictionary is None:
        print("Failed to load TransLNP model or dictionary. Exiting.")
        return

    # --- Load and Filter Generated Molecules ---
    print(f"Loading generated molecules from: {args.input_sdf}")
    suppl = Chem.SDMolSupplier(args.input_sdf, removeHs=False) # Keep Hs for now, UniMol handles removal if configured
    
    all_mols = [m for m in suppl if m is not None]
    print(f"Read {len(all_mols)} molecules from SDF.")

    valid_mols = []
    filtered_mols_data = [] # Store tuples: (original_mol, unimol_input_dict)
    
    ionizable_pattern = None
    if args.filter_smarts.lower() != "none":
        try:
            ionizable_pattern = Chem.MolFromSmarts(args.filter_smarts)
            print(f"Using SMARTS filter: {args.filter_smarts}")
        except:
            print(f"Error: Invalid SMARTS pattern provided: {args.filter_smarts}. Disabling SMARTS filter.")
            ionizable_pattern = None
            
    print("Filtering molecules (validity, SMARTS, atom count)...")
    skipped_dict = 0
    skipped_smarts = 0
    skipped_atoms = 0
    skipped_validity = 0

    for i, mol in enumerate(tqdm(all_mols, desc="Filtering")):
        # 1. Basic Validity Check (Sanitization)
        try:
            Chem.SanitizeMol(mol)
        except:
            skipped_validity += 1
            continue # Skip invalid molecules
            
        # 2. SMARTS Filter (if enabled)
        if ionizable_pattern is not None and not mol.HasSubstructMatch(ionizable_pattern):
            skipped_smarts += 1
            continue # Skip molecules without the feature

        # 3. Prepare for UniMol (Get atoms/coords, handle H removal implicitly in coords2unimol if remove_hs=True)
        try:
            conf = mol.GetConformer()
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            coordinates = conf.GetPositions()
            
            # Check atom count BEFORE converting (UniMol removes H based on its config)
            # We assume model uses remove_hs=False, so check total atoms
            if not (args.min_atoms <= len(atoms) <= args.max_atoms):
                 skipped_atoms += 1
                 continue

            # 4. Convert to UniMol input format
            # remove_hs is False matching model loading
            unimol_input = coords2unimol(atoms, coordinates, dictionary, max_atoms=args.max_atoms) 
            if unimol_input is None:
                skipped_dict += 1
                continue # Skip if atom not in dictionary

            filtered_mols_data.append((mol, unimol_input)) # Store original mol and processed data
        except Exception as e:
            print(f"Warning: Error processing molecule {i}: {e}")
            skipped_validity += 1 # Count as processing error / validity issue
            continue

    print(f"Filtering complete:")
    print(f"  Initial molecules: {len(all_mols)}")
    print(f"  Skipped (invalid RDKit): {skipped_validity}")
    print(f"  Skipped (SMARTS filter): {skipped_smarts}")
    print(f"  Skipped (atom count): {skipped_atoms}")
    print(f"  Skipped (atom not in dict): {skipped_dict}")
    print(f"  Molecules remaining for prediction: {len(filtered_mols_data)}")

    if not filtered_mols_data:
        print("No molecules passed filtering. Exiting.")
        return

    # --- Predict Scores with TransLNP ---
    print(f"Predicting scores using TransLNP model (batch size: {args.batch_size})...")
    model.eval() # Ensure model is in eval mode
    all_predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(filtered_mols_data), args.batch_size), desc="Predicting"):
            batch_indices = range(i, min(i + args.batch_size, len(filtered_mols_data)))
            batch_unimol_inputs = [filtered_mols_data[j][1] for j in batch_indices]
            
            # Collate batch
            collated_batch = batch_collate_fn_unimol(batch_unimol_inputs, dictionary)
            
            # Move to device
            for k in collated_batch:
                collated_batch[k] = collated_batch[k].to(device)

            # Predict (model forward returns logits, representation)
            logits, _ = model(**collated_batch) # Pass the dictionary directly
            
            # Store predictions (assuming logits are the scores for regression)
            all_predictions.extend(logits.squeeze(-1).cpu().numpy()) # Squeeze output dim if necessary

    print(f"Prediction complete. Obtained {len(all_predictions)} scores.")

    # --- Combine Results and Rank ---
    results = []
    for i, prediction in enumerate(all_predictions):
        original_mol = filtered_mols_data[i][0]
        # Add predicted score as a property to the molecule
        original_mol.SetDoubleProp("predicted_transfection_score", float(prediction))
        results.append((original_mol, float(prediction)))

    # Sort by predicted score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # --- Print Top N ---
    print(f"--- Top {min(args.top_n, len(results))} Molecules ---")
    for i in range(min(args.top_n, len(results))):
        mol, score = results[i]
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Molecule_{i+1}"
        smiles = Chem.MolToSmiles(mol)
        print(f"{i+1}. Name: {name}, Predicted Score: {score:.4f}, SMILES: {smiles}")

    # --- Save Results ---
    if args.output_file.lower() != "none":
        print(f"Saving {len(results)} filtered and scored molecules to: {args.output_file}")
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        writer = Chem.SDWriter(args.output_file)
        for mol, score in results:
            # Ensure the score property is set (should be already)
            if not mol.HasProp("predicted_transfection_score"):
                 mol.SetDoubleProp("predicted_transfection_score", score)
            writer.write(mol)
        writer.close()
        print("Saving complete.")

if __name__ == "__main__":
    main() 
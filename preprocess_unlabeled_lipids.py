import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import sys # For flushing output

# --- Configuration ---
SDF_PATH = "data/structures.sdf"  # Input SDF file path
OUTPUT_DIR = "data"               # Directory to save processed file
PROCESSED_UNLABELED_PKL = os.path.join(OUTPUT_DIR, "processed_unlabeled_lipids.pkl") # Output for processed unlabeled data
MAX_MOLECULES_TO_PROCESS = 10000  # Stop after processing this many molecules

# Define the atom types expected in the dataset (ensure consistency with other scripts)
# ATOM_DECODER = ['H', 'C', 'N', 'O', 'P'] # Original 5 types
# New ATOM_DECODER to match geom_with_h from GeoLDM/configs/datasets_config.py
ATOM_DECODER = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'] # 16 types
ATOM_MAP = {symbol: i for i, symbol in enumerate(ATOM_DECODER)}
NUM_ATOM_TYPES = len(ATOM_DECODER) # Should be 16

# --- Functions (Adapted from preprocess_lipids.py) ---

def generate_conformer_from_mol(mol):
    """
    Adds hydrogens, generates a 3D conformer, optimizes, and computes Gasteiger charges
    for a given RDKit molecule object (potentially read from SDF).
    Tries fallback methods if ETKDG fails.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        rdkit.Chem.Mol or None: The RDKit molecule object with added Hs, 3D conformer, and charges,
                                or None if processing fails.
    """
    try:
        if mol is None:
            return None

        # Get name for logging if available
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown_molecule"

        # Add hydrogens (important for accurate charges and geometry)
        mol = Chem.AddHs(mol, addCoords=True) # addCoords helps if input had some 3D info

        conf_id = -1
        # Try using existing conformer first if available from SDF
        if mol.GetNumConformers() > 0:
            conf_id = 0
            print(f"Info: Using existing conformer for {mol_name}.")
        else:
            # --- Attempt 1: ETKDGv3 with seed ---
            print(f"Info: Attempting ETKDGv3 (seed 42) for {mol_name}.")
            params_seed = AllChem.ETKDGv3()
            params_seed.randomSeed = 42
            params_seed.useRandomCoords = False
            conf_id = AllChem.EmbedMolecule(mol, params_seed)

            # --- Attempt 2: ETKDGv3 without seed ---
            if conf_id < 0:
                print(f"Warning: ETKDGv3 (seed 42) failed for {mol_name}. Trying without seed.")
                params_no_seed = AllChem.ETKDGv3()
                params_no_seed.useRandomCoords = False
                conf_id = AllChem.EmbedMolecule(mol, params_no_seed)

            # --- Attempt 3: ETKDGv3 with random coordinates ---
            if conf_id < 0:
                print(f"Warning: ETKDGv3 (no seed) failed for {mol_name}. Trying with random coords.")
                params_random = AllChem.ETKDGv3()
                params_random.randomSeed = 42 # Keep seed for reproducibility
                params_random.useRandomCoords = True
                conf_id = AllChem.EmbedMolecule(mol, params_random)

        # --- Check final result --- 
        if conf_id < 0:
            print(f"Error: Conformer generation failed for {mol_name} after all attempts. Skipping.")
            return None
        else:
            # Only print success if we had to generate it
            if mol.GetNumConformers() == 0: # Check if we actually generated one vs used existing
                 print(f"Success: Conformer generated for {mol_name} (attempt result code: {conf_id})")


        # Optimize the geometry using MMFF94 force field
        try:
            opt_result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            if opt_result != 0:
                 print(f"Warning: MMFF optimization did not converge (result code {opt_result}) for {mol_name}. Using best found conformer.")
        except Exception as e:
            # Sometimes optimization can fail
            print(f"Warning: MMFF optimization failed unexpectedly for {mol_name}. Using unoptimized conformer. Error: {e}")


        # Compute Gasteiger charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            print(f"Warning: Could not compute Gasteiger charges for {mol_name}. Charges will be set to 0. Error: {e}")
            # Assign a default charge of 0.0 if calculation fails
            for atom in mol.GetAtoms():
                atom.SetDoubleProp('_GasteigerCharge', 0.0)

        return mol
    except Exception as e:
        print(f"Error processing a molecule from SDF: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_molecule(mol):
    """
    Extracts relevant information from an RDKit molecule object for the dataset.
    (Modified to not require target_score).

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object with a 3D conformer.

    Returns:
        dict or None: A dictionary containing processed molecule data
                      ('positions', 'one_hot', 'charges', 'atom_mask'),
                      or None if an atom type is not in ATOM_DECODER or other error.
    """
    try:
        # Get the conformer (assuming one exists after generate_conformer_from_mol)
        conformer = mol.GetConformer()
        positions = conformer.GetPositions() # Get atom coordinates (NumAtoms x 3)

        num_atoms = mol.GetNumAtoms()
        one_hot = np.zeros((num_atoms, NUM_ATOM_TYPES), dtype=np.float32) # Will be (num_atoms, 16)
        # charges = np.zeros((num_atoms, 1)) # Original: Gasteiger charge as 1 feature
        
        # To match GEOM's include_charges=False behavior where charges don't add to feature dim:
        # We'll still calculate Gasteiger for potential future use or inspection,
        # but the 'charges' field in the output dict will be 0-dimensional.
        actual_gasteiger_charges = np.zeros((num_atoms, 1), dtype=np.float32) 
        output_charges_feature = np.zeros((num_atoms, 0), dtype=np.float32) # 0-dimensional feature

        atom_mask = np.ones((num_atoms, 1), dtype=np.float32) # Mask indicating which atoms are real

        # Extract atom features
        valid_molecule = True
        for i, atom in enumerate(mol.GetAtoms()):
            # Get atomic symbol
            symbol = atom.GetSymbol()
            if symbol not in ATOM_MAP:
                # Print only once per run for a given symbol to avoid flooding
                if not hasattr(process_molecule, "warned_symbols"):
                    process_molecule.warned_symbols = set()
                if symbol not in process_molecule.warned_symbols:
                    print(f"\nWarning: Atom type '{symbol}' not in ATOM_DECODER {ATOM_DECODER}. Skipping molecule(s) with this atom type.")
                    process_molecule.warned_symbols.add(symbol)
                valid_molecule = False
                break # Stop processing this molecule

            # One-hot encode atom type
            one_hot[i, ATOM_MAP[symbol]] = 1

            # Get Gasteiger charge
            charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
            # charges[i, 0] = charge # Store in original charges array
            actual_gasteiger_charges[i, 0] = charge # Store calculated Gasteiger charge separately

        if not valid_molecule:
            return None

        # Prepare data dictionary
        data = {
            'positions': positions.astype(np.float32),
            'one_hot': one_hot.astype(np.float32), # This is (N, 16)
            # 'charges': charges.astype(np.float32), # Original
            'charges': output_charges_feature, # This is (N, 0)
            'atom_mask': atom_mask.astype(np.float32)
            # No conditioning key needed here
            # Optionally, we could store the actual Gasteiger charges under a different key if needed for analysis
            # 'gasteiger_charges_raw': actual_gasteiger_charges 
        }
        return data
    except AttributeError:
        # Likely error if conformer doesn't exist or GetPositions fails
        print("\nError: Failed to get conformer or positions for a molecule. Skipping.")
        return None
    except Exception as e:
        print(f"\nError extracting features for a molecule: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting unlabeled lipid data preprocessing from SDF...")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

    processed_molecules = []

    print(f"Reading molecules from: {SDF_PATH}")
    # Use SDMolSupplier to read molecules iteratively from SDF
    suppl = Chem.SDMolSupplier(SDF_PATH)

    # Iterate through molecules using tqdm for progress bar
    # We don't know the total easily, so the progress bar won't show total percentage
    mol_iterator = tqdm(suppl, desc=f"Processing {os.path.basename(SDF_PATH)}")

    processed_count = 0
    skipped_count = 0
    total_read = 0 # Keep track of how many were read from SDF

    for i, mol in enumerate(mol_iterator):
        total_read = i + 1
        if mol is None:
            skipped_count += 1
            continue

        # Generate conformer, add Hs, compute charges
        processed_mol = generate_conformer_from_mol(mol)
        if processed_mol is None:
            skipped_count += 1
            continue # Skip if conformer generation/processing failed

        # Extract features
        molecule_data = process_molecule(processed_mol)
        if molecule_data is not None:
            processed_molecules.append(molecule_data)
            processed_count += 1
        else:
            skipped_count += 1

        # Update tqdm description periodically
        if total_read % 100 == 0:
             mol_iterator.set_postfix({"Read": total_read, "Processed": processed_count, "Skipped": skipped_count})

        # Check if we have reached the desired number of processed molecules
        if processed_count >= MAX_MOLECULES_TO_PROCESS:
            print(f"\nReached limit of {MAX_MOLECULES_TO_PROCESS} successfully processed molecules. Stopping SDF processing.")
            break # Exit the loop

    # Final update for tqdm after loop finishes or breaks
    mol_iterator.set_postfix({"Read": total_read, "Processed": processed_count, "Skipped": skipped_count})
    mol_iterator.close() # Close the tqdm bar cleanly
    print(f"\nFinished reading SDF. Total attempted: {total_read}, Successfully processed: {processed_count}, Skipped: {skipped_count}")


    # Save the processed data
    if processed_molecules:
        print(f"Saving {len(processed_molecules)} processed unlabeled molecules (up to limit) to {PROCESSED_UNLABELED_PKL}")
        with open(PROCESSED_UNLABELED_PKL, 'wb') as f:
            pickle.dump(processed_molecules, f)
        print("Save complete.")
    else:
        print("No molecules were successfully processed from the SDF file.")

    print("Unlabeled preprocessing finished.") 
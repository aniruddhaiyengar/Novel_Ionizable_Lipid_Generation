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

# Define the atom types expected in the dataset (ensure consistency with other scripts)
ATOM_DECODER = ['H', 'C', 'N', 'O', 'P']
ATOM_MAP = {symbol: i for i, symbol in enumerate(ATOM_DECODER)}
NUM_ATOM_TYPES = len(ATOM_DECODER)

# --- Functions (Adapted from preprocess_lipids.py) ---

def generate_conformer_from_mol(mol):
    """
    Adds hydrogens, generates a 3D conformer, optimizes, and computes Gasteiger charges
    for a given RDKit molecule object (potentially read from SDF).

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        rdkit.Chem.Mol or None: The RDKit molecule object with added Hs, 3D conformer, and charges,
                                or None if processing fails.
    """
    try:
        if mol is None:
            return None

        # Add hydrogens (important for accurate charges and geometry)
        mol = Chem.AddHs(mol, addCoords=True) # addCoords helps if input had some 3D info

        # Generate 3D coordinates using ETKDGv3 algorithm if no conformers exist
        # Or try to use existing conformer from SDF if available
        conf_id = -1
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42 # Ensure reproducibility
            conf_id = AllChem.EmbedMolecule(mol, params)

            if conf_id < 0:
                print(f"\nWarning: Could not generate conformer for a molecule from SDF. Trying fallback.")
                # Try without random seed as a fallback
                params = AllChem.ETKDGv3()
                conf_id = AllChem.EmbedMolecule(mol, params)
                if conf_id < 0:
                     print(f"\nWarning: Conformer generation failed even without seed for a molecule from SDF. Skipping.")
                     return None
        else:
             # Use the first existing conformer
             conf_id = 0


        # Optimize the geometry using MMFF94 force field
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        except Exception as e:
            # Sometimes optimization can fail
            print(f"\nWarning: MMFF optimization failed for a molecule from SDF. Using unoptimized conformer. Error: {e}")


        # Compute Gasteiger charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
             print(f"\nWarning: Could not compute Gasteiger charges for a molecule from SDF. Charges will be set to 0. Error: {e}")
             # Assign a default charge of 0.0 if calculation fails
             for atom in mol.GetAtoms():
                 atom.SetDoubleProp('_GasteigerCharge', 0.0)

        return mol
    except Exception as e:
        print(f"\nError processing a molecule from SDF: {e}")
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
        one_hot = np.zeros((num_atoms, NUM_ATOM_TYPES))
        charges = np.zeros((num_atoms, 1))
        atom_mask = np.ones((num_atoms, 1)) # Mask indicating which atoms are real

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
            charges[i, 0] = charge

        if not valid_molecule:
            return None

        # Prepare data dictionary
        data = {
            'positions': positions.astype(np.float32),
            'one_hot': one_hot.astype(np.float32),
            'charges': charges.astype(np.float32),
            'atom_mask': atom_mask.astype(np.float32)
            # No conditioning key needed here
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
    # Need to handle potential errors during file reading or molecule parsing
    suppl = Chem.SDMolSupplier(SDF_PATH)

    # Iterate through molecules using tqdm for progress bar
    # Determine total count first if possible for accurate progress bar
    # SDMolSupplier doesn't easily give count without iteration, so we estimate or omit total.
    mol_iterator = tqdm(suppl, desc=f"Processing {os.path.basename(SDF_PATH)}")

    processed_count = 0
    skipped_count = 0

    for i, mol in enumerate(mol_iterator):
        if mol is None:
            # print(f"\nWarning: Failed to read molecule at index {i} from SDF. Skipping.")
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
        if (i + 1) % 100 == 0:
             mol_iterator.set_postfix({"Processed": processed_count, "Skipped": skipped_count})

    # Final update
    mol_iterator.set_postfix({"Processed": processed_count, "Skipped": skipped_count})
    print(f"\nFinished reading SDF. Processed: {processed_count}, Skipped: {skipped_count}")


    # Save the processed data
    if processed_molecules:
        print(f"Saving {len(processed_molecules)} processed unlabeled molecules to {PROCESSED_UNLABELED_PKL}")
        with open(PROCESSED_UNLABELED_PKL, 'wb') as f:
            pickle.dump(processed_molecules, f)
        print("Save complete.")
    else:
        print("No molecules were successfully processed from the SDF file.")

    print("Unlabeled preprocessing finished.") 
import numpy as np
import pickle
import os
import logging # Use logging module for better control
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Configuration ---
SDF_PATH = "data/structures.sdf"
OUTPUT_DIR = "data"
PROCESSED_UNLABELED_PKL = os.path.join(OUTPUT_DIR, "processed_unlabeled_lipids.pkl")
MAX_MOLECULES_TO_PROCESS = 10000

ATOM_DECODER = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
ATOM_MAP = {symbol: i for i, symbol in enumerate(ATOM_DECODER)}
NUM_ATOM_TYPES = len(ATOM_DECODER)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Keep track of symbols that caused warnings to avoid flooding logs
_warned_atom_symbols = set()


def generate_conformer(mol):
    """
    Adds hydrogens, generates a 3D conformer, optimizes, and computes Gasteiger charges.
    """
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown_molecule"
    # logging.info(f"Processing: {mol_name} (Adding Hs)")
    mol = Chem.AddHs(mol, explicitOnly=False, addCoords=True)

    conf_id = -1
    if mol.GetNumConformers() > 0:
        conf_id = 0 # Use existing conformer
        # logging.debug(f"Using existing conformer for {mol_name}.")
    else:
        # logging.debug(f"Attempting conformer generation for {mol_name}.")
        params = [AllChem.ETKDGv3() for _ in range(3)]
        params[0].randomSeed = 42
        params[0].useRandomCoords = False
        params[1].useRandomCoords = False # Default seed
        params[2].randomSeed = 42
        params[2].useRandomCoords = True
        
        attempt_descs = [
            "ETKDGv3 (seed 42)",
            "ETKDGv3 (default seed)",
            "ETKDGv3 (random coords, seed 42)"
        ]

        for i, p in enumerate(params):
            conf_id = AllChem.EmbedMolecule(mol, p)
            if conf_id >= 0:
                # logging.debug(f"Conformer generated with {attempt_descs[i]} for {mol_name}.")
                break
            # else:
                # logging.warning(f"{attempt_descs[i]} failed for {mol_name}.")
        
        if conf_id < 0:
            logging.error(f"Conformer generation failed for {mol_name} after all attempts. Skipping.")
            return None

    try:
        opt_result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        if opt_result != 0:
            logging.warning(f"MMFF optimization did not converge (code {opt_result}) for {mol_name}. Using best found conformer.")
    except Exception as e:
        logging.warning(f"MMFF optimization failed for {mol_name}: {e}. Using unoptimized conformer.")

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        logging.warning(f"Gasteiger charge computation failed for {mol_name}: {e}. Charges set to 0.")
        for atom in mol.GetAtoms():
            atom.SetDoubleProp('_GasteigerCharge', 0.0)
    return mol

def extract_molecule_features(mol):
    """
    Extracts positions, one-hot encoding, charges, and atom_mask from an RDKit molecule.
    """
    global _warned_atom_symbols
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown_in_extract"
    try:
        conformer = mol.GetConformer() # Assumes conformer_id=0 is the one we want if multiple exist
        positions = conformer.GetPositions().astype(np.float32)
        num_atoms = mol.GetNumAtoms()

        # logging.debug(f"Extracting features for {mol_name}: NumAtoms={num_atoms}, Positions Shape={positions.shape}")


        one_hot = np.zeros((num_atoms, NUM_ATOM_TYPES), dtype=np.float32)
        # Charges are 0-dimensional feature vector to match GEOM non-charged features
        # Gasteiger charges are computed on mol object but not directly part of 'charges' unless modified
        output_charges_feature = np.zeros((num_atoms, 0), dtype=np.float32)
        atom_mask = np.ones((num_atoms, 1), dtype=np.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if symbol not in ATOM_MAP:
                if symbol not in _warned_atom_symbols:
                    logging.warning(f"Atom type '{symbol}' not in ATOM_DECODER. Molecules with this atom type will be skipped. Molecule: {mol_name}")
                    _warned_atom_symbols.add(symbol)
                return None
            one_hot[i, ATOM_MAP[symbol]] = 1
            # Gasteiger charge is on atom object, not added to 'output_charges_feature' here

        return {
            'positions': positions,
            'one_hot': one_hot,
            'charges': output_charges_feature,
            'atom_mask': atom_mask
        }
    except Exception as e:
        logging.error(f"Error extracting features for {mol_name}: {e}", exc_info=True)
        return None

def main():
    logging.info("Starting unlabeled lipid data preprocessing from SDF...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed_molecules_data = []
    
    logging.info(f"Reading molecules from: {SDF_PATH}")
    # removeHs=False, sanitize=True are defaults and generally good.
    suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False, sanitize=True)

    processed_count = 0
    skipped_count = 0
    total_read = 0

    # Iterating with tqdm
    # Since total number of molecules in SDF is unknown, progress bar won't show total
    # mol_iterator = tqdm(suppl, desc=f"Processing {os.path.basename(SDF_PATH)}", unit="mol")

    for i, mol_initial in enumerate(suppl):
        total_read += 1
        if mol_initial is None:
            # logging.warning(f"Molecule at SDF index {i} could not be read by RDKit.")
            skipped_count += 1
            continue

        mol_with_conformer = generate_conformer(mol_initial)
        if mol_with_conformer is None:
            skipped_count += 1
            continue

        features = extract_molecule_features(mol_with_conformer)
        if features is not None:
            processed_molecules_data.append(features)
            processed_count += 1
        else:
            skipped_count += 1
        
        if total_read % 500 == 0: # Log progress every 500 molecules
             logging.info(f"Progress: Read {total_read}, Processed {processed_count}, Skipped {skipped_count}")

        if processed_count >= MAX_MOLECULES_TO_PROCESS:
            logging.info(f"Reached limit of {MAX_MOLECULES_TO_PROCESS} successfully processed molecules.")
            break
    
    # mol_iterator.close()
    logging.info(f"Finished SDF processing. Total attempted: {total_read}, Successfully processed: {processed_count}, Skipped: {skipped_count}")

    if processed_molecules_data:
        logging.info(f"Saving {len(processed_molecules_data)} processed molecules to {PROCESSED_UNLABELED_PKL}")
        try:
            with open(PROCESSED_UNLABELED_PKL, 'wb') as f:
                pickle.dump(processed_molecules_data, f)
            logging.info("Save complete.")
        except Exception as e:
            logging.error(f"Failed to save processed data: {e}", exc_info=True)
    else:
        logging.info("No molecules were successfully processed to save.")

    logging.info("Unlabeled preprocessing finished.")

if __name__ == "__main__":
    main() 
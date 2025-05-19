import pandas as pd
import numpy as np
import pickle
import os
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Configuration ---
TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
SMILES_COL = "SMILES"
TARGET_COL = "TARGET"
CONDITIONING_KEY = "transfection_score"

OUTPUT_DIR = "data"
PROCESSED_TRAIN_PKL = os.path.join(OUTPUT_DIR, "processed_train_lipids.pkl")
PROCESSED_TEST_PKL = os.path.join(OUTPUT_DIR, "processed_test_lipids.pkl")
STATS_PKL = os.path.join(OUTPUT_DIR, "lipid_stats.pkl")

ATOM_DECODER = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
ATOM_MAP = {symbol: i for i, symbol in enumerate(ATOM_DECODER)}
NUM_ATOM_TYPES = len(ATOM_DECODER)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_warned_atom_symbols_csv = set()

# --- Functions ---

def generate_conformer_from_smiles(smiles):
    """
    Generates a 3D conformer from SMILES, adds Hs, optimizes, and computes Gasteiger charges.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Could not parse SMILES: {smiles}. Skipping.")
            return None

        # logging.info(f"Processing SMILES: {smiles} (Adding Hs)")
        mol = Chem.AddHs(mol, explicitOnly=False, addCoords=True)

        conf_id = -1
        # ETKDGv3 parameters for attempts
        params_list = [AllChem.ETKDGv3() for _ in range(3)]
        params_list[0].randomSeed = 42
        params_list[0].useRandomCoords = False
        params_list[1].useRandomCoords = False # Default seed
        params_list[2].randomSeed = 42
        params_list[2].useRandomCoords = True
        
        attempt_descs = [
            "ETKDGv3 (seed 42)",
            "ETKDGv3 (default seed)",
            "ETKDGv3 (random coords, seed 42)"
        ]

        for i, params in enumerate(params_list):
            conf_id = AllChem.EmbedMolecule(mol, params)
            if conf_id >= 0:
                # logging.debug(f"Conformer generated with {attempt_descs[i]} for SMILES: {smiles}.")
                break
            # else:
                # logging.warning(f"{attempt_descs[i]} failed for SMILES: {smiles}.")

        if conf_id < 0:
            logging.error(f"Conformer generation failed for SMILES: {smiles} after all attempts. Skipping.")
            return None
        # else:
            # logging.debug(f"Successfully generated conformer for SMILES: {smiles}.")

        try:
            opt_result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            if opt_result != 0:
                logging.warning(f"MMFF optimization did not converge (code {opt_result}) for SMILES: {smiles}.")
        except Exception as e:
            logging.warning(f"MMFF optimization failed for SMILES: {smiles}: {e}.")

        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            logging.warning(f"Gasteiger charge computation failed for SMILES: {smiles}: {e}. Charges set to 0.")
            for atom in mol.GetAtoms():
                atom.SetDoubleProp('_GasteigerCharge', 0.0)
        return mol
    except Exception as e:
        logging.error(f"General error processing SMILES {smiles}: {e}", exc_info=True)
        return None

def extract_molecule_features(mol, target_score):
    """
    Extracts features from an RDKit molecule with a conformer.
    """
    global _warned_atom_symbols_csv
    smiles_for_log = Chem.MolToSmiles(mol) # For logging if name is not available
    try:
        conformer = mol.GetConformer()
        positions = conformer.GetPositions().astype(np.float32)
        num_atoms = mol.GetNumAtoms()

        one_hot = np.zeros((num_atoms, NUM_ATOM_TYPES), dtype=np.float32)
        output_charges_feature = np.zeros((num_atoms, 0), dtype=np.float32) # 0-dim for GEOM compatibility
        atom_mask = np.ones((num_atoms, 1), dtype=np.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if symbol not in ATOM_MAP:
                if symbol not in _warned_atom_symbols_csv:
                    logging.warning(f"Atom type '{symbol}' not in ATOM_DECODER. Molecules containing it will be skipped. SMILES: {smiles_for_log}")
                    _warned_atom_symbols_csv.add(symbol)
                return None
            one_hot[i, ATOM_MAP[symbol]] = 1
            # Gasteiger charges are on the mol object, not part of 'output_charges_feature' here

        return {
            'positions': positions,
            'one_hot': one_hot,
            'charges': output_charges_feature,
            'atom_mask': atom_mask,
            CONDITIONING_KEY: float(target_score) # Ensure target_score is float
        }
    except Exception as e:
        logging.error(f"Error extracting features for SMILES {smiles_for_log}: {e}", exc_info=True)
        return None

def run_preprocessing_for_csv(csv_path, output_pkl_path):
    """
    Reads CSV, processes molecules, and saves them to a pickle file.
    Returns a list of target scores for successfully processed molecules.
    """
    logging.info(f"Starting preprocessing for: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading CSV {csv_path}: {e}", exc_info=True)
        return []

    if SMILES_COL not in df.columns or TARGET_COL not in df.columns:
        logging.error(f"Required columns '{SMILES_COL}' or '{TARGET_COL}' not in {csv_path}. Skipping.")
        return []

    processed_data_list = []
    collected_target_scores = []

    # Create tqdm iterator
    progress_bar_desc = f"Processing {os.path.basename(csv_path)}"
    tqdm_iterator = tqdm(df.iterrows(), total=df.shape[0], desc=progress_bar_desc)

    for index, row in tqdm_iterator: # Use the created iterator
        smiles = row[SMILES_COL]
        target_score = row[TARGET_COL]

        if not isinstance(smiles, str) or pd.isna(smiles) or pd.isna(target_score):
            # logging.debug(f"Skipping row {index + 2} in {os.path.basename(csv_path)}: invalid SMILES or missing target score.")
            continue

        mol_with_conformer = generate_conformer_from_smiles(smiles)
        if mol_with_conformer is None:
            continue

        features = extract_molecule_features(mol_with_conformer, target_score)
        if features is not None:
            processed_data_list.append(features)
            collected_target_scores.append(float(target_score))
            # Update tqdm description with the count of successfully processed molecules
            tqdm_iterator.set_description_str(f"{progress_bar_desc} (Processed: {len(processed_data_list)})")

    if processed_data_list:
        logging.info(f"Saving {len(processed_data_list)} processed molecules from {os.path.basename(csv_path)} to {output_pkl_path}")
        try:
            os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(processed_data_list, f)
            logging.info(f"Successfully saved {output_pkl_path}")
        except Exception as e:
            logging.error(f"Failed to save processed data to {output_pkl_path}: {e}", exc_info=True)
    else:
        logging.info(f"No molecules successfully processed from {os.path.basename(csv_path)}.")

    return collected_target_scores

def main():
    logging.info("Starting all lipid data preprocessing (from CSVs)..." )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_target_scores = run_preprocessing_for_csv(TRAIN_CSV_PATH, PROCESSED_TRAIN_PKL)
    run_preprocessing_for_csv(TEST_CSV_PATH, PROCESSED_TEST_PKL) # Test scores not used for stats

    if train_target_scores:
        scores_array = np.array(train_target_scores, dtype=np.float64)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)

        if std_score < 1e-8: # Avoid division by zero if all scores are identical
            logging.warning(f"Standard deviation of target scores is very low ({std_score}). Setting to 1.0 for stats.")
            std_score = 1.0
        
        stats = {'mean': mean_score, 'std': std_score}
        logging.info(f"Calculated statistics from training data: Mean={mean_score:.4f}, Std={std_score:.4f}")
        
        try:
            with open(STATS_PKL, 'wb') as f:
                pickle.dump(stats, f)
            logging.info(f"Successfully saved statistics to {STATS_PKL}")
        except Exception as e:
            logging.error(f"Failed to save statistics to {STATS_PKL}: {e}", exc_info=True)
    else:
        logging.warning("No training scores collected; statistics cannot be calculated or saved.")

    logging.info("All lipid preprocessing finished.")

if __name__ == "__main__":
    main() 
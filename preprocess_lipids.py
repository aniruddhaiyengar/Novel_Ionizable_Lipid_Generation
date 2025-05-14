import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import logging

# --- Configuration ---
TRAIN_CSV_PATH = "data/train.csv" # Input training CSV file path
TEST_CSV_PATH = "data/test.csv"   # Input testing CSV file path
SMILES_COL = "SMILES"             # Column name for SMILES strings
TARGET_COL = "TARGET"             # Column name for the target property (transfection score)
CONDITIONING_KEY = "transfection_score" # Key to use for the target property in the output dictionary

OUTPUT_DIR = "data"               # Directory to save processed files
PROCESSED_TRAIN_PKL = os.path.join(OUTPUT_DIR, "processed_train_lipids.pkl") # Output for processed training data
PROCESSED_TEST_PKL = os.path.join(OUTPUT_DIR, "processed_test_lipids.pkl")   # Output for processed testing data
STATS_PKL = os.path.join(OUTPUT_DIR, "lipid_stats.pkl") # Output for dataset statistics (mean, std)

# Define the atom types expected in the dataset (adjust if necessary)
# This order determines the one-hot encoding index.
# Match exactly with the model's atom decoder
ATOM_DECODER = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'] # 16 types
ATOM_MAP = {symbol: i for i, symbol in enumerate(ATOM_DECODER)}
NUM_ATOM_TYPES = len(ATOM_DECODER)

# --- Functions ---

def generate_conformer(smiles):
    """
    Generates a 3D conformer for a given SMILES string using RDKit ETKDG.
    Computes Gasteiger charges. Tries fallback methods if ETKDG fails.
    Note: Does not add hydrogens as we're working with molecules without hydrogens.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        rdkit.Chem.Mol or None: The RDKit molecule object with a 3D conformer and charges,
                                or None if conformer generation fails.
    """
    try:
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return None

        # Note: We don't add hydrogens here since we're working with molecules without hydrogens

        # --- Attempt 1: ETKDGv3 with seed ---
        params_seed = AllChem.ETKDGv3()
        params_seed.randomSeed = 42 # Ensure reproducibility
        params_seed.useRandomCoords = False # Default ETKDG
        conf_id = AllChem.EmbedMolecule(mol, params_seed)

        # --- Attempt 2: ETKDGv3 without seed ---
        if conf_id < 0:
            print(f"Warning: ETKDGv3 (seed 42) failed for SMILES: {smiles}. Trying without seed.")
            params_no_seed = AllChem.ETKDGv3()
            params_no_seed.useRandomCoords = False
            conf_id = AllChem.EmbedMolecule(mol, params_no_seed)

        # --- Attempt 3: ETKDGv3 with random coordinates ---
        if conf_id < 0:
            print(f"Warning: ETKDGv3 (no seed) failed for SMILES: {smiles}. Trying with random coords.")
            params_random = AllChem.ETKDGv3()
            # Use seed for reproducibility even with random coords
            params_random.randomSeed = 42
            params_random.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, params_random)

        # --- Check final result --- 
        if conf_id < 0:
             print(f"Error: Conformer generation failed for SMILES: {smiles} after all attempts. Skipping.")
             return None
        else:
            print(f"Success: Conformer generated for SMILES: {smiles} (attempt result code: {conf_id})")

        # Optimize the geometry using MMFF94 force field
        try:
            opt_result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            # Optional: Check opt_result, 0 is success, 1 means optimization failed to converge
            if opt_result != 0:
                 print(f"Warning: MMFF optimization did not converge (result code {opt_result}) for SMILES: {smiles}. Using best found conformer.")
        except Exception as e:
            # Sometimes optimization can fail, but the embedded conformer might still be usable
            print(f"Warning: MMFF optimization failed unexpectedly for SMILES: {smiles}. Using unoptimized conformer. Error: {e}")


        # Compute Gasteiger charges (often used in ML models)
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
             # Handle potential errors during charge calculation
             print(f"Warning: Could not compute Gasteiger charges for SMILES: {smiles}. Charges will be set to 0. Error: {e}")
             # Assign a default charge of 0.0 if calculation fails
             for atom in mol.GetAtoms():
                 atom.SetDoubleProp('_GasteigerCharge', 0.0)


        return mol
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_molecule(mol, target_score):
    """
    Extracts relevant information from an RDKit molecule object for the dataset.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object with a 3D conformer.
        target_score (float): The target property value (e.g., transfection score).

    Returns:
        dict or None: A dictionary containing processed molecule data
                      ('positions', 'one_hot', 'charges', 'atom_mask', CONDITIONING_KEY),
                      or None if an atom type is not in ATOM_DECODER.
    """
    try:
        # Get the conformer
        conformer = mol.GetConformer()
        positions = conformer.GetPositions() # Get atom coordinates (NumAtoms x 3)

        num_atoms = mol.GetNumAtoms()
        one_hot = np.zeros((num_atoms, NUM_ATOM_TYPES), dtype=np.float32) # Will be (num_atoms, 16)
        
        # To match GEOM's include_charges=False behavior where charges don't add to feature dim:
        actual_gasteiger_charges = np.zeros((num_atoms, 1), dtype=np.float32) 
        output_charges_feature = np.zeros((num_atoms, 0), dtype=np.float32) # 0-dimensional feature
        
        atom_mask = np.ones((num_atoms, 1), dtype=np.float32) # Mask indicating which atoms are real

        # Extract atom features
        valid_molecule = True
        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in ATOM_MAP:
                logging.warning(f"Invalid atom type {atom_symbol} found in molecule. Skipping.")
                valid_molecule = False
                break
            one_hot[i, ATOM_MAP[atom_symbol]] = 1
            actual_gasteiger_charges[i] = atom.GetFormalCharge()

        if not valid_molecule:
            return None

        # Create the output dictionary
        molecule_data = {
            'positions': positions,
            'one_hot': one_hot,
            'charges': output_charges_feature,  # Empty feature as per GEOM
            'atom_mask': atom_mask,
            CONDITIONING_KEY: target_score
        }

        return molecule_data

    except Exception as e:
        logging.error(f"Error processing molecule: {e}")
        return None

def preprocess_data(csv_path, output_pkl_path):
    """
    Reads a CSV, generates conformers, processes molecules, and saves to a pickle file.

    Args:
        csv_path (str): Path to the input CSV file.
        output_pkl_path (str): Path to save the output pickle file.

    Returns:
        list: A list of target scores for the processed molecules. Returns empty list on error.
    """
    print(f"Processing data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []

    if SMILES_COL not in df.columns or TARGET_COL not in df.columns:
        print(f"Error: Required columns '{SMILES_COL}' or '{TARGET_COL}' not found in {csv_path}")
        return []

    processed_molecules = []
    target_scores = []

    # Iterate through the dataframe with progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {os.path.basename(csv_path)}"):
        smiles = row[SMILES_COL]
        target_score = row[TARGET_COL]

        # Check for valid SMILES and score
        if not isinstance(smiles, str) or pd.isna(target_score):
            print(f"Warning: Skipping row {index+2} due to invalid SMILES or missing target score.")
            continue

        # Generate 3D conformer
        mol = generate_conformer(smiles)
        if mol is None:
            continue # Skip if conformer generation failed

        # Extract features
        molecule_data = process_molecule(mol, target_score)
        if molecule_data is not None:
            processed_molecules.append(molecule_data)
            target_scores.append(target_score) # Collect score for stats calculation

    # Save the processed data
    if processed_molecules:
        print(f"Saving {len(processed_molecules)} processed molecules to {output_pkl_path}")
        os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True) # Create output directory if needed
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(processed_molecules, f)
        print("Save complete.")
    else:
        print("No molecules were successfully processed.")

    return target_scores

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting lipid data preprocessing...")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

    # Process Training Data
    train_scores = preprocess_data(TRAIN_CSV_PATH, PROCESSED_TRAIN_PKL)

    # Process Testing Data
    # We don't need test scores for calculating training stats
    _ = preprocess_data(TEST_CSV_PATH, PROCESSED_TEST_PKL)

    # Calculate and Save Statistics (using training data only)
    if train_scores:
        scores_arr = np.array(train_scores).astype(np.float64) # Use float64 for precision
        mean_score = np.mean(scores_arr)
        std_score = np.std(scores_arr)

        # Prevent zero standard deviation
        if std_score < 1e-8:
            print(f"Warning: Standard deviation is very low ({std_score}). Setting to 1.0 to avoid division by zero.")
            std_score = 1.0

        stats = {'mean': mean_score, 'std': std_score}
        print(f"Calculated Stats (Training Data): Mean={mean_score:.4f}, Std={std_score:.4f}")

        print(f"Saving statistics to {STATS_PKL}")
        with open(STATS_PKL, 'wb') as f:
            pickle.dump(stats, f)
        print("Save complete.")
    else:
        print("Could not calculate statistics because no training molecules were processed.")

    print("Preprocessing finished.") 
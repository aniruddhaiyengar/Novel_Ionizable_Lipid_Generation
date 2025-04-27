# preprocess_geom_for_uae3d.py

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import copy
import urllib.request # Added for downloading
import os             # Added for path joining and checking existence

# Suppress RDKit console logs
RDLogger.DisableLog("rdApp.*")

# URLs for the raw GEOM data pickle files
RAW_URL_TRAIN = "https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle"
RAW_URL_VAL = "https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle"
RAW_URL_TEST = "https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle"

EXPECTED_FILES = ["train_data.pickle", "val_data.pickle", "test_data.pickle"]

def download_geom_data(raw_data_dir: str):
    """
    Download the raw GEOM pickle files from the specified URLs into the given directory.

    Args:
        raw_data_dir (str): The directory path where the raw data files will be saved.

    Returns:
        bool: True if all downloads were attempted successfully, False otherwise.
    """
    print(f"Attempting to download raw GEOM data to: {raw_data_dir}")
    urls = {
        "train_data.pickle": RAW_URL_TRAIN,
        "val_data.pickle": RAW_URL_VAL,
        "test_data.pickle": RAW_URL_TEST,
    }
    success = True
    for filename, url in urls.items():
        target_path = os.path.join(raw_data_dir, filename)
        if not os.path.exists(target_path):
            print(f"  Downloading {filename} from {url}...")
            try:
                # Use tqdm for progress bar during download
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                    urllib.request.urlretrieve(url, target_path, reporthook=lambda b, bsize, tsize: t.update(bsize))
                print(f"  Downloaded {filename} successfully.")
            except Exception as e:
                print(f"!!! Error downloading {filename}: {e}")
                success = False
                # Clean up partially downloaded file if error occurs
                if os.path.exists(target_path):
                    os.remove(target_path)
        else:
             print(f"  {filename} already exists. Skipping download.")

    return success


def check_raw_files_exist(raw_data_dir: str) -> bool:
    """Checks if all expected raw pickle files exist in the directory."""
    return all(os.path.exists(os.path.join(raw_data_dir, f)) for f in EXPECTED_FILES)


def preprocess_geom_split(raw_pickle_path: Path):
    """
    Loads a raw GEOM pickle file and processes molecules/conformers.

    Args:
        raw_pickle_path (Path): Path to the input pickle file (e.g., train_data.pickle).

    Returns:
        list: A list of dictionaries, where each dictionary represents a single
              conformer and contains {'rdmol': rdkit_mol_object_with_one_conformer}.
              Returns None if the file cannot be loaded or is invalid.
    """
    if not raw_pickle_path.exists():
        print(f"Error: Raw data file not found at {raw_pickle_path}")
        return None

    print(f"  >> Loading raw data from: {raw_pickle_path}")
    try:
        with open(raw_pickle_path, 'rb') as f:
            raw_data = pickle.load(f)
        if not isinstance(raw_data, list): # Basic sanity check
             print(f"Error: Loaded data from {raw_pickle_path} is not a list as expected.")
             return None
    except Exception as e:
        print(f"Error loading pickle file {raw_pickle_path}: {e}")
        return None

    processed_mols = []
    print(f"  >> Processing molecules and conformers...")
    skipped_molecules = 0
    processed_conformers = 0

    for item in tqdm(raw_data, desc=f"Processing {raw_pickle_path.stem}"):
        try:
            # --- Adjust extraction based on known GEOM pickle structure ---
            # GEOM pickles often store list of tuples: (metadata_dict, list_of_mol_objects)
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], list):
                 all_conformers_mols = item[1]
            else:
                # If structure differs, add more specific checks here or log a warning
                # print(f"Warning: Unexpected data structure for item {item_idx}. Skipping.")
                skipped_molecules += 1
                continue
            # --------------------------------------------------------------

            if not all_conformers_mols:
                skipped_molecules += 1
                continue

            # --- Process up to 5 conformers per molecule ---
            conformers_to_process = all_conformers_mols[:5]

            for conf_mol in conformers_to_process:
                if not isinstance(conf_mol, Chem.Mol):
                    continue

                # Make a deep copy to avoid modifying the original list if loaded elsewhere
                mol_copy = copy.deepcopy(conf_mol)

                # Ensure the mol object itself is valid before proceeding
                try:
                    # Use a copy for sanitization check in case it modifies the mol
                    temp_mol = copy.deepcopy(mol_copy)
                    Chem.SanitizeMol(temp_mol)
                    del temp_mol # free memory
                except Exception as sanitize_e:
                    # print(f"Warning: Skipping invalid conformer molecule: {sanitize_e}")
                    continue

                # Prepare the single-conformer Mol object expected by UAE-3D
                single_conf_mol = Chem.Mol(mol_copy) # Use the structure from the copied mol
                single_conf_mol.RemoveAllConformers()
                try:
                    # Get the single conformer from the input conf_mol
                    conformer = mol_copy.GetConformers()[0]
                    single_conf_mol.AddConformer(conformer, assignId=True)

                    processed_mols.append({'rdmol': single_conf_mol})
                    processed_conformers += 1
                except IndexError:
                    continue # Should not happen if mol_copy had a conformer
                except Exception as e_conf:
                    continue

        except Exception as e_mol:
            skipped_molecules += 1
            continue

    print(f"  >> Processed {processed_conformers} conformers.")
    if skipped_molecules > 0:
         print(f"  >> Skipped {skipped_molecules} molecule entries due to errors or structure.")
    return processed_mols

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess raw GEOM-Drugs pickle files for UAE-3D."
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        required=True,
        help="Directory to store/find raw GEOM-Drugs files (train/val/test_data.pickle)."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for output (e.g., 'data/GEOMDrugs'). The 'preprocessed' subdirectory will be created here."
    )
    args = parser.parse_args()

    raw_dir_path = Path(args.raw_path)
    output_dir = Path(args.output_root)
    preprocessed_out_dir = output_dir / 'preprocessed'

    # --- Download Step ---
    # Create raw directory if it doesn't exist
    raw_dir_path.mkdir(parents=True, exist_ok=True)

    # Check if files exist, download if necessary
    if not check_raw_files_exist(str(raw_dir_path)):
        print("Raw data files not found. Attempting download...")
        download_successful = download_geom_data(str(raw_dir_path))
        if not download_successful:
            print("!!! Aborting preprocessing due to download errors.")
            exit(1)
        # Verify again after download attempt
        if not check_raw_files_exist(str(raw_dir_path)):
             print("!!! Raw data files still missing after download attempt. Please check URLs or download manually.")
             exit(1)
    else:
        print("Raw data files found.")
    # ---------------------


    # --- Preprocessing Step ---
    # Create output directory
    preprocessed_out_dir.mkdir(parents=True, exist_ok=True)

    all_split_data = {}
    total_conformers = 0
    error_occurred = False

    # Process each split
    for split in ["train", "val", "test"]:
        print(f"\n>>> Processing split: {split}")
        pickle_file = raw_dir_path / f"{split}_data.pickle"
        split_data = preprocess_geom_split(pickle_file)

        if split_data is None:
            print(f"!!! Critical Error: Failed to process {split} split. Aborting.")
            error_occurred = True
            break

        all_split_data[split] = split_data
        total_conformers += len(split_data)

    if not error_occurred:
        print(f"\n>>> Combining data from all splits...")
        # Combine data maintaining order: train, then val, then test
        combined_data = all_split_data['train'] + all_split_data['val'] + all_split_data['test']
        print(f"Total conformers combined: {len(combined_data)} (Sum from splits: {total_conformers})")

        # Create the split dictionary with indices relative to the combined list
        len_train = len(all_split_data['train'])
        len_val = len(all_split_data['val'])

        if (len_train + len_val + len(all_split_data['test'])) != len(combined_data):
             print("!!! Warning: Length mismatch after combining splits. Check processing.")
             # Adjust test indices calculation based on actual combined length if mismatch occurs
             len_test_actual = len(combined_data) - len_train - len_val
             if len_test_actual < 0 : len_test_actual = 0 # handle edge case
        else:
             len_test_actual = len(all_split_data['test'])


        train_indices = torch.arange(0, len_train, dtype=torch.long)
        valid_indices = torch.arange(len_train, len_train + len_val, dtype=torch.long)
        test_indices = torch.arange(len_train + len_val, len_train + len_val + len_test_actual, dtype=torch.long)


        split_dict = {
            'train': train_indices,
            'valid': valid_indices,
            'test': test_indices
        }
        print(f"Split dict created: Train={len(train_indices)}, Valid={len(valid_indices)}, Test={len(test_indices)}")
        if len(combined_data) != (len(train_indices) + len(valid_indices) + len(test_indices)):
            print(f"!!! Warning: Total indices ({len(train_indices) + len(valid_indices) + len(test_indices)}) does not match combined data length ({len(combined_data)})")

        # Define output file paths
        data_out_file = preprocessed_out_dir / 'data_geom_drug_1.pt'
        split_out_file = preprocessed_out_dir / 'split_dict_geom_drug_1.pt'

        # Save the combined data and the split dictionary
        print(f"\n>>> Saving processed files...")
        try:
            torch.save(combined_data, data_out_file)
            print(f"  >> Combined data saved to: {data_out_file}")
            torch.save(split_dict, split_out_file)
            print(f"  >> Split dictionary saved to: {split_out_file}")
            print("\nPreprocessing complete.")
        except Exception as e:
            print(f"!!! Error saving output files: {e}")
            error_occurred = True # Mark error if saving fails
    else:
        print("\nPreprocessing aborted due to errors during split processing.")

    if error_occurred:
        exit(1) # Exit with error code if any step failed

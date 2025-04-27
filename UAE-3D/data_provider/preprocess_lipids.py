import argparse
import os
import random
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Suppress RDKit console logs
RDLogger.DisableLog("rdApp.*")

def split_sdf(input_sdf_path: Path, output_dir: Path, train_ratio: float = 0.9, random_seed: int = 42):
    """
    Splits a single SDF file into training and validation sets.

    Args:
        input_sdf_path (Path): Path to the input SDF file (e.g., LMSD/structures.sdf).
        output_dir (Path): Directory to save the output SDF files (e.g., data/Lipids).
        train_ratio (float): Proportion of data to use for the training set.
        random_seed (int): Random seed for reproducibility.
    """
    if not input_sdf_path.exists():
        print(f"Error: Input SDF file not found at {input_sdf_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    train_sdf_path = output_dir / "train_lipids.sdf"
    val_sdf_path = output_dir / "val_lipids.sdf"

    print(f"Reading molecules from: {input_sdf_path}")
    # Use context manager for supplier if possible, or ensure proper closing if needed
    supplier = Chem.SDMolSupplier(str(input_sdf_path), removeHs=False, sanitize=False)
    all_mols = []
    read_count = 0
    error_count = 0
    for mol in supplier:
        if mol is not None:
            # Optional: Add sanitization check here if needed
            try:
                Chem.SanitizeMol(mol)
                all_mols.append(mol)
            except Exception as e:
                # print(f"Warning: Skipping molecule due to sanitization error: {e}")
                error_count += 1
        else:
            error_count +=1
        read_count += 1
    del supplier # Explicitly delete to potentially release file handle

    if not all_mols:
        print(f"Error: No valid molecules could be read or sanitized from {read_count} attempts in the input SDF.")
        return

    print(f"Read {read_count} entries, successfully processed {len(all_mols)} valid molecules. Skipped {error_count} due to read/sanitization errors.")
    print("Splitting data...")

    # Split the list of molecules
    train_mols, val_mols = train_test_split(
        all_mols,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )

    print(f"Train set size: {len(train_mols)}")
    print(f"Validation set size: {len(val_mols)}")

    # Write training set
    print(f"Writing training set to: {train_sdf_path}")
    with Chem.SDWriter(str(train_sdf_path)) as writer:
        for mol in tqdm(train_mols, desc="Writing Train SDF"):
            writer.write(mol)

    # Write validation set
    print(f"Writing validation set to: {val_sdf_path}")
    with Chem.SDWriter(str(val_sdf_path)) as writer:
        for mol in tqdm(val_mols, desc="Writing Val SDF"):
            writer.write(mol)

    print("SDF splitting complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a single Lipid SDF file into train and validation sets."
    )
    parser.add_argument(
        "--input_sdf",
        type=str,
        required=True,
        help="Path to the input SDF file containing all lipid structures (e.g., LMSD/structures.sdf)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Lipids",
        help="Directory to save the output train_lipids.sdf and val_lipids.sdf files."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Proportion of data for the training set (default: 0.9)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)."
    )
    args = parser.parse_args()

    split_sdf(
        input_sdf_path=Path(args.input_sdf),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        random_seed=args.seed
    ) 
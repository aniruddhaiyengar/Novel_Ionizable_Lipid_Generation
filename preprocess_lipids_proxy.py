
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import os
from tqdm import tqdm

def canonicalize_smiles(smiles):
    """
    Canonicalizes a SMILES string.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        str or None: Canonicalized SMILES string, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles) # Convert SMILES to RDKit molecule object
    if mol is not None:
        # If conversion is successful, return the canonical SMILES representation
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        # Print a warning if the SMILES string is invalid
        print(f"Warning: Could not parse SMILES: {smiles}")
        return None # Return None for invalid SMILES

def generate_conformers(mol, num_conformers=1):
    """
    Generates 3D conformers for an RDKit molecule object using ETKDG.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object (must have hydrogens).
        num_conformers (int): Number of conformers to attempt generating.

    Returns:
        rdkit.Chem.Mol or None: Molecule with embedded conformer(s), or None if embedding fails.
    """
    mol = Chem.AddHs(mol) # Add hydrogens, which are necessary for 3D conformer generation
    # Configure the ETKDGv3 algorithm parameters
    params = AllChem.ETKDGv3()
    params.randomSeed = 42 # Set a random seed for reproducibility
    # Attempt to embed multiple conformers into the molecule object
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

    if not conf_ids:
        # If no conformer IDs were generated (embedding failed), print a warning
        print(f"Warning: Could not generate conformer for SMILES: {Chem.MolToSmiles(mol)}")
        return None # Return None if embedding failed

    # Attempt to optimize the generated conformer(s) using the MMFF force field
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol) # Optimize geometry
    except Exception as e:
        # Print a warning if optimization fails, but still return the unoptimized molecule
        print(f"Warning: MMFF optimization failed for SMILES: {Chem.MolToSmiles(mol)}. Error: {e}")
        # Continue without optimization if it fails

    # Return the molecule object, now containing the generated (and possibly optimized) conformer(s)
    return mol

def main(input_csv_path, output_dir, output_prefix):
    """
    Main function to process lipids: read SMILES, canonicalize, generate 3D, save.
    """
    # --- Argument Validation ---
    # Check if the input CSV file exists
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
    # Check if the output directory exists, create it if not
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True) # Create directory including parents if necessary
        print(f"Created output directory: {output_dir}")

    # --- Load Data ---
    print(f"Loading data from: {input_csv_path}")
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv_path)
        # Validate that the necessary columns 'SMILES' and 'TARGET' exist
        if 'SMILES' not in df.columns or 'TARGET' not in df.columns:
             raise ValueError("CSV must contain 'SMILES' and 'TARGET' columns.")
    except Exception as e:
        # Print an error message if loading fails
        print(f"Error loading CSV: {e}")
        return # Exit the function if loading fails

    # --- Processing Loop ---
    processed_data = [] # List to hold summary data for the output CSV
    # Initialize SDF writer to save molecules with 3D conformers
    sdf_output_path = os.path.join(output_dir, f"{output_prefix}_conformers.sdf")
    sdf_writer = Chem.SDWriter(sdf_output_path)

    print("Processing SMILES and generating conformers...")
    # Iterate through each row (molecule) in the DataFrame with a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        original_smiles = row['SMILES'] # Get the original SMILES string
        target_label = row['TARGET']    # Get the associated target label

        # --- Canonicalize SMILES ---
        cano_smiles = canonicalize_smiles(original_smiles) # Get the canonical version
        if cano_smiles is None:
            continue # Skip this molecule if canonicalization failed (invalid SMILES)

        # Create an RDKit molecule object from the canonical SMILES
        mol = Chem.MolFromSmiles(cano_smiles)
        if mol is None: # Double check molecule creation success
             print(f"Warning: Could not create molecule from canonical SMILES: {cano_smiles}")
             continue # Skip if molecule creation fails

        # --- Set Molecule Properties for SDF output ---
        mol.SetProp("_Name", f"lipid_{index}") # Set a unique name for the molecule
        mol.SetProp("Original_SMILES", original_smiles) # Store original SMILES
        mol.SetProp("Canonical_SMILES", cano_smiles)    # Store canonical SMILES
        mol.SetProp("Transfection_Label", str(target_label)) # Store label (as string for SDF compatibility)

        # --- Generate Conformer ---
        # Generate a single 3D conformer for the molecule
        mol_3d = generate_conformers(mol, num_conformers=1)

        # Check if conformer generation was successful and at least one conformer exists
        if mol_3d is not None and mol_3d.GetNumConformers() > 0:
            # --- Save to SDF ---
            sdf_writer.write(mol_3d) # Write the molecule with its 3D conformer to the SDF file

            # --- Prepare data for CSV summary output ---
            conformer = mol_3d.GetConformer(0) # Get the first (and only) conformer
            coords = conformer.GetPositions() # Get the 3D coordinates of atoms
            atom_symbols = [atom.GetSymbol() for atom in mol_3d.GetAtoms()] # Get atom symbols

            # Append summary data to the list
            processed_data.append({
                'ID': f"lipid_{index}", # Unique ID
                'Canonical_SMILES': cano_smiles, # Canonical SMILES
                'Transfection_Label': target_label, # Target label
                'Num_Atoms': mol_3d.GetNumAtoms(), # Number of atoms (including hydrogens)
                # Storing coordinates/atom symbols directly in CSV can make it very large and unwieldy.
                # The primary 3D structure is stored in the SDF file.
                # 'Atom_Symbols': atom_symbols,
                # 'Coordinates': coords.tolist()
            })
        else:
             # Print a message if conformer generation failed for a molecule
             print(f"Skipping molecule {index} due to conformer generation failure.")

    # Close the SDF writer after processing all molecules
    sdf_writer.close()
    print(f"Saved conformers to: {sdf_output_path}")

    # --- Save Processed Info CSV ---
    # Convert the list of dictionaries into a pandas DataFrame
    processed_df = pd.DataFrame(processed_data)
    # Define the path for the output CSV summary file
    output_csv_path = os.path.join(output_dir, f"{output_prefix}_processed.csv")
    # Save the DataFrame to CSV
    processed_df.to_csv(output_csv_path, index=False)
    print(f"Saved processed data summary to: {output_csv_path}")

if __name__ == "__main__":
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Preprocess lipid SMILES: Canonicalize and generate 3D conformers.")
    # Define command-line arguments
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file containing 'SMILES' and 'TARGET' columns.")
    parser.add_argument("--output_dir", type=str, default="processed_lipids", help="Directory to save the output SDF and CSV files.")
    parser.add_argument("--output_prefix", type=str, default="ionizable_lipids", help="Prefix for the output filenames (e.g., 'ionizable_lipids').")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.input_csv, args.output_dir, args.output_prefix) 
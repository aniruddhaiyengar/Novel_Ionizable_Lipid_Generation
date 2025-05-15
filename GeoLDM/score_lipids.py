from rdkit import Chem
import argparse
import os
import logging
from tqdm import tqdm
from translnp import TransLNP
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Score and Rank Generated Ionizable Lipids using TransLNP')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the SDF file containing generated molecules')
    parser.add_argument('--translnp_model_path', type=str, required=True,
                        help='Path to the pre-trained TransLNP model')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top molecules to rank and save')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the ranked molecules (SDF format). If None, will use input_path with _ranked suffix')
    
    args = parser.parse_args()
    
    # Set up output path
    if args.output_path is None:
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_ranked{ext}"
    
    # Load TransLNP model
    try:
        translnp_model = TransLNP.load(args.translnp_model_path)
        print("Successfully loaded TransLNP model")
    except Exception as e:
        logging.error(f"Failed to load TransLNP model: {e}")
        return
    
    # Load molecules
    print(f"Loading molecules from {args.input_path}")
    molecules = []
    with Chem.SDMolSupplier(args.input_path) as suppl:
        for mol in tqdm(suppl, desc="Loading molecules"):
            if mol is not None:
                molecules.append(mol)
    
    print(f"Loaded {len(molecules)} molecules")
    
    # Calculate transfection scores
    print("Calculating transfection scores...")
    scored_molecules = []
    for mol in tqdm(molecules, desc="Scoring molecules"):
        try:
            score = translnp_model.predict(mol)
            scored_molecules.append((mol, score))
        except Exception as e:
            logging.warning(f"Error scoring molecule: {e}")
            continue
    
    # Sort by score
    scored_molecules.sort(key=lambda x: x[1], reverse=True)
    top_molecules = scored_molecules[:args.top_k]
    
    # Save ranked molecules
    print(f"\nSaving top {len(top_molecules)} molecules to {args.output_path}")
    with Chem.SDWriter(args.output_path) as writer:
        for i, (mol, score) in enumerate(top_molecules, 1):
            mol.SetDoubleProp("TransfectionScore", score)
            mol.SetProp("_Name", f"Top_{i}_Score_{score:.4f}")
            writer.write(mol)
    
    # Print detailed information about top molecules
    print("\nTop molecules by transfection score:")
    for i, (mol, score) in enumerate(top_molecules, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   SMILES: {Chem.MolToSmiles(mol)}")
        print(f"   pKa: {mol.GetDoubleProp('pKa'):.2f}")
        print(f"   logP: {mol.GetDoubleProp('logP'):.2f}")
        print(f"   MW: {mol.GetDoubleProp('MW'):.1f}")
    
    # Print score statistics
    scores = [score for _, score in scored_molecules]
    print("\nScore Statistics:")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    print(f"Min: {min(scores):.4f}")
    print(f"Max: {max(scores):.4f}")
    print(f"Median: {np.median(scores):.4f}")

if __name__ == "__main__":
    main() 
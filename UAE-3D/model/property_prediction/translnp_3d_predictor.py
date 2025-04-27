# property_prediction/translnp_3d_predictor.py
import torch
import torch.nn as nn # Required for potential model parts
import os
import sys
import numpy as np
import copy
from pathlib import Path
from rdkit import Chem
from scipy.spatial import distance_matrix
import argparse # Added import

# --- Add TransLNP directory to Python path ---
# Go up one level from 'property_prediction' to project root, then into 'TransLNP'
TRANS_LNP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransLNP'))
if TRANS_LNP_DIR not in sys.path:
    sys.path.insert(0, TRANS_LNP_DIR)
# Add TransLNP/models directory specifically if needed for nested imports
TRANS_LNP_MODELS_DIR = os.path.join(TRANS_LNP_DIR, 'models')
if TRANS_LNP_MODELS_DIR not in sys.path:
     sys.path.insert(0, TRANS_LNP_MODELS_DIR)
# Add unicore path if it's bundled within TransLNP or elsewhere specific
# Example: UNICORE_DIR = os.path.join(TRANS_LNP_DIR, 'unicore_dep')
# if UNICORE_DIR not in sys.path: sys.path.insert(0, UNICORE_DIR)
# ---------------------------------------------

# --- Imports from TransLNP and UniCore ---
try:
    from models.unimol import UniMolModel, molecule_architecture # Assumed model class and default args
    from unicore.data import Dictionary # Required for atom dictionary
except ImportError as e:
    print(f"Error importing TransLNP/UniCore modules: {e}")
    print(f"Ensure TransLNP directory is structured correctly and in sys.path.")
    print(f"Attempted TransLNP path: {TRANS_LNP_DIR}")
    print("Ensure UniCore is installed or accessible.")
    raise
# -----------------------------------------

# --- Global Constants ---
# Determine weight directory relative to TransLNP base dir
WEIGHT_DIR = os.path.join(TRANS_LNP_DIR, 'weights')
# Define the specific dictionary file expected for 'molecule_all_h'
# Double-check this name in TransLNP/config.py MODEL_CONFIG if issues arise
DEFAULT_DICT_NAME_ALL_H = 'mol.dict.txt'
DEFAULT_MAX_ATOMS = 256 # Default from coords2unimol, adjust if needed
# Mapping from atomic number to symbol (includes H)
ATOMIC_NUM_TO_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
    # Add others from the dictionary if needed
}
# ---------------------


class TransLNP3DPredictor:
    """
    Predicts TransLNP score directly from 3D coordinates and atom types.
    Assumes input INCLUDES Hydrogens, matching the 'all_h' model.
    """
    def __init__(self,
                 model_weights_path='property_prediction/mol_pre_all_h_220816.pt', # <-- Updated default
                 dictionary_name=DEFAULT_DICT_NAME_ALL_H, # <-- Updated default
                 max_atoms=DEFAULT_MAX_ATOMS,
                 device='cpu'):

        self.device = torch.device(device)
        self.model_weights_path = Path(model_weights_path)
        self.dictionary_path = Path(WEIGHT_DIR) / dictionary_name
        self.max_atoms = max_atoms

        if not self.model_weights_path.exists():
             raise FileNotFoundError(f"Model weights not found: {self.model_weights_path}")
        if not self.dictionary_path.exists():
             raise FileNotFoundError(f"Atom dictionary not found: {self.dictionary_path}")

        # --- Load Atom Dictionary ---
        print(f"Loading atom dictionary from: {self.dictionary_path}")
        try:
            self.dictionary = Dictionary.load(str(self.dictionary_path))
            self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
            self.padding_idx = self.dictionary.pad()
            self.bos_idx = self.dictionary.bos()
            self.eos_idx = self.dictionary.eos()
            print(f"Dictionary loaded: size={len(self.dictionary)}, pad={self.padding_idx}, bos={self.bos_idx}, eos={self.eos_idx}")
        except Exception as e:
            print(f"!!! Error loading dictionary: {e}")
            raise
        # --------------------------

        # --- Load Model ---
        print(f"Loading TransLNP model weights from: {self.model_weights_path}")
        try:
            # 1. Instantiate the UniMolModel
            print("Instantiating UniMolModel architecture...")
            parser = argparse.ArgumentParser()
            mol_args = molecule_architecture(parser) # Get default args object
            model_args_dict = {
                 'output_dim': 1,
                 'data_type': 'molecule',
                 'remove_hs': False, # <<< Set to False for 'all_h' model
                 'encoder_embed_dim': mol_args.encoder_embed_dim,
            }
            self.model = UniMolModel(**model_args_dict)

            # 2. Load the State Dictionary
            state_dict = torch.load(self.model_weights_path, map_location='cpu')
            model_state_dict = state_dict.get('model', state_dict)

            if all(k.startswith('module.') for k in model_state_dict.keys()):
                 print("Detected 'module.' prefix in checkpoint keys, removing it.")
                 model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}

            missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                 print(f"Warning: Missing keys found during state_dict loading: {missing_keys}")
            if unexpected_keys:
                 print(f"Warning: Unexpected keys found during state_dict loading: {unexpected_keys}")

            print("Weights loaded successfully.")
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode

        except Exception as e:
            import traceback
            print(f"!!! Error loading model: {e}")
            print(traceback.format_exc())
            raise
        # ------------------

    def _prepare_input_tensors(self, atoms_symbols: list, coordinates: np.ndarray):
        """
        Converts atom symbols and coordinates (including H) to the tensor dictionary
        required by UniMolModel.forward, mimicking coords2unimol.

        Args:
            atoms_symbols (list[str]): List of atom symbols including H (e.g., ['C', 'H', 'O']).
            coordinates (np.ndarray): Numpy array of coordinates including H [num_atoms, 3].

        Returns:
            dict: Dictionary of tensors for the model, with batch dimension added.
                  Returns None on error.
        """
        try:
            n_atoms = len(atoms_symbols)
            if n_atoms == 0:
                print("Warning: Empty molecule provided.")
                return None

            # 1. Cropping (if needed)
            if n_atoms > self.max_atoms:
                print(f"Warning: Molecule ({n_atoms} atoms) exceeds max_atoms ({self.max_atoms}). Cropping.")
                indices = np.random.choice(n_atoms, self.max_atoms, replace=False)
                atoms_symbols = [atoms_symbols[i] for i in indices]
                coordinates = coordinates[indices]
                n_atoms = self.max_atoms

            # 2. Atom Tokens (using the dictionary that includes 'H')
            tokens = [self.dictionary.bos()] + [self.dictionary.index(sym) for sym in atoms_symbols] + [self.dictionary.eos()]
            src_tokens = torch.tensor(tokens, dtype=torch.long)

            # 3. Coordinates
            coords_normalized = coordinates - coordinates.mean(axis=0)
            src_coord = np.concatenate([np.zeros((1, 3)), coords_normalized, np.zeros((1, 3))], axis=0)
            src_coord = torch.tensor(src_coord, dtype=torch.float)

            # 4. Distance Matrix
            dist_matrix = distance_matrix(src_coord.numpy(), src_coord.numpy())
            src_distance = torch.tensor(dist_matrix, dtype=torch.float)

            # 5. Edge Types
            tok_row = src_tokens.view(-1, 1)
            tok_col = src_tokens.view(1, -1)
            src_edge_type = tok_row * len(self.dictionary) + tok_col

            # 6. Add Batch Dimension and Move to Device
            model_input = {
                'src_tokens': src_tokens.unsqueeze(0).to(self.device),
                'src_distance': src_distance.unsqueeze(0).to(self.device),
                'src_coord': src_coord.unsqueeze(0).to(self.device),
                'src_edge_type': src_edge_type.unsqueeze(0).to(self.device),
            }
            return model_input

        except Exception as e:
            import traceback
            print(f"Error during input tensor preparation: {e}")
            print(traceback.format_exc())
            return None

    # Removed _filter_hydrogens method

    @torch.no_grad()
    def predict(self, atomic_nums: torch.Tensor, coordinates: torch.Tensor):
        """
        Predicts the TransLNP score directly from atom numbers and coordinates.
        Expects input to INCLUDE hydrogens.

        Args:
            atomic_nums (torch.Tensor): Tensor of atomic numbers [num_atoms_with_H].
            coordinates (torch.Tensor): Tensor of atom coordinates [num_atoms_with_H, 3].

        Returns:
            float: Predicted score. Returns np.nan on error or invalid input.
        """
        if atomic_nums is None or coordinates is None or atomic_nums.numel() == 0 or coordinates.numel() == 0:
             print("Warning: Received empty input for prediction.")
             return np.nan
        if atomic_nums.shape[0] != coordinates.shape[0]:
             print(f"Warning: Mismatched shapes for atomic_nums ({atomic_nums.shape}) and coordinates ({coordinates.shape}).")
             return np.nan

        # 1. Ensure input tensors are on CPU for processing
        atomic_nums_cpu = atomic_nums.cpu()
        coordinates_cpu = coordinates.cpu()

        # 2. Convert atomic numbers to symbols (INCLUDING H)
        try:
            atom_symbols = [ATOMIC_NUM_TO_SYMBOL[num.item()] for num in atomic_nums_cpu]
        except KeyError as e:
            print(f"Error: Encountered unknown atomic number {e} not in ATOMIC_NUM_TO_SYMBOL map.")
            return np.nan

        # 3. Prepare input tensors (NO hydrogen removal)
        model_input = self._prepare_input_tensors(atom_symbols, coordinates_cpu.numpy())

        if model_input is None:
            print("Error: Failed to prepare input tensors for the model.")
            return np.nan # Preprocessing failed

        # 4. Run model inference
        try:
            output_logits, _ = self.model(**model_input)
            score = output_logits[0, 0].item()
            return score
        except Exception as e:
            import traceback
            print(f"Error during model inference: {e}")
            print(traceback.format_exc())
            return np.nan # Return NaN on error

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("Testing TransLNP3DPredictor with all_h model...")

    # --- Create test input data Ionizable lipid smiles = "CCCCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCCC=C)NCCCN(C)C" ---
    dummy_atomic_nums_h = torch.tensor([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 8, 6, 6, 6, 6, 6, 6, 8,
        6, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long) 
    
    # Approximate tetrahedral coordinates
    dummy_coordinates_h = torch.tensor([[ 1.3379e+01,  7.0180e+00, -8.0024e-01],
        [ 1.2553e+01,  5.9040e+00, -1.7181e-01],
        [ 1.2846e+01,  4.5535e+00, -8.2914e-01],
        [ 1.2131e+01,  3.3820e+00, -1.5055e-01],
        [ 1.0607e+01,  3.4018e+00, -3.3525e-01],
        [ 9.9247e+00,  2.1609e+00,  2.5556e-01],
        [ 9.8652e+00,  2.1884e+00,  1.7829e+00],
        [ 9.3982e+00,  8.7647e-01,  2.4187e+00],
        [ 7.9750e+00,  4.3950e-01,  2.0525e+00],
        [ 7.9309e+00, -5.9141e-01,  9.1794e-01],
        [ 6.5499e+00, -1.2315e+00,  7.4580e-01],
        [ 5.5185e+00, -2.8959e-01,  1.1463e-01],
        [ 4.1380e+00, -9.4811e-01, -4.9751e-03],
        [ 3.3411e+00, -8.7152e-01,  1.3027e+00],
        [ 2.1936e+00, -1.7576e+00,  1.3219e+00],
        [ 1.1025e+00, -1.5761e+00,  4.8507e-01],
        [ 1.0490e+00, -7.0313e-01, -3.7578e-01],
        [-3.3606e-02, -2.6028e+00,  7.4206e-01],
        [-1.2881e+00, -2.3618e+00, -1.1900e-01],
        [-2.0575e+00, -1.1032e+00,  3.0468e-01],
        [-3.2685e+00, -8.4995e-01, -6.0037e-01],
        [-3.9322e+00,  5.0551e-01, -3.2777e-01],
        [-4.7185e+00,  5.5654e-01,  9.7905e-01],
        [-5.9550e+00, -1.4888e-01,  8.0541e-01],
        [-6.7849e+00, -7.6633e-02,  1.8821e+00],
        [-6.4778e+00,  3.6825e-01,  2.9787e+00],
        [-8.1545e+00, -6.2260e-01,  1.5547e+00],
        [-8.7557e+00, -4.9344e-02,  2.6898e-01],
        [-8.8701e+00,  1.4810e+00,  2.8783e-01],
        [-9.3848e+00,  2.0550e+00, -1.0356e+00],
        [-1.0869e+01,  1.8100e+00, -1.3271e+00],
        [-1.1794e+01,  2.5555e+00, -3.5941e-01],
        [-1.3256e+01,  2.4685e+00, -8.1459e-01],
        [-1.4238e+01,  3.1497e+00,  1.4380e-01],
        [-1.4052e+01,  4.6395e+00,  2.1671e-01],
        [-1.4937e+01,  5.5334e+00, -2.4054e-01],
        [ 5.0127e-01, -3.9865e+00,  5.9695e-01],
        [ 9.1706e-01, -4.3291e+00, -7.6938e-01],
        [ 3.9169e-02, -5.4381e+00, -1.3617e+00],
        [ 1.8755e-01, -6.7699e+00, -6.0983e-01],
        [-5.8233e-01, -7.9128e+00, -1.1252e+00],
        [-1.8820e-01, -8.3039e+00, -2.4750e+00],
        [-2.0243e+00, -7.7006e+00, -1.0355e+00],
        [ 1.4454e+01,  6.8276e+00, -6.8607e-01],
        [ 1.3158e+01,  7.1179e+00, -1.8700e+00],
        [ 1.3152e+01,  7.9737e+00, -3.1505e-01],
        [ 1.2775e+01,  5.8536e+00,  9.0222e-01],
        [ 1.1492e+01,  6.1557e+00, -2.7924e-01],
        [ 1.2575e+01,  4.5943e+00, -1.8906e+00],
        [ 1.3927e+01,  4.3638e+00, -7.8566e-01],
        [ 1.2526e+01,  2.4498e+00, -5.7530e-01],
        [ 1.2389e+01,  3.3764e+00,  9.1417e-01],
        [ 1.0382e+01,  3.4424e+00, -1.4091e+00],
        [ 1.0175e+01,  4.3023e+00,  1.1576e-01],
        [ 8.8988e+00,  2.1230e+00, -1.3514e-01],
        [ 1.0439e+01,  1.2558e+00, -8.9975e-02],
        [ 9.2068e+00,  3.0053e+00,  2.1045e+00],
        [ 1.0855e+01,  2.4139e+00,  2.1951e+00],
        [ 9.4328e+00,  1.0236e+00,  3.5074e+00],
        [ 1.0119e+01,  7.9424e-02,  2.2017e+00],
        [ 7.3564e+00,  1.3113e+00,  1.8129e+00],
        [ 7.5303e+00, -2.3175e-02,  2.9438e+00],
        [ 8.2504e+00, -1.4322e-01, -2.8484e-02],
        [ 8.6449e+00, -1.3954e+00,  1.1413e+00],
        [ 6.6587e+00, -2.1118e+00,  9.8727e-02],
        [ 6.1973e+00, -1.6025e+00,  1.7155e+00],
        [ 5.4370e+00,  6.4434e-01,  6.8441e-01],
        [ 5.8660e+00, -2.3693e-02, -8.9123e-01],
        [ 3.5771e+00, -4.3084e-01, -7.9393e-01],
        [ 4.2449e+00, -1.9922e+00, -3.3087e-01],
        [ 3.9618e+00, -1.1672e+00,  2.1544e+00],
        [ 2.9888e+00,  1.5111e-01,  1.4754e+00],
        [ 2.0427e+00, -2.3034e+00,  2.1603e+00],
        [-3.2430e-01, -2.4907e+00,  1.7941e+00],
        [-1.0173e+00, -2.2749e+00, -1.1785e+00],
        [-1.9656e+00, -3.2212e+00, -2.9152e-02],
        [-1.3976e+00, -2.2831e-01,  2.7324e-01],
        [-2.3850e+00, -1.2172e+00,  1.3434e+00],
        [-2.9369e+00, -8.5889e-01, -1.6470e+00],
        [-3.9962e+00, -1.6622e+00, -4.9098e-01],
        [-4.6130e+00,  7.2249e-01, -1.1602e+00],
        [-3.1659e+00,  1.2887e+00, -3.2274e-01],
        [-4.9449e+00,  1.6055e+00,  1.2065e+00],
        [-4.1511e+00,  1.2766e-01,  1.8113e+00],
        [-8.0768e+00, -1.7150e+00,  1.4886e+00],
        [-8.8183e+00, -3.9074e-01,  2.3974e+00],
        [-8.1462e+00, -3.5867e-01, -5.8981e-01],
        [-9.7486e+00, -4.9197e-01,  1.2800e-01],
        [-7.8777e+00,  1.9133e+00,  4.6102e-01],
        [-9.4961e+00,  1.8021e+00,  1.1286e+00],
        [-8.7918e+00,  1.6351e+00, -1.8570e+00],
        [-9.1967e+00,  3.1361e+00, -1.0383e+00],
        [-1.1093e+01,  7.3636e-01, -1.3072e+00],
        [-1.1069e+01,  2.1558e+00, -2.3501e+00],
        [-1.1703e+01,  2.1247e+00,  6.4391e-01],
        [-1.1482e+01,  3.6055e+00, -2.9849e-01],
        [-1.3538e+01,  1.4123e+00, -8.9418e-01],
        [-1.3360e+01,  2.9057e+00, -1.8163e+00],
        [-1.4130e+01,  2.7346e+00,  1.1538e+00],
        [-1.5261e+01,  2.9175e+00, -1.7967e-01],
        [-1.3141e+01,  5.0064e+00,  6.8753e-01],
        [-1.4744e+01,  6.5992e+00, -1.5623e-01],
        [-1.5866e+01,  5.2286e+00, -7.1440e-01],
        [ 1.3216e+00, -4.0714e+00,  1.1992e+00],
        [ 9.1542e-01, -3.4653e+00, -1.4427e+00],
        [ 1.9569e+00, -4.6754e+00, -7.3954e-01],
        [-1.0066e+00, -5.1111e+00, -1.3663e+00],
        [ 3.3574e-01, -5.5695e+00, -2.4076e+00],
        [-8.0721e-02, -6.6304e+00,  4.4438e-01],
        [ 1.2475e+00, -7.0495e+00, -5.9147e-01],
        [ 8.8864e-01, -8.5051e+00, -2.5238e+00],
        [-6.9186e-01, -9.2353e+00, -2.7611e+00],
        [-4.3643e-01, -7.5482e+00, -3.2278e+00],
        [-2.3802e+00, -6.9226e+00, -1.7168e+00],
        [-2.3129e+00, -7.4322e+00, -1.4727e-02],
        [-2.5572e+00, -8.6265e+00, -1.2744e+00]], dtype=torch.float)
    # -------------------------------------------------------

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        predictor = TransLNP3DPredictor(
             # Ensure this path points to the all_h model file
            model_weights_path='../property_prediction/mol_pre_all_h_220816.pt',
            device=device
        )

        predicted_score = predictor.predict(dummy_atomic_nums_h, dummy_coordinates_h)

        if predicted_score is not None and not np.isnan(predicted_score):
            print(f"\nPredicted Score (all_h): {predicted_score:.4f}")
        else:
            print("\nPrediction failed.")

    except FileNotFoundError as e:
         print(f"\nFile Not Found Error: {e}")
         print("Ensure model weights ('mol_pre_all_h_220816.pt') and dictionary ('molecule_all_h_dict.txt') exist.")
    except ImportError as e:
         print(f"\nImport Error: {e}")
         print("Make sure TransLNP modules and UniCore are correctly installed/accessible.")
    except Exception as e:
         print(f"\nAn unexpected error occurred during testing: {e}")
         import traceback
         print(traceback.format_exc())

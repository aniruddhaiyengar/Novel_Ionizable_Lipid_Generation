import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import os
import logging

class LipidDataset(Dataset):
    """PyTorch Dataset for loading processed lipid molecule data."""
    def __init__(self, data_list):
        """
        Args:
            data_list (list): A list of dictionaries, where each dictionary
                              represents a molecule and contains keys like
                              'positions', 'one_hot', 'charges', 'atom_mask',
                              and optionally a conditioning key (e.g., 'transfection_score').
        """
        self.data_list = data_list
        
        # Create data attribute with num_atoms information and property values
        self.data = {
            'num_atoms': torch.tensor([len(item['positions']) for item in data_list])
        }
        
        # Add property values if they exist in the data
        if data_list and 'transfection_score' in data_list[0]:
            self.data['transfection_score'] = torch.tensor([item['transfection_score'] for item in data_list])

    def __len__(self):
        """Returns the number of molecules in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Returns the data dictionary for the molecule at the given index."""
        data = self.data_list[idx]
        
        # Convert positions to float tensor and add batch dimension
        positions = torch.from_numpy(data["positions"]).float()
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)  # Add batch dimension
        
        # Convert one_hot to float tensor and add batch dimension if needed
        one_hot = torch.from_numpy(data["one_hot"]).float()
        if one_hot.dim() == 2:
            one_hot = one_hot.unsqueeze(0)  # Add batch dimension
        
        # Convert charges to float tensor with proper shape
        charges = torch.from_numpy(data["charges"]).float() if "charges" in data else torch.zeros_like(one_hot[:, :, :1])
        
        # Convert atom_mask to float tensor with explicit values and proper shape
        atom_mask = torch.from_numpy(data["atom_mask"]).float()
        if atom_mask.dim() == 2:
            atom_mask = atom_mask.unsqueeze(0)  # Add batch dimension
        atom_mask = (atom_mask > 0.5).float()
        
        # Ensure edge mask is also float with proper shape
        if "edge_mask" in data:
            edge_mask = torch.from_numpy(data["edge_mask"]).float()
            if edge_mask.dim() == 2:
                edge_mask = edge_mask.unsqueeze(0)
            edge_mask = (edge_mask > 0.5).float()
        else:
            # Create default edge mask based on atom_mask if not provided
            n_nodes = atom_mask.size(1)  # Get number of nodes
            # Create adjacency matrix: [batch_size, n_nodes, n_nodes]
            edge_mask = torch.matmul(
                atom_mask.transpose(1, 2),  # [batch_size, 1, n_nodes]
                atom_mask  # [batch_size, n_nodes, 1]
            )  # Results in [batch_size, n_nodes, n_nodes]
            
            # Remove self-loops
            diag_mask = ~torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
            edge_mask = edge_mask * diag_mask
        
        # Validate shapes and types
        assert positions.dim() == 3, f"Positions should be 3D tensor, got shape {positions.shape}"
        assert one_hot.dim() == 3, f"One-hot should be 3D tensor, got shape {one_hot.shape}"
        assert atom_mask.dim() == 3, f"Atom mask should be 3D tensor, got shape {atom_mask.shape}"
        assert edge_mask.dim() == 3, f"Edge mask should be 3D tensor, got shape {edge_mask.shape}"
        assert edge_mask.shape[1] == edge_mask.shape[2], f"Edge mask should be square, got shape {edge_mask.shape}"
        
        # Create return dictionary
        return_dict = {
            "positions": positions,
            "one_hot": one_hot,
            "charges": charges,
            "atom_mask": atom_mask,
            "edge_mask": edge_mask
        }
        
        # Add transfection score if it exists
        if "transfection_score" in data:
            return_dict["transfection_score"] = torch.tensor(data["transfection_score"], dtype=torch.float32)
        
        return return_dict

def lipid_collate_fn(batch):
    """
    Custom collate function to handle batching of molecules with varying sizes.
    Pads features to the maximum size in the batch and creates edge masks.

    Args:
        batch (list): A list of data dictionaries, each returned by LipidDataset.__getitem__.

    Returns:
        dict: A dictionary containing batched and padded tensors.
    """
    if not batch:
        raise ValueError("Empty batch received")
        
    # Find the maximum number of atoms in the batch
    max_atoms = max(item['atom_mask'].shape[1] for item in batch)

    # Initialize lists to hold padded data
    batch_positions = []
    batch_one_hot = []
    batch_charges = []
    batch_atom_mask = []
    batch_transfection_scores = []

    # Pad each molecule's data
    for item in batch:
        num_atoms = item['atom_mask'].shape[1]
        padding_size = max_atoms - num_atoms

        # Convert to numpy arrays if they aren't already
        positions = item['positions'].numpy() if torch.is_tensor(item['positions']) else item['positions']
        one_hot = item['one_hot'].numpy() if torch.is_tensor(item['one_hot']) else item['one_hot']
        charges = item['charges'].numpy() if torch.is_tensor(item['charges']) else item['charges']
        atom_mask = item['atom_mask'].numpy() if torch.is_tensor(item['atom_mask']) else item['atom_mask']

        # Remove batch dimension for padding if it exists
        if positions.ndim == 3 and positions.shape[0] == 1:
            positions = positions.squeeze(0)
        if one_hot.ndim == 3 and one_hot.shape[0] == 1:
            one_hot = one_hot.squeeze(0)
        if charges.ndim == 3 and charges.shape[0] == 1:
            charges = charges.squeeze(0)
        if atom_mask.ndim == 3 and atom_mask.shape[0] == 1:
            atom_mask = atom_mask.squeeze(0)

        # Pad arrays
        padded_positions = np.pad(positions, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        padded_one_hot = np.pad(one_hot, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        padded_charges = np.pad(charges, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        padded_atom_mask = np.pad(atom_mask, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)

        # Add batch dimension back
        padded_positions = np.expand_dims(padded_positions, axis=0)
        padded_one_hot = np.expand_dims(padded_one_hot, axis=0)
        padded_charges = np.expand_dims(padded_charges, axis=0)
        padded_atom_mask = np.expand_dims(padded_atom_mask, axis=0)

        batch_positions.append(padded_positions)
        batch_one_hot.append(padded_one_hot)
        batch_charges.append(padded_charges)
        batch_atom_mask.append(padded_atom_mask)

        # Handle transfection score
        if 'transfection_score' in item:
            score = item['transfection_score']
            if torch.is_tensor(score):
                score = score.item()
            batch_transfection_scores.append(score)

    # Convert to tensors and ensure proper types
    batch_dict = {
        'positions': torch.tensor(np.concatenate(batch_positions, axis=0), dtype=torch.float32),
        'one_hot': torch.tensor(np.concatenate(batch_one_hot, axis=0), dtype=torch.float32),
        'charges': torch.tensor(np.concatenate(batch_charges, axis=0), dtype=torch.float32),
        'atom_mask': torch.tensor(np.concatenate(batch_atom_mask, axis=0), dtype=torch.float32)
    }

    # Add transfection scores if present
    if batch_transfection_scores:
        batch_dict['transfection_score'] = torch.tensor(batch_transfection_scores, dtype=torch.float32).view(-1, 1)

    # Create edge mask
    atom_mask_squeezed = batch_dict['atom_mask'].squeeze(-1)  # Remove last dimension if present
    edge_mask = atom_mask_squeezed.unsqueeze(1) * atom_mask_squeezed.unsqueeze(2)
    
    # Remove self-loops
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=edge_mask.device).unsqueeze(0)
    edge_mask = edge_mask * diag_mask
    
    batch_dict['edge_mask'] = edge_mask

    # Validate shapes and types
    assert batch_dict['positions'].dim() == 3, f"Positions should be 3D tensor, got shape {batch_dict['positions'].shape}"
    assert batch_dict['one_hot'].dim() == 3, f"One-hot should be 3D tensor, got shape {batch_dict['one_hot'].shape}"
    assert batch_dict['atom_mask'].dim() == 3, f"Atom mask should be 3D tensor, got shape {batch_dict['atom_mask'].shape}"
    assert batch_dict['edge_mask'].dim() == 3, f"Edge mask should be 3D tensor, got shape {batch_dict['edge_mask'].shape}"

    return batch_dict

def get_dataloaders(data_path, batch_size, num_workers=0, seed=42, val_split_ratio=0.1, lipid_stats_path=None, is_stage2_data=False):
    """
    Loads processed lipid data, creates Datasets and DataLoaders for train and validation.

    Tries to find a corresponding validation file (e.g., '..._val_lipids.pkl' if
    'data_path' is '..._train_lipids.pkl'). If not found, it splits the data
    loaded from 'data_path' according to 'val_split_ratio'.

    Args:
        data_path (str): Path to the processed data pickle file (e.g., training or unlabeled).
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for DataLoader. Defaults to 0.
        seed (int): Random seed for train/validation split. Defaults to 42.
        val_split_ratio (float): Fraction of data to use for validation if no validation file is found.
                                 Defaults to 0.1 (10%).
        lipid_stats_path (str, optional): Path to lipid_stats.pkl for property normalization.
        is_stage2_data (bool): Whether this is stage 2 data with property conditioning.

    Returns:
        dict: A dictionary containing 'train' and 'val' DataLoaders.
              Returns None if data loading fails.
    """
    print(f"Loading data from: {data_path}")
    try:
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        if not isinstance(all_data, list):
            raise TypeError(f"Expected a list in {data_path}, but got {type(all_data)}")
        if not all_data:
             raise ValueError(f"Pickle file {data_path} is empty.")
        print(f"Successfully loaded {len(all_data)} molecules from {data_path}.")

        # Load lipid stats if provided
        if lipid_stats_path and is_stage2_data:
            try:
                with open(lipid_stats_path, 'rb') as f:
                    lipid_stats = pickle.load(f)
                print(f"Successfully loaded lipid stats from {lipid_stats_path}")
            except Exception as e:
                print(f"Warning: Could not load lipid stats from {lipid_stats_path}: {e}")
                lipid_stats = None
        else:
            lipid_stats = None

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return None

    # Try to find a corresponding validation file
    val_data_path = None
    if '_train_' in data_path:
        potential_val_path = data_path.replace('_train_', '_val_')
        if os.path.exists(potential_val_path):
            val_data_path = potential_val_path
        else:
            # Check for test file as fallback val file (common naming)
            potential_test_path = data_path.replace('_train_', '_test_')
            if os.path.exists(potential_test_path):
                print(f"Note: Found test file '{potential_test_path}', using it for validation.")
                val_data_path = potential_test_path

    train_data = []
    val_data = []

    if val_data_path:
        print(f"Loading validation data from: {val_data_path}")
        try:
            with open(val_data_path, 'rb') as f:
                val_data = pickle.load(f)
            if not isinstance(val_data, list):
                raise TypeError(f"Expected a list in {val_data_path}, but got {type(val_data)}")
            print(f"Successfully loaded {len(val_data)} validation molecules.")
            train_data = all_data # Use all data from data_path as training data
        except FileNotFoundError:
            print(f"Error: Validation data file not found at {val_data_path}. Splitting data from {data_path}.")
            val_data_path = None # Fall back to splitting
        except Exception as e:
            print(f"Error loading validation data from {val_data_path}: {e}. Splitting data from {data_path}.")
            val_data_path = None # Fall back to splitting

    # If validation data wasn't loaded from a separate file, split the loaded data
    if not val_data_path:
        print(f"Splitting data from {data_path} into train/validation ({1-val_split_ratio:.0%}/{val_split_ratio:.0%})...")
        num_molecules = len(all_data)
        num_val = int(num_molecules * val_split_ratio)
        num_train = num_molecules - num_val

        if num_train <= 0 or num_val <= 0:
            print("Warning: Dataset too small to create a non-empty train/validation split. Using all data for training.")
            train_data = all_data
            # Create a tiny validation set with one item if possible, otherwise empty
            val_data = all_data[:1] if all_data else []
        else:
            # Perform random split
            generator = torch.Generator().manual_seed(seed)
            train_data, val_data = random_split(all_data, [num_train, num_val], generator=generator)
            # Convert map-style datasets back to lists for LipidDataset
            train_data = list(train_data)
            val_data = list(val_data)
        print(f"Split complete: {len(train_data)} train, {len(val_data)} validation molecules.")


    # Create Dataset instances
    train_dataset = LipidDataset(train_data)
    val_dataset = LipidDataset(val_data)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=lipid_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=lipid_collate_fn, drop_last=False)

    return {'train': train_loader, 'val': val_loader}


# Example Usage (optional)
if __name__ == '__main__':
    # Create dummy data for testing
    dummy_data_list = [
        {
            'positions': np.random.rand(10, 3).astype(np.float32),
            'one_hot': np.eye(5)[np.random.choice(5, 10)].astype(np.float32),
            'charges': np.random.rand(10, 1).astype(np.float32),
            'atom_mask': np.ones((10, 1)).astype(np.float32),
            'transfection_score': 0.75
        },
        {
            'positions': np.random.rand(15, 3).astype(np.float32),
            'one_hot': np.eye(5)[np.random.choice(5, 15)].astype(np.float32),
            'charges': np.random.rand(15, 1).astype(np.float32),
            'atom_mask': np.ones((15, 1)).astype(np.float32),
            'transfection_score': 0.85
        },
         {
            'positions': np.random.rand(12, 3).astype(np.float32),
            'one_hot': np.eye(5)[np.random.choice(5, 12)].astype(np.float32),
            'charges': np.random.rand(12, 1).astype(np.float32),
            'atom_mask': np.ones((12, 1)).astype(np.float32),
            'transfection_score': 0.65
        }
    ]

    # Save dummy data to a temp file
    dummy_train_path = "data/dummy_train_lipids.pkl"
    dummy_val_path = "data/dummy_val_lipids.pkl"
    os.makedirs("data", exist_ok=True)
    with open(dummy_train_path, 'wb') as f:
        pickle.dump(dummy_data_list[:2], f)
    with open(dummy_val_path, 'wb') as f:
        pickle.dump(dummy_data_list[2:], f)


    print("\n--- Testing get_dataloaders (with separate val file) ---")
    dataloaders = get_dataloaders(dummy_train_path, batch_size=2)

    if dataloaders:
        print(f"Train loader: {len(dataloaders['train'])} batches")
        print(f"Val loader: {len(dataloaders['val'])} batches")

        # Inspect first batch
        print("\nInspecting first training batch:")
        first_batch = next(iter(dataloaders['train']))
        for key, value in first_batch.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        print("\nInspecting first validation batch:")
        first_val_batch = next(iter(dataloaders['val']))
        for key, value in first_val_batch.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Clean up dummy files
    os.remove(dummy_train_path)
    os.remove(dummy_val_path)

    # --- Test splitting ---
    dummy_single_file_path = "data/dummy_single_file.pkl"
    with open(dummy_single_file_path, 'wb') as f:
        pickle.dump(dummy_data_list * 10, f) # Make it larger for splitting

    print("\n--- Testing get_dataloaders (with splitting) ---")
    dataloaders_split = get_dataloaders(dummy_single_file_path, batch_size=4, val_split_ratio=0.2)
    if dataloaders_split:
        print(f"Train loader (split): {len(dataloaders_split['train'])} batches, ~{len(dataloaders_split['train'].dataset)} samples")
        print(f"Val loader (split): {len(dataloaders_split['val'])} batches, ~{len(dataloaders_split['val'].dataset)} samples")

    os.remove(dummy_single_file_path) 
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import os

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

    def __len__(self):
        """Returns the number of molecules in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Returns the data dictionary for the molecule at the given index."""
        # Return a copy to avoid unintended modifications
        return self.data_list[idx].copy()

def lipid_collate_fn(batch):
    """
    Custom collate function to handle batching of molecules with varying sizes.
    Pads features to the maximum size in the batch and creates edge masks.

    Args:
        batch (list): A list of data dictionaries, each returned by LipidDataset.__getitem__.

    Returns:
        dict: A dictionary containing batched and padded tensors.
              Keys: 'positions', 'one_hot', 'charges', 'atom_mask', 'edge_mask',
              and the conditioning key if present in the input batch items.
    """
    # Find the maximum number of atoms in the batch
    max_atoms = 0
    for item in batch:
        # Ensure 'atom_mask' key exists
        if 'atom_mask' not in item:
            raise ValueError(f"Missing 'atom_mask' in dataset item: {item.keys()}")
        max_atoms = max(max_atoms, item['atom_mask'].shape[0])

    # Initialize lists to hold padded data for each key
    batch_positions = []
    batch_one_hot = []
    batch_charges = []
    batch_atom_mask = []
    batch_conditioning = [] # Store conditioning values if they exist

    # Check if conditioning data is present in the first item (assume consistency)
    conditioning_key = None
    if batch:
        # Find a key that is not one of the standard ones
        standard_keys = {'positions', 'one_hot', 'charges', 'atom_mask'}
        extra_keys = [k for k in batch[0].keys() if k not in standard_keys]
        if len(extra_keys) == 1:
            conditioning_key = extra_keys[0]
        elif len(extra_keys) > 1:
            print(f"Warning: Multiple potential conditioning keys found: {extra_keys}. Using the first one: {extra_keys[0]}")
            conditioning_key = extra_keys[0]

    # Pad each molecule's data and append to batch lists
    for item in batch:
        num_atoms = item['atom_mask'].shape[0]
        padding_size = max_atoms - num_atoms

        # Pad 'positions' (N, 3)
        padded_positions = np.pad(item['positions'], ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        batch_positions.append(padded_positions)

        # Pad 'one_hot' (N, num_atom_types)
        padded_one_hot = np.pad(item['one_hot'], ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        batch_one_hot.append(padded_one_hot)

        # Pad 'charges' (N, 1)
        padded_charges = np.pad(item['charges'], ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        batch_charges.append(padded_charges)

        # Pad 'atom_mask' (N, 1)
        padded_atom_mask = np.pad(item['atom_mask'], ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        batch_atom_mask.append(padded_atom_mask)

        # Append conditioning value if present
        if conditioning_key and conditioning_key in item:
            batch_conditioning.append(item[conditioning_key])

    # Stack lists into tensors
    batch_dict = {
        'positions': torch.tensor(np.stack(batch_positions), dtype=torch.float32),
        'one_hot': torch.tensor(np.stack(batch_one_hot), dtype=torch.float32),
        'charges': torch.tensor(np.stack(batch_charges), dtype=torch.float32),
        'atom_mask': torch.tensor(np.stack(batch_atom_mask), dtype=torch.float32)
    }

    # Add conditioning tensor if data was present
    if conditioning_key and batch_conditioning:
        # Assuming scalar conditioning values for now
        batch_dict[conditioning_key] = torch.tensor(batch_conditioning, dtype=torch.float32).unsqueeze(1) # Add feature dim

    # Create edge mask (batch_size, max_atoms, max_atoms)
    # An edge exists if both atoms are real (mask is 1)
    atom_mask_squeezed = batch_dict['atom_mask'].squeeze(-1) # (batch_size, max_atoms)
    edge_mask = atom_mask_squeezed.unsqueeze(1) * atom_mask_squeezed.unsqueeze(2)
    batch_dict['edge_mask'] = edge_mask

    return batch_dict

def get_dataloaders(data_path, batch_size, num_workers=0, seed=42, val_split_ratio=0.1):
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
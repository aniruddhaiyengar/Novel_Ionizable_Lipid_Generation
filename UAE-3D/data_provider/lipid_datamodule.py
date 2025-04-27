# UAE-3D/data_provider/lipid_datamodule.py
import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from rdkit import Chem # Import Chem
import numpy as np # Import numpy

# Assuming dataset and utils are available via relative imports
from .lipid_dataset import LipidDataset
from .utils import DataCollater, SimpleDataset, SimpleCollater, get_dataset_info, get_node_dist, datamodule_setup_evaluator

# Default values, might need adjustment based on lipid properties
# DEFAULT_MAX_ATOMS_LIPID = 250 # Placeholder - adjust based on your data
# DEFAULT_POSITION_STD_LIPID = 2.5 # Placeholder - adjust based on your data
# DEFAULT_ATOM_TYPES_LIPID = 16 # Placeholder (like GEOM) - adjust based on analysis
# DEFAULT_BOND_TYPES_LIPID = 4 # Placeholder (like GEOM) - adjust based on analysis

class LipidVAEDataModule(L.LightningDataModule):
    """
    DataModule for Lipid dataset (VAE Training/Fine-tuning).
    """
    def __init__(
        self,
        root: str = 'data/Lipids',
        num_workers: int = 0,
        batch_size: int = 128, # Adjust batch size based on GPU memory
        aug_rotation: bool = True,
        aug_translation: bool = True,
        aug_translation_scale: float = 0.1,
        *args, **kwargs # Capture unused arguments
    ):
        super().__init__()
        # Save hyperparameters BEFORE calculating statistics
        self.save_hyperparameters(ignore=['args', 'kwargs']) # Save relevant args
        self.root = self.hparams.root
        # Initialize stats attributes to None, they will be set in setup
        self.n_atom_types = None
        self.n_bond_types = None
        self.max_atoms = None
        self.position_std = None
        # Keep hparams accessible
        self.num_workers = self.hparams.num_workers
        self.batch_size = self.hparams.batch_size
        self.aug_rotation = self.hparams.aug_rotation
        self.aug_translation = self.hparams.aug_translation
        self.aug_translation_scale = self.hparams.aug_translation_scale


    def setup(self, stage: str | None = None):
        """Load data and calculate statistics."""
        # Load datasets only once
        if not hasattr(self, 'train_dataset'):
            print(f"Setting up train dataset from root: {self.root}")
            self.train_dataset = LipidDataset(root=self.root, split='train')
        if not hasattr(self, 'val_dataset'):
            print(f"Setting up validation dataset from root: {self.root}")
            self.val_dataset = LipidDataset(root=self.root, split='val')

        # Calculate statistics only once
        if self.max_atoms is None: # Check if stats are already calculated
            self._calculate_statistics()

    def _calculate_statistics(self):
        """Calculates statistics by iterating through train and validation sets."""
        print("Calculating dataset statistics (this might take a while)...")
        atom_symbols = set()
        bond_types = set()
        max_atoms_found = 0
        all_coords_list = [] # Use a list to append tensors

        # Iterate through training set
        print("Iterating through training set...")
        for i in tqdm(range(len(self.train_dataset)), desc="Train Stats"):
            try:
                data = self.train_dataset.get(i)
                mol = data.rdmol # Assumes LipidDataset stores rdmol
                if mol is None: continue

                num_atoms = mol.GetNumAtoms()
                if num_atoms > max_atoms_found: max_atoms_found = num_atoms

                for atom in mol.GetAtoms():
                    atom_symbols.add(atom.GetSymbol())
                for bond in mol.GetBonds():
                    bond_types.add(bond.GetBondType())

                if mol.GetNumConformers() > 0:
                     # Use the position tensor directly from the Data object if available
                    if hasattr(data, 'pos') and data.pos is not None:
                        all_coords_list.append(data.pos)
                    else:
                        # Fallback to extracting from rdmol if pos wasn't stored correctly
                        coords = mol.GetConformers()[0].GetPositions()
                        all_coords_list.append(torch.tensor(coords, dtype=torch.float))
            except Exception as e:
                print(f"Warning: Error processing train data index {i}: {e}")
                continue # Skip problematic data points

        # Iterate through validation set
        print("Iterating through validation set...")
        for i in tqdm(range(len(self.val_dataset)), desc="Validation Stats"):
            try:
                data = self.val_dataset.get(i)
                mol = data.rdmol
                if mol is None: continue

                num_atoms = mol.GetNumAtoms()
                if num_atoms > max_atoms_found: max_atoms_found = num_atoms

                for atom in mol.GetAtoms():
                    atom_symbols.add(atom.GetSymbol())
                for bond in mol.GetBonds():
                    bond_types.add(bond.GetBondType())

                if mol.GetNumConformers() > 0:
                    if hasattr(data, 'pos') and data.pos is not None:
                        all_coords_list.append(data.pos)
                    else:
                        coords = mol.GetConformers()[0].GetPositions()
                        all_coords_list.append(torch.tensor(coords, dtype=torch.float))
            except Exception as e:
                print(f"Warning: Error processing validation data index {i}: {e}")
                continue # Skip problematic data points

        # Finalize statistics
        # Add +2 buffer to max_atoms like GEOM datamodule
        self.max_atoms = max_atoms_found + 2
        # Count unique atom symbols found
        self.n_atom_types = len(atom_symbols)
        # Count unique RDKit bond types found (e.g., SINGLE, DOUBLE, etc.)
        self.n_bond_types = len(bond_types)

        # Calculate position standard deviation
        if all_coords_list:
            all_coords_tensor = torch.cat(all_coords_list, dim=0)
            self.position_std = all_coords_tensor.std().item()
            del all_coords_tensor # Free memory
        else:
            print("Warning: No coordinates found to calculate position_std. Using default value (2.5).")
            self.position_std = 2.5 # Fallback default

        print(f"Calculated Stats: MaxAtoms={self.max_atoms}, AtomTypes={self.n_atom_types} ({sorted(list(atom_symbols))}), BondTypes={self.n_bond_types} ({sorted([str(bt) for bt in bond_types])}), PosStd={self.position_std:.4f}")


    def setup_evaluator(self):
        """Optional setup for evaluation metrics (e.g., Moses)."""
        # Requires val_dataset to be loaded and stats calculated
        if not hasattr(self, 'val_dataset') or self.max_atoms is None:
            self.setup() # Ensure setup and calculations are done

        if hasattr(self, 'val_dataset'):
             self.val_rdmols = [self.val_dataset.get(i).rdmol for i in range(len(self.val_dataset))]
        else:
             self.val_rdmols = []
             print("Warning: Validation dataset not loaded, cannot set val_rdmols for evaluator.")

        # Ensure datamodule_setup_evaluator has access to self.train_dataset.get(0).rdmol if needed
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > 0:
            datamodule_setup_evaluator(self)
        else:
            print("Warning: Train dataset not loaded or empty, cannot complete evaluator setup.")

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'): self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0, # Avoid warning
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'): self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    # Define test_dataloader if you create a test split
    # def test_dataloader(self):
    #     ... implement using LipidDataset(split='test') ...

    def collate_fn(self, batch):
        # Ensure statistics are available before collating
        if self.position_std is None: self.setup()
        # Use the same collater as GEOMDrugs, passing necessary args
        return DataCollater(
            position_std=self.position_std,
            aug_rotation=self.aug_rotation,
            aug_translation=self.aug_translation,
            aug_translation_scale=self.aug_translation_scale
        )(batch)

class LipidLDMDataModule(LipidVAEDataModule):
    """
    DataModule for Lipid dataset (LDM Training/Fine-tuning/Sampling).
    Inherits train/val setup from VAE datamodule.
    Customizes test dataloader for sampling.
    """
    def __init__(self,
        root: str = 'data/Lipids',
        num_workers: int = 0,
        batch_size: int = 128,
        aug_rotation: bool = True, # Usually False for LDM training
        aug_translation: bool = False,
        aug_translation_scale: float = 0.1,
        condition_property=None, # Placeholder for future conditional gen
        num_samples=10000, # For sampling test dataloader
        *args, **kwargs
    ):
        # Pass relevant arguments to parent
        super().__init__(
            root=root,
            num_workers=num_workers,
            batch_size=batch_size,
            aug_rotation=aug_rotation,
            aug_translation=aug_translation,
            aug_translation_scale=aug_translation_scale,
            *args, **kwargs # Pass along any other args
        )
        # Save num_samples specifically if needed
        self.num_samples = num_samples
        # Override augmentation defaults specifically for LDM training if needed
        self.aug_rotation = aug_rotation # Keep passed value
        self.aug_translation = aug_translation

    def setup(self, stage: str | None = None):
        # Use parent setup for train/val datasets and statistics calculation
        super().setup(stage)
        # Setup test dataset for sampling
        if stage == "test" or stage is None:
             # Ensure stats are calculated before test setup if needed
             if self.max_atoms is None: 
                  print("Calculating stats before test setup...")
                  self._calculate_statistics()
             # Now setup the simple dataset for sampling
             self.test_dataset = SimpleDataset(self.num_samples)

    def test_dataloader(self):
        """Returns DataLoader for sampling (uses SimpleDataset)."""
        if not hasattr(self, 'test_dataset'):
             self.setup(stage="test") # Ensure test_dataset is created

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=SimpleCollater(), # Use simple collater for sampling indices
            persistent_workers=self.num_workers > 0,
        ) 
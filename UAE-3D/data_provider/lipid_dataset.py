# UAE-3D/data_provider/lipid_dataset.py
import copy
import os
import torch
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

# Assuming utils are in the same parent directory level
from .utils import featurize_mol # Use relative import

RDLogger.DisableLog("rdApp.*")

class LipidDataset(InMemoryDataset):
    """
    Dataset class for loading lipid structures from SDF files.
    Processes raw SDFs into featurized PyG Data objects.
    """
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.root_path = Path(root)
        self.split = split
        # Determine raw filename based on split
        self.raw_filename = f"{self.split}_lipids.sdf"
        super().__init__(str(self.root_path), transform, pre_transform, pre_filter)

        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Loaded processed lipid data for split '{self.split}' from {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"Processed file not found for split '{self.split}'. Processing raw data...")
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Finished processing and loaded data for split '{self.split}'.")

    @property
    def raw_dir(self):
        # Raw files are directly in the root directory specified
        return str(self.root_path)

    @property
    def processed_dir(self):
        # Save processed files in a 'processed' subdirectory
        return os.path.join(self.root_path, 'processed')

    @property
    def raw_file_names(self):
        # The name of the input SDF file for this split
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        # The name of the file where processed data will be saved
        return [f'processed_{self.split}_lipid_data.pt']

    def download(self):
        # Data is assumed to be pre-split by preprocess_lipids.py
        pass

    def process(self):
        """
        Processes raw SDF files into PyG Data objects and saves them.
        """
        raw_sdf_path = self.raw_paths[0]
        print(f"Processing raw lipid data from: {raw_sdf_path}")

        if not os.path.exists(raw_sdf_path):
             raise FileNotFoundError(f"Raw SDF file not found for split '{self.split}' at {raw_sdf_path}. Run preprocess_lipids.py first.")

        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

        # Important: Load with sanitize=False initially, handle sanitization/errors per molecule
        supplier = Chem.SDMolSupplier(raw_sdf_path, removeHs=False, sanitize=False)

        data_list = []
        num_processed = 0
        num_errors = 0
        for mol in tqdm(supplier, desc=f"Processing {os.path.basename(raw_sdf_path)}"):
            if mol is None:
                num_errors += 1
                continue

            try:
                # 1. Standardize/Sanitize (important for consistent featurization)
                Chem.SanitizeMol(mol)

                # 2. Ensure single conformer
                if mol.GetNumConformers() != 1:
                    num_errors += 1
                    continue # Skip if not exactly one conformer

                # 3. Featurize using the utility function
                # Use 'geom_with_h_1' setting as a starting point
                data = featurize_mol(mol, dataset='geom_with_h_1')

                # 4. Add position information and rdmol
                pos = mol.GetConformers()[0].GetPositions()
                data.pos = torch.tensor(pos, dtype=torch.float)
                # Store a copy to avoid issues if original mol object is modified elsewhere
                data.rdmol = copy.deepcopy(mol)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                num_processed += 1

            except Exception as e:
                # print(f"Warning: Skipping molecule due to processing error: {e}")
                num_errors += 1
                continue
        del supplier # Release file handle

        if not data_list:
             raise ValueError(f"No molecules processed successfully for split '{self.split}' from {raw_sdf_path}")

        print(f"Successfully processed {num_processed} molecules, skipped {num_errors} for split '{self.split}'.")

        data, slices = self.collate(data_list)
        print(f"Saving processed data to {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])

    # get and len methods are inherited from InMemoryDataset

    def __getitem__(self, idx):
        # Overriding __getitem__ needed for PyTorch Lightning DataLoaders
        if isinstance(idx, int):
            data = self.get(self.indices()[idx] if hasattr(self, 'indices') else idx)
            # Return a copy to prevent modification of the original object
            # Using clone() might be better if Data objects are complex
            data_copy = copy.copy(data)
            data_copy.idx = idx # Add index if needed downstream
            return data_copy
        else:
            # Handle slicing or list indexing
            return self.index_select(idx) 
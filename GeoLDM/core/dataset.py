from torch.utils.data import DataLoader
from GeoLDM.core.data.args import init_argparse
from GeoLDM.core.data.collate import PreprocessQM9
from GeoLDM.core.data.utils import initialize_datasets
import os


def retrieve_dataloaders(cfg):
    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        # Assume build_geom_dataset is in the *parent* directory or PYTHONPATH
        try:
            import build_geom_dataset 
        except ImportError:
            # If running from GeoLDM/core, need to go up two levels
            import sys
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if parent_dir not in sys.path:
                 sys.path.insert(0, parent_dir)
            import build_geom_dataset
            
        from .configs.datasets_config import get_dataset_info
        data_file = 'GeoLDM/data/geom/geom_drugs_30.npy' 
        if not os.path.exists(data_file):
             script_dir_data_file = os.path.join(os.path.dirname(__file__), '../data/geom/geom_drugs_30.npy')
             if os.path.exists(script_dir_data_file):
                 data_file = script_dir_data_file
             else:
                  raise FileNotFoundError(f"Could not find geom data file at {data_file} or {script_dir_data_file}")
                  
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets


if __name__ == '__main__':
    # This is a simple example of how to use the dataloaders
    class Config:
        def __init__(self):
            self.batch_size = 128
            self.num_workers = 0
            self.filter_n_atoms = None
            self.datadir = "GeoLDM/core/data/qm9/temp/qm9"
            self.dataset = "qm9"
            self.remove_h = False
            self.include_charges = True

    cfg = Config()
    dataloaders, charge_scale = retrieve_dataloaders(cfg)
    for i, data in enumerate(dataloaders['train']):
        print(data)
        break
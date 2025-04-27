import argparse
from pathlib import Path
import os

import lightning as L
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.utils import to_dense_batch

from data_provider.utils import add_datamodule_specific_args
from evaluation import add_evaluation_specific_args, molecule_evaluate, atom_prediction_accuracy, bond_prediction_accuracy, coordinate_prediction_rmsd
from model.autoencoder.multimodal_ae import MultiModalAE
from training_utils import add_training_specific_args, custom_callbacks, device_cast, print_args, suppress_warning
from data_provider.qm9_datamodule import QM9VAEDataModule
from data_provider.geom_drugs_datamodule import GEOMDrugsVAEDataModule
from data_provider.lipid_datamodule import LipidVAEDataModule

torch.set_float32_matmul_precision('high') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


class MultiModalAETrainer(L.LightningModule):
    def __init__(self, model, position_std, args):
        super().__init__()
        self.model = model
        self.args = args
        self.position_std = position_std

        self.test_molecule_list = []

        self.save_hyperparameters(args)

    def forward(self, batch):
        loss_dict = self.model(batch)
        return loss_dict

    @torch.no_grad()
    def accuracy_evaluate(self, atom_logits, bond_logits, coordinates, batch):
        n_atom_types = self.model.decoder.n_atom_types
        atom_accuracy = atom_prediction_accuracy(atom_logits, batch.x[:, :n_atom_types], batch.batch)

        n_bond_types = self.model.decoder.n_bond_types
        bond_types = batch.edge_attr[:, :n_bond_types+1]
        bond_types = torch.cat([(bond_types == 0).all(dim=-1, keepdim=True).float(), bond_types], dim=-1) # BT.UNSPECIFIED
        bond_accuracy = bond_prediction_accuracy(bond_logits, bond_types)

        coordinates = coordinates * self.position_std
        coordinate_rmsd = coordinate_prediction_rmsd(coordinates, batch.pos * self.position_std, batch.batch)

        return atom_accuracy, bond_accuracy, coordinate_rmsd

    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, sync_dist=True, batch_size=self.args.batch_size)

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        for key, value in loss_dict.items():
            self.log(f'valid/{key}', value, sync_dist=True, batch_size=self.args.batch_size)

        z = self.model.encode(batch) # [B * N, latent_dim]
        atom_logits, bond_logits, coordinates = self.model.decode(z, batch=batch.batch) # [B * N, n_atom_types], [B * N * N, n_bond_types + 2], [B * N, 3]

        atom_accuracy, bond_accuracy, coordinate_rmsd = self.accuracy_evaluate(atom_logits, bond_logits, coordinates, batch)

        batch_size = batch.ptr.size(0) - 1
        num_atoms = batch.x.size(0)
        num_bonds = batch.edge_index.size(1)
        self.log('valid/atom_accuracy', float(atom_accuracy), sync_dist=True, batch_size=num_atoms)
        self.log('valid/bond_accuracy', float(bond_accuracy), sync_dist=True, batch_size=num_bonds)
        self.log('valid/coordinate_rmsd', float(coordinate_rmsd), sync_dist=True, batch_size=batch_size)

        return loss_dict['loss']

    @torch.no_grad()
    def on_test_epoch_start(self):
        self.test_molecule_list = []

    def test_step(self, batch, batch_idx):
        z = self.model.encode(batch) # [B * N, latent_dim]
        atom_logits, bond_logits, coordinates = self.model.decode(z, batch=batch.batch) # [B * N, n_atom_types], [B * N * N, n_bond_types + 2], [B * N, 3]

        atom_accuracy, bond_accuracy, coordinate_rmsd = self.accuracy_evaluate(atom_logits, bond_logits, coordinates, batch)

        batch_size = batch.ptr.size(0) - 1
        num_atoms = batch.x.size(0)
        num_bonds = batch.edge_index.size(1)
        self.log('test/atom_accuracy', float(atom_accuracy), sync_dist=True, batch_size=num_atoms)
        self.log('test/bond_accuracy', float(bond_accuracy), sync_dist=True, batch_size=num_bonds)
        self.log('test/coordinate_rmsd', float(coordinate_rmsd), sync_dist=True, batch_size=batch_size)


        # Reconstruct molecules for evaluation
        _, unpadding_mask = to_dense_batch(z, batch.batch)
        batch_size, max_num_nodes = unpadding_mask.shape

        n_atom_types = self.model.decoder.n_atom_types
        atom_logits_full = torch.zeros((batch_size, max_num_nodes, n_atom_types), dtype=atom_logits.dtype, device=atom_logits.device)
        atom_logits_full.masked_scatter_(unpadding_mask.unsqueeze(-1), atom_logits) # [B, max_N, n_atom_types]
        atom_types = atom_logits_full.argmax(dim=-1) # [B, max_num_nodes]

        n_bond_types = self.model.decoder.n_bond_types
        bond_logits_full = torch.zeros((batch_size, max_num_nodes, max_num_nodes, n_bond_types + 2), dtype=bond_logits.dtype, device=bond_logits.device)
        bond_mask = unpadding_mask.unsqueeze(1) & unpadding_mask.unsqueeze(2)  # [B, N, N]
        bond_logits_full.masked_scatter_(bond_mask.unsqueeze(-1), bond_logits) # [B, max_N, max_N, n_bond_types + 2]
        bond_types = bond_logits_full[..., :-1].argmax(dim=-1) # [B, max_num_nodes, max_num_nodes], exclude self-loop

        coordinates_full = torch.zeros((batch_size, max_num_nodes, 3), dtype=coordinates.dtype, device=coordinates.device)
        coordinates_full.masked_scatter_(unpadding_mask.unsqueeze(-1), coordinates) # [B, max_N, 3]
        positions = coordinates_full * self.position_std

        for i, idx in enumerate(batch.idx):
            atom_type = atom_types[i][unpadding_mask[i]].long().cpu() # [num_nodes]
            bond_type = bond_types[i][unpadding_mask[i]][:, unpadding_mask[i]].long().cpu() # [num_nodes, num_nodes]
            position = positions[i][unpadding_mask[i]].float().cpu() # [num_nodes, 3]
            formal_charge = torch.zeros_like(atom_type) # [num_nodes]
            self.test_molecule_list.append((idx, (position, atom_type, bond_type, formal_charge)))


    @torch.no_grad()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def on_test_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        if len(self.test_molecule_list) <= 0:
            print("WARNING: No molecules generated")
            return

        if dist.is_initialized():
            gather_box = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(gather_box, self.test_molecule_list)
        else:
            gather_box = [self.test_molecule_list]

        molecule_list = {idx: molecule_data for data in gather_box for idx, molecule_data in data}
        molecule_list = list(molecule_list.values())

        self.test_molecule_list = []

        if not self.trainer.is_global_zero:
            return

        evaluation_dict, reconstructed_rdmols_3D, reconstructed_rdmols_2D = molecule_evaluate(self.trainer.datamodule, molecule_list, self.args.evaluate_3D, self.args.evaluate_2D, self.args.evaluate_moses, self.args.evaluate_align)

        for key, value in evaluation_dict.items():
            self.log(f'test/{key}', value)

        if self.args.use_wandb:
            log_dir = Path(self.loggers[1].log_dir)
        else:
            log_dir = Path(self.logger.log_dir)
        torch.save(molecule_list, log_dir / f'reconstructions_{self.current_epoch}.pt')

        if self.args.evaluate_3D:
            torch.save(reconstructed_rdmols_3D, log_dir / f'reconstructed_rdmols_3D_{self.current_epoch}.pt')
        if self.args.evaluate_2D:
            torch.save(reconstructed_rdmols_2D, log_dir / f'reconstructed_rdmols_2D_{self.current_epoch}.pt')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return optimizer

def main(args):
    print("Instantiating datamodule...")
    if args.dataset == 'qm9':
        datamodule = QM9VAEDataModule(**vars(args))
    elif args.dataset == 'drugs':
        datamodule = GEOMDrugsVAEDataModule(**vars(args))
    elif args.dataset == 'lipids':
        print(f"Using LipidVAEDataModule with root: {args.root}")
        datamodule = LipidVAEDataModule(**vars(args))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("Setting up evaluator...")
    datamodule.setup_evaluator()

    print("Instantiating model...")
    model = MultiModalAE(
        max_atoms=datamodule.max_atoms,
        n_atom_types=datamodule.n_atom_types,
        n_bond_types=datamodule.n_bond_types,
        position_std=datamodule.position_std,
        **vars(args)
    )

    trainer_model = MultiModalAETrainer(model, datamodule.position_std, args)
    trainer_model.model = torch.compile(trainer_model.model, dynamic=True, fullgraph=False, disable=args.disable_compile)

    csv_logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    if args.use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        wandb_logger = WandbLogger(project="Variational Autoencoder", name=args.filename)
        logger = [wandb_logger, csv_logger]
    else:
        logger = csv_logger

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=device_cast(args.devices),
        precision=args.precision,
        logger=logger,
        callbacks=custom_callbacks(args),
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        detect_anomaly=args.detect_anomaly,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val
    )

    if args.finetune_ckpt_path:
        print(f"Attempting to load weights from checkpoint: {args.finetune_ckpt_path}")
        if not os.path.exists(args.finetune_ckpt_path):
             print(f"!!! Error: Checkpoint file not found at {args.finetune_ckpt_path}")
             print("!!! Training will proceed with initialized weights.")
        else:
            try:
                # Load checkpoint to CPU first to avoid device mismatches
                checkpoint = torch.load(args.finetune_ckpt_path, map_location=torch.device('cpu'))

                # Extract the state dict (common key is 'state_dict')
                state_dict_key = 'state_dict'
                if state_dict_key not in checkpoint:
                    # Fallback if 'state_dict' key is missing
                    print(f"Warning: '{state_dict_key}' not found in checkpoint keys: {list(checkpoint.keys())}. Attempting to load entire checkpoint.")
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint[state_dict_key]

                # Load into model, allowing missing/unexpected keys (useful for transfer learning)
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"Info: Missing keys when loading state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Info: Unexpected keys when loading state_dict: {unexpected_keys}")
                print(f"Successfully loaded weights from {args.finetune_ckpt_path}")

            except Exception as e:
                print(f"!!! Error loading checkpoint {args.finetune_ckpt_path}: {e}")
                print("!!! Training will proceed with initialized weights.")

    if args.test_only:
        trainer.test(trainer_model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(trainer_model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        for evaluate_suffix in ['3D', '2D', 'moses', 'align']:
            trainer_model.args.__setattr__(f'evaluate_{evaluate_suffix}', True) # set all evaluation flags to True
        trainer.test(trainer_model, datamodule=datamodule)

if __name__ == '__main__':
    suppress_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='vae_experiment')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    add_training_specific_args(parser)
    add_datamodule_specific_args(parser)

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument('--learning_rate', type=float, default=1e-4)
    optimization.add_argument('--weight_decay', type=float, default=1e-5)

    MultiModalAE.add_model_specific_args(parser)

    add_evaluation_specific_args(parser)

    parser.add_argument('--dataset', type=str, default='qm9', choices=['qm9', 'drugs', 'lipids'], help='Dataset to use')
    parser.add_argument('--root', type=str, default='data/QM9', help='Root directory for dataset data')
    parser.add_argument('--finetune_ckpt_path', type=str, default=None,
                        help='Path to the checkpoint file to load weights from for fine-tuning.')

    args = parser.parse_args()
    print_args(parser, args)

    main(args)

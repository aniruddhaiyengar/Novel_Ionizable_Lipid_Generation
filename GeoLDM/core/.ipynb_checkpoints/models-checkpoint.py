import torch
from torch.distributions.categorical import Categorical

import numpy as np
# Use explicit relative path from GeoLDM
from GeoLDM.egnn.models import EGNN_dynamics_QM9, EGNN_encoder_QM9, EGNN_decoder_QM9

# Use explicit relative path from GeoLDM
from GeoLDM.equivariant_diffusion.en_diffusion import EnVariationalDiffusion, EnHierarchicalVAE, EnLatentDiffusion

import pickle
from os.path import join
import logging

def get_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_autoencoder(args, device, dataset_info, dataloader_train):
    print(f"DEBUG STAGE1: In get_autoencoder, ENTRY args.latent_nf = {getattr(args, 'latent_nf', 'NOT FOUND')}, args.include_charges = {getattr(args, 'include_charges', 'NOT FOUND')}") # Print on entry
    histogram = dataset_info['n_nodes']
    # Original calculation:
    # effective_in_node_nf_for_vae = 16 # OLD hardcoded value
    # Use dynamic calculation based on the provided dataset_info and args
    effective_in_node_nf_for_vae = len(dataset_info['atom_decoder']) + int(args.include_charges)
    print(f"DEBUG STAGE1: In get_autoencoder, AFTER effective_in_node_nf_for_vae, args.latent_nf = {getattr(args, 'latent_nf', 'NOT FOUND')}")
    
    logging.info(f"[get_autoencoder] Effective VAE in_node_nf: {effective_in_node_nf_for_vae} (from atom_decoder len {len(dataset_info['atom_decoder'])} + charges {int(args.include_charges)})")

    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    # Use getattr to safely access conditioning, default to empty list
    conditioning_args = getattr(args, 'conditioning', []) 
    # if len(args.conditioning) > 0:
    if len(conditioning_args) > 0:
        prop_dist = DistributionProperty(dataloader_train, conditioning_args)
    print(f"DEBUG STAGE1: In get_autoencoder, AFTER prop_dist, args.latent_nf = {getattr(args, 'latent_nf', 'NOT FOUND')}")

    # if args.condition_time:
    #     dynamics_in_node_nf = in_node_nf + 1
    print('Autoencoder models are _not_ conditioned on time.')
        # dynamics_in_node_nf = in_node_nf
    
    # Use the main args.nf for the VAE's hidden dimension, unless a VAE-specific one is intended and correctly handled.
    # The runtime error implies the loaded checkpoint's VAE encoder EGNN used hidden_nf=16.
    # If we want to use args.nf (256), we set it here. This will likely cause weight mismatch for this layer.
    vae_hidden_nf = args.nf 
    print(f"DEBUG: [get_autoencoder] Using hidden_nf = {vae_hidden_nf} (from args.nf) for VAE encoder/decoder.")
    print(f"DEBUG STAGE1: In get_autoencoder, AFTER vae_hidden_nf, args.latent_nf = {getattr(args, 'latent_nf', 'NOT FOUND')}")

    encoder_out_node_nf = args.latent_nf
    print(f"CRITICAL LOG STAGE1: [get_autoencoder] Initializing VAE Encoder: args.latent_nf = {args.latent_nf}, computed encoder_out_node_nf (target for EGNN_encoder_QM9.out_node_nf) = {encoder_out_node_nf}")

    encoder = EGNN_encoder_QM9(
        in_node_nf=effective_in_node_nf_for_vae,  # Use modified value (dynamic)
        context_node_nf=args.context_node_nf, 
        # out_node_nf=args.latent_nf, # OLD: Incorrect for mu, sigma
        out_node_nf=encoder_out_node_nf, # NEW: Encoder must output 2*latent_nf for mu and sigma
        n_dims=3, 
        device=device, 
        hidden_nf=vae_hidden_nf,  # Use args.nf (e.g., 256)
        act_fn=torch.nn.SiLU(), 
        n_layers=args.n_layers, # Consider if VAE should have different n_layers than main model
        attention=args.attention, 
        tanh=args.tanh, 
        mode=args.model, # This might need to be 'egnn_dynamics' or a VAE-specific mode if applicable
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, 
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, 
        aggregation_method=args.aggregation_method,
        include_charges=args.include_charges # This is for the EGNN_encoder_QM9 internal logic, if any
        )
    print(f"DEBUG STAGE1: In get_autoencoder, AFTER encoder init, args.latent_nf = {getattr(args, 'latent_nf', 'NOT FOUND')}")
    
    decoder = EGNN_decoder_QM9(
        in_node_nf=args.latent_nf, 
        context_node_nf=args.context_node_nf, 
        out_node_nf=effective_in_node_nf_for_vae, # Output should also match the 6 features
        n_dims=3, 
        device=device, 
        hidden_nf=vae_hidden_nf, # Use args.nf (e.g., 256)
        act_fn=torch.nn.SiLU(), 
        n_layers=args.n_layers, # Consistent n_layers
        attention=args.attention, 
        tanh=args.tanh, 
        mode=args.model, 
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, 
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, 
        aggregation_method=args.aggregation_method,
        include_charges=args.include_charges
        )
    print(f"DEBUG STAGE1: In get_autoencoder, AFTER decoder init, args.latent_nf = {getattr(args, 'latent_nf', 'NOT FOUND')}")

    vae = EnHierarchicalVAE(
        encoder=encoder,
        decoder=decoder,
        in_node_nf=effective_in_node_nf_for_vae, # Pass the corrected value (6)
        n_dims=3,
        latent_node_nf=args.latent_nf,
        kl_weight=args.kl_weight,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges # This is for EnHierarchicalVAE internal logic
        )
    print(f"DEBUG STAGE1: In get_autoencoder, AFTER VAE WRAPPER init, args.latent_nf = {getattr(args, 'latent_nf', 'NOT FOUND')}")

    return vae, nodes_dist, prop_dist


def get_latent_diffusion(args, device, dataset_info, dataloader_train):

    # Create (and load) the first stage model (Autoencoder).
    if args.trainable_ae:
        print("Loading VAE model from checkpoint")
        ae_path = getattr(args, 'ae_path', None)
        if ae_path is not None:
            with open(join(ae_path, 'args.pickle'), 'rb') as f:
                ae_args = pickle.load(f)
            first_stage_args = ae_args # Assign from loaded AE args
        else:
            print("Warning: trainable_ae is True, but no ae_path provided. Using main args for AE configuration.")
            first_stage_args = args # Assign main args as fallback
    else:
        first_stage_args = args # Assign main args if not training AE
    
    # CAREFUL with this -->
    # Ensure these common attributes exist, using defaults if necessary
    if not hasattr(first_stage_args, 'normalization_factor'):
        print("Setting default normalization_factor=1 for AE args")
        first_stage_args.normalization_factor = 1
    if not hasattr(first_stage_args, 'aggregation_method'):
        print("Setting default aggregation_method='sum' for AE args")
        first_stage_args.aggregation_method = 'sum'
    # Ensure cuda attribute consistency if loaded from different setup
    if not hasattr(first_stage_args, 'cuda'):
         first_stage_args.cuda = args.cuda

    # device = torch.device("cuda" if first_stage_args.cuda else "cpu") # Use device passed into function

    first_stage_model, nodes_dist, prop_dist = get_autoencoder(
        first_stage_args, device, dataset_info, dataloader_train)
    first_stage_model.to(device)

    # Safely get ae_path using getattr
    ae_path = getattr(args, 'ae_path', None) 
    # if args.ae_path is not None:
    if ae_path is not None:
        fn = 'generative_model_ema.npy' if first_stage_args.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(ae_path, fn),
                                        map_location=device)
        first_stage_model.load_state_dict(flow_state_dict)

    # Create the second stage model (Latent Diffusions).
    args.latent_nf = first_stage_args.latent_nf
    in_node_nf = args.latent_nf

    # Use getattr to safely access condition_time, default to False
    condition_time = getattr(args, 'condition_time', False)
    # if args.condition_time:
    if condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    # Use getattr to safely access probabilistic_model, default to 'diffusion'
    probabilistic_model = getattr(args, 'probabilistic_model', 'diffusion')
    if probabilistic_model == 'diffusion':
        vdm = EnLatentDiffusion(
            vae=first_stage_model,
            trainable_ae=args.trainable_ae,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(f"Unknown probabilistic_model: {probabilistic_model}")


def get_optim(args, generative_model, weight_decay=1e-12):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=weight_decay)

    return optim


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())

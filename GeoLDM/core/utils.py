import torch


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == 'qm9':
        return compute_mean_mad_from_dataloader(dataloaders['train'], properties)
    elif dataset_name == 'qm9_second_half' or dataset_name == 'qm9_second_half':
        return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)


    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms, force_unconditional=False):
    """
    Prepares the context tensor for conditioning the model.

    Args:
        conditioning (list): List of property keys to use for conditioning.
        minibatch (dict): Dictionary containing the batch data (must include keys
                          in `conditioning` as well as 'positions' and 'atom_mask').
        property_norms (dict): Dictionary containing normalization statistics
                               (mean, mad/std) for properties in `conditioning`.
        force_unconditional (bool): If True, ignores minibatch properties and
                                    returns a zero context for CFG.

    Returns:
        torch.Tensor or None: The context tensor (batch_size, n_nodes, context_nf)
                              or None if no conditioning is applied or possible.
    """
    # Ensure basic requirements are met
    if 'positions' not in minibatch or 'atom_mask' not in minibatch:
         raise ValueError("'positions' and 'atom_mask' must be in minibatch")

    batch_size, n_nodes, _ = minibatch['positions'].size()
    # Ensure atom_mask has boolean type if used for masking directly
    # Use float mask as in original model code for multiplication
    node_mask = minibatch['atom_mask'].unsqueeze(2).float()

    context_node_nf = 0
    context_list = []

    # Return None if no conditioning keys are provided
    if not conditioning:
        return None

    for key in conditioning:
        current_context = None
        # Determine the feature dimension for this property (usually 1 for scalar global properties)
        target_prop_nf = 1 # Default assumption for TransLNP

        if force_unconditional:
            # --- Logic for unconditional (CFG) ---
            current_context = torch.zeros(batch_size, n_nodes, target_prop_nf,
                                          device=minibatch['positions'].device,
                                          dtype=minibatch['positions'].dtype)
        else:
            # --- Original Logic adjusted --- 
            if key not in minibatch:
                print(f"Warning: Conditioning key '{key}' not found in this minibatch. Skipping.")
                continue

            properties = minibatch[key]

            if key not in property_norms:
                 raise KeyError(f"Normalization stats for key '{key}' not found in property_norms.")

            norm_mean = property_norms[key]['mean']
            norm_std = property_norms[key]['mad'] # Use 'mad' key as std dev

            # Ensure mean/std are tensors and on the same device/dtype
            if not isinstance(norm_mean, torch.Tensor): norm_mean = torch.tensor(norm_mean)
            if not isinstance(norm_std, torch.Tensor): norm_std = torch.tensor(norm_std)
            norm_mean = norm_mean.to(properties.device, dtype=properties.dtype)
            norm_std = norm_std.to(properties.device, dtype=properties.dtype)

            # Normalize
            if norm_std.abs() > 1e-6:
                properties = (properties - norm_mean) / norm_std
            else:
                properties = properties - norm_mean
                # print(f"Warning: Std deviation for property '{key}' is near zero. Only applying mean centering.")

            # Process based on property dimension
            if properties.dim() == 1: # Global property (batch_size,)
                if properties.size(0) != batch_size:
                     raise ValueError(f"Global property '{key}' batch size mismatch.")
                reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
                current_context = reshaped
                target_prop_nf = 1
            # Add handling for node features if needed in the future
            # elif properties.dim() == 2 or properties.dim() == 3:
            #     # Node feature logic...
            #     pass
            else:
                raise ValueError(f"Invalid tensor dimension {properties.dim()} for property '{key}'. Expected 1 for global.")

        # Append the processed context for this key
        if current_context is not None:
            context_list.append(current_context)
            context_node_nf += target_prop_nf

    if not context_list:
        return None

    context = torch.cat(context_list, dim=2)
    # Mask nodes that are padding
    context = context * node_mask # Apply float mask via multiplication

    if context.size(2) != context_node_nf:
         raise RuntimeError(f"Final context dimension mismatch.")

    return context


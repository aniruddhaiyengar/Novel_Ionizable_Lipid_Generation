import torch
import logging


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


def prepare_context(conditioning, minibatch, norms):
    """
    Prepare context tensor for conditioning the model.
    Args:
        conditioning: List of property names to condition on
        minibatch: Dictionary containing the batch data
        norms: Dictionary containing normalization statistics for each property
    Returns:
        Tensor of shape (batch_size, context_nf) containing normalized property values
    """
    if not conditioning:
        return None

    # Get device from input tensor
    device = minibatch['positions'].device
    
    # Initialize list to store normalized properties
    normalized_props = []
    
    for prop in conditioning:
        if prop not in minibatch:
            logging.error(f"Property {prop} not found in minibatch. Available keys: {list(minibatch.keys())}")
            continue
        if prop not in norms:
            logging.error(f"Normalization statistics for {prop} not found. Available norms: {list(norms.keys())}")
            continue
            
        # Get property value and ensure it's a tensor on the correct device
        property_value = minibatch[prop]
        
        # Convert to tensor if not already
        if not isinstance(property_value, torch.Tensor):
            try:
                property_value = torch.tensor(property_value, device=device, dtype=torch.float32)
            except (TypeError, ValueError) as e:
                logging.error(f"Failed to convert property {prop} to tensor. Value: {property_value}, Type: {type(property_value)}. Error: {e}")
                continue
        else:
            property_value = property_value.to(device, dtype=torch.float32)
            
        # Ensure property value has correct shape [batch_size, 1]
        if property_value.dim() == 0:
            property_value = property_value.view(1, 1)
        elif property_value.dim() == 1:
            property_value = property_value.view(-1, 1)
            
        # Get normalization statistics and ensure they're tensors on the correct device
        mean = norms[prop]['mean']
        std_dev = norms[prop]['std']
        
        # Convert normalization stats to tensors if needed
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, device=device, dtype=torch.float32)
        else:
            mean = mean.to(device, dtype=torch.float32)
            
        if not isinstance(std_dev, torch.Tensor):
            std_dev = torch.tensor(std_dev, device=device, dtype=torch.float32)
        else:
            std_dev = std_dev.to(device, dtype=torch.float32)
            
        # Add small epsilon to prevent division by zero
        std_dev = std_dev + 1e-6
            
        # Normalize property
        normalized_prop = (property_value - mean) / std_dev
        normalized_props.append(normalized_prop)
    
    if not normalized_props:
        logging.error("No properties were successfully normalized. This should not happen with valid transfection scores.")
        return None
        
    # Stack normalized properties along feature dimension
    context = torch.cat(normalized_props, dim=1)
    
    # Ensure final context tensor is on the correct device
    context = context.to(device, dtype=torch.float32)
    
    return context


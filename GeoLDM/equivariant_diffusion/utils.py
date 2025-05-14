import torch
import numpy as np
import logging


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    """
    Remove mean from x while respecting the node_mask.
    Args:
        x: Tensor of shape (batch_size, n_nodes, n_dims)
        node_mask: Tensor of shape (batch_size, n_nodes, 1) with binary values
    Returns:
        Tensor of same shape as x with mean removed
    """
    # ---- START MOVED VALIDATION ----
    # Input validation: Ensure these checks are first.
    if x is None or node_mask is None:
        raise ValueError("Both x and node_mask must be provided for remove_mean_with_mask")
    
    # Ensure proper shapes and types
    if not torch.is_tensor(x) or not torch.is_tensor(node_mask):
        raise TypeError("Both x and node_mask must be PyTorch tensors for remove_mean_with_mask")
    # ---- END MOVED VALIDATION ----

    # Debug logging for error tracking
    logging.debug(f"remove_mean_with_mask input - x type: {type(x)}, node_mask type: {type(node_mask)}")
    if x is not None: # This check is now somewhat redundant due to above, but kept for shape/dtype logging
        logging.debug(f"x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
    if node_mask is not None: # Similarly redundant for None, but good for shape/dtype logging
        logging.debug(f"node_mask shape: {node_mask.shape}, dtype: {node_mask.dtype}, device: {node_mask.device}")
    
    if x.dim() != 3:
        raise ValueError(f"x must be 3D tensor, got shape {x.shape}")
    if node_mask.dim() != 3:
        raise ValueError(f"node_mask must be 3D tensor, got shape {node_mask.shape}")
    
    # Ensure node_mask is binary and float
    node_mask = node_mask.float()
    node_mask = (node_mask > 0.5).float()
    
    # Check for masked values
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    if masked_max_abs_value > 1e-5:
        logging.warning(f"High masked value detected: {masked_max_abs_value}")
    
    # Calculate number of valid nodes per batch
    N = node_mask.sum(1, keepdims=True)
    logging.debug(f"Number of valid nodes per batch: {N.squeeze().tolist()}")
    
    # Handle case where N is 0 (no valid nodes in the batch)
    # Important: Check if ALL elements in N are zero for a given batch item, not just if any N is 0.
    # However, the original (N == 0).any() would trigger if any sample in the batch has N=0.
    # Let's refine this to per-sample handling if needed, but for now, keep original logic.
    if (N == 0).any(): # If any sample in the batch has no valid nodes
        logging.warning("Encountered a sample in the batch with no valid nodes. Returning original tensor for that sample or the whole batch based on downstream logic (currently whole batch).")
        # This return is for the whole batch if any sample has N=0.
        # A more granular approach might be needed if only some samples have N=0.
        return x # Potentially problematic if only some samples in batch are all masked
    
    # Calculate mean and remove it
    mean = torch.sum(x, dim=1, keepdim=True) / (N + 1e-8)  # Add small epsilon to prevent division by zero
    x_centered = x - mean * node_mask # Create new tensor for the result
    
    # Final validation
    if torch.isnan(x_centered).any() or torch.isinf(x_centered).any():
        logging.warning("NaN/Inf detected after mean removal. Returning original tensor x.")
        return x # Return original x, not potentially corrupted x_centered
    
    return x_centered


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

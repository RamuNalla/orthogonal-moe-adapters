import torch
import torch.nn as nn
from typing import Dict

class SubspaceExtractor:
    """Captures activations and computes the SVD subspace for Orthogonal PEFT."""
    def __init__(self, model: nn.Module, target_layer_class):
        self.model = model
        self.target_layer_class = target_layer_class
        self.covariances: Dict[str, torch.Tensor] = {}
        self.token_counts: Dict[str, int] = {}
        self.subspaces: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _get_hook(self, name: str):
        def hook(module, inp, output):
            # Use the output (base_output + adapter_output) which is in out_features space (2048).
            # inp[0] is in in_features space (5632) which mismatches the adapter's hidden_dim.
            x = output.detach().float() # Shape: (batch, seq, out_features)
            x_flat = x.view(-1, x.size(-1)) # Shape: (tokens, out_features)
            
            # Incremental Covariance: X^T * X
            cov_update = torch.matmul(x_flat.t(), x_flat)
            
            if name not in self.covariances:
                self.covariances[name] = cov_update.to(x.device)
                self.token_counts[name] = x_flat.size(0)
            else:
                self.covariances[name] += cov_update.to(x.device)
                self.token_counts[name] += x_flat.size(0)
        return hook

    def attach_hooks(self):
        """Attaches forward hooks to all injected adapter layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_layer_class):
                hook_handle = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook_handle)

    def remove_hooks(self):
        """Cleans up hooks after extraction."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_svd_subspaces(self, rank_k: int = 128):
        """Computes SVD on the covariance matrices to extract the top-K subspace."""
        print(f"Computing SVD for {len(self.covariances)} layers (Rank={rank_k})...")
        for name, cov_sum in self.covariances.items():
            # Normalize by token count to get true covariance
            cov_matrix = cov_sum / self.token_counts[name]
            
            # Perform SVD (Singular Value Decomposition)
            U, S, V = torch.linalg.svd(cov_matrix, full_matrices=False)
            
            # Keep only the top 'rank_k' eigenvectors
            top_k_subspace = U[:, :rank_k]
            self.subspaces[name] = top_k_subspace
            
        print("SVD Subspace extraction complete.")
        return self.subspaces
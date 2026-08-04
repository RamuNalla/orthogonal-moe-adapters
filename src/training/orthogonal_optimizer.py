import torch
import torch.nn as nn
from typing import Dict
from src.architecture.injector import AdapterInjectedLinear

class OrthogonalGradientController:
    """Modifies gradients in-place to ensure orthogonality to the base model's knowledge."""
    def __init__(self, model: nn.Module, subspaces: Dict[str, torch.Tensor]):
        self.model = model
        self.subspaces = subspaces

    def project_gradients(self):
        """
        To be called after `loss.backward()` but BEFORE `optimizer.step()`.
        Applies: G_ortho = G - G @ U @ U^T
        """
        for name, module in self.model.named_modules():
            # Only apply to our injected adapter layers
            if isinstance(module, AdapterInjectedLinear) and name in self.subspaces:
                U = self.subspaces[name] # Shape: (hidden_dim, rank_k)
                
                # 1. Project the Routing Gate Gradients
                gate_weight = module.moe_adapter.gate.weight
                if gate_weight.grad is not None:
                    G = gate_weight.grad # Shape: (num_experts, hidden_dim)
                    # Projection math
                    G_proj = torch.matmul(torch.matmul(G, U), U.t())
                    gate_weight.grad.sub_(G_proj) # In-place subtraction
                
                # 2. Project all Expert Down-Projections Gradients
                for expert in module.moe_adapter.experts:
                    down_weight = expert.down_proj.weight
                    if down_weight.grad is not None:
                        G = down_weight.grad # Shape: (bottleneck_dim, hidden_dim)
                        # Projection math
                        G_proj = torch.matmul(torch.matmul(G, U), U.t())
                        down_weight.grad.sub_(G_proj) # In-place subtraction
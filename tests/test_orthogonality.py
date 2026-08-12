import torch
import torch.nn as nn
import pytest

from src.architecture.injector import AdapterInjectedLinear
from src.training.orthogonal_optimizer import OrthogonalGradientController

def test_gradient_orthogonality():
    """Mathematically proves that projected gradients are strictly orthogonal to the base subspace."""
    
    hidden_dim = 32
    rank_k = 8
    
    # 1. Setup a dummy base layer and inject the MoE Adapter
    base_layer = nn.Linear(hidden_dim, hidden_dim)
    injected_layer = AdapterInjectedLinear(base_layer, bottleneck_dim=8, num_experts=2, top_k=1)
    
    # Wrap in a dummy model dictionary-like structure
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.test_layer = injected_layer
            
    model = DummyModel()
    
    # 2. Create a simulated Orthonormal Subspace (U)
    # We use QR decomposition to ensure the simulated subspace vectors are perfectly orthogonal to each other
    random_matrix = torch.randn(hidden_dim, rank_k)
    U, _ = torch.linalg.qr(random_matrix) 
    subspaces = {"test_layer": U}
    
    # 3. Simulate Forward and Backward Pass
    x = torch.randn(4, hidden_dim) # Batch 4
    output = model.test_layer(x)
    loss = output.sum()
    loss.backward()
    
    # 4. Measure Interference BEFORE Projection
    G_before = model.test_layer.moe_adapter.gate.weight.grad.clone()
    interference_before = torch.norm(torch.matmul(G_before, U)).item()
    
    # The unmodified gradient should inherently collide with the subspace
    assert interference_before > 1e-4, f"Test flawed: Initial gradient is accidentally orthogonal."
    
    # 5. Apply the SOTA Orthogonal Projection
    controller = OrthogonalGradientController(model, subspaces)
    controller.project_gradients()
    
    # 6. Measure Interference AFTER Projection
    G_after = model.test_layer.moe_adapter.gate.weight.grad
    interference_after = torch.norm(torch.matmul(G_after, U)).item()
    
    # The dot product MUST be effectively zero (accounting for floating point precision)
    assert interference_after < 1e-5, f"Orthogonality failed! Interference: {interference_after}"
    print(f"✅ Orthogonality mathematically verified. (Interference = {interference_after:.8f})")
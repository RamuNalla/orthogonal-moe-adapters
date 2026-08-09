import torch.nn as nn
from src.architecture.moe_adapter import MoEAdapterLayer

class AdapterInjectedLinear(nn.Module):
    """
    Wraps an existing frozen nn.Linear layer and runs the MoE Adapter in parallel.
    Final Output = Frozen_Output + (Scaling_Factor * Adapter_Output)
    """
    def __init__(self, original_layer: nn.Linear, bottleneck_dim: int, num_experts: int, top_k: int, scaling: float = 1.0):
        super().__init__()
        self.original_layer = original_layer
        
        # Freeze the original base model layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
        hidden_dim = original_layer.out_features  # adapter lives in output space
        
        # Attach the MoE Adapter
        self.moe_adapter = MoEAdapterLayer(
            hidden_dim=hidden_dim, 
            bottleneck_dim=bottleneck_dim, 
            num_experts=num_experts, 
            top_k=top_k
        )
        self.scaling = scaling

    def forward(self, x):
        # 1. Base model forward pass (FROZEN)
        base_output = self.original_layer(x)
        
        # 2. Adapter forward pass (TRAINABLE)
        # Adapter runs on base_output so both tensors share the same out_features dim
        adapter_output = self.moe_adapter(base_output)
        
        # 3. Combine
        return base_output + (self.scaling * adapter_output)


def inject_moe_adapters(model: nn.Module, target_module_name: str, bottleneck_dim: int = 32, num_experts: int = 4, top_k: int = 2):
    """
    Recursively traverses the model and replaces target linear layers with AdapterInjectedLinear.
    """
    injected_count = 0
    
    for name, module in model.named_children():
        # If we find the target module (e.g., 'down_proj' in Llama/Phi-3 FFN)
        if target_module_name in name and isinstance(module, nn.Linear):
            # Create the injected wrapper
            injected_module = AdapterInjectedLinear(
                original_layer=module,
                bottleneck_dim=bottleneck_dim,
                num_experts=num_experts,
                top_k=top_k
            )
            # Replace the old module with the new wrapper
            setattr(model, name, injected_module)
            injected_count += 1
        else:
            # Recursively apply to child modules
            injected_count += inject_moe_adapters(module, target_module_name, bottleneck_dim, num_experts, top_k)
            
    return injected_count
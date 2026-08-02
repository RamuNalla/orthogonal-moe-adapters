import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckExpert(nn.Module):
    """A single Parameter-Efficient Bottleneck Adapter (Expert)."""
    def __init__(self, hidden_dim, bottleneck_dim):
        super().__init__()
        # Down-projection to a smaller dimension (compression)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        # Non-linear activation
        self.act = nn.SiLU()
        # Up-projection back to original hidden dimension
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        
        # Initialize up_proj with zeros so the initial adapter output is 0.
        # This ensures the model starts with exactly its original behavior.
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        return self.up_proj(self.act(self.down_proj(x)))

class MoEAdapterLayer(nn.Module):
    """Mixture-of-Experts Adapter Layer with Top-K Routing."""
    def __init__(self, hidden_dim, bottleneck_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # The Router Gate: Maps hidden state to expert probabilities
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # The Pool of Experts
        self.experts = nn.ModuleList([
            BottleneckExpert(hidden_dim, bottleneck_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. Calculate Routing Probabilities
        # Flatten batch and seq_len for routing: (batch*seq_len, hidden_dim)
        x_flat = x.view(-1, hidden_dim) 
        gate_logits = self.gate(x_flat)
        
        # 2. Top-K Routing
        routing_weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 3. Compute Expert Outputs
        final_output = torch.zeros_like(x_flat)
        
        # Iterate over the Top-K selections
        for i in range(self.top_k):
            expert_indices = selected_experts[:, i]
            expert_weights = routing_weights[:, i].unsqueeze(-1)
            
            # For each unique expert selected in this Top-K slot
            for expert_idx in range(self.num_experts):
                # Find which tokens were routed to this specific expert
                token_mask = (expert_indices == expert_idx)
                
                if token_mask.any():
                    expert = self.experts[expert_idx]
                    # Pass only the selected tokens through the expert
                    expert_input = x_flat[token_mask]
                    expert_output = expert(expert_input)
                    
                    # Scale by routing weight and add to final output
                    scaled_output = expert_output * expert_weights[token_mask]
                    final_output[token_mask] += scaled_output
                    
        # Reshape back to original dimensions
        return final_output.view(batch_size, seq_len, hidden_dim)
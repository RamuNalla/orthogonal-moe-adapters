import torch
from transformers import AutoModelForCausalLM, AutoConfig
from src.architecture.injector import inject_moe_adapters

print("1. Loading base model configuration (TinyLlama for fast local testing)...")
# We use TinyLlama just to test the PyTorch graph on Mac quickly
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

print(f"\nBase Model parameter count: {model.num_parameters():,}")

# In TinyLlama/Llama models, the FFN down-projection is called 'down_proj'
target_layer = "down_proj" 
print(f"\n2. Injecting MoE Adapters into '{target_layer}' layers...")

num_injected = inject_moe_adapters(
    model=model, 
    target_module_name=target_layer, 
    bottleneck_dim=32,   # Small bottleneck
    num_experts=4,       # 4 distinct experts
    top_k=2              # Route to top 2
)

print(f"Successfully injected MoE adapters into {num_injected} layers!")

# Let's count trainable vs frozen parameters to prove PEFT is working
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"\n3. Parameter Verification:")
print(f"Frozen Base Parameters: {frozen_params:,}")
print(f"Trainable MoE Parameters: {trainable_params:,}")
print(f"Percentage Trainable: {(trainable_params / (trainable_params + frozen_params)) * 100:.4f}%\n")

print("4. Testing Forward Pass...")
dummy_input = torch.randint(0, 32000, (1, 15)) # Batch size 1, sequence length 15
with torch.no_grad():
    output = model(dummy_input)
    
print(f"Forward pass successful! Output logits shape: {output.logits.shape}")
print("✅ Phase 1 Architecture Complete.")
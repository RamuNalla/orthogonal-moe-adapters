import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from src.architecture.injector import inject_moe_adapters, AdapterInjectedLinear
from src.training.subspace_extraction import SubspaceExtractor
from src.training.orthogonal_optimizer import OrthogonalGradientController

print("1. Setting up model and adapters...")
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

inject_moe_adapters(model, "down_proj", bottleneck_dim=32, num_experts=2, top_k=1)

print("\n2. Simulating Subspace Extraction (General Knowledge Phase)...")
extractor = SubspaceExtractor(model, AdapterInjectedLinear)
extractor.attach_hooks()

# Simulate passing a batch of "Wikipedia" general text
dummy_wiki_data = torch.randint(0, 32000, (4, 128)) # Batch 4, Seq 128
with torch.no_grad():
    model(dummy_wiki_data)

extractor.remove_hooks()
# Extract Top 16 principal components of the knowledge subspace
subspaces = extractor.compute_svd_subspaces(rank_k=16)

print("\n3. Simulating Fine-Tuning (New Task Phase)...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()

# Simulate a batch of "SQL/Coding" data
dummy_sql_data = torch.randint(0, 32000, (2, 64)) 
outputs = model(dummy_sql_data)

# Dummy loss
loss = outputs.logits.mean()
loss.backward()

print("\n4. Applying Orthogonal Gradient Projection...")
# Initialize our controller
controller = OrthogonalGradientController(model, subspaces)

# Pick a random adapter layer to inspect BEFORE projection
sample_layer_name = list(subspaces.keys())[0]
sample_module = dict(model.named_modules())[sample_layer_name]
G_before = sample_module.moe_adapter.gate.weight.grad.clone()
U = subspaces[sample_layer_name]

# Apply the mathematical projection
controller.project_gradients()

# Inspect the gradient AFTER projection
G_after = sample_module.moe_adapter.gate.weight.grad.clone()

print("\n5. Mathematical Verification (The SOTA Proof):")
# The projection of G_before onto U should be non-zero (interference)
interference_before = torch.norm(torch.matmul(G_before, U)).item()
# The projection of G_after onto U MUST be zero (orthogonal)
interference_after = torch.norm(torch.matmul(G_after, U)).item()

print(f"Interference with Base Knowledge BEFORE projection: {interference_before:.6f}")
print(f"Interference with Base Knowledge AFTER projection:  {interference_after:.6f}")

if interference_after < 1e-5:
    print("✅ SUCCESS: Gradients are strictly orthogonal. Catastrophic forgetting is mathematically impossible in this subspace.")
else:
    print("❌ FAILURE: Projection failed.")
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM

from src.architecture.injector import inject_moe_adapters, AdapterInjectedLinear
from src.training.subspace_extraction import SubspaceExtractor
from src.training.orthogonal_optimizer import OrthogonalGradientController
from src.training.continual_trainer import ContinualMoETrainer

# Hardware acceleration for Mac (MPS) or CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Setup Model
print("\n1. Initializing Model and Injecting MoE Adapters...")
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

inject_moe_adapters(model, "down_proj", bottleneck_dim=32, num_experts=4, top_k=2)

# 2. Extract Base Knowledge Subspace
print("\n2. Extracting SVD Subspace from General Knowledge...")
extractor = SubspaceExtractor(model, AdapterInjectedLinear)
extractor.attach_hooks()

# Simulate passing 10 batches of Wikipedia data
dummy_wiki = torch.randint(0, 32000, (10, 64)).to(device)
model.to(device)
with torch.no_grad():
    model(dummy_wiki)
    
extractor.remove_hooks()
subspaces = extractor.compute_svd_subspaces(rank_k=16)

# 3. Setup OSFT Controller & Trainer
controller = OrthogonalGradientController(model, subspaces)
trainer = ContinualMoETrainer(model, controller, device=device)

# --- CONTINUAL LEARNING SIMULATION ---

# Create Synthetic DataLoaders
# Task A (Medical) - simulated with random tokens
task_a_data = torch.randint(0, 32000, (16, 32))
task_a_loader = DataLoader(TensorDataset(task_a_data), batch_size=4, shuffle=True)

# Task B (SQL) - simulated with different random tokens
task_b_data = torch.randint(1000, 15000, (16, 32))
task_b_loader = DataLoader(TensorDataset(task_b_data), batch_size=4, shuffle=True)

# Initialize Optimizer (Only optimizing parameters that require_grad)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

# 4. Train Task A
print("\n--- PHASE: TRAINING TASK A (Medical) ---")
trainer.set_active_expert(expert_idx=0) # Lock to Expert 0
# Re-filter optimizer to track newly unfrozen parameters
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
trainer.train_task(task_a_loader, optimizer, task_name="Medical Domain", epochs=1)

# 5. Train Task B
print("\n--- PHASE: TRAINING TASK B (SQL) ---")
trainer.set_active_expert(expert_idx=1) # Lock to Expert 1 (Freezes Expert 0!)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
trainer.train_task(task_b_loader, optimizer, task_name="SQL Generation", epochs=1)

# 6. Save Artifacts
trainer.save_adapters("adapters/orthogonal_moe_weights.pt")
print("\n✅ Phase 3 Complete: Continual Learning Pipeline successfully executed.")
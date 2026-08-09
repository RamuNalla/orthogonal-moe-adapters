import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict

from src.architecture.injector import AdapterInjectedLinear
from src.training.orthogonal_optimizer import OrthogonalGradientController

class ContinualMoETrainer:
    def __init__(self, model: nn.Module, controller: OrthogonalGradientController, device: str = "cpu"):
        self.model = model
        self.controller = controller
        self.device = device
        self.model.to(self.device)

    def set_active_expert(self, expert_idx: int):
        """
        Freezes all experts EXCEPT the active one.
        The Gate remains trainable so it can learn how to route new tokens.
        """
        print(f"--- Configuring Model for Expert {expert_idx} ---")
        trainable_params = 0
        frozen_params = 0

        # Freeze the entire base model first
        for param in self.model.parameters():
            param.requires_grad = False

        for name, module in self.model.named_modules():
            if isinstance(module, AdapterInjectedLinear):
                # Unfreeze the Gate
                module.moe_adapter.gate.weight.requires_grad = True
                
                # Iterate through experts and freeze/unfreeze
                for i, expert in enumerate(module.moe_adapter.experts):
                    requires_grad = (i == expert_idx)
                    expert.down_proj.weight.requires_grad = requires_grad
                    expert.up_proj.weight.requires_grad = requires_grad

        # Count parameters to verify
        for p in self.model.parameters():
            if p.requires_grad:
                trainable_params += p.numel()
            else:
                frozen_params += p.numel()

        print(f"Expert {expert_idx} Active. Trainable Params: {trainable_params:,} | Frozen Params: {frozen_params:,}\n")

    def train_task(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, task_name: str, epochs: int = 1):
        """Standard PyTorch training loop with OSFT Gradient Projection injected."""
        self.model.train()
        print(f"🚀 Starting Training for Task: {task_name}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [{task_name}]")
            
            for batch in progress_bar:
                # Move batch to device (assuming batch is a tensor of input_ids)
                inputs = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, labels=inputs)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # THE 2026 MAGIC: Project gradients orthogonally before step
                self.controller.project_gradients()
                
                # Optimizer step
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
            print(f"Task '{task_name}' - Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}")

    def save_adapters(self, save_path: str):
        """Extracts ONLY the MoE adapter weights, saving a tiny ~50MB file instead of the whole model."""
        print(f"\n💾 Saving MoE Adapter weights to {save_path}...")
        adapter_state_dict = {}
        
        for name, param in self.model.named_parameters():
            # If the parameter belongs to our MoE adapters, save it
            if "moe_adapter" in name:
                adapter_state_dict[name] = param.cpu()
                
        torch.save(adapter_state_dict, save_path)
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"✅ Saved successfully! Artifact size: {file_size_mb:.2f} MB")
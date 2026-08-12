# Orthogonal MoE Adapters for Continual LLM Learning

> **A parameter-efficient continual learning framework solving Catastrophic Forgetting in Foundation Models.**

## 📑 Abstract
Fine-tuning Large Language Models (LLMs) on sequential tasks sequentially destroys their foundational reasoning capabilities—a phenomenon known as *Catastrophic Forgetting*. Standard PEFT techniques (like LoRA) mitigate parameter scale but fail to protect the base model's knowledge subspace.

This framework introduces **Orthogonal Subspace Fine-Tuning (OSFT) combined with Mixture-of-Experts (MoE) Adapters**. By injecting Top-K routed adapters into frozen transformer blocks and projecting their gradients strictly orthogonal to the base model's principal activations, we mathematically eliminate catastrophic forgetting. 
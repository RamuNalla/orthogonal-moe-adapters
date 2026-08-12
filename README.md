# Orthogonal MoE Adapters for Continual LLM Learning

> **A parameter-efficient continual learning framework solving Catastrophic Forgetting in Foundation Models.**

## 📑 Abstract
Every time you fine-tune an LLM on something new, it quietly forgets what it already knew. Teach it medicine, and it gets worse at reasoning. Teach it SQL next, and the medical knowledge starts slipping too. This is *Catastrophic Forgetting* — and it's the reason deploying a single model across multiple tasks remains so difficult in practice.

Most PEFT methods like LoRA reduce the number of trainable parameters, but they don't actually protect the knowledge the base model already holds. The gradients still flow freely into the same directions the pretrained model relies on.

This project takes a different approach. We first study where a pretrained model concentrates its learned representations — its principal activation directions — and then hard-constrain every adapter gradient to be strictly orthogonal to those directions. On top of that, a Mixture-of-Experts router assigns each new task its own dedicated expert, which gets frozen once training is done. The result: new skills are learned in the free space the model wasn't using, and old skills stay exactly where they were. 
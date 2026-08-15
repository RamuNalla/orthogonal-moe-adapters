# Orthogonal MoE Adapters for Continual LLM Learning

> **A parameter-efficient continual learning framework solving Catastrophic Forgetting in Foundation Models.**

## Abstract
Every time you fine-tune an LLM on something new, it quietly forgets what it already knew. Teach it medicine, and it gets worse at reasoning. Teach it SQL next, and the medical knowledge starts slipping too. This is *Catastrophic Forgetting* — and it's the reason deploying a single model across multiple tasks remains so difficult in practice.

Most PEFT methods like LoRA reduce the number of trainable parameters, but they don't actually protect the knowledge the base model already holds. The gradients still flow freely into the same directions the pretrained model relies on.

This project takes a different approach. I first study where a pretrained model concentrates its learned representations — its principal activation directions — and then hard-constrain every adapter gradient to be strictly orthogonal to those directions. On top of that, a Mixture-of-Experts router assigns each new task its own dedicated expert, which gets frozen once training is done. The result: new skills are learned in the free space the model wasn't using, and old skills stay exactly where they were. 

##  Mathematical Formulation

### 1. The Core Knowledge Subspace (SVD)
Before training, I capture the base model's core knowledge by computing the covariance matrix of its activations on a general corpus (e.g., Wikipedia). I extract the top-$k$ principal components via Singular Value Decomposition (SVD):
$$ C = \frac{1}{N} \sum_{i=1}^{N} x_i x_i^T $$
$$ C = U \Sigma V^T $$
Where $U$ represents the orthonormal basis of the foundation model's knowledge space.

### 2. Orthogonal Gradient Projection
During the fine-tuning backward pass, standard gradients $G$ will inherently push the model away from $U$. I intercept the optimizer step and project the adapter gradients $G$ into the null space of $U$:
$$ G_{\perp} = G - G U U^T $$
This ensures that $\nabla W \cdot U = 0$. The new task learns strictly in the orthogonal subspace, making interference mathematically impossible.
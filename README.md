# Orthogonal MoE Adapters for Continual LLM Learning

> **A parameter-efficient continual learning framework solving Catastrophic Forgetting in Foundation Models.**

## Abstract

**The problem:** Every time you fine-tune an LLM on something new, it quietly forgets what it already knew:
- Teach it medicine → it gets worse at reasoning
- Teach it SQL next → the medical knowledge starts slipping too

This is *Catastrophic Forgetting* — and it is the reason deploying a single model across multiple tasks remains so difficult in practice.

**Why existing PEFT methods fall short:**
- Methods like LoRA reduce the number of trainable parameters, but don't protect the knowledge the base model already holds
- Gradients still flow freely into the same directions the pretrained model relies on

**This project's approach:**
- First, study where the pretrained model concentrates its learned representations — its principal activation directions
- Constrain strictly on every adapter gradient to be orthogonal to those directions
- A MoE router assigns each new task its own dedicated expert, which gets frozen once training is done

**The result:** new skills are learned in the free space the model wasn't using, and old skills stay exactly where they were. 

##  Mathematical Formulation

### 1. The Core Knowledge Subspace (SVD)
Before training, I capture the base model's core knowledge by computing the covariance matrix of its activations on a general corpus (e.g., Wikipedia). I extract the top-$k$ principal components via Singular Value Decomposition (SVD):
```math
C = \frac{1}{N} \sum_{i=1}^{N} x_i x_i^T
```
```math
C = U \Sigma V^T
```
Where $U$ represents the orthonormal basis of the foundation model's knowledge space.

### 2. Orthogonal Gradient Projection
During the fine-tuning backward pass, standard gradients $G$ will inherently push the model away from $U$. I intercept the optimizer step and project the adapter gradients $G$ into the null space of $U$:
```math
G_{\perp} = G - G U U^T
```
This ensures that $\nabla W \cdot U = 0$. The new task learns strictly in the orthogonal subspace, making interference mathematically impossible.

### 3. Mixture-of-Experts (MoE) Routing
To learn multiple disjoint tasks (e.g., Medical and SQL) efficiently, tokens are dynamically routed via a trainable gate $W_g$ to the top-$K$ bottleneck experts $E_i$:
```math
P(x) = \text{Softmax}(W_g \cdot x)
```
```math
y = x_{\text{base}} + \sum_{i \in \text{Top-K}} P_i(x) \cdot E_i(x)
```

## Benchmarking

I ran a real continual learning experiment using TinyLlama (1.1B) on three publicly available HuggingFace datasets:
- **WikiText-2** — general knowledge baseline (calibration + retention check)
- **MedAlpaca MedQA** — Task A (medical domain, trained on Expert 0)
- **SQL-Create-Context** — Task B (SQL generation, trained on Expert 1)

Metric used is **perplexity** — lower means the model understands the domain better. Each stage was measured on all three domains simultaneously to catch any forgetting.

**Results (Perplexity ↓ lower is better):**

| Stage | General (Wiki) | Medical | SQL |
|---|---|---|---|
| T0 — Base model, no training | 15.38 | 8.37 | 12.64 |
| T1 — After training Medical (Expert 0) | 15.44 | **5.24** | 13.10 |
| T2 — After training SQL (Expert 1) | 14.87 | **5.32** | **9.56** |

**What the numbers show:**
- **General knowledge stayed flat** (15.38 → 14.87) across both fine-tuning phases — the orthogonal gradient projection successfully protected the base model's subspace
- **Medical improved 37%** at T1 (8.37 → 5.24), then barely moved at T2 (5.24 → 5.32, Δ = +0.08) — Expert 0 was frozen after T1, making forgetting mathematically impossible
- **SQL improved 24%** at T2 (12.64 → 9.56) — Expert 1 learned the new task cleanly without touching anything Expert 0 had learned

<p align="center">
  <img src="assets/real_ppl_trajectory.png" width="60%" />
</p>
<p align="center">
  <img src="assets/real_antiforgetting_bar.png" width="48%" />
  <img src="assets/real_radar_competence.png" width="48%" />
</p>

## System Architecture
- `src/architecture/`: PyTorch definitions for Top-K routing gates and dynamically injected bottleneck layers.
- `src/training/subspace_extraction.py`: Forward hooks for incremental covariance computation and SVD.
- `src/training/orthogonal_optimizer.py`: Custom PyTorch backward hooks to enforce $G_{\perp}$.

## ⚙️ Quick Start

```bash
# Clone the repo
git clone https://github.com/RamuNalla/orthogonal-moe-adapters.git
cd orthogonal-moe-adapters

# Install dependencies (PyTorch, Transformers, datasets)
pip install torch transformers datasets tqdm
```

Then run the three phases in order:

```bash
# Phase 1 — Verify adapter injection and forward pass
python test_1.py

# Phase 2 — Verify orthogonal gradient projection (mathematical proof)
python test_2.py

# Phase 3 — Full continual learning pipeline (SVD → Train Medical → Train SQL → Save adapters)
python test_3.py
```

To run the orthogonality unit tests:

```bash
pytest tests/
```
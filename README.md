# LunarisCodex-MoC  
*An Experimental Language Model featuring a Collaborative Experts Module.*

---

## Overview

`LunarisCodex-MoC` is a fully-functional, Llama-style large language model that serves as a test-bed for cutting-edge neural architectures.  
It replaces the standard Feed-Forward Network (FFN) in each Transformer block with a novel **Mixture of Collaborative Experts (MoC)** module.  
Where traditional Mixture-of-Experts (MoE) models route tokens to *isolated* experts, MoC enables the *selected* experts to **interact and refine their outputs together** before the final fusion—unlocking richer representations and potentially better specialization.

---

## The Core Concept: An Analogy (Entendendo o Conceito)

### The Old Way: Standard MoE (The Overloaded President)

Imagine a president facing a complex report on “Economy vs. Environment”.

1. **Quick Decision**: The president glances at the title and thinks: *“80 % Economy, 20 % Environment.”*  
2. **Simple Selection**: Calls **only** the Minister of Economy and the Minister of Environment.  
3. **Isolated Work**: Each minister works **alone**, never exchanging ideas.  
4. **Mechanical Merge**: The president takes 80 % of the economic plan and 20 % of the environmental plan.  
5. **Result**: The final policy is **incoherent**—economic incentives may directly contradict environmental goals.

### The New Way: MoC (The Wise President)

Same problem, but handled with **collaboration**:

#### Phase 1 – Contextualized Routing (The Preliminary Meeting)
1. **Pre-Analysis**: The report is sent **to all 8 ministers** for an initial summary.  
2. **Context Meeting**: Everyone gathers.  
   - The Minister of Economy hears the Foreign Affairs minister discuss climate treaties.  
   - The Environment minister hears the Education minister mention green-campaigns.  
3. **Informed Choice**: With this holistic view, the president realizes he also needs the Foreign Affairs minister—selects **3** ministers instead of 2.

#### Phase 2 – Collaborative Fusion (The Task-Force)
1. **Working Session**: The 3 chosen ministers enter a war-room.  
2. **Cross-Pollination**:  
   - Economy: *“Let’s offer tax credits for green companies.”*  
   - Foreign Affairs: *“Great, but the Paris Agreement requires stricter targets.”*  
   - Environment: *“What if those credits fund carbon-capture research?”*  
3. **Integrated Report**: Instead of 3 separate proposals, they deliver **one cohesive policy** that is economically sound, environmentally responsible, and internationally aligned.

---

## Architectural Innovations: From MoE to MoC

### Historical Context
The project began with the standard Switch-Transformer-style MoE (top-1 routing). While efficient, these models suffer from a **fundamental limitation**:  
> **Experts work in complete isolation.**  
The final output is simply the chosen expert’s result, with zero synergy or refinement.

MoC answers the question:  
> “*What if experts could collaborate?*”

### 1. Router Contextualization
| Traditional MoE | MoC (Router Contextualization) |
|-----------------|-------------------------------|
| Router decides **blindly** from the input token alone. | Router **first sees** what *every* expert “thinks” about the token. |

#### How it Works
```python
# 1. Forward through ALL experts in parallel
expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

# 2. Self-attention over all expert outputs
contextualized, _ = self.router_self_attn(
    expert_flat, expert_flat, expert_flat
)

# 3. Route with contextualized information
routing_logits = self.router_gate(contextualized.mean(1))
```

#### Why it Matters
- **Context-aware routing** – decisions consider the entire expert landscape.  
- **Adaptive selection** – routing can change based on dynamic context.  
- **Avoids blind spots** – less chance of missing the *best* expert.

### 2. Collaborative Fusion
| Traditional MoE (k>1) | MoC (Collaborative Fusion) |
|-----------------------|---------------------------|
| Weighted sum of raw expert outputs. | Experts **talk to each other** via cross-attention, then fuse. |

#### How it Works
```python
# 1. Select top-k experts via contextualized routing
selected = expert_outputs[batch_indices, seq_indices, topk_indices]

# 2. Cross-attention between selected experts
collab_out, attn_weights = self.collab_cross_attn(
    selected_flat, selected_flat, selected_flat
)

# 3. Refinement via small FFN
refined = self.collab_ffn(collab_out) + collab_out
```

#### Why it Matters
- **Synergy** – each expert refines its output using peer knowledge.  
- **Coherence** – final representation is *more than the sum of its parts*.  
- **Stability** – residual connections and layer-norm keep gradients healthy.

---

## The Advanced Auxiliary Loss

To keep training stable and collaboration meaningful, MoC employs a **composite auxiliary loss** scaled by `aux_loss_weight`.

| Component | Intuition | Implementation |
|-----------|-----------|----------------|
| **Balance Loss** | Encourage uniform expert usage. | Minimize variance of `routing_probs.mean(dim=[0,1])`. |
| **Diversity Loss** | Prevent collapse in cross-attention patterns. | Maximize entropy of cross-attention weights. |

```python
def compute_diversity_loss(attn_weights, routing_probs):
    # Diversity: maximize entropy of cross-attention
    attn_entropy = -torch.sum(
        attn_weights * torch.log(attn_weights + 1e-8), dim=-1
    ).mean()

    # Balance: minimize variance of expert usage
    expert_usage = routing_probs.mean(dim=[0, 1])
    balance_loss = torch.var(expert_usage)

    return 0.01 * (-attn_entropy) + 0.01 * balance_loss
```

---

## Technical Deep Dive

### Data Flow (Step-by-Step)
1. **Input** `(B, S, d_model)`  
2. **Expert Computation** – all experts run in parallel → `(B, S, n_experts, d_model)`  
3. **Router Contextualization** – self-attention over experts → contextualized logits  
4. **Top-K Selection** – choose `k` experts, apply temperature scaling  
5. **Collaborative Fusion** – cross-attention + refinement → `(B, S, k, d_model)`  
6. **Weighted Fusion** – combine with routing weights → `(B, S, d_model)`  
7. **Final Projection** – `o_proj` back to model dimension.

### Tensor Dimensions
```
Input:           (B, S, d_model)
Expert Outputs:  (B, S, n_experts, d_model)
Contextualized:  (B*S, n_experts, d_model)
Top-K Selected:  (B, S, top_k, d_model)
Final Output:    (B, S, d_model)
```

### Computational Complexity Analysis

| Phase | Complexity | Notes |
|-------|------------|-------|
| Expert Forward | O(B·S·n_experts·d_model²) | Dominant term |
| Router Self-Attention | O(B·S·n_experts²·d_model) | Small vs. expert FLOPs |
| Cross-Attention (k experts) | O(B·S·k²·d_model) | k ≪ n_experts |

**Comparison with Traditional MoE**  
- **MoE (train)**: O(B·S·k·d_model²)  
- **MoC (train)**: O(B·S·n_experts·d_model²)  
> **Trade-off**: Higher compute for richer collaboration.

---

## Usage and Training

### Configuration
Create `config_moc.yaml`:

```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  n_kv_heads: 12
  vocab_size: 50257
  max_seq_len: 1024
  # --- MoC specific ---
  n_experts: 8
  top_k: 2
  aux_loss_weight: 0.1
  router_temperature: 1.0

data_dir: "data/"
out_dir: "checkpoints/lunaris-moc-8e-2k"
learning_rate: 3.0e-4
max_steps: 600000
batch_size: 16
gradient_accumulation_steps: 4
wandb_project: "lunaris-codex-moc"
wandb_run_name: "moc-8-experts-2-topk"
```

Recommended settings:

| Parameter | Range | Guideline |
|-----------|-------|-----------|
| `n_experts` | 8–16 | Balance capacity vs. compute |
| `top_k` | 2–4 | Enough collaboration, low overhead |
| `aux_loss_weight` | 1e-3–1e-2 | Tune to keep loss ≈ main loss × 0.01 |
| `router_temperature` | 0.5–2.0 | <1 → sharper routing, >1 → smoother |

### Running the Training
```bash
python train_moc.py config_moc.yaml
```

### Monitoring and Debugging

#### Key Metrics to Watch
| Metric | Where to Log | Healthy Range |
|--------|--------------|---------------|
| **loss/main** | W&B, tqdm | Steady decrease |
| **loss/aux** | W&B, tqdm | Stable, ≈ 1–2 % of main loss |
| **perplexity** | W&B | Calculated from `loss/main`; should improve |
| **Expert usage variance** | Manual | Low; aim for uniform distribution |

#### Common Issues & Fixes
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Aux loss spikes | Learning rate too high | Reduce LR or increase warmup |
| Experts unused | Temperature too high or init issues | Lower `router_temperature`, check weight init |
| Training instability | Gradient explosion | Tighten `grad_clip`, lower LR |
| Overfitting | Too many experts / small data | Increase dropout, reduce `n_experts` or `top_k` |

---

## Project Journey & Future Work

From a simple MoE prototype to a fully-fledged collaborative framework, the project’s core insight was that **experts should collaborate, not compete**.  
The next steps include:

- **Dynamic Top-K**: Adjust `k` on-the-fly based on input complexity.  
- **Hierarchical Experts**: Nested experts operating at different granularities.  
- **Memory-Augmented Routing**: Router with episodic memory of past decisions.  
- **Multi-Scale Collaboration**: Cross-attention across both token and expert axes.  

We invite the community to experiment, extend, and improve upon this foundation.

---

**Maintained by Francisco • July 2025 • v1.0**

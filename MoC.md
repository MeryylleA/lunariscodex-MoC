# Mixture of Collaborative Experts (MoC)

An advanced Mixture-of-Experts architecture with collaboration between specialists

---

## Overview

The Mixture of Collaborative Experts (MoC) is an evolution of the traditional Mixture of Experts (MoE) that introduces intelligent collaboration among the selected experts. Instead of simply combining expert outputs independently, MoC lets experts collaborate via cross-attention before the final fusion.

### Fundamental Differences from Traditional MoE

| Aspect | Traditional MoE | MoC (Our Implementation) |
|--------|------------------|--------------------------|
| Expert Selection | Simple router with softmax | Contextualized router with self-attention |
| Collaboration | None — experts are independent | Cross-attention between selected experts |
| Auxiliary Loss | Basic load balancing | Combined Diversity + Balance |
| Stability | Renormalization issues | Direct softmax on logits |

---

## Architecture Details

### Main Components

#### 1. Router Contextualization

```python
# Self-attention over ALL expert outputs
contextualized_experts, _ = self.router_self_attn(
    expert_flat, expert_flat, expert_flat
)
```

Why this matters:
- The router doesn’t decide based only on the input token
- It considers all experts’ outputs to make a more informed selection
- Enables adaptive routing based on full context

#### 2. Top-K Selection with Temperature

```python
# Temperature scaling to control distribution sharpness
routing_logits = routing_logits / self.router_temperature

# Stable top-k selection
topk_logits, topk_indices = torch.topk(routing_logits, self.top_k, dim=-1)
topk_probs = F.softmax(topk_logits, dim=-1)
```

Advantages:
- Numerically more stable than manual renormalization
- Temperature controls distribution sharpness
- Avoids division-by-zero issues

#### 3. Collaborative Fusion

```python
# Cross-attention between selected experts
collaborative_outputs, collab_attn = self.collab_cross_attn(
    selected_flat, selected_flat, selected_flat
)

# Additional refinement
refined_outputs = self.collab_ffn(collaborative_outputs) + collaborative_outputs
```

What’s different:
- Selected experts “talk” via cross-attention
- Additional FFN refinement to improve collaboration
- Residual connections for training stability

#### 4. Advanced Auxiliary Loss

```python
def compute_diversity_loss(self, cross_attn_weights, routing_probs):
    # Diversity: maximize cross-attention entropy
    attn_entropy = -torch.sum(
        cross_attn_weights * torch.log(cross_attn_weights + 1e-8), dim=-1
    ).mean()
    diversity_loss = -attn_entropy
    
    # Balance: minimize variance of expert usage
    expert_usage = routing_probs.mean(dim=[0, 1])
    balance_loss = torch.var(expert_usage)
    
    return 0.01 * diversity_loss + 0.01 * balance_loss
```

---

## Processing Flow

### Step-by-Step

1. Input Processing
   - Receives tokens: (batch_size, seq_len, d_model)
   - Processes through ALL experts in parallel

2. Router Contextualization
   - Self-attention over all expert outputs
   - Generates a contextualized representation for routing

3. Expert Selection
   - Applies temperature scaling to routing logits
   - Selects top-k experts via torch.topk()
   - Computes probabilities with stable softmax

4. Collaborative Fusion
   - Cross-attention among selected experts
   - Refinement via an additional FFN
   - Residual connections for stability

5. Final Output
   - Weighted combination of collaborative outputs
   - Final projection + auxiliary loss

### Tensor Dimensions

```
Input: (B, S, d_model)
Expert Outputs: (B, S, n_experts, d_model)
Contextualized: (B*S, n_experts, d_model)
Top-K Selection: (B, S, top_k, d_model)
Final Output: (B, S, d_model)
```

---

## Configuration

### Key Parameters

```python
@dataclass
class LunarisCodexConfig:
    # MoC Specific Parameters
    n_experts: int = 8                    # Total number of experts
    top_k: int = 2                        # How many experts to select
    aux_loss_weight: float = 1e-2         # Auxiliary loss weight
    router_temperature: float = 1.0       # Routing temperature
```

### Configuration Recommendations

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| n_experts | 8–16 | Balance capacity and efficiency |
| top_k | 2–4 | Enables collaboration without excessive overhead |
| aux_loss_weight | 1e-2 to 1e-3 | Adequate for regularization |
| router_temperature | 0.5–2.0 | 1.0 = neutral, <1.0 = sharper, >1.0 = smoother |

---

## Mathematical Analysis

### Computational Complexity

Forward Pass:
- Expert computation: O(B × S × n_experts × d_model²)
- Router self-attention: O(B × S × n_experts² × d_model)
- Cross-attention: O(B × S × top_k² × d_model)
- Total: O(B × S × n_experts × d_model²) (dominant)

Comparison with traditional MoE:
- MoE: O(B × S × top_k × d_model²)
- MoC: O(B × S × n_experts × d_model²) (during training)
- Trade-off: Higher computational cost for better quality

### Auxiliary Loss Breakdown

Diversity Loss: Encourages diverse cross-attention patterns
- Prevents expert collapse
- Maximizes attention weight entropy

Balance Loss: Ensures balanced expert usage
- Minimizes variance of expert usage
- Prevents experts from being ignored

---

## Advantages of MoC

### 1. Better Specialization
- Experts collaborate instead of competing
- Each expert can focus on specific aspects
- Intelligent combination of knowledge

### 2. Contextualized Routing
- More informed routing decisions
- Considers outputs from all experts
- Adaptive to the current context

### 3. Training Stability
- Well-balanced auxiliary loss
- Numerically stable softmax
- Residual connections preserve gradients

### 4. Flexibility
- Temperature allows fine-tuning
- Configurable for different tasks
- Scales to more experts

---

## Implementation

### Transformer Integration

```python
class Block(nn.Module):
    def __init__(self, config):
        # ... attention layers ...
        
        if config.n_experts is not None and config.n_experts > 0:
            self.feed_forward = CollaborativeExpertsModule(config)
            self.is_moe = True
        else:
            self.feed_forward = FeedForward(config)
            self.is_moe = False
```

### Training Loop Considerations

```python
# During forward pass
logits, loss, past_key_values = model(idx, targets=targets)

if loss is not None:
    total_loss, main_loss, aux_loss = loss
    # total_loss already includes the weighted auxiliary loss
    total_loss.backward()
```

---

## Monitoring and Debug

### Important Metrics

Expert Usage Distribution
```python
expert_usage = routing_probs.mean(dim=[0, 1])
print(f"Expert usage variance: {torch.var(expert_usage):.4f}")
```

Auxiliary Loss Components
```python
print(f"Diversity loss: {diversity_loss:.4f}")
print(f"Balance loss: {balance_loss:.4f}")
```

Routing Entropy
```python
routing_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1).mean()
print(f"Routing entropy: {routing_entropy:.4f}")
```

---

## Use Cases

### When to Use MoC

✅ Ideal for:
- Tasks requiring different kinds of reasoning
- Large models where efficiency matters
- Diverse datasets
- When interpretability is desired

❌ Avoid when:
- Very small models (overhead may not pay off)
- Highly specific/homogeneous tasks
- Severely limited compute resources
- Rapid prototyping (use standard FFN first)

---

## Experiments and Tuning

### Suggested Hyperparameter Sweep

```python
# Configurations to test
configs = [
    {"n_experts": 8, "top_k": 2, "router_temperature": 1.0},
    {"n_experts": 8, "top_k": 3, "router_temperature": 0.7},
    {"n_experts": 16, "top_k": 4, "router_temperature": 1.2},
]
```

### Ablation Studies
- Without Router Contextualization: Remove router self-attention
- Without Collaborative Fusion: Remove cross-attention among experts
- Auxiliary Loss Components: Test diversity vs. balance separately

---

## References and Inspirations

### Related Papers
- Switch Transformer: Foundation of modern MoE
- GLaM: Scaling MoE to massive models
- Expert Choice: Routing improvements

### Differences in Our Implementation
- Router contextualization with self-attention
- Collaborative fusion via cross-attention
- Combined auxiliary loss (diversity + balance)
- Clean integration with a Llama-style architecture

---

## Troubleshooting

### Common Issues

Auxiliary Loss Too High
- Reduce aux_loss_weight
- Check balance between diversity and balance components

Experts Not Being Used
- Increase router_temperature
- Verify weight initialization

Training Instability
- Reduce learning rate
- Check gradient clipping

Overfitting
- Increase dropout
- Reduce number of experts or top_k

---

## Ideas for Future Extensions

- Dynamic Top-K: Adjust top_k based on context
- Hierarchical Experts: Experts specialized at multiple levels
- Memory-Augmented Routing: Router with memory of past decisions
- Multi-Scale Collaboration: Cross-attention at different scales

---

Created by: Francisco  
Date: July 2025  
Version: 1.0

> "The idea is simple: instead of experts competing, they collaborate. This collaboration happens through cross-attention, allowing each expert to refine its output based on what the other experts are ‘thinking’."

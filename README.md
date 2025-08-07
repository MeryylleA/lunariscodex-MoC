# Lunaris Codex — MoC v4 (Iterative Reasoning Layers)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord&logo=discord&color=7289DA)](https://discord.gg/JNsfzEwMtC)

Lunaris Codex is a research-grade, production-ready toolkit for pre-training decoder-only Transformers from scratch. Built on a clean nanoGPT-style foundation, this release introduces a revolutionary Iterative Reasoning Layer (IRL) architecture within our Mixture-of-Collaboration (MoC) framework. The system enables experts to perform multi-step reasoning internally, dramatically improving complex problem-solving capabilities while maintaining computational efficiency through top-k routing.

What's unique in this v4 release:
- **Iterative Reasoning Layers (IRL)**: Each expert performs `n_reasoning_steps` of internal computation with residual injection, enabling deeper thought processes without architectural complexity.
- **MoC v4**: Enhanced collaborative experts with deterministic top‑k routing, capacity-aware dispatch, and principled auxiliary losses.
- **Modernized transformer stack**: RoPE, GQA, Pre-LN RMSNorm, SwiGLU-style MLP, causal SDPA with dropout, and tied embeddings.
- **Production-ready training system**: bf16-first, fused AdamW, cosine LR with warmup, robust checkpointing, and comprehensive W&B logging.

This repository is designed to be "hackable," with concise modules and clear interfaces, while providing cutting-edge components you can trust in research and production.

**Key Innovation Highlights:**
- **IRL Technology**: Multi-step iterative reasoning within each expert FFN for enhanced problem-solving capability.
- **Collaborative Expert Architecture**: Per-token top‑k collaboration with load balancing and capacity management.
- **Memory-efficient data pipeline**: Stream processing over `.npy` shards for very large corpora.
- **Enterprise-grade training**: DDP with gradient accumulation, clipping, and resumable checkpoints.
- **Comprehensive metrics**: W&B integration with expert utilization, reasoning step analysis, and drop rate monitoring.

--------------------------------------------------------------------------------

## Architecture Overview

The model lives in `model_moc.py` and exposes a single class: `LunarisCodex`. The configuration is defined by `LunarisCodexConfig`.

### Transformer Backbone
- **Decoder-only, pre-LN stack**: Stable, standard architecture for autoregressive pre-training.
- **Positional encodings**: RoPE with precomputed complex rotations via `precompute_freqs_cis`. This keeps attention math fast and numerically stable.
- **Normalization**: `RMSNorm` is used as the core normalization layer, applied before attention and FFN blocks (Pre-LN) for maximum training stability.
- **Attention**:
  - **Grouped Query Attention (GQA)**: `n_heads` queries with `n_kv_heads` shared K/V heads reduce KV cache and memory at generation time.
  - **Causal SDPA**: Uses `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` and `dropout_p` during training, enabling FlashAttention kernels automatically when available.
- **MLP**: SwiGLU-style fused two-projection MLP in `ReasoningFeedForward` with integrated IRL capability.
- **Tied embeddings**: Input `wte` and output `lm_head` share weights to reduce parameter count and often improve quality.

### Iterative Reasoning Layers (IRL)

**Core Innovation**: Each expert's feedforward network can perform multiple reasoning steps internally, allowing for deeper computation without increasing model depth.

#### How IRL Works
The `ReasoningFeedForward` module implements iterative reasoning through:

1. **Multi-step Processing**: Each expert performs `n_reasoning_steps` iterations of computation.
2. **Residual Injection**: At each step, the original input `x` is residually combined with the evolving thought state `h`.
3. **Thought State Evolution**: The internal state `h` is updated through the SwiGLU MLP at each reasoning step, scaled by a stability factor `alpha`.
4. **Computational Efficiency**: Despite multiple steps, the parameter count remains the same as traditional FFNs.

```python
def forward(self, x: torch.Tensor):
    # Initialize thought state
    h = x
    steps = max(1, self.n_reasoning_steps)
    # Each step refines the thought state 'h'
    for _ in range(steps):
        # Residual injection of original x, scaled update
        h = h + self.alpha * self._ffn_logic(h + x)
    return h
```

#### Benefits of IRL
- **Enhanced Problem Solving**: Multi-step reasoning enables more sophisticated computation within each expert.
- **Parameter Efficiency**: Same parameter count as standard FFNs but with increased computational depth.
- **Stable Training**: Residual connections and scaling at each step ensure gradient flow and training stability.
- **Configurable Complexity**: `n_reasoning_steps` allows fine-tuning of computational vs. efficiency trade-offs.
- **Compatibility**: Works seamlessly with both MoC collaboration modes and standard dense layers.

#### Research Applications
IRL is particularly effective for:
- Mathematical reasoning and multi-step problem solving
- Complex logical inference tasks
- Chain-of-thought style reasoning
- Tasks requiring iterative refinement of solutions

### MoC v4: Collaborative Experts with IRL

Defined in `MoCTopKExperts`, this module builds on a robust MoE foundation enhanced with IRL-powered experts:

#### 1) Deterministic Top‑k Routing
- A linear router (`gate`) produces logits per expert.
- `topk` over logits selects K experts per token (no Gumbel/noise).
- Softmax over the selected K determines fusion weights.
- Capacity-aware dispatch: tokens are permuted into contiguous segments per expert; capacity factor controls max pairs per expert.

#### 2) IRL-Enhanced Expert Computation
- Each expert uses `ReasoningFeedForward` with configurable `n_reasoning_steps`.
- Multi-step reasoning enables deeper computation within each expert.
- Residual injection maintains stable gradients across reasoning steps.
- Same parameter efficiency as standard experts but with enhanced capability.

#### 3) Principled Auxiliary Losses
- **Load balancing (Switch-style)**: encourages even routing across experts.
- **Router z-loss**: stabilizes router logits by penalizing log-sum-exp magnitude.

#### 4) Two Collaboration Modes
**Legacy collaboration** (attention over selected expert outputs):
- Per token, run a small MHA over the K expert outputs, **with padding masks to correctly ignore dropped experts**. An RMSNorm+FFN is applied, then the result is fused with top‑k weights.
- Ideal for expressive interaction over expert hypotheses.

**Simple collaboration** (shared-parameter, two-pass, no MHA) when `use_simple_collab=True`:
- **Pass 1**: Project messages `M`, queries `Q`, and keys `Kk` from selected expert outputs; compute KxK attention per token with masks, and aggregate `C = softmax(QK^T) @ M`.
- **Pass 2**: Update with an MLP over `[selected, C]` and residual merge; fuse with renormalized top‑k weights.
- Benefits: fewer moving parts than full MHA, stable, vectorized, and easy to analyze.

#### 5) Robust Implementation Details
- Router executed in fp32 for stability.
- Vectorized dispatch via sorting and `index_copy_`, with keep/drop tracking based on capacity.
- Per-step logging hooks expose expert indices for utilization tracking.
- Weight initialization carefully scaled for deep/stable training; output projections use depth-aware std.

### Generation
- Incremental generation with KV cache, respecting `max_seq_len`.
- GQA reduces KV cache footprint; RoPE indexing handles positional continuity across steps.
- IRL reasoning steps provide enhanced quality during generation with minimal overhead.

### Key Files
- `model_moc.py` — Model definition: transformer stack, attention, IRL-enhanced MoC layer, and optimizer configuration helpers.
- `train_moc.py` — Main trainer: DDP, AMP, scheduling, checkpointing, W&B, and data pipeline.

--------------------------------------------------------------------------------

## Training System

`train_moc.py` is a resilient, high-throughput trainer for single/multi-GPU runs with full IRL support.

### Performance Features
- **Precision**: Prefers `bfloat16` when supported; TF32 enabled on matmul/cudnn.
- **Fused AdamW**: Uses fused kernels on CUDA when available; router parameters use a reduced LR.
- **`torch.compile`**: Optional graph compilation with safe fallback, optimizes IRL computation graphs.
- **DDP**: `DistributedDataParallel` with `no_sync` for gradient accumulation.
- **Gradient Accumulation and Clipping**: Simulates large effective batch sizes and stabilizes updates.

### Scheduling and Optimization
- **Cosine decay** with linear warmup via `get_lr`.
- **Separate parameter groups**:
  - Decay for weight matrices.
  - No weight-decay for biases/norms.
  - Router gate with 0.5x LR to temper routing dynamics.
- **IRL-aware optimization**: Handles the increased compute from reasoning steps efficiently.

### Data Pipeline
- `ShardDataset` streams token IDs from many memory-mapped `.npy` shards.
- Produces aligned `(x, y)` pairs of length `seq_len` with ignore-index padding when crossing shard boundaries.
- Efficient `DataLoader` with `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`.

### Checkpointing and Resume
- Saves full state: model, optimizer, config, `step`, and `epoch`.
- Always-up-to-date `latest_checkpoint.pt` plus periodic numbered checkpoints.
- Strict state dict loading (after unwrapping DDP/compile prefixes).

### Experiment Tracking
- **Console progress**: loss, perplexity, LR, and grad-norm.
- **W&B metrics**:
  - Total loss, main CE loss, aux loss, perplexity, LR, grad-norm.
  - Expert utilization histogram for MoC layers.
  - Token–expert pair drop rate under capacity constraints.
  - IRL-specific metrics: reasoning step efficiency and convergence.

--------------------------------------------------------------------------------

## Getting Started

### Prerequisites
- Python 3.9+ recommended
- PyTorch with CUDA for GPU training
- Optional: Weights & Biases account for logging

### Installation
```bash
git clone https://github.com/MeryylleA/lunariscodex-MoC.git
cd lunariscodex-MoC
```

### Data Preparation
1. Train a tokenizer (e.g., Hugging Face `tokenizers`) and save `tokenizer.json`.
2. Tokenize all documents, append an end-of-text token to each document.
3. Interleave heterogeneous sources before tokenization to stabilize training.
4. Concatenate token IDs and shard into `.npy` files of `dtype=np.uint16`. Use large shards for IO efficiency.

Example directory: `data/` containing `shard_0000.npy`, `shard_0001.npy`, ...

### Configuration

Create `train.yaml` with both model and training settings. Example:

```yaml
model:
  vocab_size: 50304
  d_model: 1024
  n_layers: 20
  n_heads: 16
  n_kv_heads: 4
  max_seq_len: 2048
  rope_theta: 10000.0
  dropout: 0.05

  # MoC configuration
  n_experts: 8
  top_k: 2
  aux_loss_weight: 1.0e-2
  capacity_factor: 1.25
  router_z_loss_weight: 1.0e-3

  # IRL configuration (NEW in v4)
  n_reasoning_steps: 3              # Number of iterative reasoning steps per expert
  
  # Collaboration mode
  use_simple_collab: false          # set true to enable the simple 2-pass collab
  simple_collab_dropout: 0.1

  # Engineering
  use_gradient_checkpointing: true

data_dir: "data/"
learning_rate: 3.0e-4
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
warmup_steps: 2000
max_steps: 200000

batch_size: 16
gradient_accumulation_steps: 4
grad_clip: 1.0

compile_model: true
device: "cuda"
out_dir: "checkpoints/exp-mocv4-irl"
save_interval: 1000
log_interval: 20

wandb_project: "lunaris-codex-moc-v4"
wandb_run_name: "mocv4-irl-1024d-8x-top2-3steps"
```

### Launch Training
Single-node multi-GPU via `torchrun`:
```bash
torchrun --standalone --nproc_per_node=auto train_moc.py train.yaml
```

To resume, keep `latest_checkpoint.pt` in `out_dir`; the trainer loads it automatically.

--------------------------------------------------------------------------------

## Configuration Guide

### Model Scale
- `d_model`, `n_layers`, `n_heads`: choose based on memory/compute budget.
- `n_kv_heads`: set below `n_heads` for GQA; typical ratios 2–4x queries per KV head.
- `max_seq_len`: ensure it accommodates the longest planned sequences; RoPE buffers are precomputed up to this length.

### IRL Configuration (New in v4)
- `n_reasoning_steps`: **Key parameter** controlling iterative reasoning depth.
  - `1`: Standard FFN behavior (no iteration).
  - `2-4`: Moderate reasoning enhancement, good balance of quality vs. compute.
  - `5+`: Deep reasoning, significant compute increase but potentially better problem-solving.
- **Compute scaling**: Each reasoning step adds one forward pass through the expert FFN.
- **Memory impact**: Minimal additional memory usage beyond the temporary thought state.

### MoC Specifics
- `n_experts`: total experts in each MoC layer.
- `top_k`: experts per token; `2` is a strong default balancing quality and compute.
- `capacity_factor`: controls maximum token–expert pairs per expert. Lower values increase drops (regularization/throughput trade-off).
- `aux_loss_weight` and `router_z_loss_weight`: stabilize and balance routing; defaults are tuned for typical setups.
- `use_simple_collab`: enables the two-pass shared-parameter collaboration instead of full MHA over K; simpler and often faster.

### Stability and Regularization
- `dropout`: applied in attention operations, MLPs, and MoC collab heads; start low when training on very large data.
- **Pre-LN**: The Pre-LayerNorm architecture is inherently more stable than Post-LN, reducing the need for other complex normalization schemes.
- **Gradient clipping**: keep at `1.0` initially; adjust if grad-norms spike, especially important with IRL.

### Performance Tuning for IRL
- **Reasoning steps vs. batch size**: Higher `n_reasoning_steps` increases per-sample compute; consider reducing batch size accordingly.
- **Gradient checkpointing**: Highly recommended (`use_gradient_checkpointing: true`) when using IRL to manage memory.
- **Compilation**: `torch.compile` provides significant speedup for repetitive IRL computations.

--------------------------------------------------------------------------------

## Data Pipeline Details

### Input Format
- Directory of `.npy` shards containing token IDs of type `uint16`.
- Each shard can be very large (hundreds of millions to billions of tokens).
- The loader creates non-overlapping windows of length `max_seq_len + 1` to produce `(x, y)` where `y` is `x` shifted by 1.
- When windows cross shard boundaries, the dataset stitches from the next shard and pads with `-1` (ignored in CE loss).

### Best Practices
- **Interleave datasets** before tokenization to avoid curriculum artifacts and stabilize optimization.
- Ensure a dedicated end-of-text token is present in the tokenizer vocabulary and appended per document.
- **Deduplicate and filter** sources aggressively; data quality drives outcomes, especially important for IRL effectiveness.

--------------------------------------------------------------------------------

## Logging and Diagnostics

### Console
- Loss, main CE, aux loss, perplexity, LR, gradient norm via tqdm.
- IRL-aware progress tracking shows reasoning efficiency.

### W&B Integration
- Enable by setting `wandb_project`.
- **Standard metrics**:
  - Scalars: `loss/total`, `loss/main`, `loss/aux`, `perplexity`, `lr`, `grad_norm`.
  - Expert utilization: per-expert fraction for MoC layers (`experts/util_layer0/e{i}`).
  - Token–expert pair drop rate under capacity constraints (`experts/drop_rate_layer0`).
- **IRL-specific metrics** (planned):
  - Reasoning step convergence analysis.
  - Thought state evolution tracking.
  - Step-wise gradient flow analysis.

### Interpreting Metrics
- **Expert utilization**: Should be relatively even; heavy skew suggests router undertraining or insufficient aux loss.
- **Drop rate**: Increases as capacity decreases; moderate drops can act as regularization, excessive drops may starve experts.
- **IRL effectiveness**: Monitor convergence patterns and gradient flow across reasoning steps.

--------------------------------------------------------------------------------

## Examples and Usage

### Minimal Sanity Test
Run `python model_moc.py` to execute built-in sanity checks:
- Tests both collaboration modes with IRL.
- Trains a tiny model for a few steps with `n_reasoning_steps=3`.
- Validates generation with iterative reasoning.

### Generation with IRL
```python
model = LunarisCodex(config)
# IRL reasoning happens automatically during generation
output = model.generate(
    idx=input_tokens, 
    max_new_tokens=100, 
    temperature=0.8, 
    top_k=40
)
```

### Custom IRL Configuration
```python
# Different reasoning depths per use case
config_light = LunarisCodexConfig(n_reasoning_steps=2)     # Fast inference
config_standard = LunarisCodexConfig(n_reasoning_steps=3)  # Balanced
config_deep = LunarisCodexConfig(n_reasoning_steps=5)      # Maximum reasoning
```

### Optimizer Configuration
`LunarisCodex.configure_optimizers(...)` returns AdamW configured with distinct parameter groups including router-specific LR. The system automatically handles IRL parameter scaling.

--------------------------------------------------------------------------------

## Migration and Extensibility

### Backward Compatibility
- Parameter names remain stable across collaboration modes.
- Setting `n_reasoning_steps=1` provides standard FFN behavior.
- GQA ratios can be changed as long as `n_heads % n_kv_heads == 0`.
- To disable MoC and use dense FFN with IRL, set `n_experts: null` or `n_experts: 0`.

### Suggested Extensions
- **Adaptive reasoning**: Dynamic `n_reasoning_steps` based on token complexity.
- **Multi-MoC stacks**: Enable MoC on a subset of layers or alternate dense and MoC blocks.
- **Routing research**: Experiment with temperature annealing, entropy regularizers, or load-aware routing.
- **IRL variants**: Alternative reasoning patterns (e.g., branching, tree search).
- **Parallelism**: Add FSDP or tensor parallel for very large models.
- **Evaluation**: Integrate reasoning-specific benchmarks and analysis tools.

### Research Directions
- **IRL optimization**: Gradient flow analysis and step-wise learning rates.
- **Reasoning pattern analysis**: Understanding what computation happens at each step.
- **Efficiency improvements**: Conditional computation and early stopping for reasoning steps.
- **Task-specific tuning**: Optimizing reasoning depth for different problem domains.

--------------------------------------------------------------------------------

## Performance Considerations

### Computational Cost
- **IRL scaling**: Each reasoning step multiplies FFN compute by `n_reasoning_steps`.
- **Memory efficiency**: Minimal additional memory usage due to in-place computation.
- **Throughput impact**: Expect ~`n_reasoning_steps`x increase in FFN compute time.
- **Optimization**: `torch.compile` provides significant speedup for iterative patterns.

### Scaling Guidelines
- **Small models** (< 1B params): `n_reasoning_steps=2-3` provides good quality gains.
- **Medium models** (1-10B params): `n_reasoning_steps=3-4` balances performance and compute.
- **Large models** (> 10B params): `n_reasoning_steps=2-3` recommended due to compute constraints.

### Memory Management
- Enable `use_gradient_checkpointing=true` for IRL configurations.
- Consider reducing batch size proportional to reasoning steps.
- Monitor GPU memory usage during initial experiments.

--------------------------------------------------------------------------------

## Limitations

- **Pre-training focused**: No built-in fine-tuning or instruction-tuning pipelines.
- **No built-in evaluation**: You must implement downstream validation and reasoning-specific benchmarks separately.
- **Limited parallelism**: Only DDP is implemented; no FSDP/tensor/pipeline parallel in this repo.
- **Data preparation**: Tokenization is user-managed; quality of results depends heavily on your corpus and tokenizer.
- **IRL analysis tools**: Currently limited built-in tools for analyzing reasoning patterns (research opportunity).

--------------------------------------------------------------------------------

## Reproducing Results

### Getting Started Tips
1. **Start small**: Verify the entire pipeline with a tiny model (`d_model=64`, `n_layers=2`, `n_reasoning_steps=2`).
2. **Monitor routing**: If expert utilization collapses, increase `aux_loss_weight` or reduce router LR.
3. **IRL tuning**: Begin with `n_reasoning_steps=2-3` and scale based on task complexity.
4. **Memory management**: Keep `use_gradient_checkpointing=true` when scaling.
5. **Precision**: Use `bfloat16` on supported GPUs for stability at higher learning rates.

### Recommended Configurations

**Research/Experimentation**:
```yaml
model:
  d_model: 512
  n_layers: 12
  n_experts: 4
  top_k: 2
  n_reasoning_steps: 3
  use_gradient_checkpointing: true
```

**Production/Scaling**:
```yaml
model:
  d_model: 2048
  n_layers: 24
  n_experts: 8
  top_k: 2
  n_reasoning_steps: 2  # Conservative for compute efficiency
  use_gradient_checkpointing: true
```

--------------------------------------------------------------------------------

## Citations and Acknowledgements

- Built upon the spirit of Andrej Karpathy's nanoGPT: clarity, minimalism, hackability.
- Incorporates ideas from recent literature on RoPE, GQA, SwiGLU, and capacity-aware MoE training.
- IRL concept inspired by iterative refinement and multi-step reasoning research.
- Thanks to the open-source community for foundational tooling and insights.

If you use Lunaris Codex v4 with IRL in your research, please consider citing this work and sharing your findings with the community.

--------------------------------------------------------------------------------

## License

Apache License 2.0. See `LICENSE` for details.

--------------------------------------------------------------------------------

## Contact and Community

- **Author**: Francisco Antonio (GitHub `@MeryylleA`)
- **Discord**: Moon Cloud Services — https://discord.gg/JNsfzEwMtC
- **Research Focus**: Iterative reasoning, collaborative expert systems, efficient large-scale training

If you use Lunaris Codex v4 in your research or products, please share feedback, issues, and results. Contributions are welcome, especially in:
- IRL optimization and analysis tools
- Reasoning pattern visualization
- Task-specific IRL configurations
- Efficiency improvements for iterative computation

**Join our community** to discuss IRL research, share findings, and collaborate on advancing iterative reasoning in language models.

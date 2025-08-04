# Lunaris Codex — MoC v2 (Top‑k Collaborative Experts)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord&logo=discord&color=7289DA)](https://discord.gg/JNsfzEwMtC)

Lunaris Codex is a research-grade, production-ready toolkit for pre-training decoder-only Transformers from scratch. Built on a clean nanoGPT-style foundation, this release introduces a modern Llama-class core with a novel Mixture-of-Collaboration (MoC) layer: top-k routing over experts with token-wise collaborative fusion. The codebase is optimized for stability, throughput, and long-running jobs. It supports memory-mapped sharded datasets, mixed precision, `torch.compile`, DDP, robust checkpointing, and W&B logging.

What’s unique in this release:
- MoC v2: deterministic top‑k routing with capacity-aware dispatch, expert collaboration (two modes), and principled auxiliary losses.
- Modernized transformer stack: RoPE, GQA, RMSNorm, SwiGLU-style MLP, QK-norm, causal SDPA, tied embeddings.
- Training system engineered for large-scale pre-training: bf16-first, fused AdamW, cosine LR with warmup, stable resumes.

This repository is designed to be “hackable,” with concise modules and clear interfaces, while providing cutting-edge components you can trust in research and production.

Highlights
- Research-ready MoE/MoC layer with per-token top‑k collaboration and load balancing.
- Clean separation of model and trainer; single-file model for fast iteration.
- Memory-efficient data pipeline over `.npy` shards for very large corpora.
- Robust DDP training with gradient accumulation, clipping, and resumable checkpoints.
- W&B metrics, including expert utilization and drop rates.

--------------------------------------------------------------------------------

Architecture Overview

The model lives in `model_moc.py` and exposes a single class: `LunarisCodex`. The config is defined by `LunarisCodexConfig`.

Transformer backbone
- Decoder-only, pre-LN stack: Stable, standard architecture for autoregressive pre-training.
- Positional encodings: RoPE with precomputed complex rotations via `precompute_freqs_cis`. This keeps attention math fast and numerically stable.
- Normalization: `RMSNorm` throughout; optionally `QK-norm` for query/key projections inside attention for improved stability at scale.
- Attention:
  - Grouped Query Attention (GQA): `n_heads` queries with `n_kv_heads` shared K/V heads reduce KV cache and memory at generation time.
  - Causal SDPA: Uses `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True`, enabling FlashAttention kernels automatically when available.
- MLP: SwiGLU-style fused two-projection MLP in `FeedForward` (`w13` chunking).
- Tied embeddings: Input `wte` and output `lm_head` share weights to reduce parameter count and often improve quality.

MoC v2: Top‑k Collaborative Experts
Defined in `MoCTopKExperts`, this module builds on a robust MoE base with several key additions:

1) Deterministic top‑k routing
- A linear router (`gate`) produces logits per expert.
- `topk` over logits selects K experts per token (no Gumbel/noise).
- Softmax over the selected K determines fusion weights.
- Capacity-aware dispatch: tokens are permuted into contiguous segments per expert; a capacity factor controls max pairs per expert to avoid overload and improve throughput.

2) Principled auxiliary losses
- Load balancing (Switch-style): encourages even routing across experts.
- Router z-loss: stabilizes router logits by penalizing log-sum-exp magnitude.

3) Two collaboration modes
- Legacy collaboration (attention over selected expert outputs):
  - Per token, run a small MHA over the K expert outputs, apply RMSNorm+FFN, then fuse with top‑k weights.
  - Ideal if you want expressive interaction over expert hypotheses.
- Simple collaboration (shared-parameter, two-pass, no MHA) when `use_simple_collab=True`:
  - Pass 1: project messages `M`, queries `Q`, and keys `Kk` from selected expert outputs; compute KxK attention per token with masks, and aggregate `C = softmax(QK^T) @ M`.
  - Pass 2: update with an MLP over `[selected, C]` and residual merge; fuse with renormalized top‑k weights.
  - Benefits: fewer moving parts than full MHA, stable, vectorized, and easy to analyze.

4) Robust implementation details
- Router executed in fp32 for stability.
- Vectorized dispatch via sorting and `index_copy_`, with keep/drop tracking based on capacity.
- Per-step logging hooks expose expert indices for utilization tracking.
- Weight init carefully scaled for deep/stable training; output projections and MLP second layers use depth-aware std.

Generation
- Incremental generation with KV cache, respecting `max_seq_len`.
- GQA reduces KV cache footprint; RoPE indexing handles positional continuity across steps.

Key Files
- `model_moc.py` — Model definition: transformer stack, attention, MoC layer, and optimizer configuration helpers.
- `train_moc.py` — Main trainer: DDP, AMP, scheduling, checkpointing, W&B, and data pipeline.

--------------------------------------------------------------------------------

Training System

`train_moc.py` is a resilient, high-throughput trainer for single/multi-GPU runs.

Performance features
- Precision: Prefers `bfloat16` when supported; TF32 enabled on matmul/cudnn.
- Fused AdamW: Uses fused kernels on CUDA when available; router parameters use a reduced LR.
- `torch.compile`: Optional graph compilation with safe fallback.
- DDP: `DistributedDataParallel` with `no_sync` for gradient accumulation.
- Gradient Accumulation and Clipping: Simulates large effective batch sizes and stabilizes updates.

Scheduling and optimization
- Cosine decay with linear warmup via `get_lr`.
- Separate param groups:
  - Decay for weight matrices.
  - No weight-decay for biases/norms.
  - Router gate with 0.5x LR to temper routing dynamics.

Data pipeline
- `ShardDataset` streams token IDs from many memory-mapped `.npy` shards.
- Produces aligned `(x, y)` pairs of length `seq_len` with ignore-index padding when crossing shard boundaries.
- Efficient `DataLoader` with `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`.

Checkpointing and resume
- Saves full state: model, optimizer, config, `step`, and `epoch`.
- Always-up-to-date `latest_checkpoint.pt` plus periodic numbered checkpoints.
- Strict state dict loading (after unwrapping DDP/compile prefixes).

Experiment tracking
- Console progress with loss, perplexity, LR, and grad-norm.
- W&B metrics:
  - Total loss, main CE loss, aux loss, perplexity, LR, grad-norm.
  - Expert utilization histogram for the first MoC layer.
  - Token–expert pair drop rate under capacity constraints.

--------------------------------------------------------------------------------

Getting Started

Prerequisites
- Python 3.9+ recommended
- PyTorch with CUDA for GPU training
- Optional: Weights & Biases account for logging

Install
```bash
git clone https://github.com/MeryylleA/lunariscodex-MoC.git
cd lunariscodex-MoC
```

Prepare data
1) Train a tokenizer (e.g., Hugging Face `tokenizers`) and save `tokenizer.json`.
2) Tokenize all documents, append an end-of-text token to each document.
3) Interleave heterogeneous sources before tokenization to stabilize training.
4) Concatenate token IDs and shard into `.npy` files of `dtype=np.uint16`. Use large shards for IO efficiency.

Example directory: `data/` containing `shard_0000.npy`, `shard_0001.npy`, ...

Configure a run
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

  # Collaboration mode
  use_simple_collab: false        # set true to enable the simple 2-pass collab
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
out_dir: "checkpoints/exp-mocv2"
save_interval: 1000
log_interval: 20

wandb_project: "lunaris-codex-moc-v2"
wandb_run_name: "mocv2-1024d-8x-top2"
```

Launch training
Single-node multi-GPU via `torchrun`:
```bash
torchrun --standalone --nproc_per_node=auto train_moc.py train.yaml
```

To resume, keep `latest_checkpoint.pt` in `out_dir`; the trainer loads it automatically.

--------------------------------------------------------------------------------

Configuration Guide

Model scale
- `d_model`, `n_layers`, `n_heads`: choose based on memory/compute budget.
- `n_kv_heads`: set below `n_heads` for GQA; typical ratios 2–4x queries per KV head.
- `max_seq_len`: ensure it accommodates the longest planned sequences; RoPE buffers are precomputed up to this length.

MoC specifics
- `n_experts`: total experts in each MoC layer.
- `top_k`: experts per token; `2` is a strong default balancing quality and compute.
- `capacity_factor`: controls maximum token–expert pairs per expert. Lower values increase drops (regularization/throughput trade-off).
- `aux_loss_weight` and `router_z_loss_weight`: stabilize and balance routing; defaults are tuned for typical setups.
- `use_simple_collab`: enables the two-pass shared-parameter collaboration instead of full MHA over K; simpler and often faster.

Stability and regularization
- `dropout`: applied in attention projections, MLPs, and MoC collab heads; start low when training on very large data.
- QK-norm: Enabled by default in `Attention` to stabilize large models and higher learning rates.
- Gradient clipping: keep at `1.0` initially; adjust if grad-norms spike.

Precision and performance
- Prefer `bfloat16` on Ampere+ for stability and speed.
- Enable `torch.compile` to optimize graphs; safe fallbacks are in place.

Optimizer and LR schedule
- AdamW with fused kernels on CUDA when available.
- Router gate parameters use 0.5x LR by default for smoother routing dynamics.
- Linear warmup -> cosine decay; set `max_steps` to your planned token budget.

--------------------------------------------------------------------------------

Data Pipeline Details

Input format
- Directory of `.npy` shards containing token IDs of type `uint16`.
- Each shard can be very large (hundreds of millions to billions of tokens).
- The loader creates non-overlapping windows of length `max_seq_len + 1` to produce `(x, y)` where `y` is `x` shifted by 1.
- When windows cross shard boundaries, the dataset stitches from the next shard and pads with `-1` (ignored in CE loss).

Best practices
- Interleave datasets before tokenization to avoid curriculum artifacts and stabilize optimization.
- Ensure a dedicated end-of-text token is present in the tokenizer vocabulary and appended per document.
- Deduplicate and filter sources aggressively; data quality drives outcomes.

--------------------------------------------------------------------------------

Logging and Diagnostics

Console
- Loss, main CE, aux loss, perplexity, LR, gradient norm via tqdm.

W&B
- Enable by setting `wandb_project`.
- Logs:
  - Scalars: `loss/total`, `loss/main`, `loss/aux`, `perplexity`, `lr`, `grad_norm`.
  - Expert utilization: per-expert fraction for the first MoC layer (`experts/util_layer0/e{i}`).
  - Token–expert pair drop rate under capacity constraints (`experts/drop_rate_layer0`).

Interpreting MoC metrics
- Utilization should be relatively even; heavy skew suggests router undertraining or insufficient aux loss.
- Drop rate increases as capacity decreases; moderate drops can act as regularization, excessive drops may starve experts.

--------------------------------------------------------------------------------

Examples

Minimal sanity test
- Run `python model_moc.py` to execute built-in sanity checks for both collaboration modes: trains a tiny model for a few steps and tests generation.

Generation
- Use `LunarisCodex.generate(idx, max_new_tokens, temperature=..., top_k=...)` for quick sampling with KV caching and GQA.

Custom optimizers/schedulers
- `LunarisCodex.configure_optimizers(...)` returns an AdamW configured with distinct parameter groups including router-specific LR. You can swap in your own scheduler if desired.

--------------------------------------------------------------------------------

Migration and Extensibility

- The code keeps parameter names stable even when collaboration mode changes; both paths are present in the module to simplify checkpoint compatibility.
- GQA ratios can be changed as long as `n_heads % n_kv_heads == 0`.
- To disable MoC and use dense FFN, set `n_experts: null` or `n_experts: 0` in config.

Suggested extensions
- Multi-MoC stacks: enable MoC on a subset of layers or alternate dense and MoC blocks.
- Routing research: experiment with temperature annealing, entropy regularizers, or load-aware routing.
- Parallelism: add FSDP or tensor parallel for very large models.
- Evaluation: integrate with HF `transformers` for downstream task evaluation or build a small held-out validation loop.

--------------------------------------------------------------------------------

Limitations

- Pre-training focused: no built-in fine-tuning or instruction-tuning pipelines.
- No built-in evaluation suite; you must implement downstream validation separately.
- Only DDP is implemented; no FSDP/tensor/pipeline parallel in this repo.
- Data preparation/tokenization is user-managed; quality of results depends heavily on your corpus and tokenizer.

--------------------------------------------------------------------------------

Repro Tips

- Start small: verify the entire pipeline (data, training loop, checkpoints, logs) with a tiny model and short context.
- Monitor router metrics early; if utilization collapses to a few experts, increase `aux_loss_weight`, reduce LR for the router, or add warmup steps.
- Keep `use_gradient_checkpointing: true` when scaling depth or sequence length.
- Use `bfloat16` on supported GPUs for stability at higher learning rates.

--------------------------------------------------------------------------------

Citations and Acknowledgements

- Built upon the spirit of Andrej Karpathy’s nanoGPT: clarity, minimalism, hackability.
- Incorporates ideas from recent literature on RoPE, GQA, QK-norm, SwiGLU, and capacity-aware MoE training.
- Thanks to the open-source community for foundational tooling and insights.

--------------------------------------------------------------------------------

License

Apache License 2.0. See `LICENSE` for details.

--------------------------------------------------------------------------------

Contact and Community

- Author: Francisco Antonio (GitHub `@MeryylleA`)
- Discord: Moon Cloud Services — https://discord.gg/JNsfzEwMtC

If you use Lunaris Codex in your research or products, please consider sharing feedback, issues, and results. Contributions are welcome.

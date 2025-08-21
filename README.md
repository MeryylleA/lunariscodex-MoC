# Lunaris MoC — Mixture‑of‑Collaboration with Iterative Reasoning Loops

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MeryylleA/lunariscodex-MoC)

**Lunaris MoC** is a research‑grade, production‑minded Transformer featuring **Mixture‑of‑Collaboration (MoC)** blocks and **Iterative Reasoning Loops (IRL)**. Each token is *routed* to a small **Top‑K** set of experts; those expert states **collaborate** (exchange information) before a **learned fusion** produces the block output. Inside each expert, a lightweight **IRL** performs a few refinement steps to deepen computation without adding parameters.

This repository contains:

* `model_moc.py` — the model architecture (**LunarisCodex**) and configuration.
* `train_moc.py` — the training script (DDP‑ready, bf16, fused AdamW, checkpointing, rich logging).

> **Status**: mainline; actively used for training and experimentation.

---

## Table of Contents

* [Key Ideas](#key-ideas)

  * [MoC (Mixture of Collaboration)](#moc-mixture-of-collaboration)
  * [IRL (Iterative Reasoning Loops)](#irl-iterative-reasoning-loops)
  * [Routing, Capacity & Auxiliary Losses](#routing-capacity--auxiliary-losses)
  * [Backbone](#backbone)
* [Quickstart](#quickstart)

  * [Environment](#environment)
  * [Data Preparation](#data-preparation)
  * [Configuration](#configuration)
  * [Launch Training](#launch-training)
* [Training Details](#training-details)

  * [Precision & Performance](#precision--performance)
  * [Distributed Training (DDP)](#distributed-training-ddp)
  * [Checkpointing & Resume](#checkpointing--resume)
  * [Logging & Metrics Reference](#logging--metrics-reference)
* [Inference](#inference)
* [FAQ](#faq)
* [License](#license)

---

## Key Ideas

### MoC (Mixture of Collaboration)

**What**: A collaborative alternative to vanilla MoE. For each token, a learned **router** picks **Top‑K experts**. Rather than processing them independently, MoC lets those expert states **talk to each other** before fusing them.

**Per‑token flow**:

1. **Routing (fp32)**: compute router logits → select Top‑K experts; keep full probability distribution for losses/metrics.
2. **Capacity control**: each expert accepts up to a capacity; excess token–expert assignments are dropped (counted and optionally penalized).
3. **Expert compute**: each selected expert runs its **IRL FFN** on the token state.
4. **Collaboration**: run **R rounds** of attention/MLP across the **K expert states** (optionally with a learnable **mediator** token) to exchange information.
5. **Fusion**: a small **gate** mixes mediator output with the probability‑weighted sum of expert states.

**Why**: Collaboration addresses the “expert isolation” issue in classic MoE and improves credit assignment across complementary sub‑skills.

### IRL (Iterative Reasoning Loops)

Inside each expert FFN, apply **S micro‑steps** that refine the hidden state:

```python
h = x
for s in range(S):
    h = h + α * FFN(h + x)   # α ≈ 1/√S
return h
```

This raises effective depth at small extra cost, often smoothing optimization and helping compositional behavior.

### Routing, Capacity & Auxiliary Losses

* **Top‑K routing** per token with probabilities used beyond the argmax for regularization and metrics.
* **Capacity factor** bounds token–expert pairs per expert. Overflow pairs are **dropped** deterministically; the fraction is tracked.
* **Auxiliary losses** (applied to router outputs):

  * **Balance loss** encourages uniform expert utilization over a batch/window.
  * **Z‑loss** regularizes the scale of router logits via `E[(logsumexp(logits))^2]`text{logits})^2]).
  * **Drop penalty** (optional) adds a small cost proportional to the fraction of dropped token–expert pairs.

### Backbone

* Decoder‑only Transformer, Pre‑LN with **RMSNorm**.
* **RoPE** positional encoding.
* **GQA** attention (grouped KV heads) via PyTorch **SDPA/Flash** kernels.
* **SwiGLU‑style** MLP where FFN is used.
* **Tied embeddings** between token embeddings and output head.

---

## Quickstart

### Environment

* Python ≥ 3.10
* PyTorch ≥ 2.1 with CUDA (bf16 recommended on recent GPUs)

```bash
pip install -r requirements.txt
```

### Data Preparation

`train_moc.py` expects pre‑tokenized shards stored as `.npy` arrays of token IDs. The loader memory‑maps shards and emits sequences of length `max_seq_len`.

1. **Tokenize** your corpus with a BPE tokenizer (HF `tokenizers`, SentencePiece, etc.). Append an end‑of‑text token per document.
2. **Concatenate** token IDs and split into shard files (e.g., `tokens_000.npy`, `tokens_001.npy`, ...). Any integer dtype is fine; casting to `int64` happens **on GPU**.
3. Place shards under `data/` (or set `data_dir` in the config).

> Padding/targets: the dataset yields `(x, y, valid_len_y)`. Targets beyond `valid_len_y` are set to `ignore_index=-1` **on GPU** to avoid CPU‑side dtype promotion.

### Configuration

Create a YAML file (e.g., `train.yaml`):

```yaml
model:
  # Backbone
  vocab_size: 50304
  d_model: 1024
  n_layers: 20
  n_heads: 16
  n_kv_heads: 4
  max_seq_len: 2048
  dropout: 0.05

  # MoC routing
  n_experts: 8
  top_k: 2
  capacity_factor: 1.25
  aux_loss_weight: 1.0e-2
  router_z_loss_weight: 1.0e-3
  router_noise_std: 0.0

  # Collaboration among experts
  use_simple_collab: false
  use_moc_collab: true
  moc_collab_steps: 2
  moc_use_mediator: true
  moc_collab_heads: 4
  moc_collab_dropout: 0.0

  # IRL steps inside each expert
  n_reasoning_steps: 2

  # Engineering
  use_gradient_checkpointing: true
  grad_ckpt_policy: ffn

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
out_dir: "checkpoints/moc"
save_interval: 1000
log_interval: 20

wandb_project: "lunaris-moc"
wandb_run_name: "moc-1024d-8x-top2-irl2-collab2"
```

> The `TrainConfig.from_yaml()` path supplies sensible defaults if some MoC fields are omitted (e.g., `top_k`, `capacity_factor`, `aux_loss_weight`, `router_z_loss_weight`).

### Launch Training

Single node (will use all visible GPUs via `torchrun`):

```bash
torchrun --standalone --nproc_per_node=auto train_moc.py train.yaml
```

---

## Training Details

### Precision & Performance

* **bf16 autocast** on supported GPUs; attention uses **SDPA/Flash** kernels.
* **TF32** matmuls enabled on Ampere+.
* **Fused AdamW** when available.
* Optional `torch.compile()` with safe fallback.

### Distributed Training (DDP)

* **DDP** integrated via `torchrun`; uses `no_sync()` during gradient accumulation.
* `DistributedSampler` ensures shard coverage across ranks.
* Gradient accumulation and gradient clipping are built in.

### Checkpointing & Resume

* Periodic checkpoints at `save_interval`.
* The trainer **saves twice** each time: a numbered `ckpt_<step>.pt` and a rolling `latest_checkpoint.pt`.
* Resuming restores model weights, optimizer state, and step/epoch counters.

### Logging & Metrics Reference

The trainer prints concise console stats and (optionally) logs to **Weights & Biases** with the keys below. Where helpful, a short definition is provided.

#### Core losses & optimization

* **`loss/total`** — sum of main cross‑entropy loss and auxiliary router losses.
* **`loss/main`** — token cross‑entropy (ignores positions with target `-1`).
* **`loss/aux`** — sum of balance, z‑loss, and (if enabled) drop penalty.
* **`perplexity`** — `exp(loss/main)`; clipped to avoid overflow in logs.
* **`lr`** — learning rate (cosine decay with warmup; floor at 1% of peak).
* **`grad_norm`** — global gradient norm after clipping.

#### Router behavior (MoC)

* **`experts/util_layer0/e<i>`** — fraction of routed token–expert pairs that selected expert *i* (first MoC layer). Healthy training shows reasonably uniform utilization over time.
* **`experts/drop_rate_layer0`** *(if enabled)* — approximate fraction of token–expert pairs that were **dropped** by capacity in the first MoC layer. Computation:

  * Let **E** be the number of experts, **K** the Top‑K, and **N** the number of tokens in the batch. Then **N\_pairs = N × K**.
  * Capacity per expert **C = ceil((N\_pairs / E) × capacity\_factor)**.
  * For per‑expert counts $c_e$, **drop\_rate = Σ\_e max(0, c\_e − C) / N\_pairs**.
* **`viz/layer0_expert_util_cooc`** *(image, if plotting is available)* — side‑by‑side plots of (1) expert utilization bars and (2) pairwise expert **co‑occurrence** heatmap across the logging window. Useful to detect persistent expert couplings.
* **`meta/active_params_per_token`**, **`meta/total_trainable_params`**, **`meta/active_params_ratio`** — estimate of how many parameters are **active per token** given MoC’s Top‑K routing, relative to the full parameter count.

#### Throughput & system (when enabled)

* **`throughput/tok_per_s_global`** — aggregated tokens/sec across all ranks (via all‑reduce).
* **`throughput/tok_per_s_per_gpu`** — tokens/sec per GPU.
* **`throughput/samples_per_s_local`** — samples/sec on the local rank (for quick sanity checks).
* **`timing/sec_per_step_window`** — average seconds/step over the recent log window.
* **`timing/eta_sec`** — estimated seconds to completion given the current window speed.
* **`mem/current_gib`**, **`mem/max_gib`** — current and peak CUDA memory (GiB) since start.

> **Tips**: A stable run typically shows (a) slowly decreasing `loss/main`, (b) near‑uniform expert utilization, (c) low and stable drop rate at reasonable `capacity_factor`, and (d) smooth throughput without oscillations from host I/O.

---

## Inference

Basic greedy/Top‑k generation with cached decoding:

```python
from model_moc import LunarisCodex, LunarisCodexConfig
import torch

cfg = LunarisCodexConfig(max_seq_len=2048, vocab_size=50304, n_experts=8, top_k=2)
model = LunarisCodex(cfg).eval().cuda()

# idx: [B, T] start tokens
out = model.generate(idx, max_new_tokens=64, temperature=0.8, top_k=50)
```

**Practical tips**

* Keep `use_cache=True` for fast decoding.
* For latency‑sensitive serving, you can reduce collaboration or IRL rounds at decode time if your setup surfaces such knobs.
* Prefer bf16 weights and KV cache (or fp16) during inference to conserve memory.

---

## FAQ

**Is this just MoE?**  MoC uses a router like MoE, but experts **collaborate before fusion**, which changes optimization and sample efficiency.

**Do I need custom CUDA kernels?**  No. The model runs on stock PyTorch (SDPA/Flash attention). Custom kernels can improve throughput but are optional.

**How do I monitor expert behavior?**  Watch `experts/util_layer0/e<i>` and, if enabled, `experts/drop_rate_layer0`. The co‑occurrence heatmap is a powerful qualitative signal.

**Can I train with mixed precision?**  Yes — bf16 autocast is supported and recommended on recent GPUs.

**How big can I scale this?**  The trainer is DDP‑ready and has been used on multi‑GPU nodes. For larger clusters, you can integrate external engines while keeping `LunarisCodex` intact.

---

## License

Apache License 2.0 — see `LICENSE`.

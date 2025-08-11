# Lunaris MoC — Mixture of Collaboration with IRL

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Lunaris MoC is a research‑grade, production‑oriented Transformer with **Mixture‑of‑Collaboration (MoC)** blocks and **Iterative Reasoning Loops (IRL)**. The core idea: per‑token **routing selects K paths**, these paths **collaborate** (exchange information) before a **learned fusion** produces the block output. Inside each path, a lightweight **IRL** performs a few refinement steps without adding parameters.

This repository contains:

* `model_moc.py` — the architecture (**LunarisCodex**) and configuration.
* `train_moc.py` — a scalable training script (DDP‑ready, bf16, fused AdamW, checkpointing, logging).

> **Status**: mainline, ready for training and experimentation.

---

## Table of Contents

* [Key Concepts](#key-concepts)

  * [MoC (Mixture of Collaboration)](#moc-mixture-of-collaboration)
  * [IRL (Iterative Reasoning Loops)](#irl-iterative-reasoning-loops)
  * [Routing, Capacity, and Auxiliary Losses](#routing-capacity-and-auxiliary-losses)
  * [Backbone: Attention & Positional Encoding](#backbone-attention--positional-encoding)
* [Quickstart](#quickstart)

  * [Environment](#environment)
  * [Data Preparation](#data-preparation)
  * [Configuration](#configuration)
  * [Launch Training](#launch-training)
* [Training Details](#training-details)

  * [Precision & Performance](#precision--performance)
  * [Distributed Training](#distributed-training)
  * [Checkpointing & Resume](#checkpointing--resume)
  * [Logging & Metrics](#logging--metrics)
* [Inference](#inference)

  * [Practical Tips](#practical-tips)
* [FAQ](#faq)
* [License](#license)
* [Community](#community)

---

## Key Concepts

### MoC (Mixture of Collaboration)

**What it is.** A drop‑in replacement for the usual FFN block. For each token, a small **Top‑K** subset of paths ("experts") is selected by a learned router. Unlike classical MoE, MoC makes experts **talk to each other** *before* fusing.

**How it works (per token):**

1. **Routing (fp32):** compute logits → pick Top‑K experts; keep full‑distribution `router_probs` for losses.
2. **Capacity control:** each expert gets a bounded number of token‑assignments; excess pairs are dropped (counted and optionally penalized).
3. **Expert compute:** each selected expert runs its **IRL FFN** on the token state.
4. **Collaboration:** run **R rounds of MHA + FFN** across the *K expert states* (optionally add a learnable **mediator** token). This exchanges information among paths.
5. **Fusion:** a small **gate** mixes the mediator output with the probability‑weighted sum of expert states.

**Why.** Collaboration counters the “isolation” problem of MoE and improves **credit assignment** across complementary sub‑skills.

### IRL (Iterative Reasoning Loops)

Inside each expert/FFN, compute **S micro‑steps**:

```
h0 = x
for s in 1..S:
    h = h + α · FFN(h + x)        # α ≈ 1/√S
return h
```

This increases effective depth with minimal overhead. It tends to smooth optimization and can help compositional behavior.

### Routing, Capacity, and Auxiliary Losses

* **Top‑K routing:** selects experts per token; probabilities used for balancing.
* **Capacity factor:** limits token‑expert pairs per expert; overflow pairs are dropped deterministically.
* **Auxiliary losses:**

  * **Balance loss** encourages uniform expert utilization.
  * **Z‑loss** regularizes the router logits via `mean(logsumexp(logits)^2)`.
  * **Drop penalty** (optional) adds a small cost proportional to the fraction of dropped pairs.

### Backbone: Attention & Positional Encoding

* **Decoder‑only Transformer**, Pre‑LN with **RMSNorm**.
* **RoPE** positional encoding.
* **GQA** attention (grouped KV heads) with **scaled dot‑product attention (Flash/SDPA)**.
* **SwiGLU‑style** MLP where FFN is used.
* **Tied embeddings** (`wte` and `lm_head`).

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
2. **Concatenate** token IDs and split into shard files (e.g., `tokens_000.npy`, `tokens_001.npy`, ...). Any integer dtype is fine; the loader casts to `int64`.
3. Place shards under `data/` (or set `data_dir` in the config).

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

  # MoC
  n_experts: 8
  top_k: 2
  capacity_factor: 1.25
  aux_loss_weight: 1.0e-2
  router_z_loss_weight: 1.0e-3
  router_noise_std: 0.0

  # Collaboration (message passing among experts)
  use_simple_collab: false
  use_moc_collab: true
  moc_collab_steps: 2
  moc_use_mediator: true
  moc_collab_heads: 4
  moc_collab_dropout: 0.0

  # IRL
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

### Launch Training

Single node (will use all visible GPUs via `torchrun`):

```bash
torchrun --standalone --nproc_per_node=auto train_moc.py train.yaml
```

---

## Training Details

### Precision & Performance

* **bf16 autocast** on supported GPUs; attention uses **Flash/SDPA** kernels.
* **TF32** matmuls enabled by default on Ampere+.
* **Fused AdamW** used when available.
* Optional `torch.compile()` path with safe fallback.

### Distributed Training

* **DDP** is integrated; `torchrun` sets up ranks and local devices.
* The dataloader uses a **DistributedSampler** when DDP is active.
* Gradient accumulation and gradient clipping are built in.

### Checkpointing & Resume

* Periodic checkpoints at `save_interval` and a rolling `latest_checkpoint.pt`.
* Resuming restores model/optimizer state and training step/epoch.

### Logging & Metrics

* Losses: **total**, **main (CE)**, **aux**; derived **perplexity**.
* **Learning rate** and **grad‑norm**.
* **Expert utilization** (per‑expert frequency for the first MoC layer) and **approximate drop rate**.
* If `wandb_project` is set, metrics are logged to Weights & Biases.

---

## Inference

### Greedy/Top‑k Generation

`LunarisCodex.generate()` supports cached decoding with optional temperature and `top_k`:

```python
from model_moc import LunarisCodex, LunarisCodexConfig
import torch

cfg = LunarisCodexConfig(max_seq_len=2048, vocab_size=50304, n_experts=8, top_k=2)
model = LunarisCodex(cfg).eval().cuda()

# idx: [B, T] start tokens
out = model.generate(idx, max_new_tokens=64, temperature=0.8, top_k=50)
```

### Practical Tips

* Keep `use_cache` on (the implementation uses `past_key_values`).
* For latency‑sensitive serving, reduce collaboration/IRL rounds at decode time if your setup exposes such knobs.
* Prefer BF16 weights and KV cache (or FP16) during inference for memory efficiency.

---

## FAQ

**Is this MoE?**  It uses routing like MoE, but the selected paths **collaborate before fusion**; experts aren’t isolated.

**Do I need custom CUDA kernels?**  Not required. The model runs on stock PyTorch (SDPA/Flash attention). Custom kernels can improve throughput but are optional.

**Can I train with mixed precision?**  Yes — bf16 autocast is supported.

**How do I monitor expert behavior?**  Check the logged **expert utilization** and **drop rate** metrics. Uniform utilization and low, stable drop rates are healthy signs.

**How big can I scale this?**  The trainer is DDP‑ready and has been used on multi‑GPU nodes. For very large clusters, you can integrate engines like Megatron‑DeepSpeed while keeping this model class.

---

## License

Apache License 2.0 — see `LICENSE`.

---

## Community

* Author: **Francisco Antonio** (GitHub `@MeryylleA`)
* Focus: iterative reasoning, collaborative expert systems, efficient pretraining.

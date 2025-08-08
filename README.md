# Lunaris Codex — MoC v4.5 (Iterative Reasoning + Collaborative Experts)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord\&logo=discord\&color=7289DA)](https://discord.gg/JNsfzEwMtC)

Lunaris Codex is a research‑grade, production‑oriented toolkit for pre‑training decoder‑only Transformers. This **v4.5** update refines our **Mixture‑of‑Collaboration (MoC)** design and keeps **Iterative Reasoning Layers (IRL)** at its core. The model enables *per‑token* collaboration among top‑k experts while each expert runs multi‑step internal reasoning.

### What’s new in v4.5 (delta from v4)

* **MoC collaborative message passing (default)**: real expert‑to‑expert interaction via per‑token **MHA+FFN rounds** over the K selected experts, with optional **mediator token** and **learned gate** between mediator and weighted expert aggregate.
* **Vectorized sorted routing**: single argsort + contiguous per‑expert segments with **capacity control** (power‑of‑two near average×capacity\_factor). Reduces Python/kernel overhead without changing training APIs.
* **Router stability**: router logits and aux losses computed in **fp32**; optional `router_noise_std` during warmup only.
* **IRL preserved**: experts remain IRL FFNs with residual scaling `1/√steps`.
* **Backwards compatibility**: `train_moc.py` works unchanged; forward contract and optimizer setup are identical to v4.

---

## Architecture Overview

**File:** `model_moc.py` → class **`LunarisCodex`** with config **`LunarisCodexConfig`**.

### Transformer backbone

* **Decoder‑only, Pre‑LN** with `RMSNorm` before Attention and FFN.
* **RoPE** positional encoding (precomputed complex rotations, fp32 angles).
* **Attention**: **GQA** (`n_heads` queries sharing `n_kv_heads` keys/values) + **Flash SDPA** causal attention.
* **FFN**: SwiGLU‑style MLP.
* **Tied embeddings**: `wte` and `lm_head` share weights.

### IRL — Iterative Reasoning Layers

Each expert’s FFN can perform `n_reasoning_steps` refinements with residual injection and scaling, increasing compute depth **without adding parameters**.

### MoC v4.5 — Collaborative Experts

Implemented in `MoCTopKExperts`:

1. **Deterministic top‑k routing** (fp32): select experts per token; compute balance + z‑loss auxiliaries.
2. **Vectorized dispatch with capacity**: contiguous per‑expert segments, `index_copy_`, and keep/drop accounting.
3. **Collaboration (default)**:

   * Rounds of **MHA+FFN** across K expert states (+ optional **mediator** token).
   * Final **gated fusion** between mediator output and weighted expert aggregate.
4. **Simple collab (legacy)**: optional lightweight 2‑pass path (kept for ablations).

---

## Training System (unchanged vs v4)

* **Precision**: bf16 autocast in the trainer; router math in fp32 inside the model.
* **Optimizer**: fused AdamW when available; router params get reduced LR and no weight decay.
* **Checkpointing**: `use_gradient_checkpointing` with `grad_ckpt_policy: ffn` by default; `block` optional.
* **DDP / torch.compile**: supported as before. No changes to `train_moc.py` required.

---

## Get Started (v4.5 branch)

```bash
# Clone and switch to the v4.5 branch
git clone https://github.com/MeryylleA/lunariscodex-MoC.git
cd lunariscodex-MoC
git fetch origin
git checkout V4.5   # https://github.com/MeryylleA/lunariscodex-MoC/tree/V4.5
```

### Data preparation (same as v4)

* Train or provide a tokenizer (e.g., HF `tokenizers`).
* Tokenize datasets; append EOT per document.
* Concatenate IDs and shard into `.npy` files (`uint16`).

---

## Configuration (high‑impact knobs)

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

  # MoC v4.5
  n_experts: 8
  top_k: 2
  capacity_factor: 1.25
  aux_loss_weight: 1.0e-2
  router_z_loss_weight: 1.0e-3
  router_noise_std: 0.0           # enable briefly during warmup in the trainer

  # Collaboration (v4.5 defaults)
  use_simple_collab: false        # simple path off by default
  use_moc_collab: true            # collaborative message passing on
  moc_collab_steps: 2             # rounds of MHA+FFN among K experts
  moc_use_mediator: true          # add mediator token per token
  moc_collab_heads: 4             # must divide d_model; falls back safely
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
out_dir: "checkpoints/moc-v4_5"
save_interval: 1000
log_interval: 20

wandb_project: "lunaris-codex-moc-v4_5"
wandb_run_name: "moc-v4_5-1024d-8x-top2-irl2-collab2"
```

### Launch training

```bash
torchrun --standalone --nproc_per_node=auto train_moc.py train.yaml
```

> **Note:** No changes needed in `train_moc.py`. The model keeps the same forward contract and optimizer split.

---

## Upgrading from v4 → v4.5

* **Defaults**: collaboration mode switches from "simple" (off by default) to **MoC collaborative** (on by default).
* **New flags**: `use_moc_collab`, `moc_collab_steps`, `moc_use_mediator`, `moc_collab_heads`, `moc_collab_dropout`.
* **Safe ablations**: set `use_simple_collab: true` and `use_moc_collab: false` to recover the lightweight legacy path.

---

## Diagnostics & Tips

* **Expert utilization**: should be roughly even; increase `aux_loss_weight` or reduce router LR if it collapses.
* **Drop fraction**: tune `capacity_factor` (1.0–1.3 typical). Moderate drops can regularize; excessive drops starve experts.
* **IRL steps**: `2–3` is a good default; increase gradually and watch throughput.
* **Mediator gating**: the gate blends mediator vs. weighted expert aggregate; useful to monitor when analyzing collaboration behavior.

---

## License

Apache License 2.0 — see `LICENSE`.

---

## Community

* **Author**: Francisco Antonio (GitHub `@MeryylleA`)
* **Discord**: Moon Cloud Services — [https://discord.gg/JNsfzEwMtC](https://discord.gg/JNsfzEwMtC)
* **Focus**: Iterative reasoning, collaborative expert systems, efficient single‑GPU training

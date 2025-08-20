"""
Main Training Script for the LunarisCodex Language Model (MoC top-k collaborative experts).

- Apply dtype-handling patch: keep arrays in original dtype in Dataset; cast with `.to(...)` on GPU.
- Enhance terminal & W&B logging: clearer metrics, tokens/sec, ETA, and memory hints.
- Report "active parameters per token" as a dedicated log entry.
- Add a novel architectural visual log (expert utilization & co-occurrence) without editing model_moc.py.
- Save checkpoints twice every `save_interval` (numbered and latest) as required.
"""

import os
import time
import math
import glob
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
from contextlib import nullcontext

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

# Lazy import for visualization to avoid startup overhead if not used
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # We'll guard usage.

from model_moc import LunarisCodex, LunarisCodexConfig, compile_model_if_available

# ---------------------------
# Config
# ---------------------------

@dataclass
class TrainConfig:
    model: LunarisCodexConfig = field(default_factory=LunarisCodexConfig)
    data_dir: str = "data/"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    max_steps: int = 600000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    grad_clip: float = 1.0
    device: str = "cuda"
    compile_model: bool = True
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000
    save_latest_always: bool = True  # kept for backward-compat; we now always save twice per requirement
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    wandb_project: Optional[str] = "lunaris-codex-moc"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @property
    def sequence_length(self):
        return self.model.max_seq_len

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        model_config_dict = config_dict.pop("model", {})
        # Ensure top_k is carried through
        model_config = LunarisCodexConfig(**model_config_dict)

        # Backward-compat defaults
        if model_config.n_experts and getattr(model_config, "aux_loss_weight", None) is None:
            model_config.aux_loss_weight = 1e-2
        if getattr(model_config, "capacity_factor", None) is None:
            model_config.capacity_factor = 1.25
        if getattr(model_config, "router_z_loss_weight", None) is None:
            model_config.router_z_loss_weight = 1e-3
        if getattr(model_config, "top_k", None) is None:
            model_config.top_k = 2  # default top_k > 1

        config_dict['model'] = model_config

        # normalize numeric fields
        float_fields = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'grad_clip']
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps',
                      'num_epochs', 'save_interval', 'log_interval', 'num_workers', 'prefetch_factor']
        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])

        return cls(**config_dict)

# ---------------------------
# Dataset
# ---------------------------

class ShardDataset(Dataset):
    """
    Memory-efficient dataset over .npy token shards. Produces (x, y, valid_len_y) with length seq_len.
    Padding tokens are neutral at load time; we set ignore_index on GPU to avoid dtype casting on CPU.
    """
    def __init__(self, data_dir: str, sequence_length: int):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.shards:
            raise ValueError(f"No .npy files found in directory: {data_dir}")
        # Lazy memory map shards
        self.mmap_shards = [np.load(shard, mmap_mode='r') for shard in self.shards]
        self.shard_lengths = [len(shard) for shard in self.mmap_shards]
        total_tokens = sum(self.shard_lengths)
        self.total_samples = total_tokens // self.sequence_length
        self.cumulative_lengths = np.cumsum(self.shard_lengths)
        print(f"[DATA] Loaded {len(self.shards)} shards. Total tokens: {total_tokens/1e9:.2f}B.")
        print(f"[DATA] Creating {self.total_samples:,} non-overlapping samples of length {self.sequence_length}.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            x: torch Tensor (original dtype), length L
            y: torch Tensor (original dtype), length L
            valid_len_y: int count of valid targets in y (positions >= valid_len_y must be set to ignore_index on GPU)
        """
        L = self.sequence_length
        token_start_pos = idx * L
        shard_idx = np.searchsorted(self.cumulative_lengths, token_start_pos, side='right')
        local_start_idx = token_start_pos if shard_idx == 0 else token_start_pos - self.cumulative_lengths[shard_idx - 1]
        seq_len_with_target = L + 1

        # Fetch possibly across shard boundary (kept for correctness; see note below)
        if local_start_idx + seq_len_with_target <= self.shard_lengths[shard_idx]:
            seq = self.mmap_shards[shard_idx][local_start_idx: local_start_idx + seq_len_with_target]
        else:
            remaining_len = self.shard_lengths[shard_idx] - local_start_idx
            seq_part1 = self.mmap_shards[shard_idx][local_start_idx: local_start_idx + remaining_len]
            need = seq_len_with_target - remaining_len
            if shard_idx + 1 < len(self.mmap_shards):
                seq_part2 = self.mmap_shards[shard_idx + 1][:need]
                # NOTE: This concatenation happens in CPU; we keep dtype and avoid astype here by design.
                seq = np.concatenate((seq_part1, seq_part2))
            else:
                seq = seq_part1

        # Determine how many targets are valid BEFORE padding
        orig_len = int(len(seq))
        valid_len_y = int(max(0, min(L, orig_len - 1)))

        # Pad up to L+1 without forcing dtype promotion on CPU
        if orig_len < seq_len_with_target:
            pad_len = seq_len_with_target - orig_len
            # Use zero-pad in the same dtype; we will set ignore_index (-1) on GPU later.
            pad_val = np.array(0, dtype=seq.dtype)
            seq = np.pad(seq, (0, pad_len), 'constant', constant_values=pad_val)

        # Important: keep original dtype here; cast only on GPU later.
        seq_tensor = torch.from_numpy(seq)
        x, y = seq_tensor[:-1], seq_tensor[1:]
        return x, y, valid_len_y

# ---------------------------
# DDP / utils
# ---------------------------

def setup_ddp():
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        init_process_group("nccl")
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Setup complete: rank {rank}, world_size {world_size}, local_rank {local_rank}")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    if step >= config.max_steps:
        return config.learning_rate * 0.01
    decay_ratio = (step - config.warmup_steps) / max(1, (config.max_steps - config.warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)

def unwrap_model_keys(state_dict):
    unwrapped = {}
    prefixes_to_remove = ['_orig_mod.module.', 'module.', '_orig_mod.']
    for k, v in state_dict.items():
        new_k = k
        for prefix in prefixes_to_remove:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        unwrapped[new_k] = v
    return unwrapped

def compute_active_params_per_token(model: LunarisCodex) -> Tuple[int, int]:
    """
    Estimates the number of *active* parameters used per token given MoE/MoC routing.
    We count all non-expert params, and for each MoE block we add K * (per-expert params) instead of E * per-expert.
    """
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction = 0
    for block in model.transformer.h:
        if getattr(block, "is_moe", False):
            moc = block.feed_forward  # MoCTopKExperts
            if len(moc.experts) == 0:
                continue
            # Assume experts are homogenous
            per_expert = sum(p.numel() for p in moc.experts[0].parameters() if p.requires_grad)
            nE = int(getattr(moc, "n_experts", len(moc.experts)))
            k = int(getattr(moc, "top_k", 1))
            reduction += max(0, (nE - k)) * per_expert
    active = total_trainable - reduction
    return int(active), int(total_trainable)

# ---------------------------
# Training
# ---------------------------

def train(config_path: str):
    config = TrainConfig.from_yaml(config_path)
    is_ddp, rank, world_size, local_rank = setup_ddp()
    is_master = (rank == 0)

    # Seeds and perf knobs
    torch.manual_seed(1337 + rank)
    np.random.seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype) if device_type == 'cuda' else nullcontext()

    # Logging header
    if is_master:
        os.makedirs(config.out_dir, exist_ok=True)
        if config.wandb_run_name is None:
            config.wandb_run_name = f"run-moc-{time.strftime('%Y-%m-%d-%H-%M')}"
        print("-" * 72)
        print(" LUNARIS CODEX - MoC (top-k collaborative experts) Training")
        print("-" * 72)
        print(f"Model: {config.model}")
        if config.model.n_experts:
            print(f"MoC: experts={config.model.n_experts}, top_k={config.model.top_k}, "
                  f"cap_factor={config.model.capacity_factor}, aux={config.model.aux_loss_weight}, "
                  f"z={config.model.router_z_loss_weight}")
        print(f"Data: {config.data_dir}, SeqLen={config.sequence_length}, Device={config.device}, bf16={use_bf16}")
        print(f"Batch={config.batch_size} (per GPU), Accum={config.gradient_accumulation_steps}, "
              f"GlobalBatch≈{config.batch_size * config.gradient_accumulation_steps * world_size}")
        print("-" * 72)

    # W&B
    wandb = None
    if is_master and config.wandb_project:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                   name=config.wandb_run_name, config=asdict(config))

    # Data
    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.num_workers > 0 and config.persistent_workers),
        prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
        drop_last=True,
    )

    # Model
    model = LunarisCodex(config.model).to(config.device, dtype=torch.bfloat16 if use_bf16 else torch.float32)

    # Compute & announce active parameters per token (constant given K, E)
    if is_master:
        active_params, total_params = compute_active_params_per_token(model)
        ratio = active_params / max(1, total_params)
        print(f"[MODEL] Trainable params: {total_params:,} | Active per token (est.): {active_params:,} "
              f"({ratio:.2%} of total)")

    # Optional compile
    if config.compile_model and device_type == 'cuda':
        if is_master: print("[MODEL] Compiling with torch.compile ...")
        try:
            model = compile_model_if_available(model)
        except Exception as e:
            if is_master: print(f"[WARN] torch.compile failed, continuing without compile: {e}")

    # DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if is_ddp else model

    # Optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )

    # Resume
    current_step, current_epoch = 0, 0
    latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        if is_master: print(f"[RESUME] Loading checkpoint: {latest_path}")
        state = torch.load(latest_path, map_location=config.device)
        raw_model.load_state_dict(unwrap_model_keys(state['model']), strict=True)
        optimizer.load_state_dict(state['optimizer'])
        current_step = int(state.get('step', 0))
        current_epoch = int(state.get('epoch', 0))

    optimizer.zero_grad(set_to_none=True)
    if is_ddp:
        assert train_sampler is not None
        train_sampler.set_epoch(current_epoch)

    if is_master:
        print(f"\n[TRAIN] Starting at step {current_step} up to {config.max_steps} ...")
        pbar = tqdm(total=config.max_steps, desc="Steps", initial=current_step, ncols=140)

    # Log initial static metadata to W&B (active params per token)
    if is_master and wandb is not None:
        active_params, total_params = compute_active_params_per_token(raw_model)
        wandb.log({
            "meta/active_params_per_token": active_params,
            "meta/total_trainable_params": total_params,
            "meta/active_params_ratio": active_params / max(1, total_params),
            "step": current_step,
            "epoch": current_epoch,
        })

    # Training loop
    data_iter = iter(train_loader)

    # Throughput accumulators (reset each log window)
    last_log_time = time.time()
    tokens_since_log_local = 0  # local rank tokens
    samples_since_log_local = 0
    steps_since_log = 0
    # Buffer expert indices for visualization
    expert_indices_window: List[torch.Tensor] = []

    step_start_wall = time.time()

    while current_step < config.max_steps:
        current_step += 1
        steps_since_log += 1
        # LR schedule
        lr = get_lr(current_step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Accumulators for logging (losses averaged over grad-accum)
        accum_total = 0.0
        accum_main = 0.0
        accum_aux = 0.0

        # For capturing expert routing across micros in this step (layer 0)
        step_expert_indices: List[torch.Tensor] = []

        for micro in range(config.gradient_accumulation_steps):
            is_last_micro = (micro == config.gradient_accumulation_steps - 1)
            ddp_context = model.no_sync() if is_ddp and not is_last_micro else nullcontext()

            with ddp_context:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    current_epoch += 1
                    if is_ddp:
                        train_sampler.set_epoch(current_epoch)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                # Unpack dataset output; supports (x,y) or (x,y,valid_len_y)
                valid_y_len = None
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, valid_y_len = batch
                else:
                    x, y = batch

                # Move & cast ONLY on device per requirement
                x = x.to(config.device, dtype=torch.long, non_blocking=True)
                y = y.to(config.device, dtype=torch.long, non_blocking=True)

                # Apply ignore_index=-1 for padded targets on GPU using valid_len_y if provided
                if valid_y_len is not None:
                    if not torch.is_tensor(valid_y_len):
                        valid_y_len = torch.tensor(valid_y_len)
                    valid_y_len = valid_y_len.to(config.device)
                    Tlen = y.size(1)
                    ar = torch.arange(Tlen, device=config.device).unsqueeze(0)
                    pad_mask = ar >= valid_y_len.unsqueeze(1)
                    # Clone to avoid in-place on shared storage across workers
                    y = y.clone()
                    y[pad_mask] = -1

                # Tokens and samples accounting for throughput (local rank)
                # Count only valid tokens (ignore_index != -1)
                local_valid_tokens = int((y != -1).sum().item())
                tokens_since_log_local += local_valid_tokens
                samples_since_log_local += x.size(0)

                with autocast_ctx:
                    logits, loss_tuple, _, aux_list = model(x, targets=y, past_key_values=None)
                    # loss_tuple = (total, main, aux)
                    total_loss, main_loss, aux_loss = loss_tuple
                    # Scale for gradient accumulation
                    total_loss = total_loss / config.gradient_accumulation_steps

                # Accumulate for logs (use .item() after scaling)
                accum_total += float(total_loss.item())
                if main_loss is not None:
                    accum_main += float(main_loss.item()) / config.gradient_accumulation_steps
                if aux_loss is not None:
                    accum_aux += float(aux_loss.item()) / config.gradient_accumulation_steps

                # Capture first MoC layer routing indices when available (accumulate for windowed viz)
                if aux_list is not None and isinstance(aux_list, list) and len(aux_list) == 1:
                    indices_list = aux_list[0]
                    if indices_list:
                        step_expert_indices.append(indices_list[0].detach().to("cpu"))  # [B, T, K] from layer 0

                total_loss.backward()

        # Merge step indices into visualization window buffer
        if len(step_expert_indices) > 0:
            try:
                expert_indices_window.append(torch.cat(step_expert_indices, dim=0))
            except Exception:
                # Fallback if shapes mismatch across micros (shouldn't happen normally)
                expert_indices_window.extend(step_expert_indices)

        # Clip and step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if is_master:
            pbar.update(1)
            do_log = (current_step % config.log_interval == 0)
            if do_log:
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)

                # Aggregate tokens across all ranks for global throughput via all_reduce
                tokens_tensor = torch.tensor([tokens_since_log_local], device=config.device, dtype=torch.float64)
                if is_ddp:
                    torch.distributed.all_reduce(tokens_tensor, op=torch.distributed.ReduceOp.SUM)
                tokens_since_log_global = float(tokens_tensor.item())

                # Compute throughputs
                tok_per_s_global = tokens_since_log_global / elapsed
                tok_per_s_per_gpu = tok_per_s_global / max(1, world_size)
                samp_per_s_local = samples_since_log_local / elapsed

                # ETA
                steps_remaining = max(0, config.max_steps - current_step)
                # Average seconds per step over this window
                sec_per_step = elapsed / max(1, steps_since_log)
                eta_sec = steps_remaining * sec_per_step
                eta_h = int(eta_sec // 3600)
                eta_m = int((eta_sec % 3600) // 60)

                # Memory snapshot (CUDA only)
                mem_cur_gb = mem_max_gb = 0.0
                if device_type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_cur_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_max_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

                ppl = math.exp(accum_main) if accum_main < 20 else float('inf')
                pbar.set_postfix({
                    "loss": f"{accum_total:.3f}",
                    "main": f"{accum_main:.3f}",
                    "aux": f"{accum_aux:.4f}",
                    "ppl": f"{ppl:.2f}",
                    "lr": f"{lr:.2e}",
                    "gnorm": f"{float(grad_norm):.2f}",
                    "tok/s(g)": f"{tok_per_s_global:,.0f}",
                    "tok/s(gpu)": f"{tok_per_s_per_gpu:,.0f}",
                    "samp/s(gpu)": f"{samp_per_s_local:,.1f}",
                    "mem(GiB)": f"{mem_cur_gb:.2f}/{mem_max_gb:.2f}",
                    "ETA": f"{eta_h}h{eta_m:02d}m",
                })

                if wandb is not None:
                    log_data = {
                        "step": current_step,
                        "epoch": current_epoch,
                        "loss/total": accum_total,
                        "loss/main": accum_main,
                        "loss/aux": accum_aux,
                        "perplexity": ppl,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "throughput/tok_per_s_global": tok_per_s_global,
                        "throughput/tok_per_s_per_gpu": tok_per_s_per_gpu,
                        "throughput/samples_per_s_local": samp_per_s_local,
                        "timing/sec_per_step_window": sec_per_step,
                        "timing/eta_sec": eta_sec,
                        "mem/current_gib": mem_cur_gb,
                        "mem/max_gib": mem_max_gb,
                    }

                    # --- Architectural visual logs: Expert utilization & co-occurrence (layer 0) ---
                    # Aggregate indices across window
                    if len(expert_indices_window) > 0:
                        try:
                            all_idx = torch.cat(expert_indices_window, dim=0)  # [N, T, K]
                            E = int(getattr(raw_model.config, "n_experts", 0) or 0)
                            if E > 0:
                                flat_idx = all_idx.reshape(-1)
                                counts = torch.bincount(flat_idx, minlength=E).float()
                                total_pairs = counts.sum().clamp_min(1.0)
                                util = (counts / total_pairs).cpu().numpy()

                                # W&B scalars for quick glance
                                for i in range(E):
                                    log_data[f"experts/util_layer0/e{i}"] = float(util[i])

                                # Pairwise co-occurrence heatmap when top_k >= 2
                                k_sel = all_idx.shape[-1]
                                if k_sel >= 2 and plt is not None and wandb is not None:
                                    cooc = torch.zeros(E * E, dtype=torch.int64)
                                    idx2d = all_idx.view(-1, k_sel)  # [M, K]
                                    for i in range(k_sel):
                                        for j in range(i + 1, k_sel):
                                            u = idx2d[:, i]
                                            v = idx2d[:, j]
                                            flat = (u * E + v)
                                            co = torch.bincount(flat, minlength=E * E)
                                            cooc += co
                                            # Symmetrize (u,v) and (v,u)
                                            cooc += co.reshape(E, E).t().reshape(-1)
                                    cooc_mat = cooc.reshape(E, E).cpu().numpy()

                                    # Plot utilization bar and co-occurrence heatmap side-by-side
                                    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
                                    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

                                    ax0 = fig.add_subplot(gs[0, 0])
                                    ax0.bar(np.arange(E), util)
                                    ax0.set_title("Layer 0: Expert Utilization (window)")
                                    ax0.set_xlabel("Expert id")
                                    ax0.set_ylabel("Fraction of routed pairs")

                                    ax1 = fig.add_subplot(gs[0, 1])
                                    im = ax1.imshow(cooc_mat, aspect="auto", interpolation="nearest")
                                    ax1.set_title("Layer 0: Expert Pair Co-occurrence")
                                    ax1.set_xlabel("Expert v")
                                    ax1.set_ylabel("Expert u")
                                    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

                                    log_data["viz/layer0_expert_util_cooc"] = wandb.Image(fig)
                                    plt.close(fig)

                        except Exception as viz_err:
                            # Non-fatal: annotate in logs for visibility
                            log_data["viz/error"] = str(viz_err)

                    # Log constant active params per token periodically for visibility
                    active_params, total_params = compute_active_params_per_token(raw_model)
                    log_data["meta/active_params_per_token"] = active_params
                    log_data["meta/total_trainable_params"] = total_params
                    log_data["meta/active_params_ratio"] = active_params / max(1, total_params)

                    wandb.log(log_data)

                # Reset window accumulators
                last_log_time = now
                tokens_since_log_local = 0
                samples_since_log_local = 0
                steps_since_log = 0
                expert_indices_window.clear()

        # Checkpointing (save twice per requirement)
        if is_master and current_step % config.save_interval == 0:
            ckpt = {
                'model': (raw_model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'config': asdict(config),
                'step': current_step,
                'epoch': current_epoch,
            }
            # 1) Numbered checkpoint
            save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
            torch.save(ckpt, save_path)
            # 2) Latest checkpoint (always save, regardless of config.save_latest_always)
            latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
            torch.save(ckpt, latest_path)
            print(f"\n[CKPT] Saved checkpoints: {save_path}  |  {latest_path}")

    # Finalize
    if is_master:
        print("\n[TRAIN] Max steps reached. Finishing.")
        if wandb is not None:
            wandb.finish()
        pbar.close()
    if is_ddp:
        destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex MoC model (top-k collaborative experts).")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    train(args.config)

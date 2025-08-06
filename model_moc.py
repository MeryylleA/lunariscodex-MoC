import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LunarisCodexConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    vocab_size: int = 50257
    multiple_of: int = 256
    ffn_hidden_multiplier: float = 4.0
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0

    # MoE / MoC parameters
    n_experts: Optional[int] = 8
    top_k: int = 2  # fixed top-k > 1
    aux_loss_weight: float = 1e-2
    capacity_factor: float = 1.25
    router_z_loss_weight: float = 1e-3

    # Engineering
    use_gradient_checkpointing: bool = True  # safe default for single-GPU
    save_attn_weights: bool = False          # for debugging collaboration

    # New simple collab flags (default keeps legacy MHA behavior)
    use_simple_collab: bool = False
    simple_collab_dropout: float = 0.1


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)  # device set when registered as buffer
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: [B, H, T, D]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.to(dtype=xq.dtype), xk_out.to(dtype=xk.dtype)


class Attention(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        self.wqkv = nn.Linear(config.d_model, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # QK-Reorder-LN: Norm only Q and K
        self.q_norm = nn.RMSNorm(q_size, eps=1e-5)
        self.k_norm = nn.RMSNorm(kv_size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        qkv = self.wqkv(x)
        q, k, v = torch.split(
            qkv,
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present_kv = (k, v)

        if self.n_kv_heads < self.n_heads:
            n_repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))
        return y, present_kv


class FeedForward(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # Fused SwiGLU-style MLP (w13 chunk)
        self.w13 = nn.Linear(config.d_model, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        gate, up = self.w13(x).chunk(2, dim=-1)
        swiglu = F.silu(gate) * up
        return self.dropout(self.w2(swiglu))


class MoCTopKExperts(nn.Module):
    """
    MoC core on top of optimized MoE foundation.
    Key merges from model_moe.py:
      - Capacity-aware dispatch with contiguous permutation
      - Router z-loss and load-balancing loss
      - No probability scaling of expert outputs before fusion
      - Fused FFN per expert
    MoC additions from model_moc.py (core only):
      - top_k > 1 routing (fixed K)
      - cross-attention collaboration among K expert outputs per token (single round)
      - weighted fusion by normalized top-k logits

    New simple collaboration (when config.use_simple_collab=True):
      Two-pass shared-parameter collaboration over the K expert outputs per token, without multihead attention:
        Inputs per token: selected in R^{K x D}, topk_probs in R^{K}, kept_mask in {0,1}^{K}
        1) Message pass: M = msg_proj(selected); Q = q_proj(selected); Kk = k_proj(M)
           scores = Q @ Kk^T / sqrt(D), masked on dropped entries; A = softmax(scores, dim=-1)
           C = A @ M
        2) Update MLP: refined = selected + upd([selected, C])
        3) Fusion: mask dropped entries, renormalize weights over kept positions, and sum.
      Shapes:
        selected: [N, K, D]; M, Q, Kk, C, refined: [N, K, D]; scores, A: [N, K, K]; fused: [N, D]
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.n_experts is not None and config.n_experts > 0
        assert config.top_k >= 1
        self.n_experts = int(config.n_experts)
        self.top_k = int(config.top_k)
        self.aux_loss_weight = config.aux_loss_weight
        self.capacity_factor = config.capacity_factor
        self.z_loss_weight = config.router_z_loss_weight
        self.config = config

        # Router gate
        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_experts)])

        # Legacy collaboration sub-layer: MHA over K selected outputs per token
        attn_heads = min(max(1, self.top_k), config.n_heads)
        self.collab_attn = nn.MultiheadAttention(
            embed_dim=config.d_model, num_heads=attn_heads, dropout=config.dropout, batch_first=True
        )
        self.collab_norm = nn.RMSNorm(config.d_model, eps=1e-5)
        self.collab_ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model, bias=False),
        )

        # Output projection (shared for both paths)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # New simple collaboration shared-parameter projections and update MLP
        # Always define to keep state dict stable; used only if use_simple_collab=True.
        D = config.d_model
        self.msg_proj = nn.Linear(D, D, bias=False)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.upd = nn.Sequential(
            nn.Linear(2 * D, 2 * D, bias=False),
            nn.GELU(),
            nn.Dropout(config.simple_collab_dropout),
            nn.Linear(2 * D, D, bias=False),
        )

        self.save_attn_weights = bool(getattr(config, "save_attn_weights", False))

    @staticmethod
    def _load_balance_loss(router_probs: torch.Tensor, hard_assign: torch.Tensor, n_experts: int) -> torch.Tensor:
        # router_probs: [N, E] float32
        # hard_assign: [N] long
        prob_mass = router_probs.mean(dim=0)  # [E]
        tokens_one_hot = F.one_hot(hard_assign, num_classes=n_experts).to(torch.float32)
        fraction_tokens = tokens_one_hot.mean(dim=0)  # [E]
        return (prob_mass * fraction_tokens).sum() * n_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D]
        Returns:
          y: fused output [B, T, D]
          aux_loss: scalar
          expert_indices: [B, T, K] per token selected experts (for logging)
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Router in fp32 for stability
        logits = self.gate(x_flat.to(torch.float32))  # [N, E]
        # top-k selection (no gumbel; deterministic and stable)
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [N, K]
        # normalize over the K selected experts to get fusion weights
        topk_probs = F.softmax(topk_vals, dim=-1, dtype=torch.float32)  # [N, K]

        # Auxiliary losses (Switch-style)
        # For balance loss, use top-1 hard assignments from logits argmax
        top1 = torch.argmax(logits, dim=-1)  # [N]
        router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # [N, E]
        balance_loss = self._load_balance_loss(router_probs, top1, self.n_experts)
        z = torch.logsumexp(logits, dim=-1)  # [N]
        z_loss = z.pow(2).mean()
        aux_loss = self.aux_loss_weight * balance_loss + self.z_loss_weight * z_loss

        # Dispatch efficiently to experts using contiguous permutation extended to top-k:
        # Expand tokens by K, sort by target expert ids, process contiguous segments, and write back.
        N_K = N * self.top_k
        # expanded inputs for each token-k pair
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(N_K, D)  # [N*K, D]
        target_expert = topk_idx.reshape(-1)  # [N*K]
        # Sort by expert for contiguous segments
        sorted_expert, sort_idx = torch.sort(target_expert)
        x_perm = x_expanded.index_select(0, sort_idx)  # [N*K, D]
        counts = torch.bincount(sorted_expert, minlength=self.n_experts)  # [E]

        # Capacity per expert: scale with factor, based on average N*K / E
        C = int(math.ceil((N_K / max(1, self.n_experts)) * self.capacity_factor))
        keep_counts = torch.clamp(counts, max=C)

        # Offsets
        offsets = torch.cumsum(F.pad(counts, (1, 0)), dim=0)        # [E+1]
        # We only need kept segments (actual compute)
        keep_offsets = torch.cumsum(F.pad(keep_counts, (1, 0)), dim=0)  # [E+1]

        # Prepare output buffer for selected expert outputs
        expert_out_selected = x_expanded.new_zeros((N_K, D))  # [N*K, D]
        keep_mask = torch.zeros(N_K, dtype=torch.bool, device=x.device)

        # Compute per-expert segments
        for i in range(self.n_experts):
            start_all = int(offsets[i].item())
            cnt = int(counts[i].item())
            kept = int(keep_counts[i].item())
            if kept == 0:
                continue
            seg = x_perm[start_all:start_all + kept]  # [kept, D]
            y = self.experts[i](seg)  # [kept, D]
            idx_slice = sort_idx[start_all:start_all + kept]  # indices into [N*K]
            expert_out_selected.index_copy_(0, idx_slice, y)
            keep_mask.index_copy_(0, idx_slice, torch.ones(kept, dtype=torch.bool, device=keep_mask.device))

        # Reshape back to [N, K, D] with zeros where dropped
        selected = expert_out_selected.view(N, self.top_k, D)  # [N, K, D]
        kept_mask_nk = keep_mask.view(N, self.top_k)           # [N, K]

        if self.config.use_simple_collab:
            # -------------------------------
            # Simple 2-pass collaboration block
            # -------------------------------
            # a) Messages and attention over K (per token)
            # Maintain computation dtype of selected for AMP safety, upcast scores to float32 for stability if needed.
            dtype = selected.dtype
            M = self.msg_proj(selected)    # [N, K, D]
            Q = self.q_proj(selected)      # [N, K, D]
            Kk = self.k_proj(M)            # [N, K, D]

            # Compute scaled dot-product scores over K
            # Upcast to float32 for the softmax, then cast back.
            scores = torch.matmul(Q.to(torch.float32), Kk.to(torch.float32).transpose(1, 2))
            scores = scores / math.sqrt(D)

            # Mask rows/cols corresponding to dropped expert-token pairs.
            # Build [N, K, K] mask where an entry is valid only if both row and col are kept.
            km = kept_mask_nk
            if km.dtype != torch.bool:
                km = km.to(torch.bool)
            row_mask = km.unsqueeze(2)  # [N, K, 1]
            col_mask = km.unsqueeze(1)  # [N, 1, K]
            valid_pairs = row_mask & col_mask  # [N, K, K]
            scores = scores.masked_fill(~valid_pairs, -1e9)

            # Softmax across last dimension (along keys); for rows with no kept cols, softmax over all -1e9 gives NaNs.
            # Guard by setting rows with all False cols to zero after softmax using the row_mask.
            A = F.softmax(scores, dim=-1, dtype=torch.float32)  # [N, K, K], float32
            A = A * valid_pairs.to(A.dtype)
            denomA = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [N, K, 1]
            A = (A / denomA).to(dtype)  # back to original dtype

            # C = A @ M
            C = torch.matmul(A, M)  # [N, K, D]

            # b) Update using shared MLP
            refined = selected + self.upd(torch.cat([selected, C], dim=-1))  # [N, K, D]

            # c) Mask and fuse using router weights with renormalization over kept positions
            refined = refined * kept_mask_nk.unsqueeze(-1)  # zero dropped
            topk_probs_ = topk_probs.to(refined.dtype) * kept_mask_nk.to(topk_probs.dtype)
            denom = topk_probs_.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [N, 1]
            weights = (topk_probs_ / denom).to(refined.dtype)  # [N, K]
            fused = torch.sum(refined * weights.unsqueeze(-1), dim=1)  # [N, D]
            fused = self.o_proj(fused).view(B, T, D)

            expert_indices = topk_idx.view(B, T, self.top_k)
            return fused, aux_loss.to(fused.dtype), expert_indices

        # Legacy collaboration: cross-attention among K expert outputs per token (unchanged)
        x_collab_in = selected
        need_w = self.save_attn_weights
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                attn_out, attn_weights = self.collab_attn(x_collab_in, x_collab_in, x_collab_in, need_weights=need_w)
        except Exception:
            out = self.collab_attn(x_collab_in, x_collab_in, x_collab_in, need_weights=need_w)
            if need_w:
                attn_out, attn_weights = out
            else:
                attn_out = out[0] if isinstance(out, tuple) else out
                attn_weights = None

        # Residual + RMSNorm + FFN
        collab = self.collab_norm(attn_out + x_collab_in)
        refined = self.collab_ffn(collab) + collab  # [N, K, D]

        # Weighted fusion using top-k probabilities; zero-out dropped pairs
        topk_probs_ = topk_probs.to(refined.dtype)  # [N, K]
        refined = refined * kept_mask_nk.unsqueeze(-1)  # mask dropped
        topk_probs_ = topk_probs_ * kept_mask_nk.to(topk_probs_.dtype)
        denom = topk_probs_.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # re-normalize if houve drop
        weights = topk_probs_ / denom
        fused = torch.sum(refined * weights.unsqueeze(-1), dim=1)  # [N, D]
        fused = self.o_proj(fused).view(B, T, D)

        expert_indices = topk_idx.view(B, T, self.top_k)
        return fused, aux_loss.to(fused.dtype), expert_indices


class Block(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-5)

        if config.n_experts is not None and config.n_experts > 0 and config.top_k >= 1:
            self.feed_forward = MoCTopKExperts(config)
            self.is_moe = True
            print(f"Block initialized with MoC-on-MoE: {config.n_experts} experts, top_k={config.top_k}.")
        else:
            self.feed_forward = FeedForward(config)
            self.is_moe = False
            print("Block initialized with standard FeedForward network.")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        def _inner_forward(
            x_inner: torch.Tensor,
            freqs_cis_inner: torch.Tensor,
            past_kv_inner: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ):
            attn_output, new_kv = self.attention(x_inner, freqs_cis_inner, past_kv_inner)
            h = x_inner + attn_output

            aux_loss = None
            expert_indices = None
            ffn_input = self.ffn_norm(h)

            if self.is_moe:
                ffn_output, aux_loss, expert_indices = self.feed_forward(ffn_input)
            else:
                ffn_output = self.feed_forward(ffn_input)

            out = h + ffn_output
            return out, new_kv, aux_loss, expert_indices

        if self.training:
            return checkpoint(_inner_forward, x, freqs_cis, past_kv, use_reentrant=False)
        else:
            return _inner_forward(x, freqs_cis, past_kv)


class LunarisCodex(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.RMSNorm(config.d_model, eps=1e-5),
            drop=nn.Dropout(config.dropout),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        freqs_cis = precompute_freqs_cis(
            self.config.d_model // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

        num_params = self.get_num_params()
        print(f"Number of parameters: {num_params/1e6:.2f}M")
        if config.n_experts is not None and config.n_experts > 0:
            print("Note: Parameter count includes all experts. Only top_k experts per token are active at runtime.")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale output weights similar to model_moe.py
        if isinstance(module, (Attention, FeedForward, MoCTopKExperts)):
            for name, p in module.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[tuple],
        List[Tuple[torch.Tensor, torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        B, T = idx.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        assert start_pos + T <= self.config.max_seq_len, \
            f"Sequence length {start_pos + T} exceeds model's max_seq_len {self.config.max_seq_len}"

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        freqs_cis = self.freqs_cis[start_pos: start_pos + T].to(dtype=torch.float32, device=x.device)

        new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_aux_loss = x.new_zeros(())
        expert_indices_list = []

        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv, aux_loss, expert_indices = block(x, freqs_cis, past_kv_for_block)

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss.to(x.dtype)

            if expert_indices is not None:
                expert_indices_list.append(expert_indices)

            new_past_key_values.append(new_kv)

        x = self.transformer.ln_f(x)

        loss = None
        logits = self.lm_head(x) if targets is not None else self.lm_head(x[:, [-1], :])

        if targets is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            num_moe_layers = sum(1 for block in self.transformer.h if getattr(block, 'is_moe', False))
            final_aux_loss = total_aux_loss
            if num_moe_layers > 0:
                final_aux_loss = final_aux_loss / num_moe_layers

            total_loss = main_loss + final_aux_loss.to(main_loss.dtype)
            loss = (total_loss, main_loss, final_aux_loss)

            return logits, loss, new_past_key_values, [expert_indices_list]
        else:
            return logits, None, new_past_key_values, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Separate router gate parameters with a lower LR, keep decay splits
        router_params, decay_params, nodecay_params = [], [], []
        router_param_ids = set()

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'feed_forward.gate' in n and isinstance(p, torch.nn.Parameter):
                router_params.append(p)
                router_param_ids.add(id(p))
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        # Additionally catch MoCTopKExperts gate path (robust to naming)
        for n, p in self.named_parameters():
            if n.endswith('feed_forward.gate.weight'):
                if id(p) not in router_param_ids:
                    router_params.append(p)
                    router_param_ids.add(id(p))

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': router_params, 'weight_decay': 0.0, 'lr': learning_rate * 0.5},
        ]

        print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
        print(f"num router parameter tensors: {len(router_params)}, with {sum(p.numel() for p in router_params):,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0},
                {'params': router_params, 'weight_decay': 0.0, 'lr': learning_rate * 0.5},
            ],
            lr=learning_rate,
            betas=betas,
            fused=use_fused
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        past_key_values = None
        for _ in range(max_new_tokens):
            current_len = past_key_values[0][0].shape[-2] if past_key_values else idx.shape[1]
            if current_len >= self.config.max_seq_len:
                break
            idx_cond = idx if past_key_values is None else idx[:, -1:]
            logits, _, past_key_values, _ = self(idx_cond, past_key_values=past_key_values)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


def compile_model_if_available(model: nn.Module):
    try:
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled with torch.compile (max-autotune).")
    except Exception as e:
        print(f"torch.compile not enabled or failed: {e}")
    return model


if __name__ == "__main__":
    # Minimal sanity tests for both collaboration modes.
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_sanity(use_simple: bool):
        print(f"\nRunning sanity with use_simple_collab={use_simple}")
        cfg = LunarisCodexConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            vocab_size=503,
            multiple_of=16,
            ffn_hidden_multiplier=4.0,
            max_seq_len=64,
            dropout=0.0,
            n_experts=4,
            top_k=2,
            aux_loss_weight=1e-2,
            capacity_factor=1.25,
            router_z_loss_weight=1e-3,
            use_gradient_checkpointing=False,
            save_attn_weights=False,
            use_simple_collab=use_simple,
            simple_collab_dropout=0.1,
        )
        model = LunarisCodex(cfg).to(device)
        model.train()

        B, T = 2, 8
        idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

        logits, loss_tuple, _, expert_info = model(idx, targets=targets)
        total_loss, ce_loss, aux_loss = loss_tuple
        print(f"Losses -> total: {total_loss.item():.4f}, ce: {ce_loss.item():.4f}, aux: {aux_loss.item():.6f}")
        assert logits.shape == (B, T, cfg.vocab_size)
        assert isinstance(expert_info, list)

        # Check inference/generate
        model.eval()
        start = torch.randint(0, cfg.vocab_size, (B, 4), device=device)
        out = model.generate(start, max_new_tokens=5, temperature=1.0, top_k=20)
        print("Generate output shape:", out.shape)
        assert out.shape[1] >= start.shape[1]

    # Run for both modes
    run_sanity(use_simple=False)
    run_sanity(use_simple=True)
    print("\nSanity tests completed.")
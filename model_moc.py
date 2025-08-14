import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# -----------------------------------------------------------------------------
# Global backend hints for H100/GH200 (safe no-ops on CPU)
# -----------------------------------------------------------------------------
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
except Exception:
    pass

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class LunarisCodexConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    vocab_size: int = 50257
    multiple_of: int = 256
    ffn_hidden_multiplier: float = 4.0
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # MoE / MoC parameters
    n_experts: Optional[int] = 8
    top_k: int = 2
    aux_loss_weight: float = 1e-2
    capacity_factor: float = 1.25
    router_z_loss_weight: float = 1e-3
    router_noise_std: float = 0.0
    drop_penalty_weight: float = 1e-3
    # Engineering
    use_gradient_checkpointing: bool = True
    grad_ckpt_policy: str = "ffn"  # "none" | "ffn" | "block"
    save_attn_weights: bool = False
    # Collaboration modes
    use_simple_collab: bool = False        # set False: default to MoC collaborative path
    use_moc_collab: bool = True            # explicit flag (kept for clarity)
    simple_collab_dropout: float = 0.1
    moc_collab_steps: int = 2              # rounds of expert<->expert message passing
    moc_collab_heads: int = 4              # heads for expert-level attention
    moc_collab_dropout: float = 0.0        # dropout inside MoC collaboration
    moc_use_mediator: bool = True          # add a learnable mediator token per token
    # IRL: iterative reasoning steps inside each FFN/expert
    n_reasoning_steps: int = 1

# -----------------------------------------------------------------------------
# Rotary Embeddings
# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: [B, H, T, D]
    B, H, T, D = xq.shape
    xq_ = torch.view_as_complex(xq.reshape(B, H, T, D // 2, 2).to(torch.float32))
    xk_ = torch.view_as_complex(xk.reshape(B, H, T, D // 2, 2).to(torch.float32))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2], fp32
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(B, H, T, D).to(dtype=xq.dtype)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(B, H, T, D).to(dtype=xk.dtype)
    return xq_out, xk_out

# -----------------------------------------------------------------------------
# Attention (Flash SDP, GQA)
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        assert config.n_heads % config.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads (GQA)"
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.wqkv = nn.Linear(config.d_model, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.wqkv(x)  # [B, T, (H+2*Hkv)*D]
        q, k, v = torch.split(
            qkv,
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        # heads-first
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, T, D]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE (fp32 angles)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # KV cache append
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_kv = (k, v)

        # GQA: expand KV heads if needed
        if self.n_kv_heads < self.n_heads:
            n_repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)

        # Causal attention with Flash SDP; dropout only in training
        attn_dropout = self.dropout.p if self.training and self.dropout.p > 0 else 0.0
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_dropout)  # [B, H, T, D]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))
        return y, present_kv

# -----------------------------------------------------------------------------
# Reasoning FFN (IRL inside)
# -----------------------------------------------------------------------------
class ReasoningFeedForward(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        # Fused SwiGLU-style MLP (w13 chunk)
        self.w13 = nn.Linear(config.d_model, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # IRL steps & residual scaling
        self.n_reasoning_steps = int(getattr(config, "n_reasoning_steps", 1))
        self.alpha = 1.0 / math.sqrt(max(1, self.n_reasoning_steps))

    def _ffn_logic(self, z: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(z).chunk(2, dim=-1)
        swiglu = F.silu(gate) * up
        return self.dropout(self.w2(swiglu))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        steps = max(1, self.n_reasoning_steps)
        for _ in range(steps):
            h = h + self.alpha * self._ffn_logic(h + x)
        return h

# -----------------------------------------------------------------------------
# Helper to pick a valid head count for MoC collaboration
# -----------------------------------------------------------------------------

def _pick_heads(d_model: int, preferred: int) -> int:
    h = min(preferred, 8)
    while h > 1 and d_model % h != 0:
        h -= 1
    return max(1, h)

# -----------------------------------------------------------------------------
# MoC-on-Top-K Experts (vectorized routing + collaborative message passing)
# -----------------------------------------------------------------------------
class MoCTopKExperts(nn.Module):
    """
    Research-first MoC core: vectorized sorted routing + capacity + *collaboration between experts*.
    - Router/aux in fp32 with optional noise.
    - Experts are IRL FFNs.
    - Collaboration: R rounds of MHA+FFN across the K selected experts (+ optional mediator token),
      then gated fusion between mediator and weighted expert aggregate.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.n_experts is not None and config.n_experts > 0
        assert config.top_k >= 1
        self.n_experts = int(config.n_experts)
        self.top_k = int(config.top_k)
        self.aux_loss_weight = float(config.aux_loss_weight)
        self.capacity_factor = float(config.capacity_factor)
        self.z_loss_weight = float(config.router_z_loss_weight)
        self.drop_penalty_weight = float(getattr(config, "drop_penalty_weight", 0.0))
        self.config = config

        # Router gate (fp32 compute later)
        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([ReasoningFeedForward(config) for _ in range(self.n_experts)])

        D = config.d_model
        # Collaboration stack
        h = _pick_heads(D, config.moc_collab_heads)
        self.collab_attn = nn.MultiheadAttention(embed_dim=D, num_heads=h, dropout=config.moc_collab_dropout, batch_first=True)
        self.collab_norm1 = nn.RMSNorm(D, eps=1e-6)
        self.collab_ffn = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        self.collab_norm2 = nn.RMSNorm(D, eps=1e-6)
        self.moc_collab_steps = max(1, int(config.moc_collab_steps))
        if config.moc_use_mediator:
            self.mediator = nn.Parameter(torch.empty(1, 1, D))
            nn.init.normal_(self.mediator, mean=0.0, std=0.02)
        else:
            self.mediator = None
        # Gating between mediator and weighted expert sum
        self.fuse_gate = nn.Linear(D, 1, bias=True)

        # Output projection
        self.o_proj = nn.Linear(D, D, bias=False)

        # Simple path preserved for compatibility when explicitly requested
        self.msg_proj = nn.Linear(D, D, bias=False)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)

        self.save_attn_weights = bool(getattr(config, "save_attn_weights", False))
        self.last_attn_weights = None

    @staticmethod
    def _load_balance_loss_topk(router_probs, topk_idx, topk_probs, n_experts: int) -> torch.Tensor:
        N, E = router_probs.shape
        assign = router_probs.new_zeros(N, E)
        for j in range(topk_idx.size(1)):
            assign.scatter_add_(1, topk_idx[:, j:j + 1], topk_probs[:, j:j + 1])
        fraction_tokens = assign.mean(dim=0)  # [E]
        prob_mass = router_probs.mean(dim=0)  # [E]
        return (prob_mass * fraction_tokens).sum() * n_experts

    def _capacity_limit(self, total_slots: int) -> int:
        avg = total_slots / max(1, self.n_experts)
        target = int(math.ceil(avg * self.capacity_factor))
        if target <= 0:
            return 0
        return 1 << int(math.ceil(math.log2(target)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Router in fp32 (+ optional noise during training)
        logits = self.gate(x_flat.to(torch.float32))  # [N, E]
        if self.training and getattr(self.config, "router_noise_std", 0.0) > 0.0:
            logits = logits + torch.randn_like(logits) * float(self.config.router_noise_std)

        # Top-k (dispatch) and fp32 probs for losses
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)               # [N, K]
        topk_probs = F.softmax(topk_vals, dim=-1, dtype=torch.float32)               # [N, K]
        router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)                # [N, E]

        balance_loss = self._load_balance_loss_topk(router_probs, topk_idx, topk_probs, self.n_experts)
        z = torch.logsumexp(logits, dim=-1)
        z_loss = (z * z).mean()
        aux_loss = self.aux_loss_weight * balance_loss + self.z_loss_weight * z_loss

        # Expand dispatch by K
        N_K = N * self.top_k
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(N_K, D)  # [N*K, D]
        target_expert = topk_idx.reshape(-1)                                         # [N*K]
        prio = topk_vals.reshape(-1)                                                 # [N*K]

        # Capacity per expert (power-of-two)
        C = self._capacity_limit(N_K)

        # Outputs for per-expert path
        expert_out_selected = x_expanded.new_zeros((N_K, D))
        keep_mask = torch.zeros(N_K, dtype=torch.bool, device=x.device)
        total_dropped = 0

        # Sorted routing
        idx_sorted = torch.argsort(target_expert)
        counts = torch.bincount(target_expert, minlength=self.n_experts)
        seg_ends = counts.cumsum(0)
        seg_starts = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long), seg_ends[:-1]])

        for e in range(self.n_experts):
            s = int(seg_starts[e].item()); t = int(seg_ends[e].item())
            if t <= s:
                continue
            span = idx_sorted[s:t]
            n_e = int(span.numel())
            if C <= 0:
                total_dropped += n_e
                continue
            if n_e > C:
                prio_e = prio.index_select(0, span)
                keep_local = torch.topk(prio_e, k=C, largest=True, sorted=False).indices
                span = span.index_select(0, keep_local)
                total_dropped += n_e - C

            seg_x = x_expanded.index_select(0, span)
            y = self.experts[e](seg_x)
            expert_out_selected.index_copy_(0, span, y)
            keep_mask.index_fill_(0, span, True)

        # Drop penalty
        if self.drop_penalty_weight > 0.0 and N_K > 0:
            drop_frac = float(total_dropped) / float(N_K)
            aux_loss = aux_loss + x_flat.new_tensor(self.drop_penalty_weight * drop_frac, dtype=aux_loss.dtype)

        # Arrange [N, K, D] and masks
        expert_states = expert_out_selected.view(N, self.top_k, D)
        kept_mask_nk = keep_mask.view(N, self.top_k)
        expert_states = expert_states * kept_mask_nk.unsqueeze(-1)

        # >>> PATCH: normalização estável em fp32 + fallback quando nenhum expert foi mantido <<<
        topk_probs_masked_f32 = topk_probs * kept_mask_nk.to(topk_probs.dtype)            # [N, K], fp32
        denom_f32 = topk_probs_masked_f32.sum(dim=-1, keepdim=True)                       # [N, 1], fp32
        no_kept = (denom_f32 == 0)                                                        # [N, 1] bool
        safe_denom_f32 = torch.where(no_kept, torch.ones_like(denom_f32), denom_f32)
        weights_f32 = topk_probs_masked_f32 / safe_denom_f32                               # [N, K], fp32
        weights_f32 = torch.where(no_kept, torch.zeros_like(weights_f32), weights_f32)     # zera linhas sem kept
        weights = weights_f32.to(expert_states.dtype)
        # <<< PATCH END >>>

        # --- Collaboration paths ---
        if self.config.use_simple_collab and not self.config.use_moc_collab:
            # Lightweight fusion (legacy simple path)
            fused = torch.sum(expert_states * weights.unsqueeze(-1), dim=1)  # [N, D]
            fused = self.o_proj(fused).view(B, T, D)
            expert_indices = topk_idx.view(B, T, self.top_k).to(torch.long)
            return fused, aux_loss.to(fused.dtype), expert_indices

        # MoC collaborative message passing among experts (+ optional mediator)
        # tokens_in: [N, K(+1), D]
        tokens_in = expert_states
        key_padding_mask = ~kept_mask_nk  # True = ignore pad
        if self.mediator is not None:
            m = self.mediator.expand(N, 1, D).to(tokens_in.dtype)
            tokens_in = torch.cat([tokens_in, m], dim=1)
            pad_ext = torch.zeros(N, 1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            key_padding_mask = torch.cat([key_padding_mask, pad_ext], dim=1)
        # R rounds of Attn + FFN
        tokens = tokens_in
        need_w = self.save_attn_weights
        attn_w = None
        for _ in range(self.moc_collab_steps):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out, attn_w = self.collab_attn(tokens, tokens, tokens, need_weights=need_w, key_padding_mask=key_padding_mask)
            tokens = self.collab_norm1(tokens + out)
            tokens = tokens + self.collab_ffn(self.collab_norm2(tokens))
        if need_w:
            self.last_attn_weights = attn_w

        # Split back
        if self.mediator is not None:
            mediator_out = tokens[:, -1, :]
            expert_refined = tokens[:, :-1, :]
        else:
            mediator_out = tokens.mean(dim=1)
            expert_refined = tokens

        expert_refined = expert_refined * (~key_padding_mask[:, :self.top_k]).unsqueeze(-1)
        weighted = torch.sum(expert_refined * weights.unsqueeze(-1), dim=1)  # [N, D]

        # Gate between mediator and weighted expert aggregate
        gamma = torch.sigmoid(self.fuse_gate(x_flat)).view(N, 1)
        fused = gamma * mediator_out + (1.0 - gamma) * weighted

        fused = self.o_proj(fused).view(B, T, D)
        expert_indices = topk_idx.view(B, T, self.top_k).to(torch.long)
        return fused, aux_loss.to(fused.dtype), expert_indices

# -----------------------------------------------------------------------------
# Transformer Block (checkpoint policies: none/ffn/block)
# -----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        self.attn_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        if config.n_experts is not None and config.n_experts > 0 and config.top_k >= 1:
            self.feed_forward = MoCTopKExperts(config)
            self.is_moe = True
            print(f"Block with MoC collaboration: {config.n_experts} experts, top_k={config.top_k}.")
        else:
            self.feed_forward = ReasoningFeedForward(config)
            self.is_moe = False
            print("Block with standard ReasoningFeedForward.")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        def _inner_full(x_inner: torch.Tensor, freqs_cis_inner: torch.Tensor, past_kv_inner: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
            attn_output, new_kv = self.attention(self.attn_norm(x_inner), freqs_cis_inner, past_kv_inner)
            h = x_inner + attn_output
            ffn_input = self.ffn_norm(h)
            if self.is_moe:
                ffn_output, aux_loss, expert_indices = self.feed_forward(ffn_input)
            else:
                ffn_output = self.feed_forward(ffn_input)
                aux_loss = ffn_output.new_zeros(())
                expert_indices = torch.empty(0, dtype=torch.long, device=ffn_output.device)
            out = h + ffn_output
            return out, new_kv, aux_loss, expert_indices

        if self.training and self.config.use_gradient_checkpointing:
            if self.config.grad_ckpt_policy == "block":
                # PATCH: checkpoint sem passar tuplas não-tensor
                def _inner_ckpt(x_inner: torch.Tensor, freqs_cis_inner: torch.Tensor):
                    return _inner_full(x_inner, freqs_cis_inner, past_kv)
                return checkpoint(_inner_ckpt, x, freqs_cis, use_reentrant=False)
            elif self.config.grad_ckpt_policy == "ffn":
                attn_output, new_kv = self.attention(self.attn_norm(x), freqs_cis, past_kv)
                h = x + attn_output
                ffn_input = self.ffn_norm(h)

                def _ffn_only(ffn_in: torch.Tensor):
                    if self.is_moe:
                        y, aux, experts = self.feed_forward(ffn_in)
                    else:
                        y = self.feed_forward(ffn_in)
                        aux = y.new_zeros(())
                        experts = torch.empty(0, dtype=torch.long, device=y.device)
                    return y, aux, experts

                ffn_output, aux_loss, expert_indices = checkpoint(_ffn_only, ffn_input, use_reentrant=False)
                out = h + ffn_output
                return out, new_kv, aux_loss, (None if expert_indices.numel() == 0 else expert_indices)
            else:
                return _inner_full(x, freqs_cis, past_kv)
        else:
            return _inner_full(x, freqs_cis, past_kv)

# -----------------------------------------------------------------------------
# Top-level Model
# -----------------------------------------------------------------------------
class LunarisCodex(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.RMSNorm(config.d_model, eps=1e-6),
            drop=nn.Dropout(config.dropout),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # tie weights
        self.transformer.wte.weight = self.lm_head.weight
        freqs_cis = precompute_freqs_cis(
            self.config.d_model // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        # Init
        self.apply(self._init_weights)
        self._rescale_out_projections()
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

    def _rescale_out_projections(self):
        # Scale only output projections: Attention.o_proj, FFN.w2, MoC.o_proj
        denom = math.sqrt(2 * self.config.n_layers)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, Attention):
                    m.o_proj.weight.mul_(1.0 / denom)
                elif isinstance(m, ReasoningFeedForward):
                    m.w2.weight.mul_(1.0 / denom)
                elif isinstance(m, MoCTopKExperts):
                    m.o_proj.weight.mul_(1.0 / denom)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[tuple], List[Tuple[torch.Tensor, torch.Tensor]], Optional[List[torch.Tensor]]]:
        B, T = idx.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        assert start_pos + T <= self.config.max_seq_len, \
            f"Sequence length {start_pos + T} exceeds model's max_seq_len {self.config.max_seq_len}"

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        freqs_cis = self.freqs_cis[start_pos: start_pos + T].to(dtype=torch.float32, device=x.device)

        new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_aux_loss = x.new_zeros(())
        expert_indices_list: List[torch.Tensor] = []

        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv, aux_loss, expert_indices = block(x, freqs_cis, past_kv_for_block)
            total_aux_loss = total_aux_loss + aux_loss.to(x.dtype)
            if expert_indices is not None:
                expert_indices_list.append(expert_indices)
            new_past_key_values.append(new_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) if targets is not None else self.lm_head(x[:, [-1], :])

        loss = None
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
        # Separate router params (lower LR, no WD); keep stable naming
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
        for n, p in self.named_parameters():
            if n.endswith('feed_forward.gate.weight'):
                if id(p) not in router_param_ids:
                    router_params.append(p)
                    router_param_ids.add(id(p))
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
            fused=use_fused,
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
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
        if was_training:
            self.train()
        return idx

# -----------------------------------------------------------------------------
# Compile helper (kept for training compatibility)
# -----------------------------------------------------------------------------

def compile_model_if_available(model: nn.Module, mode: str = "max-autotune"):
    try:
        model = torch.compile(model, mode=mode)
        print(f"Model compiled with torch.compile ({mode}).")
    except Exception as e:
        print(f"torch.compile not enabled or failed: {e}")
    return model


if __name__ == "__main__":
    # Minimal sanity tests
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        router_noise_std=0.02,
        drop_penalty_weight=1e-3,
        use_gradient_checkpointing=True,
        grad_ckpt_policy="ffn",
        save_attn_weights=False,
        use_simple_collab=False,
        use_moc_collab=True,
        simple_collab_dropout=0.1,
        moc_collab_steps=2,
        moc_collab_heads=4,
        moc_collab_dropout=0.0,
        moc_use_mediator=True,
        n_reasoning_steps=2,
    )
    model = LunarisCodex(cfg).to(device)

    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    model.train()
    logits, loss_tuple, _, expert_info = model(idx, targets=targets)
    total_loss, ce_loss, aux_loss = loss_tuple
    print(f"Losses -> total: {total_loss.item():.4f}, ce: {ce_loss.item():.4f}, aux: {aux_loss.item():.6f}")
    assert logits.shape == (B, T, cfg.vocab_size)

    model.eval()
    out = model.generate(idx[:, :4], max_new_tokens=5)
    print("Generate output shape:", out.shape)
    print("Sanity complete.")

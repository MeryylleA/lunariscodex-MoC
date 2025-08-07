import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Global backend hints for H100/GH200
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
except Exception:
    pass

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
    router_noise_std: float = 0.0          # training-time router noise; e.g., 0.01–0.1
    drop_penalty_weight: float = 1e-3      # penalize overflow drops
    # Engineering
    use_gradient_checkpointing: bool = True
    grad_ckpt_policy: str = "ffn"          # "none" | "ffn" | "block"
    save_attn_weights: bool = False
    # Simple collab flags
    use_simple_collab: bool = True
    simple_collab_dropout: float = 0.1
    # IRL: number of iterative reasoning steps inside each expert FFN
    n_reasoning_steps: int = 1

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
        # Reshape to heads-first
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, T, D]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)  # [B, H, T, D], [B, Hkv, T, D]

        # KV cache append
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_kv = (k, v)

        # GQA: repeat KV heads if needed
        if self.n_kv_heads < self.n_heads:
            n_repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)

        # Causal attention with Flash SDP; pass dropout in training
        attn_dropout = self.dropout.p if self.training and self.dropout.p > 0 else 0.0
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_dropout)  # [B, H, T, D]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))
        return y, present_kv

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
        # IRL steps e escala residual
        self.n_reasoning_steps = int(getattr(config, "n_reasoning_steps", 1))
        self.alpha = 1.0 / math.sqrt(max(1, self.n_reasoning_steps))

    def _ffn_logic(self, z: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(z).chunk(2, dim=-1)
        swiglu = F.silu(gate) * up
        return self.dropout(self.w2(swiglu))

    def forward(self, x: torch.Tensor):
        h = x
        steps = max(1, self.n_reasoning_steps)
        for _ in range(steps):
            h = h + self.alpha * self._ffn_logic(h + x)
        return h

class MoCTopKExperts(nn.Module):
    """
    MoC core on top of optimized MoE foundation with simple collaboration by default.
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
        self.experts = nn.ModuleList([ReasoningFeedForward(config) for _ in range(self.n_experts)])

        # Legacy collaboration sub-layer (kept for compatibility)
        D = config.d_model
        attn_heads = min(max(1, self.top_k), config.n_heads)
        if D % attn_heads != 0:
            attn_heads = 1  # garante divisibilidade
        self.collab_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=attn_heads, dropout=config.dropout, batch_first=True
        )
        self.collab_norm = nn.RMSNorm(D, eps=1e-6)
        self.collab_ffn = nn.Sequential(
            nn.Linear(D, D, bias=False),
            nn.GELU(),
            nn.Linear(D, D, bias=False),
        )
        # Output projection
        self.o_proj = nn.Linear(D, D, bias=False)

        # Simple collaboration projections
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
        self.last_attn_weights = None

    @staticmethod
    def _load_balance_loss_topk(
        router_probs: torch.Tensor, topk_idx: torch.Tensor, topk_probs: torch.Tensor, n_experts: int
    ) -> torch.Tensor:
        # router_probs: [N, E] float32
        # topk_idx/topk_probs: [N, K]
        N, E = router_probs.shape
        assign = router_probs.new_zeros(N, E)  # expected assignment mass per expert
        for j in range(topk_idx.size(1)):
            assign.scatter_add_(1, topk_idx[:, j:j+1], topk_probs[:, j:j+1])
        fraction_tokens = assign.mean(dim=0)  # [E]
        prob_mass = router_probs.mean(dim=0)  # [E]
        return (prob_mass * fraction_tokens).sum() * n_experts

    def _capacity_limit(self, total_slots: int) -> int:
        # Power-of-two capacity close to average * capacity_factor
        avg = total_slots / max(1, self.n_experts)
        target = int(math.ceil(avg * self.capacity_factor))
        if target <= 0:
            return 0
        return 1 << int(math.ceil(math.log2(target)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D]
        Returns:
          y: fused output [B, T, D]
          aux_loss: scalar Tensor
          expert_indices: [B, T, K] per token selected experts (long), or empty tensor if none
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Router em fp32 (+ ruído opcional)
        logits = self.gate(x_flat.to(torch.float32))  # [N, E]
        if self.training and getattr(self.config, "router_noise_std", 0.0) > 0.0:
            logits = logits + torch.randn_like(logits) * float(self.config.router_noise_std)

        # top-k selection
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [N, K]
        topk_probs = F.softmax(topk_vals, dim=-1, dtype=torch.float32)  # [N, K]

        # Aux losses (balance + z-loss)
        router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # [N, E]
        balance_loss = self._load_balance_loss_topk(router_probs, topk_idx, topk_probs, self.n_experts)
        z = torch.logsumexp(logits, dim=-1)
        z_loss = (z * z).mean()
        aux_loss = self.aux_loss_weight * balance_loss + self.z_loss_weight * z_loss

        # Dispatch expandido por K
        N_K = N * self.top_k
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(N_K, D)  # [N*K, D]
        target_expert = topk_idx.reshape(-1)  # [N*K]
        prio = topk_vals.reshape(-1)          # [N*K]

        # Capacidade por expert (alinhada a potências de 2)
        C = self._capacity_limit(N_K)

        # Buffers de saída
        expert_out_selected = x_expanded.new_zeros((N_K, D))  # [N*K, D]
        keep_mask = torch.zeros(N_K, dtype=torch.bool, device=x.device)
        total_dropped = 0

        # Processa experts em segmentos (simples e robusto)
        for e in range(self.n_experts):
            idx_e = (target_expert == e).nonzero(as_tuple=False).squeeze(-1)
            n_e = idx_e.numel()
            if n_e == 0:
                continue
            if C <= 0:
                total_dropped += n_e
                continue
            if n_e <= C:
                kept_idx = idx_e
            else:
                prio_e = prio.index_select(0, idx_e)
                _, local_top = torch.topk(prio_e, k=C, largest=True, sorted=False)
                kept_idx = idx_e.index_select(0, local_top)
                total_dropped += int(n_e - C)

            seg = x_expanded.index_select(0, kept_idx)  # [kept, D]
            y = self.experts[e](seg)                    # [kept, D]
            expert_out_selected.index_copy_(0, kept_idx, y)
            keep_mask.index_fill_(0, kept_idx, True)

        # Reformatar para [N, K, D]
        selected = expert_out_selected.view(N, self.top_k, D)   # [N, K, D]
        kept_mask_nk = keep_mask.view(N, self.top_k)            # [N, K]

        # Penaliza drops
        if N_K > 0 and self.config.drop_penalty_weight > 0:
            drop_frac = 1.0 - keep_mask.to(torch.float32).mean()
            aux_loss = aux_loss + self.config.drop_penalty_weight * drop_frac

        if self.config.use_simple_collab:
            dtype = selected.dtype
            M = self.msg_proj(selected)    # [N, K, D]
            Q = self.q_proj(selected)      # [N, K, D]
            Kk = self.k_proj(M)            # [N, K, D]

            # Scores em fp32
            scores = torch.matmul(Q.to(torch.float32), Kk.to(torch.float32).transpose(1, 2))
            scores = scores / math.sqrt(D)

            # Máscara de pares válidos
            km = kept_mask_nk.to(torch.bool)
            if self.top_k == 2:
                s00 = scores[:, 0, 0]; s01 = scores[:, 0, 1]
                s10 = scores[:, 1, 0]; s11 = scores[:, 1, 1]
                k0 = km[:, 0]; k1 = km[:, 1]
                neg_min = torch.finfo(scores.dtype).min
                r0c0 = torch.where(k0 & k0, s00, torch.tensor(neg_min, dtype=scores.dtype, device=scores.device))
                r0c1 = torch.where(k0 & k1, s01, torch.tensor(neg_min, dtype=scores.dtype, device=scores.device))
                r1c0 = torch.where(k1 & k0, s10, torch.tensor(neg_min, dtype=scores.dtype, device=scores.device))
                r1c1 = torch.where(k1 & k1, s11, torch.tensor(neg_min, dtype=scores.dtype, device=scores.device))
                r0 = torch.stack([r0c0, r0c1], dim=-1)
                r1 = torch.stack([r1c0, r1c1], dim=-1)
                rows = torch.stack([r0, r1], dim=1)  # [N, 2, 2]
                A = F.softmax(rows, dim=-1, dtype=torch.float32)
                valid_pairs = (km.unsqueeze(2) & km.unsqueeze(1)).to(A.dtype)  # [N,2,2]
                A = A * valid_pairs
                denomA = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                A = (A / denomA).to(dtype)
            else:
                row_mask = km.unsqueeze(2)  # [N, K, 1]
                col_mask = km.unsqueeze(1)  # [N, 1, K]
                valid_pairs = row_mask & col_mask  # [N, K, K]
                scores = scores.masked_fill(~valid_pairs, torch.finfo(scores.dtype).min)
                A = F.softmax(scores, dim=-1, dtype=torch.float32)  # [N, K, K]
                A = A * valid_pairs.to(A.dtype)
                denomA = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                A = (A / denomA).to(dtype)

            # Combinação de contexto e fusão
            Cctx = torch.matmul(A, M)  # [N, K, D]
            refined = selected + self.upd(torch.cat([selected, Cctx], dim=-1))  # [N, K, D]

            refined = refined * kept_mask_nk.unsqueeze(-1)
            topk_probs_ = (topk_probs * kept_mask_nk.to(topk_probs.dtype)).to(refined.dtype)
            denom = topk_probs_.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [N, 1]
            weights = (topk_probs_ / denom)  # [N, K]
            fused = torch.sum(refined * weights.unsqueeze(-1), dim=1)  # [N, D]
            fused = self.o_proj(fused).view(B, T, D)
            expert_indices = topk_idx.view(B, T, self.top_k).to(torch.long)
            return fused, aux_loss.to(fused.dtype), expert_indices

        # Legacy collaboration: MHA sobre K com máscara
        x_collab_in = selected  # [N, K, D]
        need_w = self.save_attn_weights
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                attn_out, attn_weights = self.collab_attn(
                    x_collab_in, x_collab_in, x_collab_in,
                    need_weights=need_w,
                    key_padding_mask=~kept_mask_nk  # True mascara
                )
        except Exception:
            out = self.collab_attn(
                x_collab_in, x_collab_in, x_collab_in,
                need_weights=need_w,
                key_padding_mask=~kept_mask_nk
            )
            attn_out, attn_weights = (out if need_w else (out[0] if isinstance(out, tuple) else out, None))

        if need_w:
            self.last_attn_weights = attn_weights

        collab = self.collab_norm(attn_out + x_collab_in)
        refined = self.collab_ffn(collab) + collab  # [N, K, D]
        topk_probs_ = topk_probs.to(refined.dtype)  # [N, K]
        refined = refined * kept_mask_nk.unsqueeze(-1)
        topk_probs_ = topk_probs_ * kept_mask_nk.to(topk_probs_.dtype)
        denom = topk_probs_.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = topk_probs_ / denom
        fused = torch.sum(refined * weights.unsqueeze(-1), dim=1)  # [N, D]
        fused = self.o_proj(fused).view(B, T, D)
        expert_indices = topk_idx.view(B, T, self.top_k).to(torch.long)
        return fused, aux_loss.to(fused.dtype), expert_indices

class Block(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        # Pre-LN na atenção
        self.attn_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        self.attention = Attention(config)

        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-6)
        if config.n_experts is not None and config.n_experts > 0 and config.top_k >= 1:
            self.feed_forward = MoCTopKExperts(config)
            self.is_moe = True
            print(f"Block initialized with MoC-on-MoE: {config.n_experts} experts, top_k={config.top_k}.")
        else:
            self.feed_forward = ReasoningFeedForward(config)
            self.is_moe = False
            print("Block initialized with standard ReasoningFeedForward network.")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        def _inner_full(
            x_inner: torch.Tensor,
            freqs_cis_inner: torch.Tensor,
            past_kv_inner: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ):
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

        # Checkpoint policy
        if self.training and self.config.use_gradient_checkpointing:
            if self.config.grad_ckpt_policy == "block":
                return checkpoint(_inner_full, x, freqs_cis, past_kv, use_reentrant=False)
            elif self.config.grad_ckpt_policy == "ffn":
                # Atenção sem checkpoint, checkpoint apenas no FFN
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
        # Escala apenas as projeções de saída: Attention.o_proj, FFN.w2, MoC.o_proj
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

        # RoPE freqs como slice fp32 no device
        freqs_cis = self.freqs_cis[start_pos: start_pos + T].to(dtype=torch.float32, device=x.device)

        new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_aux_loss = x.new_zeros(())
        expert_indices_list = []

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
        # Separar router params com LR menor
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
        # Robustez a naming
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
            fused=use_fused
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

def compile_model_if_available(model: nn.Module, mode: str = "max-autotune"):
    try:
        model = torch.compile(model, mode=mode)
        print(f"Model compiled with torch.compile ({mode}).")
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
            router_noise_std=0.05,      # small noise for exploration
            drop_penalty_weight=1e-3,   # overflow penalty
            use_gradient_checkpointing=True,
            grad_ckpt_policy="ffn",
            save_attn_weights=False,
            use_simple_collab=use_simple,
            simple_collab_dropout=0.1,
            n_reasoning_steps=3,  # test IRL with >1 steps
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

        # Inference/generate
        model.eval()
        start = torch.randint(0, cfg.vocab_size, (B, 4), device=device)
        out = model.generate(start, max_new_tokens=5, temperature=1.0, top_k=20)
        print("Generate output shape:", out.shape)
        assert out.shape[1] >= start.shape[1]

    run_sanity(use_simple=False)
    run_sanity(use_simple=True)
    print("\nSanity tests completed.")

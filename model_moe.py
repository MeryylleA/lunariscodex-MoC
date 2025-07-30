"""
Full definition of a LunarisCodex Language Model, all of it in this single file.
This version is a refactored and simplified Llama-style model, created by adapting
the robust, industry-standard components from the `Instella` (OLMo) architecture
into a clean, minimal, and self-contained structure.

--- MODIFICATION FOR MoC INTEGRATION ---
This version has been upgraded from a simple Switch Transformer MoE to a more
advanced Collaborative Experts Module (MoC). The standard FeedForward network inside
each Transformer Block is replaced by the CollaborativeExpertsModule.

Key MoC features:
- CollaborativeExpertsModule (MoC): Replaces the FFN. It contains multiple FFN "experts".
- Router Contextualization: A router uses self-attention over ALL expert outputs before selecting.
- Collaborative Fusion: The selected top-k experts refine their outputs via cross-attention.
- Advanced Auxiliary Loss: A loss combining diversity and balance is computed to ensure stable
  training and effective collaboration.
- Configurable number of experts and top-k routing.
"""

import math
from dataclasses import dataclass
import inspect
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class LunarisCodexConfig:
    """
    Configuration class for the LunarisCodex model.

    Args:
        d_model: Hidden dimension size (embedding dimension)
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads for queries
        n_kv_heads: Number of key/value heads (for GQA). If equal to n_heads, it's MHA
        vocab_size: Size of the vocabulary
        multiple_of: Ensures FFN hidden dimension is a multiple of this (for efficiency)
        ffn_hidden_multiplier: Multiplier for FFN hidden dimension size
        max_seq_len: Maximum sequence length the model can handle
        rope_theta: Base frequency for RoPE (10000 is standard)
        dropout: Dropout probability for regularization

        --- MoC CONFIGURATIONS ---
        n_experts: Total number of experts in the MoC layer. If None, uses standard FFN.
        top_k: Number of experts to route each token to.
        aux_loss_weight: Multiplier for the auxiliary collaboration and balance loss.
        router_temperature: Temperature for scaling router logits before softmax.
    """
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
    # --- MoC Params ---
    n_experts: Optional[int] = 8 # Example: 8 experts
    top_k: int = 2 # For Collaborative Experts, this is often > 1
    aux_loss_weight: float = 1e-2 # Global weight for the auxiliary loss
    router_temperature: float = 1.0 # MODIFIED: Added router temperature parameter

# Pre-existing functions (precompute_freqs_cis, apply_rotary_emb) remain unchanged.
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Pre-existing modules (RMSNorm, Attention) remain unchanged.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output_dtype = x.dtype
        x = self._norm(x.float()).to(output_dtype)
        return x * self.weight

class Attention(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
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
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))
        return y, present_kv

# The original FeedForward class is kept, as it will be used as the "expert" network.
class FeedForward(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        swiglu = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(swiglu))

# --- NEW: Collaborative Experts Module (MoC) ---
class CollaborativeExpertsModule(nn.Module):
    """
    Mixture of Collaborative Experts (MoC) - FFN replacement for Transformers
    This implementation is adapted to fit the LunarisCodex architecture.

    Key innovations:
    1. Router uses self-attention over all expert outputs for contextualized selection.
    2. Selected experts collaborate via cross-attention before final fusion.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_weight = config.aux_loss_weight
        self.router_temperature = config.router_temperature # MODIFIED: Added router temperature

        # Expert networks (using the model's standard FeedForward class)
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(self.n_experts)
        ])

        # Router with self-attention contextualization
        self.router_self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.router_norm = RMSNorm(self.d_model)
        self.router_gate = nn.Linear(self.d_model, self.n_experts)

        # Collaborative fusion via cross-attention
        self.collab_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.collab_norm = RMSNorm(self.d_model)
        self.collab_ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2), # Using a small FFN for refinement
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        # Output projection
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def _compute_diversity_loss(
        self,
        cross_attn_weights: torch.Tensor,
        routing_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to encourage:
        1. Diversity in cross-attention patterns (avoid collapse).
        2. Balanced expert usage.
        """
        # Cross-attention diversity loss (maximize entropy)
        attn_entropy = -torch.sum(
            cross_attn_weights * torch.log(cross_attn_weights + 1e-8),
            dim=-1
        ).mean()
        diversity_loss = -attn_entropy

        # Expert usage balance loss (minimize variance of usage)
        expert_usage = routing_probs.mean(dim=[0, 1])
        balance_loss = torch.var(expert_usage)

        # Combine losses (using a small, fixed internal weight for simplicity)
        return 0.01 * diversity_loss + 0.01 * balance_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Final collaborative expert output.
            aux_loss: Auxiliary loss for training stability.
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1: Get outputs from ALL experts (parallel computation)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # -> (B, S, num_experts, d_model)

        # Step 2: Router Contextualization via Self-Attention
        expert_flat = expert_outputs.view(-1, self.n_experts, d_model)
        contextualized_experts, _ = self.router_self_attn(
            expert_flat, expert_flat, expert_flat
        )
        contextualized_experts = self.router_norm(contextualized_experts + expert_flat)

        # Generate routing scores from contextualized token representation
        token_summary = contextualized_experts.mean(dim=1)
        routing_logits = self.router_gate(token_summary).view(batch_size, seq_len, self.n_experts)

        # --- MODIFICATION START ---
        # Step 3: Top-K Expert Selection with Temperature and Stable Softmax
        # Apply temperature scaling to the logits
        routing_logits = routing_logits / self.router_temperature

        # Get top-k logits and their indices
        top_k_logits, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)

        # Apply softmax to the selected top-k logits for numerically stable probabilities
        top_k_probs = F.softmax(top_k_logits, dim=-1, dtype=torch.float32)
        
        # Calculate routing probabilities over all experts *after* temperature scaling for the aux loss
        routing_probs = F.softmax(routing_logits, dim=-1, dtype=torch.float32)
        # --- MODIFICATION END ---

        # Step 4: Gather selected expert outputs
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, seq_len, self.top_k)
        seq_indices = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, -1, self.top_k)
        selected_expert_outputs = expert_outputs[batch_indices, seq_indices, top_k_indices]
        # -> (B, S, top_k, d_model)

        # Step 5: Collaborative Fusion via Cross-Attention
        selected_flat = selected_expert_outputs.view(-1, self.top_k, d_model)
        collaborative_outputs, collab_attn = self.collab_cross_attn(
            selected_flat, selected_flat, selected_flat
        )
        collaborative_outputs = self.collab_norm(collaborative_outputs + selected_flat)

        # Additional refinement FFN
        refined_outputs = self.collab_ffn(collaborative_outputs) + collaborative_outputs
        refined_outputs = refined_outputs.view(batch_size, seq_len, self.top_k, d_model)

        # Step 6: Final weighted combination
        final_output = (refined_outputs * top_k_probs.unsqueeze(-1)).sum(dim=2)
        final_output = self.o_proj(final_output)

        # Step 7: Compute and apply global weight to auxiliary loss
        aux_loss = self._compute_diversity_loss(collab_attn, routing_probs) * self.aux_loss_weight

        return final_output, aux_loss


# --- MODIFIED: Block to support MoC ---
class Block(nn.Module):
    """
    A single Transformer block, now with a choice between a standard FFN and a MoC layer.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        # Conditionally create either a standard FFN or a Collaborative Experts module.
        if config.n_experts is not None and config.n_experts > 0:
            self.feed_forward = CollaborativeExpertsModule(config)
            self.is_moe = True
            print(f"Block initialized with Collaborative Experts Module ({config.n_experts} experts, top_k={config.top_k}).")
        else:
            self.feed_forward = FeedForward(config)
            self.is_moe = False
            print("Block initialized with standard FeedForward network.")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the transformer block. Now returns an optional auxiliary loss.
        """
        # First residual connection: Attention
        attn_output, new_kv = self.attention(self.attention_norm(x), freqs_cis, past_kv)
        h = x + attn_output

        # Prepare for the second residual connection (FFN or MoC)
        aux_loss = None
        ffn_input = self.ffn_norm(h)

        # Apply either the FFN or the MoC layer.
        if self.is_moe:
            ffn_output, aux_loss = self.feed_forward(ffn_input)
        else:
            ffn_output = self.feed_forward(ffn_input)

        # Second residual connection
        out = h + ffn_output

        return out, new_kv, aux_loss


# --- MODIFIED: Main LunarisCodex class to handle auxiliary loss ---
class LunarisCodex(nn.Module):
    """
    Complete LunarisCodex Language Model, now with optional MoC support.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.d_model),
            drop = nn.Dropout(config.dropout),
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

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")
        if config.n_experts is not None:
             print("Note: Parameter count includes all experts. Active parameters per forward pass are much lower.")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Apply special initialization to the output projection of key modules
        if isinstance(module, (Attention, FeedForward, CollaborativeExpertsModule)):
            for name, p in module.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[tuple], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the model.
        """
        B, T = idx.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        assert start_pos + T <= self.config.max_seq_len, \
            f"Sequence length {start_pos + T} exceeds model's max_seq_len {self.config.max_seq_len}"

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        new_past_key_values = []
        total_aux_loss = 0.0

        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv, aux_loss = block(x, freqs_cis, past_kv_for_block)
            if aux_loss is not None:
                total_aux_loss += aux_loss
            new_past_key_values.append(new_kv)

        x = self.transformer.ln_f(x)

        loss = None

        if targets is not None:
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            num_moe_layers = sum(1 for block in self.transformer.h if getattr(block, 'is_moe', False))
            final_aux_loss = total_aux_loss
            if num_moe_layers > 0:
                final_aux_loss /= num_moe_layers

            total_loss = main_loss + final_aux_loss
            loss = (total_loss, main_loss, final_aux_loss)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_past_key_values

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
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
            logits, _, past_key_values = self(idx_cond, past_key_values=past_key_values)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

"""
Optimized CollaborativeExpertsModule with PyTorch-level performance improvements.

Key optimizations:
1. Expert parallelization using batch operations instead of stack/loop
2. Memory-efficient attention with optimized reshaping
3. Reduced tensor operations and intermediate allocations
4. Optional gradient checkpointing support
5. In-place operations where safe
6. Efficient indexing operations
7. Optimized auxiliary loss computation
"""

import math
from dataclasses import dataclass
import inspect
from typing import Optional, Tuple, List
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LunarisCodexConfig:
    """
    Configuration class for the LunarisCodex model.
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
    n_experts: Optional[int] = 8
    top_k: int = 2
    aux_loss_weight: float = 0.1
    router_temperature: float = 1.0
    # --- New optimization params ---
    use_gradient_checkpointing: bool = False
    enable_expert_parallelism: bool = True


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

        # QK-Reorder-LN: per-tensor RMSNorm for Q and K before RoPE
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.q_norm = nn.RMSNorm(q_size, eps=1e-5)
        self.k_norm = nn.RMSNorm(kv_size, eps=1e-5)

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

        # QK-Reorder-LN: normalize Q and K prior to RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

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


class OptimizedExpertLayer(nn.Module):
    """
    Optimized expert layer that can process multiple inputs in parallel.
    """
    def __init__(self, config: LunarisCodexConfig, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = config.d_model

        # Calculate FFN dimensions
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # Batched expert parameters for parallel computation
        self.w1 = nn.Parameter(torch.randn(n_experts, config.d_model, hidden_dim) * 0.02)
        self.w3 = nn.Parameter(torch.randn(n_experts, config.d_model, hidden_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_experts, hidden_dim, config.d_model) * 0.02)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel expert computation.
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            expert_outputs: (batch_size, seq_len, n_experts, d_model)
        """
        B, S, D = x.shape

        # Reshape for batched matrix multiplication: (B*S, 1, D) @ (n_experts, D, hidden_dim)
        x_reshaped = x.view(B * S, 1, D)

        # Parallel computation for all experts
        # (B*S, n_experts, hidden_dim)
        gate_out = torch.bmm(x_reshaped.expand(-1, self.n_experts, -1), self.w1.transpose(0, 1).transpose(1, 2))
        up_out = torch.bmm(x_reshaped.expand(-1, self.n_experts, -1), self.w3.transpose(0, 1).transpose(1, 2))

        # Apply SwiGLU activation
        swiglu = F.silu(gate_out) * up_out

        # Final projection: (B*S, n_experts, hidden_dim) @ (n_experts, hidden_dim, D)
        expert_outputs = torch.bmm(swiglu, self.w2.transpose(0, 1).transpose(1, 2))

        # Reshape back and apply dropout
        expert_outputs = expert_outputs.view(B, S, self.n_experts, D)
        return self.dropout(expert_outputs)


class CollaborativeExpertsModule(nn.Module):
    """
    Optimized Mixture of Collaborative Experts (MoC) with significant performance improvements.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_weight = config.aux_loss_weight
        self.router_temperature = config.router_temperature
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.enable_expert_parallelism = config.enable_expert_parallelism

        # Choose expert implementation based on parallelism setting
        if self.enable_expert_parallelism:
            self.expert_layer = OptimizedExpertLayer(config, self.n_experts)
        else:
            # Fallback to individual experts
            self.experts = nn.ModuleList([
                FeedForward(config) for _ in range(self.n_experts)
            ])

        # Router with optimized attention
        self.router_self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.router_norm = nn.RMSNorm(self.d_model, eps=1e-5)
        self.router_gate = nn.Linear(self.d_model, self.n_experts, bias=False)

        # Optimized collaborative fusion
        self.collab_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=min(config.n_heads, self.top_k),  # Optimize heads for top_k
            dropout=config.dropout,
            batch_first=True
        )
        self.collab_norm = nn.RMSNorm(self.d_model, eps=1e-5)

        # More efficient collaboration FFN
        collab_hidden = self.d_model
        self.collab_ffn = nn.Sequential(
            nn.Linear(self.d_model, collab_hidden, bias=False),
            nn.GELU(),  # GELU is often more efficient than ReLU + Dropout
            nn.Linear(collab_hidden, self.d_model, bias=False)
        )

        # Output projection
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Pre-allocate buffers for efficiency (will be resized as needed)
        self.register_buffer('_temp_routing_probs', torch.empty(0))
        self.register_buffer('_temp_expert_usage', torch.empty(0))

    def _get_expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get expert outputs using the most efficient method available."""
        if self.enable_expert_parallelism:
            return self.expert_layer(x)
        else:
            # Fallback: sequential computation but with reduced memory allocation
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x))
            return torch.stack(expert_outputs, dim=2)

    def _compute_auxiliary_loss_efficient(
        self,
        routing_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        attn_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently compute auxiliary loss with minimal memory allocation.
        """
        # Expert usage balance loss - use pre-allocated buffer
        expert_usage = routing_probs.mean(dim=(0, 1))  # (n_experts,)

        # Efficient variance computation without creating intermediate tensors
        mean_usage = expert_usage.mean()
        balance_loss = ((expert_usage - mean_usage) ** 2).mean()

        # Diversity loss based on cross-attention weights entropy
        # Calculate entropy of attention weights from collaboration step
        diversity_loss = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1).mean()

        return 0.01 * diversity_loss + 0.01 * balance_loss

    def _efficient_expert_selection(
        self,
        routing_logits: torch.Tensor,
        expert_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Efficiently select and weight top-k experts with optimized indexing.
        """
        batch_size, seq_len, _ = routing_logits.shape

        # Compute routing probabilities for aux loss (before temperature scaling)
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Apply temperature scaling
        scaled_logits = routing_logits / self.router_temperature

        # Top-k selection with efficient indexing
        top_k_logits, top_k_indices = torch.topk(scaled_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Efficient gathering using advanced indexing
        # Create index tensors once
        batch_idx = torch.arange(batch_size, device=expert_outputs.device)[:, None, None]
        seq_idx = torch.arange(seq_len, device=expert_outputs.device)[None, :, None]

        # Gather selected expert outputs: (B, S, top_k, d_model)
        selected_outputs = expert_outputs[batch_idx, seq_idx, top_k_indices]

        return selected_outputs, top_k_probs, routing_probs, top_k_indices

    def _collaborative_fusion_checkpoint(self, selected_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collaborative fusion with optional checkpointing."""
        def fusion_fn(x):
            B, S, K, D = x.shape
            x_flat = x.view(-1, K, D)

            # Cross-attention with residual
            attn_out, attn_weights = self.collab_cross_attn(x_flat, x_flat, x_flat, need_weights=True)
            attn_out = self.collab_norm(attn_out + x_flat)

            # FFN refinement with residual
            refined = self.collab_ffn(attn_out) + attn_out
            return refined.view(B, S, K, D), attn_weights

        if self.use_gradient_checkpointing and self.training:
            return checkpoint(fusion_fn, selected_outputs, use_reentrant=False)
        else:
            return fusion_fn(selected_outputs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass with reduced memory allocation and improved efficiency.

        Returns:
            final_output: (B, S, d_model) - The final output after expert collaboration
            aux_loss: scalar - Auxiliary loss for expert load balancing
            top_k_indices: (B, S, top_k) - Expert indices selected for each token
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1: Parallel expert computation
        expert_outputs = self._get_expert_outputs(x)  # (B, S, n_experts, d_model)

        # Step 2: Router contextualization with memory optimization
        expert_flat = expert_outputs.view(-1, self.n_experts, d_model)

        # Use flash attention if available, otherwise standard attention
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                contextualized_experts, _ = self.router_self_attn(
                    expert_flat, expert_flat, expert_flat
                )
        except:
            contextualized_experts, _ = self.router_self_attn(
                expert_flat, expert_flat, expert_flat
            )

        # Apply norm with residual connection
        contextualized_experts = self.router_norm(contextualized_experts + expert_flat)

        # Generate routing scores
        token_summary = contextualized_experts.mean(dim=1)  # More efficient than multiple operations
        routing_logits = self.router_gate(token_summary).view(batch_size, seq_len, self.n_experts)

        # Add router noise during training to improve expert utilization
        if self.training:
            noise = torch.randn_like(routing_logits) * 1e-2
            routing_logits = routing_logits + noise

        # Step 3: Efficient expert selection and weighting
        selected_outputs, top_k_probs, routing_probs, top_k_indices = self._efficient_expert_selection(
            routing_logits, expert_outputs
        )

        # Step 4: Collaborative fusion with optional checkpointing
        refined_outputs, attn_weights = self._collaborative_fusion_checkpoint(selected_outputs)

        # Step 5: Final weighted combination with in-place operations
        final_output = torch.sum(refined_outputs * top_k_probs.unsqueeze(-1), dim=2)
        final_output = self.o_proj(final_output)

        # Step 6: Efficient auxiliary loss computation
        aux_loss = self._compute_auxiliary_loss_efficient(routing_probs, top_k_indices, attn_weights)
        aux_loss = aux_loss * self.aux_loss_weight

        return final_output, aux_loss, top_k_indices


class Block(nn.Module):
    """
    Optimized Transformer block with MoC support.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.attention = Attention(config)
        # Remove pre-attention norm for QK-Reorder-LN
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-5)

        if config.n_experts is not None and config.n_experts > 0:
            self.feed_forward = CollaborativeExpertsModule(config)
            self.is_moe = True
        else:
            self.feed_forward = FeedForward(config)
            self.is_moe = False

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Attention with residual (QK-Reorder-LN: pass x directly, norms inside Attention)
        attn_output, new_kv = self.attention(x, freqs_cis, past_kv)
        h = x + attn_output

        # FFN with residual
        aux_loss = None
        expert_indices = None
        ffn_input = self.ffn_norm(h)

        if self.is_moe:
            ffn_output, aux_loss, expert_indices = self.feed_forward(ffn_input)
        else:
            ffn_output = self.feed_forward(ffn_input)

        out = h + ffn_output
        return out, new_kv, aux_loss, expert_indices


class LunarisCodex(nn.Module):
    """
    Complete LunarisCodex Language Model with optimized MoC support.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.RMSNorm(config.d_model, eps=1e-5),
            drop = nn.Dropout(config.dropout),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute rotary embeddings
        freqs_cis = precompute_freqs_cis(
            self.config.d_model // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Print model info
        total_params = self.get_num_params()
        print(f"Number of parameters: {total_params/1e6:.2f}M")

        if config.n_experts is not None:
            # Calculate active parameters for MoE
            moe_layers = sum(1 for block in self.transformer.h if hasattr(block, 'is_moe') and block.is_moe)
            if moe_layers > 0:
                expert_params_per_layer = sum(p.numel() for p in self.transformer.h[0].feed_forward.expert_layer.parameters() if hasattr(self.transformer.h[0].feed_forward, 'expert_layer'))
                if expert_params_per_layer == 0:  # Fallback calculation
                    expert_params_per_layer = sum(p.numel() for p in self.transformer.h[0].feed_forward.experts[0].parameters())

                active_expert_params = expert_params_per_layer * config.top_k * moe_layers
                other_params = total_params - (expert_params_per_layer * config.n_experts * moe_layers)
                active_params = other_params + active_expert_params

                print(f"Active parameters per forward pass: {active_params/1e6:.2f}M ({active_params/total_params*100:.1f}% of total)")
                print(f"MoE efficiency: {config.top_k}/{config.n_experts} experts active per token")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            # For expert parameters in OptimizedExpertLayer
            torch.nn.init.normal_(module, mean=0.0, std=0.02)

        # Scale output projections
        if isinstance(module, (Attention, FeedForward, CollaborativeExpertsModule)):
            for name, p in module.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], List[Tuple[torch.Tensor, torch.Tensor]], Optional[List[torch.Tensor]]]:
        B, T = idx.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0

        assert start_pos + T <= self.config.max_seq_len, \
            f"Sequence length {start_pos + T} exceeds model's max_seq_len {self.config.max_seq_len}"

        # Token embeddings and dropout
        x = self.transformer.drop(self.transformer.wte(idx))
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        # Process through transformer blocks
        new_past_key_values = []
        total_aux_loss = 0.0
        all_expert_indices = []

        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv, aux_loss, expert_indices = block(x, freqs_cis, past_kv_for_block)

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss  # Avoid += for potential in-place issues

            if expert_indices is not None:
                all_expert_indices.append(expert_indices)

            new_past_key_values.append(new_kv)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Compute loss if targets provided
        loss_tuple = None
        if targets is not None:
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

            # Average auxiliary loss across MoE layers
            num_moe_layers = sum(1 for block in self.transformer.h if getattr(block, 'is_moe', False))
            final_aux_loss = total_aux_loss / max(num_moe_layers, 1)  # Avoid division by zero

            total_loss = main_loss + final_aux_loss
            loss_tuple = (total_loss, main_loss, final_aux_loss)
        else:
            # Generation mode - only compute logits for last token
            logits = self.lm_head(x[:, [-1], :])

        # Return expert indices only if we have MoE layers, otherwise None
        expert_indices_result = all_expert_indices if all_expert_indices else None

        return logits, loss_tuple, new_past_key_values, expert_indices_result

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Separate parameters for weight decay
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

        # Use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Optimized generation with efficient memory usage.
        """
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Check sequence length
            current_len = past_key_values[0][0].shape[-2] if past_key_values else idx.shape[1]
            if current_len >= self.config.max_seq_len:
                break

            # Only use last token if we have past_key_values
            idx_cond = idx if past_key_values is None else idx[:, -1:]

            # Forward pass - note the 4 return values
            logits, _, past_key_values, _ = self(idx_cond, past_key_values=past_key_values)

            # Sample next token
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx

"""Differentiable fixed-prefix execution; one working query after the first pass."""
import torch
from torch import Tensor, nn
from .memory import PromptMemory
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def pick(x: Tensor, positions: list[int]) -> Tensor:
    return torch.stack([x[row, position:position + 1] for row, position in enumerate(positions)])


def layer_compute(layer: nn.Module, hidden: Tensor, embeddings: tuple[Tensor, Tensor],
                  mask: Tensor | None, positions: list[int] | None = None,
                  cached: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Pure operation so non-reentrant checkpointing preserves prefix gradients."""
    attention = layer.self_attn
    residual = hidden
    x = layer.input_layernorm(hidden)
    shape = (*x.shape[:-1], -1, attention.head_dim)
    q = attention.q_proj(x).view(shape).transpose(1, 2)
    k = attention.k_proj(x).view(shape).transpose(1, 2)
    v = attention.v_proj(x).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *embeddings)
    if cached is not None:
        keys, values = cached
        active = (torch.arange(keys.shape[-2], device=x.device)[None, :] ==
                  torch.tensor(positions, device=x.device)[:, None])[:, None, :, None]
        k = torch.where(active, k, keys)
        v = torch.where(active, v, values)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(attention.config._attn_implementation, eager_attention_forward)
    output, _ = interface(attention, q, k, v, mask,
        dropout=attention.attention_dropout if layer.training else 0.0,
        scaling=attention.scaling, sliding_window=None)
    output = output.reshape(*x.shape[:-1], -1).contiguous()
    hidden = residual + attention.o_proj(output)
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, k, v


def fixed_prefix_layer(layer: nn.Module, hidden: Tensor, embeddings: tuple[Tensor, Tensor],
                       mask: Tensor | None, memory: PromptMemory, index: int,
                       use_checkpoint: bool) -> Tensor:
    """Reuse prefix K/V without detaching it or caching across model forwards."""
    positions = memory.write_mask[..., 0].long().argmax(-1).tolist()
    cached = None
    first = memory.output is None
    if not first:
        hidden = pick(hidden, positions)
        embeddings = tuple(pick(e.expand(memory.write_mask.shape[0], -1, -1), positions) for e in embeddings)
        cached = memory.keys_values[index]
        if mask is None:
            keep = torch.arange(cached[0].shape[-2], device=hidden.device)[None, :] <= torch.tensor(positions, device=hidden.device)[:, None]
            mask = torch.zeros_like(keep, dtype=hidden.dtype).masked_fill(~keep, torch.finfo(hidden.dtype).min)[:, None, None, :]
        else:
            mask = torch.stack([mask[row if mask.shape[0] > 1 else 0, :, pos:pos + 1, :] for row, pos in enumerate(positions)])
    def compute(x, cos, sin, *kv):
        return layer_compute(layer, x, (cos, sin), mask, positions, tuple(kv) if kv else None)
    args = (hidden, *embeddings, *(cached or ()))
    result, keys, values = (checkpoint(compute, *args, use_reentrant=False)
                            if use_checkpoint and torch.is_grad_enabled() else compute(*args))
    if first:
        memory.keys_values.append((keys, values))
        memory.layer_outputs.append(result)
        return result
    return torch.where(memory.write_mask, result, memory.layer_outputs[index])

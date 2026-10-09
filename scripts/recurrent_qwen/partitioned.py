"""Inference-only causal factorization for count-dependent historical models.

Identical prompt prefixes are computed once per graph, at *each* loop for
full-sequence recurrence. Count-specific suffixes remain separate continuous
states. Fixed-prompt recurrence retains first-pass layer K/V, replacing only
the working position. No symbols, labels, depth values or stops enter R.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .prefix import layer_compute


def suffix_layer(layer: nn.Module, hidden: Tensor, embeddings: tuple[Tensor, Tensor],
                 mask: Tensor, prefix: tuple[Tensor, Tensor],
                 replace_positions: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Queries [B,Q,H], rotary K/V prefix [B,KV,P,D]; full additive query mask."""
    attention = layer.self_attn
    x = layer.input_layernorm(hidden)
    shape = (*x.shape[:-1], -1, attention.head_dim)
    q = attention.q_proj(x).view(shape).transpose(1, 2)
    k = attention.k_proj(x).view(shape).transpose(1, 2)
    v = attention.v_proj(x).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *embeddings)
    if replace_positions is None:
        k, v = torch.cat((prefix[0], k), -2), torch.cat((prefix[1], v), -2)
    else:
        # Work on fresh tensors: original fixed memory is immutable across loops.
        k_full, v_full = prefix[0].clone(), prefix[1].clone()
        rows = torch.arange(len(hidden), device=hidden.device)
        k_full[rows, :, replace_positions, :] = k[:, :, 0, :]
        v_full[rows, :, replace_positions, :] = v[:, :, 0, :]
        k, v = k_full, v_full
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(attention.config._attn_implementation, eager_attention_forward)
    output, _ = interface(attention, q, k, v, mask,
                         dropout=0.0, scaling=attention.scaling, sliding_window=None)
    output = output.reshape(*x.shape[:-1], -1).contiguous()
    result = hidden + attention.o_proj(output)
    result = result + layer.mlp(layer.post_attention_layernorm(result))
    return result, k, v


@dataclass
class PartitionedResult:
    """Readouts [graph*count, loop, symbol], stop logits or None, final R working."""
    scores: Tensor
    stops: Tensor | None
    working: Tensor


@torch.inference_mode()
def partitioned_forward(model: nn.Module, prefix_ids: Tensor, suffix_ids: Tensor,
                        suffix_mask: Tensor, graph_indices: Tensor,
                        token_ids: list[int], loops: int, horizons: Tensor | None = None) -> PartitionedResult:
    """Exact causal factorization, modulo documented FP32 kernel roundoff.

    Prefix [G,P] has no padding, suffix [B,S] is right padded. Each suffix maps
    to one prefix via [B] graph_indices. Positions match the dense concatenation.
    Supports historical full_sequence/fixed_prompt models, without routed inputs,
    bridges, direct heads or private recurrent controllers. Never modifies model.
    """
    if (model.training or model.router is not None or model.bridge is not None or
            model.state_head is not None or loops < 1):
        raise ValueError('Partitioned path requires an unmodified historical eval model')
    if prefix_ids.ndim != 2 or suffix_ids.shape != suffix_mask.shape or suffix_ids.ndim != 2:
        raise ValueError('Expected prefix and aligned suffix token matrices')
    if (not suffix_mask.any(-1).all() or not torch.equal(suffix_mask.long().cumprod(-1), suffix_mask.long())
            or graph_indices.shape != (len(suffix_ids),) or
            (graph_indices < 0).any() or (graph_indices >= len(prefix_ids)).any()):
        raise ValueError('Need right-padded suffixes and valid graph indices')
    prefix = model.embed_tokens(prefix_ids)
    suffix = model.embed_tokens(suffix_ids)
    device, dtype = prefix.device, prefix.dtype
    p, s = prefix.shape[1], suffix.shape[1]
    pp = torch.arange(p, device=device)[None]
    sp = torch.arange(p, p + s, device=device)[None]
    pe, se = model.rotary_emb(prefix, pp), model.rotary_emb(suffix, sp)
    negative = torch.finfo(dtype).min
    pm = torch.zeros(p, p, device=device, dtype=dtype).masked_fill(
        torch.arange(p, device=device)[None] > torch.arange(p, device=device)[:, None], negative)[None, None]
    keep = torch.cat((torch.ones(len(suffix), p, device=device, dtype=torch.bool), suffix_mask.bool()), -1)
    allowed = (torch.arange(p + s, device=device)[None] <= sp[0, :, None])[None] & keep[:, None, :]
    sm = torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, negative)[:, None]
    ends = suffix_mask.sum(-1).long() - 1
    rows = torch.arange(len(suffix), device=device)
    am = sm[rows, :, ends, :].unsqueeze(2)
    ae = tuple(e.expand(len(suffix), -1, -1)[rows, ends, :].unsqueeze(1) for e in se)

    def block(layers, ph, sh, record=False):
        memory = []
        for layer in layers:
            ph, k, v = layer_compute(layer, ph, pe, pm)
            sh, keys, values = suffix_layer(layer, sh, se, sm,
                                           (k.index_select(0, graph_indices), v.index_select(0, graph_indices)))
            if record:
                memory.append((keys, values))
        return ph, sh, memory

    def working_block(layers, state, memory):
        for layer, kv in zip(layers, memory, strict=True):
            state, _, _ = suffix_layer(layer, state, ae, am, kv, ends + p)
        return state

    prefix, suffix, _ = block(model.prelude.layers, prefix, suffix)
    fixed = model.recurrence_mode == 'fixed_prompt'
    scores, stops = [], []
    original_batch = len(suffix_ids)
    active_rows = torch.arange(original_batch, device=device)
    crossed = torch.zeros(original_batch, device=device, dtype=torch.bool)
    if horizons is not None and (horizons.shape != (original_batch,) or (horizons < 1).any() or (horizons > loops).any()):
        raise ValueError('Requested readout horizons must fit the loop cap')
    r_memory = c_memory = None
    working = None
    selected = model.lm_head.weight[token_ids]
    for loop in range(loops):
        if fixed and loop:
            working = working_block(model.recurrent.layers, working, r_memory)
            decoded = working_block(model.coda.layers, working, c_memory)
        else:
            prefix, suffix, r_memory = block(model.recurrent.layers, prefix, suffix, fixed)
            working = suffix[rows, ends, :].unsqueeze(1)
            _, decoded_suffix, c_memory = block(model.coda.layers, prefix, suffix, fixed)
            decoded = decoded_suffix[rows, ends, :].unsqueeze(1)
        score = nn.functional.linear(model.norm(decoded[:, 0]), selected).float()
        if not torch.isfinite(score).all():
            raise FloatingPointError('Nonfinite partitioned readout')
        full_score = score.new_zeros(original_batch, score.shape[-1])
        full_score[active_rows] = score
        scores.append(full_score)
        if model.completion_head is not None:
            stop = model.completion_head(working[:, 0]).float()
            if not torch.isfinite(stop).all():
                raise FloatingPointError('Nonfinite partitioned completion head')
            full_stop = stop.new_full((original_batch,), float('-inf'))
            full_stop[active_rows] = stop
            stops.append(full_stop)
            crossed[active_rows] |= stop >= 0
        if horizons is not None and loop + 1 < loops:
            need = horizons[active_rows] > loop + 1
            if model.completion_head is not None:
                need |= ~crossed[active_rows]
            if not need.any():
                for _ in range(loop + 1, loops):
                    scores.append(full_score.new_zeros(full_score.shape))
                    if stops: stops.append(full_stop.new_full(full_stop.shape, float('-inf')))
                break
            active_rows = active_rows[need]
            graph_indices = graph_indices[need]
            suffix = suffix[need]
            working = working[need]
            ends = ends[need]
            rows = torch.arange(len(active_rows), device=device)
            sm = sm[need]; am = am[need]
            ae = tuple(e[need] for e in ae)
            if fixed:
                r_memory = [(key[need], value[need]) for key, value in r_memory]
                c_memory = [(key[need], value[need]) for key, value in c_memory]
    return PartitionedResult(torch.stack(scores, 1), torch.stack(stops, 1) if stops else None, working[:, 0])

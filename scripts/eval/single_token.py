"""Exact next-token likelihood scoring for single-token multiple-choice tasks."""

from typing import Any

import torch
from torch import Tensor


def encode_pair(tokenizer: Any, context: str, continuation: str) -> tuple[tuple[int, ...], int]:
    """Use the harness's causal token boundary convention; reject other targets.

    Trailing context whitespace belongs to the continuation. No special tokens
    are inserted. Checking the prefix prevents a retokenized boundary from being
    silently scored as a different context.
    """
    if not context or not continuation:
        raise ValueError("Single-token scoring requires nonempty context and continuation")
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    prefix = tokenizer.encode(context, add_special_tokens=False)
    complete = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not prefix or complete[:len(prefix)] != prefix or len(complete) != len(prefix) + 1:
        raise ValueError("Target must be exactly one token after an unchanged context prefix")
    return tuple(prefix), complete[-1]


def last_logits(model: Any, input_ids: Tensor, attention_mask: Tensor, *, recurrent: bool) -> Tensor:
    """Return full-vocabulary [B,V] logits, with exactly one pass and no KV cache.

    Inputs are left-padded [B,S], so every last position is a valid prompt token.
    Both paths project only that position, matching LM-head matrix shapes.
    """
    if not bool(attention_mask[:, -1].all()):
        raise ValueError("Every last position must be an unmasked prompt token")
    if recurrent:
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       num_loops=1, logits_mode="answer").logits
    else:
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False, logits_to_keep=1).logits[:, -1]
    if logits.ndim != 2 or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Expected finite full-vocabulary [B,V] logits")
    return logits


def padded_inputs(contexts: list[tuple[int, ...]], *, pad_token_id: int,
                  device: torch.device) -> tuple[Tensor, Tensor]:
    """Left-pad contexts with an explicit mask; never truncate a prompt."""
    if not contexts or any(not row for row in contexts):
        raise ValueError("Need nonempty contexts")
    width = max(map(len, contexts))
    ids = torch.full((len(contexts), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(contexts):
        ids[i, -len(row):] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, -len(row):] = 1
    return ids, mask

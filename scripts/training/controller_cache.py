"""Broader frozen prompt features paired with count-invariant executor prefixes."""
from collections.abc import Callable

import torch
from torch import Tensor

from scripts.dataset.pointer import SYMBOLS
from scripts.training.data import collate, encode_tasks
from scripts.training.objective import forward_symbols


def fit_suffix_initializer(head, context: Tensor, counts: list[int]) -> dict:
    """Fit a linear prompt reader using training labels only; no recurrent labels.

    Exact suffix embeddings are independent of graph/start for this prompt
    template. Fit their first graph's count variants and verify all repeats.
    The learned weights, not targets or a parser, initialize native memory.
    """
    if head.kind != 'affine_suffix' or len(context) % len(counts) or context.shape[1] != head.context.in_features:
        raise ValueError('Need an affine suffix head and aligned count variants')
    rows = context.reshape(-1, len(counts), context.shape[1])
    torch.testing.assert_close(rows, rows[:1].expand_as(rows), atol=0, rtol=0)
    x = rows[0].double()
    a = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    labels = torch.tensor(counts, dtype=x.dtype)
    fit = torch.linalg.lstsq(a, labels, rcond=1e-8, driver='gelsd')
    with torch.no_grad():
        head.context.weight.copy_(fit.solution[:-1][None].to(head.context.weight))
        head.context.bias.copy_(fit.solution[-1:].to(head.context.bias))
    predicted = head.context(rows[0].to(head.context.weight)).detach().cpu().flatten()
    return {'method': 'training-only linear least squares, rcond=1e-8', 'training_counts': counts,
            'feature_rank': int(fit.rank), 'training_max_absolute_error': (predicted - labels.float()).abs().max().item(),
            'graph_invariant_suffix_verified': True, 'updates': 'initializer frozen after supervised fit'}


def prompt_features(model, tokenizer, token_ids: list[int], tasks: list, batch_size: int,
                    progress: Callable[[int], None] | None = None) -> Tensor:
    """Cache [B,H_context] native initialization inputs; no R/C or count parser."""
    if not tasks or batch_size < 1 or model.router is None:
        raise ValueError('Need tasks, positive batch size and an isolated executor')
    model.eval().requires_grad_(False)
    device = str(next(model.parameters()).device)
    items = encode_tasks(tasks, tokenizer, dict(zip(SYMBOLS, token_ids, strict=True)), 512)
    chunks = []
    for start in range(0, len(items), batch_size):
        batch = collate(items[start:start + batch_size], tokenizer.pad_token_id, device)
        chunks.append(model.controller_prompt_state(batch['input_ids'], batch['attention_mask']).cpu())
        if progress:
            progress(min(start + batch_size, len(items)))
    result = torch.cat(chunks)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite controller prompt features')
    return result


def pair_prefixes(old_tasks: list, old_features: dict[str, Tensor], tasks: list, loops: int) -> dict[str, Tensor]:
    """Reuse actual R states by (rule table, start), verifying old count invariance."""
    if loops < 1 or loops > old_features['working'].shape[1] or len(old_tasks) != len(old_features['working']):
        raise ValueError('Need aligned cached tasks and an available loop prefix')
    indices = {}
    for i, task in enumerate(old_tasks):
        key = (task.mapping_sha256, task.initial_state)
        if key in indices:
            prior = indices[key]
            torch.testing.assert_close(old_features['working'][i, :loops], old_features['working'][prior, :loops],
                                       atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(old_features['predictions'][i, :loops],
                                       old_features['predictions'][prior, :loops], atol=0, rtol=0)
        else:
            indices[key] = i
    selected = []
    for task in tasks:
        key = (task.mapping_sha256, task.initial_state)
        if key not in indices:
            raise ValueError('New prompt has no cached executor trajectory for its graph/start')
        selected.append(indices[key])
    idx = torch.tensor(selected)
    return {key: old_features[key][idx, :loops].clone() for key in ('working', 'predictions')}


def check_native_prefix(model, tokenizer, token_ids: list[int], tasks: list,
                        context: Tensor, working: Tensor, batch_size: int = 4) -> float:
    """Guard new count/padding variants against actual unchanged executor forwards."""
    items = encode_tasks(tasks, tokenizer, dict(zip(SYMBOLS, token_ids, strict=True)), 512)
    observed, contexts = [], []
    hooks = [model.completion_head.context.register_forward_pre_hook(lambda m, x: contexts.append(x[0].detach())),
             model.completion_head.observation.register_forward_pre_hook(lambda m, x: observed.append(x[0].detach()))]
    max_error = 0.
    try:
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                observed.clear(); contexts.clear()
                batch = collate(items[start:start + batch_size], tokenizer.pad_token_id,
                                str(next(model.parameters()).device))
                forward_symbols(model, batch['input_ids'], batch['attention_mask'], token_ids,
                                num_loops=working.shape[1])
                actual = torch.stack(observed, 1).cpu()
                expected = working[start:start + batch_size]
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                torch.testing.assert_close(contexts[0].cpu(), context[start:start + batch_size], atol=2e-5, rtol=1e-4)
                max_error = max(max_error, (actual - expected).abs().max().item())
    finally:
        for hook in hooks:
            hook.remove()
    return max_error

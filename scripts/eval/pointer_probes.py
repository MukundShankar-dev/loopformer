"""Small paired prompt interventions and hidden-state diagnostics; no training."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import math
from typing import Any, Iterator

import torch

from scripts.dataset.pointer import PointerExample, SYMBOLS, render_prompt, validate_example
from scripts.training.data import EncodedExample, collate
from scripts.training.objective import symbolic_scores


def task_slice(task: PointerExample, start_step: int, end_step: int) -> PointerExample:
    """Same table, reference start at start_step, transitions through end_step.

    Steps are original trajectory indices; 0 is the original initial state.
    Altered start/depth changes the prompt, so this is not an isolated hidden-state intervention.
    """
    if not 0 <= start_step < end_step <= task.task_depth:
        raise ValueError('Require 0 <= start_step < end_step <= task depth')
    start = task.initial_state if start_step == 0 else task.intermediate_states[start_step - 1]
    states = task.intermediate_states[start_step:end_step]
    result = replace(task, example_id=f'{task.example_id}:slice-{start_step}-{end_step}',
                     initial_state=start, intermediate_states=states, final_state=states[-1],
                     task_depth=len(states), prompt=render_prompt(task.mapping, start, len(states)))
    validate_example(result)
    return result


def state_change(previous: torch.Tensor, current: torch.Tensor) -> dict:
    """Scalar state diagnostics; callers distinguish answer position vs whole sequence."""
    a, b = previous.float().reshape(-1), current.float().reshape(-1)
    norm = torch.linalg.vector_norm(a)
    delta = torch.linalg.vector_norm(b - a)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12)
    return {'rms': b.square().mean().sqrt().item(),
            'update_rms': delta.item() / math.sqrt(b.numel()),
            'relative_update_norm': (delta / norm).item() if norm.item() else None,
            'cosine_to_previous': cosine.item() if norm.item() and torch.linalg.vector_norm(b).item() else None}


def inspect_example(model: torch.nn.Module, item: EncodedExample, token_ids: list[int], pad_id: int, *, rule_control: str | None = None,
                    prefix_tokens: int = 0, restart_after: int = 6) -> list[dict]:
    """Capture one unpadded example at a time; only scalar summaries leave this call."""
    if rule_control not in (None, "refresh", "noop"):
        raise ValueError("Unknown rule control")
    device = str(next(model.parameters()).device)
    batch = collate([item], pad_id, device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            control = (rule_context_control(model, prefix_tokens, restart_after, refresh=rule_control == 'refresh')
                       if rule_control else nullcontext())
            with control:
                result = model(batch['input_ids'], batch['attention_mask'], num_loops=len(item.targets), return_hidden_states=True)
            scores = symbolic_scores(result.loop_logits, token_ids)[0].float()
            values = scores.cpu().tolist()
            previous = result.initial_hidden_state
            rows = []
            for t, (hidden, logits) in enumerate(zip(result.hidden_states, values, strict=True), 1):
                rank = sorted(range(len(logits)), key=lambda j: (-logits[j], j))
                target = item.targets[t - 1]
                weights = [math.exp(v - logits[rank[0]]) for v in logits]
                total = sum(weights)
                entropy = -sum((w / total) * math.log(w / total) for w in weights if w)
                row = {'loop': t, 'prediction': SYMBOLS[rank[0]], 'target': SYMBOLS[target],
                       'correct': rank[0] == target, 'target_rank': rank.index(target) + 1,
                       'top_symbols': ' '.join(SYMBOLS[j] for j in rank[:3]),
                       'answer_entropy': entropy,
                       'predicted_margin': logits[rank[0]] - logits[rank[1]],
                       'target_margin': logits[target] - max(v for j, v in enumerate(logits) if j != target)}
                for scope, a, b in [('answer', previous[:, -1], hidden[:, -1]), ('sequence', previous, hidden)]:
                    row.update({scope + '_' + k: v for k, v in state_change(a, b).items()})
                rows.append(row)
                previous = hidden
            return rows
    finally:
        model.train(was_training)


def rule_prefix_length(tokenizer: Any, prompt: str) -> int:
    """Token boundary before Start, including the rule-line newline; reject crossings."""
    boundary = prompt.index('\nStart:') + 1
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    if any(start < boundary < end for start, end in offsets):
        raise ValueError('A token crosses the Rules/Start boundary; cannot isolate rule positions')
    count = sum(end <= boundary for start, end in offsets)
    if not 0 < count < len(offsets) or any(start < boundary for start, end in offsets[count:]):
        raise ValueError('Invalid contiguous rule-token prefix')
    return count


@contextmanager
def rule_context_control(model: torch.nn.Module, prefix_tokens: int, restart_after: int, *, refresh: bool) -> Iterator[None]:
    """Inference-only hook: replace [1,:prefix,H] at R input after K loops.

    Refresh uses h0 from P; the no-op copies the current prefix. Remaining
    positions retain the current state. Hooks and captured tensors are scoped
    to one forward, including exceptional exits; model weights never change.
    """
    if model.training or torch.is_grad_enabled():
        raise ValueError('Rule-context controls require eval mode and no_grad')
    if prefix_tokens < 1 or restart_after < 1:
        raise ValueError('Positive prefix length and restart boundary required')
    initial = None
    calls = 0

    def capture(module, args, output):
        nonlocal initial
        if output.shape[0] != 1 or prefix_tokens >= output.shape[1]:
            raise ValueError('Rule control requires batch 1 and a proper prefix')
        initial = output[:, :prefix_tokens].detach().clone()

    def intervene(module, args):
        nonlocal calls
        calls += 1
        if initial is None:
            raise ValueError('Prelude must run before recurrence')
        if calls <= restart_after:
            return None
        hidden, *rest = args
        prefix = initial if refresh else hidden[:, :prefix_tokens]
        return (torch.cat((prefix, hidden[:, prefix_tokens:]), dim=1), *rest)

    handles = []
    try:
        handles.append(model.prelude.register_forward_hook(capture))
        handles.append(model.recurrent.register_forward_pre_hook(intervene))
        yield
    finally:
        for handle in handles:
            handle.remove()

"""Small paired prompt interventions and hidden-state diagnostics; no training."""

from dataclasses import replace
import math

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


def inspect_example(model: torch.nn.Module, item: EncodedExample, token_ids: list[int], pad_id: int) -> list[dict]:
    """Capture one unpadded example at a time; only scalar summaries leave this call."""
    device = str(next(model.parameters()).device)
    batch = collate([item], pad_id, device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
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

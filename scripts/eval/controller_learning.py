"""Bounded diagnostic fits on frozen features; no updates to the pointer executor."""
from copy import deepcopy
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.eval.controller_features import replay
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.objective import completion_loss


def fit_probe(features: Tensor, labels: Tensor, partitions: list[str], classes: int,
              *, nonlinear: bool, steps: int, seed: int, device: str,
              shuffled: bool = False) -> tuple[Tensor, dict]:
    """Train-only feature standardization; choose on graph-disjoint dev CE.

    Return all predictions, including a held-out graph partition that never
    influences fitting or selection. A failed probe is not proof of absent info.
    """
    if steps < 1 or len(features) != len(labels) or len(partitions) != len(labels):
        raise ValueError('Invalid probe shapes or step budget')
    torch.manual_seed(seed)
    x, y = features.to(device), labels.to(device)
    masks = {p: torch.tensor([s == p for s in partitions], device=device) for p in ('fit', 'dev', 'holdout')}
    if not all(m.any() for m in masks.values()) or y.min() < 0 or y.max() >= classes:
        raise ValueError('Need all graph partitions and valid class labels')
    mean, scale = x[masks['fit']].mean(0), x[masks['fit']].std(0, unbiased=False).clamp_min(1e-4)
    x = (x - mean) / scale
    train_x, train_y = x[masks['fit']], y[masks['fit']].clone()
    if shuffled:
        train_y = train_y[torch.randperm(len(train_y), device=device)]
    width = x.shape[-1]
    probe = (nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, classes))
             if nonlinear else nn.Linear(width, classes)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.003, weight_decay=0.01)
    best_loss, best_step, best = float('inf'), 0, None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_x), train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite diagnostic probe loss')
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                dev_loss = F.cross_entropy(probe(x[masks['dev']]), y[masks['dev']]).item()
            if dev_loss < best_loss:
                best_loss, best_step, best = dev_loss, step, deepcopy(probe.state_dict())
    probe.load_state_dict(best)
    with torch.no_grad():
        prediction = probe(x).argmax(-1)
    return prediction.cpu(), {'selected_step': best_step, 'dev_cross_entropy': best_loss,
        'chance_accuracy': 1 / classes, 'shuffled_fit_labels': shuffled,
        'accuracy': {p: (prediction[m] == y[m]).float().mean().item() for p, m in masks.items()}}


def stop_rows(logits: Tensor, predictions: Tensor, tasks: list, metadata: list[dict],
              condition: str) -> list[dict]:
    """Exact first-crossing timing; distinguish missing stop from cap-time success."""
    logits, predictions = logits.detach().cpu(), predictions.detach().cpu()
    if logits.ndim != 2 or predictions.shape != logits.shape or len(tasks) != len(logits) or len(metadata) != len(tasks):
        raise ValueError('Stopping rows need aligned [questions, loops] logits and predictions')
    rows = []
    for scores, predicted, task, meta in zip(logits, predictions, tasks, metadata, strict=True):
        hits = (scores >= 0).nonzero().flatten()
        first = int(hits[0]) + 1 if len(hits) else None
        executed = first if first is not None else len(scores)
        prediction = chr(int(predicted[executed - 1]) + ord('A'))
        correct = prediction == task.final_state
        rows.append({'condition': condition, **meta, 'first_stop': first, 'executed_loops': executed,
            'exact_stop': first == task.task_depth, 'early_stop': first is not None and first < task.task_depth,
            'late_stop': first is not None and first > task.task_depth, 'missing_stop': first is None,
            'prediction': prediction, 'target': task.final_state,
            'stop_probability': float(scores[executed - 1].sigmoid()),
            'executor_nominal_correct': [chr(int(p) + ord('A')) for p in predicted[:task.task_depth]] == task.intermediate_states,
            'stopped_answer_correct': correct, 'joint_success': correct and first == task.task_depth})
    return rows


def aggregate_stops(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('Cannot summarize empty stop cohort')
    return {'questions': len(rows), **{key: sum(row[key] for row in rows) / len(rows)
        for key in ('exact_stop', 'early_stop', 'late_stop', 'missing_stop', 'stopped_answer_correct', 'joint_success')}}


def fit_controller(head: RecurrentController, features: dict[str, Tensor], depths: Tensor,
                   fit_indices: Tensor, *, steps: int, learning_rate: float, device: str,
                   progress: Callable[[int, dict], None] | None = None) -> tuple[RecurrentController, list[dict]]:
    """Full-batch tiny-set fit with the existing stop loss and no post-depth labels.

    Selection uses fit loss only. No held-out graph or count affects selection.
    Cached executor observations are detached; controller recurrence uses full BPTT.
    """
    if steps < 1 or learning_rate <= 0 or not len(fit_indices):
        raise ValueError('Need nonempty fit examples, positive learning rate and steps')
    fitted = deepcopy(head).to(device).train().requires_grad_(True)
    context = features['context'][fit_indices].to(device)
    working = features['working'][fit_indices].to(device)
    d = depths[fit_indices].to(device)
    mask = torch.arange(working.shape[1], device=device)[None, :] < d[:, None]
    optimizer = torch.optim.AdamW(fitted.parameters(), lr=learning_rate, weight_decay=0.0)
    history, best_loss, best = [], float('inf'), None
    previous_gradient_norm = None
    for step in range(steps + 1):
        logits, _, _ = replay(fitted, context, working)
        loss, _ = completion_loss(logits, mask)
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                has = (logits >= 0).any(-1)
                first = (logits >= 0).int().argmax(-1) + 1
                row = {'step': step, 'loss': loss.item(), 'exact_stop': (has & (first == d)).float().mean().item(),
                       'previous_gradient_norm': previous_gradient_norm, 'learning_rate': learning_rate}
            history.append(row)
            if row['loss'] < best_loss:
                best_loss, best = row['loss'], deepcopy(fitted.state_dict())
            if progress:
                progress(step, row)
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(fitted.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        previous_gradient_norm = float(norm)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite controller gradient')
    fitted.load_state_dict(best)
    return fitted.eval().requires_grad_(False), history

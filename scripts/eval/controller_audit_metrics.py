"""Read-only controller fit and gradient diagnostics on immutable cached inputs."""
from collections import defaultdict
from itertools import combinations
import random

import torch
from torch import Tensor, nn

from scripts.eval.controller_features import replay
from scripts.training.controller import evaluate_controller
from scripts.training.controller_remaining import remaining_predictions
from scripts.training.objective import completion_loss


def loss_components(logits: Tensor, predicted: Tensor, depths: Tensor, scale: float) -> dict[str, Tensor]:
    """Split the actual objective without changing its per-example/time weighting.

    initial + intermediate + terminal equals remaining_loss exactly. There are
    no interior targets at N=1. t=0 is initialization; stop logits start at t=1.
    """
    t = torch.arange(predicted.shape[1], device=predicted.device)[None, :]
    errors = ((predicted - (depths[:, None] - t)) / scale).square() / (depths[:, None] + 1)
    return {
        'stop': completion_loss(logits, t[:, 1:] <= depths[:, None])[0],
        'initial': errors[:, 0].mean(),
        'intermediate': (errors * ((t > 0) & (t < depths[:, None]))).sum(-1).mean(),
        'terminal': errors.gather(1, depths[:, None]).mean(),
    }


def cohort(depth: int, trained_depths: list[int]) -> str:
    return 'trained' if depth in trained_depths else 'interpolation' if depth <= max(trained_depths) else 'extrapolation'


def gradient_indices(tasks: list, graphs: int, seed: int) -> list[int]:
    """Select graphs, keeping every requested-count variant of each selected graph."""
    identities = sorted({t.mapping_sha256 for t in tasks})
    if graphs < 1 or graphs > len(identities):
        raise ValueError(f'Need 1..{len(identities)} gradient graphs, got {graphs}')
    random.Random(seed).shuffle(identities)
    chosen = set(identities[:graphs])
    return [i for i, task in enumerate(tasks) if task.mapping_sha256 in chosen]


def fit_audit(head: nn.Module, readout: nn.Linear, features: dict, tasks: list,
              trained_depths: list[int], device: str, scale: float, batch_size: int) -> tuple[dict, list[dict], list[dict]]:
    """Reuse shared exact-stop scoring; add paired numerical/stopping error records."""
    traces = []
    metrics, decisions = evaluate_controller(head, features, tasks, trained_depths, device,
        batch_size=batch_size, remaining_readout=readout, remaining_scale=scale, traces=traces)
    grouped = defaultdict(list)
    for trace in traces:
        grouped[trace['example_id']].append(trace)
    rows = []
    for task, decision in zip(tasks, decisions, strict=True):
        trace = grouped[task.example_id]
        nominal = trace[:task.task_depth + 1]
        errors = [abs(r['predicted_remaining'] - r['target_remaining']) for r in nominal]
        first_zero = next((r['loop'] for r in trace[1:] if r['predicted_remaining'] <= .5), None)
        first_error = next((i for i, e in enumerate(errors) if e > .5), None)
        rows.append({**decision, 'mapping_sha256': task.mapping_sha256,
            'cohort': cohort(task.task_depth, trained_depths),
            'initial_prediction': nominal[0]['predicted_remaining'], 'initial_abs_error': errors[0],
            'initial_within_half': errors[0] <= .5, 'remaining_trajectory_within_half': max(errors) <= .5,
            'first_remaining_error_loop': first_error,
            'remaining_mae': sum(errors) / len(errors), 'terminal_abs_error': errors[-1],
            'decrement_mae': sum(abs(b['predicted_remaining'] - a['predicted_remaining'] + 1)
                                 for a, b in zip(nominal, nominal[1:])) / task.task_depth,
            'first_zero': first_zero, 'zero_exact': first_zero == task.task_depth,
            'wrong_time_correct_letter': decision['stopped_answer_correct'] and not decision['exact_stop']})
    cross = []
    for name in metrics:
        selected = [r for r in rows if (r['depth'] == int(name[6:]) if name.startswith('depth_') else r['cohort'] == name)]
        failures = [r['first_remaining_error_loop'] for r in selected if r['first_remaining_error_loop'] is not None]
        metrics[name].update(
            remaining_error_examples=len(failures),
            first_remaining_error_loop_mean=sum(failures) / len(failures) if failures else None,
            wrong_time_correct_letter_rate=sum(r['wrong_time_correct_letter'] for r in selected) / len(selected))
        for signal in ('initial_within_half', 'remaining_trajectory_within_half', 'zero_exact'):
            correct = [r for r in selected if r[signal]]
            metrics[name][f'{signal}_examples'] = len(correct)
            metrics[name][f'exact_stop_given_{signal}'] = sum(r['exact_stop'] for r in correct) / len(correct) if correct else None
            for numerical_correct in (False, True):
                for stop_correct in (False, True):
                    count = sum(r[signal] == numerical_correct and r['exact_stop'] == stop_correct for r in selected)
                    cross.append({'cohort': name, 'signal': signal, 'numerical_correct': numerical_correct,
                                  'stop_correct': stop_correct, 'count': count, 'cohort_questions': len(selected)})
    return metrics, rows, cross


def gradient_audit(head: nn.Module, readout: nn.Linear, context: Tensor, working: Tensor,
                   depths: Tensor, scale: float, weight: float, batch_size: int,
                   progress=None) -> tuple[list[dict], list[dict], list[dict]]:
    """Measure objective gradients without backward(), optimizer steps or .grad writes.

    Per-batch vectors and their example-weighted panel mean are measured separately.
    Auxiliary gradients in stop-only controls are explicitly counterfactual: the
    deployed training objective has zero auxiliary weight on controller parameters.
    Adam's historical moments are not modeled; these are raw objective gradients.
    """
    head.eval().requires_grad_(True)
    readout.eval().requires_grad_(True)
    named = [(f'controller.{n}', p) for n, p in head.named_parameters()] + [(f'auxiliary.{n}', p) for n, p in readout.named_parameters()]
    params = [p for _, p in named]
    lengths = [p.numel() for p in params]
    slices, offset = defaultdict(list), 0
    for (name, _), length in zip(named, lengths, strict=True):
        module = '.'.join(name.split('.')[:2]) if name.startswith('controller.') else 'auxiliary'
        slices[module].append(slice(offset, offset + length))
        if name.startswith('controller.'):
            slices['controller'].append(slice(offset, offset + length))
        offset += length
    totals = {k: torch.zeros(offset, device=context.device) for k in ('stop', 'initial', 'intermediate', 'terminal')}
    total_losses = dict.fromkeys(totals, 0.)
    norms, cosines, batches = [], [], []

    def record(vectors: dict[str, Tensor], losses: dict[str, float], scope: str, batch: int, count: int) -> None:
        vectors = {**vectors, 'remaining': vectors['initial'] + vectors['intermediate'] + vectors['terminal']}
        # Head gradients in the control are stop-only. The separate passive
        # readout optimizer still receives its unweighted regression gradient.
        controller_indices = torch.cat([torch.arange(s.start, s.stop, device=context.device) for s in slices['controller']])
        vectors['objective'] = vectors['stop'] + weight * vectors['remaining']
        norm = vectors['objective'][controller_indices].norm().item()
        clip = min(1., 1. / max(norm, 1e-30))
        batches.append({'scope': scope, 'batch': batch, 'questions': count, **losses,
                        'remaining': sum(losses[k] for k in ('initial', 'intermediate', 'terminal')),
                        'objective': losses['stop'] + weight * sum(losses[k] for k in ('initial', 'intermediate', 'terminal')),
                        'controller_norm': norm, 'hypothetical_clip_scale': clip,
                        'remaining_weight': weight})
        for module, pieces in slices.items():
            parts = {k: torch.cat([v[s] for s in pieces]) for k, v in vectors.items()}
            for component, vector in parts.items():
                # The objective vector describes only the controller optimizer;
                # passive-readout training is reported as raw remaining gradient.
                if module == 'auxiliary' and component == 'objective':
                    continue
                norms.append({'scope': scope, 'batch': batch, 'module': module, 'component': component,
                              'norm': vector.norm().item(), 'questions': count,
                              'auxiliary_active_on_controller': weight > 0})
            for a, b in combinations(('stop', 'initial', 'intermediate', 'terminal', 'remaining'), 2):
                denominator = parts[a].norm() * parts[b].norm()
                value = (parts[a] @ parts[b] / denominator).clamp(-1, 1).item() if denominator.item() > 0 else None
                cosines.append({'scope': scope, 'batch': batch, 'module': module, 'left': a, 'right': b,
                                'cosine': value, 'questions': count})

    for start in range(0, len(depths), batch_size):
        end = min(start + batch_size, len(depths))
        logits, initial, memories = replay(head, context[start:end], working[start:end])
        predicted = remaining_predictions(readout, initial, memories, scale)
        losses = loss_components(logits, predicted, depths[start:end], scale)
        vectors = {}
        for name, loss in losses.items():
            grads = torch.autograd.grad(loss, params, retain_graph=name != 'terminal', allow_unused=True)
            vector = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                                for g, p in zip(grads, params, strict=True)])
            if not torch.isfinite(vector).all() or not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite audit gradient/loss')
            vectors[name] = vector
            totals[name] += vector * ((end - start) / len(depths))
            total_losses[name] += loss.item() * ((end - start) / len(depths))
        record(vectors, {k: v.item() for k, v in losses.items()}, 'batch', start // batch_size, end - start)
        if progress:
            progress(end)
    record(totals, total_losses, 'panel_mean', -1, len(depths))
    return norms, cosines, batches

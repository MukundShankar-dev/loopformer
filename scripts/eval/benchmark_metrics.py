"""Graph-cluster quality metrics, separate from model inference and stop policy."""
from math import sqrt

import numpy as np
import torch
from torch import Tensor

from scripts.dataset.pointer import PointerExample, SYMBOLS, execute


METRICS = ('nominal_final_correct', 'complete_trajectory', 'exact_stop', 'early_stop',
           'late_stop', 'missing_stop', 'stopped_answer_correct', 'joint_success',
           'strict_success', 'correct_letter_wrong_time')


def quality_arrays(tasks: list[PointerExample], predictions: Tensor, stops: list[int | None],
                   max_depth: int, direct_predictions: Tensor | None = None) -> tuple[dict, list[dict]]:
    """Boolean [graphs,counts] outcomes; a missing signal is never an exact stop."""
    if predictions.ndim != 2 or len(tasks) != len(predictions) or len(stops) != max_depth:
        raise ValueError('Unaligned graph trajectories/count stops')
    cap = predictions.shape[1]
    if cap < max_depth or any(t is not None and not 1 <= t <= cap for t in stops):
        raise ValueError('Prediction budget does not cover requests/stops')
    targets = torch.tensor([[SYMBOLS.index(s) for s in execute(dict(t.mapping), t.initial_state, cap)] for t in tasks])
    correct = predictions.cpu() == targets
    strict = correct[:, :max_depth].cumprod(1).bool()
    counts = torch.arange(1, max_depth + 1)
    first = torch.tensor([t if t is not None else cap for t in stops])
    signaled = torch.tensor([t is not None for t in stops])
    exact = signaled & (first == counts)
    stopped_correct = predictions.cpu()[:, first - 1] == targets[:, :max_depth]
    expand = lambda value: value[None].expand(len(tasks), -1).numpy()
    arrays = {'nominal_final_correct': correct[:, :max_depth].numpy(), 'complete_trajectory': strict.numpy(),
        'exact_stop': expand(exact), 'early_stop': expand(signaled & (first < counts)),
        'late_stop': expand(signaled & (first > counts)), 'missing_stop': expand(~signaled),
        'stopped_answer_correct': stopped_correct.numpy(), 'joint_success': (stopped_correct & exact).numpy(),
        'strict_success': (strict & exact).numpy(),
        'correct_letter_wrong_time': (stopped_correct & ~exact).numpy()}
    loop_rows = []
    survived = torch.ones(len(tasks), dtype=torch.bool)
    for i in range(max_depth):
        eligible = int(survived.sum())
        row = {'loop': i + 1, 'graphs': len(tasks), 'accuracy': correct[:, i].float().mean().item(),
            'correct_prefix_graphs': eligible,
            'conditional_transition_accuracy': correct[survived, i].float().mean().item() if eligible else None,
            'complete_prefix_accuracy': strict[:, i].float().mean().item()}
        if direct_predictions is not None:
            row.update(r_accuracy=(direct_predictions[:, i].cpu() == targets[:, i]).float().mean().item(),
                       r_c_agreement=(direct_predictions[:, i].cpu() == predictions[:, i].cpu()).float().mean().item())
        loop_rows.append(row)
        survived &= correct[:, i]
    return arrays, loop_rows


def clustered_summary(arrays: dict, metadata: list[dict], mask: np.ndarray, *, repeats: int, seed: int) -> dict:
    """Resample whole graphs within seed/mode strata; all horizons travel together."""
    if not mask.any() or repeats < 1:
        raise ValueError('Need a nonempty count cohort and bootstrap repeats')
    groups = {}
    for i, row in enumerate(metadata):
        groups.setdefault((row['dataset_seed'], row['graph_mode']), []).append(i)
    values = np.stack([arrays[k][:, mask].mean(1) for k in METRICS], 1)
    rng = np.random.default_rng(seed)
    samples = np.zeros((repeats, len(METRICS)))
    for indices in groups.values():
        indices = np.asarray(indices)
        chosen = rng.choice(indices, size=(repeats, len(indices)), replace=True)
        samples += values[chosen].sum(1)
    samples /= len(metadata)
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {'graphs': len(metadata), 'queries': int(len(metadata) * mask.sum()),
        'counts': (np.flatnonzero(mask) + 1).tolist(),
        'metrics': {k: {'rate': float(values[:, i].mean()), 'graph_bootstrap_95': bounds[:, i].tolist()}
                    for i, k in enumerate(METRICS)}}


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """Marginal graph-level binomial interval; nondegenerate at zero/all successes."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError('Invalid graph success count')
    z = 1.959963984540054
    rate = successes / n
    denominator = 1 + z*z/n
    center = (rate + z*z/(2*n)) / denominator
    half = z*sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denominator
    return max(0., center-half), min(1., center+half)


def grouped_rows(arrays: dict, metadata: list[dict]) -> list[dict]:
    """Per-count rates and explicit seed/mode/orbit strata, without horizon pseudo-counts."""
    rows = []
    groups = {'all': list(range(len(metadata)))}
    for key in ('dataset_seed', 'graph_mode', 'transient_length', 'cycle_period'):
        for value in sorted({m[key] for m in metadata}):
            groups[f'{key}={value}'] = [i for i, m in enumerate(metadata) if m[key] == value]
    for name, indices in groups.items():
        for j in range(next(iter(arrays.values())).shape[1]):
            row = {'stratum': name, 'depth': j + 1, 'graphs': len(indices),
                   **{k: float(v[indices, j].mean()) for k, v in arrays.items()}}
            for key in ('joint_success', 'complete_trajectory', 'exact_stop'):
                lo, hi = wilson_interval(int(arrays[key][indices, j].sum()), len(indices))
                row.update({key + '_wilson_95_low': lo, key + '_wilson_95_high': hi})
            rows.append(row)
    return rows


def criteria_result(arrays: dict, criteria: dict) -> dict:
    keys = {'joint_success': 'joint_success', 'complete_trajectory': 'complete_trajectory', 'exact_stop': 'exact_stop'}
    checks = {}
    for name, key in keys.items():
        checks[f'overall_{name}'] = float(arrays[key].mean()) >= criteria[f'overall_{name}']
        checks[f'minimum_depth_{name}'] = float(arrays[key].mean(0).min()) >= criteria[f'minimum_depth_{name}']
    return {'passed': all(checks.values()), 'checks': checks, 'thresholds': criteria}

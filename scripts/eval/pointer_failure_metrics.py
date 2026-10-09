"""Offline decoded-trajectory diagnostics; raw-table reference, no model inputs.

Decoded symbols are observations of a readout, not proof of hidden-state content.
Repeated loops on a graph are dependent; first-error tables count each graph once.
"""
from collections import Counter

import numpy as np


CATEGORIES = ('repeat previous state', 'earlier visited state', 'later reachable state', 'outside start orbit')
MODES = ('random_function', 'permutation', 'full_cycle')


def reference_trace(mapping: dict[str, str], start: str, loops: int) -> str:
    result = []
    state = start
    for _ in range(loops):
        state = mapping[state]
        result.append(state)
    return ''.join(result)


def first_error_category(mapping: dict[str, str], start: str, targets: str,
                         predicted: str, index: int) -> str:
    """Priority makes cyclic/visited categories mutually exclusive; index is zero-based."""
    if predicted == targets[index]:
        raise ValueError('Categorize only an incorrect decoded symbol')
    previous = start if index == 0 else targets[index - 1]
    if predicted == previous:
        return CATEGORIES[0]
    if predicted in start + targets[:index]:
        return CATEGORIES[1]
    orbit, state = set(), start
    while state not in orbit:
        orbit.add(state)
        state = mapping[state]
    return CATEGORIES[2] if predicted in orbit else CATEGORIES[3]


def failure_metrics(tasks: list, rows: list[dict], maximum: int) -> dict:
    """One aligned trajectory per graph; strict prefixes can never regain success."""
    if len(tasks) != len(rows) or maximum < 1:
        raise ValueError('Need aligned graph tasks and readouts')
    correct, targets, first, episodes = [], [], [], []
    confusion = np.zeros((26, 26), dtype=np.int64)
    category_counts = Counter()
    # Categories after a previously wrong C decode, excluding the first loop.
    dynamics = Counter()
    first_readouts = Counter()
    exposure = Counter()
    for task, row in zip(tasks, rows, strict=True):
        if row['mapping_sha256'] != task.mapping_sha256 or row['example_id'] != task.example_id:
            raise ValueError('Graph/readout identity mismatch')
        mapping = dict(task.mapping)
        target = reference_trace(mapping, task.initial_state, maximum)
        prediction = row['predictions'][:maximum]
        direct = row['r_predictions'][:maximum]
        if len(prediction) != maximum or len(direct) != maximum:
            raise ValueError('Missing loop readouts')
        hits = np.asarray([p == t for p, t in zip(prediction, target, strict=True)])
        errors = np.flatnonzero(~hits)
        saved = int(row['first_error_loop']) if row['first_error_loop'] else None
        actual = int(errors[0]) + 1 if len(errors) else None
        if saved != actual:
            raise ValueError('Saved first-error position disagrees with independent reference')
        mode = row['graph_mode']; exposure[mode] += 1
        if len(errors):
            i = int(errors[0]); category = first_error_category(mapping, task.initial_state, target, prediction[i], i)
            confusion[ord(target[i]) - 65, ord(prediction[i]) - 65] += 1
            category_counts[mode, category] += 1
            readout = ('R correct / C wrong' if direct[i] == target[i] else
                       'R and C wrong, same letter' if direct[i] == prediction[i] else 'R and C wrong, different letters')
            first_readouts[readout] += 1
            first.append({**{k: row[k] for k in ('graph_index', 'example_id', 'graph_mode', 'dataset_seed', 'cycle_period', 'transient_length')},
                'loop': i + 1, 'target': target[i], 'prediction': prediction[i], 'r_prediction': direct[i],
                'category': category, 'readout_category': readout,
                'one_reference_step_ahead': prediction[i] == mapping[target[i]],
                'correct_final_at_max': bool(hits[-1]), 'wrong_loops': int((~hits).sum())})
        for i in range(1, maximum):
            if not hits[i - 1]:
                kind = ('recover to reference' if hits[i] else
                        'follow edge from prior decode' if prediction[i] == mapping[prediction[i - 1]] else
                        'other wrong transition')
                dynamics[mode, kind] += 1
            if not hits[i] and hits[i - 1]:
                episodes.append({'graph_index': row['graph_index'], 'graph_mode': mode, 'loop': i + 1})
        correct.append(hits); targets.append(target)
    array = np.stack(correct)
    categories = [{'graph_mode': mode, 'category': category, 'graphs': category_counts[mode, category],
        'total_graphs': exposure[mode], 'failed_graphs': sum(category_counts[mode, c] for c in CATEGORIES)}
        for mode in MODES for category in CATEGORIES]
    dynamics_rows = [{'graph_mode': mode, 'category': category, 'transitions': dynamics[mode, category],
        'prior_wrong_transitions': sum(dynamics[mode, c] for c in
            ('recover to reference', 'follow edge from prior decode', 'other wrong transition'))}
        for mode in MODES for category in ('recover to reference', 'follow edge from prior decode', 'other wrong transition')]
    return {'correct': array, 'targets': targets, 'first_errors': first, 'categories': categories,
        'dynamics': dynamics_rows, 'error_episodes': episodes,
        'confusion': confusion, 'first_readouts': dict(first_readouts),
        'summary': {'graphs': len(tasks), 'maximum_loop': maximum, 'first_failed_graphs': len(first),
            'last_first_error_loop': max((r['loop'] for r in first), default=None),
            'error_episode_entries_after_first': len(episodes) - sum(r['loop'] > 1 for r in first),
            'failed_graphs_correct_final': sum(r['correct_final_at_max'] for r in first),
            'first_errors_one_reference_step_ahead': sum(r['one_reference_step_ahead'] for r in first),
            'first_error_readouts': dict(first_readouts),
            'after_wrong_readout_transitions': sum(dynamics.values()),
            'limitations': 'Decoded-edge consistency is observational, not a causal hidden-state test; repeated graph loops are dependent.'}}

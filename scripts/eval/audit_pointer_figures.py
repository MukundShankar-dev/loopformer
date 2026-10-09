"""Recompute every benchmark plot input from raw graphs and saved observations.

This is an offline scoring audit, not another inference run. Endpoint correctness,
strict prefix correctness and actual stop timing are deliberately kept separate.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from math import isclose, sqrt
from pathlib import Path
import re

import numpy as np

from scripts.eval.audit_checkpoint_comparison import audit_comparison
from scripts.eval.pointer_task import sha256_file


def rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def equal(actual: str | float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance):
        raise ValueError(f'{label}: saved {actual}, independently computed {expected}')


def interval(successes: int, total: int) -> tuple[float, float]:
    """Independent Wilson calculation for the exported pointwise graph bands."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0, center - half), min(1, center + half)


def audit_plot_tables(results: Path, graphs_file: Path) -> dict:
    """Check raw targets, every decision, every stratum and every per-loop series."""
    summary = json.loads((results / 'summary.json').read_text())
    manifest = json.loads((results / 'dataset_manifest.json').read_text())
    if sha256_file(graphs_file) != manifest['graphs_sha256']:
        raise ValueError('Raw graph bytes do not match the benchmark')
    tasks = [json.loads(line) for line in graphs_file.read_text().splitlines()]
    graphs = rows(results / 'graphs.csv')
    counts = rows(results / 'controller_counts.csv')
    n, maximum, cap = len(tasks), summary['config']['max_depth'], summary['config']['safety_cap']
    if len(graphs) != n or n != summary['graphs'] or summary['queries'] != n * maximum:
        raise ValueError('Graph/query coverage differs')
    if [int(r['depth']) for r in counts] != list(range(1, maximum + 1)):
        raise ValueError('Controller count coverage differs')
    targets, predictions, direct, fingerprints = [], [], [], set()
    first_errors = Counter()
    for i, (task, row) in enumerate(zip(tasks, graphs, strict=True)):
        mapping = dict(task['mapping'])
        fingerprint = hashlib.sha256(json.dumps(sorted(mapping.items()), separators=(',', ':')).encode()).hexdigest()
        if (fingerprint != task['mapping_sha256'] or fingerprint != row['mapping_sha256']
                or row['example_id'] != task['example_id'] or int(row['graph_index']) != i
                or fingerprint in fingerprints):
            raise ValueError('Graph identity mismatch')
        fingerprints.add(fingerprint)
        if 'graph_metadata' in manifest and any(row[k] != str(v) for k, v in manifest['graph_metadata'][i].items()):
            raise ValueError('Saved graph strata differ from the original dataset manifest')
        lines = task['prompt'].splitlines()
        if (len(lines) != 4 or dict(re.findall(r'\( ([A-Z]), ([A-Z])\)', lines[0])) != mapping
                or lines[1:] != [f'Start: {task["initial_state"]}', f'Steps: {maximum}', 'Answer:']):
            raise ValueError('Raw prompt does not match the graph')
        state, trace, visited = task['initial_state'], [], {}
        while state not in visited:
            visited[state] = len(visited)
            state = mapping[state]
        transient, period = visited[state], len(visited) - visited[state]
        if int(row['transient_length']) != transient or int(row['cycle_period']) != period:
            raise ValueError('Orbit metadata differs from raw graph')
        state = task['initial_state']
        for _ in range(cap):
            state = mapping[state]
            trace.append(state)
        if task['intermediate_states'] != trace[:maximum] or task['final_state'] != trace[maximum - 1]:
            raise ValueError('Dataset targets differ from raw traversal')
        if len(row['predictions']) != cap or len(row['r_predictions']) != cap:
            raise ValueError('Saved C/R readout coverage differs')
        first = next((t for t, (p, s) in enumerate(zip(row['predictions'][:maximum], trace), 1) if p != s), None)
        if row['first_error_loop'] != (str(first) if first else ''):
            raise ValueError('First C-readout error differs')
        first_errors[str(first) if first else 'none'] += 1
        targets.append(trace); predictions.append(list(row['predictions'])); direct.append(list(row['r_predictions']))
    target, prediction, r_prediction = map(np.asarray, (targets, predictions, direct))
    hits = prediction[:, :maximum] == target[:, :maximum]
    prefixes = np.logical_and.accumulate(hits, axis=1)
    stops = np.array([int(r['first_stop']) if r['first_stop'] else 0 for r in counts])
    depth = np.arange(1, maximum + 1)
    signaled = stops != 0
    if ((stops < 0) | (stops > cap)).any():
        raise ValueError('Stop outside safety cap')
    executed = np.where(signaled, stops, cap)
    for j, row in enumerate(counts):
        flags = {'exact_stop': bool(signaled[j] and stops[j] == depth[j]),
            'early_stop': bool(signaled[j] and stops[j] < depth[j]),
            'late_stop': bool(signaled[j] and stops[j] > depth[j]), 'missing_stop': not bool(signaled[j])}
        if any(k in row and row[k] != str(v) for k, v in flags.items()):
            raise ValueError('Controller count flags differ from first stop')
        if 'executed_loops' in row and int(row['executed_loops']) != executed[j]:
            raise ValueError('Controller count executed loops differ')
    stopped_hits = prediction[:, executed - 1] == target[:, :maximum]
    exact = signaled & (stops == depth)
    arrays = {'nominal_final_correct': hits, 'complete_trajectory': prefixes,
        'exact_stop': np.broadcast_to(exact, hits.shape),
        'early_stop': np.broadcast_to(signaled & (stops < depth), hits.shape),
        'late_stop': np.broadcast_to(signaled & (stops > depth), hits.shape),
        'missing_stop': np.broadcast_to(~signaled, hits.shape), 'stopped_answer_correct': stopped_hits,
        'joint_success': stopped_hits & exact, 'strict_success': prefixes & exact,
        'correct_letter_wrong_time': stopped_hits & ~exact}
    with gzip.open(results / 'decisions.csv.gz', 'rt') as handle:
        saved = csv.DictReader(handle)
        for i in range(n):
            for j in range(maximum):
                row = next(saved, None)
                if row is None or int(row['graph_index']) != i or int(row['depth']) != j + 1:
                    raise ValueError('Decision coverage/order differs')
                for key, array in arrays.items():
                    if row[key] != str(bool(array[i, j])):
                        raise ValueError(f'Decision {key} differs at graph {i}, count {j + 1}')
                if int(row['executed_loops']) != executed[j] or row['first_stop'] != counts[j]['first_stop']:
                    raise ValueError('Decision stop disagrees with controller record')
        if next(saved, None) is not None:
            raise ValueError('Extra decisions')
    groups = {'all': list(range(n))}
    for key in ('dataset_seed', 'graph_mode', 'transient_length', 'cycle_period'):
        for value in {r[key] for r in graphs}:
            groups[f'{key}={value}'] = [i for i, r in enumerate(graphs) if r[key] == value]
    grouped = rows(results / 'per_count_and_stratum.csv')
    seen = set()
    for row in grouped:
        key, j = row['stratum'], int(row['depth']) - 1
        if key not in groups or not 0 <= j < maximum or (key, j) in seen:
            raise ValueError('Wrong or duplicate stratum/count')
        seen.add((key, j)); indices = groups[key]
        if int(row['graphs']) != len(indices):
            raise ValueError('Per-stratum graph denominator differs')
        for metric, array in arrays.items():
            successes = int(array[indices, j].sum())
            equal(row[metric], successes / len(indices), f'{key}/{j + 1}/{metric}')
            if metric in ('joint_success', 'complete_trajectory', 'exact_stop'):
                for side, bound in zip(('low', 'high'), interval(successes, len(indices)), strict=True):
                    equal(row[f'{metric}_wilson_95_{side}'], bound, f'{key}/{j + 1}/{metric}/{side}')
    if len(seen) != len(groups) * maximum:
        raise ValueError('Missing stratum/count')
    loop_rows = rows(results / 'per_loop.csv')
    if [int(r['loop']) for r in loop_rows] != list(range(1, maximum + 1)):
        raise ValueError('Per-loop coverage differs')
    for j, row in enumerate(loop_rows):
        eligible = np.ones(n, dtype=bool) if j == 0 else prefixes[:, j - 1]
        if int(row['graphs']) != n or int(row['correct_prefix_graphs']) != eligible.sum():
            raise ValueError('Per-loop population or conditional denominator differs')
        expected = {'accuracy': hits[:, j].mean(), 'complete_prefix_accuracy': prefixes[:, j].mean(),
            'r_accuracy': (r_prediction[:, j] == target[:, j]).mean(),
            'r_c_agreement': (r_prediction[:, j] == prediction[:, j]).mean()}
        if eligible.any():
            expected['conditional_transition_accuracy'] = hits[eligible, j].mean()
        elif row['conditional_transition_accuracy']:
            raise ValueError('Empty conditional cohort must be undefined')
        for metric, value in expected.items():
            # Original loop exports use a float32 tensor mean, unlike strata.
            equal(row[metric], value, f'loop {j + 1}/{metric}', tolerance=6e-8)
    for metric, array in arrays.items():
        equal(summary['cohorts']['all']['metrics'][metric]['rate'], array.mean(), f'overall/{metric}')
    checkpoints = sorted({t for t in (1, 12, 30, 31, 64, 128, maximum) if t <= maximum})
    return {'passed': True, 'graphs': n, 'decisions_checked': n * maximum,
        'stratum_count_rows_checked': len(grouped), 'loop_rows_checked': len(loop_rows),
        'first_c_error_counts': dict(first_errors),
        'depth_checks': [{'requested_steps': t, 'denominator_graphs': n,
            'all_steps_correct_graphs': int(prefixes[:, t - 1].sum()),
            'final_answer_and_exact_stop_graphs': int(arrays['joint_success'][:, t - 1].sum()),
            'exact_stop_graphs': int(arrays['exact_stop'][:, t - 1].sum())} for t in checkpoints],
        'metric_definitions': {'joint_success': 'correct C letter at actual stop AND first stop equals N',
            'complete_trajectory': 'every C letter at loops 1..N matches the reference; no recovery',
            'first_error': 'first incorrect intermediate C letter; not a first joint-success failure'},
        'inputs_sha256': {name: sha256_file(results / name) for name in
            ('graphs.csv', 'controller_counts.csv', 'per_count_and_stratum.csv', 'per_loop.csv', 'decisions.csv.gz', 'summary.json')},
        'raw_graphs_sha256': sha256_file(graphs_file)}


def audit_failure_exports(results: Path, graphs_file: Path, output: Path) -> dict:
    """Independently traverse tables to check every failure-figure CSV cell."""
    tasks = [json.loads(line) for line in graphs_file.read_text().splitlines()]
    graphs = rows(results / 'graphs.csv')
    maximum = json.loads((results / 'summary.json').read_text())['config']['max_depth']
    categories, dynamics, confusion, final_counts, period_errors, period_sizes = (Counter() for _ in range(6))
    expected_first, episodes, mode_sizes, mode_failures = {}, set(), Counter(), Counter()
    for task, graph in zip(tasks, graphs, strict=True):
        mapping, state, reference = dict(task['mapping']), task['initial_state'], []
        for _ in range(maximum):
            state = mapping[state]; reference.append(state)
        predicted, direct = graph['predictions'][:maximum], graph['r_predictions'][:maximum]
        correct = [p == s for p, s in zip(predicted, reference, strict=True)]
        mode, index, period = graph['graph_mode'], graph['graph_index'], int(graph['cycle_period'])
        mode_sizes[mode] += 1; period_sizes[period] += 1
        final_counts[reference[-1], 'graphs'] += 1
        final_counts[reference[-1], 'wrong'] += not correct[-1]
        for j, hit in enumerate(correct):
            period_errors[period, j + 1] += not hit
            if j and not correct[j - 1]:
                kind = ('recover to reference' if hit else 'follow edge from prior decode'
                        if predicted[j] == mapping[predicted[j - 1]] else 'other wrong transition')
                dynamics[mode, kind] += 1
            if j and correct[j - 1] and not hit:
                episodes.add((index, mode, j + 1))
        if all(correct):
            continue
        mode_failures[mode] += 1
        j = correct.index(False); p = predicted[j]
        previous = task['initial_state'] if j == 0 else reference[j - 1]
        orbit, state = set(), task['initial_state']
        while state not in orbit:
            orbit.add(state); state = mapping[state]
        category = ('repeat previous state' if p == previous else 'earlier visited state'
            if p in [task['initial_state'], *reference[:j]] else 'later reachable state'
            if p in orbit else 'outside start orbit')
        categories[mode, category] += 1
        confusion[reference[j], p] += 1
        expected_first[index] = {'loop': str(j + 1), 'target': reference[j], 'prediction': p,
            'r_prediction': direct[j], 'category': category,
            'readout_category': 'R correct / C wrong' if direct[j] == reference[j] else
                'R and C wrong, same letter' if direct[j] == p else 'R and C wrong, different letters',
            'one_reference_step_ahead': str(p == mapping[reference[j]]),
            'correct_final_at_max': str(correct[-1]), 'wrong_loops': str(correct.count(False))}
    saved_first = rows(output / 'first_errors.csv')
    if len(saved_first) != len(expected_first) or len({r['graph_index'] for r in saved_first}) != len(saved_first):
        raise ValueError('First-error export coverage differs')
    for row in saved_first:
        expected = expected_first.get(row['graph_index'])
        if expected is None or any(row[k] != v for k, v in expected.items()):
            raise ValueError('First-error export differs from raw traversal')
    category_rows = rows(output / 'categories.csv')
    for row in category_rows:
        mode = row['graph_mode']
        if (int(row['graphs']) != categories[mode, row['category']] or int(row['total_graphs']) != mode_sizes[mode]
                or int(row['failed_graphs']) != mode_failures[mode]):
            raise ValueError('Category export denominator/count differs')
    dynamic_rows = rows(output / 'dynamics.csv')
    for row in dynamic_rows:
        denominator = sum(v for (mode, _), v in dynamics.items() if mode == row['graph_mode'])
        if (int(row['transitions']) != dynamics[row['graph_mode'], row['category']]
                or int(row['prior_wrong_transitions']) != denominator):
            raise ValueError('Dynamics export denominator/count differs')
    saved_episodes = rows(output / 'error_episodes.csv')
    if {(r['graph_index'], r['graph_mode'], int(r['loop'])) for r in saved_episodes} != episodes or len(saved_episodes) != len(episodes):
        raise ValueError('Error-episode export differs')
    for row in rows(output / 'first_error_confusion.csv'):
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if int(row[letter]) != confusion[row['target'], letter]:
                raise ValueError('First-error confusion cell differs')
    for row in rows(output / 'final_error_by_letter.csv'):
        letter = row['letter']; denominator = final_counts[letter, 'graphs']; errors = final_counts[letter, 'wrong']
        if int(row['graphs']) != denominator or int(row['wrong_graphs']) != errors:
            raise ValueError('Final-letter denominator/count differs')
        if denominator:
            equal(row['error_rate'], errors / denominator, f'final letter {letter}')
        elif row['error_rate']:
            raise ValueError('Empty final-letter cohort must be undefined')
    cycle_rows = rows(output / 'cycle_period_by_loop.csv')
    if len(cycle_rows) != len(period_sizes) * maximum:
        raise ValueError('Cycle matrix coverage differs')
    for row in cycle_rows:
        period, loop = int(row['period']), int(row['loop'])
        if int(row['graphs']) != period_sizes[period]:
            raise ValueError('Cycle-period denominator differs')
        equal(row['error_rate'], period_errors[period, loop] / period_sizes[period], f'period {period}/loop {loop}')
    return {'passed': True, 'failing_graphs_checked': len(expected_first),
        'confusion_cells_checked': 26 * 26, 'final_letter_rates_checked': 26,
        'cycle_matrix_cells_checked': len(cycle_rows), 'error_episode_entries_checked': len(episodes),
        'category_cells_checked': len(category_rows), 'dynamics_cells_checked': len(dynamic_rows)}


def audit_numeric(path: Path) -> dict:
    """Check individual stopping flags rather than only a cumulative total."""
    summary = json.loads((path / 'summary.json').read_text())
    data = rows(path / 'countdown.csv')
    if [int(r['requested']) for r in data] != list(range(1, summary['rollout_max'] + 1)):
        raise ValueError('Numeric request coverage differs')
    total, first_failed = 0, None
    for row in data:
        count = int(row['requested']); stop = int(row['first_stop']) if row['first_stop'] else None
        flags = {'exact_stop': stop == count, 'early_stop': stop is not None and stop < count,
            'late_stop': stop is not None and stop > count, 'missing_stop': stop is None}
        if any(row[k] != str(v) for k, v in flags.items()):
            raise ValueError('Numeric flags differ from actual first stop')
        total += flags['exact_stop']
        if not flags['exact_stop'] and first_failed is None:
            first_failed = count
    if total != summary['exact_stop_questions'] or first_failed != summary['first_failed_stop_count']:
        raise ValueError('Numeric summary differs')
    inputs = {name: sha256_file(path / name) for name in ('summary.json', 'countdown.csv')}
    if 'reader_max' in summary:
        reader = rows(path / 'reader.csv')
        if [int(r['requested']) for r in reader] != list(range(1, summary['reader_max'] + 1)):
            raise ValueError('Numeric reader coverage differs')
        errors = [abs(float(r['initial_memory']) - int(r['requested'])) for r in reader]
        for row, error in zip(reader, errors, strict=True):
            equal(row['absolute_error'], error, 'reader absolute error')
        equal(summary['reader_max_absolute_error'], max(errors), 'reader maximum error')
        equal(summary['reader_mean_absolute_error'], sum(errors) / len(errors), 'reader mean error')
        for row in data:
            equal(row['initial_memory'], reader[int(row['requested']) - 1]['initial_memory'], 'reader/countdown initialization')
        inputs['reader.csv'] = sha256_file(path / 'reader.csv')
    return {'passed': True, 'requests_checked': len(data), 'exact_stop_questions': total,
        'first_failed_stop_count': first_failed,
        'inputs_sha256': inputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--numeric-results', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'benchmark': audit_plot_tables(args.results, args.graphs),
        'comparison': audit_comparison(args.comparison, args.graphs),
        'numeric': {str(p): audit_numeric(p) for p in args.numeric_results},
        'audit_source_sha256': sha256_file(Path(__file__)),
        'scope': 'Independent rescoring of saved observations; no new model inference.'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(f'Plot-input audit passed: {result["benchmark"]["decisions_checked"]:,} benchmark '
          f'and {result["comparison"]["decisions_checked"]:,} comparison decisions')


if __name__ == '__main__':
    main()

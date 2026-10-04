"""Compare nominal recurrent execution on matched full-loop evaluation traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scripts.eval.loop_metrics import summarize_trajectories, write_csv


def compare_runs(paths: list[Path]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Return depth accuracy, conditional transitions, paired outcomes and provenance.

    Risk sets contain only examples correct at every preceding nominal loop.
    A zero denominator produces null, not zero accuracy. Post-task loops are excluded.
    """
    depth_rows, risk_rows, paired_rows, provenance = [], [], [], {}
    reference_identity = reference_settings = reference = None
    for path in paths:
        summary = json.loads((path / 'summary.json').read_text())
        if summary['status'] != 'complete' or 'stop_policy' in summary:
            raise ValueError(f'Expected completed full-loop evaluation: {path}')
        examples, _ = summarize_trajectories(path / 'trajectories.csv')
        identity = {e['example_id']: (e['task_depth'], e['initial_state'], e['targets']) for e in examples}
        settings = {k: summary[k] for k in ('data_sha256', 'loops', 'dtype', 'batch_size', 'scoring', 'prompt_format')}
        if reference is None:
            reference_identity, reference_settings = identity, settings
            reference = {e['example_id']: e for e in examples}
        elif identity != reference_identity or settings != reference_settings:
            raise ValueError('Checkpoints must use identical examples, targets and evaluation settings')
        label = path.name
        provenance[label] = {'checkpoint': summary['checkpoint'], **settings,
                             'summary_sha256': hashlib.sha256((path / 'summary.json').read_bytes()).hexdigest(),
                             'trajectories_sha256': hashlib.sha256((path / 'trajectories.csv').read_bytes()).hexdigest()}
        for depth in sorted({e['task_depth'] for e in examples}):
            members = [e for e in examples if e['task_depth'] == depth]
            errors = Counter(str(e['first_error_loop'] or 'none') for e in members)
            depth_rows.append({'checkpoint': label, 'task_depth': depth, 'examples': len(members),
                               'trajectory_accuracy': sum(e['trajectory_correct'] for e in members) / len(members),
                               'mean_correct_prefix_length': sum(e['correct_prefix_length'] for e in members) / len(members),
                               'first_error_counts': json.dumps(dict(errors), sort_keys=True)})
            for loop in range(1, depth + 1):
                at_risk = sum(e['correct_prefix_length'] >= loop - 1 for e in members)
                failures = sum(e['first_error_loop'] == loop for e in members)
                risk_rows.append({'checkpoint': label, 'task_depth': depth, 'loop': loop,
                                  'correct_prefix_examples': at_risk, 'first_failures': failures,
                                  'conditional_transition_accuracy': (at_risk - failures) / at_risk if at_risk else None})
        for e in examples:
            base = reference[e['example_id']]
            paired_rows.append({'checkpoint': label, 'reference': paths[0].name, 'example_id': e['example_id'],
                                'task_depth': e['task_depth'], 'trajectory_correct': e['trajectory_correct'],
                                'reference_correct': base['trajectory_correct'], 'first_error_loop': e['first_error_loop'],
                                'correct_prefix_delta': e['correct_prefix_length'] - base['correct_prefix_length']})
    return depth_rows, risk_rows, paired_rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    # The already-selected checkpoint is the paired reference, not an OOD selection.
    paths = [args.results / f'step-{step:06d}' for step in (5000, 4500, 5500, 7500)]
    depths, risks, pairs, provenance = compare_runs(paths)
    for name, rows in [('depth_comparison', depths), ('conditional_transitions', risks), ('paired_examples', pairs)]:
        write_csv(args.results / f'{name}.csv', rows)
    (args.results / 'comparison.json').write_text(json.dumps(provenance, indent=2) + '\n')
    table = Table('Checkpoint', *[str(d) for d in range(13, 21)])
    for path in paths:
        values = {r['task_depth']: r['trajectory_accuracy'] for r in depths if r['checkpoint'] == path.name}
        table.add_row(path.name, *[f'{values[d]:.1%}' for d in range(13, 21)])
    Console().print(table)
    Console().print(f'Saved comparison CSVs and provenance to {args.results}')


if __name__ == '__main__':
    main()

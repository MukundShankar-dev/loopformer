"""Independent raw-table audit of paired frozen checkpoint decision metrics."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from scripts.eval.pointer_task import sha256_file


def audit_comparison(results: Path, graphs_file: Path) -> dict:
    summary = json.loads((results / 'summary.json').read_text())
    freeze = json.loads((results / 'freeze.json').read_text())
    if summary['status'] != 'complete' or not summary['inference_files_unchanged']:
        raise ValueError('Need a completed frozen comparison')
    if sha256_file(graphs_file) != freeze['source_graphs_sha256']:
        raise ValueError('Raw graph bytes differ from freeze')
    graphs = [json.loads(line) for line in graphs_file.read_text().splitlines()]
    with (results / 'decisions.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    expected = {(arm['name'], i, count) for arm in summary['config']['models']
                for i in freeze['selected_graph_indices'] for count in summary['config']['counts']}
    seen, totals = set(), {a['name']: Counter() for a in summary['config']['models']}
    for row in rows:
        model, graph_index, depth = row['model'], int(row['graph_index']), int(row['depth'])
        key = model, graph_index, depth
        if key not in expected or key in seen:
            raise ValueError('Wrong or duplicated model/graph/count coverage')
        seen.add(key)
        graph = graphs[graph_index]
        if row['mapping_sha256'] != graph['mapping_sha256'] or row['example_id'] != f'{graph["example_id"]}-steps-{depth}':
            raise ValueError('Decision belongs to another graph/count')
        mapping, state = dict(graph['mapping']), graph['initial_state']
        targets = []
        for _ in range(depth):
            state = mapping[state]; targets.append(state)
        target = ''.join(targets)
        prediction = row['forced_predictions']
        if len(prediction) != depth or row['targets'] != target or row['target'] != target[-1]:
            raise ValueError('Nominal trace length or raw-rule targets differ')
        first_error = next((i + 1 for i, (a, b) in enumerate(zip(prediction, target, strict=True)) if a != b), None)
        saved_error = int(row['first_error_loop']) if row['first_error_loop'] else None
        if saved_error != first_error:
            raise ValueError('First-error index differs')
        first_stop = int(row['first_stop']) if row['first_stop'] else None
        if first_stop is not None and not 1 <= first_stop <= summary['config']['safety_cap']:
            raise ValueError('First stop outside safety cap')
        executed = first_stop or summary['config']['safety_cap']
        if int(row['executed_loops']) != executed:
            raise ValueError('Executed loops/cap differs')
        exact = first_stop == depth
        correct = row['prediction'] == target[-1]
        # If actual stopping falls in the nominal trace, its letter must agree.
        if executed <= depth and row['prediction'] != prediction[executed - 1]:
            raise ValueError('Stopped decode disagrees with forced nominal prefix')
        flags = {'nominal_final_correct': prediction[-1] == target[-1], 'complete_trajectory': first_error is None,
            'exact_stop': exact, 'early_stop': first_stop is not None and first_stop < depth,
            'late_stop': first_stop is not None and first_stop > depth, 'missing_stop': first_stop is None,
            'stopped_answer_correct': correct, 'joint_success': correct and exact,
            'strict_success': first_error is None and exact, 'correct_letter_wrong_time': correct and not exact}
        for name, value in flags.items():
            if row[name] != str(value):
                raise ValueError(f'Decision {name} differs from independent arithmetic')
            totals[model][name] += value
    if seen != expected:
        raise ValueError('Incomplete matched panel')
    for model, counts in totals.items():
        if not summary['models'][model]['native_fidelity']['passed'] or not summary['models'][model]['inference_files_unchanged']:
            raise ValueError('Model native fidelity/freeze failed')
        for metric, count in counts.items():
            if abs(summary['models'][model][metric] - count / summary['questions_per_model']) > 1e-12:
                raise ValueError('Summary aggregate differs from decision arithmetic')
    return {'passed': True, 'decisions_checked': len(rows), 'graphs': summary['graphs'],
        'models': len(summary['models']), 'success_counts': {k: dict(v) for k, v in totals.items()},
        'inputs_sha256': {name: sha256_file(results / name) for name in ('summary.json', 'decisions.csv', 'freeze.json')},
        'graphs_sha256': sha256_file(graphs_file), 'audit_source_sha256': sha256_file(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('eval/pointer_benchmark/checkpoint-comparison-20261008'))
    parser.add_argument('--graphs', type=Path, required=True)
    args = parser.parse_args()
    output = args.results / 'comparison_reference_audit.json'
    if output.exists():
        raise ValueError('Audit exists; refuse overwrite')
    result = audit_comparison(args.results, args.graphs)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Independent audit passed {result["decisions_checked"]:,} decisions')


if __name__ == '__main__':
    main()

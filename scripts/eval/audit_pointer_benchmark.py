"""Independently check saved benchmark targets, flags and aggregate arithmetic."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from scripts.eval.benchmark_metrics import METRICS
from scripts.eval.pointer_task import sha256_file


def audit_native_results(results: Path, data: Path) -> dict:
    """Traverse raw graphs and check native targets, first crossings and verdicts."""
    summary = json.loads((results / 'summary.json').read_text())
    if (summary['status'] != 'complete' or summary['stop_policy'] != 'completion'
            or summary['data_sha256'] != sha256_file(data) or summary['limit'] is not None
            or summary['test_mode']):
        raise ValueError('Need a complete, unfiltered native result with original data')
    tasks = [json.loads(line) for line in data.read_text().splitlines()]
    by_id = {t['example_id']: t for t in tasks}
    decisions = list(csv.DictReader((results / 'decisions.csv').open()))
    if (len(by_id) != len(tasks) or len(decisions) != len(tasks)
            or {r['example_id'] for r in decisions} != set(by_id)
            or summary['selected_examples'] != len(tasks)):
        raise ValueError('Native task/decision coverage differs')
    trajectories = {key: [] for key in by_id}
    with (results / 'trajectories.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['example_id'] not in trajectories:
                raise ValueError('Unknown native trajectory task')
            trajectories[row['example_id']].append(row)
    totals = Counter(); rows_checked = 0
    threshold = summary['stop_threshold']
    for decision in decisions:
        task = by_id[decision['example_id']]
        mapping, depth = dict(task['mapping']), task['task_depth']
        lines = task['prompt'].splitlines()
        if (len(lines) != 4 or dict(re.findall(r'\( ([A-Z]), ([A-Z])\)', lines[0])) != mapping
                or lines[1] != f'Start: {task["initial_state"]}' or lines[2] != f'Steps: {depth}'
                or lines[3] != 'Answer:'):
            raise ValueError('Native raw prompt differs from structured task')
        state, reference = task['initial_state'], []
        for _ in range(depth):
            state = mapping[state]; reference.append(state)
        if task['intermediate_states'] != reference or task['final_state'] != state:
            raise ValueError('Native dataset target differs from raw graph traversal')
        rows = trajectories[task['example_id']]
        executed = int(decision['executed_loops'])
        if (not 1 <= executed <= summary['loops'] or len(rows) != executed
                or int(decision['task_depth']) != depth or int(decision['budget']) != summary['loops']):
            raise ValueError('Native loop coverage or budget differs')
        for loop, row in enumerate(rows, 1):
            target = reference[loop - 1] if loop <= depth else ''
            expected = {'task_depth': depth, 'loop': loop, 'intermediate_target': target,
                'intermediate_correct': row['prediction'] == target if target else '',
                'final_target': state, 'final_correct': row['prediction'] == state,
                'stopped': loop == executed}
            if any(row[k] != str(v) for k, v in expected.items()):
                raise ValueError('Native trajectory target or verdict differs')
            if loop < executed and float(row['stop_probability']) >= threshold:
                raise ValueError('Native execution continues past the first stop crossing')
            rows_checked += 1
        signaled = float(rows[-1]['stop_probability']) >= threshold
        if not signaled and executed != summary['loops']:
            raise ValueError('Missing native stop must use the safety cap')
        predictions = [r['prediction'] for r in rows]
        correct = predictions[-1] == state
        expected = {'prediction': predictions[-1], 'target': state, 'correct': correct,
            'complete_trajectory': predictions[:depth] == reference, 'policy': 'completion',
            'stop_reason': 'head' if signaled else 'cap', 'exact_stop': signaled and executed == depth,
            'early_stop': signaled and executed < depth, 'late_stop': signaled and executed > depth,
            'cap_fallback': not signaled, 'joint_success': signaled and executed == depth and correct}
        if any(decision[k] != str(v) for k, v in expected.items()):
            raise ValueError('Native decision verdict differs from executed trajectory')
        for key, value in expected.items():
            if isinstance(value, bool):
                totals[key] += value
    for key in ('exact_stop', 'early_stop', 'late_stop', 'cap_fallback', 'joint_success'):
        if abs(summary['completion'][key + '_rate'] - totals[key] / len(tasks)) > 1e-12:
            raise ValueError('Native aggregate stopping verdict differs')
    if (abs(summary['accuracy'] - totals['correct'] / len(tasks)) > 1e-12
            or abs(summary['complete_trajectory_accuracy'] - totals['complete_trajectory'] / len(tasks)) > 1e-12
            or summary['executed_example_loops'] != rows_checked):
        raise ValueError('Native aggregate answer/trajectory arithmetic differs')
    return {'passed': True, 'native_questions': len(tasks), 'transition_rows_checked': rows_checked,
        'success_counts': dict(totals), 'data_sha256': sha256_file(data),
        'inputs_sha256': {name: sha256_file(results / name) for name in ('summary.json', 'decisions.csv', 'trajectories.csv')},
        'audit_source_sha256': sha256_file(Path(__file__))}


def audit_benchmark(results: Path, graphs_file: Path) -> dict:
    """Audit raw saved symbols against graph execution; never run/fit a model.

    Deliberately do not call quality_arrays or the task reference executor: an
    independent scalar traversal checks the inputs and scoring implementation.
    """
    summary = json.loads((results / 'summary.json').read_text())
    manifest = json.loads((results / 'dataset_manifest.json').read_text())
    if (summary['status'] != 'complete' or not summary['native_fidelity']['passed']
            or not summary['inference_files_unchanged'] or sha256_file(graphs_file) != manifest['graphs_sha256']):
        raise ValueError('Need completed, native-verified results with original graph bytes')
    tasks = [json.loads(line) for line in graphs_file.read_text().splitlines()]
    graphs = list(csv.DictReader((results / 'graphs.csv').open()))
    counts = list(csv.DictReader((results / 'controller_counts.csv').open()))
    maximum, cap = summary['config']['max_depth'], summary['config']['safety_cap']
    if (len(graphs) != len(tasks) or len(tasks) != summary['graphs'] or len(counts) != maximum
            or [int(r['depth']) for r in counts] != list(range(1,maximum+1))):
        raise ValueError('Saved graph/count coverage differs from protocol')
    references, prefixes = [], []
    identities, first_errors = set(), Counter()
    for i,(task,row) in enumerate(zip(tasks,graphs,strict=True)):
        mapping = dict(task['mapping'])
        lines = task['prompt'].splitlines()
        if (len(lines)!=4 or dict(re.findall(r'\( ([A-Z]), ([A-Z])\)',lines[0]))!=mapping
                or lines[1]!=f'Start: {task["initial_state"]}' or lines[2]!=f'Steps: {maximum}'
                or lines[3]!='Answer:'):
            raise ValueError('Prompt fields differ from structured graph/request')
        fingerprint = hashlib.sha256(json.dumps(sorted(mapping.items()),separators=(',',':')).encode()).hexdigest()
        if fingerprint != task['mapping_sha256'] or fingerprint != row['mapping_sha256'] or fingerprint in identities:
            raise ValueError('Graph identity mismatch or duplicate')
        identities.add(fingerprint)
        if int(row['graph_index']) != i or len(row['predictions']) != cap:
            raise ValueError('Saved graph order/trajectory budget differs')
        state, states = task['initial_state'], []
        for _ in range(cap):
            state = mapping[state]; states.append(state)
        if task['intermediate_states'] != states[:maximum] or task['final_state'] != states[maximum-1]:
            raise ValueError('Dataset targets differ from independent traversal')
        correct_prefix, prefix, first = True, [], None
        for loop,(prediction,target) in enumerate(zip(row['predictions'][:maximum],states,strict=False),1):
            if prediction != target and first is None:
                first = loop
            correct_prefix = correct_prefix and prediction == target
            prefix.append(correct_prefix)
        if row['first_error_loop'] != (str(first) if first else ''):
            raise ValueError('First error annotation differs')
        first_errors[str(first) if first else 'none'] += 1
        references.append(states); prefixes.append(prefix)
    excluded = set()
    old_data = Path(summary['config']['existing_data'])
    for filename, expected_hash in manifest['old_dataset_sha256'].items():
        path = old_data/filename
        if sha256_file(path)!=expected_hash:
            raise ValueError('Original dataset bytes changed')
        for line in path.open():
            excluded.add(json.loads(line)['mapping_sha256'])
    for filename, expected_hash in manifest.get('extra_exclusion_sha256',{}).items():
        path = Path(filename)
        if sha256_file(path)!=expected_hash:
            raise ValueError('Additional exclusion bytes changed')
        for line in path.open():
            excluded.add(json.loads(line)['mapping_sha256'])
    if identities & excluded:
        raise ValueError('Benchmark graph overlaps an excluded table')
    totals = Counter(); by_count = [Counter() for _ in range(maximum)]
    with gzip.open(results / 'decisions.csv.gz','rt') as handle:
        rows = csv.DictReader(handle)
        checked = 0
        for i in range(len(tasks)):
            for j in range(maximum):
                row = next(rows,None)
                if row is None or int(row['graph_index']) != i or int(row['depth']) != j+1:
                    raise ValueError('Decision coverage/order differs')
                first = int(counts[j]['first_stop']) if counts[j]['first_stop'] else None
                executed = first or cap
                if row['first_stop'] != counts[j]['first_stop'] or int(row['executed_loops']) != executed:
                    raise ValueError('Decision stop differs from saved controller')
                final = graphs[i]['predictions'][j] == references[i][j]
                stopped = graphs[i]['predictions'][executed-1] == references[i][j]
                exact = first == j+1
                values = dict(nominal_final_correct=final,complete_trajectory=prefixes[i][j],exact_stop=exact,
                    early_stop=first is not None and first<j+1,late_stop=first is not None and first>j+1,
                    missing_stop=first is None,stopped_answer_correct=stopped,joint_success=exact and stopped,
                    strict_success=exact and prefixes[i][j],correct_letter_wrong_time=stopped and not exact)
                for key,value in values.items():
                    if row[key] != str(value):
                        raise ValueError(f'Decision flag {key} differs at graph {i}, count {j+1}')
                    totals[key] += value; by_count[j][key] += value
                checked += 1
        if next(rows,None) is not None:
            raise ValueError('Unexpected extra decisions')
    all_rows = [r for r in csv.DictReader((results / 'per_count_and_stratum.csv').open()) if r['stratum']=='all']
    if len(all_rows) != maximum:
        raise ValueError('Per-count aggregate coverage differs')
    for row in all_rows:
        j = int(row['depth'])-1
        if int(row['graphs']) != len(tasks) or any(abs(float(row[k])-by_count[j][k]/len(tasks))>1e-12 for k in METRICS):
            raise ValueError('Per-count aggregate arithmetic differs')
    for key in METRICS:
        if abs(summary['cohorts']['all']['metrics'][key]['rate']-totals[key]/checked)>1e-12:
            raise ValueError('Overall aggregate arithmetic differs')
    return {'passed':True,'graphs':len(tasks),'decisions_checked':checked,'excluded_graphs':len(excluded),
        'graph_overlap':0,'success_counts':dict(totals),
        'first_error_counts':dict(sorted(first_errors.items())),
        'graphs_sha256':sha256_file(graphs_file),'summary_sha256':sha256_file(results/'summary.json'),
        'decisions_sha256':sha256_file(results/'decisions.csv.gz'),'audit_source_sha256':sha256_file(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--graphs',type=Path)
    parser.add_argument('--native-results',type=Path)
    parser.add_argument('--native-data',type=Path)
    args = parser.parse_args()
    if args.native_results is not None:
        if args.native_data is None or args.results is not None or args.graphs is not None:
            parser.error('Use --native-results and --native-data together, without benchmark arguments')
        destination = args.native_results/'native_reference_audit.json'
        if destination.exists():
            raise ValueError('Audit output exists; refuse overwrite')
        result = audit_native_results(args.native_results, args.native_data)
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Native audit passed: {result["native_questions"]:,} queries, {result["transition_rows_checked"]:,} transitions')
        return
    if args.results is None or args.graphs is None or args.native_data is not None:
        parser.error('Use --results and --graphs together')
    destination = args.results/'result_reference_audit.json'
    if destination.exists():
        raise ValueError('Audit output exists; refuse overwrite')
    result = audit_benchmark(args.results,args.graphs)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Independent audit passed: {result["decisions_checked"]:,} decisions on {result["graphs"]:,} graphs')


if __name__ == '__main__':
    main()

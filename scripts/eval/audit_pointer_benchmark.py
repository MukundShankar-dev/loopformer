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
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--graphs',type=Path,required=True)
    args = parser.parse_args()
    destination = args.results/'result_reference_audit.json'
    if destination.exists():
        raise ValueError('Audit output exists; refuse overwrite')
    result = audit_benchmark(args.results,args.graphs)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Independent audit passed: {result["decisions_checked"]:,} decisions on {result["graphs"]:,} graphs')


if __name__ == '__main__':
    main()

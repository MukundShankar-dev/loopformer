"""Plot inputs must distinguish recovery, strict prefixes and stopping."""
from dataclasses import replace
import csv
import gzip
import json

import pytest
import torch

from scripts.dataset.pointer import generate_unconditioned_example, mapping_fingerprint, render_prompt
from scripts.eval.audit_pointer_figures import audit_plot_tables, audit_numeric
from scripts.eval.benchmark_metrics import quality_arrays, grouped_rows
from scripts.eval.pointer_task import sha256_file


def write_rows(path, data):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)


@pytest.fixture
def recovered_results(tmp_path):
    mapping = {chr(65 + i): 'A' for i in range(26)}
    mapping.update(A='B', B='C', C='A')
    pairs = [list(item) for item in mapping.items()]
    task = replace(generate_unconditioned_example(8, 4, 'benchmark', 0),
        mapping=pairs, mapping_sha256=mapping_fingerprint(mapping), initial_state='A',
        intermediate_states=list('BCAB'), final_state='B', prompt=render_prompt(pairs, 'A', 4))
    graph_file = tmp_path / 'raw.jsonl'
    graph_file.write_text(json.dumps(task.to_dict()) + '\n')
    metadata = dict(graph_index=0, example_id=task.example_id, mapping_sha256=task.mapping_sha256,
        dataset_seed=8, graph_mode='random_function', transient_length=0, cycle_period=3)
    # Incorrect at loop 2, then recovers at loops 3 and 4. Stops at every requested N.
    predicted = torch.tensor([[ord(c) - 65 for c in 'BXABC']])
    direct = torch.tensor([[ord(c) - 65 for c in 'BCABC']])
    arrays, loops = quality_arrays([task], predicted, [1, 2, 3, 4], 4, direct)
    write_rows(tmp_path / 'graphs.csv', [{**metadata, 'predictions': 'BXABC',
        'r_predictions': 'BCABC', 'first_error_loop': 2}])
    write_rows(tmp_path / 'controller_counts.csv', [dict(depth=n, first_stop=n) for n in range(1, 5)])
    write_rows(tmp_path / 'per_count_and_stratum.csv', grouped_rows(arrays, [metadata]))
    write_rows(tmp_path / 'per_loop.csv', loops)
    decisions = [{**metadata, 'depth': n, 'first_stop': n, 'executed_loops': n,
        **{k: bool(v[0, n - 1]) for k, v in arrays.items()}} for n in range(1, 5)]
    with gzip.open(tmp_path / 'decisions.csv.gz', 'wt') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader(); writer.writerows(decisions)
    summary = {'graphs': 1, 'queries': 4, 'config': {'max_depth': 4, 'safety_cap': 5},
        'cohorts': {'all': {'metrics': {k: {'rate': float(v.mean())} for k, v in arrays.items()}}}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    (tmp_path / 'dataset_manifest.json').write_text(json.dumps({'graphs_sha256': sha256_file(graph_file)}))
    return tmp_path, graph_file


def test_recovery_changes_endpoint_success_without_repairing_prefix(recovered_results):
    path, graphs = recovered_results
    audit = audit_plot_tables(path, graphs)
    last = audit['depth_checks'][-1]
    assert last == {'requested_steps': 4, 'denominator_graphs': 1,
        'all_steps_correct_graphs': 0, 'final_answer_and_exact_stop_graphs': 1, 'exact_stop_graphs': 1}
    assert audit['first_c_error_counts'] == {'2': 1}


@pytest.mark.parametrize('filename,column,replacement,error', [
    ('per_count_and_stratum.csv', 'joint_success', '0.2', 'joint_success'),
    ('per_count_and_stratum.csv', 'complete_trajectory_wilson_95_low', '0.2', 'complete_trajectory/low'),
    ('per_loop.csv', 'correct_prefix_graphs', '0', 'denominator'),
    ('per_loop.csv', 'accuracy', '0.2', 'accuracy'),
])
def test_rejects_corrupt_plot_rates_intervals_and_denominators(recovered_results, filename, column, replacement, error):
    path, graphs = recovered_results
    with (path / filename).open() as handle:
        data = list(csv.DictReader(handle))
    data[0][column] = replacement
    write_rows(path / filename, data)
    with pytest.raises(ValueError, match=error):
        audit_plot_tables(path, graphs)


def test_numeric_audit_rejects_wrong_individual_flags_even_with_right_total(tmp_path):
    summary = {'rollout_max': 2, 'exact_stop_questions': 1, 'first_failed_stop_count': 2}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    data = [dict(requested=1, first_stop=1, exact_stop=True, early_stop=False, late_stop=False, missing_stop=False),
        dict(requested=2, first_stop=1, exact_stop=False, early_stop=True, late_stop=False, missing_stop=False)]
    write_rows(tmp_path / 'countdown.csv', data)
    assert audit_numeric(tmp_path)['exact_stop_questions'] == 1
    data[1]['early_stop'] = False
    write_rows(tmp_path / 'countdown.csv', data)
    with pytest.raises(ValueError, match='Numeric flags'):
        audit_numeric(tmp_path)

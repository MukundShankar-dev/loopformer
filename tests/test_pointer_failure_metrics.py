"""Independent first-error taxonomy and decoded-transition denominators."""
from dataclasses import replace

import pytest

from scripts.dataset.pointer import generate_unconditioned_example
from scripts.eval.pointer_failure_metrics import first_error_category, failure_metrics, reference_trace


def test_cyclic_category_priority_and_outside_orbit():
    mapping = dict(A='B', B='C', C='A', D='D')
    targets = reference_trace(mapping, 'A', 5)
    assert targets == 'BCABC'
    assert first_error_category(mapping, 'A', targets, 'A', 0) == 'repeat previous state'
    assert first_error_category(mapping, 'A', targets, 'B', 2) == 'earlier visited state'
    assert first_error_category(mapping, 'A', targets, 'C', 0) == 'later reachable state'
    assert first_error_category(mapping, 'A', targets, 'D', 0) == 'outside start orbit'
    with pytest.raises(ValueError):
        first_error_category(mapping, 'A', targets, 'B', 0)


def test_first_errors_count_once_and_recovery_does_not_erase_them():
    task = replace(generate_unconditioned_example(31, 5, 'benchmark', 0),
        initial_state='A', mapping=[['A', 'B'], ['B', 'C'], ['C', 'A'], ['D', 'D']])
    row = {'mapping_sha256': task.mapping_sha256, 'example_id': task.example_id, 'graph_index': 0,
           'graph_mode': 'random_function', 'dataset_seed': 307, 'cycle_period': 3, 'transient_length': 0,
           'predictions': 'BDDCC', 'r_predictions': 'BCABC', 'first_error_loop': 2}
    result = failure_metrics([task], [row], 5)
    assert result['summary']['first_failed_graphs'] == 1
    assert result['summary']['failed_graphs_correct_final'] == 1
    assert result['confusion'].sum() == 1
    dynamics = {r['category']: r for r in result['dynamics'] if r['graph_mode'] == 'random_function'}
    assert dynamics['follow edge from prior decode']['transitions'] == 1
    assert dynamics['recover to reference']['transitions'] == 1
    assert dynamics['other wrong transition']['transitions'] == 1
    assert all(r['prior_wrong_transitions'] == 3 for r in dynamics.values())
    row['first_error_loop'] = 3
    with pytest.raises(ValueError):
        failure_metrics([task], [row], 5)

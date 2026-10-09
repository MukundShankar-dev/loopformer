"""Comparisons cannot hide execution errors or reuse changed executor weights."""
from dataclasses import replace
import pytest
import torch
from scripts.dataset.pointer import generate_unconditioned_example
from scripts.eval.checkpoint_comparison import assert_executor_tensors, compose_row, executor_spec, nominal_fields


def test_controller_change_allowed_but_executor_change_rejected():
    old = {'recurrent.weight': torch.tensor([2.]), 'completion_head.weight': torch.tensor([1.])}
    new = {'recurrent.weight': torch.tensor([2.]), 'completion_head.different': torch.tensor([4.])}
    assert assert_executor_tensors(old, new) == 1
    new['recurrent.weight'] += .001
    with pytest.raises(AssertionError):
        assert_executor_tensors(old, new)
    spec = {'executor': {'isolated': True, 'controller_size': 128}, 'completion_head': {'intermediate': 128},
            'controller_training': {'seed': 61}, 'recurrence_mode': 'fixed_prompt'}
    changed = {**spec, 'executor': {'isolated': True, 'controller_kind': 'shared_number', 'controller_size': 1}}
    assert executor_spec(spec) == executor_spec(changed)
    changed['executor']['isolated'] = False
    assert executor_spec(spec) != executor_spec(changed)


def test_recovery_is_final_correct_but_never_a_complete_trajectory():
    task = generate_unconditioned_example(31, 3, 'benchmark', 0)
    predicted = [ord(c) - 65 for c in task.intermediate_states]
    predicted[1] = (predicted[1] + 1) % 26
    row = nominal_fields(task, predicted)
    assert row['nominal_final_correct'] and not row['complete_trajectory'] and row['first_error_loop'] == 2
    with pytest.raises(ValueError):
        nominal_fields(task, predicted[:2])


def test_cyclic_letter_at_wrong_loop_and_missing_cap_remain_failures():
    task = replace(generate_unconditioned_example(30, 3, 'benchmark', 0), intermediate_states=['A'] * 3, final_state='A')
    for first, missing in ((1, False), (None, True)):
        stopped = {'first_stop': first, 'exact_stop': False, 'missing_stop': missing,
                   'stopped_answer_correct': True, 'joint_success': False}
        row = compose_row(task, {'depth': 3}, stopped, [0, 0, 0])
        assert row['complete_trajectory'] and row['correct_letter_wrong_time']
        assert not row['joint_success'] and not row['strict_success']


def test_independent_audit_rejects_wrong_time_success(tmp_path):
    import csv
    import json
    from scripts.eval.audit_checkpoint_comparison import audit_comparison
    from scripts.eval.loop_metrics import write_csv
    from scripts.eval.pointer_task import sha256_file
    from scripts.eval.checkpoint_comparison import aggregate
    task = replace(generate_unconditioned_example(30, 3, 'benchmark', 0), initial_state='A',
        mapping=[[chr(i + 65), 'A'] for i in range(26)], intermediate_states=['A'] * 3, final_state='A')
    graphs = tmp_path / 'graphs.jsonl'; graphs.write_text(json.dumps(task.to_dict()) + '\n')
    output = tmp_path / 'result'; output.mkdir()
    rows = []
    for n in (1, 2, 3):
        variant = replace(task, task_depth=n, intermediate_states=['A'] * n)
        row = compose_row(variant, {'model': 'old', 'graph_index': 0, 'depth': n,
            'mapping_sha256': task.mapping_sha256, 'example_id': f'{task.example_id}-steps-{n}'},
            {'first_stop': 1, 'executed_loops': 1, 'exact_stop': n == 1, 'early_stop': n > 1,
             'late_stop': False, 'missing_stop': False, 'stopped_answer_correct': True,
             'joint_success': n == 1, 'prediction': 'A', 'target': 'A'}, [0] * n)
        rows.append(row)
    summary = {'status': 'complete', 'inference_files_unchanged': True, 'graphs': 1,
        'questions_per_model': 3, 'config': {'models': [{'name': 'old'}], 'counts': [1, 2, 3], 'safety_cap': 4},
        'models': {'old': {**aggregate(rows), 'native_fidelity': {'passed': True}, 'inference_files_unchanged': True}}}
    (output / 'summary.json').write_text(json.dumps(summary))
    (output / 'freeze.json').write_text(json.dumps({'source_graphs_sha256': sha256_file(graphs), 'selected_graph_indices': [0]}))
    write_csv(output / 'decisions.csv', rows)
    assert audit_comparison(output, graphs)['decisions_checked'] == 3
    rows[1]['joint_success'] = True
    write_csv(output / 'decisions.csv', rows)
    with pytest.raises(ValueError, match='joint_success'):
        audit_comparison(output, graphs)


def test_assemble_mixed_schemas_without_inference_and_reject_changed_checkpoint(tmp_path):
    import csv
    import json
    from scripts.eval.checkpoint_comparison import aggregate, assemble_comparison
    from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
    from scripts.eval.loop_metrics import write_csv
    output = tmp_path / 'result'; output.mkdir()
    models, hashes = [], {}
    for name in ('legacy', 'gru'):
        model = tmp_path / name; model.mkdir()
        (model / 'metadata.json').write_text('{}')
        models.append({'name': name, 'model': str(model)})
        hashes[name] = checkpoint_hashes(model)
        arm = output / name; arm.mkdir()
        task = replace(generate_unconditioned_example(30, 1, 'benchmark', 0), intermediate_states=['A'], final_state='A')
        row = compose_row(task, {'model': name, 'model_label': name, 'graph_index': 0, 'depth': 1,
            'dataset_seed': 307, 'cycle_period': 1, 'transient_length': 0, 'example_id': 'a', 'mapping_sha256': 'graph'},
            {'first_stop': 1, 'executed_loops': 1, 'exact_stop': True, 'early_stop': False,
             'late_stop': False, 'missing_stop': False, 'stopped_answer_correct': True,
             'joint_success': True, 'prediction': 'A', 'target': 'A'}, [0])
        if name == 'gru':
            row.update(condition='matched', executor_nominal_correct=True, stop_probability=.99)
        write_csv(arm / 'decisions.csv', [row])
        (arm / 'summary.json').write_text(json.dumps({**aggregate([row]),
            'native_fidelity': {'passed': True}, 'inference_files_unchanged': True}))
    config = {'models': models, 'counts': [1]}
    (output / 'freeze.json').write_text(json.dumps({'config': config, 'inference_sha256': hashes, 'selected_graph_indices': [0]}))
    result = assemble_comparison(config, output)
    assert result['questions_per_model'] == 1
    with (output / 'decisions.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and all(r['joint_success'] == 'True' for r in rows)
    assert all('stop_probability' not in r for r in rows)
    (tmp_path / 'gru/metadata.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed checkpoint'):
        assemble_comparison(config, output)

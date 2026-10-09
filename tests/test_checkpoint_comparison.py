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

import csv
import json

import pytest

from scripts.eval.compare_checkpoints import compare_runs


def make_run(path, predictions):
    path.mkdir()
    summary = dict(status='complete', checkpoint=str(path), data_sha256='same', loops=3,
                   dtype='float32', batch_size=16, scoring='symbols', prompt_format='plain')
    (path / 'summary.json').write_text(json.dumps(summary))
    rows = []
    for loop, (prediction, target) in enumerate(zip(predictions, 'BCD'), 1):
        rows.append(dict(example_id='a', task_depth=3, loop=loop, split='depth_test', seed=47,
                         initial_state='A', prediction=prediction, intermediate_target=target,
                         intermediate_correct=prediction == target, final_correct=prediction == 'D'))
    with (path / 'trajectories.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_risk_sets_and_paired_recovery(tmp_path):
    a = make_run(tmp_path / 'a', 'BXD')
    b = make_run(tmp_path / 'b', 'BCD')
    depths, risks, pairs, _ = compare_runs([a, b])
    assert depths[0]['trajectory_accuracy'] == 0  # final recovery is not a perfect trajectory
    assert [r['conditional_transition_accuracy'] for r in risks[:3]] == [1, 0, None]
    assert pairs[1]['correct_prefix_delta'] == 2
    assert depths[1]['trajectory_accuracy'] == 1


def test_reject_mismatched_evaluations(tmp_path):
    a = make_run(tmp_path / 'a', 'BCD')
    b = make_run(tmp_path / 'b', 'BCD')
    path = b / 'summary.json'
    summary = json.loads(path.read_text())
    summary['data_sha256'] = 'different'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='identical'):
        compare_runs([a, b])

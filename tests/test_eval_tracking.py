import argparse
import json
from pathlib import Path

import pytest

from scripts.eval.tracking import report_saved_run
from scripts.eval.paired_steps import variants
from scripts.dataset.pointer import generate_example


def args(mode):
    return argparse.Namespace(wandb_mode=mode, wandb_project='loopformer', wandb_entity=None)


def test_disabled_needs_no_artifacts(tmp_path):
    report_saved_run(tmp_path, args('disabled'))
    assert list(tmp_path.iterdir()) == []


def test_offline_evaluation_upload(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps({'status':'complete','accuracy':.5,'checkpoint':'model/step-5'}))
    (tmp_path/'examples.csv').write_text('example_id,task_depth,trajectory_correct\nx,14,True\n')
    (tmp_path/'depth_comparison.csv').write_text('checkpoint,task_depth,trajectory_accuracy\na,13,0.5\nb,13,0.3\n')
    (tmp_path/'conditional_transitions.csv').write_text('checkpoint,task_depth,loop,conditional_transition_accuracy\na,13,1,1\na,13,2,\nb,13,1,0.5\n')
    (tmp_path/'adapter_model.pt').write_bytes(b'never upload weights')
    report_saved_run(tmp_path,args('offline'))
    saved=json.loads((tmp_path/'wandb_eval_run.json').read_text())
    assert set(saved['artifact_sha256']) == {'summary.json','examples.csv','depth_comparison.csv','conditional_transitions.csv'}
    assert saved['project'] == 'loopformer'


def test_reject_incomplete_upload(tmp_path):
    (tmp_path/'summary.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError,match='completed'):
        report_saved_run(tmp_path,args('offline'))


def test_two_digit_matched_variants():
    source=generate_example(47,20,'depth_test',0)
    tasks=variants([source],1,[12,14,16,18,20])
    assert len(tasks)==5
    assert all(t.mapping==source.mapping and t.initial_state==source.initial_state for t in tasks)
    assert all(t.intermediate_states==source.intermediate_states[:t.task_depth] for t in tasks)
    assert all(t.intermediate_states[:12]==tasks[0].intermediate_states for t in tasks)
    with pytest.raises(ValueError):
        variants([source],1,[20,12])

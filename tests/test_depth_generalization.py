"""Depth exposure, initialization identity, and fair evaluation pairing."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.eval.depth_comparison import compare_depth_runs
from scripts.training.initialization import validate_initialization
from scripts.recurrent_qwen.checkpoint import FORMAT


def test_initialization_rejects_incompatible_weights_and_hidden_exposure(tmp_path):
    spec = {key: 1 for key in ('base_model', 'revision', 'recurrent_start', 'recurrent_end', 'lora_rank',
                              'lora_alpha', 'symbols', 'token_ids', 'prompt_format', 'loss_vocabulary')}
    source = {**spec, 'format': FORMAT, 'train_max_depth': 4}
    (tmp_path / 'recurrent_config.json').write_text(json.dumps(source))
    assert validate_initialization(tmp_path, {**spec, 'train_max_depth': 6}) == source
    for key in spec:
        with pytest.raises(ValueError, match=key):
            validate_initialization(tmp_path, {**spec, key: 2, 'train_max_depth': 6})
    with pytest.raises(ValueError, match='recorded training depth'):
        validate_initialization(tmp_path, {**spec, 'train_max_depth': 3})


def test_absolute_pairing_relative_offsets_and_comparison_cli(tmp_path):
    reference, candidate = [], []
    for role, maximum, paths in [('reference', 4, reference), ('candidate', 6, candidate)]:
        for i, depths in enumerate((range(1, 9), range(9, 17))):
            p = tmp_path / f'{role}-{i}';p.mkdir();paths.append(p)
            summary = {'status': 'complete', 'test_mode': False, 'limit': None, 'data_sha256': f'data-{i}',
                       'selected_examples': 1000, 'loops': max(depths), 'dtype': 'float32', 'device': 'cpu',
                       'batch_size': 1, 'seed': 17, 'scoring': 'argmax', 'prompt_format': 'dataset_raw',
                       'source_sha256': {}, 'checkpoint': role, 'local_checkpoint_sha256': {'adapter': role},
                       'model': {'base_model': 'same', 'revision': 'same', 'symbols': [], 'token_ids': [], 'train_max_depth': maximum},
                       'by_depth': {str(d): {'examples': 125, 'trajectory_accuracy': .5, 'loss': 1.} for d in depths},
                       'depth_by_loop': {str(d): {str(d): {'final_accuracy': .6}} for d in depths}}
            (p / 'summary.json').write_text(json.dumps(summary))
    rows = compare_depth_runs(reference, candidate)
    assert len(rows) == 32
    assert {(r['role'], r['task_depth']) for r in rows if r['steps_beyond_training'] == 2} == {('reference', 6), ('candidate', 8)}
    assert next(r for r in rows if r['role'] == 'candidate' and r['task_depth'] == 6)['region'] == 'trained'
    assert all(r['region'] == 'far_ood' for r in rows if r['task_depth'] >= 9)
    p = candidate[0] / 'summary.json';s=json.loads(p.read_text());s['data_sha256']='wrong';p.write_text(json.dumps(s))
    with pytest.raises(ValueError, match='data_sha256'):
        compare_depth_runs(reference, candidate)
    # Plan-only orchestration must not need adapter weights or run inference.
    for name, maximum in [('model4', 4), ('model6', 6)]:
        model = tmp_path / name;model.mkdir()
        (model / 'recurrent_config.json').write_text(json.dumps({'train_max_depth': maximum}))
    data = tmp_path / 'data.jsonl';data.write_text('{}\n')
    output = tmp_path / 'unused'
    result = subprocess.run([sys.executable, '-m', 'scripts.eval.depth_generalization',
                             '--reference-model', str(tmp_path / 'model4'), '--model', str(tmp_path / 'model6'),
                             '--data', str(data), '--output', str(output), '--dry-run'],
                            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True)
    assert 'no inference or writes' in result.stdout and not output.exists()

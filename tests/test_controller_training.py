"""Frozen executor fidelity, selection discipline and controller-training handoff."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from test_executor_upgrade import setup, configured
from test_pointer_launchers import launcher_repo
from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset
from scripts.dataset.pointer import SYMBOLS, generate_unconditioned_example
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint, save_checkpoint
from scripts.training.controller import ControllerConfig, export_checkpoint, learning_rate, select_graphs, selection_key


def test_count_pairing_and_selection_ignore_unseen_counts():
    tasks = [generate_unconditioned_example(i + 40, 1, 'train', i) for i in range(6)]
    panel = select_graphs(tasks, 3, [1, 2, 4], 83)
    assert len(panel) == 9 and len({t.mapping_sha256 for t in panel}) == 3
    for i in range(0, 9, 3):
        assert [t.task_depth for t in panel[i:i+3]] == [1, 2, 4]
        assert panel[i].mapping == panel[i+2].mapping
    first = {'trained': {'exact_stop': .8, 'loss': .5}, 'extrapolation': {'exact_stop': 0}}
    second = {'trained': {'exact_stop': .7, 'loss': .1}, 'extrapolation': {'exact_stop': 1}}
    assert selection_key(first) < selection_key(second)
    assert selection_key({'trained': {'exact_stop': .8, 'loss': .4}}) < selection_key(first)
    config = ControllerConfig()
    config.validate()
    assert learning_rate(config, config.warmup_steps) == config.learning_rate
    assert learning_rate(config, config.steps) == pytest.approx(config.learning_rate * config.min_lr_ratio)
    with pytest.raises(ValueError):
        replace(config, steps=0).validate()


def source_checkpoint(setup):
    base, tokenizer, tokens, _, path = setup
    options = dict(train_scope='recurrent', bridge=True, direct_readout=True, controller_size=8,
                   isolated=True, prefix_reuse=True, gradient_checkpointing=False)
    model = configured(base, tokenizer, **options)
    spec = dict(base_model=str(path / 'base'), revision='main', recurrent_start=1, recurrent_end=2,
        recurrence_mode='fixed_prompt', executor=options, completion_head={'intermediate': 8},
        symbols=list(SYMBOLS), token_ids=list(tokens.values()), prompt_format='dataset_raw',
        loss_vocabulary='symbols', train_max_depth=2, train_depths=[1, 2])
    save_checkpoint(path / 'source', model, tokenizer, spec, {})
    return model, tokenizer, tokens, path


def test_export_preserves_every_noncontroller_tensor_and_loads(setup):
    model, _, _, path = source_checkpoint(setup)
    original = torch.load(path / 'source/adapter_model.pt', weights_only=True)
    with torch.no_grad():
        model.completion_head.readout.bias.add_(1)
    export_checkpoint(path / 'source', path / 'export', model.completion_head, {'step': 3})
    changed = torch.load(path / 'export/adapter_model.pt', weights_only=True)
    assert original.keys() == changed.keys()
    for key in original:
        if not key.startswith('completion_head.'):
            torch.testing.assert_close(original[key], changed[key], atol=0, rtol=0)
    assert not torch.equal(original['completion_head.readout.bias'], changed['completion_head.readout.bias'])
    loaded, _, spec = load_recurrent_checkpoint(path / 'export')
    assert spec['controller_training']['step'] == 3
    for key, value in model.completion_head.state_dict().items():
        torch.testing.assert_close(loaded.completion_head.state_dict()[key], value, atol=0, rtol=0)


def test_controller_cli_cache_training_export_and_native_eval(setup):
    _, tokenizer, tokens, path = source_checkpoint(setup)
    dataset = DatasetConfig(seed=61, graph_mode='mixture', paired_horizons=True,
        train_count=8, validation_count=6, test_count=2, depth_test_count=2, max_train_depth=2, max_eval_depth=3)
    write_dataset(path / 'data', generate_dataset(dataset, tokenizer, tokens), dataset, tokens, {})
    config = ControllerConfig(source=str(path / 'source'), data=str(path / 'data'), device='cpu',
        train_graphs=3, validation_graphs=2, extraction_batch_size=2, batch_size=4, steps=4,
        warmup_steps=1, eval_every=2, log_every=1, evaluation_max_depth=3)
    (path / 'config.json').write_text(json.dumps(config.to_dict()))
    original = {p.name: p.read_bytes() for p in (path / 'source').iterdir() if p.is_file()}
    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', module, *map(str, args), '--wandb-mode', 'disabled'], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    run('scripts.training.train_controller', '--config', path / 'config.json', '--output', path / 'preview', '--dry-run')
    assert not (path / 'preview').exists()
    run('scripts.training.train_controller', '--config', path / 'config.json', '--output', path / 'run')
    run('scripts.training.train_controller', '--config', path / 'config.json', '--output', path / 'reuse',
        '--features-cache', path / 'run/features.pt')
    a = json.loads((path / 'run/summary.json').read_text()); b = json.loads((path / 'reuse/summary.json').read_text())
    assert a['status'] == 'complete' and a['best_step'] == b['best_step'] and a['validation'] == b['validation']
    assert json.loads((path / 'run/best_checkpoint.json').read_text())['path'] == 'best'
    state = torch.load(path / 'run/best/adapter_model.pt', weights_only=True)
    source = torch.load(path / 'source/adapter_model.pt', weights_only=True)
    selected = torch.load(path / 'run/best_controller.pt', weights_only=True)
    for name, value in state.items():
        expected = selected[name.removeprefix('completion_head.')] if name.startswith('completion_head.') else source[name]
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    run('scripts.eval.loop_test', '--model', path / 'run/best', '--data', path / 'data/validation.jsonl',
        '--device', 'cpu', '--loops', 3, '--stop-policy', 'completion', '--stop-threshold', .5, '--output', path / 'stopped')
    assert json.loads((path / 'stopped/summary.json').read_text())['status'] == 'complete'
    for name, payload in original.items():
        assert (path / 'source' / name).read_bytes() == payload


@pytest.mark.parametrize('fail_call', [None, 1, 3])
def test_eval_controller_launcher_uses_export_and_stops_on_failure(launcher_repo, fail_call):
    root, run = launcher_repo
    shutil.copy(Path(__file__).resolve().parents[1] / 'eval_controller.sh', root)
    env = {**os.environ, **({'FAIL_CALL': str(fail_call)} if fail_call else {})}
    result = subprocess.run(['bash', 'eval_controller.sh', str(run)], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == (7 if fail_call else 0), result.stderr
    calls = [json.loads(l) for l in (root / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == (fail_call or 3)
    for call in calls:
        assert call[call.index('--model') + 1] == str(run / 'step-003250')
    assert list((root / 'eval/pointer_diagnostics').glob('*/run.log'))


@pytest.mark.parametrize('dry,fail', [(True, False), (False, False), (False, True)])
def test_train_launcher_dry_run_and_failure_logging(launcher_repo, dry, fail):
    root, _ = launcher_repo
    shutil.copy(Path(__file__).resolve().parents[1] / 'train_controller.sh', root)
    stub = root / 'bin/python'
    stub.write_text(stub.read_text().replace("if 'scripts.training.train_pointer' in args:",
        "if 'scripts.training.train_pointer' in args or 'scripts.training.train_controller' in args:"))
    args = ['bash', 'train_controller.sh'] + (['--dry-run'] if dry else [])
    result = subprocess.run(args, cwd=root, env={**os.environ, 'FAIL_TRAIN': '9' if fail else '0'}, capture_output=True, text=True)
    assert result.returncode == (9 if fail else 0)
    call = json.loads((root / 'calls.jsonl').read_text().splitlines()[0])
    assert 'scripts.training.train_controller' in call
    logs = list((root / 'models/stage1_pointer').glob('controller-seed61-launch-*.log'))
    if dry:
        assert not logs and not (root / 'models/stage1_pointer/controller-seed61').exists()
    else:
        assert len(logs) == 1
        assert 'training stderr' in logs[0].read_text()

"""Countdown labels remain supervision; paired controls preserve native inference."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_executor_upgrade import setup
from test_controller_training import source_checkpoint
from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset
from scripts.eval.controller_features import replay
from scripts.eval.controller_comparison import paired_summary
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.controller import ControllerConfig
from scripts.training.controller_remaining import remaining_loss, remaining_predictions, remaining_metrics, initialization_loss
from scripts.training.compare_controllers import comparison_runs


def test_remaining_targets_masking_and_error_metrics():
    predicted = torch.tensor([[2., 1., 0., 100.], [3., 2., 1., 0.]], requires_grad=True)
    depths = torch.tensor([2, 3])
    assert remaining_loss(predicted, depths, 12).item() == 0
    (remaining_loss(predicted + .25, depths, 12)).backward()
    assert predicted.grad[0, 3] == 0
    assert predicted.grad[0, 0] > 0 and predicted.grad[1, 3] > 0
    metrics = remaining_metrics(predicted.detach(), torch.tensor([[-1., 1., -1.], [-1., -1., 1.]]), depths)
    assert metrics['remaining_initial_mae'] == metrics['remaining_decrement_mae'] == 0
    assert metrics['remaining_zero_exact'] == metrics['remaining_stop_zero_agreement'] == 1
    assert metrics['remaining_stop_near_zero_rate'] == 1
    # A correctly timed diagnostic zero is not itself the actual head decision.
    bad_stop = remaining_metrics(predicted.detach(), torch.ones(2, 3), depths)
    assert bad_stop['remaining_zero_exact'] == 1 and bad_stop['remaining_stop_zero_agreement'] == 0
    missing = remaining_metrics(torch.full((2, 4), 10.), -torch.ones(2, 3), depths)
    assert missing['remaining_zero_missing'] == 1 and missing['remaining_stop_near_zero_rate'] is None
    with pytest.raises(ValueError):
        remaining_loss(predicted, torch.tensor([0, 4]), 12)


def test_initialization_weight_is_independent_of_rollout_length():
    predicted = torch.tensor([[3., 99., 99., 99.], [4., 99., 99., 99.]], requires_grad=True)
    depths = torch.tensor([2, 3])
    loss = initialization_loss(predicted, depths, 12)
    torch.testing.assert_close(loss, torch.tensor(1 / 144))
    loss.backward()
    torch.testing.assert_close(predicted.grad[:, 0], torch.full((2,), 1 / 144))
    assert torch.count_nonzero(predicted.grad[:, 1:]) == 0
    for invalid in [replace(ControllerConfig(), initial_loss_weight=1),
                    replace(ControllerConfig(), initial_loss_weight=-1)]:
        with pytest.raises(ValueError):
            invalid.validate()


def test_readout_gradient_scope_and_no_feedback():
    torch.manual_seed(2)
    head = RecurrentController(6, 4)
    readout = torch.nn.Linear(4, 1)
    context = torch.randn(2, 6, requires_grad=True)
    working = torch.randn(2, 3, 6, requires_grad=True)
    logits, initial, memory = replay(head, context, working)
    original = logits.detach().clone()
    predicted = remaining_predictions(readout, initial, memory, 12, detach=True)
    remaining_loss(predicted, torch.tensor([2, 3]), 12).backward()
    assert all(p.grad is None for p in head.parameters())
    assert readout.weight.grad.abs().sum() > 0
    predicted = remaining_predictions(readout, initial, memory, 12)
    remaining_loss(predicted, torch.tensor([2, 3]), 12).backward()
    for module in [head.context, head.cell]:
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    assert head.readout.weight.grad is None
    assert context.grad is None and working.grad is None
    with torch.no_grad():
        readout.weight.fill_(10000)
        changed, _, _ = replay(head, context, working)
    torch.testing.assert_close(changed, original, atol=0, rtol=0)


def test_comparison_plan_preserves_panel_and_rejects_invalid_config(tmp_path):
    config = tmp_path / 'base.json';config.write_text(json.dumps(ControllerConfig().to_dict()))
    spec = {'training_config': str(config), 'optimizer_seeds': [83, 89, 97],
            'remaining_loss_weight': 1., 'remaining_scale': 12.}
    runs = comparison_runs(spec)
    assert len(runs) == 6
    for (_, a), (_, b) in zip(runs[::2], runs[1::2]):
        assert replace(a, remaining_loss_weight=1) == b
        assert a.seed == 83
    for invalid in [replace(ControllerConfig(), remaining_loss_weight=1),
                    replace(ControllerConfig(), remaining_scale=0),
                    replace(ControllerConfig(), optimizer_seed=-1)]:
        with pytest.raises(ValueError):invalid.validate()
    with pytest.raises(ValueError):comparison_runs({**spec, 'optimizer_seeds': [83, 83]})
    summary = paired_summary([{'split': 'validation', 'cohort': 'trained', 'optimizer_seed': seed,
                               'arm': arm, 'exact_stop': value}
        for seed in [83, 89] for arm, value in [('stop_only', .5), ('remaining', .75)]])
    assert summary[0]['paired_exact_stop_delta_mean'] == .25
    assert summary[0]['paired_exact_stop_delta_sd'] == 0


def test_matched_suite_cache_export_and_native_eval(setup):
    _, tokenizer, tokens, path = source_checkpoint(setup)
    dataset = DatasetConfig(seed=61, graph_mode='mixture', paired_horizons=True,
        train_count=8, validation_count=6, test_count=2, depth_test_count=2, max_train_depth=2, max_eval_depth=3)
    write_dataset(path / 'data', generate_dataset(dataset, tokenizer, tokens), dataset, tokens, {})
    config = ControllerConfig(source=str(path / 'source'), data=str(path / 'data'), device='cpu',
        train_graphs=3, validation_graphs=2, extraction_batch_size=2, batch_size=4, steps=4,
        warmup_steps=1, eval_every=2, log_every=1, evaluation_max_depth=3)
    config_path = path / 'config.json';config_path.write_text(json.dumps(config.to_dict()))
    env = {**os.environ, 'WANDB_MODE': 'disabled'}
    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', module, *map(str, args)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    run('scripts.training.train_controller', '--config', config_path, '--output', path / 'baseline')
    spec = {'training_config': str(config_path), 'features_cache': str(path / 'baseline/features.pt'),
            'output': str(path / 'paired'), 'optimizer_seeds': [83], 'remaining_loss_weight': 1., 'remaining_scale': 12.}
    # Use absolute output roots; evaluation output is explicitly bound to this fixture.
    comparison_path = path / 'compare.json';comparison_path.write_text(json.dumps(spec))
    run('scripts.training.compare_controllers', '--config', comparison_path, '--dry-run')
    assert not (path / 'paired').exists()
    runs = comparison_runs(spec)
    entries = []
    for name, cfg in runs:
        cp = path / f'{name}.json';cp.write_text(json.dumps(cfg.to_dict()))
        output = path / 'paired' / name
        run('scripts.training.train_controller', '--config', cp, '--output', output,
            '--features-cache', spec['features_cache'])
        entries.append({'name': name, 'path': str(output), 'optimizer_seed': 83,
                        'arm': 'remaining' if cfg.remaining_loss_weight else 'stop_only'})
    original = torch.load(path / 'baseline/last_controller_state.pt', weights_only=True)['head']
    passive = torch.load(path / 'paired/seed-83-stop_only/last_controller_state.pt', weights_only=True)['head']
    for name in original:torch.testing.assert_close(original[name], passive[name], atol=0, rtol=0)
    manifest = {'comparison': spec, 'runs': entries, 'features_cache': spec['features_cache']}
    (path / 'paired/comparison.json').write_text(json.dumps(manifest))
    # Full comparison includes one tiny-model deep extraction and frozen replay for both arms.
    run('scripts.eval.controller_comparison', '--run', path / 'paired', '--output', path / 'evaluation')
    summary = json.loads((path / 'evaluation/summary.json').read_text())
    assert summary['status'] == 'complete'
    assert set(summary['metrics']) == {'seed-83-stop_only', 'seed-83-remaining'}
    assert summary['deep']['live_replay_max_logit_error'] < 2e-5
    selected = path / 'paired/seed-83-remaining'
    assert (selected / 'remaining_trajectories.csv').is_file()
    checkpoint = selected / 'best'
    assert not any('remaining' in k for k in torch.load(checkpoint / 'adapter_model.pt', weights_only=True))
    source = torch.load(path / 'source/adapter_model.pt', weights_only=True)
    exported = torch.load(checkpoint / 'adapter_model.pt', weights_only=True)
    for name in source:
        if not name.startswith('completion_head.'):
            torch.testing.assert_close(source[name], exported[name], atol=0, rtol=0)
    run('scripts.eval.loop_test', '--model', checkpoint, '--data', path / 'data/validation.jsonl',
        '--device', 'cpu', '--loops', 3, '--stop-policy', 'completion', '--stop-threshold', .5, '--output', path / 'native')
    # Reusing the deep cache must produce identical metrics, without Qwen extraction.
    run('scripts.eval.controller_comparison', '--run', path / 'paired', '--output', path / 'evaluation-reuse',
        '--deep-features-cache', path / 'evaluation/deep_features.pt')
    reused = json.loads((path / 'evaluation-reuse/summary.json').read_text())
    assert reused['metrics'] == summary['metrics']
    # The repair retains export/replay contracts while adding direct loop-zero supervision.
    repaired_config = path / 'repair.json'
    repaired_config.write_text(json.dumps(replace(config, remaining_readout=True,
        remaining_loss_weight=1., initial_loss_weight=12.).to_dict()))
    run('scripts.training.train_controller', '--config', repaired_config, '--output', path / 'repair',
        '--features-cache', spec['features_cache'])
    run('scripts.eval.controller_candidate', '--run', path / 'repair',
        '--deep-features-cache', path / 'evaluation/deep_features.pt', '--output', path / 'repair-eval')
    repair_result = json.loads((path / 'repair-eval/summary.json').read_text())
    assert repair_result['metrics']['best']['validation'] == json.loads((path / 'repair/summary.json').read_text())['validation']


@pytest.mark.parametrize('fail_call', [None, 3])
def test_suite_orchestration_and_failure_propagation(tmp_path, monkeypatch, fail_call):
    from scripts.training import compare_controllers
    config = tmp_path / 'base.json';config.write_text(json.dumps(ControllerConfig().to_dict()))
    cache = tmp_path / 'features.pt';cache.touch()
    output = tmp_path / 'comparison'
    spec = {'training_config': str(config), 'features_cache': str(cache), 'output': str(output),
            'optimizer_seeds': [83, 89, 97], 'remaining_loss_weight': 1, 'remaining_scale': 12}
    path = tmp_path / 'comparison.json';path.write_text(json.dumps(spec))
    calls = []
    def fake_run(command, *, check):
        calls.append(command)
        assert check
        if len(calls) == fail_call:
            raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(compare_controllers.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['compare', '--config', str(path), '--dry-run'])
    compare_controllers.main()
    assert not output.exists() and not calls
    monkeypatch.setattr(sys, 'argv', ['compare', '--config', str(path)])
    if fail_call:
        with pytest.raises(subprocess.CalledProcessError):compare_controllers.main()
        assert len(calls) == fail_call
        assert json.loads((output / 'comparison.json').read_text())['status'] == 'running'
    else:
        compare_controllers.main()
        assert len(calls) == 7
        assert all(c[c.index('--features-cache')+1] == str(cache) for c in calls[:6])
        assert calls[-1][calls[-1].index('-m')+1] == 'scripts.eval.controller_comparison'
        assert json.loads((output / 'comparison.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('dry,fail', [(True, False), (False, False), (False, True)])
def test_comparison_shell_logs_and_preserves_exit_code(tmp_path, dry, fail):
    import shutil
    import shlex
    shutil.copy(Path(__file__).resolve().parents[1] / 'train_controller_remaining.sh', tmp_path)
    bin_dir = tmp_path / '.venv/bin';bin_dir.mkdir(parents=True)
    (bin_dir / 'activate').write_text(f'export PATH={shlex.quote(str(bin_dir))}:"$PATH"\n')
    stub = bin_dir / 'python'
    stub.write_text(f'#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n'
                    "Path('call.json').write_text(json.dumps(sys.argv[1:]))\n"
                    "print('suite stderr',file=sys.stderr)\n" + f'sys.exit({7 if fail else 0})\n')
    stub.chmod(0o755)
    result = subprocess.run(['bash', 'train_controller_remaining.sh'] + (['--dry-run'] if dry else []),
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == (7 if fail else 0)
    assert 'scripts.training.compare_controllers' in json.loads((tmp_path / 'call.json').read_text())
    logs = list(tmp_path.glob('models/stage1_pointer/controller-remaining-launch-*.log'))
    assert len(logs) == (0 if dry else 1)
    if logs:assert 'suite stderr' in logs[0].read_text()

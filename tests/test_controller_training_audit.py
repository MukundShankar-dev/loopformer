"""Scientific contracts for the no-update audit, using synthetic saved features."""
from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from scripts.dataset.pointer import generate_unconditioned_example
from scripts.eval.controller_audit_io import load_inputs
from scripts.eval.controller_audit_metrics import loss_components, gradient_audit, fit_audit, gradient_indices
from scripts.eval.controller_features import replay
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.controller import ControllerConfig, evaluate_controller, select_graphs
from scripts.training.controller_remaining import remaining_predictions, remaining_loss


def test_components_and_gradients_reconstruct_real_objective():
    torch.manual_seed(3)
    head, readout = RecurrentController(6, 4), torch.nn.Linear(4, 1)
    context, working = torch.randn(5, 6), torch.randn(5, 4, 6)
    depths = torch.tensor([1, 2, 3, 4, 2])
    before = {n: p.clone() for n, p in head.named_parameters()}
    logits, initial, memories = replay(head, context, working)
    predicted = remaining_predictions(readout, initial, memories, 12.)
    components = loss_components(logits, predicted, depths, 12.)
    auxiliary = remaining_loss(predicted, depths, 12.)
    torch.testing.assert_close(sum(components[k] for k in ('initial', 'intermediate', 'terminal')), auxiliary)
    reference = torch.autograd.grad(components['stop'] + .7 * auxiliary, list(head.parameters()))
    norm = torch.cat([g.flatten() for g in reference]).norm().item()
    rows, cosines, losses = gradient_audit(head, readout, context, working, depths, 12., .7, 2)
    measured = next(r for r in rows if r['scope'] == 'panel_mean' and r['module'] == 'controller' and r['component'] == 'objective')
    assert measured['norm'] == pytest.approx(norm, rel=2e-5, abs=1e-7)
    assert len([r for r in losses if r['scope'] == 'batch']) == 3
    assert all(-1 <= r['cosine'] <= 1 for r in cosines if r['cosine'] is not None)
    assert any(r['cosine'] is None for r in cosines)  # structurally unused heads
    for n, p in head.named_parameters():
        torch.testing.assert_close(p, before[n], atol=0, rtol=0)
        assert p.grad is None
    assert all(p.grad is None for p in readout.parameters())
    # No direct interior supervision at N=1, and no target after N.
    pred = torch.randn(1, 5, requires_grad=True)
    parts = loss_components(torch.zeros(1, 4), pred, torch.tensor([1]), 12.)
    assert parts['intermediate'].item() == 0
    gradient = torch.autograd.grad(parts['initial'] + parts['terminal'], pred)[0]
    assert gradient[0, 2:].count_nonzero() == 0


def test_stop_control_auxiliary_gradients_are_counterfactual():
    torch.manual_seed(4)
    head, readout = RecurrentController(6, 4), torch.nn.Linear(4, 1)
    context, working, depths = torch.randn(3, 6), torch.randn(3, 3, 6), torch.tensor([1, 3, 2])
    rows, _, _ = gradient_audit(head, readout, context, working, depths, 12., 0., 3)
    r = {v['component']: v for v in rows if v['scope'] == 'panel_mean' and v['module'] == 'controller'}
    assert r['objective']['norm'] == r['stop']['norm']
    assert r['remaining']['norm'] > 0
    assert r['remaining']['auxiliary_active_on_controller'] is False


def test_numerical_success_cannot_hide_wrong_time_cyclic_answer(monkeypatch):
    from scripts.eval import controller_audit_metrics as module
    from scripts.eval.controller_learning import stop_rows, aggregate_stops
    # A self-loop makes the final letter correct at every time, including early.
    task = next(t for i in range(200)
                if (t := generate_unconditioned_example(i, 2, 'validation', 0)).intermediate_states[0] == t.final_state)
    traces = [dict(example_id=task.example_id, loop=t, predicted_remaining=float(2-t),
                   target_remaining=2-t, stop_probability=.9 if t else None) for t in range(3)]
    def evaluate(*args, **kwargs):
        kwargs['traces'].extend(traces)
        decisions = stop_rows(torch.tensor([[1., 1.]]),
            torch.tensor([[ord(task.final_state) - ord('A')] * 2]), [task],
            [{'example_id': task.example_id, 'depth': 2}], 'controller')
        return {'trained': aggregate_stops(decisions)}, decisions
    monkeypatch.setattr(module, 'evaluate_controller', evaluate)
    _, rows, cross = fit_audit(None, None, {}, [task], [2], 'cpu', 12., 1)
    assert rows[0]['zero_exact'] and rows[0]['remaining_trajectory_within_half']
    assert rows[0]['wrong_time_correct_letter'] and not rows[0]['joint_success']
    assert next(r['count'] for r in cross if r['signal'] == 'zero_exact' and r['numerical_correct'] and not r['stop_correct']) == 1


@pytest.fixture
def audit_bundle(tmp_path):
    torch.manual_seed(2)
    root = tmp_path / 'comparison'; root.mkdir()
    data = tmp_path / 'data'; data.mkdir()
    train = [generate_unconditioned_example(40+i, 1, 'train', i) for i in range(4)]
    validation = [generate_unconditioned_example(70+i, 1, 'validation', i) for i in range(2)]
    for split, tasks in [('train', train), ('validation', validation)]:
        (data / f'{split}.jsonl').write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in tasks))
    (data / 'manifest.json').write_text(json.dumps({'splits': {split: {'count': len(panel), 'sha256': sha256_file(data / f'{split}.jsonl')} for split, panel in [('train', train), ('validation', validation)]}}))
    trained = [1, 3]
    config = ControllerConfig(data=str(data), device='cpu', train_graphs=4, validation_graphs=2,
        steps=4, warmup_steps=1, evaluation_max_depth=4, remaining_readout=True, remaining_loss_weight=1.)
    tasks = {'train': select_graphs(train, 4, trained, config.seed),
             'validation': select_graphs(validation, 2, [1, 2, 3, 4], config.seed+1)}
    head, readout = RecurrentController(6, 4), torch.nn.Linear(4, 1)
    features = {}
    for split, panel in tasks.items():
        loops = 3 if split == 'train' else 4
        context, working = torch.randn(len(panel), 6), torch.randn(len(panel), loops, 6)
        with torch.no_grad(): logits, _, _ = replay(head, context, working)
        features[split] = {'context': context, 'working': working, 'logits': logits,
                           'predictions': torch.zeros(len(panel), loops, dtype=torch.long)}
    identity = {'source_sha256': 'a'*64, 'source_config_sha256': 'b'*64,
        **{f'{split}_ids': [t.example_id for t in panel] for split, panel in tasks.items()},
        **{f'{split}_data_sha256': sha256_file(data / f'{split}.jsonl') for split in tasks}}
    cache = {'format': 'controller-training-features-v1', 'width': 6, 'controller_width': 4,
             'head': head.state_dict(), 'features': features, 'identity': identity}
    cache_path = tmp_path / 'features.pt';torch.save(cache, cache_path)
    entries = []
    for name, weight in [('stop_only', 0.), ('remaining', 1.)]:
        p = root / name; p.mkdir()
        cfg = replace(config, remaining_loss_weight=weight)
        (p / 'config.json').write_text(json.dumps(cfg.to_dict()))
        torch.save(head.state_dict(), p / 'best_controller.pt')
        torch.save({'state_dict': readout.state_dict(), 'scale': 12., 'step': 3, 'width': 4}, p / 'best_remaining_readout.pt')
        torch.save({'head': head.state_dict(), 'remaining_readout': readout.state_dict(), 'step': 4}, p / 'last_controller_state.pt')
        (p / 'best_selection.json').write_text(json.dumps({'step': 3}))
        metrics, _ = evaluate_controller(head, features['validation'], tasks['validation'], trained, 'cpu', remaining_readout=readout)
        (p / 'summary.json').write_text(json.dumps({'status': 'complete', 'best_step': 3, 'validation': metrics}))
        write_csv(p / 'train_panel.csv', [{'example_id': t.example_id, 'mapping_sha256': t.mapping_sha256,
            'depth': t.task_depth} for t in tasks['train']])
        (p / 'validation_tasks.jsonl').write_text(''.join(json.dumps(t.to_dict())+'\n' for t in tasks['validation']))
        meta = {'status': 'complete', 'features_sha256': sha256_file(cache_path), 'trained_depths': trained,
            'source_checkpoint_sha256': identity['source_sha256'], 'source_config_sha256': identity['source_config_sha256'],
            **{f'{s}_data_sha256': identity[f'{s}_data_sha256'] for s in tasks},
            'train_panel_sha256': sha256_file(p / 'train_panel.csv'), 'validation_tasks_sha256': sha256_file(p / 'validation_tasks.jsonl')}
        (p / 'run.json').write_text(json.dumps(meta))
        entries.append({'name': name, 'path': str(p), 'arm': name, 'optimizer_seed': 83})
    (root / 'comparison.json').write_text(json.dumps({'status': 'complete', 'runs': entries, 'features_cache': str(cache_path)}))
    return root, cache_path, tasks


def test_cli_full_audit_is_read_only_and_has_no_pretrained_dependency(audit_bundle, tmp_path):
    root, cache, _ = audit_bundle
    before = {p: sha256_file(p) for p in root.rglob('*') if p.is_file()}
    before[cache] = sha256_file(cache)
    output = tmp_path / 'audit'
    command = [sys.executable, '-m', 'scripts.eval.controller_training_audit', '--run', str(root),
               '--device', 'cpu', '--gradient-graphs', '2', '--gradient-batch-size', '3',
               '--output', str(output), '--wandb-mode', 'disabled']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['weights_unchanged']
    assert len(summary['fit']) == 4  # both arms and best/final
    for p, digest in before.items(): assert sha256_file(p) == digest
    with (output / 'predictions.csv').open() as handle: rows = list(csv.DictReader(handle))
    assert len(rows) == 64 and {r['split'] for r in rows} == {'train', 'validation'}
    with (output / 'gradient_norms.csv').open() as handle: rows = list(csv.DictReader(handle))
    assert {'controller.context', 'controller.cell', 'controller.observation', 'controller.readout', 'auxiliary'} <= {r['module'] for r in rows}
    assert {'trained', 'interpolation', 'extrapolation'} == {r['cohort'] for r in rows}
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and 'Output exists' in again.stderr


def test_cache_and_checkpoint_mismatch_fail_closed(audit_bundle):
    root, path, tasks = audit_bundle
    a = gradient_indices(tasks['train'], 2, 107)
    assert a == gradient_indices(tasks['train'], 2, 107)
    assert len(a) == 4 and len({tasks['train'][i].mapping_sha256 for i in a}) == 2
    cache = torch.load(path, weights_only=True)
    cache['identity']['train_ids'].reverse()
    torch.save(cache, path)
    with pytest.raises(ValueError, match='task order'):
        load_inputs(root, None, ['best'])


def test_readout_step_mismatch_rejected(audit_bundle):
    from scripts.eval.controller_audit_io import load_heads
    root, path, _ = audit_bundle
    cache, _, entries, _ = load_inputs(root, None, ['best'])
    p = Path(entries[0]['path']) / 'best_remaining_readout.pt'
    saved = torch.load(p, weights_only=True); saved['step'] = 99; torch.save(saved, p)
    with pytest.raises(ValueError, match='metadata mismatch'):
        load_heads(entries[0], 'best', cache, 'cpu')


def test_dry_run_needs_no_artifacts_and_writes_nothing(tmp_path):
    output = tmp_path / 'out'
    result = subprocess.run([sys.executable, '-m', 'scripts.eval.controller_training_audit',
        '--run', str(tmp_path / 'absent'), '--output', str(output), '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0 and not output.exists()


@pytest.mark.parametrize('dry,code', [(True, 0), (False, 0), (False, 7)])
def test_launcher_logs_and_propagates_failure(tmp_path, dry, code):
    shutil.copy(Path(__file__).resolve().parents[1] / 'audit_controller.sh', tmp_path)
    (tmp_path / '.venv/bin').mkdir(parents=True)
    (tmp_path / '.venv/bin/activate').write_text('export PATH="' + str(tmp_path / 'bin') + ':$PATH"\n')
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin/python').write_text(f'#!/bin/sh\necho audit-stub\nexit {code}\n')
    (tmp_path / 'bin/python').chmod(0o755)
    result = subprocess.run(['bash', str(tmp_path / 'audit_controller.sh')] + (['--dry-run'] if dry else []), capture_output=True, text=True)
    assert result.returncode == code
    logs = list(tmp_path.glob('eval/pointer_diagnostics/*.log'))
    assert len(logs) == (0 if dry else 1)
    if logs: assert 'audit-stub' in logs[0].read_text()

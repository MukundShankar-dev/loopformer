"""Controller diagnostics preserve execution and separate probe/train/eval labels."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from test_executor_upgrade import setup, configured
from test_pointer_launchers import launcher_repo
from scripts.dataset.pointer import generate_unconditioned_example, SYMBOLS
from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset
from scripts.eval.controller_features import make_panel, extract_features, replay
from scripts.eval.controller_learning import fit_probe, fit_controller, stop_rows
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.recurrent_qwen.checkpoint import save_checkpoint


def panel():
    source = [generate_unconditioned_example(i + 13, 1, 'validation', i) for i in range(6)]
    return make_panel(source, (2, 2, 2), [1, 2, 3], 71)


def test_panel_keeps_graphs_disjoint_and_all_count_loop_combinations():
    tasks, meta = panel()
    assert len(tasks) == 18
    partitions = {p: {m['mapping_sha256'] for m in meta if m['partition'] == p} for p in ('fit', 'dev', 'holdout')}
    assert len(set.union(*partitions.values())) == 6
    for graph in range(6):
        group = [t for t, m in zip(tasks, meta) if m['graph_index'] == graph]
        assert [t.task_depth for t in group] == [1, 2, 3]
        assert all(t.mapping == group[0].mapping and t.initial_state == group[0].initial_state for t in group)
    assert panel()[1] == meta
    with pytest.raises(ValueError, match='development'):
        make_panel([generate_unconditioned_example(1, 1, 'test', 0)], (1, 1, 1), [1], 1)


def test_live_capture_replays_without_changing_weights_or_executor(setup):
    base, tokenizer, tokens, _, _ = setup
    model = configured(base, tokenizer, controller_size=8, prefix_reuse=True)
    before = deepcopy(model.state_dict())
    tasks, meta = panel()
    features, head, error = extract_features(model, tokenizer, list(tokens.values()), tasks, 3, 4)
    assert error < 2e-5
    assert features['memory'].shape == (18, 3, 8)
    assert features['context'].shape == (18, 16)
    assert not any(x.requires_grad for x in features.values())
    assert not model.completion_head.context._forward_pre_hooks
    assert not model.completion_head.cell._forward_hooks
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    with torch.no_grad():
        logits, _, _ = replay(head, features['context'], features['working'])
    torch.testing.assert_close(logits, features['logits'])
    # Matched count variants produce identical executor predictions.
    for start in range(0, 18, 3):
        assert torch.equal(features['predictions'][start], features['predictions'][start + 2])


def test_probe_heldout_labels_cannot_change_fit_or_selection():
    torch.set_num_threads(1)
    y = torch.arange(3).repeat(12)
    x = torch.nn.functional.one_hot(y, 3).float()
    parts = ['fit'] * 18 + ['dev'] * 9 + ['holdout'] * 9
    a, info = fit_probe(x, y, parts, 3, nonlinear=False, steps=150, seed=3, device='cpu')
    changed = y.clone(); changed[-9:] = (changed[-9:] + 1) % 3
    b, other = fit_probe(x, changed, parts, 3, nonlinear=False, steps=150, seed=3, device='cpu')
    assert torch.equal(a, b)
    assert info['selected_step'] == other['selected_step']
    assert info['dev_cross_entropy'] == other['dev_cross_entropy']
    assert info['accuracy']['holdout'] == 1
    assert other['accuracy']['holdout'] == 0


def test_tiny_fit_updates_only_copy_and_learns_without_counter_inputs():
    torch.set_num_threads(1); torch.manual_seed(3)
    head = RecurrentController(4, 16)
    original = deepcopy(head.state_dict())
    depths = torch.tensor([1, 2, 3])
    features = {'context': torch.eye(4)[:3].requires_grad_(), 'working': torch.zeros(3, 3, 4, requires_grad=True)}
    fitted, history = fit_controller(head, features, depths, torch.arange(3), steps=200,
                                    learning_rate=.02, device='cpu')
    assert history[-1]['loss'] < .1 * history[0]['loss']
    assert history[-1]['exact_stop'] == 1
    assert features['context'].grad is None and features['working'].grad is None
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    assert any(not torch.equal(value, original[name]) for name, value in fitted.state_dict().items())


def test_exact_stop_is_not_correct_letter_or_budget_fallback():
    task = generate_unconditioned_example(1, 3, 'validation', 0)
    predicted = torch.full((3, 3), ord(task.final_state) - ord('A'))
    logits = torch.tensor([[1., 1., 1.], [-1., -1., -1.], [-1., -1., 0.]])
    rows = stop_rows(logits, predicted, [task] * 3, [{}] * 3, 'test')
    assert all(r['stopped_answer_correct'] for r in rows)
    assert [r['exact_stop'] for r in rows] == [False, False, True]
    assert [r['joint_success'] for r in rows] == [False, False, True]
    assert rows[1]['missing_stop'] and rows[1]['executed_loops'] == 3


def test_cli_extract_fit_cache_reuse_and_source_fidelity(setup):
    base, tokenizer, tokens, _, path = setup
    options = dict(train_scope='recurrent', bridge=True, direct_readout=True, controller_size=8,
                   isolated=True, prefix_reuse=True, gradient_checkpointing=False)
    model = configured(base, tokenizer, **options)
    spec = dict(base_model=str(path / 'base'), revision='main', recurrent_start=1, recurrent_end=2,
        recurrence_mode='fixed_prompt', executor=options, completion_head={'intermediate': 8},
        symbols=list(SYMBOLS), token_ids=list(tokens.values()), prompt_format='dataset_raw',
        loss_vocabulary='symbols', train_max_depth=2, train_depths=[1, 2])
    save_checkpoint(path / 'checkpoint', model, tokenizer, spec, {})
    config = DatasetConfig(seed=61, graph_mode='mixture', paired_horizons=True,
        train_count=6, validation_count=12, test_count=2, depth_test_count=2, max_train_depth=2, max_eval_depth=3)
    write_dataset(path / 'data', generate_dataset(config, tokenizer, tokens), config, tokens, {})
    before = {p.name: p.read_bytes() for p in (path / 'checkpoint').iterdir() if p.is_file()}
    common = ['--device', 'cpu', '--fit-graphs', '2', '--dev-graphs', '2', '--holdout-graphs', '2',
              '--tiny-graphs', '1', '--max-depth', '3', '--probe-steps', '2', '--controller-steps', '2',
              '--wandb-mode', 'disabled']
    def run(*args):
        result = subprocess.run([sys.executable, '-m', 'scripts.eval.controller_diagnostic', *common, *map(str, args)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    run('--model', path / 'checkpoint', '--data', path / 'data/validation.jsonl', '--output', path / 'out')
    run('--features-cache', path / 'out/features.pt', '--output', path / 'reused')
    first = json.loads((path / 'out/summary.json').read_text())
    second = json.loads((path / 'reused/summary.json').read_text())
    assert first['probes'] == second['probes'] and first['stopping'] == second['stopping']
    assert first['status'] == 'complete'
    assert set(first['stopping']) == {'original', 'continued_tiny', 'fresh_tiny'}
    rows = list(csv.DictReader((path / 'out/decisions.csv').open()))
    assert len(rows) == 3 * 6 * 3
    for name, data in before.items():
        assert (path / 'checkpoint' / name).read_bytes() == data


@pytest.mark.parametrize('fail', [False, True])
def test_launcher_records_stderr_and_propagates_failure(launcher_repo, fail):
    import os
    root, run = launcher_repo
    shutil.copy(Path(__file__).resolve().parents[1] / 'diagnose_controller.sh', root)
    env = {**os.environ, **({'FAIL_CALL': '1'} if fail else {})}
    result = subprocess.run(['bash', 'diagnose_controller.sh', str(run)], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == (7 if fail else 0), result.stdout + result.stderr
    call = json.loads((root / 'calls.jsonl').read_text().splitlines()[0])
    assert call[2] == 'scripts.eval.controller_diagnostic'
    assert call[call.index('--model') + 1] == str(run / 'step-003250')
    logs = list((root / 'eval/pointer_diagnostics').glob('*.log'))
    assert len(logs) == 1
    if fail:
        assert 'eval failed' in logs[0].read_text()


def test_capture_removes_hooks_after_replay_failure(setup, monkeypatch):
    base, tokenizer, tokens, _, _ = setup
    model = configured(base, tokenizer, controller_size=8)
    def fail(*args, **kwargs):
        raise RuntimeError('simulated replay failure')
    monkeypatch.setattr('scripts.eval.controller_features.replay', fail)
    tasks, _ = panel()
    with pytest.raises(RuntimeError, match='simulated'):
        extract_features(model, tokenizer, list(tokens.values()), tasks[:2], 3, 2)
    assert not model.completion_head.context._forward_pre_hooks
    assert not model.completion_head.context._forward_hooks
    assert not model.completion_head.observation._forward_pre_hooks
    assert not model.completion_head.cell._forward_hooks


def test_real_two_cycle_same_letter_at_early_and_late_loops_is_not_success():
    from dataclasses import replace
    from scripts.dataset.pointer import execute, mapping_fingerprint, render_prompt, validate_example
    task = generate_unconditioned_example(1, 3, 'validation', 0)
    mapping = {s: s for s in SYMBOLS}
    mapping.update(A='B', B='A')
    pairs = [list(pair) for pair in mapping.items()]
    states = execute(mapping, 'A', 3)
    task = replace(task, mapping=pairs, mapping_sha256=mapping_fingerprint(mapping), initial_state='A',
                   intermediate_states=states, final_state=states[-1], prompt=render_prompt(pairs, 'A', 3))
    validate_example(task)
    # B is correct at requested loop 3, but also appears at loops 1 and 5.
    predictions = torch.tensor([[1, 0, 1, 0, 1]] * 3)
    logits = torch.tensor([[1., -1., -1., -1., -1.], [-1., -1., -1., -1., 1.], [-1., -1., 1., -1., -1.]])
    rows = stop_rows(logits, predictions, [task] * 3, [{}] * 3, 'cycle')
    assert all(r['stopped_answer_correct'] for r in rows)
    assert [r['first_stop'] for r in rows] == [1, 5, 3]
    assert [r['joint_success'] for r in rows] == [False, False, True]
    assert rows[0]['early_stop'] and rows[1]['late_stop']

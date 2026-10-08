"""Short-prefix counting targets, immutable feature reuse, and portable training."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest
import torch

from test_executor_upgrade import setup
from test_controller_training import source_checkpoint
from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset
from scripts.dataset.pointer import generate_unconditioned_example
from scripts.eval.controller_features import extract_features
from scripts.training.controller import ControllerConfig, select_graphs
from scripts.training.controller_cache import prompt_features, pair_prefixes
from scripts.training.controller_remaining import prefix_losses, remaining_loss
from scripts.training.objective import completion_loss


def test_prefix_objective_does_not_invent_terminal_at_training_cap():
    logits = torch.zeros(2, 3, requires_grad=True)
    depths = torch.tensor([2, 8])
    predicted = torch.tensor([[2., 1., 0., 99.], [8., 7., 6., 5.]], requires_grad=True)
    stop, numeric = prefix_losses(logits, predicted, depths, 12)
    assert numeric.item() == 0
    stop.backward()
    assert logits.grad[0, 0] > 0 and logits.grad[0, 1] < 0 and logits.grad[0, 2] == 0
    assert (logits.grad[1] > 0).all()  # every observed pass must continue, including last
    _, numeric = prefix_losses(logits.detach(), predicted + .5, depths, 12)
    numeric.backward()
    assert predicted.grad[0, 3] == 0 and (predicted.grad[1] > 0).all()
    # Fully observed requests keep the historical objective's exact semantics.
    scores = torch.randn(2, 4)
    depths = torch.tensor([1, 4])
    numbers = torch.randn(2, 5)
    stop, numeric = prefix_losses(scores, numbers, depths, 12)
    torch.testing.assert_close(stop, completion_loss(scores, torch.arange(4)[None, :] < depths[:, None])[0])
    torch.testing.assert_close(numeric, remaining_loss(numbers, depths, 12))


def test_prompt_features_match_actual_initializer_without_running_r(setup):
    model, tokenizer, tokens, _ = source_checkpoint(setup)
    tasks = [generate_unconditioned_example(47, d, 'train', i) for i, d in enumerate([1, 12, 64])]
    native, _, _ = extract_features(model, tokenizer, list(tokens.values()), tasks, 64, 2)
    calls = []
    hook = model.recurrent.register_forward_hook(lambda *args: calls.append(1))
    try:
        features = prompt_features(model, tokenizer, list(tokens.values()), tasks, 2)
    finally:
        hook.remove()
    assert not calls and not features.requires_grad
    torch.testing.assert_close(features, native['context'], atol=0, rtol=0)
    prefixes = pair_prefixes(tasks, native, [tasks[2], tasks[0]], 3)
    torch.testing.assert_close(prefixes['working'][0], native['working'][0, :3], atol=1e-6, rtol=1e-5)
    damaged = {**native, 'working': native['working'].clone()}
    damaged['working'][1, 0, 0] += 1
    with pytest.raises(AssertionError):
        pair_prefixes(tasks, damaged, tasks, 3)


def test_prepared_cache_prefix_training_and_selection_are_portable(setup):
    _, tokenizer, tokens, path = source_checkpoint(setup)
    dataset = DatasetConfig(seed=61, graph_mode='mixture', paired_horizons=True,
        train_count=8, validation_count=6, test_count=2, depth_test_count=2, max_train_depth=2, max_eval_depth=3)
    write_dataset(path / 'data', generate_dataset(dataset, tokenizer, tokens), dataset, tokens, {})
    config = ControllerConfig(source=str(path / 'source'), data=str(path / 'data'), device='cpu',
        train_graphs=3, validation_graphs=2, extraction_batch_size=2, batch_size=4, steps=4,
        warmup_steps=1, eval_every=2, log_every=1, evaluation_max_depth=3)
    (path / 'base.json').write_text(json.dumps(config.to_dict()))
    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', module, *map(str, args)],
            env={**os.environ, 'WANDB_MODE': 'disabled'}, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    run('scripts.training.train_controller', '--config', path / 'base.json', '--output', path / 'baseline')
    prefix = replace(config, remaining_readout=True, remaining_loss_weight=1, initial_loss_weight=12,
                     requested_counts=[1, 2, 4, 6], training_loops=2)
    (path / 'prefix.json').write_text(json.dumps(prefix.to_dict()))
    run('scripts.training.prepare_controller_prefix', '--config', path / 'prefix.json',
        '--features-cache', path / 'baseline/features.pt', '--output', path / 'features.pt', '--dry-run')
    assert not (path / 'features.pt').exists()
    run('scripts.training.prepare_controller_prefix', '--config', path / 'prefix.json',
        '--features-cache', path / 'baseline/features.pt', '--output', path / 'features.pt')
    cache = torch.load(path / 'features.pt', weights_only=True)
    assert cache['features']['train']['working'].shape[:2] == (12, 2)
    assert cache['extraction']['native_prefix_max_error'] < 2e-5
    run('scripts.training.train_controller', '--config', path / 'prefix.json',
        '--features-cache', path / 'features.pt', '--output', path / 'prefix-run')
    metadata = json.loads((path / 'prefix-run/run.json').read_text())
    assert metadata['trained_depths'] == [1, 2, 4, 6] and metadata['training_loops'] == 2
    summary = json.loads((path / 'prefix-run/summary.json').read_text())
    assert summary['validation']['selection']['questions'] == 4
    source = torch.load(path / 'source/adapter_model.pt', weights_only=True)
    exported = torch.load(path / 'prefix-run/best/adapter_model.pt', weights_only=True)
    for name in source:
        if not name.startswith('completion_head.'):
            torch.testing.assert_close(source[name], exported[name], atol=0, rtol=0)
    run('scripts.eval.loop_test', '--model', path / 'prefix-run/best', '--data', path / 'data/validation.jsonl',
        '--device', 'cpu', '--loops', 6, '--stop-policy', 'completion', '--stop-threshold', .5,
        '--output', path / 'native')

"""A learned internal counter is not an externally supplied loop clock."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import torch
import pytest

from test_executor_upgrade import setup
from test_controller_training import source_checkpoint
from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset
from scripts.eval.controller_features import replay
from scripts.recurrent_qwen.interfaces import AffineSuffixController
from scripts.training.controller import ControllerConfig
from scripts.training.controller_cache import fit_suffix_initializer
from scripts.training.controller_remaining import prefix_losses
from scripts.training.repeat_affine_controller import acceptance


def test_repeat_gate_checks_every_count_and_actual_stop_not_letters():
    def result():
        return {'status': 'complete', 'threshold': .5,
                'native_checks': [{'passed': True, 'questions': 20}, {'passed': True, 'questions': 24}],
                'metrics': {'best': {'all': {f'depth_{n}': {'questions': 32, 'exact_stop': 1.,
                    'stopped_answer_correct': 1.} for n in range(1, 65)}}}}
    results = {seed: result() for seed in (83, 89, 97)}
    assert acceptance(results)['passed']
    results[89]['metrics']['best']['all']['depth_64']['exact_stop'] = 0
    assert not acceptance(results)['passed']  # A correct cyclic final letter is insufficient.
    del results[97]['metrics']['best']['all']['depth_64']
    with pytest.raises(ValueError):
        acceptance(results)


def test_affine_update_is_learned_and_stop_gradients_are_isolated():
    head = AffineSuffixController(3)
    memory = torch.tensor([[2.]])
    _, following = head.advance(torch.randn(1, 3), memory)
    torch.testing.assert_close(following, memory)  # No -1 update is installed at construction.
    features = torch.randn(2, 24)
    working = torch.randn(2, 3, 3, requires_grad=True)
    logits, initial, states = replay(head, features, working)
    stop, numeric = prefix_losses(logits, torch.cat((initial[:, None], states), 1).squeeze(-1), torch.tensor([2, 3]), 12)
    stop.backward()
    assert head.cell.weight.grad is None and head.context.weight.grad is None
    assert head.readout.weight.grad is not None and working.grad is None
    numeric.backward()
    assert head.cell.bias.grad is not None and head.context.weight.grad is not None
    assert working.grad is None


def test_linear_initializer_uses_training_labels_and_learns_additivity():
    head = AffineSuffixController(1)
    # Synthetic frozen token features encode independent tens/ones categories.
    context = torch.zeros(5, 8)
    counts = [1, 2, 11, 12, 21]
    for i, n in enumerate(counts):
        context[i, n % 10] = 1
        context[i, 3 + n // 10] = 1
    metadata = fit_suffix_initializer(head, context.repeat(3, 1), counts)
    assert metadata['training_max_absolute_error'] < 1e-5
    unseen = torch.zeros(1, 8); unseen[0, 2] = 1; unseen[0, 5] = 1
    torch.testing.assert_close(head.initialize(unseen), torch.tensor([[22.]]), atol=1e-5, rtol=1e-5)


def test_affine_cache_training_export_and_actual_stopping(setup):
    _, tokenizer, tokens, path = source_checkpoint(setup)
    data = DatasetConfig(seed=61, graph_mode='mixture', paired_horizons=True,
        train_count=8, validation_count=6, test_count=2, depth_test_count=2, max_train_depth=2, max_eval_depth=3)
    write_dataset(path / 'data', generate_dataset(data, tokenizer, tokens), data, tokens, {})
    config = ControllerConfig(source=str(path / 'source'), data=str(path / 'data'), device='cpu',
        train_graphs=3, validation_graphs=2, extraction_batch_size=2, batch_size=4, steps=8,
        warmup_steps=1, eval_every=4, log_every=1, evaluation_max_depth=3)
    (path / 'base.json').write_text(json.dumps(config.to_dict()))
    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', module, *map(str, args)],
            env={**os.environ, 'WANDB_MODE': 'disabled'}, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    run('scripts.training.train_controller', '--config', path / 'base.json', '--output', path / 'base-run')
    affine = replace(config, requested_counts=[1, 2, 4, 6], training_loops=2, remaining_readout=True,
        remaining_loss_weight=1, controller_kind='affine_suffix', learning_rate=.01)
    (path / 'affine.json').write_text(json.dumps(affine.to_dict()))
    run('scripts.training.prepare_controller_prefix', '--config', path / 'affine.json',
        '--features-cache', path / 'base-run/features.pt', '--output', path / 'affine-features.pt')
    run('scripts.training.train_controller', '--config', path / 'affine.json',
        '--features-cache', path / 'affine-features.pt', '--output', path / 'affine-run')
    exported = torch.load(path / 'affine-run/best/adapter_model.pt', weights_only=True)
    source = torch.load(path / 'source/adapter_model.pt', weights_only=True)
    for name in source:
        if not name.startswith('completion_head.'):
            torch.testing.assert_close(exported[name], source[name], atol=0, rtol=0)
    spec = json.loads((path / 'affine-run/best/recurrent_config.json').read_text())
    assert spec['executor']['controller_kind'] == 'affine_suffix'
    run('scripts.eval.loop_test', '--model', path / 'affine-run/best', '--data', path / 'data/validation.jsonl',
        '--device', 'cpu', '--loops', 6, '--stop-policy', 'completion', '--stop-threshold', .5,
        '--output', path / 'native')
    from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
    from scripts.eval.controller_features import extract_features
    from scripts.eval.pointer_task import load_examples
    from scripts.eval.controller_learning import stop_rows
    from scripts.training.controller import evaluate_controller
    model, tokenizer, loaded_spec = load_recurrent_checkpoint(path / 'affine-run/best')
    tasks = load_examples(path / 'data/validation.jsonl')
    features, head, _ = extract_features(model, tokenizer, loaded_spec['token_ids'], tasks, 6, 2)
    metrics, decisions = evaluate_controller(head, features, tasks, affine.requested_counts, 'cpu')
    import csv
    native = list(csv.DictReader((path / 'native/decisions.csv').open()))
    for actual, cached in zip(native, decisions, strict=True):
        assert int(actual['executed_loops']) == cached['executed_loops']
        assert actual['exact_stop'] == str(cached['exact_stop'])

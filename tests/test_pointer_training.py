"""Stage 1 scientific contracts, using only random tiny Qwen models."""

from dataclasses import replace
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from scripts.dataset.pointer import SYMBOLS, generate_example
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint, restore_adapters, save_checkpoint
from scripts.training.config import TrainingConfig, read_config
from scripts.training.data import EncodedExample, collate, select_tasks
from scripts.training.evaluation import evaluate
from scripts.training.gates import initialize
from scripts.training.objective import batch_metrics, combine_metrics, step_loss, symbolic_scores
from scripts.training.runner import resume_identity, train
from scripts.eval.recurrent_pointer import evaluate_recurrent_batches

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_training(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.set_num_threads(1)
    torch.manual_seed(17)
    vocabulary = {symbol: index for index, symbol in enumerate(['[PAD]', '[EOS]', '[UNK]', *SYMBOLS])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=1024, tie_word_embeddings=True, pad_token_id=0, eos_token_id=1,
        attn_implementation='eager',
    )).eval()
    base_path = tmp_path / 'base'
    base.save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)
    config = replace(TrainingConfig(), model=str(base_path), revision='main', recurrent_start=1,
                     recurrent_end=2, lora_rank=2, lora_alpha=4, train_max_depth=2,
                     validation_max_depth=3, epochs=2, gradient_accumulation=2,
                     eval_every=2, save_every=1, warmup_steps=0, learning_rate=.001)
    items = []
    for index in range(5):
        task = generate_example(17 + index, 1 + index % 2, 'train', index)
        items.append(EncodedExample(task, tokenizer.encode(task.prompt), [SYMBOLS.index(s) for s in task.intermediate_states]))
    token_ids = list(range(3, 29))
    spec = {'base_model': str(base_path), 'revision': 'main', 'recurrent_start': 1, 'recurrent_end': 2,
            'lora_rank': 2, 'lora_alpha': 4, 'symbols': list(SYMBOLS), 'token_ids': token_ids,
            'prompt_format': 'dataset_raw', 'loss_vocabulary': 'symbols', 'train_max_depth': 2}
    model, gate = initialize(base, config, collate(items[:1], 0, 'cpu'), token_ids)
    return model, tokenizer, config, items, spec, gate


def test_loss_exact_weighting_mask_and_accumulation():
    scores = torch.tensor([[[3., 0.], [100., -100.]], [[0., 1.], [2., 0.]], [[1., 0.], [0., 3.]]], requires_grad=True)
    targets = torch.tensor([[0, 1], [1, 1], [0, 1]])
    mask = torch.tensor([[True, False], [True, True], [True, True]])
    loss, losses = step_loss(scores, targets, mask)
    ce = torch.nn.functional.cross_entropy(scores.flatten(0, 1), targets.flatten(), reduction='none').reshape(3, 2)
    expected = (ce[0, 0] + ce[1].mean() + ce[2].mean()) / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    reference = scores.grad.clone()
    assert torch.count_nonzero(reference[0, 1]) == 0
    scores.grad = None
    parts = []
    for lo, hi in [(0, 2), (2, 3)]:
        subloss, sublosses = step_loss(scores[lo:hi], targets[lo:hi], mask[lo:hi])
        (subloss * (hi - lo) / 3).backward()
        parts.append(batch_metrics(scores[lo:hi], targets[lo:hi], mask[lo:hi], sublosses))
    torch.testing.assert_close(scores.grad, reference)
    metrics = combine_metrics(parts)
    assert metrics['loss'] == pytest.approx(loss.item())
    assert metrics['intermediate_accuracy'] == 4 / 5
    assert metrics['trajectory_accuracy'] == 2 / 3
    assert metrics['per_loop']['2']['count'] == 2
    with pytest.raises(FloatingPointError):
        step_loss(scores.detach() * float('nan'), targets, mask)


def test_config_and_subset_are_explicit_and_reproducible():
    config = read_config(ROOT / 'configs/stage1_pointer_overfit.json')
    assert config.train_limit == 32 and config.train_max_depth == 4
    tasks = [generate_example(i + 1, d, 'train', i) for i in range(12) for d in (1, 2, 3)]
    a = select_tasks(tasks, 3, 17, limit=7)
    assert a == select_tasks(tasks, 3, 17, limit=7)
    assert a != select_tasks(tasks, 3, 18, limit=7)
    assert [task.task_depth for task in a] == [1, 2, 3, 1, 2, 3, 1]
    for bad in (replace(config, loss_vocabulary='all'), replace(config, gradient_accumulation=0)):
        with pytest.raises(ValueError):
            bad.validate()


def test_gates_checkpoint_roundtrip_and_gradient_scope(tiny_training, tmp_path):
    model, tokenizer, config, items, spec, gate = tiny_training
    assert gate['passed'] and all(norm > 0 for norm in gate['hidden_gradient_norms'])
    frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    batch = collate(items[:2], 0, 'cpu')
    result = model(batch['input_ids'], batch['attention_mask'], num_loops=2)
    loss, _ = step_loss(symbolic_scores(result.loop_logits, spec['token_ids']), batch['targets'], batch['target_mask'])
    loss.backward()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    optimizer.step()
    assert all(p.grad is None and torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    path = tmp_path / 'checkpoint'
    save_checkpoint(path, model, tokenizer, spec, {})
    loaded, saved_tokenizer, saved_spec = load_recurrent_checkpoint(path)
    with torch.no_grad():
        reference = model(batch['input_ids'], batch['attention_mask'], num_loops=3)
        restored = loaded(batch['input_ids'], batch['attention_mask'], num_loops=3)
    for first, second in zip(reference.loop_logits, restored.loop_logits):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert saved_spec['token_ids'] == spec['token_ids']
    rows, _ = next(evaluate_recurrent_batches(loaded, saved_tokenizer, [item.task for item in items[:2]], saved_spec, batch_size=2))
    for row, logits in zip(rows, [restored.loop_logits[0][0], restored.loop_logits[1][1]]):
        assert row['prediction'] == SYMBOLS[int(logits[spec['token_ids']].argmax())]
        assert row['generated_tokens'] == 0
    assert [row['readout_loop'] for row in rows] == [1, 2]
    assert [row['executed_loops'] for row in rows] == [2, 2]


def test_nominal_metrics_exclude_post_completion(tiny_training, tmp_path):
    model, tokenizer, _, items, spec, _ = tiny_training
    output = tmp_path / 'trajectory.csv'
    summary = evaluate(model, items[:2], spec['token_ids'], tokenizer.pad_token_id, loops=3, output=output)
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 6
    assert [summary['per_loop'][str(t)]['count'] for t in (1, 2)] == [2, 1]
    assert '3' not in summary['per_loop']
    assert sum(row['post_completion'] == 'True' for row in rows) == 3
    assert all(row['intermediate_target'] == '' for row in rows if row['post_completion'] == 'True')
    nominal = [row for row in rows if row['loop'] == row['task_depth']]
    assert summary['final_accuracy'] == sum(row['final_correct'] == 'True' for row in nominal) / 2
    assert summary['depth_by_loop']['1']['3']['count'] == 1


def test_resume_matches_uninterrupted_updates_including_short_tail(tiny_training, tmp_path):
    model, tokenizer, config, items, spec, _ = tiny_training
    initial = tmp_path / 'initial'
    save_checkpoint(initial, model, tokenizer, spec, {})
    identity = resume_identity(config, {'data': 'test'})
    full = tmp_path / 'full'
    full.mkdir()
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    result = train(model, tokenizer, items, items[:2], items[:2], config, full, spec, identity)
    assert result['step'] == 6
    expected = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    split_model, _, _ = load_recurrent_checkpoint(initial)
    first = tmp_path / 'first'
    first.mkdir()
    train(split_model, tokenizer, items, items[:2], items[:2], replace(config, max_steps=2), first, spec, identity)
    saved = first / 'step-000002'
    resume = torch.load(saved / 'training_state.pt', weights_only=True)
    assert resume['next_offset'] == 4 and resume['next_epoch'] == 0
    resumed_model, _, _ = load_recurrent_checkpoint(saved)
    second = tmp_path / 'second'
    second.mkdir()
    resumed = train(resumed_model, tokenizer, items, items[:2], items[:2], config, second, spec, identity, resume=resume)
    for name, p in resumed_model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p, expected[name], rtol=0, atol=0)
    assert resumed['validation'] == result['validation']
    events = [json.loads(line) for line in (full / 'metrics.jsonl').read_text().splitlines()]
    assert [event['train']['examples'] for event in events if event['event'] == 'train'] == [2, 2, 1, 2, 2, 1]
    assert json.loads((full / 'last_checkpoint.json').read_text())['step'] == 6


def test_saved_recurrent_checkpoint_runs_through_naive_cli(tiny_training, tmp_path):
    model, tokenizer, _, items, spec, _ = tiny_training
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(checkpoint, model, tokenizer, spec, {})
    data = tmp_path / 'test.jsonl'
    data.write_text(''.join(json.dumps(generate_example(17 + i, i + 1, 'test', i).to_dict()) + '\n' for i in range(3)))
    output = tmp_path / 'results'
    command = [sys.executable, '-m', 'scripts.eval.naive_test', '--model', str(checkpoint),
               '--data', str(data), '--output', str(output), '--test', '--batch-size', '2']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'recurrent checkpoint' in result.stdout and 'readout:' in result.stdout
    summary = json.loads((output / 'summary.json').read_text())
    rows = list(csv.DictReader((output / 'predictions.csv').open()))
    assert summary['total'] == 3 and summary['generated_tokens'] == 0
    assert summary['scoring'] == 'allowed_symbol_argmax_at_requested_depth'
    assert [int(row['readout_loop']) for row in rows] == [1, 2, 3]
    assert summary['correct'] == sum(row['correct'] == 'True' for row in rows)
    rejected = subprocess.run([*command, '--prompt-format', 'chat', '--output', str(tmp_path / 'reject')], cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'Recurrent checkpoints fix' in rejected.stderr


def test_training_cli_dry_run_and_toy_update(tmp_path):
    """Real cached tokenizer + random weights; exercise the complete CLI wiring."""
    import hashlib
    from transformers import AutoTokenizer
    from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip('Pinned tokenizer must be cached for the CLI integration check')
    torch.manual_seed(17)
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=256, tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        attn_implementation='eager',
    ))
    base_path = tmp_path / 'base'
    base.save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)
    manifest = {'splits': {}}
    for split in ('train', 'validation'):
        content = ''.join(json.dumps(generate_example((17 if split == 'train' else 117) + i, 1 + i % 2, split, i).to_dict()) + '\n' for i in range(4))
        (tmp_path / f'{split}.jsonl').write_text(content)
        manifest['splits'][split] = {'sha256': hashlib.sha256(content.encode()).hexdigest(), 'count': 4}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    config = replace(TrainingConfig(), model=str(base_path), revision='main',
                     train_data=str(tmp_path / 'train.jsonl'), validation_data=str(tmp_path / 'validation.jsonl'),
                     recurrent_start=1, recurrent_end=2, lora_rank=2, lora_alpha=4,
                     train_max_depth=2, validation_max_depth=2, validation_per_depth=1,
                     train_probe_per_depth=1, gradient_accumulation=2, max_steps=1)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / 'run'
    command = [sys.executable, '-m', 'scripts.training.train_pointer', '--config', str(config_path), '--output', str(output)]
    preview = subprocess.run([*command, '--dry-run'], cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'Dry run passed' in preview.stdout and not output.exists()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'Run complete' in result.stdout and 'Rules:' not in result.stdout
    metadata = json.loads((output / 'run.json').read_text())
    assert metadata['status'] == 'complete' and metadata['gate']['passed']
    state = torch.load(output / 'step-000001/training_state.pt', weights_only=True)
    assert state['step'] == 1 and state['next_offset'] == 2
    config_path.write_text(json.dumps(replace(config, max_steps=2).to_dict()))
    continuation = tmp_path / 'resumed'
    subprocess.run([*command, '--output', str(continuation), '--resume', str(output / 'step-000001')], cwd=ROOT, capture_output=True, text=True, check=True)
    summary = json.loads((continuation / 'summary.json').read_text())
    assert summary['step'] == 2 and summary['next_epoch'] == 1

    # A new depth stage loads adapters, but must not reuse optimizer/cursor state.
    source = continuation / 'step-000002'
    (source / 'training_state.pt').unlink()
    for split in ('train', 'validation'):
        content = ''.join(json.dumps(generate_example((217 if split == 'train' else 317) + i, 1 + i % 3, split, i).to_dict()) + '\n' for i in range(6))
        (tmp_path / f'{split}.jsonl').write_text(content)
        manifest['splits'][split] = {'sha256': hashlib.sha256(content.encode()).hexdigest(), 'count': 6}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    deeper_config = replace(config, train_max_depth=3, validation_max_depth=3, max_steps=1)
    config_path.write_text(json.dumps(deeper_config.to_dict()))
    deeper = tmp_path / 'deeper'
    warm_command = [*command, '--output', str(deeper), '--init-from', str(source)]
    subprocess.run([*warm_command, '--dry-run'], cwd=ROOT, capture_output=True, text=True, check=True)
    assert not deeper.exists()
    subprocess.run(warm_command, cwd=ROOT, capture_output=True, text=True, check=True)
    initial = torch.load(deeper / 'step-000000/adapter_model.pt', weights_only=True)
    expected = torch.load(source / 'adapter_model.pt', weights_only=True)
    assert all(torch.equal(initial[k], expected[k]) for k in expected)
    state = torch.load(deeper / 'step-000001/training_state.pt', weights_only=True)
    assert state['step'] == 1 and state['next_offset'] == 2
    saved_spec = json.loads((deeper / 'step-000001/recurrent_config.json').read_text())
    assert saved_spec['train_max_depth'] == 3 and saved_spec['initialization']['source_train_max_depth'] == 2
    config_path.write_text(json.dumps(replace(deeper_config, max_steps=2).to_dict()))
    subprocess.run([*command, '--output', str(tmp_path / 'deeper_resume'), '--resume', str(deeper / 'step-000001')],
                   cwd=ROOT, capture_output=True, text=True, check=True)

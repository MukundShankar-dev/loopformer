"""Architecture/data contracts for the isolated executor, without pretrained weights."""
from copy import deepcopy
from dataclasses import replace
import json
import re

import pytest
import torch
from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import Fuse
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast

from scripts.dataset.pointer import SYMBOLS, generate_unconditioned_example, validate_example, render_prompt
from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset, verify_dataset
from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.recurrent_qwen.checkpoint import save_checkpoint, load_recurrent_checkpoint
from scripts.training.config import TrainingConfig
from scripts.training.data import encode_tasks, collate
from scripts.training.objective import completion_loss, step_loss
from scripts.training.executor_gates import validate_executor


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    pattern = r" ?[A-Z]|[a-z]+|[0-9]|\n|."
    corpus = [render_prompt([[s, s] for s in SYMBOLS], 'A', 1234567890), *[' ' + s for s in SYMBOLS]]
    vocab = ['[PAD]', '[UNK]', '[EOS]', *sorted(set(t for text in corpus for t in re.findall(pattern, text)))]
    backend = Tokenizer(WordLevel({v: i for i, v in enumerate(vocab)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Split(Regex(pattern), behavior='isolated')
    backend.decoder = Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024,
        pad_token_id=0, eos_token_id=2, attn_implementation='eager')).eval()
    base.save_pretrained(tmp_path / 'base')
    tokenizer.save_pretrained(tmp_path / 'base')
    tasks = [generate_unconditioned_example(27, d, 'train', i) for i, d in enumerate([2, 12, 128])]
    tokens = {s: tokenizer.encode(' ' + s, add_special_tokens=False)[0] for s in SYMBOLS}
    items = encode_tasks(tasks, tokenizer, tokens, 512)
    return base, tokenizer, tokens, items, tmp_path


def configured(base, tokenizer, **kwargs):
    model = RecurrentQwen(deepcopy(base), 1, 2, recurrence_mode='fixed_prompt')
    model.configure_executor(tokenizer, **kwargs)
    return model.eval()


def test_count_invariance_and_gradient_isolation(setup):
    base, tokenizer, tokens, items, _ = setup
    model = configured(base, tokenizer)
    batch = collate(items, 0, 'cpu')
    result = model(batch['input_ids'], batch['attention_mask'], num_loops=3, readout_token_ids=list(tokens.values()))
    for scores in result.loop_logits:
        torch.testing.assert_close(scores[0], scores[1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(scores[0], scores[2], atol=1e-6, rtol=1e-5)
    assert result.stop_logits.shape == (3, 3)
    assert not torch.equal(result.stop_logits[0], result.stop_logits[2])
    assert validate_executor(model, collate(items[:1], 0, 'cpu'), list(tokens.values()))['passed']


@pytest.mark.parametrize('scope', ['lora', 'recurrent', 'full'])
@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
def test_prefix_forward_gradient_equivalence(setup, scope, attention):
    base, tokenizer, tokens, items, _ = setup
    base.set_attn_implementation(attention)
    dense = RecurrentQwen(deepcopy(base), 1, 2, recurrence_mode='fixed_prompt')
    if scope == 'lora':
        from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
        attach_recurrent_lora(dense, rank=2, alpha=4)
    dense.configure_executor(tokenizer, train_scope=scope)
    dense.eval()
    fast = deepcopy(dense)
    fast.prefix_reuse = True
    for block in (fast.prelude, fast.recurrent, fast.coda):
        block.prefix_reuse = True
        block.gradient_checkpointing = True
    batch = collate(items[:2], 0, 'cpu')
    outputs = []
    for model in (dense, fast):
        output = model(batch['input_ids'], batch['attention_mask'], num_loops=3,
                       readout_token_ids=list(tokens.values()))
        loss = sum(x.square().mean() for x in output.loop_logits) + output.state_logits.square().mean() + output.stop_logits.square().mean()
        loss.backward()
        outputs.append(output)
    for a, b in zip(outputs[0].loop_logits, outputs[1].loop_logits):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for (name, a), (_, b) in zip(dense.named_parameters(), fast.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=lambda x: name + ': ' + x)
        else:
            assert b.grad is None


def test_selected_projection_and_bridge_first_pass(setup):
    base, tokenizer, tokens, items, _ = setup
    model = configured(base, tokenizer, isolated=False, controller_size=0)
    batch = collate(items[:1], 0, 'cpu')
    with torch.no_grad():
        full = model(batch['input_ids'], batch['attention_mask']).logits
        selected = model(batch['input_ids'], batch['attention_mask'], readout_token_ids=list(tokens.values())).logits
        reference = base(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits[:, -1]
    torch.testing.assert_close(full, reference, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(selected, full[:, list(tokens.values())])


@pytest.mark.parametrize('scope', ['recurrent', 'full'])
def test_checkpoint_roundtrip(setup, scope):
    base, tokenizer, tokens, items, path = setup
    options = dict(train_scope=scope, bridge=True, direct_readout=True, controller_size=8,
                   isolated=True, prefix_reuse=True, gradient_checkpointing=True)
    model = configured(base, tokenizer, **options)
    spec = dict(base_model=str(path / 'base'), revision='main', recurrent_start=1, recurrent_end=2,
        recurrence_mode='fixed_prompt', executor=options, completion_head={'intermediate': 8},
        symbols=list(SYMBOLS), token_ids=list(tokens.values()), prompt_format='dataset_raw', loss_vocabulary='symbols')
    save_checkpoint(path / 'checkpoint', model, tokenizer, spec, {})
    loaded, _, saved = load_recurrent_checkpoint(path / 'checkpoint')
    assert saved['format'] == 'loopformer-executor-v2'
    batch = collate(items[:1], 0, 'cpu')
    with torch.no_grad():
        a = model(batch['input_ids'], batch['attention_mask'], num_loops=3)
        b = loaded(batch['input_ids'], batch['attention_mask'], num_loops=3)
    for x, y in zip(a.loop_logits, b.loop_logits):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    torch.testing.assert_close(a.stop_logits, b.stop_logits, atol=0, rtol=0)


def test_unconditioned_seed_replay_and_pairs(setup):
    _, tokenizer, tokens, _, path = setup
    config = DatasetConfig(seed=61, graph_mode='mixture', paired_horizons=True,
        train_count=6, validation_count=6, test_count=6, depth_test_count=8, max_train_depth=3, max_eval_depth=7)
    data = generate_dataset(config, tokenizer, tokens)
    for split, records in data.items():
        width = len(config.depths(split))
        assert len({r['mapping_sha256'] for r in records[:width]}) == (width if split == 'train' else 1)
        for r in records:
            assert r['schema_version'] == 2
    write_dataset(path / 'data', data, config, tokens, {})
    verify_dataset(path / 'data', tokenizer)
    a = generate_unconditioned_example(19, 2, 'train', 0)
    b = generate_unconditioned_example(19, 128, 'train', 1)
    assert a.mapping == b.mapping and a.initial_state == b.initial_state
    assert a.intermediate_states == b.intermediate_states[:2]
    validate_example(b)


def test_new_training_eval_cli_and_diagnostics(setup):
    import subprocess
    import sys
    base, tokenizer, tokens, _, path = setup
    from transformers import AutoTokenizer
    from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION, validate_symbols
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("CLI integration needs the cached pinned tokenizer")
    tokens = validate_symbols(tokenizer)
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.eos_token_id = tokenizer.eos_token_id
    base.save_pretrained(path / 'base')
    tokenizer.save_pretrained(path / 'base')
    dataset_config = DatasetConfig(seed=61, graph_mode='mixture', paired_horizons=True,
        train_count=4, validation_count=4, test_count=4, depth_test_count=4,
        max_train_depth=2, max_eval_depth=4)
    dataset = generate_dataset(dataset_config, tokenizer, tokens)
    write_dataset(path / 'data', dataset, dataset_config, tokens, {})
    config = replace(TrainingConfig(), model=str(path / 'base'), revision='main',
        train_data=str(path / 'data/train.jsonl'), validation_data=str(path / 'data/validation.jsonl'),
        recurrent_start=1, recurrent_end=2, recurrence_mode='fixed_prompt', train_scope='recurrent',
        isolated_controller=True, reentry_bridge=True, state_loss_weight=.25,
        prefix_reuse=True, gradient_checkpointing=True, completion_loss_weight=.1,
        completion_head_hidden_size=8, max_prompt_tokens=512, train_max_depth=2,
        validation_max_depth=2, validation_per_depth=1, train_probe_per_depth=1,
        batch_size=2, gradient_accumulation=1, max_steps=2, eval_every=1, save_every=1,
        warmup_steps=0, lr_schedule='cosine', schedule_steps=2, threads=1)
    (path / 'config.json').write_text(json.dumps(config.to_dict()))
    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', module, *map(str, args)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    run('scripts.training.train_pointer', '--config', path / 'config.json', '--output', path / 'run', '--wandb-mode', 'disabled')
    checkpoint = path / 'run/step-000002'
    metrics = [json.loads(line) for line in (path / 'run/metrics.jsonl').read_text().splitlines()]
    event = next(e for e in metrics if e['event'] == 'train')
    assert event['train']['direct_state']['examples'] == 2
    assert set(event['gradient_norm_by_module']) >= {'recurrent', 'completion_head', 'bridge', 'state_head'}
    run('scripts.eval.loop_test', '--model', checkpoint, '--data', path / 'data/validation.jsonl',
        '--output', path / 'eval', '--batch-size', 2, '--loops', 4, '--record-states', '--wandb-mode', 'disabled')
    run('scripts.eval.loop_test', '--model', checkpoint, '--data', path / 'data/validation.jsonl',
        '--output', path / 'stopped', '--loops', 4, '--stop-policy', 'completion', '--stop-threshold', .5, '--wandb-mode', 'disabled')
    run('scripts.eval.executor_diagnostic', '--model', checkpoint, '--data', path / 'data/validation.jsonl',
        '--output', path / 'diagnostic', '--device', 'cpu', '--count', 1, '--depths', 1, 2, 4, '--wandb-mode', 'disabled')
    summary = json.loads((path / 'diagnostic/summary.json').read_text())
    assert summary['status'] == 'complete'
    run('scripts.training.native_control', '--data', path / 'data', '--output', path / 'native',
        '--device', 'cpu', '--steps', 2, '--batch-size', 2, '--width', 16,
        '--train-max-depth', 2, '--eval-max-depth', 4, '--wandb-mode', 'disabled')
    import csv
    assert all(int(row['common_prefix_disagreements']) == 0 for row in csv.DictReader((path / 'diagnostic/pairs.csv').open()))


def test_native_control_has_learned_autonomous_transitions(setup):
    from scripts.training.native_control import NativeExecutor
    from scripts.training.objective import forward_symbols
    from scripts.training.evaluation import evaluate
    _, tokenizer, tokens, items, _ = setup
    model = NativeExecutor(tokenizer, 16)
    batch = collate(items[:1], 0, 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
    losses = []
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        _, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'], list(tokens.values()), num_loops=2)
        loss, _ = step_loss(scores, batch['targets'], batch['target_mask'])
        loss.backward()
        assert model.query.weight.grad is not None and model.query.weight.grad.abs().sum() > 0
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2
    metrics = evaluate(model, items[:1], list(tokens.values()), 0)
    assert metrics['trajectory_accuracy'] == 1


def test_controller_loss_cannot_change_executor_optimizer_update(setup):
    from scripts.training.runner import clip_training_gradients
    base, tokenizer, tokens, items, _ = setup
    first = configured(base, tokenizer)
    second = deepcopy(first)
    second.completion_head.readout.bias.data.add_(10)
    batch = collate(items[:1], 0, 'cpu')
    for model, weight in ((first, .1), (second, 10000.)):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        result = model(batch['input_ids'], batch['attention_mask'], num_loops=2, readout_token_ids=list(tokens.values()))
        scores = torch.stack(result.loop_logits, dim=1)
        pointer, _ = step_loss(scores, batch['targets'], batch['target_mask'])
        halt, _ = completion_loss(result.stop_logits, batch['target_mask'])
        (pointer + weight * halt).backward()
        clip_training_gradients(model, .01, True)
        optimizer.step()
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        if not name.startswith('completion_head.'):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

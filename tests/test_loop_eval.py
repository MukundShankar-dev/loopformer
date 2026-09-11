"""Full-loop scoring contracts and offline saved-checkpoint CLI integration."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.dataset.pointer import SYMBOLS, generate_example
from scripts.eval.loop_metrics import depth_loop_rows, summarize_trajectories
from scripts.training.data import EncodedExample
from scripts.training.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_first_error_recovery_post_completion_and_margins(tmp_path):
    tasks = [generate_example(17, 1, 'test', 0), generate_example(18, 3, 'test', 1)]
    items = [EncodedExample(t, [i], [SYMBOLS.index(s) for s in t.intermediate_states]) for i, t in enumerate(tasks)]
    a, b = [item.targets for item in items]
    predictions = [[a[0], (a[0] + 1) % 26, a[0], a[0]], [b[0], b[0], b[2], b[2]]]

    class PrescribedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, num_loops):
            assert not self.training and not torch.is_grad_enabled()
            logits = []
            for loop in range(num_loops):
                scores = torch.zeros(len(input_ids), 26)
                for row, index in enumerate(input_ids[:, 0].tolist()):
                    scores[row, predictions[index][loop]] = 2
                logits.append(scores)
            return SimpleNamespace(loop_logits=tuple(logits))

    model = PrescribedModel()
    output = tmp_path / 'trajectories.csv'
    summary = evaluate(model, items, list(range(26)), 0, loops=4, batch_size=2, output=output)
    assert model.training and model.weight.grad is None
    assert summary['trajectory_accuracy'] == .5
    assert summary['final_accuracy'] == 1
    assert summary['intermediate_accuracy'] == .75
    assert [v['count'] for v in summary['per_loop'].values()] == [2, 1, 1]
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 8
    for row in rows:
        if row['post_completion'] == 'True':
            assert row['intermediate_margin'] == row['intermediate_loss'] == row['intermediate_target'] == ''
        else:
            assert float(row['intermediate_margin']) == (2 if row['intermediate_correct'] == 'True' else -2)
    examples, diagnostics = summarize_trajectories(output)
    assert [x['first_error_loop'] for x in examples] == ['', 2]
    assert [x['correct_prefix_length'] for x in examples] == [1, 1]
    assert [x['first_final_correct_loop'] for x in examples] == [1, 3]
    assert diagnostics['overall']['nominal_repeated_predictions'] == 1
    assert diagnostics['overall']['nominal_adjacent_pairs'] == 2
    assert diagnostics['overall']['first_error_counts'] == {'none': 1, '2': 1}
    matrix = depth_loop_rows(summary)
    assert len(matrix) == 8
    assert all(r['intermediate_accuracy'] == r['intermediate_loss'] == '' for r in matrix if r['post_completion'])
    with pytest.raises(ValueError, match='Loop budget'):
        evaluate(model, items, list(range(26)), 0, loops=2, batch_size=2)


def test_saved_checkpoint_loop_cli_matches_naive_readout(tmp_path):
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from scripts.recurrent_qwen.checkpoint import save_checkpoint
    from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
    from scripts.recurrent_qwen.model import RecurrentQwen

    torch.set_num_threads(1)
    torch.manual_seed(17)
    vocabulary = {s: i for i, s in enumerate(['[PAD]', '[EOS]', '[UNK]', *['Ġ' + s for s in SYMBOLS]])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=1, attn_implementation='eager',
    )).eval()
    base_path = tmp_path / 'base'
    base.save_pretrained(base_path)
    model = RecurrentQwen(base, 1, 2)
    attach_recurrent_lora(model, rank=2, alpha=4)
    spec = {'base_model': str(base_path), 'revision': 'main', 'recurrent_start': 1, 'recurrent_end': 2,
            'lora_rank': 2, 'lora_alpha': 4, 'symbols': list(SYMBOLS), 'token_ids': list(range(3, 29)),
            'prompt_format': 'dataset_raw', 'loss_vocabulary': 'symbols', 'train_max_depth': 4}
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(checkpoint, model, tokenizer, spec, {})
    # Inference must not depend on optimizer state.
    (checkpoint / 'training_state.pt').unlink()
    data = tmp_path / 'test.jsonl'
    tasks = [generate_example(17 + i, d, 'test', i) for i, d in enumerate((1, 2, 10))]
    data.write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in tasks))
    output = tmp_path / 'loops'
    command = [sys.executable, '-m', 'scripts.eval.loop_test', '--model', str(checkpoint),
               '--data', str(data), '--output', str(output), '--batch-size', '2', '--threads', '1', '--test']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'TEST MODE' in result.stdout and 'Per-loop readout' in result.stdout
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['loops'] == 10
    assert summary['examples'] == 3 and summary['executed_example_loops'] == 30
    assert summary['per_loop']['1']['count'] == 3 and summary['per_loop']['3']['count'] == 1
    trajectories = list(csv.DictReader((output / 'trajectories.csv').open()))
    assert len(trajectories) == 30
    assert len(list(csv.DictReader((output / 'examples.csv').open()))) == 3
    assert len(list(csv.DictReader((output / 'depth_by_loop.csv').open()))) == 30
    naive_output = tmp_path / 'naive'
    subprocess.run([sys.executable, '-m', 'scripts.eval.naive_test', '--model', str(checkpoint),
                    '--data', str(data), '--output', str(naive_output), '--batch-size', '2', '--threads', '1'],
                   cwd=ROOT, capture_output=True, text=True, check=True)
    naive = list(csv.DictReader((naive_output / 'predictions.csv').open()))
    nominal = [row for row in trajectories if row['loop'] == row['task_depth']]
    assert [row['prediction'] for row in nominal] == [row['prediction'] for row in naive]
    assert summary['final_accuracy'] == sum(row['correct'] == 'True' for row in naive) / 3
    rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'Output already exists' in rejected.stderr
    rejected = subprocess.run([*command, '--output', str(tmp_path / 'too_short'), '--loops', '2'], cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'deepest selected task' in rejected.stderr
    assert not (tmp_path / 'too_short').exists()

    # The terminal CLI uses the same model path and scoring, but persists changed
    # prompts and explicit dynamics. No optimizer state or pretrained downloads.
    terminal_output = tmp_path / 'terminal'
    terminal_command = [sys.executable, '-m', 'scripts.eval.overscaling_test', '--model', str(checkpoint),
                        '--data', str(data), '--output', str(terminal_output), '--loops', '11',
                        '--threads', '1', '--batch-size', '2', '--test']
    subprocess.run(terminal_command, cwd=ROOT, capture_output=True, text=True, check=True)
    terminal_summary = json.loads((terminal_output / 'summary.json').read_text())
    assert terminal_summary['status'] == 'complete'
    assert terminal_summary['task_variant'] == 'absorbing-terminal-v1'
    assert terminal_summary['executed_example_loops'] == 33
    assert len(list(csv.DictReader((terminal_output / 'transitions.csv').open()))) == 30
    terminal_data = terminal_output / 'tasks.jsonl'
    transformed = [json.loads(line) for line in terminal_data.read_text().splitlines()]
    assert all(dict(t['mapping'])[t['final_state']] == t['final_state'] for t in transformed)
    from scripts.eval.pointer_task import sha256_file
    assert sha256_file(terminal_data) == terminal_summary['transformed_tasks_sha256']
    terminal_naive = tmp_path / 'terminal_naive'
    subprocess.run([sys.executable, '-m', 'scripts.eval.naive_test', '--model', str(checkpoint),
                    '--data', str(terminal_data), '--output', str(terminal_naive), '--threads', '1'],
                   cwd=ROOT, capture_output=True, text=True, check=True)
    expected = list(csv.DictReader((terminal_naive / 'predictions.csv').open()))
    terminal_rows = list(csv.DictReader((terminal_output / 'trajectories.csv').open()))
    nominal = [r for r in terminal_rows if r['loop'] == r['task_depth']]
    assert [r['prediction'] for r in nominal] == [r['prediction'] for r in expected]
    rejected = subprocess.run([*terminal_command, '--output', str(tmp_path / 'no_extra'), '--loops', '10'],
                              cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'greater than' in rejected.stderr
    assert not (tmp_path / 'no_extra').exists()

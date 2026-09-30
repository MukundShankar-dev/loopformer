"""Risk-set, paired-intervention, and profiling contracts on tiny local models."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.dataset.pointer import SYMBOLS, generate_example
from scripts.eval.failure_diagnostics import analyze_failures
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_probes import task_slice, state_change
from scripts.eval.profiling import distribution, module_ranges, training_update
from scripts.training.data import EncodedExample

ROOT = Path(__file__).resolve().parents[1]


def test_failure_risk_sets_and_reference_validation(tmp_path):
    tasks = [generate_example(51+i, 4, 'validation', i) for i in range(5)]
    rows = []
    for i, task in enumerate(tasks):
        path = [task.initial_state, *task.intermediate_states]
        wrong = [path[1], path[0], path[3], next(s for s in SYMBOLS if s not in path), None][i]
        for t in range(1, 6):
            target = path[t] if t <= 4 else ''
            prediction = wrong if t == 2 and wrong else (target or path[-1])
            rows.append({'example_id': task.example_id, 'loop': t, 'task_depth': 4,
                         'initial_state': path[0], 'prediction': prediction, 'intermediate_target': target,
                         'intermediate_correct': prediction == target if target else '',
                         'final_target': path[-1], 'final_correct': prediction == path[-1]})
    file = tmp_path/'trace.csv'; write_csv(file, rows)
    errors, risk, summary = analyze_failures(file, tasks, 5)
    assert [r['category'] for r in errors] == ['repeat_previous', 'earlier_path', 'future_path', 'off_path']
    assert [r['correct_prefix_examples'] for r in risk] == [5, 5, 1, 1]
    assert risk[1]['conditional_failure_rate'] == .8 and summary['complete_trajectories'] == 1
    write_csv(file, rows[:20])
    _, empty_risk, _ = analyze_failures(file, tasks[:4], 5)
    assert empty_risk[2]['correct_prefix_examples'] == 0
    assert empty_risk[2]['conditional_failure_rate'] == ''
    # Recovering later does not re-enter the prefix risk set.
    rows[1]['intermediate_correct'] = True
    write_csv(file, rows)
    with pytest.raises(ValueError, match='mismatch'):
        analyze_failures(file, tasks, 5)
    write_csv(file, rows[:-1])
    with pytest.raises(ValueError):
        analyze_failures(file, tasks, 5)


def test_suffix_and_depth_cue_preserve_reference_transitions():
    task = generate_example(51, 8, 'validation', 0)
    original = task.to_dict()
    suffix = task_slice(task, 6, 8)
    assert suffix.initial_state == task.intermediate_states[5]
    assert suffix.intermediate_states == task.intermediate_states[6:]
    assert suffix.mapping == task.mapping and suffix.task_depth == 2
    shorter = task_slice(task, 0, 7)
    assert shorter.initial_state == task.initial_state
    assert shorter.intermediate_states == task.intermediate_states[:7]
    assert task.to_dict() == original
    with pytest.raises(ValueError):
        task_slice(task, 8, 8)
    values = state_change(torch.ones(2), 2*torch.ones(2))
    assert values['rms'] == 2 and values['relative_update_norm'] == 1
    assert values['cosine_to_previous'] == pytest.approx(1)
    assert state_change(torch.zeros(2), torch.ones(2))['relative_update_norm'] is None
    assert distribution([4., 1., 2., 3.])['median_seconds'] == 2.5


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from scripts.recurrent_qwen.checkpoint import save_checkpoint
    from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
    from scripts.recurrent_qwen.model import RecurrentQwen
    torch.set_num_threads(1); torch.manual_seed(17)
    vocab = {s: i for i, s in enumerate(['[PAD]', '[EOS]', '[UNK]', *['Ġ'+s for s in SYMBOLS]])}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False); backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=1, attn_implementation='eager')).eval()
    base_path = tmp_path/'base'; base.save_pretrained(base_path)
    model = RecurrentQwen(base, 1, 2); attach_recurrent_lora(model, rank=2, alpha=4)
    spec = {'base_model': str(base_path), 'revision': 'main', 'recurrent_start': 1, 'recurrent_end': 2,
            'lora_rank': 2, 'lora_alpha': 4, 'symbols': list(SYMBOLS), 'token_ids': list(range(3, 29)),
            'prompt_format': 'dataset_raw', 'loss_vocabulary': 'symbols', 'train_max_depth': 2}
    checkpoint = tmp_path/'checkpoint'; save_checkpoint(checkpoint, model, tokenizer, spec, {})
    return checkpoint, model, tokenizer, spec


def test_profiling_hooks_preserve_forward_and_training_gradient_scope(tiny_checkpoint):
    _, model, _, spec = tiny_checkpoint
    model.eval(); ids = torch.tensor([[3,4,5]])
    with torch.no_grad():
        expected = model(ids, num_loops=2).logits.clone()
        with module_ranges(model):
            actual = model(ids, num_loops=2).logits
        assert torch.equal(expected, actual)
    assert not model.recurrent._forward_hooks and not model.recurrent._forward_pre_hooks
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    tasks = [generate_example(i, d, 'train', i) for i,d in enumerate([1,2])]
    items = [EncodedExample(t, [3,4,5], [SYMBOLS.index(s) for s in t.intermediate_states]) for t in tasks]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002)
    stats = training_update(model, items, spec['token_ids'], 0, optimizer, 2)
    assert stats['executed_transitions'] == 4 and stats['supervised_transitions'] == 3
    for n,p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p,frozen[n]) and p.grad is None


def test_diagnostic_and_profiler_clis_are_offline_and_checkpoint_safe(tmp_path, tiny_checkpoint):
    checkpoint, _, _, _ = tiny_checkpoint
    before = {p.name:p.read_bytes() for p in checkpoint.iterdir() if p.is_file()}
    tasks = [generate_example(101+i, d, 'validation', i) for i,d in enumerate([2,3])]
    data = tmp_path/'validation.jsonl';data.write_text(''.join(json.dumps(t.to_dict())+'\n' for t in tasks))
    def run(module, *args):
        subprocess.run([sys.executable, '-m', module, *map(str,args)], cwd=ROOT, capture_output=True, text=True, check=True)
    common = ['--model',checkpoint,'--data',data,'--threads','1']
    profile = tmp_path/'profile'
    run('scripts.eval.profile_pointer',*common,'--output',profile,'--loops','3','--limit','2','--batch-size','2','--warmup','1','--repeats','2')
    saved = json.loads((profile/'summary.json').read_text())
    assert saved['status']=='complete' and len(saved['timing']['samples_seconds'])==2
    assert (profile/'trace.json.gz').stat().st_size > 0
    assert 'pointer/recurrent' in {r['name'] for r in json.loads((profile/'ranges.json').read_text())}
    probes = tmp_path/'probes'
    run('scripts.eval.probe_pointer',*common,'--output',probes,'--restart-after','1','--limit','2')
    probe = json.loads((probes/'summary.json').read_text()); assert probe['status']=='complete'
    assert len(list(csv.DictReader((probes/'states.csv').open()))) == 11
    loops = tmp_path/'loops'
    run('scripts.eval.loop_test',*common,'--output',loops,'--loops','3','--batch-size','2')
    run('scripts.eval.diagnose_pointer',loops,'--data',data,'--output',tmp_path/'failures')
    assert json.loads((tmp_path/'failures/summary.json').read_text())['examples']==2
    train_data=tmp_path/'train.jsonl'
    train_data.write_text(''.join(json.dumps(generate_example(i,d,'train',i).to_dict())+'\n' for i,d in enumerate([1,2])))
    train_output=tmp_path/'profile_train'
    run('scripts.eval.profile_pointer','--model',checkpoint,'--data',train_data,'--mode','train','--output',train_output,
        '--batch-size','1','--effective-batch','2','--train-max-depth','2','--warmup','1','--repeats','1','--threads','1')
    assert json.loads((train_output/'summary.json').read_text())['repeats'][0]['metrics']['supervised_transitions']==3
    assert before == {p.name:p.read_bytes() for p in checkpoint.iterdir() if p.is_file()}

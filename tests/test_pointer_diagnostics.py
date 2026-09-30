"""Risk-set, paired-intervention, and profiling contracts on tiny local models."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.dataset.pointer import SYMBOLS, execute, generate_example, parse_mapping
from scripts.eval.failure_diagnostics import analyze_failures
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_probes import task_slice, state_change
from scripts.eval.rule_edits import paired_rule_edits
from scripts.eval.rule_edit_probe import _summarize
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


def test_paired_rule_edits_change_one_edge_and_preserve_prefix():
    task = generate_example(51, 12, 'depth_test', 0)
    original = task.to_dict()
    for loop in (6, 10):
        relevant, irrelevant = paired_rule_edits(task, loop)
        for edit in (relevant, irrelevant):
            mapping = parse_mapping(edit.prompt.splitlines()[0].removeprefix('Rules: '))
            changed = [source for source in SYMBOLS if mapping[source] != dict(task.mapping)[source]]
            assert changed == [edit.source]
            assert edit.prompt.splitlines()[1:] == task.prompt.splitlines()[1:]
            assert list(edit.targets) == execute(mapping, task.initial_state, loop)
            assert list(edit.targets[:-1]) == task.intermediate_states[:loop - 1]
            assert edit.new_destination not in [task.initial_state, *task.intermediate_states]
        assert relevant.source == ([task.initial_state, *task.intermediate_states][loop - 1])
        assert relevant.expected == relevant.new_destination
        assert irrelevant.source not in [task.initial_state, *task.intermediate_states]
        assert irrelevant.expected == task.intermediate_states[loop - 1]
    assert task.to_dict() == original
    with pytest.raises(ValueError, match='within'):
        paired_rule_edits(task, 13)


def test_rule_edit_counts_use_retained_prefix_and_matched_cases():
    rows = [
        {'example_id': 'a', 'phase': 'late', 'condition': 'relevant', 'edited_prefix_correct': True,
         'edited_correct': True, 'prediction_changed': True, 'baseline_correct': False},
        {'example_id': 'a', 'phase': 'late', 'condition': 'irrelevant', 'edited_prefix_correct': True,
         'edited_correct': False, 'prediction_changed': False, 'baseline_correct': False},
        {'example_id': 'b', 'phase': 'late', 'condition': 'relevant', 'edited_prefix_correct': False,
         'edited_correct': True, 'prediction_changed': True, 'baseline_correct': False},
        {'example_id': 'b', 'phase': 'late', 'condition': 'irrelevant', 'edited_prefix_correct': True,
         'edited_correct': False, 'prediction_changed': False, 'baseline_correct': False},
    ]
    summary = _summarize(rows)['late']
    assert summary['relevant']['attempted'] == 2
    assert summary['relevant']['prefix_retained'] == 1
    assert summary['relevant']['follows_edited_target'] == 1
    assert summary['matched']['both_prefixes_retained'] == 1
    assert summary['matched']['relevant_follows_new_edge'] == 1


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


def test_rule_refresh_only_replaces_prefix_after_boundary(tiny_checkpoint):
    from scripts.eval.pointer_probes import rule_context_control
    _, model, _, _ = tiny_checkpoint
    model.eval()
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        expected = model(ids, num_loops=3, return_hidden_states=True)
        with rule_context_control(model, 2, 1, refresh=False):
            noop = model(ids, num_loops=3, return_hidden_states=True)
        assert all(torch.equal(a,b) for a,b in zip(expected.loop_logits,noop.loop_logits))
        seen = []
        with rule_context_control(model, 2, 1, refresh=True):
            # Registered second, this sees the input after intervention.
            hook = model.recurrent.register_forward_pre_hook(lambda module,args: seen.append(args[0].clone()))
            try:
                actual = model(ids, num_loops=3, return_hidden_states=True)
            finally:
                hook.remove()
        assert torch.equal(actual.loop_logits[0], expected.loop_logits[0])
        for i in range(1,3):
            assert torch.equal(seen[i][:,:2], actual.initial_hidden_state[:,:2])
            assert torch.equal(seen[i][:,2:], actual.hidden_states[i-1][:,2:])
        with pytest.raises(RuntimeError, match='deliberate'):
            with rule_context_control(model, 2, 1, refresh=True):
                raise RuntimeError('deliberate')
        assert not model.recurrent._forward_pre_hooks and not model.prelude._forward_hooks
    with pytest.raises(ValueError, match='no_grad'):
        with rule_context_control(model, 2, 1, refresh=True):
            pass


def test_control_probe_cli_records_prompt_and_scoring_horizons(tmp_path, tiny_checkpoint):
    checkpoint, _, _, _ = tiny_checkpoint
    before = {p.name:p.read_bytes() for p in checkpoint.iterdir() if p.is_file()}
    tasks = [generate_example(201+i,d,'validation',i) for i,d in enumerate([1,3])]
    data=tmp_path/'tasks.jsonl'
    data.write_text(''.join(json.dumps(t.to_dict())+'\n' for t in tasks))
    out=tmp_path/'controls'
    subprocess.run([sys.executable,'-m','scripts.eval.probe_pointer','--model',str(checkpoint),
        '--data',str(data),'--output',str(out),'--controls','--restart-after','1',
        '--limit','2','--threads','1'],cwd=ROOT,capture_output=True,text=True,check=True)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='complete' and summary['counts']['rule_noop']['prediction_changes']==0
    records=[json.loads(line) for line in (out/'tasks.jsonl').read_text().splitlines()]
    suffix=next(r for r in records if r['variant']=='suffix_original_steps')
    assert suffix['displayed_steps']==3 and suffix['scoring_loops']==2
    assert 'Steps: 3' in suffix['model_prompt'] and 'Steps: 2' in suffix['task']['prompt']
    assert suffix['task']['intermediate_states']==tasks[1].intermediate_states[1:]
    assert 0<suffix['rule_prefix_tokens']<suffix['answer_position']
    assert summary['quality_by_depth']['rule_refresh']['1']['examples']==1


def test_rule_edit_probe_cli_preserves_checkpoint_and_saved_prompts(tmp_path, tiny_checkpoint):
    from scripts.eval.pointer_probes import inspect_example
    from scripts.training.data import encode_tasks
    checkpoint, model, tokenizer, spec = tiny_checkpoint
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()}
    token_map = dict(zip(spec['symbols'], spec['token_ids'], strict=True))
    chosen = None
    for seed in range(300):
        task = generate_example(10000 + seed, 2, 'depth_test', 0)
        item = encode_tasks([task], tokenizer, token_map, model.config.max_position_embeddings)[0]
        rows = inspect_example(model, item, spec['token_ids'], tokenizer.pad_token_id)
        if rows[0]['correct'] and not rows[1]['correct']:
            chosen = task
            break
    assert chosen is not None, 'Tiny model should provide a late first-error fixture'
    data = tmp_path/'depth_test.jsonl'
    data.write_text(json.dumps(chosen.to_dict())+'\n')
    output = tmp_path/'rule_edits'
    subprocess.run([sys.executable,'-m','scripts.eval.rule_edit_probe','--model',str(checkpoint),
                    '--data',str(data),'--output',str(output),'--early-loop','1',
                    '--limit','1','--threads','1'],cwd=ROOT,capture_output=True,text=True,check=True)
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['eligible_late_first_errors'] == 1
    assert len(list(csv.DictReader((output/'pairs.csv').open()))) == 4
    inputs = [json.loads(line) for line in (output/'inputs.jsonl').read_text().splitlines()]
    assert len(inputs) == 4 and all(record['prompt'].endswith('Steps: 2\nAnswer:') for record in inputs)
    assert before == {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()}
    assert before=={p.name:p.read_bytes() for p in checkpoint.iterdir() if p.is_file()}


def test_rule_boundary_rejects_crossing_tokens():
    from scripts.eval.pointer_probes import rule_prefix_length
    class CrossingTokenizer:
        def __call__(self, prompt, **kwargs):
            return {'offset_mapping': [(0,len(prompt))]}
    with pytest.raises(ValueError,match='crosses'):
        rule_prefix_length(CrossingTokenizer(),'Rules: ( A, B)\nStart: A\nSteps: 1\nAnswer:')


def test_rule_boundary_includes_merged_closing_parenthesis_and_newline():
    from scripts.eval.pointer_probes import rule_prefix_length
    prompt='Rules: ( A, B)\nStart: A\nSteps: 1\nAnswer:'
    boundary=prompt.index('Start:')
    class MergedTokenizer:
        def __call__(self, prompt, **kwargs):
            return {'offset_mapping': [(0,boundary-2),(boundary-2,boundary),(boundary,len(prompt))]}
    assert rule_prefix_length(MergedTokenizer(),prompt)==2


def test_control_shell_logs_and_stops_on_failure(tmp_path):
    import os
    import shutil
    shutil.copy(ROOT/'probe_controls.sh',tmp_path/'probe_controls.sh')
    (tmp_path/'.venv/bin').mkdir(parents=True)
    (tmp_path/'.venv/bin/activate').write_text('')
    checkpoint=tmp_path/'models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500'
    checkpoint.mkdir(parents=True)
    for name in ['recurrent_config.json','adapter_model.pt','tokenizer.json']:
        (checkpoint/name).touch()
    stub=tmp_path/'.venv/bin/python'
    stub.write_text('#!/usr/bin/env bash\necho "stub call $*"\necho "stub stderr" >&2\nexit "${PROBE_TEST_EXIT:-0}"\n')
    stub.chmod(0o755)
    env={**os.environ,'PATH':str(stub.parent)+os.pathsep+os.environ['PATH']}
    success=subprocess.run(['bash','probe_controls.sh'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert success.returncode==0 and success.stdout.count('stub call')==2
    failure=subprocess.run(['bash','probe_controls.sh'],cwd=tmp_path,env={**env,'PROBE_TEST_EXIT':'7'},capture_output=True,text=True)
    assert failure.returncode==7 and failure.stdout.count('stub call')==1
    logs=list((tmp_path/'eval/pointer_probes').glob('*/run.log'))
    assert len(logs)==2 and all('stub stderr' in p.read_text() for p in logs)

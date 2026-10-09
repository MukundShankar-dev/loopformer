"""Shared reading extrapolates positions/lengths without adding training counts."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_executor_upgrade import setup
from test_controller_training import source_checkpoint
from scripts.dataset.benchmark import count_variant
from scripts.dataset.pointer import generate_unconditioned_example
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.interfaces import SharedNumberController
from scripts.recurrent_qwen.number_reader import MAX_NUMBER_TOKENS, SharedNumberReader, number_token_ids
from scripts.training.controller import ControllerConfig, export_checkpoint, select_graphs
from scripts.training.controller_cache import prompt_features
from scripts.training.data import collate, encode_tasks
from scripts.training.number_reader import fit_number_reader


TRAIN_COUNTS = [n for n in range(1,64) if n not in [9,17,29,41,53]]


def packed(strings, embeddings):
    context = torch.zeros(len(strings), MAX_NUMBER_TOKENS, embeddings.shape[1]+1)
    for row, text in enumerate(strings):
        for column, token in enumerate(text):
            context[row,column,:-1] = embeddings[int(token)]
            context[row,column,-1] = 1
    return context.flatten(1)


def test_shared_reader_learns_radix_and_unseen_digit_positions_and_lengths():
    torch.manual_seed(17)
    embeddings = torch.randn(10,16)
    reader = SharedNumberReader(16)
    assert reader.gain.item() == 1  # No decimal radix is installed.
    result = fit_number_reader(reader, packed([str(n) for n in TRAIN_COUNTS], embeddings), TRAIN_COUNTS)
    assert result['learned_gain'] == pytest.approx(10)
    assert not result['digit_value_labels']
    assert result['training_counts'] == TRAIN_COUNTS
    counts = list(range(1,10001))
    with torch.no_grad():
        predictions = reader(packed([str(n) for n in counts],embeddings)).flatten()
    torch.testing.assert_close(predictions, torch.tensor(counts,dtype=torch.float32), atol=.01, rtol=1e-6)
    # An alternative labeling learns another radix with the identical fitter.
    strings = [str(n) for n in range(1,7)] + [a+b for a in '123456' for b in '0123456']
    labels = [int(s) if len(s)==1 else int(s[0])*7+int(s[1]) for s in strings]
    other = SharedNumberReader(16)
    assert fit_number_reader(other,packed(strings,embeddings),labels)['learned_gain'] == pytest.approx(7)
    torch.testing.assert_close(other(packed(['106'],embeddings)),torch.tensor([[55.]]),atol=1e-4,rtol=0)


def test_number_routing_rejects_format_and_tokenizer_contract_violations(setup):
    _, tokenizer, _, _, _ = setup
    prompt = 'Rules: ( A, B)\nStart: A\nSteps: 100\nAnswer:'
    ids = tuple(tokenizer.encode(prompt,add_special_tokens=False))
    assert tokenizer.decode(list(number_token_ids(tokenizer,ids))) == '100'
    for bad in (prompt.replace('100','01'),prompt.replace('100','123456789'),prompt.replace('Steps','Count')):
        with pytest.raises(ValueError):
            number_token_ids(tokenizer,tuple(tokenizer.encode(bad,add_special_tokens=False)))


def test_shared_controller_export_native_padding_and_count_free_executor(setup):
    model, tokenizer, tokens, path = source_checkpoint(setup)
    head = SharedNumberController(model.config.hidden_size)
    model.completion_head = head
    task = generate_unconditioned_example(17,1,'train',0)
    training = [count_variant(task,n) for n in TRAIN_COUNTS]
    context = prompt_features(model,tokenizer,list(tokens.values()),training,16)
    fit_number_reader(head.context,context,TRAIN_COUNTS)
    # Test fixture cell/readout; production repair copies the trained values.
    with torch.no_grad():
        head.cell.bias.fill_(-1)
        head.readout.weight.fill_(-4); head.readout.bias.fill_(2)
    original = torch.load(path/'source/adapter_model.pt',weights_only=True)
    export_checkpoint(path/'source',path/'shared',head,{'step':0})
    weights = torch.load(path/'shared/adapter_model.pt',weights_only=True)
    for k,v in original.items():
        if not k.startswith('completion_head.'):
            torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
    loaded, _, spec = load_recurrent_checkpoint(path/'shared')
    assert spec['executor']['controller_kind']=='shared_number'
    variants = [count_variant(task,n) for n in (9,70,100,256)]
    items = encode_tasks(variants,tokenizer,tokens,512)
    batch = collate(items,tokenizer.pad_token_id,'cpu')
    native_context = loaded.controller_prompt_state(batch['input_ids'],batch['attention_mask'])
    torch.testing.assert_close(loaded.completion_head.initialize(native_context),torch.tensor([[9.],[70.],[100.],[256.]]),atol=1e-4,rtol=0)
    left_ids = torch.full_like(batch['input_ids'],tokenizer.pad_token_id)
    left_mask = torch.zeros_like(batch['attention_mask'])
    for row in range(len(items)):
        valid = batch['input_ids'][row][batch['attention_mask'][row].bool()]
        left_ids[row,-len(valid):]=valid;left_mask[row,-len(valid):]=1
    torch.testing.assert_close(loaded.controller_prompt_state(left_ids,left_mask),native_context,atol=0,rtol=0)
    with torch.no_grad():
        result = loaded(batch['input_ids'],batch['attention_mask'],num_loops=3,readout_token_ids=list(tokens.values()))
    for scores in result.loop_logits:
        torch.testing.assert_close(scores,scores[:1].expand_as(scores),atol=1e-6,rtol=1e-5)
    # Replay every learned update; no step count is passed into the head.
    memory = loaded.completion_head.initialize(native_context)
    stops = [None]*4
    for loop in range(1,273):
        logits,memory = loaded.completion_head.advance(torch.randn(4,model.config.hidden_size),memory)
        for i,score in enumerate(logits):
            if stops[i] is None and score >= 0:
                stops[i]=loop
    assert stops==[9,70,100,256]
    assert not (path/'shared/summary.json').exists()


def test_repair_keeps_original_training_protocol():
    config=json.loads(Path('configs/controller_affine.json').read_text())
    repair=json.loads(Path('configs/controller_number_reader.json').read_text())
    assert repair['training_config']=='configs/controller_affine.json'
    assert config['requested_counts']==TRAIN_COUNTS and config['training_loops']==12
    assert MAX_NUMBER_TOKENS==8


def test_reader_repair_cli_preserves_countdown_and_refuses_changed_training_panel(setup):
    from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset
    from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
    from scripts.eval.loop_metrics import write_csv
    from scripts.eval.pointer_task import sha256_file
    from scripts.training.data import read_tasks
    model,tokenizer,tokens,path=source_checkpoint(setup)
    dataset=DatasetConfig(seed=61,graph_mode='mixture',paired_horizons=True,
        train_count=4,validation_count=2,test_count=2,depth_test_count=2,max_train_depth=2,max_eval_depth=3)
    write_dataset(path/'data',generate_dataset(dataset,tokenizer,tokens),dataset,tokens,{})
    config=ControllerConfig(source=str(path/'source'),data=str(path/'data'),device='cpu',
        train_graphs=2,validation_graphs=1,training_loops=12,requested_counts=TRAIN_COUNTS,
        remaining_readout=True,remaining_loss_weight=1,controller_kind='affine_suffix')
    training=path/'training.json';training.write_text(json.dumps(config.to_dict()))
    old=path/'old-run'
    from scripts.recurrent_qwen.interfaces import AffineSuffixController
    old_head=AffineSuffixController(model.config.hidden_size)
    export_checkpoint(path/'source',old/'best',old_head,{'trained_depths':TRAIN_COUNTS,
        'training_loops':12,'config':config.to_dict()})
    before=checkpoint_hashes(old/'best')
    selected=select_graphs(read_tasks(path/'data/train.jsonl','train'),2,TRAIN_COUNTS,config.seed)
    write_csv(old/'train_panel.csv',[{'example_id':t.example_id} for t in selected])
    (old/'run.json').write_text(json.dumps({'train_data_sha256':sha256_file(path/'data/train.jsonl')}))
    settings={'source':str(old/'best'),'training_config':str(training),'output':str(path/'new-run')}
    file=path/'repair.json';file.write_text(json.dumps(settings))
    args=[sys.executable,'-m','scripts.training.repair_number_reader','--config',str(file)]
    preview=subprocess.run(args+['--dry-run'],capture_output=True,text=True)
    assert preview.returncode==0 and not (path/'new-run').exists()
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert checkpoint_hashes(old/'best')==before
    original=torch.load(old/'best/adapter_model.pt',weights_only=True)
    exported=torch.load(path/'new-run/best/adapter_model.pt',weights_only=True)
    for name in original:
        if not name.startswith('completion_head.context.'):
            torch.testing.assert_close(original[name],exported[name],atol=0,rtol=0)
    summary=json.loads((path/'new-run/summary.json').read_text())
    assert summary['training_loops']==12 and summary['fit']['training_counts']==TRAIN_COUNTS
    assert subprocess.run(args,capture_output=True,text=True).returncode!=0
    config_dict=config.to_dict();config_dict['requested_counts']=TRAIN_COUNTS[:-1]
    training.write_text(json.dumps(config_dict))
    settings['output']=str(path/'invalid-run');file.write_text(json.dumps(settings))
    bad=subprocess.run(args,capture_output=True,text=True)
    assert bad.returncode!=0 and 'actual training config differs' in bad.stderr
    assert not (path/'invalid-run').exists()

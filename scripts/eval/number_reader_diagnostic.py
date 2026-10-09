"""Separate number reading from countdown drift; no executor-quality claim."""
import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import track

from scripts.dataset.pointer import render_prompt
from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.eval.loop_metrics import write_csv
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.data import read_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/stage1_pointer/controller-shared-number-seed83/best'))
    parser.add_argument('--output', type=Path, default=Path('eval/pointer_benchmark/shared-number-20261008/numeric'))
    parser.add_argument('--data', type=Path, default=Path('data/pointer/seed-61-independent/train.jsonl'))
    parser.add_argument('--reader-max', type=int, default=10000)
    parser.add_argument('--rollout-max', type=int, default=4096)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = parser.parse_args()
    if not 1 <= args.rollout_max <= args.reader_max <= 10000 or args.output.exists():
        raise ValueError('Need 1 <= rollout maximum <= reader maximum <= 10000 and a new output path')
    before = checkpoint_hashes(args.model)
    torch.set_num_threads(4)
    model, tokenizer, _ = load_recurrent_checkpoint(args.model,device=args.device)
    model.eval().requires_grad_(False)
    if model.completion_head.kind != 'shared_number':
        raise ValueError('Need the shared-number frozen controller')
    task = read_tasks(args.data,'train')[0]
    initial = []
    with torch.inference_mode():
        for start in track(range(1,args.reader_max+1,128),description='Read raw number fields'):
            prompts = [render_prompt(task.mapping,task.initial_state,n)
                       for n in range(start,min(start+128,args.reader_max+1))]
            batch = tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to(args.device)
            context = model.controller_prompt_state(batch['input_ids'],batch['attention_mask'])
            initial.append(model.completion_head.initialize(context).cpu())
        initial = torch.cat(initial).flatten()
        memory = initial[:args.rollout_max,None].to(args.device)
        stops = torch.zeros(args.rollout_max,dtype=torch.long,device=args.device)
        observations = torch.zeros(args.rollout_max,model.config.hidden_size,device=args.device)
        for loop in track(range(1,args.rollout_max+17),description='Frozen countdown only'):
            logits,memory = model.completion_head.advance(observations,memory)
            stops = torch.where((stops==0)&(logits>=0),loop,stops)
        stops = stops.cpu().tolist()
    labels = torch.arange(1,args.reader_max+1)
    errors = (initial.double()-labels).abs()
    if checkpoint_hashes(args.model) != before:
        raise ValueError('Frozen checkpoint changed during diagnostic')
    args.output.mkdir(parents=True)
    write_csv(args.output/'reader.csv',[{'requested':n,'initial_memory':float(initial[n-1]),
        'absolute_error':float(errors[n-1])} for n in range(1,args.reader_max+1)])
    write_csv(args.output/'countdown.csv',[{'requested':n,'initial_memory':float(initial[n-1]),
        'first_stop':stops[n-1] or None,'exact_stop':stops[n-1]==n,'early_stop':0<stops[n-1]<n,
        'late_stop':stops[n-1]>n,'missing_stop':stops[n-1]==0} for n in range(1,args.rollout_max+1)])
    failed = [n for n in range(1,args.rollout_max+1) if stops[n-1]!=n]
    result = {'status':'complete','checkpoint':str(args.model),'inference_sha256':before,
        'reader_max':args.reader_max,'reader_max_absolute_error':float(errors.max()),
        'reader_mean_absolute_error':float(errors.mean()),'rollout_max':args.rollout_max,
        'exact_stop_questions':args.rollout_max-len(failed),'first_failed_stop_count':failed[0] if failed else None,
        'countdown_gain':float(model.completion_head.cell.weight.item()),
        'countdown_offset':float(model.completion_head.cell.bias.item()),
        'inference_files_unchanged':True,'executor_executed':False,
        'interpretation':'Numeric diagnostic only; no fit, retuning, or pointer-quality claim'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    Console().print(f'Reader max error: {result["reader_max_absolute_error"]:g} · exact countdown: {result["exact_stop_questions"]}/{args.rollout_max} · first failure: {result["first_failed_stop_count"]}')


if __name__ == '__main__':
    main()

"""Configure ordinary trace SFT with matched data/capacity; no recurrence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from importlib.metadata import version

import torch
from rich.console import Console
from transformers import AutoTokenizer

from scripts.eval.loading import load_model
from scripts.eval.pointer_task import sha256_file
from scripts.eval.trace_baselines import weight_hashes
from scripts.trace_task import encode_trace
from scripts.training.data import read_tasks,select_tasks
from scripts.training.trace_runner import run


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/trace_sft.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--wandb-mode',choices=('online','offline','disabled'),default=os.environ.get('WANDB_MODE','disabled'))
    args=parser.parse_args();c=json.loads(args.config.read_text());output=args.output or Path(c['output'])
    if args.smoke_test and not args.output:parser.error('Smoke test requires a separate --output')
    if output.exists() and not args.resume and not args.dry_run:parser.error('Output exists; choose fresh path or explicit --resume')
    reference=json.loads(Path(c['reference_config']).read_text())
    for key in ('model','revision','train_data','validation_data','seed','train_depths','learning_rate',
                'warmup_steps','weight_decay','max_grad_norm','epochs','lr_schedule','schedule_steps','min_lr_ratio'):
        if c[key]!=reference[key]:raise ValueError(f'Trace SFT differs from executor protocol at {key}')
    if c['batch_size']*c['gradient_accumulation']!=reference['batch_size']*reference['gradient_accumulation']:
        raise ValueError('Trace SFT must match executor effective batch')
    if c['deterministic']!=reference.get('deterministic',True):raise ValueError('Deterministic setting differs from executor')
    if c['train_scope']=='middle' and (c['train_layer_start'],c['train_layer_end'])!=(reference['recurrent_start'],reference['recurrent_end']):
        raise ValueError('Adapted layer range differs from R')
    for key in ('train','validation'):
        if sha256_file(Path(c[f'{key}_data']))!=c[f'{key}_sha256']:raise ValueError(f'{key} dataset hash differs')
    tokenizer=AutoTokenizer.from_pretrained(c['model'],revision=c['revision'],local_files_only=True)
    template=Path(c['prompt']).read_text()
    tasks=select_tasks(read_tasks(Path(c['train_data']),'train'),12,c['seed'],depths=c['train_depths'])
    val=select_tasks(read_tasks(Path(c['validation_data']),'validation'),12,c['seed'],per_depth=2 if args.smoke_test else c['validation_per_depth'])
    if args.smoke_test:tasks=tasks[:160]
    encoded=[encode_trace(t,tokenizer,template,c['max_sequence_tokens']) for t in tasks]
    validation=[encode_trace(t,tokenizer,template,c['max_sequence_tokens']) for t in val]
    identity=dict(config=c,initial_weights_sha256=weight_hashes(c['model'],c['revision']),
        prompt_sha256=sha256_file(Path(c['prompt'])),reference_config_sha256=sha256_file(Path(c['reference_config'])),
        train_ids_sha256=hashlib.sha256('\n'.join(t.example_id for t in tasks).encode()).hexdigest(),
        validation_ids_sha256=hashlib.sha256('\n'.join(t.example_id for t in val).encode()).hexdigest(),
        smoke_test=args.smoke_test,sources_sha256={p:sha256_file(Path(p)) for p in
            ('scripts/trace_task.py','scripts/training/train_trace.py','scripts/training/trace_runner.py',
             'scripts/eval/pointer_task.py','scripts/training/data.py','scripts/training/runner.py',
             'scripts/eval/cached_generation.py',
             'scripts/training/precision.py','scripts/dataset/pointer.py')},
        packages={p:version(p) for p in ('torch','transformers','tokenizers')})
    Console().print(f'[cyan]Ordinary trace SFT · {len(tasks):,} examples · 1 epoch · {len(tasks)//16:,} updates[/cyan]\n'
        f'Train scope {c["train_scope"]} · microbatch {c["batch_size"]} · accumulation {c["gradient_accumulation"]} · effective batch 16 · {c["precision"]}/{c["attention"]}\n'
        f'Maximum encoded training length {max(len(x.input_ids) for x in encoded)} · no truncation\nOutput: {output}')
    if args.dry_run:return
    if c['device']=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA required by this explicit config')
    if c['device']=='cuda' and c['deterministic'] and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA, or use trace_baselines.sh')
    torch.manual_seed(c['seed']);torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(c['deterministic'])
    model,tokenizer,loaded=load_model(str(args.resume) if args.resume else c['model'],base_model=None,tokenizer_source=None,
        revision=c['revision'],device=c['device'],dtype='float32',download=False)
    model.set_attn_implementation(c['attention'])
    atomic_metadata = dict(loaded=loaded, tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                          chat_template=tokenizer.chat_template)
    output.mkdir(parents=True,exist_ok=True)
    from scripts.eval.paper_baseline import atomic_json
    atomic_json(output/'loaded.json',atomic_metadata)
    result=run(model,tokenizer,encoded,validation,c,identity,output,smoke=args.smoke_test,resume=args.resume,wandb_mode=args.wandb_mode)
    Console().print(f'[green]Complete · best trained-count response CE {result["best_loss"]:.4f}[/green]')


if __name__=='__main__':main()

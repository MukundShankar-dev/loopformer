"""Cached ordinary trace generation on the existing graph/count benchmark.

Every query supplies rules, start and requested count once. Gold traces never
enter generation. Atomic count-major chunks make the long benchmark resumable.
"""
import argparse
import gc
import gzip
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from scripts.dataset.benchmark import count_variant
from scripts.eval.loading import load_model
from scripts.eval.cached_generation import cached_batches
from scripts.eval.paper_baseline import atomic_json, text_hash
from scripts.eval.paper_suite import load_graphs
from scripts.eval.pointer_task import load_examples, sha256_file
from scripts.training.tracking import tracking_run


def weight_hashes(source: str, revision: str) -> dict:
    """Resolve cached weights only; hash bytes independently of loaded precision."""
    path=Path(source)
    if not path.is_dir():path=Path(snapshot_download(source,revision=revision,local_files_only=True))
    files=sorted(path.glob('*.safetensors'))
    if not files:raise ValueError(f'No safetensors weights in {path}')
    return {p.name:sha256_file(p) for p in files}


def generate(model, tokenizer, tasks: list, template: str, config: dict, speed: dict) -> tuple[list, float]:
    rows=[];seconds=0.
    for part,elapsed in cached_batches(model,tokenizer,tasks,template,
            batch_size=speed['batch_size'],max_new_tokens=config['max_new_tokens'],
            cache=speed['cache'],compiled=speed['compiled'],
            max_cache_len=config['max_cache_len'] if speed['cache']=='static' else None):
        rows.extend(part);seconds+=elapsed
    return rows,seconds


def profile(model,tokenizer,tasks: list,template: str,config: dict,output: Path) -> dict:
    """Choose throughput only, never task accuracy; compiled decode must match eager.

    The profile cohort is validation data. Long outputs are not forced. Compile
    warmup is measured separately and excluded from steady-state throughput.
    """
    candidates=[];console=Console()
    for b in config['batch_candidates']:
        samples=[tasks[i%len(tasks)] for i in range(b)]
        speed=dict(batch_size=b,cache='dynamic',compiled=False)
        console.print(f'[cyan]Profiling cached generation · batch {b}[/cyan]')
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            generate(model,tokenizer,samples,template,config,speed)
            rows,seconds=generate(model,tokenizer,samples,template,config,speed)
        except torch.OutOfMemoryError:
            candidates.append(dict(**speed,rejected='CUDA capacity exceeded'))
            gc.collect();torch.cuda.empty_cache();continue
        candidates.append(dict(**speed,seconds=seconds,questions_per_second=b/seconds,
            tokens_per_second=sum(r['generated_tokens'] for r in rows)/seconds,
            cuda_peak_bytes=torch.cuda.max_memory_allocated()))
        atomic_json(output/'speed_profile.json',dict(status='profiling',candidates=candidates))
    feasible=[x for x in candidates if 'questions_per_second' in x]
    if not feasible:raise ValueError('No profiled inference batch fits the available GPU')
    selected=max(feasible,key=lambda x:x['questions_per_second'])
    if not config['profile_compilation']:
        atomic_json(output/'speed_profile.json',dict(selected=selected,candidates=candidates,
            compilation='Disabled after preflight static-cache decode was slower and changed continuations'))
        return selected
    b=selected['batch_size'];samples=[tasks[i%len(tasks)] for i in range(b)]
    eager,_=generate(model,tokenizer,samples,template,config,selected)
    compiled=dict(batch_size=b,cache='static',compiled=True)
    console.print(f'[cyan]Profiling compiled cached decode · batch {b} (first call compiles)[/cyan]')
    torch.cuda.reset_peak_memory_stats();started=perf_counter()
    try:
        warm,_=generate(model,tokenizer,samples,template,config,compiled)
        warmup=perf_counter()-started
        rows,seconds=generate(model,tokenizer,samples,template,config,compiled)
    except torch.OutOfMemoryError:
        candidates.append(dict(**compiled,rejected='CUDA capacity exceeded'))
        if hasattr(model,'_cache'):del model._cache
        gc.collect();torch.cuda.empty_cache()
        atomic_json(output/'speed_profile.json',dict(selected=selected,candidates=candidates))
        return selected
    same=all(a['generated_token_ids']==z['generated_token_ids'] for a,z in zip(eager,rows,strict=True))
    candidate=dict(**compiled,seconds=seconds,questions_per_second=b/seconds,
        tokens_per_second=sum(r['generated_tokens'] for r in rows)/seconds,
        warmup_seconds=warmup,exact_eager_continuations=same,
        cuda_peak_bytes=torch.cuda.max_memory_allocated())
    candidates.append(candidate)
    if same and candidate['questions_per_second']>selected['questions_per_second']:selected=candidate
    if selected['cache']=='dynamic' and hasattr(model,'_cache'):
        del model._cache;gc.collect();torch.cuda.empty_cache()
    report=dict(selected=selected,candidates=candidates,scope='Validation requests only; throughput selection, no quality tuning')
    atomic_json(output/'speed_profile.json',report)
    return selected


def native_check(model,tokenizer,tasks: list,metadata: list,template: str,config: dict,speed: dict) -> dict:
    """Record batch/compile sensitivity, without selecting weights on this opened panel."""
    samples=[];seen=set()
    for task,meta in zip(tasks,metadata,strict=True):
        key=meta['dataset_seed'],meta['graph_mode']
        if key not in seen:
            seen.add(key);samples.extend(count_variant(task,n) for n in config['native_counts'])
    batched,_=generate(model,tokenizer,samples,template,config,speed)
    native,_=generate(model,tokenizer,samples,template,config,dict(batch_size=1,cache='dynamic',compiled=False))
    fields=('correct','complete_trajectory','exact_stop','strict_success')
    comparisons=[dict(example_id=a['example_id'],raw_equal=a['generated_token_ids']==b['generated_token_ids'],
        scores_equal=all(a[k]==b[k] for k in fields),batched_ids=a['generated_token_ids'],native_ids=b['generated_token_ids'])
        for a,b in zip(batched,native,strict=True)]
    return dict(questions=len(samples),exact_continuations=all(r['raw_equal'] for r in comparisons),
        exact_scores=all(r['scores_equal'] for r in comparisons),observations=comparisons,
        scope='BF16 batch-shape sensitivity audit. Disagreements are reported, not hidden or used to select a checkpoint.')


def save_chunk(path: Path,rows: list,seconds: float,indices: list[int],depth: int,identity: str) -> None:
    temporary=path.with_suffix('.tmp.gz')
    with gzip.open(temporary,'wt') as handle:
        handle.write(json.dumps(dict(generation_seconds=seconds,indices=indices,depth=depth,freeze_sha256=identity))+'\n')
        for index,row in zip(indices,rows,strict=True):
            record={k:v for k,v in row.items() if k not in ('prompt','model_input')}
            record.update(graph_index=index,input_sha256=text_hash(row['model_input']))
            handle.write(json.dumps(record)+'\n')
    temporary.replace(path)


def read_chunk(path: Path) -> tuple[dict,list]:
    with gzip.open(path,'rt') as handle:
        header=json.loads(next(handle));rows=[json.loads(line) for line in handle]
    return header,rows


def extract(config: dict,root: Path,arm: str,*,smoke: bool=False,profile_only: bool=False,wandb_mode: str='disabled') -> None:
    source=config['model'] if arm=='original' else str(Path(config['training'])/'best')
    output=root/arm;output.mkdir(parents=True,exist_ok=True)
    if sha256_file(Path(config['graphs']))!=config['graphs_sha256']:raise ValueError('Benchmark graph checksum differs')
    if sha256_file(Path(config['test']))!=config['test_sha256']:raise ValueError('Opened test checksum differs')
    if sha256_file(Path(config['validation']))!=config['validation_sha256']:raise ValueError('Profile validation checksum differs')
    if config['requests']!=list(range(1,257)):raise ValueError('Main protocol requires every requested count 1–256')
    tasks,metadata=load_graphs(Path(config['graphs']))
    if len(tasks)!=config['graph_count']:raise ValueError('Benchmark graph coverage differs')
    counts=config['requests']
    if smoke:tasks=tasks[:3];metadata=metadata[:3];counts=[1,2,3,12]
    hashes=weight_hashes(source,config['revision']);template=Path(config['prompt']).read_text()
    identity=dict(config=config,arm=arm,smoke=smoke,weights_sha256=hashes,
        prompt_sha256=text_hash(template),test_sha256=sha256_file(Path(config['test'])),
        packages={k:version(k) for k in ('torch','transformers','tokenizers')},
        sources_sha256={p:sha256_file(Path(p)) for p in ('scripts/trace_task.py','scripts/eval/trace_baselines.py',
            'scripts/eval/cached_generation.py','scripts/eval/pointer_task.py')})
    freeze=output/'freeze.json'
    if freeze.exists():
        saved=json.loads(freeze.read_text())
        if saved['identity']!=identity:raise ValueError('Resume weights/protocol/prompt/code identity differs')
        speed=saved['speed']
        if (output/'summary.json').exists() and json.loads((output/'summary.json').read_text())['status']=='complete':return
    else:speed=None
    torch.set_num_threads(config['threads'])
    model,tokenizer,loaded=load_model(source,base_model=None,tokenizer_source=None,revision=config['revision'],
        device=config['device'],dtype=config['dtype'],download=False)
    model.requires_grad_(False);model.set_attn_implementation(config['attention']);model.eval()
    tokenizer.save_pretrained(output/'tokenizer');(output/'prompt.txt').write_text(template)
    atomic_json(output/'loaded.json',dict(**loaded,chat_template=tokenizer.chat_template,
        tokenizer_sha256=text_hash(tokenizer.backend_tokenizer.to_str()),eos_token_id=model.generation_config.eos_token_id))
    if profile_only:
        validation=load_examples(Path(config['validation']))
        profile(model,tokenizer,[t for t in validation if t.task_depth==12][:128],template,config,output)
        return
    if speed is None:
        if smoke:speed=dict(batch_size=3,cache='dynamic',compiled=False)
        else:
            validation=load_examples(Path(config['validation']))
            speed=profile(model,tokenizer,[t for t in validation if t.task_depth==12][:128],template,config,output)
        atomic_json(freeze,dict(identity=identity,speed=speed))
    frozen_hash=sha256_file(freeze)
    if not (output/'native_fidelity.json').exists():
        atomic_json(output/'progress.json',dict(status='native sensitivity audit',queries=0,total=len(tasks)*len(counts)))
        audit=native_check(model,tokenizer,tasks,metadata,template,config,speed) if not smoke else dict(smoke=True)
        atomic_json(output/'native_fidelity.json',audit)
    chunks=output/'chunks';chunks.mkdir(exist_ok=True)
    done=correct=tokens=0;generation=0.;hashes_out={};start=perf_counter();b=speed['batch_size']
    total=len(tasks)*len(counts);console=Console()
    with tracking_run(output,identity,config,mode=wandb_mode,project='loopformer') as tracker, \
            Progress(TextColumn(f'{arm} trace'),BarColumn(),TextColumn('{task.completed}/{task.total}'),
                TimeElapsedColumn(),TimeRemainingColumn(),TextColumn('{task.fields[metrics]}')) as progress:
        job=progress.add_task('generation',total=total,metrics='')
        for depth in counts:
            for i in range(0,len(tasks),b):
                indices=list(range(i,min(i+b,len(tasks))));path=chunks/f'n{depth:03d}-g{i:04d}.jsonl.gz'
                if not path.exists():
                    batch=[count_variant(tasks[j],depth) for j in indices]
                    rows,seconds=generate(model,tokenizer,batch,template,config,speed)
                    save_chunk(path,rows,seconds,indices,depth,frozen_hash)
                header,rows=read_chunk(path)
                if (header['indices']!=indices or header['depth']!=depth or header['freeze_sha256']!=frozen_hash
                    or len(rows)!=len(indices)):raise ValueError(f'Chunk coverage/identity differs: {path}')
                done+=len(rows);correct+=sum(r['strict_success'] for r in rows)
                tokens+=sum(r['generated_tokens'] for r in rows);generation+=header['generation_seconds']
                hashes_out[path.name]=sha256_file(path)
                record=dict(status='extracting',queries=done,total=total,requested_depth=depth,
                    strict_success=correct/done,generation_seconds=generation,generated_tokens=tokens,
                    tokens_per_second=tokens/generation,questions_per_second=done/generation,
                    estimated_remaining_seconds=(total-done)*generation/done)
                atomic_json(output/'progress.json',record)
                progress.update(job,completed=done,metrics=f'{correct/done:.1%} strict · {tokens/generation:.0f} tok/s · N={depth}')
                tracker.log(dict(event='validation',step=done,validation=record))
        test=load_examples(Path(config['test']));test=test[:3] if smoke else test
        testpath=output/'reserved_test.jsonl.gz'
        if not testpath.exists():
            console.print('[cyan]Running the previously opened reserved test split[/cyan]')
            rows,seconds=generate(model,tokenizer,test,template,config,speed)
            save_chunk(testpath,rows,seconds,list(range(len(test))),0,frozen_hash)
        if weight_hashes(source,config['revision'])!=hashes:raise ValueError('Checkpoint changed during inference')
        summary=dict(status='complete',queries=done,total=total,strict_success=correct/done,generated_tokens=tokens,
            generation_seconds=generation,questions_per_second=done/generation,tokens_per_second=tokens/generation,
            chunks_sha256=hashes_out,reserved_test_sha256=sha256_file(testpath),weights_unchanged=True,
            freeze_sha256=frozen_hash,wall_seconds_this_invocation=perf_counter()-start,smoke=smoke)
        atomic_json(output/'summary.json',summary);atomic_json(output/'progress.json',summary);tracker.summary(summary)
    del model;gc.collect();torch.cuda.empty_cache()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/pointer_trace_baselines.json'))
    parser.add_argument('--output',type=Path);parser.add_argument('--arm',choices=('original','sft'),required=True)
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--wandb-mode',choices=('online','offline','disabled'),default='disabled')
    args=parser.parse_args();config=json.loads(args.config.read_text());root=args.output or Path(config['output'])
    if (args.smoke_test or args.profile_only) and not args.output:parser.error('Preflight needs a separate --output')
    root.mkdir(parents=True,exist_ok=True)
    protocol=root/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=config:raise ValueError('Output protocol differs')
    atomic_json(protocol,config)
    extract(config,root,args.arm,smoke=args.smoke_test,profile_only=args.profile_only,wandb_mode=args.wandb_mode)


if __name__=='__main__':main()

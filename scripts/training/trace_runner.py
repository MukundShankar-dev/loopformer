"""Ordinary-Qwen response SFT; standard autoregression, no recurrent wrapper."""
import json
import hashlib
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import torch
from rich.progress import Progress,BarColumn,TextColumn,TimeElapsedColumn,TimeRemainingColumn

from scripts.eval.paper_baseline import atomic_json
from scripts.eval.pointer_task import synchronize
from scripts.eval.cached_generation import cached_batches
from scripts.trace_task import TraceExample, trace_collate, trace_loss, set_trainable
from scripts.training.data import EncodedExample
from scripts.training.precision import autocast_context
from scripts.training.runner import epoch_order,rng_state,restore_rng
from scripts.training.tracking import tracking_run


def frozen_hash(model: Any) -> str:
    """Verify frozen tensor bytes before/after optimization, including tied weights."""
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rate_at_step(config: dict, step: int) -> float:
    """Same warmup/cosine schedule as the successful executor; steps are one-based."""
    factor=min(1.,step/max(1,config['warmup_steps']))
    if step>config['warmup_steps']:
        fraction=min(1.,(step-config['warmup_steps'])/(config['schedule_steps']-config['warmup_steps']))
        factor=config['min_lr_ratio']+(1-config['min_lr_ratio'])*.5*(1+math.cos(math.pi*fraction))
    return config['learning_rate']*factor


def validate(model: Any, items: list[TraceExample], config: dict, pad_id: int) -> dict:
    """Teacher-forced development loss; never describe it as free-generation accuracy."""
    model.eval(); sums={};b=config['batch_size']
    with torch.inference_mode():
        for depth in sorted({x.task.task_depth for x in items}):
            cohort=[x for x in items if x.task.task_depth==depth]
            row=sums.setdefault(depth,dict(examples=0,loss_sum=0.,tokens=0,correct=0,trajectory=0))
            for i in range(0,len(cohort),b):
                part=cohort[i:i+b];batch=trace_collate(part,pad_id,config['device'])
                with autocast_context(config['device'],config['precision']):loss,metrics=trace_loss(model,batch)
                row['examples']+=len(part);row['loss_sum']+=float(loss)*len(part)
                for key in ('tokens','correct','trajectory'):row[key]+=int(metrics[key])
    by_depth={str(d):dict(examples=v['examples'],loss=v['loss_sum']/v['examples'],
        teacher_forced_token_accuracy=v['correct']/v['tokens'],
        teacher_forced_response_accuracy=v['trajectory']/v['examples']) for d,v in sums.items()}
    trained=[v for d,v in by_depth.items() if int(d) in config['train_depths']]
    loss=sum(v['loss']*v['examples'] for v in trained)/sum(v['examples'] for v in trained)
    model.train()
    return dict(selection_loss=loss,by_depth=by_depth,scope='Teacher-forced response plus EOS CE; trained requested counts only for selection')


def free_validation(model: Any, tokenizer: Any, items: list[TraceExample], config: dict) -> dict:
    """Small fixed generated-trace panel; never used to choose the checkpoint."""
    selected=[]
    for depth in sorted({x.task.task_depth for x in items}):
        selected.extend([x.task for x in items if x.task.task_depth==depth][:config['free_validation_per_depth']])
    model.eval();rows=[]
    with autocast_context(config['device'],config['precision']):
        for part,_ in cached_batches(model,tokenizer,selected,Path(config['prompt']).read_text(),
            batch_size=config['batch_size'],max_new_tokens=config['free_validation_max_new_tokens'],
            cache='dynamic'):rows.extend(part)
    model.train()
    return {str(d):dict(examples=sum(r['task_depth']==d for r in rows),**{
        key:sum(r[key] for r in rows if r['task_depth']==d)/sum(r['task_depth']==d for r in rows)
        for key in ('correct','complete_trajectory','exact_stop','strict_success')})
        for d in sorted({r['task_depth'] for r in rows})}


def checkpoint(model: Any,tokenizer: Any,path: Path,state: dict) -> None:
    """Atomic complete HF checkpoint; keep only best and last, not every full model."""
    temporary=path.with_name(path.name+'.writing')
    if temporary.exists():shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary,safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save(state,temporary/'training_state.pt')
    atomic_json(temporary/'trace_training.json',{k:state[k] for k in ('step','best_loss','identity')})
    old=path.with_name(path.name+'.previous')
    if old.exists():shutil.rmtree(old)
    if path.exists():path.rename(old)
    temporary.rename(path)
    if old.exists():shutil.rmtree(old)


def run(model: Any,tokenizer: Any,train: list[TraceExample],validation: list[TraceExample],
        config: dict,identity: dict,output: Path,*,smoke: bool=False,resume: Path|None=None,
        wandb_mode: str='disabled') -> dict:
    """Exactly one visit per training example per epoch; checkpoint only optimizer boundaries."""
    names=set_trainable(model,config['train_scope'],config['train_layer_start'],config['train_layer_end'])
    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameters=[p for p in model.parameters() if p.requires_grad]
    frozen_before=frozen_hash(model)
    optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=config['weight_decay'],
                                 fused=config['device']=='cuda')
    b=config['batch_size'];acc=config['gradient_accumulation'];effective=b*acc
    max_steps=10 if smoke else math.ceil(len(train)/effective)*config['epochs']
    step=0;best=float('inf');offset=0;epoch=0
    if resume:
        saved=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Trace resume identity changed')
        optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],config['device'])
        step,best,offset,epoch=(saved[k] for k in ('step','best_loss','next_offset','next_epoch'))
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'config.json',config);atomic_json(output/'run.json',identity)
    if resume and (output/'metrics.jsonl').exists():
        rows=[json.loads(l) for l in (output/'metrics.jsonl').read_text().splitlines()]
        shutil.copy2(output/'metrics.jsonl', output/'metrics.before_resume.jsonl')
        (output/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=step))
    started=perf_counter();initial_step=step
    order_items=[EncodedExample(x.task,x.input_ids,x.response_ids) for x in train]
    model.train()
    def state(next_epoch: int,next_offset: int) -> dict:
        return dict(step=step,best_loss=best,next_epoch=next_epoch,next_offset=next_offset,
            optimizer=optimizer.state_dict(),rng=rng_state(config['device']),identity=identity)
    with tracking_run(output,identity,config,mode=wandb_mode,project='loopformer') as tracker, \
         (output/'metrics.jsonl').open('a' if resume else 'w') as log, \
         Progress(TextColumn('{task.description}'),BarColumn(),TextColumn('{task.completed}/{task.total}'),
                  TimeElapsedColumn(),TimeRemainingColumn(),TextColumn('{task.fields[metrics]}')) as progress:
        job=progress.add_task('Ordinary trace SFT',total=max_steps,completed=step,metrics='')
        def record(event: dict) -> None:
            log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();tracker.log(event)
            atomic_json(output/'progress.json',dict(status='training',step=step,total_steps=max_steps,
                elapsed_seconds=perf_counter()-started,**{k:v for k,v in event.items() if k not in ('event','step')}))
        if not resume:
            val=validate(model,validation,config,tokenizer.pad_token_id)
            val['free_generation']=free_validation(model,tokenizer,validation,config)
            record(dict(event='validation',step=0,validation=val))
        for e in range(epoch,config['epochs']):
            order=epoch_order(order_items,effective,config['seed']+e,config['bucket_by_depth'])
            start_offset=offset if e==epoch else 0
            for start in range(start_offset,len(order),effective):
                if step>=max_steps:break
                indices=order[start:start+effective];optimizer.zero_grad(set_to_none=True)
                synchronize(model.device);began=perf_counter()
                if config['device']=='cuda':torch.cuda.reset_peak_memory_stats()
                loss_sum=0.;tokens=0;correct=0
                for m in range(0,len(indices),b):
                    items=[train[i] for i in indices[m:m+b]]
                    batch=trace_collate(items,tokenizer.pad_token_id,config['device'])
                    with autocast_context(config['device'],config['precision']):loss,metrics=trace_loss(model,batch)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite trace SFT loss')
                    (loss*len(items)/len(indices)).backward()
                    loss_sum+=float(loss.detach())*len(items)/len(indices)
                    tokens+=int(metrics['tokens']);correct+=int(metrics['correct'])
                norm=torch.nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'],error_if_nonfinite=True)
                rate=rate_at_step(config,step+1)
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.step();step+=1;synchronize(model.device);seconds=perf_counter()-began
                event=dict(event='train',step=step,learning_rate=rate,train=dict(loss=loss_sum,
                    teacher_forced_token_accuracy=correct/tokens,response_tokens=tokens,
                    examples=len(indices),gradient_norm=float(norm),update_seconds=seconds,
                    examples_per_second=len(indices)/seconds,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated() if config['device']=='cuda' else 0))
                record(event);progress.update(job,completed=step,metrics=f'CE {loss_sum:.3f} · {len(indices)/seconds:.1f} ex/s')
                next_offset=start+len(indices);next_epoch=e
                if next_offset==len(order):next_epoch=e+1;next_offset=0
                if step%config['eval_every']==0 or step==max_steps:
                    val=validate(model,validation,config,tokenizer.pad_token_id)
                    val['free_generation']=free_validation(model,tokenizer,validation,config)
                    record(dict(event='validation',step=step,validation=val))
                    if val['selection_loss']<best:
                        best=val['selection_loss'];checkpoint(model,tokenizer,output/'best',state(next_epoch,next_offset))
                        atomic_json(output/'best_checkpoint.json',dict(step=step,path=str(output/'best'),selection_loss=best))
                if step%config['save_every']==0 or step==max_steps:
                    checkpoint(model,tokenizer,output/'last',state(next_epoch,next_offset))
            if step>=max_steps:break
        frozen_after=frozen_hash(model)
        if frozen_before!=frozen_after:raise ValueError('Ordinary SFT changed frozen parameter bytes')
        result=dict(status='complete',steps=step,examples=len(train),epochs=config['epochs'],
            effective_batch=effective,trainable_parameters=sum(p.numel() for p in parameters),
            total_parameters=sum(p.numel() for p in model.parameters()),
            frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
            frozen_weights_unchanged=True,frozen_parameters_sha256=frozen_after,
            trainable_names=names,best_loss=best,elapsed_seconds=perf_counter()-started,
            updates_this_invocation=step-initial_step,best_checkpoint=str(output/'best'),
            selection='Minimum teacher-forced trained-count validation response CE; no benchmark selection')
        atomic_json(output/'summary.json',result);atomic_json(output/'progress.json',result);tracker.summary(result)
    return result

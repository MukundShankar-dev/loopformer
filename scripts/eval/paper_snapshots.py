"""Full-panel executor learning progression with a fixed final-timer composition.

These are frozen executor snapshots, not historical autonomous solvers. Training
step 1000 has monitor records but no retained weights and is never reconstructed.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from rich.progress import track

from scripts.eval.paper_suite import load_graphs,save_chunk
from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.eval.pointer_task import sha256_file,synchronize
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.data import collate,encode_tasks
from scripts.training.objective import forward_symbols
from scripts.dataset.pointer import SYMBOLS


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/pointer_analysis_protocol.json'))
    parser.add_argument('--output',type=Path,default=Path('eval/pointer_analysis/paper-20261009'))
    args=parser.parse_args();config=json.loads(args.config.read_text())
    tasks,_=load_graphs(Path(config['graphs']));cap=config['safety_cap'];batch_size=64
    root=args.output/'executor_snapshots';root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(239);torch.use_deterministic_algorithms(True)
    for step in config['final_executor_history']['retained_weight_steps']:
        path=Path(config['final_executor_history']['run'])/f'step-{step:06d}'
        directory=root/f'step-{step:06d}';directory.mkdir(exist_ok=True)
        identity=dict(checkpoint_sha256=checkpoint_hashes(path),graph_sha256=sha256_file(Path(config['graphs'])),
                      dtype='float32',attention='sdpa',batch_size=batch_size,cap=cap,
                      recurrent_source_sha256={str(p):sha256_file(p) for p in Path('scripts/recurrent_qwen').glob('*.py')})
        frozen=directory/'freeze.json'
        if frozen.exists() and json.loads(frozen.read_text())!=identity:raise ValueError('Snapshot resume identity changed')
        frozen.write_text(json.dumps(identity,indent=2)+'\n')
        if (directory/'summary.json').exists():continue
        model,tokenizer,spec=load_recurrent_checkpoint(path,device='cuda')
        model.eval().requires_grad_(False);model.config._attn_implementation='sdpa'
        if model.router is None:raise ValueError('Snapshot R must be count-free')
        items=encode_tasks(tasks,tokenizer,dict(zip(SYMBOLS,spec['token_ids'],strict=True)),512)
        began=perf_counter()
        for start in track(range(0,len(items),batch_size),description=f'Frozen executor update {step}'):
            end=min(start+batch_size,len(items));chunk=directory/f'graphs-{start:04d}-{end:04d}.npz'
            if chunk.exists():continue
            batch=collate(items[start:end],tokenizer.pad_token_id,'cuda',cap)
            with torch.inference_mode():result,scores=forward_symbols(model,batch['input_ids'],batch['attention_mask'],spec['token_ids'],num_loops=cap)
            if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite snapshot readout')
            save_chunk(chunk,predictions=scores.argmax(-1).cpu().numpy().astype('uint8'),
                       direct=result.state_logits.argmax(-1).cpu().numpy().astype('uint8'),graph_indices=np.arange(start,end))
        synchronize(torch.device('cuda'))
        if checkpoint_hashes(path)!=identity['checkpoint_sha256']:raise ValueError('Snapshot weights changed')
        (directory/'summary.json').write_text(json.dumps(dict(status='complete',step=step,graphs=len(tasks),
            elapsed_seconds=perf_counter()-began,weights_unchanged=True,
            scope='Frozen count-free executor; final timer composition is diagnostic, not its historical controller'),indent=2)+'\n')
        del model;torch.cuda.empty_cache()


if __name__=='__main__':main()

"""Tighten the existing countdown fit without increasing count/depth exposure."""
import argparse
import csv
import json
from pathlib import Path

import torch
from rich.console import Console

from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.eval.pointer_task import sha256_file
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.controller import ControllerConfig, export_checkpoint, select_graphs
from scripts.training.controller_cache import prompt_features
from scripts.training.data import read_tasks
from scripts.training.number_reader import refine_countdown


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/controller_number_precision.json'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    settings=json.loads(args.config.read_text())
    config=ControllerConfig(**json.loads(Path(settings['training_config']).read_text()));config.validate()
    source,output=Path(settings['source']),Path(settings['output'])
    if args.dry_run:
        Console().print(f'Preview: same {len(config.requested_counts)} labels, {config.training_loops} loops, two fitted scalars; no loading/writes.');return
    if output.exists():
        raise ValueError('Countdown output exists; refuse overwrite')
    before=checkpoint_hashes(source)
    spec=json.loads((source/'recurrent_config.json').read_text())
    prior=spec['controller_training']
    original_source=Path(prior['source'])
    original_spec=json.loads((original_source/'recurrent_config.json').read_text())
    original_run=json.loads((original_source.parent/'run.json').read_text())
    if (spec['executor']['controller_kind']!='shared_number' or prior['trained_depths']!=config.requested_counts
            or prior['training_loops']!=config.training_loops or original_spec['controller_training']['config']!=config.to_dict()
            or sha256_file(Path(config.data)/'train.jsonl')!=original_run['train_data_sha256']):
        raise ValueError('Training exposure/source differs from original controller')
    tasks=select_graphs(read_tasks(Path(config.data)/'train.jsonl','train'),config.train_graphs,config.requested_counts,config.seed)
    with (original_source.parent/'train_panel.csv').open() as handle:
        if [r['example_id'] for r in csv.DictReader(handle)]!=[t.example_id for t in tasks]:
            raise ValueError('Original training graph/count selection changed')
    torch.set_num_threads(4)
    model,tokenizer,loaded=load_recurrent_checkpoint(source,device='cpu')
    model.eval().requires_grad_(False)
    context=prompt_features(model,tokenizer,loaded['token_ids'],tasks[:len(config.requested_counts)],config.extraction_batch_size)
    fit=refine_countdown(model.completion_head,context,config.requested_counts,config.training_loops)
    provenance={'source':str(source),'source_checkpoint_sha256':before['adapter_model.pt'],
        'trained_depths':config.requested_counts,'training_loops':config.training_loops,
        'optimization_scope':'countdown_cell_only','config':settings,'fit':fit,
        'selection':'One predetermined training-label fit; no stress/graph benchmark selection'}
    export_checkpoint(source,output/'best',model.completion_head,provenance)
    old=torch.load(source/'adapter_model.pt',weights_only=True)
    new=torch.load(output/'best/adapter_model.pt',weights_only=True)
    if set(old)!=set(new):raise ValueError('Export tensor names changed')
    preserved=[k for k in old if not k.startswith('completion_head.cell.')]
    for k in preserved:torch.testing.assert_close(old[k],new[k],atol=0,rtol=0)
    if checkpoint_hashes(source)!=before:raise ValueError('Source files changed')
    (output/'summary.json').write_text(json.dumps({'status':'complete',**provenance,
        'unchanged_non_cell_tensors':len(preserved),'source_inference_sha256':before,
        'exported_inference_sha256':checkpoint_hashes(output/'best'),
        'config_sha256':sha256_file(args.config)},indent=2)+'\n')
    Console().print(f'Fitted only cell · gain {fit["learned_gain"]:.9g} · offset {fit["learned_offset"]:.9g} · training MSE {fit["deployed_mse"]:.3g}')


if __name__=='__main__':main()

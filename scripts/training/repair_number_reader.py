"""Replace only a frozen controller's initializer, retaining its training range."""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from rich.console import Console

from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.eval.pointer_task import sha256_file
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.interfaces import SharedNumberController
from scripts.training.controller import ControllerConfig, export_checkpoint, select_graphs
from scripts.training.controller_cache import prompt_features
from scripts.training.data import read_tasks
from scripts.training.number_reader import fit_number_reader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/controller_number_reader.json'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    training = ControllerConfig(**json.loads(Path(settings['training_config']).read_text())); training.validate()
    source, output = Path(settings['source']), Path(settings['output'])
    console = Console()
    console.print(f'[cyan]Reader-only repair · same {len(training.requested_counts)} count labels · maximum {max(training.requested_counts)} · inherited {training.training_loops}-loop supervision[/cyan]')
    if args.dry_run:
        console.print('Preview only: no model/data loading, fitting or writes.'); return
    if output.exists():
        raise ValueError('Reader output exists; refuse overwrite')
    before = checkpoint_hashes(source)
    spec = json.loads((source / 'recurrent_config.json').read_text())
    prior = spec['controller_training']
    if (spec['executor'].get('controller_kind') != 'affine_suffix'
            or prior['trained_depths'] != training.requested_counts
            or prior['training_loops'] != training.training_loops
            or prior['config'] != training.to_dict()):
        raise ValueError('Source architecture or actual training config differs from the unchanged-range control')
    data = Path(training.data)
    original_run = json.loads((source.parent / 'run.json').read_text())
    if sha256_file(data / 'train.jsonl') != original_run['train_data_sha256']:
        raise ValueError('Training data differs from original controller fit')
    tasks = select_graphs(read_tasks(data / 'train.jsonl', 'train'), training.train_graphs,
                          training.requested_counts, training.seed)
    with (source.parent / 'train_panel.csv').open() as handle:
        original_ids = [row['example_id'] for row in csv.DictReader(handle)]
    if original_ids != [t.example_id for t in tasks]:
        raise ValueError('Training graph/count panel differs from original controller')
    torch.set_num_threads(4)
    model, tokenizer, loaded_spec = load_recurrent_checkpoint(source, device='cpu')
    model.eval().requires_grad_(False)
    old_head = model.completion_head
    head = SharedNumberController(model.config.hidden_size)
    head.cell.load_state_dict(old_head.cell.state_dict())
    head.readout.load_state_dict(old_head.readout.state_dict())
    model.completion_head = head
    # Count-only input is graph invariant. Fit one graph's exact existing variants.
    fit_tasks = tasks[:len(training.requested_counts)]
    context = prompt_features(model, tokenizer, loaded_spec['token_ids'], fit_tasks, training.extraction_batch_size)
    final_context = prompt_features(model, tokenizer, loaded_spec['token_ids'], tasks[-len(fit_tasks):], training.extraction_batch_size)
    torch.testing.assert_close(context, final_context, atol=0, rtol=0)
    fit = fit_number_reader(head.context, context, training.requested_counts)
    provenance = {'optimization_scope': 'number_reader_only', 'source_checkpoint_sha256': before['adapter_model.pt'],
        'source': str(source), 'trained_depths': prior['trained_depths'], 'training_loops': prior['training_loops'],
        'config': settings, 'fit': fit, 'selection': 'Single predetermined fit; no validation/benchmark selection'}
    export_checkpoint(source, output / 'best', head, provenance)
    original = torch.load(source / 'adapter_model.pt', map_location='cpu', weights_only=True)
    exported = torch.load(output / 'best/adapter_model.pt', map_location='cpu', weights_only=True)
    kept = {k for k in original if not k.startswith('completion_head.context.')}
    if kept != {k for k in exported if not k.startswith('completion_head.context.')}:
        raise ValueError('Non-reader tensor set changed')
    for name in kept:
        torch.testing.assert_close(original[name], exported[name], atol=0, rtol=0)
    if checkpoint_hashes(source) != before:
        raise ValueError('Source inference files changed')
    (output / 'summary.json').write_text(json.dumps({'status': 'complete', **provenance,
        'source_inference_sha256': before, 'exported_inference_sha256': checkpoint_hashes(output / 'best'),
        'unchanged_non_reader_tensors': len(kept), 'training_data_sha256': original_run['train_data_sha256'],
        'original_train_panel_sha256': sha256_file(source.parent / 'train_panel.csv'),
        'config_sha256': sha256_file(args.config), 'command': sys.argv}, indent=2) + '\n')
    console.print(f'[green]Reader fitted[/green] · learned gain {fit["learned_gain"]:g} · train max error {fit["training_max_absolute_error"]:g} · {len(kept)} other tensors unchanged')
    console.print(f'Portable checkpoint: {output / "best"}')


if __name__ == '__main__':
    main()

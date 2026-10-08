"""Prepare broader count exposure without retraining or deeply unrolling Qwen."""
import argparse
import json
from pathlib import Path

import torch
from rich.console import Console

from scripts.eval.controller_diagnostic import progress_bar
from scripts.eval.pointer_task import sha256_file
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.controller import ControllerConfig, select_graphs
from scripts.training.controller_cache import prompt_features, pair_prefixes, check_native_prefix
from scripts.training.data import read_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/controller_prefix.json'))
    parser.add_argument('--features-cache', type=Path, default=Path('models/stage1_pointer/controller-seed61/features.pt'))
    parser.add_argument('--output', type=Path, default=Path('models/stage1_pointer/controller-prefix-features.pt'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = ControllerConfig(**json.loads(args.config.read_text())); config.validate()
    if not config.requested_counts or not config.training_loops:
        raise ValueError('Need explicit broader counts and a training loop budget')
    console = Console()
    console.print(f'Frozen prompt cache · {config.train_graphs} graphs × {len(config.requested_counts)} counts · {config.training_loops} reused R loops')
    if args.dry_run:
        console.print('Preview only: no loading or writes.'); return
    if args.output.exists():
        raise ValueError('Cache output exists; choose a new path')
    torch.set_num_threads(4)
    cache = torch.load(args.features_cache, weights_only=True, map_location='cpu')
    original_hash = sha256_file(args.features_cache)
    source, data = Path(config.source), Path(config.data)
    spec = json.loads((source / 'recurrent_config.json').read_text())
    identity = {key: sha256_file(path) for key, path in {
        'source_sha256': source / 'adapter_model.pt', 'source_config_sha256': source / 'recurrent_config.json',
        'train_data_sha256': data / 'train.jsonl', 'validation_data_sha256': data / 'validation.jsonl'}.items()}
    if cache['format'] != 'controller-training-features-v1' or any(cache['identity'][k] != v for k, v in identity.items()):
        raise ValueError('Original cache source/data differ')
    raw_train = read_tasks(data / 'train.jsonl', 'train')
    original_graphs = len(cache['identity']['train_ids']) // len(spec['train_depths'])
    original_tasks = select_graphs(raw_train, original_graphs, spec['train_depths'], config.seed)
    if [t.example_id for t in original_tasks] != cache['identity']['train_ids']:
        raise ValueError('Original graph selection differs')
    tasks = select_graphs(raw_train, config.train_graphs, config.requested_counts, config.seed)
    validation = select_graphs(read_tasks(data / 'validation.jsonl', 'validation'), config.validation_graphs,
                              list(range(1, config.evaluation_max_depth + 1)), config.seed + 1)
    if ([t.example_id for t in validation] != cache['identity']['validation_ids'] or
            {t.mapping_sha256 for t in tasks} & {t.mapping_sha256 for t in validation}):
        raise ValueError('Validation cache selection differs or graphs overlap')
    train_features = pair_prefixes(original_tasks, cache['features']['train'], tasks, config.training_loops)
    del cache['features']['train'], raw_train
    model, tokenizer, loaded_spec = load_recurrent_checkpoint(source, device=config.device)
    model.config._attn_implementation = 'sdpa'; model.eval().requires_grad_(False)
    with progress_bar() as progress:
        bar = progress.add_task('Cache frozen full-prompt P features', total=len(tasks))
        context = prompt_features(model, tokenizer, loaded_spec['token_ids'], tasks, config.extraction_batch_size,
                                  progress=lambda done: progress.update(bar, completed=done))
    train_features['context'] = context
    # Independent of outcomes: first graph, extrema and a central requested count.
    chosen = [g * len(config.requested_counts) + i for g in range(min(4, config.train_graphs))
              for i in sorted({0, len(config.requested_counts) // 2, len(config.requested_counts) - 1})]
    error = check_native_prefix(model, tokenizer, loaded_spec['token_ids'], [tasks[i] for i in chosen],
                                context[chosen], train_features['working'][chosen])
    val_error = check_native_prefix(model, tokenizer, loaded_spec['token_ids'], validation[:4],
        cache['features']['validation']['context'][:4], cache['features']['validation']['working'][:4])
    identity.update(train_ids=[t.example_id for t in tasks], validation_ids=[t.example_id for t in validation])
    result = {**cache, 'identity': identity, 'features': {'train': train_features, 'validation': cache['features']['validation']},
              'replay_error': {'train': None, 'validation': cache['replay_error']['validation']},
              'extraction': {**cache['extraction'], 'prefix_cache_source': str(args.features_cache),
                  'prefix_cache_sha256': original_hash, 'requested_counts': config.requested_counts,
                  'training_loops': config.training_loops, 'native_prefix_max_error': error,
                  'native_validation_max_error': val_error}}
    if sha256_file(source / 'adapter_model.pt') != identity['source_sha256'] or sha256_file(args.features_cache) != original_hash:
        raise RuntimeError('Frozen source/cache changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    args.output.with_suffix('.json').write_text(json.dumps({k: v for k, v in result.items() if k not in ('head', 'features')}, indent=2) + '\n')
    console.print(f'[green]Cache validated[/green] · native R error {error:g} · {args.output}')


if __name__ == '__main__':
    main()

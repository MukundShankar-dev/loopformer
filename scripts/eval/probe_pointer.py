"""Run fixed small suffix/depth-cue probes on a retained recurrent checkpoint."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from rich.console import Console
from rich.progress import track

from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_probes import inspect_example, task_slice
from scripts.eval.pointer_task import load_examples, sha256_file
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.data import encode_tasks


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--limit', type=int, default=32, help='Fixed prefix, selected before observing failures')
    parser.add_argument('--restart-after', type=int, default=6)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.restart_after, args.threads) < 1 or args.output.exists():
        parser.error('Positive counts and a new output directory required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS unavailable')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
    tasks = load_examples(args.data, args.limit)
    if any(t.task_depth <= args.restart_after for t in tasks):
        parser.error('Every selected task must be deeper than --restart-after')
    console = Console()
    with console.status('Loading checkpoint for small paired probes…'):
        model, tokenizer, spec = load_recurrent_checkpoint(args.model, device=args.device, download=args.download)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'command': [sys.executable, '-m', 'scripts.eval.probe_pointer', *sys.argv[1:]],
                'checkpoint': str(args.model.resolve()), 'adapter_sha256': sha256_file(args.model/'adapter_model.pt'),
                'checkpoint_metadata_sha256': sha256_file(args.model/'recurrent_config.json'),
                'data_sha256': sha256_file(args.data), 'example_ids': [t.example_id for t in tasks],
                'device': args.device, 'dtype': 'float32', 'batch_size': 1, 'seed': args.seed,
                'restart_after': args.restart_after, 'torch': torch.__version__, 'tokenizer_sha256': sha256_file(args.model/'tokenizer.json'),
                'source_sha256': {str(p.relative_to(root)): sha256_file(p) for folder in ('eval', 'training', 'recurrent_qwen', 'dataset') for p in (root/'scripts'/folder).glob('*.py')},
                'semantics': 'Paired changed-prompt diagnostics, not causal isolation of hidden state. Prefix selected before model outputs. Hidden summaries include h0->h1; h0 has no target. Not a speed benchmark.'}
    summary = args.output/'summary.json'; summary.write_text(json.dumps(metadata, indent=2)+'\n')
    token_map = dict(zip(spec['symbols'], spec['token_ids'], strict=True))
    all_rows, pairs, changed_tasks = [], [], []
    for task in track(tasks, description='Probing', console=console):
        variants = {'original': task, 'suffix': task_slice(task, args.restart_after, task.task_depth),
                    'shorter_depth': task_slice(task, 0, task.task_depth - 1)}
        observations = {}
        for name, variant in variants.items():
            item = encode_tasks([variant], tokenizer, token_map, model.config.max_position_embeddings)[0]
            observations[name] = inspect_example(model, item, spec['token_ids'], tokenizer.pad_token_id)
            for row in observations[name]:
                all_rows.append({'example_id': task.example_id, 'variant': name, 'task_depth': variant.task_depth, **row})
            changed_tasks.append({'source_id': task.example_id, 'variant': name, 'task': variant.to_dict()})
        original, suffix, shorter = (observations[n] for n in variants)
        for i, fresh in enumerate(suffix):
            old = original[args.restart_after + i]
            pairs.append({'example_id': task.example_id, 'comparison': 'suffix', 'original_loop': old['loop'],
                          'variant_loop': fresh['loop'], 'original_correct': old['correct'], 'variant_correct': fresh['correct'],
                          'same_prediction': old['prediction'] == fresh['prediction'],
                          'original_prefix_correct': all(r['correct'] for r in original[:old['loop']-1])})
        for old, changed in zip(original, shorter):
            pairs.append({'example_id': task.example_id, 'comparison': 'depth_cue', 'original_loop': old['loop'],
                          'variant_loop': changed['loop'], 'original_correct': old['correct'], 'variant_correct': changed['correct'],
                          'same_prediction': old['prediction'] == changed['prediction'],
                          'original_prefix_correct': all(r['correct'] for r in original[:old['loop']-1])})
    write_csv(args.output/'states.csv', all_rows); write_csv(args.output/'pairs.csv', pairs)
    tasks_path = args.output/'tasks.jsonl'
    tasks_path.write_text(''.join(json.dumps(x)+'\n' for x in changed_tasks))
    counts = {}
    for comparison in ('suffix', 'depth_cue'):
        cohort = [r for r in pairs if r['comparison'] == comparison]
        first = [r for r in cohort if r['original_prefix_correct'] and not r['original_correct']]
        counts[comparison] = {'paired_steps': len(cohort), 'prediction_changes': sum(not r['same_prediction'] for r in cohort),
                              'original_first_errors': len(first), 'variant_correct_at_original_first_error': sum(r['variant_correct'] for r in first)}
    metadata.update(status='complete', counts=counts, tasks_sha256=sha256_file(tasks_path))
    summary.write_text(json.dumps(metadata, indent=2)+'\n')
    console.print(f'[green]Saved[/green] paired outcomes and hidden-state summaries to {args.output}')


if __name__ == '__main__':
    main()

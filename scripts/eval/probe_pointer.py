"""Run fixed small suffix/depth-cue probes on a retained recurrent checkpoint."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table

from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_probes import inspect_example, task_slice, rule_prefix_length
from scripts.dataset.pointer import render_prompt
from scripts.dataset.symbols import validate_prompt_tokens
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
    parser.add_argument('--controls', action='store_true', help='Add original-Steps suffix and rule-refresh/no-op controls; allow shallow tasks')
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
    if not args.controls and any(t.task_depth <= args.restart_after for t in tasks):
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
                'controls': args.controls, 'restart_after': args.restart_after, 'torch': torch.__version__, 'tokenizer_sha256': sha256_file(args.model/'tokenizer.json'),
                'source_sha256': {str(p.relative_to(root)): sha256_file(p) for folder in ('eval', 'training', 'recurrent_qwen', 'dataset') for p in (root/'scripts'/folder).glob('*.py')},
                'semantics': 'Paired changed-prompt diagnostics, not causal isolation of hidden state. Prefix selected before model outputs. Hidden summaries include h0->h1; h0 has no target. Not a speed benchmark.'}
    summary = args.output/'summary.json'; summary.write_text(json.dumps(metadata, indent=2)+'\n')
    token_map = dict(zip(spec['symbols'], spec['token_ids'], strict=True))
    all_rows, pairs, changed_tasks = [], [], []
    for task in track(tasks, description='Probing', console=console):
        variants = {'original': task}
        if task.task_depth > args.restart_after:
            variants.update(suffix=task_slice(task, args.restart_after, task.task_depth),
                            shorter_depth=task_slice(task, 0, task.task_depth - 1))
            if args.controls:
                variants['suffix_original_steps'] = variants['suffix']
        if args.controls:
            variants.update(rule_refresh=task, rule_noop=task)
        observations = {}
        for name, variant in variants.items():
            item = encode_tasks([variant], tokenizer, token_map, model.config.max_position_embeddings)[0]
            prompt = variant.prompt
            if name == 'suffix_original_steps':
                # Keep the reference task valid; record the deliberately mismatched
                # displayed Steps separately from its execution/scoring horizon.
                prompt = render_prompt(variant.mapping, variant.initial_state, task.task_depth)
                validate_prompt_tokens(tokenizer, prompt, token_map)
                item.input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(item.input_ids) > model.config.max_position_embeddings:
                    raise ValueError('Altered prompt exceeds model context')
            prefix = rule_prefix_length(tokenizer, prompt) if args.controls else None
            control = {'rule_refresh': 'refresh', 'rule_noop': 'noop'}.get(name)
            observations[name] = inspect_example(model, item, spec['token_ids'], tokenizer.pad_token_id,
                rule_control=control, prefix_tokens=prefix or 0, restart_after=args.restart_after)
            for row in observations[name]:
                all_rows.append({'example_id': task.example_id, 'variant': name, 'task_depth': variant.task_depth, **row})
            changed_tasks.append({'source_id': task.example_id, 'variant': name, 'task': variant.to_dict(),
                                  'model_prompt': prompt, 'input_ids': item.input_ids,
                                  'input_tokens': len(item.input_ids), 'answer_position': len(item.input_ids)-1,
                                  'rule_prefix_tokens': prefix, 'scoring_loops': len(item.targets),
                                  'displayed_steps': task.task_depth if name == 'suffix_original_steps' else variant.task_depth})
        comparisons = [('original', 'suffix', 'suffix', args.restart_after),
                       ('original', 'shorter_depth', 'depth_cue', 0),
                       ('original', 'suffix_original_steps', 'suffix_original_steps', args.restart_after),
                       ('suffix', 'suffix_original_steps', 'suffix_steps_cue', 0),
                       ('original', 'rule_refresh', 'rule_refresh', 0),
                       ('original', 'rule_noop', 'rule_noop', 0)]
        for base_name, name, comparison, offset in comparisons:
            if name not in observations:
                continue
            baseline = observations[base_name]
            for i, changed in enumerate(observations[name]):
                old = baseline[offset+i]
                if old['target'] != changed['target']:
                    raise ValueError('Paired transitions have different reference targets')
                pairs.append({'example_id': task.example_id, 'comparison': comparison,
                              'baseline_variant': base_name, 'original_loop': old['loop'],
                              'variant_loop': changed['loop'], 'original_correct': old['correct'],
                              'variant_correct': changed['correct'],
                              'same_prediction': old['prediction'] == changed['prediction'],
                              'original_prefix_correct': all(r['correct'] for r in baseline[:old['loop']-1])})
        if args.controls:
            original = observations['original']
            if original != observations['rule_noop']:
                raise ValueError('No-op control changed observations; stop and inspect execution')
            if original[:args.restart_after] != observations['rule_refresh'][:args.restart_after]:
                raise ValueError('Rule refresh changed the untouched initial loops')
    write_csv(args.output/'states.csv', all_rows); write_csv(args.output/'pairs.csv', pairs)
    tasks_path = args.output/'tasks.jsonl'
    tasks_path.write_text(''.join(json.dumps(x)+'\n' for x in changed_tasks))
    counts = {}
    for comparison in sorted({r['comparison'] for r in pairs}):
        cohort = [r for r in pairs if r['comparison'] == comparison]
        first = [r for r in cohort if r['original_prefix_correct'] and not r['original_correct']]
        counts[comparison] = {'paired_steps': len(cohort), 'prediction_changes': sum(not r['same_prediction'] for r in cohort),
                              'original_first_errors': len(first), 'variant_correct_at_original_first_error': sum(r['variant_correct'] for r in first)}
    trajectories = {}
    for row in all_rows:
        trajectories.setdefault((row['variant'], row['example_id'], row['task_depth']), []).append(row['correct'])
    example_rows = []
    for (variant, key, depth), values in trajectories.items():
        first = next((i for i, correct in enumerate(values, 1) if not correct), None)
        example_rows.append({'example_id': key, 'variant': variant, 'scoring_depth': depth,
                             'complete_trajectory': all(values), 'correct_steps': sum(values),
                             'first_error_loop': first if first is not None else '',
                             'correct_prefix_length': first-1 if first is not None else depth})
    write_csv(args.output/'examples.csv', example_rows)
    quality = {}
    for variant in sorted({key[0] for key in trajectories}):
        quality[variant] = {}
        for depth in sorted({key[2] for key in trajectories if key[0] == variant}):
            values = [v for k, v in trajectories.items() if k[0] == variant and k[2] == depth]
            quality[variant][str(depth)] = {'examples': len(values), 'complete_trajectories': sum(all(v) for v in values),
                'correct_steps': sum(sum(v) for v in values), 'steps': sum(len(v) for v in values)}
    for comparison, counts_row in counts.items():
        cohort = [r for r in pairs if r['comparison'] == comparison]
        counts_row['correct_to_wrong_steps'] = sum(r['original_correct'] and not r['variant_correct'] for r in cohort)
        counts_row['wrong_to_correct_steps'] = sum(not r['original_correct'] and r['variant_correct'] for r in cohort)
        grouped = {}
        for r in cohort:
            grouped.setdefault(r['example_id'], []).append(r)
        counts_row['lost_complete_aligned_segments'] = sum(all(r['original_correct'] for r in rs) and not all(r['variant_correct'] for r in rs) for rs in grouped.values())
        counts_row['gained_complete_aligned_segments'] = sum(not all(r['original_correct'] for r in rs) and all(r['variant_correct'] for r in rs) for rs in grouped.values())
    metadata.update(status='complete', counts=counts, quality_by_depth=quality, tasks_sha256=sha256_file(tasks_path))
    summary.write_text(json.dumps(metadata, indent=2)+'\n')
    table = Table(title='Complete scored trajectories (suffix tasks have shorter horizons)')
    table.add_column('Variant'); table.add_column('Correct / examples', justify='right')
    for variant, depths in quality.items():
        table.add_row(variant, f"{sum(v['complete_trajectories'] for v in depths.values())} / {sum(v['examples'] for v in depths.values())}")
    console.print(table)
    console.print(f'[green]Saved[/green] paired outcomes and hidden-state summaries to {args.output}')


if __name__ == '__main__':
    main()

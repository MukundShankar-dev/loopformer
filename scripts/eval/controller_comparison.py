"""Compare all controller arms using shared frozen observations and existing stop scoring."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from time import perf_counter

import torch
from rich.console import Console
from rich.table import Table

from scripts.dataset.pointer import PointerExample, validate_example
from scripts.eval.controller_diagnostic import progress_bar
from scripts.eval.controller_features import extract_features
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file
from scripts.eval.tracking import add_tracking_arguments, report_saved_run
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.controller import evaluate_controller
from scripts.training.data import read_tasks


def paired_summary(results: list[dict]) -> list[dict]:
    """Report every paired seed; never choose an arm or seed on unseen counts."""
    keys = sorted({(r['split'], r['cohort']) for r in results})
    rows = []
    for split, cohort in keys:
        group = [r for r in results if (r['split'], r['cohort']) == (split, cohort)]
        seeds = sorted({r['optimizer_seed'] for r in group})
        deltas = []
        for seed in seeds:
            arms = {r['arm']: r for r in group if r['optimizer_seed'] == seed}
            if set(arms) != {'stop_only', 'remaining'} or sum(r['optimizer_seed'] == seed for r in group) != 2:
                raise ValueError('Comparison needs both arms for every seed')
            deltas.append(arms['remaining']['exact_stop'] - arms['stop_only']['exact_stop'])
        rows.append({'split': split, 'cohort': cohort, 'seeds': len(seeds),
                     'stop_only_mean': mean(r['exact_stop'] for r in group if r['arm'] == 'stop_only'),
                     'remaining_mean': mean(r['exact_stop'] for r in group if r['arm'] == 'remaining'),
                     'paired_exact_stop_delta_mean': mean(deltas),
                     'paired_exact_stop_delta_sd': stdev(deltas) if len(deltas) > 1 else None,
                     'paired_exact_stop_delta_min': min(deltas), 'paired_exact_stop_delta_max': max(deltas)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--deep-features-cache', type=Path)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'comparison.json').read_text())
    entries = manifest['runs']
    if not entries:
        raise ValueError('No comparison runs')
    configs = [json.loads((Path(r['path']) / 'config.json').read_text()) for r in entries]
    base = configs[0]
    for config in configs:
        differences = {key for key in base if config[key] != base[key]}
        if differences - {'optimizer_seed', 'remaining_loss_weight'}:
            raise ValueError(f'Unmatched training configurations: {differences}')
    device = base['device']
    cache_path = Path(manifest['features_cache'])
    cache_hash = sha256_file(cache_path)
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    if cache.get('format') != 'controller-training-features-v1':
        raise ValueError('Need a controller training cache')
    # Keep only validation observations after reading the original shared cache.
    validation_features = cache['features'].pop('validation')
    cache['features'].clear()
    validation_path = Path(entries[0]['path']) / 'validation_tasks.jsonl'
    validation_tasks = [PointerExample(**json.loads(line)) for line in validation_path.read_text().splitlines()]
    for task in validation_tasks:
        validate_example(task)
        if task.split != 'validation':
            raise ValueError('Need development validation tasks')
    if [t.example_id for t in validation_tasks] != cache['identity']['validation_ids']:
        raise ValueError('Validation tasks differ from cached panel order')
    tasks_hash = sha256_file(validation_path)
    summaries = []
    for entry in entries:
        run = Path(entry['path'])
        metadata = json.loads((run / 'run.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        if (summary['status'] != 'complete' or metadata['features_sha256'] != cache_hash
                or metadata['source_checkpoint_sha256'] != cache['identity']['source_sha256']
                or sha256_file(run / 'validation_tasks.jsonl') != tasks_hash
                or metadata['validation_tasks_sha256'] != tasks_hash):
            raise ValueError(f'Mismatched cache/source/validation panel: {run}')
        summaries.append(summary)
    output = args.output or Path('eval/pointer_diagnostics') / args.run.name
    output.mkdir(parents=True, exist_ok=False)
    console = Console()
    torch.set_num_threads(4)
    panels = {'validation': (validation_tasks, validation_features)}
    deep_metadata = None
    if not args.validation_only:
        data = Path(base['data']) / 'depth_test.jsonl'
        tasks = read_tasks(data, 'depth_test')
        if max(t.task_depth for t in tasks) > 64:
            raise ValueError('This comparison has a fixed 64-loop budget')
        identity = {'source_sha256': cache['identity']['source_sha256'], 'data_sha256': sha256_file(data),
                    'source_config_sha256': cache['identity']['source_config_sha256'],
                    'loops': 64, 'device': device, 'precision': 'float32', 'attention': 'sdpa',
                    'batch_size': base['extraction_batch_size'], 'ids': [t.example_id for t in tasks]}
        began = perf_counter()
        if args.deep_features_cache:
            deep = torch.load(args.deep_features_cache, map_location='cpu', weights_only=True)
            if deep['identity'] != identity:
                raise ValueError('Deep cache source/data/settings differ')
            deep_path = args.deep_features_cache
        else:
            source = Path(base['source'])
            if (sha256_file(source / 'adapter_model.pt') != identity['source_sha256']
                    or sha256_file(source / 'recurrent_config.json') != identity['source_config_sha256']):
                raise ValueError('Frozen source differs from training cache')
            model, tokenizer, spec = load_recurrent_checkpoint(source, device=device)
            model.config._attn_implementation = 'sdpa'
            with progress_bar() as progress:
                bar = progress.add_task('Cache frozen deep observations once', total=len(tasks))
                values, _, error = extract_features(model, tokenizer, spec['token_ids'], tasks, 64,
                    base['extraction_batch_size'], progress=lambda done: progress.update(bar, completed=done))
            deep = {'identity': identity, 'features': {k: values[k] for k in ('context', 'working', 'predictions')},
                    'live_replay_max_logit_error': error}
            del model, tokenizer, values
            if device == 'cuda':
                torch.cuda.empty_cache()
            deep_path = output / 'deep_features.pt'
            torch.save(deep, deep_path)
        deep_metadata = {'cache': str(deep_path), 'sha256': sha256_file(deep_path),
                         'identity': identity, 'live_replay_max_logit_error': deep['live_replay_max_logit_error'],
                         'load_or_extraction_seconds': perf_counter() - began}
        panels['depth_test'] = tasks, deep['features']
    results, all_metrics = [], {}
    for entry, summary in zip(entries, summaries, strict=True):
        run = Path(entry['path'])
        head = RecurrentController(cache['width'], cache['controller_width']).to(device)
        head.load_state_dict(torch.load(run / 'best_controller.pt', map_location=device, weights_only=True))
        saved = torch.load(run / 'best_remaining_readout.pt', map_location=device, weights_only=True)
        if saved['step'] != summary['best_step'] or saved['scale'] != base['remaining_scale']:
            raise ValueError('Readout does not belong to selected controller')
        readout = torch.nn.Linear(cache['controller_width'], 1).to(device)
        readout.load_state_dict(saved['state_dict'])
        child = output / entry['name']
        child.mkdir()
        all_metrics[entry['name']] = {}
        spec = json.loads((run / 'best/recurrent_config.json').read_text())
        for split, (tasks, features) in panels.items():
            console.print(f"[cyan]{entry['name']}[/cyan] · {split}")
            traces = []
            metrics, decisions = evaluate_controller(head, features, tasks, spec['train_depths'], device,
                remaining_readout=readout, remaining_scale=saved['scale'], traces=traces)
            # Full question-level decisions; bounded per-loop trace sample chosen
            # by panel order, not correctness. All aggregates use the full panel.
            graph_sample = set(list(dict.fromkeys(t.mapping_sha256 for t in tasks))[:8])
            trace_ids = {t.example_id for t in tasks if t.mapping_sha256 in graph_sample}
            write_csv(child / f'{split}_remaining_traces.csv', [r for r in traces if r['example_id'] in trace_ids])
            write_csv(child / f'{split}_decisions.csv', decisions)
            all_metrics[entry['name']][split] = metrics
            results.extend({'run': entry['name'], 'arm': entry['arm'], 'optimizer_seed': entry['optimizer_seed'],
                            'best_step': summary['best_step'], 'split': split, 'cohort': cohort, **values}
                           for cohort, values in metrics.items())
    # A predeclared small native check guards exported inference against replay.
    # It never picks the winning seed or tunes a threshold on unseen requests.
    native_checks = []
    for split, (tasks, _) in panels.items():
        (output / f'{split}_native_tasks.jsonl').write_text(
            ''.join(json.dumps(t.to_dict()) + '\n' for t in tasks[:16]))
    first_seed = min(entry['optimizer_seed'] for entry in entries)
    for entry in entries:
        if entry['optimizer_seed'] != first_seed:
            continue
        child = output / entry['name']
        for split, (tasks, features) in panels.items():
            native = child / f'{split}_native_check'
            budget = features['working'].shape[1]
            subprocess.run([sys.executable, '-u', '-m', 'scripts.eval.loop_test',
                '--model', str(Path(entry['path']) / 'best'), '--device', device,
                '--data', str(output / f'{split}_native_tasks.jsonl'), '--limit', '16',
                '--loops', str(budget), '--stop-policy', 'completion', '--stop-threshold', '0.5',
                '--attention', 'sdpa', '--output', str(native), '--wandb-mode', 'disabled'], check=True)
            with (child / f'{split}_decisions.csv').open() as handle:
                cached_decisions = {r['example_id']: r for r in csv.DictReader(handle)}
            with (native / 'decisions.csv').open() as handle:
                native_rows = list(csv.DictReader(handle))
            checked = 0
            for row in native_rows:
                cached = cached_decisions[row['example_id']]
                for field in ('prediction', 'executed_loops', 'exact_stop', 'joint_success'):
                    if row[field] != cached[field]:
                        raise ValueError(f"Native/replay mismatch: {entry['name']} {row['example_id']} {field}")
                if row['cap_fallback'] != cached['missing_stop']:
                    raise ValueError('Native/replay missing-stop mismatch')
                checked += 1
            if checked != min(16, len(tasks)):
                raise ValueError('Native check did not cover its full predeclared panel')
            native_checks.append({'run': entry['name'], 'split': split, 'matched_questions': checked,
                                  'native_questions': len(native_rows), 'budget': budget, 'passed': True})
    paired = paired_summary(results)
    write_csv(output / 'comparison.csv', results)
    write_csv(output / 'paired_comparison.csv', paired)
    result = {'status': 'complete', 'mode': 'cached controller replay, not measured adaptive inference latency',
              'stop_threshold': .5, 'validation_tasks_sha256': tasks_hash, 'training_features_sha256': cache_hash,
              'source_checkpoint_sha256': cache['identity']['source_sha256'], 'comparison': manifest['comparison'], 'runs': entries,
              'selected_heads_sha256': {entry['name']: {
                  name: sha256_file(Path(entry['path']) / name)
                  for name in ('best_controller.pt', 'best_remaining_readout.pt')}
                  for entry in entries},
              'deep': deep_metadata, 'metrics': all_metrics, 'native_checks': native_checks,
              'trace_sampling': 'first eight graph identities in each panel; metrics/decisions use all questions',
              'source_sha256': {str(p): sha256_file(p)
                               for pattern in ('scripts/training/*.py', 'scripts/eval/*.py', 'scripts/recurrent_qwen/*.py')
                               for p in Path('.').glob(pattern)}}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    table = Table(title='Mean exact stopping across matched seeds')
    for label in ('Split', 'Counts', 'Stop only', 'Remaining', 'Paired change'):
        table.add_column(label)
    for row in paired:
        if not row['cohort'].startswith('depth_'):
            table.add_row(row['split'], row['cohort'], f"{row['stop_only_mean']:.1%}",
                          f"{row['remaining_mean']:.1%}", f"{row['paired_exact_stop_delta_mean']:+.1%}")
    console.print(table)
    report_saved_run(output, args)
    console.print(f'Saved comparison: {output}')


if __name__ == '__main__':
    main()

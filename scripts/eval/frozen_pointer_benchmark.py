"""Frozen finite-range pointer benchmark; compose shared inference and exact-stop scoring."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from scripts.dataset.benchmark import count_variant, exclusion_hashes, independent_graphs
from scripts.dataset.pointer import SYMBOLS
from scripts.eval.benchmark_metrics import (METRICS, clustered_summary, criteria_result, grouped_rows, quality_arrays)
from scripts.eval.controller_features import replay
from scripts.eval.controller_learning import stop_rows
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file, synchronize
from scripts.eval.tracking import add_tracking_arguments, report_saved_run
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.controller_cache import prompt_features
from scripts.training.data import collate, encode_tasks
from scripts.training.objective import forward_symbols


def checkpoint_hashes(path: Path) -> dict[str, str]:
    """All inference files; training optimizer state is not loaded by evaluation."""
    return {p.name: sha256_file(p) for p in sorted(path.iterdir())
            if p.is_file() and p.name != 'training_state.pt'}


def controller_panel(model, tokenizer, token_ids: list[int], tasks: list, metadata: list[dict],
                     maximum: int, cap: int) -> tuple[torch.Tensor, list[int | None], list[dict]]:
    """Use native suffix reading/update, verify graph invariance across every stratum."""
    representatives = []
    seen = set()
    for task, meta in zip(tasks, metadata, strict=True):
        key = (meta['dataset_seed'], meta['graph_mode'])
        if key not in seen:
            representatives.append(task); seen.add(key)
    variants = [count_variant(t, count) for t in representatives for count in range(1, maximum + 1)]
    context = prompt_features(model, tokenizer, token_ids, variants, 64)
    context = context.reshape(len(representatives), maximum, -1)
    torch.testing.assert_close(context, context[:1].expand_as(context), atol=0, rtol=0)
    device = next(model.parameters()).device
    # Affine timing ignores R observations; expanded zeros are not a state/count input.
    working = torch.zeros(1, 1, model.config.hidden_size, device=device).expand(maximum, cap, -1)
    with torch.no_grad():
        logits, initial, _ = replay(model.completion_head, context[0].to(device), working)
    logits = logits.cpu()
    selected = variants[:maximum]
    decisions = stop_rows(logits, torch.zeros(maximum, cap, dtype=torch.long), selected,
                          [{'example_id': t.example_id, 'depth': t.task_depth} for t in selected], 'counter')
    stops = [r['first_stop'] for r in decisions]
    rows = [{'depth': i + 1, 'initial_memory': float(initial[i, 0]), 'first_stop': stops[i],
             'executed_loops': r['executed_loops'], 'exact_stop': r['exact_stop'],
             'early_stop': r['early_stop'], 'late_stop': r['late_stop'],
             'missing_stop': r['missing_stop'], 'stop_probability': r['stop_probability']}
            for i, r in enumerate(decisions)]
    return logits, stops, rows


def executor_panel(model, tokenizer, token_ids: list[int], tasks: list, *, cap: int,
                   batch_size: int, console: Console) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """One count-free trajectory per graph, without retaining full hidden vectors."""
    device = next(model.parameters()).device
    items = encode_tasks(tasks, tokenizer, dict(zip(SYMBOLS, token_ids, strict=True)), 512)
    predictions, direct = [], []
    synchronize(device); began = perf_counter()
    with Progress(TextColumn('Frozen executor'), BarColumn(), TextColumn('{task.completed:.0f}/{task.total:.0f} graphs'),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress, torch.inference_mode():
        bar = progress.add_task('Executor', total=len(items))
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size], tokenizer.pad_token_id, str(device), cap)
            result, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'], token_ids, num_loops=cap)
            if not torch.isfinite(scores).all() or not torch.isfinite(result.stop_logits).all():
                raise FloatingPointError('Nonfinite frozen model readout')
            predictions.append(scores.argmax(-1).cpu())
            if result.state_logits is not None:
                direct.append(result.state_logits.argmax(-1).cpu())
            progress.update(bar, completed=min(start + batch_size, len(items)))
    synchronize(device)
    return torch.cat(predictions), torch.cat(direct) if direct else None, perf_counter() - began


def native_fidelity(config: dict, output: Path, tasks: list, metadata: list[dict], predictions: torch.Tensor,
                    controller_logits: torch.Tensor, device: str) -> dict:
    """Predeclared real stopped calls; require matching both successes and failures."""
    chosen, counters = [], {}
    for i, (task, meta) in enumerate(zip(tasks, metadata, strict=True)):
        key = (meta['dataset_seed'], meta['graph_mode'])
        count = counters.get(key, 0)
        if count < config['native_graphs_per_seed_mode']:
            chosen.extend((i, count_variant(task, depth)) for depth in config['native_counts'])
        counters[key] = count + 1
    selected = [t for _, t in chosen]
    expected = stop_rows(torch.stack([controller_logits[t.task_depth - 1] for _, t in chosen]),
        torch.stack([predictions[i] for i, _ in chosen]), selected,
        [{'example_id': t.example_id, 'depth': t.task_depth} for t in selected], 'benchmark')
    return verify_native_decisions(config, output, selected, expected, device)


def verify_native_decisions(config: dict, output: Path, selected: list, expected: list[dict],
                            device: str) -> dict:
    """Check arbitrary controller replay against the existing actual-stop evaluator.

    Unlike scalar controller_panel, this supports graph-dependent GRU decisions.
    Both successes and failures, including cap fallbacks, must match exactly.
    """
    path = output / 'native_tasks.jsonl'
    path.write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in selected))
    subprocess.run([sys.executable, '-u', '-m', 'scripts.eval.loop_test', '--model', config['model'],
        '--data', str(path), '--device', device, '--attention', 'sdpa', '--loops', str(config['safety_cap']),
        '--stop-policy', 'completion', '--stop-threshold', str(config['threshold']),
        '--output', str(output / 'native'), '--wandb-mode', 'disabled'], check=True)
    actual = {r['example_id']: r for r in csv.DictReader((output / 'native/decisions.csv').open())}
    if set(actual) != {t.example_id for t in selected}:
        raise ValueError('Native panel coverage differs')
    mismatch = []
    for row in expected:
        other = actual[row['example_id']]
        keys = ('prediction', 'executed_loops', 'exact_stop', 'early_stop', 'late_stop', 'joint_success')
        failed = [k for k in keys if str(row[k]) != other[k]]
        if str(row['missing_stop']) != other['cap_fallback']:
            failed.append('cap_fallback')
        if failed:
            mismatch.append({'example_id': row['example_id'], 'keys': failed})
    result = {'passed': not mismatch, 'questions': len(selected), 'mismatches': mismatch}
    (output / 'native_fidelity.json').write_text(json.dumps(result, indent=2) + '\n')
    if mismatch:
        raise ValueError('Native/reuse fidelity failed; do not accept benchmark metrics')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pointer_benchmark.json'))
    parser.add_argument('--output', type=Path, default=Path('eval/pointer_benchmark/frozen-20261008/independent'))
    parser.add_argument('--dataset-output', type=Path, default=Path('data/pointer/benchmark-seeds211-223-227'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--freeze', type=Path, default=Path('eval/pointer_benchmark/frozen-20261008/freeze.json'))
    parser.add_argument('--dry-run', action='store_true')
    add_tracking_arguments(parser)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config['threshold'] != .5 or config['safety_cap'] < config['max_depth'] or config['batch_size'] < 1
            or config['native_graphs_per_seed_mode'] < 1 or any(not 1 <= d <= config['max_depth'] for d in config['native_counts'])):
        raise ValueError('Invalid declared benchmark configuration')
    console = Console(); n = len(config['seeds']) * len(config['graph_modes']) * config['graphs_per_seed_mode']
    console.print(f'[bold cyan]Frozen benchmark · {n:,} new graphs × {config["max_depth"]} counts · safety cap {config["safety_cap"]}[/bold cyan]')
    if args.dry_run:
        console.print('Preview only: no loading, dataset reads, generation or writes.'); return
    if args.output.exists() or args.dataset_output.exists():
        raise ValueError('Benchmark or dataset output exists; refuse overwrite')
    model_path = Path(config['model']); before = checkpoint_hashes(model_path)
    frozen = json.loads(args.freeze.read_text())
    if (frozen['checkpoint'] != config['model'] or frozen['inference_sha256'] != before
            or frozen['threshold'] != config['threshold'] or frozen['safety_cap'] != config['safety_cap']
            or ('protocol_sha256' in frozen and frozen['protocol_sha256'] != sha256_file(args.config))):
        raise ValueError('Model or stopping policy differs from pre-test freeze')
    torch.set_num_threads(4); torch.manual_seed(239); torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True); args.dataset_output.mkdir(parents=True)
    source_files = ['scripts/eval/frozen_pointer_benchmark.py', 'scripts/eval/benchmark_metrics.py',
                    'scripts/dataset/benchmark.py', 'scripts/dataset/pointer.py', 'scripts/eval/controller_learning.py',
                    'scripts/recurrent_qwen/model.py', 'scripts/recurrent_qwen/interfaces.py']
    provenance = {'status': 'running', 'config': config, 'config_sha256': sha256_file(args.config),
        'checkpoint': config['model'], 'checkpoint_sha256': before,
        'source_sha256': {p: sha256_file(Path(p)) for p in source_files},
        'command': sys.argv, 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(), 'torch': torch.__version__, 'device': args.device,
        'dtype': 'float32', 'attention': 'sdpa', 'batch_size': config['batch_size'],
        'target_semantics': 'One R loop is one transition; exact first crossing at N required',
        'compute_scope': 'One forced count-free trajectory per graph; controller replay for every count; bounded native fidelity panel'}
    (args.output / 'summary.json').write_text(json.dumps(provenance, indent=2) + '\n')
    excluded = exclusion_hashes(Path(config['existing_data']))
    extra_exclusions = {}
    for filename in config.get('exclusion_files', []):
        path = Path(filename)
        extra_exclusions[filename] = sha256_file(path)
        for line in path.read_text().splitlines():
            excluded.add(json.loads(line)['mapping_sha256'])
    tasks, metadata = independent_graphs(config, excluded)
    graph_file = args.dataset_output / 'graphs.jsonl'
    graph_file.write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in tasks))
    manifest = {'generator': 'frozen-pointer-benchmark-v1', 'config': config, 'graphs': n,
        'queries': n * config['max_depth'], 'exclusion_graphs': len(excluded),
        'old_dataset_sha256': {p.name: sha256_file(p) for p in Path(config['existing_data']).glob('*.jsonl')},
        'graphs_sha256': sha256_file(graph_file), 'extra_exclusion_sha256': extra_exclusions, 'graph_metadata': metadata}
    (args.dataset_output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    model, tokenizer, spec = load_recurrent_checkpoint(model_path, device=args.device)
    model.eval().requires_grad_(False); model.config._attn_implementation = 'sdpa'
    if model.router is None or model.completion_head.kind not in ('affine_suffix', 'shared_number') or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Reuse requires a frozen isolated executor and an executor-independent scalar controller')
    logits, stops, controller_rows = controller_panel(model, tokenizer, spec['token_ids'], tasks, metadata,
                                                     config['max_depth'], config['safety_cap'])
    write_csv(args.output / 'controller_counts.csv', controller_rows)
    predictions, direct, seconds = executor_panel(model, tokenizer, spec['token_ids'], tasks,
        cap=config['safety_cap'], batch_size=config['batch_size'], console=console)
    arrays, loop_rows = quality_arrays(tasks, predictions, stops, config['max_depth'], direct)
    write_csv(args.output / 'per_loop.csv', loop_rows)
    write_csv(args.output / 'per_count_and_stratum.csv', grouped_rows(arrays, metadata))
    graph_rows = [{**meta, 'predictions': ''.join(SYMBOLS[int(p)] for p in predictions[i]),
                   'r_predictions': ''.join(SYMBOLS[int(p)] for p in direct[i]) if direct is not None else '',
                   'first_error_loop': next((j+1 for j,v in enumerate(arrays['nominal_final_correct'][i]) if not v), None)}
                  for i, meta in enumerate(metadata)]
    write_csv(args.output / 'graphs.csv', graph_rows)
    with gzip.open(args.output / 'decisions.csv.gz', 'wt', newline='') as handle:
        keys = ['graph_index', 'dataset_seed', 'graph_mode', 'depth', 'first_stop', 'executed_loops', *METRICS]
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator='\n'); writer.writeheader()
        for i, meta in enumerate(metadata):
            for j in range(config['max_depth']):
                writer.writerow({k: meta[k] for k in ('graph_index', 'dataset_seed', 'graph_mode')} |
                    {'depth': j+1, 'first_stop': stops[j], 'executed_loops': stops[j] or config['safety_cap']} |
                    {k: bool(arrays[k][i,j]) for k in METRICS})
    cohorts = {}; counts = np.arange(1, config['max_depth']+1)
    masks = {'all': np.ones(config['max_depth'], dtype=bool), '1_12': counts <= 12,
        '13_63': (counts >= 13) & (counts <= 63), '64_99': (counts >= 64) & (counts <= 99),
        '100_256': counts >= 100, 'held_out_values': np.isin(counts, [9,17,29,41,53])}
    for name, mask in masks.items():
        if mask.any():
            cohorts[name] = clustered_summary(arrays, metadata, mask, repeats=config['bootstrap_repeats'], seed=config['bootstrap_seed'])
    fidelity = native_fidelity(config, args.output, tasks, metadata, predictions, logits, args.device)
    if checkpoint_hashes(model_path) != before:
        raise ValueError('Frozen inference files changed during benchmark')
    result = {**provenance, 'status': 'complete', 'graphs': n, 'queries': n * config['max_depth'],
        'cohorts': cohorts, 'acceptance': criteria_result(arrays, config['criteria']), 'native_fidelity': fidelity,
        'inference_files_unchanged': True, 'executor_evaluation_seconds': seconds,
        'forced_executor_example_loops': n * config['safety_cap'], 'native_fidelity_questions': fidelity['questions'],
        'confidence_interval_scope': 'Percentile bootstrap of whole graphs within dataset-seed/graph-mode strata; horizons stay clustered',
        'limitations': ['26-state cyclic graphs', 'fixed raw prompt template', 'three-digit token-layout boundary',
                        'reused executor trajectories, not native adaptive latency', 'single frozen model seed']}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    console.print(f'Completed · quality criteria passed: {result["acceptance"]["passed"]} · native matches {fidelity["questions"]}/{fidelity["questions"]}')
    for name, values in cohorts.items():
        m = values['metrics']; console.print(f'{name}: joint {m["joint_success"]["rate"]:.2%} · exact stop {m["exact_stop"]["rate"]:.2%} · trajectory {m["complete_trajectory"]["rate"]:.2%}')
    report_saved_run(args.output, args)


if __name__ == '__main__':
    main()

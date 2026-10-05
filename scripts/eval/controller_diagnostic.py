"""One bounded controller diagnosis: count access, memory probes and tiny-set fits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from importlib.metadata import version
import platform
from pathlib import Path
import sys
from time import perf_counter

import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from scripts.dataset.pointer import PointerExample, validate_example
from scripts.eval.controller_features import extract_features, make_panel, replay
from scripts.eval.controller_learning import aggregate_stops, fit_controller, fit_probe, stop_rows
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file, synchronize
from scripts.eval.tracking import add_tracking_arguments, report_saved_run
from scripts.recurrent_qwen.interfaces import RecurrentController


def progress_bar() -> Progress:
    return Progress(TextColumn('{task.description}'), BarColumn(), TextColumn('{task.completed}/{task.total}'),
                    TimeElapsedColumn(), TimeRemainingColumn(),
                    console=Console(force_terminal=True, width=100), refresh_per_second=1)


def probe_suite(features: dict, metadata: list[dict], steps: int, seed: int,
                device: str, output: Path) -> list[dict]:
    depths = torch.tensor([m['depth'] - 1 for m in metadata])
    parts = [m['partition'] for m in metadata]
    loops = features['memory'].shape[1]
    points = sorted({1, 4, 8, 12, loops} & set(range(1, loops + 1)))
    cases = [('prompt_count', features['context'], depths, parts, loops),
             ('initial_count', features['initial'], depths, parts, loops)]
    cases += [(f'memory_count_loop_{t}', features['memory'][:, t - 1], depths, parts, loops) for t in points]
    cases.append(('memory_elapsed', features['memory'].flatten(0, 1),
                  torch.arange(loops).repeat(len(metadata)), [p for p in parts for _ in range(loops)], loops))
    runs = [(name, x, y, partitions, classes, nonlinear, False)
            for name, x, y, partitions, classes in cases for nonlinear in (False, True)]
    runs.append((*cases[0], False, True))
    results, confusion = [], []
    with progress_bar() as progress:
        bar = progress.add_task('Count / memory probes', total=len(runs))
        for name, x, y, partitions, classes, nonlinear, shuffled in runs:
            kind = 'mlp' if nonlinear else 'linear'
            prediction, metrics = fit_probe(x, y, partitions, classes, nonlinear=nonlinear,
                steps=steps, seed=seed, device=device, shuffled=shuffled)
            condition = f'{name}/{kind}' + ('/shuffled_labels' if shuffled else '')
            for partition in ('fit', 'dev', 'holdout'):
                indices = [i for i, p in enumerate(partitions) if p == partition]
                results.append({'condition': condition, 'partition': partition, 'examples': len(indices),
                    'accuracy': metrics['accuracy'][partition], 'chance_accuracy': metrics['chance_accuracy'],
                    'selected_step': metrics['selected_step'], 'dev_cross_entropy': metrics['dev_cross_entropy']})
                counts = Counter((int(y[i]) + 1, int(prediction[i]) + 1) for i in indices)
                confusion.extend({'condition': condition, 'partition': partition, 'label': label,
                                  'prediction': pred, 'count': count} for (label, pred), count in sorted(counts.items()))
            # Save after every completed fit so interruption still leaves useful output.
            write_csv(output / 'probes.csv', results)
            write_csv(output / 'probe_confusion.csv', confusion)
            progress.advance(bar)
    return results


def controller_suite(head: RecurrentController, features: dict, tasks: list, metadata: list[dict],
                     trained_depths: list[int], tiny_graphs: int, steps: int, lr: float,
                     seed: int, device: str, output: Path) -> tuple[dict, list[dict]]:
    fit_graphs = sorted({m['graph_index'] for m in metadata if m['partition'] == 'fit'})[:tiny_graphs]
    if len(fit_graphs) != tiny_graphs:
        raise ValueError('Not enough fit graphs for the tiny controller set')
    depths = torch.tensor([t.task_depth for t in tasks])
    selected = torch.tensor([i for i, m in enumerate(metadata)
                             if m['graph_index'] in fit_graphs and m['depth'] in trained_depths])
    rows, trajectories, histories, metrics = [], [], [], {}
    for condition in ('original', 'continued_tiny', 'fresh_tiny'):
        candidate = head
        if condition != 'original':
            if condition == 'fresh_tiny':
                torch.manual_seed(seed)
                candidate = RecurrentController(head.context[1].in_features, head.intermediate)
            with progress_bar() as progress:
                bar = progress.add_task(condition, total=steps)
                fitted, history = fit_controller(candidate, features, depths, selected, steps=steps,
                    learning_rate=lr, device=device,
                    progress=lambda n, row: progress.update(bar, completed=n, description=f'{condition} · fit exact {row["exact_stop"]:.1%}'))
            histories.extend({'condition': condition, **row} for row in history)
            write_csv(output / 'controller_fit.csv', histories)
            torch.save({k: v.cpu() for k, v in fitted.state_dict().items()}, output / f'{condition}.pt')
            candidate = fitted
        candidate = candidate.to(device).eval().requires_grad_(False)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tasks), 128):
                logits, _, _ = replay(candidate, features['context'][start:start + 128].to(device),
                                     features['working'][start:start + 128].to(device))
                chunks.append(logits.cpu())
        logits = torch.cat(chunks)
        if condition == 'original':
            torch.testing.assert_close(logits, features['logits'], atol=2e-5, rtol=1e-4)
        decisions = stop_rows(logits, features['predictions'], tasks, metadata, condition)
        for i, row in enumerate(decisions):
            row['graph_role'] = 'tiny_fit' if row['graph_index'] in fit_graphs else row['partition']
            row['count_role'] = ('trained' if row['depth'] in trained_depths else
                                 'interpolation' if row['depth'] <= max(trained_depths) else 'extrapolation')
            trajectories.extend({'condition': condition, 'example_id': row['example_id'],
                'partition': row['partition'], 'depth': row['depth'], 'loop': t + 1,
                'stop_probability': float(p), 'pointer_prediction': chr(int(features['predictions'][i, t]) + ord('A')),
                'nominal_target': tasks[i].intermediate_states[t] if t < row['depth'] else '',
                'final_target': tasks[i].final_state, 'supervised': t < row['depth'],
                'stop_target': t + 1 == row['depth'] if t < row['depth'] else ''}
                for t, p in enumerate(logits[i].sigmoid().tolist()))
        rows.extend(decisions)
        cohorts = sorted({(r['graph_role'], r['count_role']) for r in decisions})
        metrics[condition] = {f'{g}/{c}': aggregate_stops([r for r in decisions if r['graph_role'] == g and r['count_role'] == c])
                              for g, c in cohorts}
        write_csv(output / 'decisions.csv', rows)
        write_csv(output / 'trajectories.csv', trajectories)
        candidate.cpu()
    by_depth = []
    for condition in metrics:
        for partition in ('tiny_fit', 'fit', 'dev', 'holdout'):
            for depth in sorted(set(depths.tolist())):
                group = [r for r in rows if r['condition'] == condition and r['graph_role'] == partition and r['depth'] == depth]
                if group:
                    by_depth.append({'condition': condition, 'graph_role': partition, 'depth': depth, **aggregate_stops(group)})
    write_csv(output / 'stopping_by_depth.csv', by_depth)
    return metrics, by_depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--features-cache', type=Path, help='Rerun fits from an existing features.pt, without loading Qwen')
    parser.add_argument('--data', type=Path, default=Path('data/pointer/seed-61-independent/validation.jsonl'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fit-graphs', type=int, default=32)
    parser.add_argument('--dev-graphs', type=int, default=8)
    parser.add_argument('--holdout-graphs', type=int, default=24)
    parser.add_argument('--tiny-graphs', type=int, default=8)
    parser.add_argument('--max-depth', type=int, default=16)
    parser.add_argument('--probe-steps', type=int, default=400)
    parser.add_argument('--controller-steps', type=int, default=1000)
    parser.add_argument('--controller-lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=71)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if bool(args.model) == bool(args.features_cache):
        parser.error('Provide exactly one of --model or --features-cache')
    if min(args.batch_size, args.fit_graphs, args.dev_graphs, args.holdout_graphs, args.tiny_graphs,
           args.probe_steps, args.controller_steps) < 1 or not 2 <= args.max_depth <= 64 or args.controller_lr <= 0:
        parser.error('Invalid sizes, depth (2..64), steps or learning rate')
    if args.tiny_graphs > args.fit_graphs:
        parser.error('--tiny-graphs cannot exceed --fit-graphs')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu explicitly for tiny checks')
    output = args.output or Path('eval/pointer_diagnostics') / f'controller-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}'
    output.mkdir(parents=True, exist_ok=False)
    console = Console()
    console.print('[bold]Controller diagnosis[/bold] · frozen executor · FP32 · threshold 0.5')
    started = perf_counter()
    if args.features_cache:
        cache = torch.load(args.features_cache, map_location='cpu', weights_only=True)
        if cache.get('format') != 'controller-features-v1':
            raise ValueError('Unsupported controller feature cache')
        tasks = [PointerExample(**row) for row in cache['tasks']]
        metadata, features, provenance = cache['metadata'], cache['features'], cache['provenance']
        head = RecurrentController(features['context'].shape[-1], features['initial'].shape[-1])
        head.load_state_dict(cache['head'])
        console.print('Using cached graph partitions and depths; extraction arguments do not resample the cache.')
    else:
        from scripts.training.data import read_tasks
        from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
        tasks, metadata = make_panel(read_tasks(args.data, 'validation'),
            (args.fit_graphs, args.dev_graphs, args.holdout_graphs), list(range(1, args.max_depth + 1)), args.seed)
        model, tokenizer, spec = load_recurrent_checkpoint(args.model, device=args.device)
        model.config._attn_implementation = 'sdpa'
        digest = sha256_file(args.model / 'adapter_model.pt')
        with progress_bar() as progress:
            bar = progress.add_task('Frozen feature extraction', total=len(tasks))
            features, head, replay_error = extract_features(model, tokenizer, spec['token_ids'], tasks,
                args.max_depth, args.batch_size, progress=lambda done: progress.update(bar, completed=done))
        synchronize(torch.device(args.device))
        if sha256_file(args.model / 'adapter_model.pt') != digest:
            raise RuntimeError('Source checkpoint changed during extraction')
        provenance = {'model': str(args.model), 'checkpoint_sha256': digest,
            'config_sha256': sha256_file(args.model / 'recurrent_config.json'), 'data': str(args.data),
            'data_sha256': sha256_file(args.data), 'spec': spec, 'extraction_seed': args.seed,
            'graph_counts': [args.fit_graphs, args.dev_graphs, args.holdout_graphs],
            'batch_size': args.batch_size, 'attention': 'sdpa', 'precision': 'float32',
            'live_replay_max_logit_error': replay_error,
            'source_sha256': {str(p): sha256_file(p) for pattern in ('scripts/recurrent_qwen/*.py',
                'scripts/eval/controller_*.py', 'scripts/training/*.py', 'scripts/dataset/*.py') for p in Path('.').glob(pattern)}}
        cache = {'format': 'controller-features-v1', 'features': features, 'head': head.state_dict(),
                 'tasks': [t.to_dict() for t in tasks], 'metadata': metadata, 'provenance': provenance}
        torch.save(cache, output / 'features.pt')
        del model, tokenizer
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    for task in tasks:
        validate_example(task)
    n, loops, width = features['working'].shape
    if n != len(tasks) or len(metadata) != n or features['context'].shape != (n, width):
        raise ValueError('Feature cache and task shapes disagree')
    for key in ('memory', 'logits', 'predictions'):
        if features[key].shape[:2] != (n, loops):
            raise ValueError(f'Invalid cached {key} shape')
    if not all(torch.isfinite(value).all() for value in features.values()):
        raise ValueError('Nonfinite cached features')
    partitions = {}
    for task, meta in zip(tasks, metadata, strict=True):
        if (meta['example_id'] != task.example_id or meta['mapping_sha256'] != task.mapping_sha256
                or meta['depth'] != task.task_depth):
            raise ValueError('Feature row metadata and task disagree')
        previous = partitions.setdefault(meta['mapping_sha256'], meta['partition'])
        if previous != meta['partition']:
            raise ValueError('Graph leakage between diagnostic partitions')
    for graph in {m['graph_index'] for m in metadata}:
        group = [m for m in metadata if m['graph_index'] == graph]
        if sorted(m['depth'] for m in group) != list(range(1, loops + 1)):
            raise ValueError('Every graph must have every requested count exactly once')
    # Labels and numeric loop indices belong only to diagnostic probes / losses.
    depths = sorted({t.task_depth for t in tasks})
    if depths != list(range(1, features['memory'].shape[1] + 1)):
        raise ValueError('Count/progress probes require all counts at every recorded loop')
    trained_depths = provenance['spec'].get('train_depths') or list(range(1, provenance['spec']['train_max_depth'] + 1))
    if not set(trained_depths) <= set(depths):
        raise ValueError('Feature budget must cover every original training count')
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in tasks))
    write_csv(output / 'panel.csv', metadata)
    probes = probe_suite(features, metadata, args.probe_steps, args.seed, args.device, output)
    stopping, _ = controller_suite(head, features, tasks, metadata, trained_depths,
        args.tiny_graphs, args.controller_steps, args.controller_lr, args.seed, args.device, output)
    synchronize(torch.device(args.device))
    summary = {'status': 'complete', 'schema_version': 1,
        'command': [sys.executable, '-m', 'scripts.eval.controller_diagnostic', *sys.argv[1:]],
        'python': platform.python_version(), 'packages': {name: version(name) for name in ('torch', 'transformers', 'rich')},
        'device': args.device, 'dtype': 'float32', 'seed': args.seed, 'model': provenance['model'],
        'data': provenance['data'], 'data_sha256': provenance['data_sha256'], 'extraction': provenance,
        'features_cache': str(args.features_cache or output / 'features.pt'),
        'features_sha256': sha256_file(args.features_cache or output / 'features.pt'), 'depths': depths,
        'trained_depths': trained_depths, 'tiny_graphs': args.tiny_graphs,
        'probe_steps': args.probe_steps, 'controller_steps': args.controller_steps, 'controller_lr': args.controller_lr,
        'source_sha256': {str(p): sha256_file(p) for pattern in ('scripts/eval/*.py', 'scripts/recurrent_qwen/*.py',
            'scripts/training/*.py', 'scripts/dataset/*.py') for p in Path('.').glob(pattern)},
        'tasks_sha256': sha256_file(output / 'tasks.jsonl'),
        'probes': {f'{r["condition"]}/{r["partition"]}': r['accuracy'] for r in probes},
        'stopping': stopping, 'wall_seconds': perf_counter() - started,
        'interpretation': ['All graph partitions are development data, not reserved confirmation.',
            'Probe success shows accessible information, not causal use; failure does not prove absence.',
            'Count probes include all requested counts in fit; they do not measure unseen-count extrapolation.',
            'Controller fits use only original trained counts on the tiny fit graphs; selection uses fit loss.',
            'Exact stopping is primary; repeated correct letters can inflate stopped-answer accuracy.',
            'Fits consume cached FP32 frozen features; no Qwen parameters or source checkpoint files are updated.',
            'A bounded tiny-fit failure does not prove the controller architecture is incapable.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    table = Table(title='Exact stopping · threshold 0.5')
    table.add_column('Controller'); table.add_column('Tiny fit / trained counts'); table.add_column('New graphs / trained counts')
    for name, metrics in stopping.items():
        table.add_row(name, f'{metrics["tiny_fit/trained"]["exact_stop"]:.1%}', f'{metrics["holdout/trained"]["exact_stop"]:.1%}')
    console.print(table)
    console.print(f'[green]Saved diagnostic report and CSVs:[/green] {output}')
    report_saved_run(output, args)


if __name__ == '__main__':
    main()

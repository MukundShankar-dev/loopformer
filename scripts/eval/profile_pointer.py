"""Short warmed checkpoint profiling, with optional disposable backward updates."""

import argparse
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
from rich.console import Console

from scripts.eval.pointer_task import load_examples, sha256_file
from scripts.eval.profiling import distribution, measure, memory, module_ranges, training_update
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.data import encode_tasks, select_tasks
from scripts.training.evaluation import evaluate


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--mode', choices=('eval', 'train'), default='eval')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='CUDA profiler or CPU implementation check')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=128, help='Eval prefix size; training uses one balanced effective-batch group')
    parser.add_argument('--loops', type=int, default=16, help='Evaluation budget; training uses nominal depth per microbatch')
    parser.add_argument('--effective-batch', type=int, default=8)
    parser.add_argument('--train-max-depth', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('batch_size', 'limit', 'loops', 'effective_batch', 'train_max_depth', 'warmup', 'repeats', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.output.exists():
        parser.error('Output already exists')
    if args.mode == 'train' and (args.batch_size > args.effective_batch or args.effective_batch % args.batch_size):
        parser.error('Training batch must divide --effective-batch; use batch 4 or 8 for the matched update group')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    console = Console()
    with console.status('Loading checkpoint and preparing fixed profiling examples…'):
        tasks = load_examples(args.data, args.limit if args.mode == 'eval' else None)
        if args.mode == 'train':
            if any(t.split != 'train' for t in tasks):
                parser.error('Disposable training profiling requires the training split')
            tasks = select_tasks(tasks, args.train_max_depth, args.seed, limit=args.effective_batch)
            if len(tasks) != args.effective_batch:
                parser.error('Not enough training examples')
        elif max(t.task_depth for t in tasks) > args.loops:
            parser.error('Loop budget must cover nominal targets')
        model, tokenizer, spec = load_recurrent_checkpoint(args.model, device=args.device, download=args.download)
        if args.mode == 'train' and spec.get('loss_reduction', 'example_mean') != 'example_mean':
            parser.error('Training profiler currently measures the retained example-mean baseline only')
        items = encode_tasks(tasks, tokenizer, dict(zip(spec['symbols'], spec['token_ids'], strict=True)), model.config.max_position_embeddings)
    # Snapshot only trainable adapters. Restores happen outside measured intervals.
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    initial = {n: p.detach().clone() for n, p in parameters.items()} if args.mode == 'train' else {}
    if args.mode == 'train' and not parameters:
        raise ValueError('Checkpoint loader supplied no trainable adapters')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'mode': args.mode, 'command': [sys.executable, '-m', 'scripts.eval.profile_pointer', *sys.argv[1:]],
               'started_utc': datetime.now(timezone.utc).isoformat(), 'settings': vars(args) | {'model': str(args.model), 'data': str(args.data), 'output': str(args.output)},
               'checkpoint': str(args.model.resolve()), 'adapter_sha256': sha256_file(args.model / 'adapter_model.pt'),
               'metadata_sha256': sha256_file(args.model / 'recurrent_config.json'),
               'tokenizer_sha256': sha256_file(args.model / 'tokenizer.json'), 'data_sha256': sha256_file(args.data),
               'example_ids': [t.example_id for t in tasks], 'python': platform.python_version(),
               'packages': {n: version(n) for n in ('torch', 'transformers', 'peft')},
               'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
               'cuda_runtime': torch.version.cuda, 'dtype': 'float32', 'attention': 'eager',
               'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'source_sha256': {str(p.relative_to(root)): sha256_file(p) for folder in ('eval', 'training', 'recurrent_qwen', 'dataset') for p in (root/'scripts'/folder).glob('*.py')},
               'scope': 'Warmed eval includes collation, forward, scoring and CSV export; excludes loading/encoding. Train is one disposable update with fresh AdamW, not resume or steady-state trainer throughput. No checkpoint saves, validation, or dashboard in train timing.'}
    summary_path = args.output/'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')

    def prepare():
        if args.mode == 'eval':
            return None
        with torch.no_grad():
            for n, p in parameters.items():
                p.copy_(initial[n])
        torch.manual_seed(args.seed)
        return torch.optim.AdamW(list(parameters.values()), lr=.0002, weight_decay=0, foreach=False)

    def operation(optimizer, *, trace=False):
        if args.mode == 'train':
            return training_update(model, items, spec['token_ids'], tokenizer.pad_token_id, optimizer, args.batch_size)
        return evaluate(model, items[:args.batch_size] if trace else items, spec['token_ids'], tokenizer.pad_token_id,
                        batch_size=args.batch_size, loops=args.loops, profile_ranges=trace,
                        output=args.output / ('profiled_batch.csv' if trace else 'trajectories.csv'))

    times, measurements = [], []
    with console.status(f'Warming up ({args.warmup}), then measuring {args.repeats} repeats…'):
        for i in range(args.warmup + args.repeats):
            optimizer = prepare()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            elapsed, result = measure(lambda: operation(optimizer), device)
            if i >= args.warmup:
                times.append(elapsed); measurements.append({'seconds': elapsed, 'memory': memory(device), 'metrics': result})
            del optimizer
    # Preserve completed measurements even if CUDA tracing is unavailable later.
    summary.update(timing=distribution(times), repeats=measurements, phase='trace_pending')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    # Compare the same single-batch/update workload with and without profiler.
    optimizer = prepare()
    trace_baseline, _ = measure(lambda: operation(optimizer, trace=True), device)
    del optimizer
    optimizer = prepare()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with console.status('Capturing one batch/update with CPU/GPU operator attribution…'):
        with torch.profiler.profile(activities=activities) as prof:
            with module_ranges(model):
                trace_seconds, _ = measure(lambda: operation(optimizer, trace=True), device)
    trace = args.output/'trace.json'
    prof.export_chrome_trace(str(trace))
    with trace.open('rb') as src, gzip.open(args.output/'trace.json.gz', 'wb') as dst:
        import shutil
        shutil.copyfileobj(src, dst)
    trace.unlink()
    ranges = [{'name': event.key, 'calls': event.count, 'cpu_total_us': event.cpu_time_total,
               'cpu_self_us': event.self_cpu_time_total,
               'device_total_us': getattr(event, 'device_time_total', 0),
               'device_self_us': getattr(event, 'self_device_time_total', 0)}
              for event in prof.key_averages() if event.key.startswith('pointer/')]
    (args.output/'ranges.json').write_text(json.dumps(ranges, indent=2)+'\n')
    (args.output/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total', row_limit=60))
    adapter_diagnostics = None
    if args.mode == 'train':
        del optimizer
        optimizer = prepare()
        diagnostic_seconds, adapter_diagnostics = measure(lambda: training_update(
            model, items, spec['token_ids'], tokenizer.pad_token_id, optimizer, args.batch_size,
            diagnostics=True), device)
        adapter_diagnostics['diagnostic_seconds'] = diagnostic_seconds
    for n, p in parameters.items():
        if n in initial:
            with torch.no_grad():
                p.copy_(initial[n])
            p.grad = None
    stats = distribution(times)
    summary.update(status='complete', phase='complete', timing=stats, repeats=measurements,
                   cuda_kernel_events=sum(event.device_type == torch.autograd.DeviceType.CUDA for event in prof.events()),
                   adapter_diagnostics=adapter_diagnostics,
                   examples_per_second=len(items)/stats['median_seconds'],
                   supervised_transitions_per_second=sum(len(x.targets) for x in items)/stats['median_seconds'],
                   executed_transitions_per_second=(len(items)*args.loops if args.mode == 'eval' else measurements[0]['metrics']['executed_transitions'])/stats['median_seconds'],
                   trace_unprofiled_seconds=trace_baseline, trace_profiled_seconds=trace_seconds,
                   profiler_interval_ratio=trace_seconds/trace_baseline,
                   trace_scope='one evaluation batch including CSV export' if args.mode == 'eval' else 'one effective optimizer update',
                   profiler_note='Overlapping/nested inclusive operator times must not be added; trace is diagnostic, warmed unprofiled repeats determine throughput. Trace export/profiler teardown excluded from interval ratio.')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    console.print(f"[green]Complete[/green] · median {stats['median_seconds']:.3f}s · p95 {stats['p95_seconds']:.3f}s · {summary['examples_per_second']:.2f} examples/s")
    console.print(f'Saved summary.json, operators.txt and trace.json.gz to {args.output}')


if __name__ == '__main__':
    main()

"""Read-only progress from saved suite artifacts; never imports or loads a model."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from rich.console import Console
from rich.table import Table


STAGES = (
    ('paper_reuse_audit', 'Verify frozen source'),
    ('paper_suite', 'Model extraction'),
    ('paper_baseline', 'Ordinary Qwen baseline'),
    ('paper_snapshots', 'Executor snapshots'),
    ('paper_metrics', 'Metrics and bootstrap'),
    ('paper_structure', 'Graph/failure analysis'),
    ('audit_paper_analysis', 'Independent decision audit'),
    ('plot_paper_analysis', 'Render final figures'),
    ('retire_pointer_plots', 'Retire old plot exports'),
    ('paper_report', 'Write completion report'),
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # Summaries are small non-atomic writes. A watcher can catch one mid-write.
        return {}


def _active_process(root: Path) -> dict:
    """Identify the owning Linux/WSL stage, including loading and native gates."""
    modules = {f'scripts.eval.{module}': label for module, label in STAGES}
    proc = Path('/proc')
    candidate = {}
    if not proc.exists():
        return {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = entry.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if '-m' not in args:
                continue
            module = args[args.index('-m') + 1]
            if module not in modules:
                continue
            option = '--output' if module.endswith(('paper_suite', 'paper_snapshots')) else '--input'
            directory = Path(args[args.index(option) + 1]) if option in args else Path('eval/pointer_analysis/paper-20261009')
            if not directory.is_absolute():
                directory = entry.joinpath('cwd').resolve() / directory
            if directory.resolve() == root.resolve():
                result = dict(module=module.rsplit('.', 1)[-1], label=modules[module], pid=int(entry.name))
                if module.endswith('paper_snapshots'):
                    candidate = result  # A baseline prerequisite may own its child process.
                else:
                    return result
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue  # A process may exit while its metadata is being read.
    return candidate


def _chunk_totals(directory: Path, total: int) -> tuple[int, float]:
    seen: set[int] = set()
    seconds = 0.0
    for path in directory.glob('graphs-*.npz'):
        if '.tmp.' in path.name:
            continue
        with ZipFile(path) as archive:
            indices = np.load(BytesIO(archive.read('graph_indices.npy')), allow_pickle=False)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(f'Invalid graph indices in {path}')
            ids = {int(i) for i in indices}
            if len(ids) != len(indices) or seen.intersection(ids) or any(i < 0 or i >= total for i in ids):
                raise ValueError(f'Duplicate or out-of-range graph coverage in {path}')
            seen.update(ids)
            if 'seconds.npy' in archive.namelist():
                seconds += float(np.load(BytesIO(archive.read('seconds.npy')), allow_pickle=False))
    return len(seen), seconds


def suite_status(root: Path) -> dict:
    """Report committed graph chunks and passing gates, never padded predictions."""
    config = _read_json(root / 'protocol.json')
    if not config:
        return {'status': 'not started', 'models': [], 'activity': 'Waiting for protocol.json'}
    state = next((label for marker, label in (
        ('COMPLETE', 'complete'), ('FAILED', 'failed'), ('PAUSED', 'paused'), ('RUNNING', 'running')
    ) if (root / marker).exists()), 'incomplete')
    process = _active_process(root)
    rows = []
    for arm in config['models']:
        directory = root / arm['name']
        graphs, seconds = _chunk_totals(directory, config['graph_count'])
        summary = _read_json(directory / 'summary.json')
        native = _read_json(directory / 'native_fidelity.json')
        complete = summary.get('status') == 'complete' and graphs == config['graph_count']
        average = seconds / graphs if graphs else None
        rows.append(dict(model=arm['name'], graphs=graphs, total=config['graph_count'],
            native_checked=native.get('passed') is True,
            state='complete' if complete else 'partial' if graphs else 'pending',
            initialized=(directory / 'freeze.json').exists(),
            inference_seconds=seconds,
            remaining_inference_seconds=(config['graph_count'] - graphs) * average if average is not None else None))
    snapshots = []
    for step in config['final_executor_history']['retained_weight_steps']:
        directory = root / 'executor_snapshots' / f'step-{step:06d}'
        graphs, _ = _chunk_totals(directory, config['graph_count'])
        summary = _read_json(directory / 'summary.json')
        snapshots.append(dict(step=step, graphs=graphs, total=config['graph_count'],
            complete=summary.get('status') == 'complete' and graphs == config['graph_count'],
            initialized=(directory / 'freeze.json').exists()))
    baseline_progress = _read_json(root / 'ordinary_qwen/progress.json')
    baseline_total = config['graph_count'] * len(config.get('requests', range(1,257)))
    baseline_queries = baseline_progress.get('queries', 0)
    if not 0 <= baseline_queries <= baseline_total:
        raise ValueError('Ordinary baseline progress outside frozen query count')
    baseline_complete = baseline_progress.get('status') == 'complete' and baseline_queries == baseline_total
    baseline_seconds = baseline_progress.get('evaluation_seconds', 0)
    baseline = dict(queries=baseline_queries, total=baseline_total,
        state='complete' if baseline_complete else 'extracting' if process.get('module') == 'paper_baseline' else 'partial' if baseline_queries else 'pending',
        native_checked=_read_json(root / 'ordinary_qwen/native_fidelity.json').get('passed') is True,
        remaining_inference_seconds=(baseline_total-baseline_queries)*baseline_seconds/baseline_queries if baseline_queries else None)
    done = {
        'paper_reuse_audit': _read_json(root / 'source_reuse_audit.json').get('passed') is True,
        'paper_suite': all(r['state'] == 'complete' for r in rows),
        'paper_baseline': baseline_complete,
        'paper_snapshots': all(r['complete'] for r in snapshots),
        'paper_metrics': _read_json(root / 'reference_audit.json').get('passed') is True,
        'paper_structure': (root / 'first_error_exposures.csv').exists() and process.get('module') != 'paper_structure',
        'audit_paper_analysis': _read_json(root / 'independent_audit.json').get('passed') is True and
                                _read_json(root / 'ordinary_qwen/independent_audit.json').get('passed') is True,
        'plot_paper_analysis': _read_json(root / 'plots/manifest.json').get('status') == 'complete',
        'retire_pointer_plots': (root / 'retired_plot_exports.json').exists() and process.get('module') != 'retire_pointer_plots',
        'paper_report': state == 'complete',
    }
    activity = process.get('label', 'Starting / between stages' if state == 'running' else state.capitalize())
    if state == 'running' and process.get('module') == 'paper_suite':
        current = next((r for r in rows if r['state'] != 'complete' and r['initialized']), None)
        if current:
            current['state'] = 'extracting' if current['native_checked'] else 'loading / native checks'
            activity = f"{current['model']} — {current['state']}"
    elif state == 'running' and process.get('module') == 'paper_snapshots':
        current = next((r for r in snapshots if not r['complete'] and r['initialized']), None)
        if current:
            activity = f"Executor snapshot {current['step']} — {current['graphs']}/{current['total']} graphs"
    elif state == 'running' and process.get('module') == 'paper_baseline':
        activity = f"Ordinary Qwen — {baseline_queries:,}/{baseline_total:,} queries" if baseline['native_checked'] else 'Ordinary Qwen — loading / native generation checks'
    if state in ('paused', 'failed', 'complete'):
        activity = dict(paused='Paused; completed chunks retained', failed='Failed; inspect run.log', complete='Complete')[state]
    stages = [dict(module=module, label=label, state='done' if done[module] else
                   'running' if state == 'running' and process.get('module') == module else 'pending')
              for module, label in STAGES]
    return dict(status=state, models=rows, snapshots=snapshots,
        ordinary_baseline=baseline,
        snapshots_complete=sum(r['complete'] for r in snapshots), snapshots_total=len(snapshots),
        scored=done['paper_metrics'], independently_audited=done['audit_paper_analysis'],
        plots_ready=done['plot_paper_analysis'], stages=stages, activity=activity, process=process,
        eta_scope='Per-model extraction ETA only; native gates, later models, scoring and plots excluded')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('eval/pointer_analysis/paper-20261009'))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = suite_status(args.input)
    if args.json:
        print(json.dumps(result))
        return
    table = Table(title=f"Pointer paper suite · {result['status']}")
    for column in ('Model', 'Graphs', 'Native gate', 'State', 'Extraction ETA'):
        table.add_column(column)
    for row in result['models']:
        eta = row['remaining_inference_seconds']
        label = f'{eta / 3600:.1f} h' if eta is not None else 'unmeasured'
        table.add_row(row['model'], f"{row['graphs']}/{row['total']}",
                      'passed' if row['native_checked'] else 'pending', row['state'], label)
    console = Console()
    console.print(table)
    console.print(result['activity'])
    if result['models']:
        console.print(f"Executor snapshots: {result['snapshots_complete']}/{result['snapshots_total']} · independent audit: {result['independently_audited']} · figures: {result['plots_ready']}")
        console.print(result['eta_scope'])


if __name__ == '__main__':
    main()

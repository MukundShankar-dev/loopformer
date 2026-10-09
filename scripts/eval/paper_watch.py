"""Live Rich monitor for the frozen paper suite, locally or over SSH."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from scripts.eval.paper_status import suite_status


MODEL_LABELS = {
    'ce_full': 'CE-only', 'joint_full': 'Joint completion', 'fixed6': 'Fixed memory · depth 6',
    'fixed12': 'Fixed memory · depth 12', 'gru': 'GRU controller', 'final': 'Final controller',
}
COLORS = {'complete': 'green', 'done': 'green', 'running': 'cyan', 'extracting': 'cyan',
          'loading / native checks': 'cyan', 'paused': 'yellow', 'failed': 'red'}


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return '—'
    minutes = max(0, round(seconds / 60))
    if seconds > 0 and minutes == 0:
        return '<1 min'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m' if hours else f'{minutes} min'


def fetch_status(root: Path, remote: str | None, remote_root: str) -> dict:
    """SSH only fetches JSON metadata; no evaluator is launched or signaled."""
    if remote is None:
        return suite_status(root)
    if remote.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@:-]+', remote):
        raise ValueError('Use an SSH hostname/alias such as desktop')
    command = f'cd {shlex.quote(remote_root)} && .venv/bin/python -m scripts.eval.paper_status --input {shlex.quote(str(root))} --json'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', remote, command],
                            text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise ConnectionError(result.stderr.strip() or result.stdout.strip() or f'SSH exited {result.returncode}')
    return json.loads(result.stdout)


def dashboard(result: dict | None, location: str, updated: str, error: str | None = None) -> Group:
    """Render distinct extraction coverage and pipeline stages, never a time percent."""
    panels = []
    status = result.get('status', 'connecting') if result else 'connecting'
    header = Text(f'Pointer evaluation  ·  {location}  ·  ', style='bold')
    header.append(status.upper(), style=COLORS.get(status, 'yellow'))
    activity = Text(result.get('activity', 'Reading progress…') if result else 'Reading progress…', style='cyan')
    panels.append(Panel(Group(header, activity), border_style=COLORS.get(status, 'blue')))
    if error:
        warning = Text('Progress refresh failed; displayed values are from the last successful update.\n', style='bold yellow')
        warning.append(error, style='yellow')
        panels.append(Panel(warning, border_style='yellow'))
    if result and result.get('models'):
        table = Table(expand=True, show_edge=False)
        for name in ('Model', 'Progress', 'Graphs', 'Stage', 'ETA*'):
            table.add_column(name, no_wrap=name not in ('Model', 'Stage'))
        for row in result['models']:
            color = COLORS.get(row['state'], 'bright_black')
            percent = 100 * row['graphs'] / row['total']
            bar = ProgressBar(total=row['total'], completed=row['graphs'], width=12,
                              complete_style=color, finished_style='green')
            progress = Table.grid(padding=(0, 1))
            progress.add_row(bar, Text(f'{percent:5.1f}%', style=color))
            table.add_row(MODEL_LABELS.get(row['model'], row['model']), progress,
                          f"{row['graphs']:,}/{row['total']:,}", Text(row['state'], style=color),
                          'done' if row['state'] == 'complete' else format_eta(row['remaining_inference_seconds']))
        panels.append(Panel(table, title='Model extraction', border_style='blue'))
        baseline = result.get('ordinary_baseline')
        if baseline:
            color = COLORS.get(baseline['state'], 'bright_black')
            bar = Table.grid(padding=(0, 1))
            bar.add_row(Text('Ordinary Qwen · 3-shot', style='bold'),
                ProgressBar(total=baseline['total'], completed=baseline['queries'], width=24, complete_style=color, finished_style='green'),
                Text(f"{baseline['queries']:,}/{baseline['total']:,} queries · {baseline['state']} · ETA {format_eta(baseline['remaining_inference_seconds'])}", style=color))
            panels.append(Panel(bar, title='Final-answer baseline · generation, not recurrent loops', border_style='blue'))
        snapshots = result.get('snapshots', [])
        if snapshots:
            total = sum(r['total'] for r in snapshots)
            graphs = sum(r['graphs'] for r in snapshots)
            snapshot_bar = Table.grid(padding=(0, 1))
            snapshot_bar.add_row(Text('Executor snapshots', style='bold'),
                                ProgressBar(total=total, completed=graphs, width=24, complete_style='cyan', finished_style='green'),
                                Text(f'{100 * graphs / total:.1f}% · {graphs:,}/{total:,} graphs · {result["snapshots_complete"]}/{len(snapshots)} checkpoints complete'))
            panels.append(snapshot_bar)
        stages = Text()
        for row in result.get('stages', []):
            symbol = {'done': '✓', 'running': '▶', 'pending': '·'}[row['state']]
            stages.append(f"{symbol} {row['label']}   ", style=COLORS.get(row['state'], 'dim'))
        panels.append(Panel(stages, title='Pipeline · ✓ done · ▶ running · · pending', border_style='blue'))
    panels.append(Text('* ETA is remaining extraction time for that model, estimated from completed chunks.\nNative checks, subsequent models and reporting are excluded; pending ETA is unmeasured.', style='dim'))
    panels.append(Text(f'Last successful update: {updated or "not yet"}  ·  refresh every few seconds  ·  Ctrl+C exits this monitor only', style='dim'))
    return Group(*panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('eval/pointer_analysis/paper-20261009'))
    parser.add_argument('--remote', help='SSH hostname/alias, e.g. desktop; omitted for a local run')
    parser.add_argument('--remote-root', default='/home/mukund/loopformer', help='Repository directory on SSH host')
    parser.add_argument('--interval', type=float, default=5, help='Seconds between refreshes (minimum 2)')
    parser.add_argument('--once', action='store_true', help='Print one dashboard and exit')
    args = parser.parse_args()
    if not 2 <= args.interval <= 3600:
        parser.error('--interval must be between 2 and 3600 seconds')
    console = Console()
    if not args.once and not console.is_terminal:
        parser.error('Live display requires a terminal; use --once or paper_status --json for pipes')
    location = args.remote or 'local'
    if args.once:
        result = fetch_status(args.input, args.remote, args.remote_root)
        console.print(dashboard(result, location, datetime.now().strftime('%H:%M:%S')))
        return
    result = None
    updated = ''
    try:
        with Live(dashboard(None, location, updated), console=console, screen=True, auto_refresh=False) as live:
            while True:
                error = None
                try:
                    result = fetch_status(args.input, args.remote, args.remote_root)
                    updated = datetime.now().strftime('%H:%M:%S')
                except (ConnectionError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    error = str(exc)
                live.update(dashboard(result, location, updated, error), refresh=True)
                if error is None and result and result['status'] in ('complete', 'failed'):
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        console.print('[dim]Monitor closed. The evaluation is unaffected.[/dim]')
        return
    console.print(dashboard(result, location, updated))


if __name__ == '__main__':
    main()

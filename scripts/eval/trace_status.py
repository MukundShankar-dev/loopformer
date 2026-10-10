"""Read-only progress display for the ordinary trace baseline pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table


def snapshot() -> dict:
    config=json.loads(Path('configs/pointer_trace_baselines.json').read_text());root=Path(config['output'])
    paths=dict(pipeline=root/'pipeline.json',training=Path(config['training'])/'progress.json',
        original=root/'original'/'progress.json',sft=root/'sft'/'progress.json')
    return {name:json.loads(path.read_text()) if path.exists() else {} for name,path in paths.items()}


def table(state: dict) -> Table:
    result=Table(title=f"Trace baselines · {state['pipeline'].get('stage','not launched')}")
    for name in ('Stage','Progress','State','Rate / ETA'):result.add_column(name)
    for name in ('training','original','sft'):
        row=state[name];done=row.get('step',row.get('queries',row.get('steps',0)));total=row.get('total_steps',row.get('total',row.get('steps',0)))
        status=row.get('status','pending');color='green' if status=='complete' else 'cyan' if row else 'dim'
        seconds=row.get('estimated_remaining_seconds')
        speed=row.get('questions_per_second',row.get('train',{}).get('examples_per_second'))
        detail=f'{speed:.1f}/s' if speed is not None else ''
        if seconds is not None:detail+=f' · {seconds/60:.0f} min left'
        result.add_row(name,f'[{color}]{done:,}/{total:,}[/{color}]',f'[{color}]{status}[/{color}]',detail)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote');parser.add_argument('--watch',action='store_true');parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    def read() -> dict:
        if args.remote:
            text=subprocess.check_output(['ssh',args.remote,'cd /home/mukund/loopformer && .venv/bin/python -m scripts.eval.trace_status --json'],text=True)
            return json.loads(text)
        return snapshot()
    if args.json:print(json.dumps(read()));return
    if not args.watch:Console().print(table(read()));return
    try:
        with Live(table(read()),refresh_per_second=1) as live:
            while True:time.sleep(5);live.update(table(read()))
    except KeyboardInterrupt:pass


if __name__=='__main__':main()

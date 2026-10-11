"""Read-only progress display for the ordinary trace baseline pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress_bar import ProgressBar
from rich.panel import Panel
from rich.console import Group
from rich.text import Text


def snapshot() -> dict:
    config=json.loads(Path('configs/pointer_trace_baselines.json').read_text());root=Path(config['output'])
    paths=dict(pipeline=root/'pipeline.json',training=Path(config['training'])/'progress.json',
        original=root/'original'/'progress.json',sft=root/'sft'/'progress.json')
    state={name:json.loads(path.read_text()) if path.exists() else {} for name,path in paths.items()}
    for arm in ('original','sft'):
        if not state[arm]:state[arm]=dict(status='pending',queries=0,total=config['graph_count']*len(config['requests']))
    training_config=json.loads(Path('configs/trace_sft.json').read_text())
    state['effective_batch']=training_config['batch_size']*training_config['gradient_accumulation']
    if not state['training']:
        state['training']=dict(status='pending',step=0,total_steps=training_config['schedule_steps'])
    best=Path(config['training'])/'best_checkpoint.json'
    state['checkpoint']=json.loads(best.read_text()) if best.exists() else {}
    return state


def duration(seconds: float | None) -> str:
    if seconds is None:return '—'
    minutes=max(0,round(seconds/60))
    return f'{minutes//60}h {minutes%60:02d}m' if minutes>=60 else f'{minutes}m'


def table(state: dict) -> Table:
    result=Table(expand=True)
    for name in ('Stage','Progress','Coverage','State','Throughput','ETA*'):result.add_column(name)
    pipeline=state['pipeline'];paused=pipeline.get('status') in ('stopped','failed')
    labels=dict(training='Full Qwen SFT',original='Original Qwen · trace',sft='SFT Qwen · trace')
    for name in ('training','original','sft'):
        row=state[name];done=row.get('step',row.get('queries',row.get('steps',0)));total=row.get('total_steps',row.get('total',row.get('steps',0)))
        status=row.get('status','pending')
        if paused and status not in ('pending','complete'):status=pipeline['status']
        color='green' if status=='complete' else 'dim' if status=='pending' else 'yellow' if paused else 'cyan'
        seconds=row.get('estimated_remaining_seconds')
        speed=row.get('questions_per_second',row.get('train',{}).get('examples_per_second'))
        if name=='training' and speed:
            seconds=(total-done)*state['effective_batch']/speed
        if paused or status in ('pending','complete'):seconds=None
        detail=f'{speed:.1f} '+('ex/s' if name=='training' else 'q/s') if speed is not None else ''
        if 'tokens_per_second' in row:detail+=f" · {row['tokens_per_second']:.0f} tok/s"
        if name!='training' and 'requested_depth' in row:status+=f" · N={row['requested_depth']}"
        result.add_row(labels[name],ProgressBar(total=max(1,total),completed=done,width=16,complete_style=color),
            f'[{color}]{done:,}/{total:,} ({done/max(1,total):.1%})[/{color}]',
            f'[{color}]{status}[/{color}]',detail,duration(seconds))
    stage=pipeline.get('stage','');status='complete' if pipeline.get('status')=='complete' else 'running' if stage=='independent-audits-and-plots' else 'pending'
    if paused and stage=='independent-audits-and-plots':status=pipeline['status']
    result.add_row('Independent audits + plots','—','—',status,'CPU · no model inference','—')
    return result


def dashboard(state: dict) -> Group:
    pipeline=state['pipeline'];status=pipeline.get('status','not launched')
    color='green' if status=='complete' else 'yellow' if status in ('stopped','failed') else 'cyan'
    header=Panel(Text(f"Ordinary trace baselines · {status.upper()}\n{pipeline.get('stage','not launched')}",style=color),border_style=color)
    checkpoint=state.get('checkpoint',{}).get('step')
    footer=Text('* ETA covers the current stage only. Training excludes validation/saves; generation rate may change with depth.\n'
                'Audits/plots show stage status without an invented progress percentage.\n'
                f"Saved training checkpoint: {'update '+str(checkpoint) if checkpoint is not None else 'none'} · "
                'refresh every 5 seconds · Ctrl+C closes this monitor only.',style='dim')
    return Group(header,table(state),footer)


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
    if not args.watch:Console().print(dashboard(read()));return
    try:
        with Live(dashboard(read()),refresh_per_second=1) as live:
            while True:time.sleep(5);live.update(dashboard(read()))
    except KeyboardInterrupt:pass


if __name__=='__main__':main()

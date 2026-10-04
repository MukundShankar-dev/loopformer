"""Optional scalar experiment tracking; local artifacts remain authoritative."""
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator


def scalar_metrics(values: dict, prefix: str = '') -> dict:
    """Flatten numeric leaves; omit strings, lists and undefined metrics."""
    result = {}
    for key, value in values.items():
        name = f'{prefix}/{key}' if prefix else str(key)
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)):
            result[name] = value
    return result


class Tracker:
    def __init__(self, run=None):
        self.run = run

    def log(self, event: dict) -> None:
        if self.run is None or event.get('event') not in ('train', 'validation'):
            return
        kind = event['event']
        values = {}
        for key in ('train', 'validation', 'train_probe'):
            if key in event:
                values.update(scalar_metrics(event[key], key))
        values.update(scalar_metrics({k: v for k, v in event.items()
                                      if k not in ('train', 'validation', 'train_probe', 'step')}, kind))
        # W&B's internal event index is separate: train and validation can share a step.
        self.run.log({'optimizer_step': event['step'], **values})

    def summary(self, result: dict) -> None:
        if self.run is not None:
            self.run.summary.update(scalar_metrics(result, 'final'))


@contextmanager
def tracking_run(output: Path, metadata: dict, config: dict, *, mode: str,
                 project: str, entity: str | None = None, name: str | None = None) -> Iterator[Tracker]:
    """Create one run per training invocation, including checkpoint continuations."""
    if mode == 'disabled':
        yield Tracker()
        return
    import wandb

    # SDK internal state is ignored by Git. Do not upload model files or source code.
    with wandb.init(project=project, entity=entity, name=name or output.name, mode=mode,
                    dir=str(output.resolve()), config={'training': config, 'provenance': metadata},
                    settings=wandb.Settings(quiet=True, console='off', disable_code=True)) as run:
        run.define_metric('optimizer_step')
        run.define_metric('*', step_metric='optimizer_step')
        (output / 'wandb_run.json').write_text(json.dumps(
            {'id': run.id, 'url': run.url, 'project': run.project, 'entity': run.entity,
             'mode': mode, 'sdk_version': wandb.__version__}, indent=2) + '\n')
        yield Tracker(run)

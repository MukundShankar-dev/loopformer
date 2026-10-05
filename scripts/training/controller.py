"""Controller-only optimization on cached, immutable executor observations."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil

import torch
from torch import Tensor

from scripts.eval.controller_features import replay
from scripts.eval.controller_learning import aggregate_stops, stop_rows
from scripts.eval.executor_diagnostic import matched_tasks
from scripts.training.objective import completion_loss


@dataclass(frozen=True)
class ControllerConfig:
    source: str = 'models/stage1_pointer/executor_r-seed61/step-002250'
    data: str = 'data/pointer/seed-61-independent'
    device: str = 'cuda'
    seed: int = 83
    train_graphs: int = 2048
    validation_graphs: int = 128
    extraction_batch_size: int = 16
    batch_size: int = 256
    steps: int = 3000
    warmup_steps: int = 100
    learning_rate: float = 0.0003
    min_lr_ratio: float = 0.1
    eval_every: int = 100
    log_every: int = 10
    evaluation_max_depth: int = 16

    def validate(self) -> None:
        for key in ('train_graphs', 'validation_graphs', 'extraction_batch_size', 'batch_size',
                    'steps', 'eval_every', 'log_every', 'evaluation_max_depth'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.device not in ('cpu', 'cuda', 'mps') or not 1 <= self.evaluation_max_depth <= 64:
            raise ValueError('Need a supported device and evaluation depth <=64')
        if (type(self.seed) is not int or not 0 <= self.seed < 2**63 or type(self.warmup_steps) is not int
                or not 0 <= self.warmup_steps < self.steps):
            raise ValueError('Invalid seed/warmup budget')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError('Invalid learning rate/schedule')

    def to_dict(self) -> dict:
        return asdict(self)


def select_graphs(tasks: list, count: int, depths: list[int], seed: int) -> list:
    """Sample distinct tables once, then pair each graph/start with all counts."""
    unique = list({t.mapping_sha256: t for t in tasks}.values())
    random.Random(seed).shuffle(unique)
    return matched_tasks(unique, count, depths)


def learning_rate(config: ControllerConfig, step: int) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    fraction = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    return config.learning_rate * (config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2)


def selection_key(metrics: dict) -> tuple[float, float]:
    """Only trained-count development exact timing, then its stop BCE; lower wins."""
    return -metrics['trained']['exact_stop'], metrics['trained']['loss']


def evaluate_controller(head, features: dict[str, Tensor], tasks: list, trained_depths: list[int],
                        device: str, batch_size: int = 256) -> tuple[dict, list[dict]]:
    """Cached first-crossing evaluation; gold labels cannot enter replay."""
    head.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            scores, _, _ = replay(head, features['context'][start:start + batch_size].to(device),
                                 features['working'][start:start + batch_size].to(device))
            logits.append(scores.cpu())
    logits = torch.cat(logits)
    metadata = [{'example_id': t.example_id, 'depth': t.task_depth} for t in tasks]
    rows = stop_rows(logits, features['predictions'], tasks, metadata, 'controller')
    depths = torch.tensor([t.task_depth for t in tasks])
    mask = torch.arange(logits.shape[1])[None, :] < depths[:, None]
    cohorts = {'trained': depths.new_tensor([d in trained_depths for d in depths.tolist()], dtype=torch.bool),
               'interpolation': depths.new_tensor([d <= max(trained_depths) and d not in trained_depths for d in depths.tolist()], dtype=torch.bool),
               'extrapolation': depths > max(trained_depths)}
    cohorts.update({f'depth_{d}': depths == d for d in sorted(set(depths.tolist()))})
    metrics = {}
    for name, chosen in cohorts.items():
        if chosen.any():
            selected = [r for r, valid in zip(rows, chosen.tolist(), strict=True) if valid]
            metrics[name] = {**aggregate_stops(selected), 'loss': completion_loss(logits[chosen], mask[chosen])[0].item()}
    return metrics, rows


def export_checkpoint(source: Path, destination: Path, head, provenance: dict) -> None:
    """Keep the executor payload bitwise intact; replace only completion_head.*.

    Use the existing full executor format, not a head-only file masquerading as
    a complete checkpoint. Never infer the saved tensor set from requires_grad.
    """
    spec = json.loads((source / 'recurrent_config.json').read_text())
    if spec.get('format') != 'loopformer-executor-v2' or not spec.get('executor', {}).get('isolated'):
        raise ValueError('Need an isolated full executor checkpoint')
    weights = torch.load(source / 'adapter_model.pt', map_location='cpu', weights_only=True)
    new = {f'completion_head.{name}': value.detach().cpu().clone() for name, value in head.state_dict().items()}
    expected = {name for name in weights if name.startswith('completion_head.')}
    if set(new) != expected:
        raise ValueError('Controller tensor names differ from source checkpoint')
    for name, value in new.items():
        if value.shape != weights[name].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid controller tensor {name}')
    weights.update(new)
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in ('adapter_model.pt', 'training_state.pt', 'recurrent_config.json'):
            shutil.copy2(path, destination / path.name)
    torch.save(weights, destination / 'adapter_model.pt')
    # Explicit marker prevents confusing this export with a resumable joint run.
    torch.save({'optimization_scope': 'controller_only', **provenance}, destination / 'training_state.pt')
    (destination / 'recurrent_config.json').write_text(json.dumps({**spec, 'controller_training': provenance}, indent=2) + '\n')

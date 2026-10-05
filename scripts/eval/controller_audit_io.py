"""Validated loading of saved controller runs; never instantiate the executor."""
import csv
import json
from pathlib import Path

import torch

from scripts.eval.pointer_task import sha256_file, load_examples
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.controller import ControllerConfig, select_graphs
from scripts.training.data import read_tasks


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_inputs(run: Path, cache_path: Path | None, checkpoints: list[str]) -> tuple[dict, dict, list[dict], dict]:
    """Verify data, panel order and provenance across all arms before writing output."""
    manifest = read_json(run / 'comparison.json')
    entries = manifest['runs']
    if manifest.get('status') != 'complete' or not entries or len({e['name'] for e in entries}) != len(entries):
        raise ValueError('Need a completed comparison with unique run names')
    path = cache_path or Path(manifest['features_cache'])
    if not path.is_file():
        raise ValueError(f'Missing cached features: {path}. Run this audit on the training desktop; no extraction is performed.')
    cache_hash = sha256_file(path)
    cache = torch.load(path, map_location='cpu', weights_only=True)
    if cache.get('format') != 'controller-training-features-v1':
        raise ValueError('Expected controller-training-features-v1')
    base = ControllerConfig(**read_json(Path(entries[0]['path']) / 'config.json'))
    base.validate()
    source_run = read_json(Path(entries[0]['path']) / 'run.json')
    trained = source_run['trained_depths']
    data = Path(base.data)
    for split in ('train', 'validation'):
        if sha256_file(data / f'{split}.jsonl') != cache['identity'][f'{split}_data_sha256']:
            raise ValueError(f'{split} dataset hash differs from feature cache')
    tasks = {
        'train': select_graphs(read_tasks(data / 'train.jsonl', 'train'), base.train_graphs, trained, base.seed),
        'validation': load_examples(Path(entries[0]['path']) / 'validation_tasks.jsonl'),
    }
    if {t.mapping_sha256 for t in tasks['train']} & {t.mapping_sha256 for t in tasks['validation']}:
        raise ValueError('Train/validation graphs overlap')
    for split, panel in tasks.items():
        if any(t.split != split for t in panel):
            raise ValueError(f'Unexpected split in {split} panel')
        if [t.example_id for t in panel] != cache['identity'][f'{split}_ids']:
            raise ValueError(f'{split} task order differs from cache')
        f = cache['features'][split]
        loops = max(trained) if split == 'train' else base.evaluation_max_depth
        shapes = {'context': (len(panel), cache['width']), 'working': (len(panel), loops, cache['width']),
                  'predictions': (len(panel), loops), 'logits': (len(panel), loops)}
        for key, shape in shapes.items():
            if f[key].shape != shape or not torch.isfinite(f[key]).all():
                raise ValueError(f'Invalid cached {split}/{key}')
        if any(f[k].dtype != torch.float32 for k in ('context', 'working', 'logits')):
            raise ValueError('Audit requires the original float32 cache')
        if f['predictions'].min() < 0 or f['predictions'].max() >= 26:
            raise ValueError('Invalid symbolic prediction index')
    files = {str(path): cache_hash, str(run / 'comparison.json'): sha256_file(run / 'comparison.json')}
    for name in ('train.jsonl', 'validation.jsonl', 'manifest.json'):
        files[str(data / name)] = sha256_file(data / name)
    for entry in entries:
        p = Path(entry['path'])
        config = ControllerConfig(**read_json(p / 'config.json'))
        config.validate()
        if (entry['optimizer_seed'] != (config.optimizer_seed if config.optimizer_seed is not None else config.seed)
                or entry['arm'] != ('remaining' if config.remaining_loss_weight > 0 else 'stop_only')):
            raise ValueError('Manifest seed/arm does not match the saved configuration')
        if {k for k, v in base.to_dict().items() if config.to_dict()[k] != v} - {'optimizer_seed', 'remaining_loss_weight'}:
            raise ValueError('Comparison configs are not matched')
        meta, summary = read_json(p / 'run.json'), read_json(p / 'summary.json')
        if (meta.get('status') != 'complete' or summary.get('status') != 'complete'
                or meta['features_sha256'] != cache_hash or meta['trained_depths'] != trained
                or meta['source_checkpoint_sha256'] != cache['identity']['source_sha256']
                or meta['source_config_sha256'] != cache['identity']['source_config_sha256']
                or meta['train_data_sha256'] != cache['identity']['train_data_sha256']
                or meta['validation_data_sha256'] != cache['identity']['validation_data_sha256']):
            raise ValueError(f'Cache/source/data provenance mismatch: {p}')
        for name, hash_key in [('train_panel.csv', 'train_panel_sha256'), ('validation_tasks.jsonl', 'validation_tasks_sha256')]:
            if sha256_file(p / name) != meta[hash_key] or meta[hash_key] != source_run[hash_key]:
                raise ValueError(f'Panel differs across runs: {p / name}')
        with (p / 'train_panel.csv').open() as handle:
            panel = list(csv.DictReader(handle))
        if [(r['example_id'], int(r['depth']), r['mapping_sha256']) for r in panel] != [
                (t.example_id, t.task_depth, t.mapping_sha256) for t in tasks['train']]:
            raise ValueError('Training panel does not match reconstructed tasks')
        if not config.remaining_readout:
            raise ValueError('This audit requires the saved numerical readout in both arms')
        required = ['config.json', 'run.json', 'summary.json', 'train_panel.csv', 'validation_tasks.jsonl']
        if 'best' in checkpoints:
            required += ['best_controller.pt', 'best_remaining_readout.pt', 'best_selection.json']
        if 'last' in checkpoints:
            required += ['last_controller_state.pt']
        for name in required:
            if not (p / name).is_file():
                raise ValueError(f'Missing saved audit input: {p / name}')
            files[str(p / name)] = sha256_file(p / name)
    return cache, tasks, entries, {'input_sha256': files, 'trained_depths': trained, 'base_config': base.to_dict()}


def load_heads(entry: dict, checkpoint: str, cache: dict, device: str) -> tuple[RecurrentController, torch.nn.Linear, int, ControllerConfig]:
    """Load lightweight best/final heads only, with a matching numerical readout."""
    path = Path(entry['path'])
    config = ControllerConfig(**read_json(path / 'config.json'))
    if checkpoint == 'best':
        head_state = torch.load(path / 'best_controller.pt', map_location='cpu', weights_only=True)
        saved = torch.load(path / 'best_remaining_readout.pt', map_location='cpu', weights_only=True)
        step = read_json(path / 'best_selection.json')['step']
        if (saved['step'] != step or read_json(path / 'summary.json')['best_step'] != step
                or saved['scale'] != config.remaining_scale or saved['width'] != cache['controller_width']):
            raise ValueError('Selected controller/readout checkpoint metadata mismatch')
        readout_state = saved['state_dict']
    elif checkpoint == 'last':
        saved = torch.load(path / 'last_controller_state.pt', map_location='cpu', weights_only=True)
        head_state, readout_state, step = saved['head'], saved['remaining_readout'], saved['step']
        if step != config.steps or readout_state is None:
            raise ValueError('Final controller/readout missing or incomplete')
    else:
        raise ValueError('Checkpoint must be best or last')
    head = RecurrentController(cache['width'], cache['controller_width'])
    readout = torch.nn.Linear(cache['controller_width'], 1)
    head.load_state_dict(head_state)
    readout.load_state_dict(readout_state)
    if not all(torch.isfinite(p).all() for module in (head, readout) for p in module.parameters()):
        raise ValueError('Nonfinite controller/readout weights')
    return head.to(device), readout.to(device), step, config

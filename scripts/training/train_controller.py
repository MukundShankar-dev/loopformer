"""Train only the isolated controller; cache Qwen once and export a portable best model."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import torch
from rich.console import Console

from scripts.eval.controller_diagnostic import progress_bar
from scripts.eval.controller_features import extract_features, replay
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file, synchronize
from scripts.eval.tracking import add_tracking_arguments
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.controller import ControllerConfig, evaluate_controller, export_checkpoint, learning_rate, selection_key, select_graphs
from scripts.training.data import read_tasks
from scripts.training.objective import completion_loss
from scripts.training.controller_remaining import remaining_predictions, remaining_loss, initialization_loss, prefix_losses
from scripts.training.tracking import tracking_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/controller_seed61.json'))
    parser.add_argument('--output', type=Path, default=Path('models/stage1_pointer/controller-seed61'))
    parser.add_argument('--features-cache', type=Path, help='Reuse a previous controller-training cache; starts a new optimizer')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--smoke-test', action='store_true')
    add_tracking_arguments(parser)
    args = parser.parse_args()
    config = ControllerConfig(**json.loads(args.config.read_text()))
    if args.smoke_test:
        config = replace(config, train_graphs=2, validation_graphs=2, batch_size=16, steps=10,
                         warmup_steps=1, eval_every=5, log_every=1)
        args.output = args.output.with_name(args.output.name + f'-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}')
    config.validate()
    source, data = Path(config.source), Path(config.data)
    spec = json.loads((source / 'recurrent_config.json').read_text())
    if spec.get('format') != 'loopformer-executor-v2' or not spec.get('executor', {}).get('isolated'):
        raise ValueError('Controller training requires an isolated executor checkpoint')
    if not (source / 'adapter_model.pt').is_file():
        raise ValueError('Complete source checkpoint weights are required on this device')
    trained_depths = config.requested_counts or spec.get('train_depths') or list(range(1, spec['train_max_depth'] + 1))
    training_loops = config.training_loops or max(trained_depths)
    if training_loops >= config.evaluation_max_depth:
        raise ValueError('Evaluation budget must extend beyond the training loop budget')
    if training_loops < max(trained_depths) and not args.features_cache and not args.dry_run:
        raise ValueError('Short-prefix training requires an explicitly prepared frozen feature cache')
    console = Console()
    console.print('[bold]Controller-only training[/bold] · executor frozen · existing controller initialization')
    console.print(f'{config.train_graphs:,} training graphs × {len(trained_depths)} counts · {config.steps:,} updates · batch {config.batch_size}')
    console.print(f'Trained counts: {trained_depths} · training loops {training_loops} · threshold 0.5')
    console.print('Select on seen-count validation timing within the training loop budget, then BCE')
    console.print(f'Output: {args.output}')
    if args.dry_run:
        console.print(json.dumps(config.to_dict(), indent=2))
        console.print('Preview only: no model loading, extraction, updates or file writes.')
        return
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; change device explicitly for local tiny tests')
    if args.output.exists():
        raise ValueError('Output already exists; choose a new --output (no overwrite)')
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    began = perf_counter()
    train_all = read_tasks(data / 'train.jsonl', 'train')
    validation_all = read_tasks(data / 'validation.jsonl', 'validation')
    if {t.mapping_sha256 for t in train_all} & {t.mapping_sha256 for t in validation_all}:
        raise ValueError('Train/validation rule tables overlap')
    train_tasks = select_graphs(train_all, config.train_graphs, trained_depths, config.seed)
    validation_tasks = select_graphs(validation_all, config.validation_graphs,
                                   list(range(1, config.evaluation_max_depth + 1)), config.seed + 1)
    del train_all, validation_all
    identity = {'source_sha256': sha256_file(source / 'adapter_model.pt'),
                'source_config_sha256': sha256_file(source / 'recurrent_config.json'),
                'train_data_sha256': sha256_file(data / 'train.jsonl'),
                'validation_data_sha256': sha256_file(data / 'validation.jsonl'),
                'train_ids': [t.example_id for t in train_tasks],
                'validation_ids': [t.example_id for t in validation_tasks]}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config.to_dict(), indent=2) + '\n')
    write_csv(output / 'train_panel.csv', [{'example_id': t.example_id, 'mapping_sha256': t.mapping_sha256,
        'seed': t.seed, 'start': t.initial_state, 'depth': t.task_depth} for t in train_tasks])
    (output / 'validation_tasks.jsonl').write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in validation_tasks))
    if args.features_cache:
        cache = torch.load(args.features_cache, map_location='cpu', weights_only=True)
        if cache.get('format') != 'controller-training-features-v1' or cache['identity'] != identity:
            raise ValueError('Cache source/data/selected tasks differ from this run')
        head = RecurrentController(cache['width'], cache['controller_width'])
        head.load_state_dict(cache['head'])
    else:
        model, tokenizer, loaded_spec = load_recurrent_checkpoint(source, device=config.device)
        model.config._attn_implementation = 'sdpa'
        cache = {'format': 'controller-training-features-v1', 'identity': identity,
                 'extraction': {'device': config.device, 'precision': 'float32', 'attention': 'sdpa',
                                'batch_size': config.extraction_batch_size},
                 'features': {}, 'replay_error': {}}
        for name, tasks, loops in [('train', train_tasks, training_loops),
                                   ('validation', validation_tasks, config.evaluation_max_depth)]:
            with progress_bar() as progress:
                bar = progress.add_task(f'Cache frozen {name} features', total=len(tasks))
                values, head, error = extract_features(model, tokenizer, loaded_spec['token_ids'], tasks,
                    loops, config.extraction_batch_size, progress=lambda done: progress.update(bar, completed=done))
            cache['features'][name] = {key: values[key] for key in ('context', 'working', 'predictions', 'logits')}
            cache['replay_error'][name] = error
        cache.update(width=model.config.hidden_size, controller_width=head.intermediate, head=head.state_dict())
        del model, tokenizer, values
        torch.save(cache, output / 'features.pt')
        if config.device == 'cuda':
            torch.cuda.empty_cache()
    if sha256_file(source / 'adapter_model.pt') != identity['source_sha256']:
        raise RuntimeError('Source executor checkpoint changed')
    synchronize(torch.device(config.device))
    extraction_seconds = perf_counter() - began
    features = cache['features']
    for name, tasks, loops in [('train', train_tasks, training_loops),
                               ('validation', validation_tasks, config.evaluation_max_depth)]:
        f = features[name]
        if (f['context'].shape != (len(tasks), cache['width']) or
                f['working'].shape != (len(tasks), loops, cache['width']) or
                f['predictions'].shape != (len(tasks), loops)):
            raise ValueError(f'Invalid {name} feature shapes')
        if not all(torch.isfinite(t).all() for t in f.values()):
            raise ValueError(f'Nonfinite {name} features')
    metadata = {'status': 'running', 'source': str(source), 'source_checkpoint_sha256': identity['source_sha256'],
        'source_config_sha256': identity['source_config_sha256'], 'data': str(data),
        'train_data_sha256': identity['train_data_sha256'], 'validation_data_sha256': identity['validation_data_sha256'],
        'source_sha256': {str(p): sha256_file(p) for pattern in ('scripts/training/*.py', 'scripts/eval/*.py', 'scripts/recurrent_qwen/*.py') for p in Path('.').glob(pattern)},
        'train_panel_sha256': sha256_file(output / 'train_panel.csv'),
        'validation_tasks_sha256': sha256_file(output / 'validation_tasks.jsonl'),
        'features_cache': str(args.features_cache or output / 'features.pt'),
        'features_sha256': sha256_file(args.features_cache or output / 'features.pt'),
        'extraction': cache['extraction'], 'live_replay_error': cache['replay_error'],
        'extraction_seconds': extraction_seconds, 'python': platform.python_version(), 'torch': torch.__version__,
        'command': [sys.executable, '-m', 'scripts.training.train_controller', *sys.argv[1:]],
        'selection': 'maximize seen-count development exact stopping within training loop budget; tie-break BCE',
        'optimization_scope': 'controller_only', 'trained_depths': trained_depths,
        'training_loops': training_loops,
        'trainable_parameters': sum(p.numel() for p in head.parameters()),
        'remaining_readout_parameters': head.intermediate + 1 if config.remaining_readout else 0,
        'remaining_gradient_scope': ('controller_and_readout' if config.remaining_loss_weight or config.initial_loss_weight else 'detached_readout_only')
                                   if config.remaining_readout else None,
        'cached_tensor_bytes': sum(t.numel() * t.element_size() for group in features.values() for t in group.values())}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    head = head.to(config.device).requires_grad_(True)
    optimizer_seed = config.seed if config.optimizer_seed is None else config.optimizer_seed
    torch.manual_seed(optimizer_seed)
    remaining_readout = torch.nn.Linear(head.intermediate, 1).to(config.device) if config.remaining_readout else None
    if remaining_readout is not None:
        console.print(f'Remaining-work readout · weight {config.remaining_loss_weight:g} · fixed scale {config.remaining_scale:g}'
                      + (' · detached measurement control' if config.remaining_loss_weight == 0 else ' · trains controller memory'))
        if config.initial_loss_weight:
            console.print(f'Independent initialization loss · weight {config.initial_loss_weight:g} · no trajectory averaging')
    console.print(f'Frozen features ready · {sum(p.numel() for p in head.parameters()):,} controller parameters · starting optimizer')
    torch.manual_seed(optimizer_seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=0.0)
    readout_optimizer = (torch.optim.AdamW(remaining_readout.parameters(), lr=config.learning_rate, weight_decay=0.0)
                         if remaining_readout is not None else None)
    context = features['train']['context'].to(config.device)
    working = features['train']['working'].to(config.device)
    depths = torch.tensor([t.task_depth for t in train_tasks], device=config.device)
    masks = torch.arange(working.shape[1], device=config.device)[None, :] < depths[:, None]
    generator = torch.Generator().manual_seed(optimizer_seed)
    order, offset = torch.empty(0, dtype=torch.long), 0
    best_key, best_step, best_metrics = (float('inf'), float('inf')), 0, None
    validation_history = []
    if config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    training_began = perf_counter()
    with tracking_run(output, metadata, config.to_dict(), mode=args.wandb_mode,
                      project=args.wandb_project, entity=args.wandb_entity) as tracker, (output / 'metrics.jsonl').open('w') as log:
        def record(event: dict) -> None:
            log.write(json.dumps(event) + '\n'); log.flush(); tracker.log(event)

        def validate(step: int) -> dict:
            nonlocal best_key, best_step, best_metrics
            metrics, rows = evaluate_controller(head, features['validation'], validation_tasks, trained_depths, config.device,
                remaining_readout=remaining_readout, remaining_scale=config.remaining_scale,
                training_loops=training_loops if config.training_loops is not None else None)
            record({'event': 'validation', 'step': step, 'validation': metrics})
            validation_history.extend({'step': step, 'cohort': name, **values} for name, values in metrics.items())
            write_csv(output / 'validation_history.csv', validation_history)
            if step == 0:
                write_csv(output / 'baseline_decisions.csv', rows)
            key = selection_key({'trained': metrics['selection']} if 'selection' in metrics else metrics)
            if key < best_key:
                best_key, best_step, best_metrics = key, step, metrics
                torch.save({k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, output / 'best_controller.pt')
                if remaining_readout is not None:
                    torch.save({'state_dict': remaining_readout.state_dict(), 'scale': config.remaining_scale,
                                'step': step, 'width': head.intermediate}, output / 'best_remaining_readout.pt')
                write_csv(output / 'best_decisions.csv', rows)
                (output / 'best_selection.json').write_text(json.dumps({'step': step, 'metrics': metrics}, indent=2) + '\n')
            return metrics

        latest = validate(0)
        last_validation_step = 0
        # Verify cached baseline logits also agree when replayed by the training evaluator.
        with torch.no_grad():
            count = min(16, len(validation_tasks))
            logits, _, _ = replay(head, features['validation']['context'][:count].to(config.device),
                                 features['validation']['working'][:count].to(config.device))
            torch.testing.assert_close(logits.cpu(), features['validation']['logits'][:count], atol=2e-5, rtol=1e-4)
        with progress_bar() as progress:
            bar = progress.add_task('Controller training', total=config.steps)
            for step in range(1, config.steps + 1):
                if offset >= len(order):
                    order = torch.randperm(len(train_tasks), generator=generator)
                    offset = 0
                idx = order[offset:offset + config.batch_size].to(config.device)
                offset += len(idx)
                head.train()
                lr = learning_rate(config, step)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                logits, initial, memories = replay(head, context[idx], working[idx])
                stop_loss = (completion_loss(logits, masks[idx])[0] if config.training_loops is None
                             else logits.new_zeros(()))
                auxiliary = None
                initial_loss = None
                loss = stop_loss
                if remaining_readout is not None:
                    readout_optimizer.zero_grad(set_to_none=True)
                    for group in readout_optimizer.param_groups:
                        group['lr'] = lr
                    predicted = remaining_predictions(remaining_readout, initial, memories, config.remaining_scale,
                                                       detach=config.remaining_loss_weight == 0 and config.initial_loss_weight == 0)
                    if config.training_loops is not None:
                        stop_loss, auxiliary = prefix_losses(logits, predicted, depths[idx], config.remaining_scale)
                    else:
                        auxiliary = remaining_loss(predicted, depths[idx], config.remaining_scale)
                    initial_loss = initialization_loss(predicted, depths[idx], config.remaining_scale)
                    loss = (stop_loss + config.remaining_loss_weight * auxiliary
                            + config.initial_loss_weight * initial_loss)
                # At weight zero the auxiliary trains only a detached measurement probe.
                backward_loss = (loss + auxiliary if auxiliary is not None and config.remaining_loss_weight == 0
                                 and config.initial_loss_weight == 0 else loss)
                backward_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if readout_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(remaining_readout.parameters(), 1.0, error_if_nonfinite=True)
                    readout_optimizer.step()
                if step % config.log_every == 0 or step == 1 or step == config.steps:
                    with torch.no_grad():
                        exact = (logits >= 0).any(-1) & ((logits >= 0).int().argmax(-1) + 1 == depths[idx])
                    record({'event': 'train', 'step': step, 'train': {'loss': loss.item(), 'stop_loss': stop_loss.item(),
                            **({'remaining_loss': auxiliary.item()} if auxiliary is not None else {}),
                            **({'initial_loss': initial_loss.item()} if initial_loss is not None else {}),
                            'exact_stop': exact.float().mean().item()},
                        'learning_rate': lr, 'gradient_norm_before_clip': float(norm),
                        'training_elapsed_seconds': perf_counter() - training_began,
                        'cuda_peak_bytes': torch.cuda.max_memory_allocated() if config.device == 'cuda' else None})
                if step % config.eval_every == 0 or step == config.steps:
                    latest = validate(step)
                    last_validation_step = step
                    torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                                'remaining_readout': remaining_readout.state_dict() if remaining_readout is not None else None,
                                'remaining_optimizer': readout_optimizer.state_dict() if readout_optimizer is not None else None},
                               output / 'last_controller_state.pt')
                if step % config.log_every == 0 or step == config.steps:
                    progress.update(bar, completed=step,
                        description=f'Controller · val@{last_validation_step} {latest.get("selection", latest["trained"])["exact_stop"]:.1%}')
        # Publish only the selected controller with the original frozen executor.
        head.load_state_dict(torch.load(output / 'best_controller.pt', map_location=config.device, weights_only=True))
        if remaining_readout is not None:
            saved_readout = torch.load(output / 'best_remaining_readout.pt', map_location=config.device, weights_only=True)
            remaining_readout.load_state_dict(saved_readout['state_dict'])
            traces = []
            evaluate_controller(head, features['validation'], validation_tasks, trained_depths, config.device,
                remaining_readout=remaining_readout, remaining_scale=config.remaining_scale, traces=traces,
                training_loops=training_loops if config.training_loops is not None else None)
            write_csv(output / 'remaining_trajectories.csv', traces)
        provenance = {'source_checkpoint_sha256': identity['source_sha256'], 'step': best_step,
                      'optimization_scope': 'controller_only', 'trained_depths': trained_depths,
                      'training_loops': training_loops,
                      'config': config.to_dict(), 'selection': metadata['selection']}
        if sha256_file(source / 'adapter_model.pt') != identity['source_sha256']:
            raise RuntimeError('Source checkpoint changed before export')
        export_checkpoint(source, output / 'best', head, provenance)
        (output / 'best_checkpoint.json').write_text(json.dumps({'path': 'best', 'step': best_step}) + '\n')
        synchronize(torch.device(config.device))
        summary = {'status': 'complete', 'best_step': best_step, 'validation': best_metrics,
                   'best_checkpoint': str(output / 'best'), 'extraction_seconds': extraction_seconds,
                   'training_and_export_seconds': perf_counter() - training_began, 'wall_seconds': perf_counter() - began,
                   'frozen_executor_source_sha256': identity['source_sha256'],
                   'exported_checkpoint_sha256': sha256_file(output / 'best/adapter_model.pt')}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        tracker.summary(summary)
    metadata['status'] = 'complete'
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    console.print(f'[green]Complete[/green] · best step {best_step} · trained-count validation exact stop {best_metrics["trained"]["exact_stop"]:.2%}')
    console.print(f'Saved portable checkpoint: {output / "best"}')


if __name__ == '__main__':
    main()

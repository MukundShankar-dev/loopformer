"""Frozen paired checkpoint diagnostics, composing existing inference and scoring.

Count-free execution is reused only after exact non-controller identity checks.
Legacy count-dependent R runs separately for every requested count. No fitting.
"""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import torch
from rich.console import Console
from rich.progress import track

from scripts.dataset.benchmark import count_variant
from scripts.dataset.pointer import PointerExample, SYMBOLS, validate_example
from scripts.eval.audit_pointer_benchmark import audit_native_results
from scripts.eval.controller_features import extract_features, replay
from scripts.eval.controller_learning import stop_rows
from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes, controller_panel, verify_native_decisions
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.controller_cache import prompt_features
from scripts.training.data import collate, encode_tasks
from scripts.training.objective import forward_symbols


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def executor_spec(spec: dict) -> dict:
    """Exclude only controller architecture/fit metadata; preserve every R option."""
    result = {k: v for k, v in spec.items() if k not in ('controller_training', 'completion_head')}
    result['executor'] = {k: v for k, v in spec['executor'].items()
                          if k not in ('controller_kind', 'controller_size')}
    return result


def assert_executor_tensors(old: dict, new: dict) -> int:
    preserved = {k for k in old if not k.startswith('completion_head.')}
    if preserved != {k for k in new if not k.startswith('completion_head.')}:
        raise ValueError('Executor tensor names differ; do not reuse trajectories')
    for name in preserved:
        torch.testing.assert_close(old[name], new[name], atol=0, rtol=0)
    return len(preserved)


def verify_reuse(source: Path, candidate: Path, source_summary: dict) -> dict:
    if checkpoint_hashes(source) != source_summary['checkpoint_sha256']:
        raise ValueError('Confirmed source checkpoint changed')
    specs = [json.loads((p / 'recurrent_config.json').read_text()) for p in (source, candidate)]
    if executor_spec(specs[0]) != executor_spec(specs[1]):
        raise ValueError('Non-controller architecture differs; do not reuse trajectories')
    old_files, new_files = checkpoint_hashes(source), checkpoint_hashes(candidate)
    for name, value in old_files.items():
        if name not in ('adapter_model.pt', 'recurrent_config.json') and new_files.get(name) != value:
            raise ValueError('Tokenizer/inference payload changed')
    code = {}
    for path in sorted(Path('scripts/recurrent_qwen').glob('*.py')):
        original = subprocess.check_output(['git', 'show', f'{source_summary["git_head"]}:{path}'])
        expected = hashlib.sha256(original).hexdigest()
        if sha256_file(path) != expected:
            raise ValueError(f'Recurrent inference implementation changed: {path}')
        code[str(path)] = expected
    old = torch.load(source / 'adapter_model.pt', weights_only=True, map_location='cpu')
    new = torch.load(candidate / 'adapter_model.pt', weights_only=True, map_location='cpu')
    tensors = assert_executor_tensors(old, new)
    return {'passed': True, 'unchanged_non_controller_tensors': tensors,
            'non_controller_spec_unchanged': True, 'tokenizer_bytes_unchanged': True,
            'inference_code_sha256': code}


def nominal_fields(task, predictions: list[int]) -> dict:
    """Requested-depth forced trace, independent of the controller's actual stop."""
    if len(predictions) < task.task_depth:
        raise ValueError('Forced trajectory must cover requested depth')
    predicted = ''.join(SYMBOLS[p] for p in predictions[:task.task_depth])
    target = ''.join(task.intermediate_states)
    first = next((i + 1 for i, (a, b) in enumerate(zip(predicted, target, strict=True)) if a != b), None)
    return {'forced_predictions': predicted, 'targets': target, 'first_error_loop': first,
            'nominal_final_correct': predicted[-1] == target[-1], 'complete_trajectory': first is None}


METRICS = ('nominal_final_correct', 'complete_trajectory', 'exact_stop', 'early_stop',
           'late_stop', 'missing_stop', 'stopped_answer_correct', 'joint_success',
           'strict_success', 'correct_letter_wrong_time')
STOP_METRICS = METRICS[2:]


def aggregate(rows: list[dict]) -> dict:
    """Absent learned stopping is undefined, never scored as success or failure."""
    result = {'queries': len(rows)}
    for key in METRICS:
        values = [r[key] for r in rows if r[key] is not None]
        if values and len(values) != len(rows):
            raise ValueError('Mixed metric applicability within an arm')
        result[key] = sum(values) / len(values) if values else None
    return result


def compose_row(task, metadata: dict, stopped: dict, predictions: list[int]) -> dict:
    # Keep a common decision schema across legacy/native and controller replay.
    # Probability traces remain available in the native evaluator artifacts.
    shared = {k: v for k, v in stopped.items() if k not in ('condition', 'executor_nominal_correct', 'stop_probability')}
    row = {**metadata, **shared, **nominal_fields(task, predictions)}
    row['strict_success'] = row['complete_trajectory'] and row['exact_stop']
    row['correct_letter_wrong_time'] = row['stopped_answer_correct'] and not row['exact_stop']
    return row


def assemble_comparison(config: dict, output: Path) -> dict:
    """Validate completed frozen arms and assemble exports without model inference."""
    freeze = json.loads((output / 'freeze.json').read_text())
    if freeze['config'] != config:
        raise ValueError('Cannot assemble under a changed protocol')
    combined, results = [], {}
    for arm in config['models']:
        name = arm['name']; arm_output = output / name
        summary = json.loads((arm_output / 'summary.json').read_text())
        if (not summary['native_fidelity']['passed'] or not summary['inference_files_unchanged']
                or checkpoint_hashes(Path(arm['model'])) != freeze['inference_sha256'][name]):
            raise ValueError('Cannot assemble unverified or changed checkpoint')
        rows = read_csv(arm_output / 'decisions.csv')
        if len(rows) != summary['queries']:
            raise ValueError('Arm decision coverage differs')
        for row in rows:
            for key in ('condition', 'executor_nominal_correct', 'stop_probability'):
                row.pop(key, None)
            for key in METRICS:
                headless = arm['mode'] == 'forced_only' if 'mode' in arm else False
                if headless and key in STOP_METRICS and row[key] == '':
                    row[key] = None
                    continue
                if row[key] not in ('True', 'False'):
                    raise ValueError('Invalid boolean result')
                row[key] = row[key] == 'True'
            for key in ('depth', 'graph_index', 'dataset_seed', 'cycle_period', 'transient_length'):
                row[key] = int(row[key])
            for key in ('first_stop', 'first_error_loop', 'executed_loops'):
                row[key] = int(row[key]) if row[key] else None
        measured = aggregate(rows)
        if any((measured[key] is None) != (summary[key] is None) or
               (measured[key] is not None and abs(measured[key] - summary[key]) > 1e-12) for key in METRICS):
            raise ValueError('Saved arm summary differs from decisions')
        results[name] = summary
        combined.extend(rows)
    write_csv(output / 'decisions.csv', combined)
    result = {'status': 'complete', 'config': config, 'graphs': len(freeze['selected_graph_indices']),
        'questions_per_model': len(freeze['selected_graph_indices']) * len(config['counts']),
        'models': results, 'inference_files_unchanged': True,
        'scope': 'Paired descriptive diagnostic on opened benchmark; legacy training/capacity/data differ; isolated arms have identical non-controller tensors.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def legacy_evaluation(config: dict, output: Path, tasks: list, metadata: list[dict],
                      device: str, batch_size: int) -> tuple[list[dict], dict]:
    """Actual learned-stop calls plus forced nominal traces for coupled legacy R."""
    if config.get('mode') == 'forced_only':
        return forced_only_evaluation(config, output, tasks, metadata, device, batch_size)
    path = output / 'native_tasks.jsonl'
    path.write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in tasks))
    subprocess.run([sys.executable, '-u', '-m', 'scripts.eval.loop_test', '--model', config['model'],
        '--data', str(path), '--device', device, '--attention', 'sdpa', '--loops', str(config['safety_cap']),
        '--stop-policy', 'completion', '--stop-threshold', '.5', '--output', str(output / 'native'),
        '--wandb-mode', 'disabled'], check=True)
    audit = audit_native_results(output / 'native', path)
    (output / 'native/native_reference_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    decisions = {r['example_id']: r for r in read_csv(output / 'native/decisions.csv')}
    native_predictions = {}
    for row in read_csv(output / 'native/trajectories.csv'):
        native_predictions.setdefault(row['example_id'], []).append(row['prediction'])
    model, tokenizer, spec = load_recurrent_checkpoint(Path(config['model']), device=device)
    model.eval().requires_grad_(False); model.config._attn_implementation = 'sdpa'
    if model.router is not None:
        raise ValueError('Legacy arm must retain its count-dependent prompt routing')
    rows, matched = [], 0
    for depth in track(sorted({t.task_depth for t in tasks}), description='Legacy forced nominal traces'):
        indices = [i for i, t in enumerate(tasks) if t.task_depth == depth]
        items = encode_tasks([tasks[i] for i in indices], tokenizer,
                             dict(zip(SYMBOLS, spec['token_ids'], strict=True)), 512)
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size], tokenizer.pad_token_id, device, depth)
            with torch.inference_mode():
                _, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'],
                                            spec['token_ids'], num_loops=depth)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite legacy forced readout')
            for i, predicted in zip(indices[start:start + batch_size], scores.argmax(-1).cpu().tolist(), strict=True):
                task, actual = tasks[i], decisions[tasks[i].example_id]
                prefix = native_predictions[task.example_id][:depth]
                if prefix != [SYMBOLS[p] for p in predicted[:len(prefix)]]:
                    raise ValueError('Legacy batched forced prefix differs from actual stopped inference')
                matched += len(prefix)
                missing = actual['cap_fallback'] == 'True'
                stopped = {k: actual[k] == 'True' for k in ('exact_stop', 'early_stop', 'late_stop', 'joint_success')}
                stopped.update(first_stop=None if missing else int(actual['executed_loops']),
                    executed_loops=int(actual['executed_loops']), missing_stop=missing,
                    stopped_answer_correct=actual['correct'] == 'True', prediction=actual['prediction'], target=actual['target'])
                rows.append(compose_row(task, metadata[i], stopped, predicted))
    return rows, {'passed': True, 'questions': len(tasks), 'native_nominal_prefix_matches': matched,
                  'native_reference_audit': audit}


def forced_only_evaluation(config: dict, output: Path, tasks: list, metadata: list[dict],
                           device: str, batch_size: int) -> tuple[list[dict], dict]:
    """Historical CE checkpoint: externally forced depth, no invented stop policy.

    R sees the full prompt, so each displayed count is executed separately.
    Independently invoked batch-one calls verify selected batched predictions.
    """
    model, tokenizer, spec = load_recurrent_checkpoint(Path(config['model']), device=device)
    model.eval().requires_grad_(False); model.config._attn_implementation = 'sdpa'
    if model.router is not None or model.completion_head is not None:
        raise ValueError('Forced-only historical arm must have no controller or router')
    rows, native = [], []
    first_indices, seen = set(), set()
    for meta in metadata:
        stratum = meta['dataset_seed'], meta['graph_mode']
        if stratum not in seen:
            first_indices.add(meta['graph_index']); seen.add(stratum)
    for depth in track(sorted({t.task_depth for t in tasks}), description='CE-only forced traces'):
        indices = [i for i, task in enumerate(tasks) if task.task_depth == depth]
        items = encode_tasks([tasks[i] for i in indices], tokenizer,
                             dict(zip(SYMBOLS, spec['token_ids'], strict=True)), 512)
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size], tokenizer.pad_token_id, device, depth)
            with torch.inference_mode():
                _, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'], spec['token_ids'], num_loops=depth)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite CE-only readout')
            predicted = scores.argmax(-1).cpu().tolist()
            for offset, (i, trace) in enumerate(zip(indices[start:start + batch_size], predicted, strict=True)):
                task = tasks[i]
                row = {**metadata[i], **nominal_fields(task, trace),
                       **dict.fromkeys(STOP_METRICS), 'first_stop': None,
                       'executed_loops': None, 'prediction': None, 'target': task.final_state}
                rows.append(row)
                if metadata[i]['graph_index'] in first_indices and depth in config['native_counts']:
                    single = collate([items[start + offset]], tokenizer.pad_token_id, device, depth)
                    with torch.inference_mode():
                        _, actual = forward_symbols(model, single['input_ids'], single['attention_mask'], spec['token_ids'], num_loops=depth)
                    if not torch.isfinite(actual).all():
                        raise FloatingPointError('Nonfinite CE-only batch-one readout')
                    letters = ''.join(SYMBOLS[p] for p in actual.argmax(-1)[0].cpu().tolist())
                    if letters != row['forced_predictions']:
                        raise ValueError('CE-only batch-one trace differs from batched forced trace')
                    native.append({**metadata[i], 'forced_predictions': letters, 'targets': row['targets']})
    write_csv(output / 'forced_native.csv', native)
    return rows, {'passed': True, 'questions': len(native), 'scope': 'Batch-one forced inference; learned stopping not applicable',
                  'trace_sha256': sha256_file(output / 'forced_native.csv')}


def reuse_saved_arm(arm: dict, config: dict, freeze: dict, output: Path) -> None:
    """Reuse already audited calls only with identical inputs, policy and weights."""
    from scripts.eval.audit_checkpoint_comparison import audit_comparison
    source = Path(arm['saved_results'])
    old = json.loads((source / 'freeze.json').read_text())
    old_arm = next(a for a in old['config']['models'] if a['name'] == arm['name'])
    if (old_arm['model'] != arm['model'] or old_arm['mode'] != arm['mode'] or
            old['inference_sha256'][arm['name']] != freeze['inference_sha256'][arm['name']] or
            old['selected_graph_indices'] != freeze['selected_graph_indices'] or
            old['source_graphs_sha256'] != freeze['source_graphs_sha256'] or
            old['source_summary_sha256'] != freeze['source_summary_sha256'] or
            any(old['config'][key] != config[key] for key in ('counts', 'native_counts', 'threshold', 'safety_cap', 'batch_size', 'device'))):
        raise ValueError('Saved arm differs from the new frozen paired protocol')
    for path in sorted(Path('scripts/recurrent_qwen').glob('*.py')):
        original = subprocess.check_output(['git', 'show', f'{old["git_head"]}:{path}'])
        if hashlib.sha256(original).hexdigest() != sha256_file(path):
            raise ValueError('Inference implementation changed since saved calls')
    audit = audit_comparison(source, Path(config['graphs']))
    shutil.copytree(source / arm['name'], output)
    rows = read_csv(output / 'decisions.csv')
    for row in rows:
        row['model_label'] = arm['label']
    write_csv(output / 'decisions.csv', rows)
    summary = json.loads((output / 'summary.json').read_text())
    summary['label'] = arm['label']
    summary['reused_observations'] = True
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'saved_arm_reuse.json').write_text(json.dumps({'passed': True, 'source': str(source),
        'source_freeze_sha256': sha256_file(source / 'freeze.json'), 'source_audit': audit,
        'native_files_copied': True, 'checkpoint_and_inference_code_unchanged': True}, indent=2) + '\n')


def isolated_evaluation(config: dict, output: Path, graphs: list, graph_metadata: list[dict],
                        tasks: list, metadata: list[dict], predictions: torch.Tensor,
                        device: str, batch_size: int, native_counts: list[int]) -> tuple[list[dict], dict]:
    """Replay native controller inputs; GRU always observes the actual R vectors."""
    model, tokenizer, spec = load_recurrent_checkpoint(Path(config['model']), device=device)
    model.eval().requires_grad_(False); model.config._attn_implementation = 'sdpa'
    cap = config['safety_cap']
    if model.router is None:
        raise ValueError('Need isolated executor for shared graph trajectories')
    if model.completion_head.kind in ('affine_suffix', 'shared_number'):
        logits, _, counts = controller_panel(model, tokenizer, spec['token_ids'], graphs, graph_metadata, 256, cap)
        write_csv(output / 'controller_counts.csv', counts)
        question_logits = torch.stack([logits[t.task_depth - 1] for t in tasks])
    else:
        features, _, error = extract_features(model, tokenizer, spec['token_ids'], graphs, cap, batch_size)
        torch.testing.assert_close(features['predictions'], predictions, atol=0, rtol=0)
        # The count-free vectors are captured once, but the GRU initialization is
        # recomputed from every full Rules/Start/Steps prompt, never broadcast.
        context = prompt_features(model, tokenizer, spec['token_ids'], tasks, batch_size)
        lookup = {m['graph_index']: i for i, m in enumerate(graph_metadata)}
        chunks = []
        for start in track(range(0, len(tasks), 64), description='Replay graph-dependent GRU'):
            indices = [lookup[m['graph_index']] for m in metadata[start:start + 64]]
            with torch.inference_mode():
                repeated, _, _ = replay(model.completion_head, context[start:start + 64].to(device),
                                         features['working'][indices].to(device))
            chunks.append(repeated.cpu())
        question_logits = torch.cat(chunks)
        (output / 'feature_fidelity.json').write_text(json.dumps({'passed': True,
            'max_live_replay_logit_error': error, 'source_predictions_exact': True,
            'graph_count': len(graphs), 'context_per_raw_prompt': True}) + '\n')
    lookup = {m['graph_index']: i for i, m in enumerate(graph_metadata)}
    question_predictions = predictions[[lookup[m['graph_index']] for m in metadata]]
    decisions = stop_rows(question_logits, question_predictions, tasks, metadata, 'matched')
    rows = [compose_row(t, m, r, p.tolist()) for t, m, r, p in
            zip(tasks, metadata, decisions, question_predictions, strict=True)]
    # Selection is the first graph in each declared seed/mode stratum, by order.
    first_indices = set()
    seen = set()
    for meta in graph_metadata:
        key = meta['dataset_seed'], meta['graph_mode']
        if key not in seen:
            first_indices.add(meta['graph_index']); seen.add(key)
    selected = [i for i, m in enumerate(metadata) if m['graph_index'] in first_indices and m['depth'] in native_counts]
    fidelity = verify_native_decisions(config, output, [tasks[i] for i in selected],
                                      [decisions[i] for i in selected], device)
    audit = audit_native_results(output / 'native', output / 'native_tasks.jsonl')
    (output / 'native/native_reference_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return rows, fidelity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pointer_checkpoint_comparison.json'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--assemble-only', action='store_true', help='Validate completed saved arms and assemble CSV/summary, without inference')
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    console = Console()
    console.print(f'[cyan]Frozen checkpoint comparison: {len(config["models"])} models · {config["graphs_per_stratum"]} graphs per stratum · {len(config["counts"])} counts[/cyan]')
    if args.dry_run:
        console.print(config); return
    output = Path(config['output'])
    if args.assemble_only:
        if (output / 'summary.json').exists():
            raise ValueError('Completed comparison exists; refuse overwrite')
        assemble_comparison(config, output)
        console.print('[green]Assembled verified saved arms without inference.[/green]'); return
    if (output.exists() or config['threshold'] != .5 or config['safety_cap'] != 272
            or config['counts'] != sorted(set(config['counts']))
            or not config['counts'] or not 1 <= min(config['counts']) <= max(config['counts']) <= 256
            or config['batch_size'] < 1 or config['graphs_per_stratum'] < 1
            or not set(config['native_counts']) <= set(config['counts'])
            or any(a['mode'] not in ('legacy', 'isolated', 'forced_only') for a in config['models'])
            or len({a['name'] for a in config['models']}) != len(config['models'])):
        raise ValueError('Need fresh output and declared stopping protocol')
    source = Path(config['source_results'])
    summary = json.loads((source / 'summary.json').read_text())
    audit = json.loads((source / 'result_reference_audit.json').read_text())
    manifest = json.loads((source / 'dataset_manifest.json').read_text())
    if not audit['passed'] or audit['summary_sha256'] != sha256_file(source / 'summary.json'):
        raise ValueError('Need audited completed source benchmark')
    if sha256_file(Path(config['graphs'])) != manifest['graphs_sha256']:
        raise ValueError('Benchmark graph bytes changed')
    all_graphs = [PointerExample(**json.loads(line)) for line in Path(config['graphs']).read_text().splitlines()]
    for graph in all_graphs:
        validate_example(graph)
    source_rows = read_csv(source / 'graphs.csv')
    selected, strata = [], Counter()
    for i, meta in enumerate(manifest['graph_metadata']):
        key = meta['dataset_seed'], meta['graph_mode']
        if strata[key] < config['graphs_per_stratum']:
            selected.append(i)
        strata[key] += 1
    graphs = [all_graphs[i] for i in selected]
    graph_metadata = [manifest['graph_metadata'][i] for i in selected]
    predictions = torch.tensor([[SYMBOLS.index(c) for c in source_rows[i]['predictions']] for i in selected])
    tasks = [count_variant(t, depth) for t in graphs for depth in config['counts']]
    metadata = [{**m, 'example_id': t.example_id, 'depth': depth} for m, graph in zip(graph_metadata, graphs, strict=True)
                for depth in config['counts'] for t in [count_variant(graph, depth)]]
    before = {arm['name']: checkpoint_hashes(Path(arm['model'])) for arm in config['models']}
    output.mkdir(parents=True)
    freeze = {'config': config, 'config_sha256': sha256_file(args.config),
        'inference_sha256': before, 'source_summary_sha256': sha256_file(source / 'summary.json'),
        'source_graphs_sha256': sha256_file(Path(config['graphs'])), 'selected_graph_indices': selected,
        'panel_selection': 'First graphs in each seed/mode, independent of outcomes; opened-panel diagnosis, not fresh confirmation',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (output / 'freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    torch.set_num_threads(4); torch.manual_seed(239); torch.use_deterministic_algorithms(True)
    for arm in config['models']:
        console.print(f'[bold cyan]{arm["label"]}[/bold cyan]'); began = perf_counter()
        arm_output = output / arm['name']
        if arm.get('saved_results'):
            reuse_saved_arm(arm, config, freeze, arm_output)
            console.print('[green]Reused audited observations with unchanged weights, inputs and inference code.[/green]')
            continue
        arm_output.mkdir()
        settings = {**config, 'model': arm['model'], 'mode': arm['mode']}
        if arm['mode'] in ('legacy', 'forced_only'):
            rows, fidelity = legacy_evaluation(settings, arm_output, tasks, metadata, config['device'], config['batch_size'])
        else:
            identity = verify_reuse(Path(summary['checkpoint']), Path(arm['model']), summary)
            (arm_output / 'executor_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
            rows, fidelity = isolated_evaluation(settings, arm_output, graphs, graph_metadata, tasks, metadata,
                predictions, config['device'], config['batch_size'], config['native_counts'])
        for row in rows:
            row.update(model=arm['name'], model_label=arm['label'])
        write_csv(arm_output / 'decisions.csv', rows)
        per_count = [{'model': arm['name'], 'depth': n, **aggregate([r for r in rows if r['depth'] == n])} for n in config['counts']]
        write_csv(arm_output / 'per_count.csv', per_count)
        if checkpoint_hashes(Path(arm['model'])) != before[arm['name']]:
            raise ValueError('Frozen inference files changed')
        arm_summary = {'label': arm['label'], **aggregate(rows), 'seconds': perf_counter() - began,
            'native_fidelity': fidelity, 'inference_files_unchanged': True}
        (arm_output / 'summary.json').write_text(json.dumps(arm_summary, indent=2) + '\n')
        gc.collect(); torch.cuda.empty_cache()
    assemble_comparison(config, output)
    console.print('[green]All frozen comparison arms complete, reference-audited, and fidelity checked.[/green]')


if __name__ == '__main__':
    main()

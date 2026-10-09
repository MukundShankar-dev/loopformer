"""Resumable full-population extraction using shared model/scoring interfaces.

One independent count-specific suffix state for each historical query; one
count-free rollout per graph for isolated models. Frozen weights, FP32/SDPA.
"""
import argparse
from collections import Counter
import csv
import gc
import gzip
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from scripts.dataset.benchmark import count_variant
from scripts.dataset.pointer import PointerExample, SYMBOLS, validate_example, orbit_structure
from scripts.eval.checkpoint_comparison import assert_executor_tensors, executor_spec
from scripts.eval.controller_features import replay
from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes, controller_panel
from scripts.eval.pointer_task import sha256_file, synchronize
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.partitioned import partitioned_forward
from scripts.training.controller_cache import prompt_features
from scripts.training.data import collate, encode_tasks
from scripts.training.objective import forward_symbols


def load_graphs(path: Path) -> tuple[list, list[dict]]:
    tasks, metadata = [], []
    for i, line in enumerate(path.open()):
        record = json.loads(line)
        task = PointerExample(**{k: record[k] for k in PointerExample.__dataclass_fields__})
        validate_example(task)
        # IDs contain seed and mode; compare metadata independently with manifest.
        parts = task.example_id.split('-')
        mode = parts[2]
        tasks.append(task)
        metadata.append(dict(graph_index=i, example_id=task.example_id,
            mapping_sha256=task.mapping_sha256, dataset_seed=int(parts[1]), graph_mode=mode,
            **orbit_structure(dict(task.mapping), task.initial_state)))
    if len({t.mapping_sha256 for t in tasks}) != len(tasks):
        raise ValueError('Duplicate graph tables')
    return tasks, metadata


def partition_inputs(tasks: list, tokenizer, counts: list[int], device: str):
    """Tokenize actual prompts and find a shared causal token prefix per graph."""
    families = [[tokenizer.encode(count_variant(t, n).prompt, add_special_tokens=False)
                 for n in counts] for t in tasks]
    boundaries = []
    for family in families:
        boundary = 0
        for tokens in zip(*family):
            if len(set(tokens)) > 1:
                break
            boundary += 1
        if boundary == min(map(len, family)):
            # Single-count smoke tests still leave the Answer position writable.
            boundary -= 1
        boundaries.append(boundary)
    # All graphs share the shortest boundary. Prefixes may differ across graphs.
    p = min(boundaries)
    if p < 1:
        raise ValueError('No reusable causal prefix')
    prefix = torch.tensor([f[0][:p] for f in families], device=device)
    suffixes = [tokens[p:] for family in families for tokens in family]
    width = max(map(len, suffixes))
    suffix = torch.full((len(suffixes), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(suffix)
    for i, tokens in enumerate(suffixes):
        suffix[i, :len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, :len(tokens)] = 1
    indices = torch.arange(len(tasks), device=device).repeat_interleave(len(counts))
    return prefix, suffix, mask, indices


def fidelity(model, tokenizer, spec, tasks: list, metadata: list[dict], counts: list[int],
             cap: int, device: str, *, historical: bool) -> dict:
    """Independent original batch-one model calls; include count and stratum extremes."""
    selected, seen = [], set()
    for task, meta in zip(tasks, metadata, strict=True):
        stratum = meta['dataset_seed'], meta['graph_mode']
        if stratum not in seen:
            seen.add(stratum); selected.append(task)
    rows = []
    for task in selected:
        if historical:
            inputs = partition_inputs([task], tokenizer, counts, device)
            fast = partitioned_forward(model, *inputs, spec['token_ids'], cap)
        for j, n in enumerate(counts):
            variant = count_variant(task, n)
            item = encode_tasks([variant], tokenizer, dict(zip(SYMBOLS, spec['token_ids'], strict=True)), 512)
            batch = collate(item, tokenizer.pad_token_id, device, cap)
            with torch.inference_mode():
                native, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'],
                    spec['token_ids'], num_loops=n if model.completion_head is None else cap,
                    **({'completion_threshold': .5} if model.completion_head is not None else {}))
            if historical:
                used = scores.shape[1]
                expected = fast.scores[j, :used]
                # Numerical fidelity and exact observed decisions are separate checks.
                torch.testing.assert_close(scores[0], expected, atol=2e-3, rtol=2e-3)
                matches = torch.equal(scores[0].argmax(-1), expected.argmax(-1))
                if native.stop_logits is not None:
                    crossing = (fast.stops[j] >= 0).nonzero().flatten()
                    actual = int(crossing[0]) + 1 if len(crossing) else None
                    stopped = bool((native.stop_logits[0] >= 0).any())
                    matches = matches and actual == (used if stopped else None)
                if not matches:
                    raise ValueError(f'Partition/native decoded fidelity failed: {task.example_id}, N={n}')
                error = float((scores[0] - expected).abs().max())
            else:
                # Isolated replay is checked separately against these native observations.
                error = None
            rows.append(dict(example_id=variant.example_id, depth=n, loops=scores.shape[1],
                             predictions=''.join(SYMBOLS[x] for x in scores[0].argmax(-1).cpu().tolist()),
                             first_stop=(scores.shape[1] if native.stop_logits is not None and
                                         (native.stop_logits[0] >= 0).any() else None), max_logit_error=error))
    return {'passed': True, 'questions': len(rows), 'observations': rows,
            'scope': 'Original batch-one stopped calls; forced-only calls run to each requested depth'}


def save_chunk(path: Path, **arrays) -> None:
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def extract_arm(arm: dict, config: dict, output: Path, tasks: list, metadata: list[dict],
                *, device: str, graph_batch: int, limit: int | None = None) -> None:
    """Atomic per-graph chunks allow interruption/resumption without retraining."""
    name, path = arm['name'], Path(arm['path'])
    directory = output / name; directory.mkdir(parents=True, exist_ok=True)
    source_files = [*Path('scripts/recurrent_qwen').glob('*.py'), Path(__file__),
                    Path('scripts/eval/controller_features.py'), Path('scripts/training/controller_cache.py')]
    identity = dict(protocol=config, checkpoint_sha256=checkpoint_hashes(path), graph_sha256=sha256_file(Path(config['graphs'])),
                    code_sha256={str(p.relative_to(Path.cwd())) if p.is_absolute() else str(p): sha256_file(p) for p in source_files},
                    device=device, dtype='float32', attention='sdpa', graph_batch=graph_batch)
    frozen = directory / 'freeze.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('Resume identity differs: weights, protocol, implementation or batch shape changed')
    frozen.write_text(json.dumps(identity, indent=2) + '\n')
    if (directory / 'summary.json').exists():
        saved = json.loads((directory / 'summary.json').read_text())
        if saved['status'] == 'complete':
            Console().print(f'[green]{name}: complete, retained verified chunks[/green]'); return
    model, tokenizer, spec = load_recurrent_checkpoint(path, device=device)
    model.eval().requires_grad_(False); model.config._attn_implementation = 'sdpa'
    counts, cap = config['requests'], config['safety_cap']
    historical = arm['mode'] in ('forced_only', 'legacy')
    source = Path(config['source_results'])
    source_rows = list(csv.DictReader((source / 'graphs.csv').open()))
    if len(source_rows) != len(tasks) or any(r['mapping_sha256'] != t.mapping_sha256 or r['example_id'] != t.example_id
            for r,t in zip(source_rows,tasks,strict=True)):
        raise ValueError('Verified source graph identities differ')
    if historical != (model.router is None):
        raise ValueError('Registry and checkpoint routing disagree')
    if not historical:
        source_summary = json.loads((source / 'summary.json').read_text())
        if checkpoint_hashes(Path(config['models'][-1]['path'])) != source_summary['checkpoint_sha256']:
            raise ValueError('Verified final source weights changed')
        source_spec = json.loads((Path(config['models'][-1]['path']) / 'recurrent_config.json').read_text())
        if executor_spec(spec) != executor_spec(source_spec):
            raise ValueError('Source and isolated executor architecture differ')
        compared = assert_executor_tensors(torch.load(Path(config['models'][-1]['path']) / 'adapter_model.pt', weights_only=True, map_location='cpu'),
                                          torch.load(path / 'adapter_model.pt', weights_only=True, map_location='cpu'))
        (directory / 'reuse_audit.json').write_text(json.dumps(dict(passed=True, identical_executor_tensors=compared,
                source_graphs_sha256=sha256_file(source / 'graphs.csv'), source_summary_sha256=sha256_file(source / 'summary.json')), indent=2))
    # Native checks are deliberately bounded and predeclared; no weights are fit.
    audit_path = directory / 'native_fidelity.json'
    if not audit_path.exists():
        Console().print(f'[cyan]{name}: checking original batch-one inference[/cyan]')
        audit = fidelity(model, tokenizer, spec, tasks, metadata, [1, 6, 12, 32, 128, 256], cap, device, historical=historical)
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    audit = json.loads(audit_path.read_text())
    if not audit['passed']:
        raise ValueError('Native fidelity did not pass')
    native = {r['example_id']: r for r in audit['observations']}
    if not historical and model.completion_head.kind == 'shared_number':
        _, scalar_stops, controller_rows = controller_panel(model, tokenizer, spec['token_ids'], tasks, metadata, max(counts), cap)
        (directory / 'controller_counts.json').write_text(json.dumps(controller_rows, indent=2) + '\n')
    total = min(len(tasks), limit) if limit else len(tasks)
    began = perf_counter(); processed = 0
    with Progress(TextColumn(name), BarColumn(), TextColumn('{task.completed:.0f}/{task.total:.0f} graphs'),
                  TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        bar = progress.add_task(name, total=total)
        for start in range(0, total, graph_batch):
            end = min(start + graph_batch, total)
            chunk = directory / f'graphs-{start:04d}-{end:04d}.npz'
            if not chunk.exists():
                subset = tasks[start:end]
                synchronize(torch.device(device)); tick = perf_counter()
                if historical:
                    result = partitioned_forward(model, *partition_inputs(subset, tokenizer, counts, device), spec['token_ids'], cap, horizons=torch.tensor(counts,device=device).repeat(end-start))
                    predictions = result.scores.argmax(-1).cpu().numpy().astype('uint8').reshape(end-start, len(counts), cap)
                    if result.stops is None:
                        stops = np.full((end-start, len(counts)), -2, dtype='int16')  # undefined
                    else:
                        logits = result.stops.cpu().numpy().reshape(end-start, len(counts), cap)
                        stops = np.where((logits >= 0).any(-1), (logits >= 0).argmax(-1) + 1, -1).astype('int16')
                    direct = np.empty((0,), dtype='uint8')
                else:
                    if model.completion_head.kind == 'shared_number':
                        predictions = np.array([[ord(c)-65 for c in r['predictions'][:cap]] for r in source_rows[start:end]], dtype='uint8')
                        direct = np.array([[ord(c)-65 for c in r['r_predictions'][:cap]] for r in source_rows[start:end]], dtype='uint8')
                        stops = np.broadcast_to(np.array([x if x is not None else -1 for x in scalar_stops], dtype='int16'), (end-start, len(counts))).copy()
                    else:
                        # Capture actual R states once; GRU initialization still sees each full question.
                        captures = []
                        hook = model.completion_head.observation.register_forward_pre_hook(lambda m, args: captures.append(args[0].detach()))
                        try:
                            items = encode_tasks(subset, tokenizer, dict(zip(SYMBOLS, spec['token_ids'], strict=True)), 512)
                            batch = collate(items, tokenizer.pad_token_id, device, cap)
                            with torch.inference_mode():
                                result, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'], spec['token_ids'], num_loops=cap)
                            working = torch.stack(captures, 1)
                        finally:
                            hook.remove()
                        predictions = scores.argmax(-1).cpu().numpy().astype('uint8')
                        expected = np.array([[ord(c)-65 for c in r['predictions'][:cap]] for r in source_rows[start:end]], dtype='uint8')
                        if not np.array_equal(predictions, expected):
                            raise ValueError('Captured isolated executor differs from verified source')
                        direct = result.state_logits.argmax(-1).cpu().numpy().astype('uint8')
                        variants = [count_variant(t, n) for t in subset for n in counts]
                        context = prompt_features(model, tokenizer, spec['token_ids'], variants, 64).to(device)
                        firsts=[]
                        for offset in range(0,len(variants),1024):
                            stop=min(offset+1024,len(variants))
                            lookup=torch.arange(offset,stop,device=device)//len(counts)
                            with torch.inference_mode():
                                logits,_,_=replay(model.completion_head,context[offset:stop],working[lookup])
                            hits=(logits>=0).cpu().numpy()
                            firsts.append(np.where(hits.any(-1),hits.argmax(-1)+1,-1).astype('int16'))
                        stops=np.concatenate(firsts).reshape(end-start,len(counts))
                for local, task in enumerate(subset):
                    for j, n in enumerate(counts):
                        check = native.get(count_variant(task, n).example_id)
                        if check is not None:
                            pred = predictions[local, j] if historical else predictions[local]
                            letters = ''.join(SYMBOLS[x] for x in pred[:check['loops']])
                            if letters != check['predictions'] or (check['first_stop'] or -1) != int(stops[local,j]) and stops[local,j] != -2:
                                raise ValueError('Population chunk disagrees with original native fidelity calls')
                synchronize(torch.device(device)); seconds = perf_counter() - tick
                save_chunk(chunk, predictions=predictions, stops=stops, direct=direct,
                           graph_indices=np.arange(start,end), counts=np.array(counts), seconds=np.array(seconds))
                processed += end-start
            progress.update(bar, completed=end)
    complete = total == len(tasks)
    result = dict(status='complete' if complete else 'partial', graphs=total, counts=counts,
        questions=total*len(counts), native_fidelity=True, weights_unchanged=checkpoint_hashes(path)==identity['checkpoint_sha256'],
        new_graphs=processed, elapsed_seconds=perf_counter()-began, independent_count_specific_execution=historical)
    if not result['weights_unchanged']:
        raise ValueError('Frozen checkpoint changed during evaluation')
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    del model; gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pointer_analysis_protocol.json'))
    parser.add_argument('--output', type=Path, default=Path('eval/pointer_analysis/paper-20261009'))
    parser.add_argument('--model', action='append', help='Registry ID; default all six')
    parser.add_argument('--graph-batch', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Smoke only, never marked complete')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if config['requests'] != list(range(1,257)) or config['threshold'] != .5 or config['safety_cap'] != 272 or args.graph_batch < 1:
        raise ValueError('Paper protocol must retain all counts, fixed threshold/cap and positive batch')
    tasks, metadata = load_graphs(Path(config['graphs']))
    strata = Counter((r['dataset_seed'],r['graph_mode']) for r in metadata)
    if len(tasks)!=config['graph_count'] or len(strata)!=9 or set(strata.values())!={150}:
        raise ValueError('Full 1350 graph, nine-stratum panel required')
    source = json.loads((Path(config['source_results'])/'dataset_manifest.json').read_text())
    if source['graphs_sha256'] != sha256_file(Path(config['graphs'])):
        raise ValueError('Opened population graph bytes differ')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'graph_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.set_num_threads(4); torch.manual_seed(239); torch.use_deterministic_algorithms(True)
    names = args.model or [a['name'] for a in config['models']]
    if set(names) - {a['name'] for a in config['models']}:
        raise ValueError('Unknown registry ID')
    for arm in config['models']:
        if arm['name'] in names:
            extract_arm(arm, config, args.output, tasks, metadata, device=args.device,
                        graph_batch=args.graph_batch if arm['mode']!='isolated' else 64,limit=args.limit)


if __name__=='__main__': main()

"""Resumable ordinary-Qwen generation on the full, unchanged paper graph panel.

Uses the existing ordinary evaluator. One atomic compressed file per graph holds
every requested count, raw continuations and reconstructible input hashes. No
recurrent transition, stop-loop or trajectory semantics are assigned to tokens.
"""
import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from scripts.dataset.benchmark import count_variant
from scripts.eval.loading import load_model
from scripts.eval.paper_suite import load_graphs
from scripts.eval.pointer_task import evaluate_batches, sha256_file

DEFAULT_CONFIG = Path('configs/pointer_analysis_baseline.json')


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_chunk(path: Path, graph_index: int, counts: list[int]) -> tuple[dict, list[dict]]:
    """Require all and only the frozen requests in graph-major/count order."""
    with gzip.open(path, 'rt') as handle:
        header = json.loads(next(handle))
        rows = [json.loads(line) for line in handle]
    if (header['graph_index'] != graph_index or [r['task_depth'] for r in rows] != counts or
            any(r['graph_index'] != graph_index for r in rows)):
        raise ValueError(f'Baseline graph/query coverage differs: {path}')
    return header, rows


def write_chunk(path: Path, graph_index: int, rows: list[dict], seconds: float,
                wall_seconds: float) -> None:
    temporary = path.with_suffix('.tmp.gz')
    with gzip.open(temporary, 'wt') as handle:
        handle.write(json.dumps(dict(graph_index=graph_index, generation_seconds=seconds,
                                     evaluation_seconds=wall_seconds)) + '\n')
        for row in rows:
            # Exact rendered input is reproducible from graph, frozen template and
            # saved tokenizer. Avoid repeating ~1 KB of instructions 345,600 times.
            record = {k: v for k, v in row.items() if k not in ('prompt', 'model_input')}
            record.update(graph_index=graph_index, input_sha256=text_hash(row['model_input']))
            handle.write(json.dumps(record) + '\n')
    temporary.replace(path)


def native_gate(model, tokenizer, tasks: list, metadata: list[dict], template: str,
                config: dict) -> dict:
    """Compare 54 predeclared batched continuations to independent batch-one calls."""
    samples, seen = [], set()
    for task, meta in zip(tasks, metadata, strict=True):
        key = meta['dataset_seed'], meta['graph_mode']
        if key not in seen:
            seen.add(key)
            samples.extend(count_variant(task, n) for n in config['native_counts'])
    options = dict(max_new_tokens=config['max_new_tokens'], prompt_format=config['prompt_format'])
    batched = [row for rows, _ in evaluate_batches(model, tokenizer, samples, template,
               batch_size=config['batch_size'], **options) for row in rows]
    native = [row for rows, _ in evaluate_batches(model, tokenizer, samples, template,
              batch_size=1, **options) for row in rows]
    keys = ('example_id', 'response', 'generated_token_ids', 'stop_reason', 'prediction', 'correct')
    if any(any(a[k] != b[k] for k in keys) for a, b in zip(batched, native, strict=True)):
        raise ValueError('Ordinary baseline batched generation differs from native batch-one observations')
    return dict(passed=True, questions=len(samples), counts=config['native_counts'],
                scope='Exact continuations/decisions on first graph of each seed/mode stratum',
                observations=[{k: row[k] for k in keys} for row in native])


def extract(root: Path, config_path: Path = DEFAULT_CONFIG) -> None:
    """Load cached pinned ordinary weights once, resume committed graph chunks."""
    protocol = json.loads((root / 'protocol.json').read_text())
    config = json.loads(config_path.read_text())
    if (config['dtype'] != 'float32' or config['attention'] != 'sdpa' or config['use_cache'] or
            config['do_sample'] or config['prompt_format'] != 'chat' or config['batch_size'] < 1):
        raise ValueError('Baseline must retain explicit FP32/SDPA greedy cache-disabled chat generation')
    graphs = Path(protocol['graphs'])
    prompt = Path(config['prompt'])
    if sha256_file(graphs) != config['graphs_sha256'] or sha256_file(prompt) != config['prompt_sha256']:
        raise ValueError('Frozen baseline graph/prompt bytes differ')
    tasks, metadata = load_graphs(graphs)
    counts = protocol['requests']
    if len(tasks) != protocol['graph_count'] or counts != list(range(1, 257)):
        raise ValueError('Baseline requires the entire frozen graph/count panel')
    snapshot = Path(snapshot_download(config['model'], revision=config['revision'], local_files_only=True))
    sources = [Path(__file__), Path('scripts/eval/pointer_task.py'), Path('scripts/eval/loading.py'),
               Path('scripts/eval/paper_suite.py'), Path('scripts/dataset/benchmark.py'), Path('scripts/dataset/pointer.py')]
    identity = dict(config=config, protocol_sha256=sha256_file(root / 'protocol.json'), device='cuda',
        sources_sha256={str(p): sha256_file(p) for p in sources},
        snapshot_sha256={p.name: sha256_file(p) for p in sorted(snapshot.iterdir())
                         if p.is_file() and p.suffix in ('.safetensors', '.json', '.txt', '.model')},
        packages={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', 'huggingface-hub')},
        cuda_version=torch.version.cuda, matmul_allow_tf32=False, cudnn_allow_tf32=False)
    directory = root / 'ordinary_qwen'; directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / 'freeze.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('Ordinary baseline resume identity changed')
    atomic_json(frozen, identity)
    (directory / 'prompt.txt').write_text(prompt.read_text())
    completed, correct, tokens, generation_seconds, evaluation_seconds, committed = 0, 0, 0, 0., 0., {}
    for i in range(len(tasks)):
        path = directory / f'graph-{i:04d}.jsonl.gz'
        if path.exists():
            header, rows = read_chunk(path, i, counts)
            completed += len(rows); correct += sum(r['correct'] for r in rows)
            tokens += sum(r['generated_tokens'] for r in rows)
            generation_seconds += header['generation_seconds']; evaluation_seconds += header['evaluation_seconds']
            committed[path.name] = sha256_file(path)
    def progress_value(status: str) -> dict:
        return dict(status=status, queries=completed, total=len(tasks)*len(counts), graphs=len(committed),
            correct=correct, accuracy=correct/completed if completed else None, generated_tokens=tokens,
            generation_seconds=generation_seconds, evaluation_seconds=evaluation_seconds,
            generated_tokens_per_second=tokens/generation_seconds if generation_seconds else None,
            questions_per_second=completed/evaluation_seconds if evaluation_seconds else None,
            chunks_sha256=committed, freeze_sha256=sha256_file(frozen), weights_unchanged=True,
            timing_scope='Generation: synchronized model.generate including prefill, tokens include first EOS. Evaluation: graph variant construction, tokenization, generation and decoding; excludes loading, native gates, hashing and persistence.')
    saved = directory / 'progress.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        if any(committed.get(name) != digest for name, digest in previous['chunks_sha256'].items()):
            raise ValueError('Baseline committed raw chunk changed or disappeared')
    atomic_json(saved, progress_value('extracting'))
    if len(committed) == len(tasks):
        if not json.loads((directory / 'native_fidelity.json').read_text())['passed']:
            raise ValueError('Baseline native gate absent/failed')
        atomic_json(saved, progress_value('complete')); atomic_json(directory/'summary.json', progress_value('complete'))
        return
    torch.set_num_threads(4); torch.manual_seed(config['seed']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Console().print(f'[cyan]Ordinary Qwen · {len(tasks)*len(counts):,} questions · 3-shot · FP32/SDPA · batch {config["batch_size"]}[/cyan]')
    model, tokenizer, loaded = load_model(config['model'], base_model=None, tokenizer_source=None,
        revision=config['revision'], device='cuda', dtype=config['dtype'], download=False)
    if loaded['resolved_model_revision'] != config['revision']:
        raise ValueError('Loaded ordinary model revision differs')
    model.set_attn_implementation(config['attention']); model.requires_grad_(False)
    tokenizer.save_pretrained(directory / 'tokenizer')
    atomic_json(directory / 'loaded.json', dict(**loaded, tokenizer_sha256=text_hash(tokenizer.backend_tokenizer.to_str()),
        chat_template=tokenizer.chat_template, eos_token_id=model.generation_config.eos_token_id))
    template = prompt.read_text()
    gate = directory / 'native_fidelity.json'
    if not gate.exists():
        atomic_json(gate, native_gate(model, tokenizer, tasks, metadata, template, config))
    if not json.loads(gate.read_text())['passed']:
        raise ValueError('Ordinary baseline native gate failed')
    with Progress(TextColumn('{task.description}'), BarColumn(), TextColumn('{task.completed:,}/{task.total:,}'),
                  TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        task_id = progress.add_task('Ordinary Qwen generation', total=len(tasks)*len(counts), completed=completed)
        for i, task in enumerate(tasks):
            path = directory / f'graph-{i:04d}.jsonl.gz'
            if path.name in committed:
                continue
            began = perf_counter(); variants = [count_variant(task, n) for n in counts]
            rows, seconds = [], 0.
            for batch, elapsed in evaluate_batches(model, tokenizer, variants, template,
                    batch_size=config['batch_size'], max_new_tokens=config['max_new_tokens'], prompt_format=config['prompt_format']):
                rows.extend(batch); seconds += elapsed
                progress.advance(task_id, len(batch))
            wall_seconds = perf_counter()-began
            write_chunk(path, i, rows, seconds, wall_seconds)
            committed[path.name] = sha256_file(path); completed += len(rows)
            correct += sum(r['correct'] for r in rows); tokens += sum(r['generated_tokens'] for r in rows)
            generation_seconds += seconds; evaluation_seconds += wall_seconds
            atomic_json(saved, progress_value('extracting'))
    current = {p.name: sha256_file(p) for p in snapshot.iterdir() if p.name in identity['snapshot_sha256']}
    if current != identity['snapshot_sha256']:
        raise ValueError('Cached ordinary model/tokenizer bytes changed during extraction')
    atomic_json(saved, progress_value('complete')); atomic_json(directory/'summary.json', progress_value('complete'))
    del model; gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('eval/pointer_analysis/paper-20261009'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    extract(args.input, args.config)


if __name__ == '__main__':
    main()

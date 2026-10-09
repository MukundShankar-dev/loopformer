"""Independent raw-generation audit and paired final-letter baseline statistics."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from scripts.eval.benchmark_metrics import wilson_interval
from scripts.eval.loop_metrics import write_csv
from scripts.eval.paper_baseline import atomic_json, read_chunk, text_hash
from scripts.eval.paper_suite import load_graphs
from scripts.eval.pointer_task import sha256_file

RANGES = (('all', 1, 256), ('1-6', 1, 6), ('7-12', 7, 12), ('13-32', 13, 32),
          ('33-64', 33, 64), ('65-128', 65, 128), ('129-256', 129, 256))


def audit_row(row: dict, task, count: int, target: str, tokenizer, template: str,
              config: dict, eos_ids: set[int]) -> tuple[bool, bool, bool, int]:
    """Reconstruct inputs/labels and decode raw token IDs independently of scoring."""
    if row['example_id'] != f'{task.example_id}-steps-{count}' or row['target'] != target:
        raise ValueError('Ordinary query identity/target differs from raw rules')
    prompt = template.format(rules=' '.join(f'( {a}, {b})' for a, b in task.mapping),
                             start=task.initial_state, steps=count).strip()
    model_input = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                                               tokenize=False, add_generation_prompt=True)
    if text_hash(model_input) != row['input_sha256']:
        raise ValueError('Ordinary rendered input hash differs')
    ids = json.loads(row['generated_token_ids'])
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('Invalid raw continuation IDs')
    eos = ids[-1] in eos_ids
    if (len(ids) > config['max_new_tokens'] or len(ids) != row['generated_tokens'] or
            any(i in eos_ids for i in ids[:-1]) or
            row['stop_reason'] != ('eos' if eos else 'max_new_tokens') or
            (not eos and len(ids) != config['max_new_tokens'])):
        raise ValueError('Ordinary continuation budget/EOS semantics differ')
    response = tokenizer.decode(ids, skip_special_tokens=True)
    prediction = response.strip()
    valid = len(prediction) == 1 and 'A' <= prediction <= 'Z'
    correct = valid and prediction == target
    if (row['response'] != response or row['valid_answer'] is not valid or row['correct'] is not correct or
            row['prediction'] != (prediction if valid else '')):
        raise ValueError('Ordinary score differs from independently decoded continuation')
    return correct, valid, not eos, ord(prediction)-65 if valid else 26


def analyze(root: Path) -> None:
    """Audit all raw responses, then cluster-bootstrap graph-level final accuracy."""
    directory = root / 'ordinary_qwen'
    protocol = json.loads((root / 'protocol.json').read_text())
    frozen = json.loads((directory / 'freeze.json').read_text()); config = frozen['config']
    summary = json.loads((directory / 'summary.json').read_text())
    native = json.loads((directory / 'native_fidelity.json').read_text())
    if (summary['status'] != 'complete' or not summary['weights_unchanged'] or not native['passed'] or
            summary['freeze_sha256'] != sha256_file(directory/'freeze.json')):
        raise ValueError('Ordinary baseline incomplete/unverified')
    if frozen['protocol_sha256'] != sha256_file(root/'protocol.json') or config['graphs_sha256'] != sha256_file(Path(protocol['graphs'])):
        raise ValueError('Ordinary baseline panel identity differs')
    template = (directory / 'prompt.txt').read_text()
    if text_hash(template) != config['prompt_sha256']:
        raise ValueError('Ordinary saved prompt differs')
    tokenizer = AutoTokenizer.from_pretrained(directory/'tokenizer', local_files_only=True)
    loaded = json.loads((directory / 'loaded.json').read_text())
    if (text_hash(tokenizer.backend_tokenizer.to_str()) != loaded['tokenizer_sha256'] or
            tokenizer.chat_template != loaded['chat_template'] or loaded['resolved_model_revision'] != config['revision']):
        raise ValueError('Ordinary saved tokenizer/revision differs')
    eos = loaded['eos_token_id']
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    tasks, metadata = load_graphs(Path(protocol['graphs']))
    counts = protocol['requests']; shape = len(tasks), len(counts)
    if summary['total'] != shape[0]*shape[1] or summary['queries'] != summary['total']:
        raise ValueError('Ordinary baseline summary coverage differs')
    correct, valid, budget = (np.zeros(shape, dtype=bool) for _ in range(3))
    predictions = np.full(shape, 26, dtype='uint8')
    hashes = {}; generated_tokens = 0
    expected_files = {f'graph-{i:04d}.jsonl.gz' for i in range(len(tasks))}
    if {p.name for p in directory.glob('graph-*.jsonl.gz')} != expected_files or set(summary['chunks_sha256']) != expected_files:
        raise ValueError('Ordinary graph chunk set differs')
    for i, task in enumerate(tasks):
        path = directory / f'graph-{i:04d}.jsonl.gz'; hashes[path.name] = sha256_file(path)
        if hashes[path.name] != summary['chunks_sha256'][path.name]:
            raise ValueError('Ordinary raw chunk hash differs')
        _, rows = read_chunk(path, i, counts)
        # Scalar dictionary execution does not reuse stored dataset labels or the
        # generation evaluator's score_response/reference-execution functions.
        state = task.initial_state; mapping = dict(task.mapping)
        for j, (count, row) in enumerate(zip(counts, rows, strict=True)):
            if count != j+1:
                raise ValueError('Independent audit expects consecutive requested counts')
            state = mapping[state]
            correct[i,j], valid[i,j], budget[i,j], predictions[i,j] = audit_row(
                row, task, count, state, tokenizer, template, config, eos_ids)
            generated_tokens += row['generated_tokens']
    if (int(correct.sum()) != summary['correct'] or generated_tokens != summary['generated_tokens'] or
            abs(float(correct.mean()) - summary['accuracy']) > 1e-12):
        raise ValueError('Ordinary summary disagrees with raw generation audit')
    np.savez_compressed(directory/'outcomes.npz', correct=correct, valid=valid, token_budget=budget,
                        predictions=predictions, counts=np.asarray(counts))
    rows = []
    for j, count in enumerate(counts):
        low, high = wilson_interval(int(correct[:,j].sum()), len(tasks))
        rows.append(dict(requested_depth=count, graphs=len(tasks), correct=int(correct[:,j].sum()),
            accuracy=float(correct[:,j].mean()), low=low, high=high,
            invalid_answers=int((~valid[:,j]).sum()), token_budget_stops=int(budget[:,j].sum())))
    write_csv(directory/'quality_by_depth.csv', rows)
    strata = []
    for field in ('graph_mode', 'dataset_seed', 'cycle_period', 'transient_length'):
        for value in sorted({m[field] for m in metadata}):
            mask = np.array([m[field] == value for m in metadata])
            for j, count in enumerate(counts):
                strata.append(dict(field=field, value=value, requested_depth=count, graphs=int(mask.sum()),
                    correct=int(correct[mask,j].sum()), accuracy=float(correct[mask,j].mean()),
                    invalid_answers=int((~valid[mask,j]).sum()), token_budget_stops=int(budget[mask,j].sum())))
    write_csv(directory/'strata.csv', strata)
    rng = np.random.default_rng(protocol['uncertainty']['bootstrap_seed']); groups = {}
    for i, meta in enumerate(metadata):
        groups.setdefault((meta['dataset_seed'], meta['graph_mode']), []).append(i)
    draws = np.concatenate([rng.choice(ids, size=(protocol['uncertainty']['bootstrap_repeats'], len(ids)), replace=True)
                            for ids in groups.values()], 1)
    intervals, deltas = [], []
    with np.load(root/'outcomes.npz', allow_pickle=False) as recurrent:
        for label, low, high in RANGES:
            mask = (np.array(counts) >= low) & (np.array(counts) <= high)
            means = correct[:,mask].mean(1)
            lo, hi = np.quantile(means[draws].mean(1), [.025, .975])
            intervals.append(dict(depth_range=label, rate=float(means.mean()), low=float(lo), high=float(hi), graphs=len(tasks)))
            for arm in protocol['models']:
                for metric in ('nominal_final_correct', 'stopped_answer_correct'):
                    key = f'{arm["name"]}__{metric}'
                    if key not in recurrent.files:
                        continue  # CE-only has no autonomous return.
                    values = recurrent[key]
                    if values.shape != shape:
                        raise ValueError('Paired recurrent/ordinary query matrix differs')
                    difference = values[:,mask].mean(1)-means
                    lo, hi = np.quantile(difference[draws].mean(1), [.025, .975])
                    deltas.append(dict(model=arm['name'], metric=metric, depth_range=label,
                        recurrent_minus_ordinary=float(difference.mean()), low=float(lo), high=float(hi), graphs=len(tasks)))
    write_csv(directory/'cluster_intervals.csv', intervals)
    write_csv(directory/'paired_deltas.csv', deltas)
    frequency = Counter(predictions.ravel().tolist())
    atomic_json(directory/'independent_audit.json', dict(passed=True, queries=summary['queries'], graphs=len(tasks),
        accuracy=float(correct.mean()), invalid_answers=int((~valid).sum()), token_budget_stops=int(budget.sum()),
        prediction_counts={chr(65+k) if k < 26 else 'invalid': v for k,v in sorted(frequency.items())},
        chunks_sha256=hashes, freeze_sha256=sha256_file(directory/'freeze.json'),
        native_sha256=sha256_file(directory/'native_fidelity.json'), loaded_sha256=sha256_file(directory/'loaded.json'),
        outcomes_sha256=sha256_file(directory/'outcomes.npz'),
        recurrent_outcomes_sha256=sha256_file(root/'outcomes.npz'),
        analysis_artifacts_sha256={name: sha256_file(directory/name) for name in
            ('quality_by_depth.csv','strata.csv','cluster_intervals.csv','paired_deltas.csv')},
        analysis_code_sha256=sha256_file(Path(__file__)),
        scope='Every raw continuation decoded; independently reconstructed prompts/targets/scores; final-letter metrics only'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('eval/pointer_analysis/paper-20261009'))
    args = parser.parse_args(); analyze(args.input)


if __name__ == '__main__':
    main()

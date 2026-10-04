"""Learned structured pointer executor: a positive control, not a Qwen result.

The input parser exposes the supplied edges as key/value symbols. Learned
attention must select an edge from the current continuous state. No reference
intermediate state, decoded symbol, equality lookup, or step count is fed back.
"""
import argparse
import json
from pathlib import Path
from functools import lru_cache
from typing import Any

import torch
from torch import nn

from scripts.dataset.pointer import SYMBOLS, parse_mapping
from scripts.recurrent_qwen.outputs import RecurrentOutput


class NativeExecutor(nn.Module):
    supports_symbol_readout = True
    completion_head = None

    def __init__(self, tokenizer: Any, width: int = 64) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.width = width
        self.symbols = nn.Embedding(26, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, 26)
        self.parse = lru_cache(maxsize=4096)(self._parse)

    def _parse(self, ids):
        text = self.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        lines = text.splitlines()
        if len(lines) != 4 or not lines[1].startswith('Start: '):
            raise ValueError('Native control requires a raw pointer prompt')
        mapping = parse_mapping(lines[0].removeprefix('Rules: '))
        return [SYMBOLS.index(mapping[s]) for s in SYMBOLS], SYMBOLS.index(lines[1].removeprefix('Start: '))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, num_loops: int,
                readout_token_ids: list[int] | None = None) -> RecurrentOutput:
        programs = [self.parse(tuple(row[mask.bool()].tolist())) for row, mask in zip(input_ids, attention_mask)]
        targets = torch.tensor([p[0] for p in programs], device=input_ids.device)
        starts = torch.tensor([p[1] for p in programs], device=input_ids.device)
        state = self.symbols(starts)
        keys = self.key(self.norm(self.symbols.weight))
        values = self.symbols(targets)
        logits = []
        for _ in range(num_loops):
            attention = (self.query(self.norm(state)) @ keys.T / self.width**.5).softmax(-1)
            state = (attention.unsqueeze(-1) * values).sum(1)
            logits.append(self.readout(self.norm(state)))
        return RecurrentOutput(loop_logits=tuple(logits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/pointer/seed-61-independent'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train-max-depth', type=int, default=12)
    parser.add_argument('--eval-max-depth', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--wandb-mode', choices=('disabled', 'online', 'offline'), default='disabled')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.width, args.train_max_depth, args.eval_max_depth) < 1:
        parser.error('Budgets and sizes must be positive')
    if args.output.exists():
        parser.error('Output already exists')
    from transformers import AutoTokenizer
    from rich.progress import Progress
    from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION, validate_symbols
    from scripts.training.data import read_tasks, encode_tasks, collate
    from scripts.training.objective import step_loss, forward_symbols
    from scripts.training.evaluation import evaluate
    from scripts.training.tracking import tracking_run
    from scripts.eval.pointer_task import sha256_file

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    token_map = validate_symbols(tokenizer)
    train_tasks = [t for t in read_tasks(args.data / 'train.jsonl', 'train') if t.task_depth <= args.train_max_depth]
    dev_tasks = [t for t in read_tasks(args.data / 'validation.jsonl', 'validation') if t.task_depth <= args.train_max_depth]
    deep_tasks = [t for t in read_tasks(args.data / 'depth_test.jsonl', 'depth_test') if t.task_depth <= args.eval_max_depth]
    if not train_tasks or not dev_tasks or not deep_tasks:
        parser.error('All development cohorts must be nonempty')
    graph_sets = [{t.mapping_sha256 for t in tasks} for tasks in (train_tasks, dev_tasks, deep_tasks)]
    if any(graph_sets[i] & graph_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Control splits share graphs')
    train_items, dev_items, deep_items = [encode_tasks(tasks, tokenizer, token_map, 512)
                                        for tasks in (train_tasks, dev_tasks, deep_tasks)]
    model = NativeExecutor(tokenizer, args.width).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    generator = torch.Generator().manual_seed(args.seed)
    args.output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {'kind': 'learned_structured_executor_control', 'config': config,
        'data_sha256': {split: sha256_file(args.data / f'{split}.jsonl') for split in ('train', 'validation', 'depth_test')},
        'source_sha256': sha256_file(Path(__file__)), 'optimizer': 'AdamW', 'sampling': 'seeded with replacement',
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'claim': 'structured-memory learnability control; not Qwen or language generalization'}
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    token_ids = list(token_map.values())
    with tracking_run(args.output, metadata, config, mode=args.wandb_mode, project='loopformer') as tracker, \
            Progress() as progress, (args.output / 'metrics.jsonl').open('w') as log:
        task = progress.add_task('Learned native control', total=args.steps)
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randint(len(train_items), (args.batch_size,), generator=generator).tolist()
            batch = collate([train_items[i] for i in indices], tokenizer.pad_token_id, args.device)
            optimizer.zero_grad(set_to_none=True)
            _, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'], token_ids,
                                         num_loops=batch['targets'].shape[1])
            loss, _ = step_loss(scores, batch['targets'], batch['target_mask'])
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            event = {'event': 'train', 'step': step, 'train': {'loss': loss.item()}, 'gradient_norm_before_clip': norm.item()}
            log.write(json.dumps(event) + '\n')
            tracker.log(event)
            progress.update(task, advance=1, description=f'Native control · CE {loss.item():.3f}')
        metrics = {name: evaluate(model, items, token_ids, tokenizer.pad_token_id,
                    batch_size=args.batch_size, output=args.output / f'{name}.csv')
                   for name, items in (('validation', dev_items), ('depth_test', deep_items))}
        tracker.summary(metrics)
    torch.save(model.state_dict(), args.output / 'native_model.pt')
    (args.output / 'summary.json').write_text(json.dumps({'status': 'complete', **metadata, **metrics}, indent=2) + '\n')
    print(f'Saved native control and development metrics: {args.output}')


if __name__ == '__main__':
    main()

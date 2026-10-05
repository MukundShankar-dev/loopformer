"""Read-only capture of the actual controller interface on matched development graphs."""
from copy import deepcopy
from dataclasses import replace
import random
from collections.abc import Callable

import torch
from torch import Tensor

from scripts.dataset.pointer import PointerExample, SYMBOLS, execute, render_prompt
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.data import collate, encode_tasks
from scripts.training.objective import forward_symbols


def make_panel(tasks: list[PointerExample], graph_counts: tuple[int, int, int],
               depths: list[int], seed: int) -> tuple[list[PointerExample], list[dict]]:
    """Split by rule-table identity before making count variants; no confirmation data."""
    if len(graph_counts) != 3 or min(graph_counts) < 1 or not depths or sorted(set(depths)) != depths:
        raise ValueError('Need three positive graph counts and sorted unique depths')
    if min(depths) < 1 or max(depths) > 256:
        raise ValueError('Depths must be within 1..256')
    unique = {}
    for task in tasks:
        if task.split != 'validation':
            raise ValueError('Controller diagnostics use development validation graphs only')
        unique.setdefault(task.mapping_sha256, task)
    graphs = list(unique.values())
    random.Random(seed).shuffle(graphs)
    if len(graphs) < sum(graph_counts):
        raise ValueError(f'Need {sum(graph_counts)} distinct development graphs, found {len(graphs)}')
    variants, metadata = [], []
    offset = 0
    for partition, count in zip(('fit', 'dev', 'holdout'), graph_counts, strict=True):
        for graph_index, task in enumerate(graphs[offset:offset + count], offset):
            for depth in depths:
                states = execute(dict(task.mapping), task.initial_state, depth)
                variant = replace(task, schema_version=2, example_id=f'{task.example_id}-controller-{depth}',
                    task_depth=depth, intermediate_states=states, final_state=states[-1],
                    prompt=render_prompt(task.mapping, task.initial_state, depth))
                variants.append(variant)
                metadata.append({'example_id': variant.example_id, 'graph_index': graph_index,
                    'mapping_sha256': task.mapping_sha256, 'partition': partition, 'depth': depth})
        offset += count
    return variants, metadata


def replay(head: RecurrentController, context: Tensor, working: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Replay cached [B,H] prompt and [B,T,H] executor observations, without labels.

    The hidden controller memory stays differentiable through every update. There
    is no time/count feature, oracle input, or external state reset between loops.
    """
    initial = head.initialize(context)
    memory = initial
    logits, memories = [], []
    for state in working.unbind(1):
        logit, memory = head.advance(state, memory)
        logits.append(logit)
        memories.append(memory)
    return torch.stack(logits, 1), initial, torch.stack(memories, 1)


def extract_features(model, tokenizer, token_ids: list[int], tasks: list[PointerExample],
                     loops: int, batch_size: int,
                     progress: Callable[[int], None] | None = None) -> tuple[dict[str, Tensor], RecurrentController, float]:
    """Capture only vectors via hooks, not full sequence states; FP32 eval only.

    All tasks run every loop, even past their requested stop, to balance probe
    labels. Cached replay must reproduce the live head's complete logit sequence.
    Hooks are always removed; callers retain an unchanged model state dict.
    """
    if not isinstance(model.completion_head, RecurrentController) or model.router is None:
        raise ValueError('Need an isolated executor checkpoint with recurrent controller')
    if not tasks or batch_size < 1 or loops < max(t.task_depth for t in tasks):
        raise ValueError('Need nonempty tasks, positive batch size and a budget covering all counts')
    model.eval().requires_grad_(False)
    head = model.completion_head
    device = next(model.parameters()).device
    items = encode_tasks(tasks, tokenizer, dict(zip(SYMBOLS, token_ids, strict=True)), 512)
    captured: dict[str, list[Tensor]] = {}
    def save(name: str, value: Tensor) -> None:
        captured.setdefault(name, []).append(value.detach().clone())
    hooks = [
        head.context.register_forward_pre_hook(lambda module, inputs: save('context', inputs[0])),
        head.context.register_forward_hook(lambda module, inputs, output: save('initial', output)),
        head.observation.register_forward_pre_hook(lambda module, inputs: save('working', inputs[0])),
        head.cell.register_forward_hook(lambda module, inputs, output: save('memory', output)),
    ]
    # This copy has no capture hooks. Copies of nn.Module also copy hook registries.
    replay_head = RecurrentController(model.config.hidden_size, head.intermediate).to(device)
    replay_head.load_state_dict(head.state_dict())
    replay_head.eval().requires_grad_(False)
    chunks: dict[str, list[Tensor]] = {}
    max_error = 0.0
    try:
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                captured.clear()
                batch = collate(items[start:start + batch_size], tokenizer.pad_token_id, str(device), loops)
                result, scores = forward_symbols(model, batch['input_ids'], batch['attention_mask'], token_ids, num_loops=loops)
                values = {'context': captured['context'][0], 'initial': captured['initial'][0],
                          'working': torch.stack(captured['working'], 1),
                          'memory': torch.stack(captured['memory'], 1), 'logits': result.stop_logits,
                          'predictions': scores.argmax(-1)}
                repeated, initial, memory = replay(replay_head, values['context'], values['working'])
                for actual, expected in [(repeated, values['logits']), (initial, values['initial']), (memory, values['memory'])]:
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
                max_error = max(max_error, (repeated - values['logits']).abs().max().item())
                for name, value in values.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'Nonfinite captured {name}')
                    chunks.setdefault(name, []).append(value.cpu())
                if progress:
                    progress(min(start + batch_size, len(items)))
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(values) for name, values in chunks.items()}, deepcopy(replay_head).cpu(), max_error

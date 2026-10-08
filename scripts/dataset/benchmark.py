"""Independent, seed-replayable graph panels for frozen pointer benchmarks."""
from dataclasses import replace
import json
from pathlib import Path

from .pointer import (PointerExample, example_seed, execute, generate_unconditioned_example,
                      orbit_structure, render_prompt, validate_example)


def count_variant(task: PointerExample, count: int) -> PointerExample:
    """Change requested horizon/labels only; never condition graph sampling on it."""
    states = execute(dict(task.mapping), task.initial_state, count)
    result = replace(task, example_id=f'{task.example_id}-steps-{count}', task_depth=count,
                     intermediate_states=states, final_state=states[-1],
                     prompt=render_prompt(task.mapping, task.initial_state, count))
    validate_example(result)
    return result


def independent_graphs(config: dict, excluded: set[str]) -> tuple[list[PointerExample], list[dict]]:
    """Balanced seed/mode strata; reject only graph identity overlap, never difficulty."""
    if config['graphs_per_seed_mode'] < 1 or not 1 <= config['max_depth'] <= 256:
        raise ValueError('Need positive graph count and a maximum depth in 1..256')
    if len(set(config['seeds'])) != len(config['seeds']) or len(set(config['graph_modes'])) != len(config['graph_modes']):
        raise ValueError('Repeated seeds or graph modes')
    used = set(excluded)
    tasks, metadata = [], []
    for seed in config['seeds']:
        for mode in config['graph_modes']:
            for index in range(config['graphs_per_seed_mode']):
                for attempt in range(1000):
                    task = generate_unconditioned_example(example_seed(seed, f'benchmark-{mode}', index, attempt),
                        config['max_depth'], 'benchmark', index, mode)
                    if task.mapping_sha256 not in used:
                        break
                else:
                    raise ValueError('Unable to sample a new rule table after 1000 attempts')
                used.add(task.mapping_sha256)
                task = replace(task, example_id=f'benchmark-{seed}-{mode}-{index:06d}-{task.seed:016x}')
                validate_example(task)
                tasks.append(task)
                metadata.append({'graph_index': len(tasks) - 1, 'example_id': task.example_id,
                    'mapping_sha256': task.mapping_sha256, 'dataset_seed': seed, 'graph_mode': mode,
                    **orbit_structure(dict(task.mapping), task.initial_state)})
    return tasks, metadata


def exclusion_hashes(directory: Path) -> set[str]:
    """Read graph fingerprints, including reserved splits, solely to prevent overlap."""
    hashes = set()
    for path in sorted(directory.glob('*.jsonl')):
        for line in path.open():
            record = json.loads(line)
            hashes.add(record['mapping_sha256'])
    if not hashes:
        raise ValueError('No old dataset graph fingerprints found')
    return hashes

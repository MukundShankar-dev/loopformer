"""Predeclared pointer counterfactuals with independently executed targets."""

from dataclasses import dataclass
import hashlib
import random

from scripts.dataset.pointer import PointerExample, SYMBOLS, execute, render_prompt


@dataclass(frozen=True)
class Variant:
    name: str
    prompt: str
    targets: tuple[str, ...]
    displayed_steps: int
    changed_source: str | None = None
    old_destination: str | None = None
    new_destination: str | None = None
    probe_loop: int | None = None
    branch: str | None = None
    symbol_map: tuple[str, ...] | None = None


def _rng(task: PointerExample, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"mechanism-v1:{task.example_id}:{purpose}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _variant(task: PointerExample, name: str, pairs: list[list[str]], start: str,
             displayed_steps: int, horizon: int,
             **metadata: str | int | tuple[str, ...] | None) -> Variant:
    mapping = dict(pairs)
    if len(pairs) != len(SYMBOLS) or set(mapping) != set(SYMBOLS):
        raise ValueError("Variant must keep one outgoing rule for every symbol")
    targets = tuple(execute(mapping, start, horizon))
    return Variant(name, render_prompt(pairs, start, displayed_steps), targets, displayed_steps, **metadata)


def structural_variants(task: PointerExample) -> tuple[Variant, ...]:
    """Same problem under rule order, symbol names, depth cue, and unused edge."""
    path = [task.initial_state, *task.intermediate_states]
    table = dict(task.mapping)
    shuffled = [pair.copy() for pair in task.mapping]
    _rng(task, "order").shuffle(shuffled)
    if shuffled == task.mapping:
        shuffled = shuffled[1:] + shuffled[:1]
    reordered = _variant(task, "rule_order", shuffled, task.initial_state,
                         task.task_depth, task.task_depth)

    names = list(SYMBOLS)
    _rng(task, "symbols").shuffle(names)
    rename = dict(zip(SYMBOLS, names, strict=True))
    relabeled_pairs = [[rename[source], rename[destination]] for source, destination in task.mapping]
    relabeled = _variant(task, "symbol_rename", relabeled_pairs, rename[task.initial_state],
                         task.task_depth, task.task_depth,
                         symbol_map=tuple(rename[symbol] for symbol in SYMBOLS))
    if relabeled.targets != tuple(rename[state] for state in task.intermediate_states):
        raise AssertionError("Symbol relabeling changed the transition structure")

    depth_cue = _variant(task, "steps_plus_one", task.mapping, task.initial_state,
                         task.task_depth + 1, task.task_depth)
    unused = [symbol for symbol in SYMBOLS if symbol not in path]
    if not unused:
        raise ValueError("No source outside the nominal path")
    source = _rng(task, "unused-source").choice(unused)
    candidates = [symbol for symbol in SYMBOLS if symbol != table[source]]
    destination = _rng(task, "unused-destination").choice(candidates)
    edited = [[key, destination if key == source else value] for key, value in task.mapping]
    irrelevant = _variant(task, "irrelevant_edge", edited, task.initial_state,
                          task.task_depth, task.task_depth, changed_source=source,
                          old_destination=table[source], new_destination=destination)
    if reordered.targets != tuple(task.intermediate_states) or depth_cue.targets != reordered.targets or irrelevant.targets != reordered.targets:
        raise AssertionError("A structure-preserving control changed the reference path")
    return reordered, relabeled, depth_cue, irrelevant


def directed_edge_variants(task: PointerExample, loop: int, *, exclude_prediction: str,
                           branch: str, replacements: int = 2) -> tuple[Variant, ...]:
    """Two balanced off-path destinations, each with a matched irrelevant edit.

    Excluding the existing prediction prevents an unchanged answer from being
    counted as a response to the counterfactual. Early edits execute the whole
    changed branch; critical edits execute only through the probed transition.
    """
    if not 1 <= loop <= task.task_depth or branch not in ("early", "critical") or replacements < 1:
        raise ValueError("Invalid directed-edge probe")
    path = [task.initial_state, *task.intermediate_states]
    mapping = dict(task.mapping)
    candidates = [symbol for symbol in SYMBOLS if symbol not in path and symbol != exclude_prediction]
    _rng(task, f"targets:{branch}:{loop}").shuffle(candidates)
    if len(candidates) < replacements:
        raise ValueError("Not enough off-path replacements")
    source = path[loop - 1]
    horizon = task.task_depth if branch == "early" else loop
    result = []
    for index, destination in enumerate(candidates[:replacements]):
        unused_sources = [symbol for symbol in SYMBOLS if symbol not in path and mapping[symbol] != destination]
        irrelevant_source = _rng(task, f"control:{branch}:{loop}:{index}").choice(unused_sources)
        for condition, changed_source in (("relevant", source), ("irrelevant", irrelevant_source)):
            pairs = [[key, destination if key == changed_source else value] for key, value in task.mapping]
            variant = _variant(task, f"{branch}_{condition}_{index + 1}", pairs,
                               task.initial_state, task.task_depth, horizon,
                               changed_source=changed_source, old_destination=mapping[changed_source],
                               new_destination=destination, probe_loop=loop, branch=branch)
            if variant.targets[:loop - 1] != tuple(task.intermediate_states[:loop - 1]):
                raise AssertionError("Directed edit changed an earlier reference transition")
            target = destination if condition == "relevant" else task.intermediate_states[loop - 1]
            if variant.targets[loop - 1] != target:
                raise AssertionError("Directed edit has the wrong counterfactual target")
            result.append(variant)
    return tuple(result)

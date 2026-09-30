"""Reference-checked, single-edge prompt interventions for pointer diagnostics."""

from dataclasses import dataclass

from scripts.dataset.pointer import PointerExample, SYMBOLS, execute, render_prompt


@dataclass(frozen=True)
class RuleEdit:
    condition: str
    loop: int
    source: str
    old_destination: str
    new_destination: str
    expected: str
    prompt: str
    targets: tuple[str, ...]


def paired_rule_edits(task: PointerExample, loop: int) -> tuple[RuleEdit, RuleEdit]:
    """Change one relevant or unused edge; preserve every reference target before loop.

    The off-path destination is chosen from symbols absent from the complete
    reference path. Only the prefix through ``loop`` is scored: edited later
    paths can differ from the original and need not satisfy dataset sampling rules.
    """
    if not 1 <= loop <= task.task_depth:
        raise ValueError("Probe loop must be within the task depth")
    path = [task.initial_state, *task.intermediate_states]
    unused = [symbol for symbol in SYMBOLS if symbol not in path]
    if not unused:
        raise ValueError("No off-path symbol available")
    mapping = dict(task.mapping)
    relevant_source = path[loop - 1]
    irrelevant_source = next((symbol for symbol in unused if mapping[symbol] != unused[0]), None)
    if irrelevant_source is None:
        raise ValueError("No irrelevant rule can be changed to the selected destination")
    destination = unused[0]
    edits = []
    for condition, source in (("relevant", relevant_source), ("irrelevant", irrelevant_source)):
        old = mapping[source]
        pairs = [[key, destination if key == source else value] for key, value in task.mapping]
        if sum(a != b for a, b in zip(task.mapping, pairs, strict=True)) != 1:
            raise AssertionError("Expected exactly one modified edge")
        targets = tuple(execute(dict(pairs), task.initial_state, loop))
        if targets[:-1] != tuple(task.intermediate_states[:loop - 1]):
            raise AssertionError("Edited an earlier reference transition")
        expected = destination if condition == "relevant" else task.intermediate_states[loop - 1]
        if targets[-1] != expected:
            raise AssertionError("Counterfactual target does not match the edited edge")
        edits.append(RuleEdit(condition, loop, source, old, destination, expected,
                              render_prompt(pairs, task.initial_state, task.task_depth), targets))
    return edits[0], edits[1]

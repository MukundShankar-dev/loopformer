"""Deterministic, evaluation-only absorbing-terminal variants of pointer tasks."""

from dataclasses import replace

from .pointer import PointerExample, execute, mapping_fingerprint, render_prompt, validate_example


TRANSFORM_VERSION = "absorbing-terminal-v1"


def absorbing_terminal(example: PointerExample) -> PointerExample:
    """Replace only the final state's outgoing edge, preserving nominal execution.

    The original d+1 path states are distinct, so that edge is not followed until
    loop d+1. The changed prompt is a new distribution, not a retention label
    retroactively applied to the original task. Source IDs/seeds support pairing.
    """
    validate_example(example)
    final = example.final_state
    pairs = [[source, final if source == final else target] for source, target in example.mapping]
    result = replace(
        example, mapping=pairs, mapping_sha256=mapping_fingerprint(dict(pairs)),
        prompt=render_prompt(pairs, example.initial_state, example.task_depth),
    )
    validate_example(result)
    states = execute(dict(pairs), result.initial_state, result.task_depth + 2)
    if states != [*example.intermediate_states, final, final]:
        raise ValueError("Terminal transform changed the nominal trajectory")
    return result

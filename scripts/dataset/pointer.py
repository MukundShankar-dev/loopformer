"""Explicit-step pointer programs and exact, one-based execution targets."""

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re
from typing import Sequence


SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PAIR = re.compile(r"\(\s*([A-Z])\s*,\s*([A-Z])\s*\)")


def parse_mapping(text: str) -> dict[str, str]:
    """Parse pairs such as '(A,C) (C,D) (D, E)', rejecting ambiguous rules."""
    matches = list(PAIR.finditer(text))
    if not matches or PAIR.sub("", text).strip():
        raise ValueError("Mapping must be whitespace-separated (A,C) pairs")
    mapping = {}
    for match in matches:
        source, target = match.groups()
        if source in mapping:
            raise ValueError(f"Duplicate source symbol: {source}")
        mapping[source] = target
    return mapping


def execute(mapping: dict[str, str], start: str, steps: int) -> list[str]:
    """Return states after transitions 1..steps, excluding the initial state.

    This reference interpreter permits cycles and partial tables, provided every
    requested transition exists. The dataset sampler applies stricter constraints.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    states = []
    current = start
    for step in range(1, steps + 1):
        if current not in mapping:
            raise ValueError(f"No rule for {current!r} at step {step}")
        current = mapping[current]
        states.append(current)
    return states


def check_predictions(
    mapping: dict[str, str], start: str, predictions: Sequence[str], steps: int,
) -> list[bool]:
    """Score decoded symbols at steps 1..N against the exact nominal trajectory.

    Predictions may be a prefix (too few model loops), but cannot extend beyond
    the requested depth. Invalid/multi-symbol answers are incorrect; no lenient
    extraction of a letter from prose is performed. Leading whitespace is allowed.
    """
    targets = execute(mapping, start, steps)
    if isinstance(predictions, str) or len(predictions) > steps:
        raise ValueError("Provide a sequence of at most 'steps' decoded predictions")
    return [prediction.strip() == target for prediction, target in zip(predictions, targets)]


def render_prompt(pairs: list[list[str]], start: str, steps: int) -> str:
    # Spaces before symbols prevent Qwen punctuation/letter token merges.
    rules = " ".join(f"( {source}, {target})" for source, target in pairs)
    return f"Rules: {rules}\nStart: {start}\nSteps: {steps}\nAnswer:"


def mapping_fingerprint(mapping: dict[str, str]) -> str:
    """Ignore rule order, start, and depth when detecting reused rule tables."""
    payload = json.dumps(sorted(mapping.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def example_seed(master_seed: int, split: str, index: int, attempt: int = 0) -> int:
    payload = f"pointer-v1:{master_seed}:{split}:{index}:{attempt}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


@dataclass
class PointerExample:
    schema_version: int
    example_id: str
    family: str
    split: str
    seed: int
    task_depth: int
    mapping: list[list[str]]
    mapping_sha256: str
    initial_state: str
    intermediate_states: list[str]
    final_state: str
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_example(seed: int, depth: int, split: str, index: int) -> PointerExample:
    """Sample a full table, conditioned on a non-repeating nominal path.

    Every example uses all 26 symbols, keeping rule count independent of depth.
    The d+1 distinct path states prevent an early/repeated final answer; remaining
    edges are independent random choices and can cycle outside the nominal path.
    Rule order is shuffled so the solution is not printed in execution order.
    """
    if type(depth) is not int or not 1 <= depth < len(SYMBOLS):
        raise ValueError("depth must be an integer between 1 and 25")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed)
    path = rng.sample(SYMBOLS, depth + 1)
    mapping = dict(zip(path[:-1], path[1:]))
    for symbol in SYMBOLS:
        if symbol not in mapping:
            mapping[symbol] = rng.choice(SYMBOLS)
    pairs = [[source, mapping[source]] for source in SYMBOLS]
    rng.shuffle(pairs)
    states = execute(mapping, path[0], depth)
    return PointerExample(
        schema_version=1, example_id=f"{split}-{index:06d}-{seed:016x}",
        family="pointer", split=split, seed=seed, task_depth=depth,
        mapping=pairs, mapping_sha256=mapping_fingerprint(mapping),
        initial_state=path[0], intermediate_states=states, final_state=states[-1],
        prompt=render_prompt(pairs, path[0], depth),
    )


def validate_example(example: PointerExample) -> None:
    """Reparse the actual prompt and independently recompute every saved target."""
    if example.schema_version != 1 or example.family != "pointer":
        raise ValueError("Unsupported pointer record schema/family")
    if type(example.task_depth) is not int or not 1 <= example.task_depth < len(SYMBOLS):
        raise ValueError("Invalid task depth")
    if any(len(pair) != 2 for pair in example.mapping):
        raise ValueError("Every mapping entry must have two symbols")
    mapping = dict(example.mapping)
    if len(example.mapping) != len(SYMBOLS) or set(mapping) != set(SYMBOLS):
        raise ValueError("Each vocabulary symbol must have exactly one rule")
    if not set(mapping.values()) <= set(SYMBOLS):
        raise ValueError("Mapping target outside vocabulary")
    if example.mapping_sha256 != mapping_fingerprint(mapping):
        raise ValueError("Mapping fingerprint mismatch")
    if example.prompt != render_prompt(example.mapping, example.initial_state, example.task_depth):
        raise ValueError("Prompt disagrees with structured task")
    parsed = parse_mapping(example.prompt.splitlines()[0].removeprefix("Rules: "))
    states = execute(parsed, example.initial_state, example.task_depth)
    if states != example.intermediate_states or example.final_state != states[-1]:
        raise ValueError("Intermediate/final labels disagree with reference execution")
    if len(set([example.initial_state, *states])) != example.task_depth + 1:
        raise ValueError("Nominal path must not repeat a state")

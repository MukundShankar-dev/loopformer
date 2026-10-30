"""Convert EvalPlus MBPP+ into Problems, computing expected outputs in the sandbox.

Run with the probe environment, which has EvalPlus:
    .venv-probe/bin/python -m scripts.codetasks.mbppplus --output data/agent/mbppplus-v0.2.0.jsonl

Inputs come from `evalplus.data.get_mbpp_plus`, which restores tuples and sets.
Expected outputs come from running each canonical solution in the sandbox on a
deep copy of each input. The public test is the input of the example assert in
the statement (the first original MBPP input if it cannot be matched); hidden
tests are every base and plus input.

Matching EvalPlus's checker: absolute tolerance 1e-6 when a task's expected
values are floats and it sets none, with relative 1e-7 applied by the grader.
Excluded, and listed in the run's output: the tasks EvalPlus checks with
special oracles (set equality, "not None", or a custom function), which
input/expected equality cannot express, and tasks whose reference solution
fails or exceeds REFERENCE_LIMITS in the sandbox.
Remaining differences: one process per case instead of one per sample, a
fixed per-case time limit instead of 4x the reference time, and the system
Python rather than EvalPlus's.
"""
import argparse
import ast
import json
from pathlib import Path

from scripts.codetasks.problems import Problem, write_problems
from scripts.sandbox.grader import TestCase, Tests
from scripts.sandbox.runner import Limits, run
from scripts.sandbox.values import decode, encode

SOURCE = "mbppplus-v0.2.0"
SPECIAL_ORACLE_ENTRY_POINTS = {
    "are_equivalent", "sum_div", "surface_Area", "digit_distance_nums",  # custom checks
    "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly", "larg_nnum",
    "intersection_array", "find_dissimilar", "Diff",  # set equality
    "check_str", "text_match_three", "text_starta_endb",  # "not None"
}
FLOAT_ATOL = 1e-6
REFERENCE = """\
import copy, importlib.util, json, sys
spec = importlib.util.spec_from_file_location('values', '/code/values.py')
values = importlib.util.module_from_spec(spec); spec.loader.exec_module(values)
request = json.loads(sys.stdin.read())
namespace = {}
exec(request['solution'], namespace)
function = namespace[request['function']]
outputs = []
for arguments in values.decode(request['inputs']):
    try:
        outputs.append(values.encode(function(*copy.deepcopy(arguments))))
    except Exception as error:
        outputs.append({'__error__': type(error).__name__})
print(json.dumps(outputs))
"""
REFERENCE_LIMITS = Limits(wall_seconds=300, cpu_seconds=300, memory_bytes=2 << 30, output_bytes=256 << 20,
                          file_bytes=512 << 20)


def is_floats(value) -> bool:
    """EvalPlus's test for enforcing a float tolerance: a float, or a non-empty list or tuple of floats."""
    if isinstance(value, float):
        return True
    return isinstance(value, (list, tuple)) and bool(value) and all(isinstance(item, float) for item in value)


class ReferenceFailed(Exception):
    pass


def reference_outputs(solution: str, function: str, inputs: list) -> list:
    """Encoded outputs of a trusted reference solution, one per input, computed in the sandbox."""
    request = json.dumps({"solution": solution, "function": function, "inputs": encode(inputs)})
    values_source = (Path(__file__).parents[1] / "sandbox" / "values.py").read_text()
    result = run({"reference.py": REFERENCE, "values.py": values_source}, entry="reference.py", stdin=request,
                 limits=REFERENCE_LIMITS)
    if result.exit_code != 0 or result.stdout_truncated:
        raise ReferenceFailed(f"exit {result.exit_code}, OOM {result.oom_killed}, timed out {result.timed_out}, "
                              f"truncated {result.stdout_truncated}: {result.stderr[-300:]}")
    return json.loads(result.stdout)


def statement_example(prompt: str, function: str, inputs: list) -> int | None:
    """Index of the input used by the statement's `assert function(...) == ...`, if it is a literal call."""
    asserts = "\n".join(line for line in prompt.splitlines() if line.strip().startswith("assert "))
    try:
        tree = ast.parse(asserts)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function:
            try:
                arguments = [ast.literal_eval(argument) for argument in node.args]
            except ValueError:
                return None
            matches = [index for index, candidate in enumerate(inputs) if list(candidate) == arguments]
            return matches[0] if matches else None
    return None


def convert(task: dict) -> Problem:
    """One MBPP+ task as a Problem; raises ReferenceFailed for tasks that must be excluded."""
    function = task["entry_point"]
    if function in SPECIAL_ORACLE_ENTRY_POINTS:
        raise ReferenceFailed("special oracle in EvalPlus")
    inputs = [list(arguments) for arguments in [*task["base_input"], *task["plus_input"]]]
    outputs = reference_outputs(task["prompt"] + task["canonical_solution"], function, inputs)
    errors = [output for output in outputs if isinstance(output, dict) and "__error__" in output]
    if errors:
        raise ReferenceFailed(f"reference raised on {len(errors)} inputs: {errors[:3]}")
    atol = task["atol"] or (FLOAT_ATOL if any(is_floats(decode(output)) for output in outputs) else 0.0)
    cases = tuple(TestCase(json.dumps(encode(arguments)), json.dumps(output))
                  for arguments, output in zip(inputs, outputs))
    example = statement_example(task["prompt"], function, inputs)
    public = cases[example if example is not None else 0]
    return Problem(id=task["task_id"], source=SOURCE, statement=task["prompt"].strip().strip('"').strip(),
                   public=Tests("call", (public,), function=function, atol=atol),
                   hidden=Tests("call", cases, function=function, atol=atol))


def main() -> None:
    from evalplus.data import get_mbpp_plus  # only in the probe environment
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = get_mbpp_plus()
    problems, excluded = [], {}
    for task_id, task in tasks.items():
        try:
            problems.append(convert(task))
        except ReferenceFailed as reason:
            excluded[task_id] = str(reason)
    write_problems(args.output, problems)
    args.output.with_suffix(".excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    console = Console()
    for task_id, reason in excluded.items():
        console.print(f"excluded {task_id}: {reason[:160]}", markup=False)
    console.print(f"{len(problems)} of {len(tasks)} MBPP+ tasks written to {args.output}; "
                  f"{sum(len(p.hidden.cases) for p in problems)} hidden cases")


if __name__ == "__main__":
    main()

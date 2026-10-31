"""Convert EvalPlus HumanEval+ or MBPP+ into a problem set, computing expected outputs in the sandbox.

Run with the probe environment, which has EvalPlus:
    .venv-probe/bin/python -m scripts.codetasks.evalplus --dataset mbpp --output data/agent/mbppplus-v0.2.0
    .venv-probe/bin/python -m scripts.codetasks.evalplus --dataset humaneval --output data/agent/humanevalplus-v0.1.10

Inputs come from EvalPlus's loaders, which restore tuples and sets. Expected
outputs come from running each canonical solution in the sandbox on a deep
copy of each input. The public test is the input of the statement's first
example call (an MBPP `assert` or a HumanEval `>>>` doctest), or the first
original input if no example can be matched; hidden tests are every base and
plus input. The statement is MBPP's description, or HumanEval's prompt
(imports, signature and docstring). HumanEval's prompt code also runs before
the solution (the tests' `imports`), as in the original HumanEval harness,
which executes prompt + completion: EvalPlus instead prefills the reply so the
model copies the prompt, which our turn format cannot do, and without it the
model drops the prompt's imports (`List`) and helpers. Showing the prompt in a
code block does not help: the model then often copies the stub without a body
(docs/history.md, M0d). EvalPlus requires self-contained code, so this is more
lenient than EvalPlus exactly for solutions that use a name the prompt defines.

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
import sys
from pathlib import Path

from scripts.codetasks.problems import Problem, in_memory, write_problems
from scripts.codetasks.reference import FLOAT_ATOL, ReferenceFailed, is_floats, reference_outputs
from scripts.sandbox.grader import TestCase, Tests
from scripts.sandbox.values import decode, encode

SOURCES = {"humaneval": "humanevalplus-v0.1.10", "mbpp": "mbppplus-v0.2.0"}  # EvalPlus 0.3.1's default versions
SPECIAL_ORACLE_ENTRY_POINTS = {
    "humaneval": {"find_zero"},  # any root within atol
    "mbpp": {
        "are_equivalent", "sum_div", "surface_Area", "digit_distance_nums",  # custom checks
        "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly", "larg_nnum",
        "intersection_array", "find_dissimilar", "Diff",  # set equality
        "check_str", "text_match_three", "text_starta_endb",  # "not None"
    },
}


def statement_example(prompt: str, function: str, inputs: list) -> int | None:
    """Index of the input used by the statement's first example call (`assert f(...)` or `>>> f(...)`), if literal."""
    examples = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("assert "):
            examples.append(line)
        elif line.startswith(">>> "):
            examples.append(line[4:])
    try:
        tree = ast.parse("\n".join(examples))
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


def convert(task: dict, dataset: str) -> Problem:
    """One EvalPlus task as a Problem; raises ReferenceFailed for tasks that must be excluded."""
    function = task["entry_point"]
    if function in SPECIAL_ORACLE_ENTRY_POINTS[dataset]:
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
    if dataset == "mbpp":
        text, imports = task["prompt"].strip().strip('"').strip(), ""
    else:
        text, imports = task["prompt"].strip(), task["prompt"]
    return Problem(id=task["task_id"], source=SOURCES[dataset], statement=text,
                   public=Tests("call", (public,), function=function, atol=atol, imports=imports),
                   hidden=in_memory(Tests("call", cases, function=function, atol=atol, imports=imports)))


def main() -> None:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus  # only in the probe environment
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True, help="problem-set directory to create")
    args = parser.parse_args()
    sys.set_int_max_str_digits(0)  # decoding the trusted reference outputs, some of which are huge integers
    tasks = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    problems, excluded = [], {}
    for task_id, task in tasks.items():
        try:
            problems.append(convert(task, args.dataset))
        except ReferenceFailed as reason:
            excluded[task_id] = str(reason)
    write_problems(args.output, problems)
    (args.output / "excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    console = Console()
    for task_id, reason in excluded.items():
        console.print(f"excluded {task_id}: {reason[:160]}", markup=False)
    console.print(f"{len(problems)} of {len(tasks)} {args.dataset} tasks written to {args.output}; "
                  f"{sum(len(p.hidden().cases) for p in problems)} hidden cases")


if __name__ == "__main__":
    main()

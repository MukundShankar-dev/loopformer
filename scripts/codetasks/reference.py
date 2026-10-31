"""Expected outputs from a trusted reference solution, computed in the sandbox.

Benchmarks that ship inputs and a reference solution but no expected values
(EvalPlus, KodCode's asserts) get them by calling the reference on a deep copy
of each input inside the sandbox. Values cross as tagged JSON
(scripts/sandbox/values.py). With `repeats` > 1 every input runs again and any
difference fails the reference, which catches solutions that depend on
randomness, time or state left by an earlier call.
"""
import json
from pathlib import Path

from scripts.sandbox.runner import Limits, run
from scripts.sandbox.values import encode

FLOAT_ATOL = 1e-6  # EvalPlus's tolerance when expected values are floats and a task sets none
REFERENCE = """\
import copy, importlib.util, json, os, sys
sys.set_int_max_str_digits(0)  # values.py encodes huge integers as decimal strings
result = os.fdopen(os.dup(1), 'w')
os.dup2(2, 1)  # the reference's own prints go to stderr
spec = importlib.util.spec_from_file_location('values', '/code/values.py')
values = importlib.util.module_from_spec(spec); spec.loader.exec_module(values)
request = json.loads(sys.stdin.read())
namespace = {}
exec(request['solution'], namespace)
function = namespace[request['function']]
runs = []
for _ in range(request['repeats']):
    outputs = []
    for arguments in values.decode(request['inputs']):
        try:
            outputs.append(values.encode(function(*copy.deepcopy(arguments))))
        except Exception as error:
            outputs.append({'__error__': type(error).__name__})
    runs.append(outputs)
result.write(json.dumps(runs))
result.close()
"""
REFERENCE_LIMITS = Limits(wall_seconds=300, cpu_seconds=300, memory_bytes=2 << 30, output_bytes=256 << 20,
                          file_bytes=512 << 20)


class ReferenceFailed(Exception):
    pass


def is_floats(value) -> bool:
    """EvalPlus's test for enforcing a float tolerance: a float, or a non-empty list or tuple of floats."""
    if isinstance(value, float):
        return True
    return isinstance(value, (list, tuple)) and bool(value) and all(isinstance(item, float) for item in value)


def reference_outputs(solution: str, function: str, inputs: list, limits: Limits = REFERENCE_LIMITS,
                      repeats: int = 1) -> list:
    """Encoded outputs of `function` in `solution`, one per input; errors appear as {"__error__": name}."""
    request = json.dumps({"solution": solution, "function": function, "inputs": encode(inputs), "repeats": repeats})
    values_source = (Path(__file__).parents[1] / "sandbox" / "values.py").read_text()
    result = run({"reference.py": REFERENCE, "values.py": values_source}, entry="reference.py", stdin=request,
                 limits=limits)
    if result.exit_code != 0 or result.stdout_truncated:
        raise ReferenceFailed(f"exit {result.exit_code}, OOM {result.oom_killed}, timed out {result.timed_out}, "
                              f"truncated {result.stdout_truncated}: {result.stderr[-300:]}")
    try:
        runs = json.loads(result.stdout)
    except ValueError as error:
        raise ReferenceFailed(f"unreadable output: {error}") from None
    if any(outputs != runs[0] for outputs in runs[1:]):
        raise ReferenceFailed("nondeterministic: outputs differ between runs")
    return runs[0]

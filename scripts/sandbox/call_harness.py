"""Runs inside the sandbox for call-style tests; copied in as /code/harness.py.

Reads `{"function": name, "arguments": [...], "imports": code}` from stdin,
loads /code/solution.py after running `imports` in its namespace, calls `name`
(a module function, or a method of `Solution()` in LeetCode style) and writes
the return value to the original stdout. Arguments and result use the tagged
JSON of /code/values.py (scripts/sandbox/values.py). The solution's own prints
go to stderr.

With `"batch": [arguments, ...]` in place of `"arguments"`, the solution loads
once and each case is called in turn (with a fresh `Solution()`), writing one
line per case: `{"case": i, "value": ...}`, or `{"case": i}` if the call
raised. Exceptions that are not `Exception`s (`exit()`) end the run. The host
decides only returned values from a batch and re-runs every other case alone
(scripts/sandbox/grader.py).

The solution runs in this process and can tamper with anything here. That is
acceptable because the host trusts the output only as "the value the solution
produced": the expected value never enters the sandbox.
"""
import importlib.util
import json
import os
import sys


def load(name: str, imports: str = ""):
    spec = importlib.util.spec_from_file_location(name, f"/code/{name}.py")
    module = importlib.util.module_from_spec(spec)
    exec(compile(imports, "imports.py", "exec"), module.__dict__)
    spec.loader.exec_module(module)
    return module


def function(module, name: str):
    target = getattr(module, name, None)
    if target is None and hasattr(module, "Solution"):
        target = getattr(module.Solution(), name, None)
    if not callable(target):
        raise AttributeError(f"solution defines no function {name!r}")
    return target


def main() -> None:
    sys.set_int_max_str_digits(0)  # values.py encodes huge integers as decimal strings
    values = load("values")
    request = json.loads(sys.stdin.read())
    arguments = values.decode(request["arguments"]) if "batch" not in request else None
    result = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # anything the solution prints goes to stderr
    module = load("solution", request["imports"])
    if arguments is not None:
        result.write(json.dumps(values.encode(function(module, request["function"])(*arguments))))
        result.close()
        return
    for case, encoded in enumerate(request["batch"]):
        try:
            line = json.dumps({"case": case, "value": values.encode(
                function(module, request["function"])(*values.decode(encoded)))})
        except Exception:
            line = json.dumps({"case": case})
        result.write(line + "\n")
        result.flush()
    result.close()


main()

"""Runs inside the sandbox for call-style tests; copied in as /code/harness.py.

Reads `{"function": name, "arguments": [...]}` from stdin, loads
/code/solution.py, calls `name` (a module function, or a method of `Solution()`
in LeetCode style) and writes the return value to the original stdout.
Arguments and result use the tagged JSON of /code/values.py
(scripts/sandbox/values.py). The solution's own prints go to stderr.

The solution runs in this process and can tamper with anything here. That is
acceptable because the host trusts the output only as "the value the solution
produced": the expected value never enters the sandbox.
"""
import importlib.util
import json
import os
import sys


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, f"/code/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    values = load("values")
    request = json.loads(sys.stdin.read())
    arguments = values.decode(request["arguments"])
    result = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # anything the solution prints goes to stderr
    module = load("solution")
    name = request["function"]
    target = getattr(module, name, None)
    if target is None and hasattr(module, "Solution"):
        target = getattr(module.Solution(), name, None)
    if not callable(target):
        raise AttributeError(f"solution defines no function {name!r}")
    encoded = values.encode(target(*arguments))
    result.write(json.dumps(encoded))
    result.close()


main()

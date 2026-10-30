"""Runs inside the sandbox for call-style tests; copied in as /code/harness.py.

Reads `{"function": name, "arguments": [...]}` from stdin, loads
/code/solution.py, calls `name` (a module function, or a method of `Solution()`
in LeetCode style) and writes the JSON-encoded return value to the original
stdout. The solution's own prints are redirected to stderr.

The solution runs in this process and can tamper with anything here. That is
acceptable because the host trusts the output only as "the value the solution
produced": the expected value never enters the sandbox.
"""
import importlib.util
import json
import os
import sys


def encode(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"return value of type {type(value).__name__} is not JSON-serializable")


def main() -> None:
    request = json.loads(sys.stdin.read())
    result = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # anything the solution prints goes to stderr
    spec = importlib.util.spec_from_file_location("solution", "/code/solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = request["function"]
    target = getattr(module, name, None)
    if target is None and hasattr(module, "Solution"):
        target = getattr(module.Solution(), name, None)
    if not callable(target):
        raise AttributeError(f"solution defines no function {name!r}")
    value = target(*request["arguments"])
    result.write(json.dumps(value, default=encode))
    result.close()


main()

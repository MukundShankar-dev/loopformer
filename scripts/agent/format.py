"""The agent's turn format: instructions, grammar, parsing and observations.

A turn is exactly one tool call:

    run_python
    ```python
    <code>
    ```
    ```stdin
    <optional input>
    ```

The other tools are `run_public_tests` and `submit`, each with one python
block. `grammar(tools)` is the EBNF that constrained decoding enforces, so a
generated turn always parses; the single-turn and repair conditions use the
same format with only `submit` allowed. Observations are rendered as user messages.

There is no free-text field: with one, the base model under greedy decoding
often never ended it and so never submitted (docs/history.md, M0d).
"""
import json
import re
from dataclasses import dataclass

from scripts.sandbox.grader import Grade, Tests
from scripts.sandbox.runner import RunResult
from scripts.sandbox.values import decode

TOOLS = ("run_python", "run_public_tests", "submit")
OBSERVATION_CHARACTERS = 2000  # of stdout and of stderr, each, shown to the agent
# Mirrors `grammar` rule for rule; tests/test_agent.py checks that both accept the same turns.
BODY = r"(?:[^`]|`[^`]|``[^`])*"
TURN = re.compile(rf"\A(?:(?P<tool>run_python)\n```python\n(?P<code>{BODY})```(?:\n```stdin\n(?P<stdin>{BODY})```)?"
                  rf"|(?P<other>run_public_tests|submit)\n```python\n(?P<other_code>{BODY})```)\n?\Z")

TOOL_HELP = {
    "run_python": "run_python: runs your code as a script, with optional standard input in a ```stdin block. "
                  "You see stdout, stderr and the exit status.",
    "run_public_tests": "run_public_tests: runs your code against the public tests. You see each test's result.",
    "submit": "submit: submits your code as the final answer. This ends the episode.",
}
REPAIR_SUBMIT_HELP = ("submit: submits your code. It is run against the public tests; if one fails, you see each "
                      "test's result and submit a revision. The episode ends when the public tests pass or turns "
                      "run out, and your last submission is the final answer.")


@dataclass(frozen=True)
class Turn:
    tool: str
    code: str
    stdin: str | None


def instructions(tools: tuple[str, ...], max_turns: int, repair: bool = False) -> str:
    """System prompt for the allowed tools; identical across conditions except for the tool list."""
    if not tools or any(tool not in TOOLS for tool in tools) or "submit" not in tools:
        raise ValueError(f"tools must be a subset of {TOOLS} that includes submit")
    stdin_line = "```stdin\n<optional input>\n```\n" if "run_python" in tools else ""
    return (
        "You are a Python programming agent. Solve the problem in Python using only the standard library. "
        "Code you run or submit must be complete and self-contained, including every import it needs.\n"
        f"Each turn, write exactly one tool call. You have at most {max_turns} turns.\n\nTools:\n"
        + "\n".join(f"- {REPAIR_SUBMIT_HELP if repair else TOOL_HELP[tool]}" for tool in tools)
        + "\n\nFormat:\n<tool name>\n```python\n<complete code>\n```\n" + stdin_line
    )


def grammar(tools: tuple[str, ...]) -> str:
    """EBNF (xgrammar dialect) accepting exactly the turns `TURN` parses, for these tools."""
    calls = []
    if "run_python" in tools:
        calls.append('"run_python\\n" code ("\\n```stdin\\n" body "```")?')
    if "run_public_tests" in tools:
        calls.append('"run_public_tests\\n" code')
    if "submit" in tools:
        calls.append('"submit\\n" code')
    return "\n".join([
        # The optional final newline lets the model close a block with its usual fence-and-newline token.
        f'root ::= ({" | ".join(calls)}) "\\n"?',
        'code ::= "```python\\n" body "```"',
        # Any text without three consecutive backticks.
        'body ::= ([^`] | "`" [^`] | "``" [^`])*',
    ])


def parse_turn(text: str) -> Turn:
    match = TURN.match(text)
    if match is None:
        raise ValueError(f"turn does not match the format: {text[:200]!r}")
    if match["tool"]:
        return Turn(match["tool"], match["code"], match["stdin"])
    return Turn(match["other"], match["other_code"], None)


def _clip(text: str) -> str:
    if len(text) <= OBSERVATION_CHARACTERS:
        return text
    return text[:OBSERVATION_CHARACTERS] + f"\n... [{len(text) - OBSERVATION_CHARACTERS} more characters]"


def render_run(result: RunResult) -> str:
    status = "timed out" if result.timed_out else "out of memory" if result.oom_killed else f"exit code {result.exit_code}"
    parts = [f"run_python finished: {status}, {result.wall_seconds:.2f} s."]
    for name, text, truncated in [("stdout", result.stdout, result.stdout_truncated),
                                  ("stderr", result.stderr, result.stderr_truncated)]:
        if text:
            parts.append(f"{name}{' (truncated)' if truncated else ''}:\n{_clip(text)}")
    return "\n".join(parts)


def _shown_value(encoded: str) -> str:
    """repr of a tagged-JSON value for display; the raw text if it is not valid (it may be untrusted)."""
    try:
        return repr(decode(json.loads(encoded)))
    except (ValueError, RecursionError, TypeError):
        return encoded


def render_public_tests(tests: Tests, grade: Grade, callee: str | None, tool: str = "run_public_tests") -> str:
    """`callee` is how call tests invoke the function (`f` or `Solution().f`); None for stdio tests."""
    passed = sum(result.outcome == "pass" for result in grade.results)
    lines = [f"{tool}: {passed} of {len(grade.results)} public tests passed."]
    for number, (case, result) in enumerate(zip(tests.cases, grade.results), start=1):
        if tests.kind == "call":
            arguments = ", ".join(repr(argument) for argument in decode(json.loads(case.input)))
            call = f"{callee}({arguments})"
            got = _shown_value(result.stdout) if result.outcome == "pass" or result.outcome == "wrong_answer" else None
            line = f"Test {number}: {call} expected {_shown_value(case.expected)}"
        else:
            got = result.stdout if result.outcome in ("pass", "wrong_answer") else None
            line = f"Test {number}: input {case.input!r} expected output {case.expected!r}"
        line += f", got {_clip(got)}" if got is not None else ""
        line += f": {result.outcome}"
        if result.outcome not in ("pass", "wrong_answer") and result.stderr:
            line += f"\n{_clip(result.stderr)}"
        lines.append(line)
    return "\n".join(lines)

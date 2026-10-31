"""Grade LiveCodeBench generations with its own evaluator, inside the bubblewrap sandbox.

    python -m scripts.codeprobe.evaluate_livecodebench --lcb ~/src/LiveCodeBench --problems <dir> \\
        --generations <run>/generations.json --output <run>/eval_all.json

LiveCodeBench's `codegen_metrics` executes every sample, so it runs only in
the sandbox: one run per problem, because a problem's hidden tests reach
190 MB and the whole release does not fit in memory at once. Inside: the
probe environment and its interpreter, the LiveCodeBench checkout at /lcb
and the problem's hidden tests at /tests.json, all read-only. Generations
arrive on stdin and verdicts leave on stdout. The output is LiveCodeBench's
`eval_all` list (one row per problem with `graded_list`), which
scripts/codeprobe/headroom.py reads.

As with scripts/codeprobe/evaluate_sandboxed.py, verdicts are computed beside
the code being judged, so an adversarial sample could forge them; fine for
base-model numbers comparable with published ones, not for training rewards.
"""
import argparse
import json
import os
from pathlib import Path

from rich.console import Console

from scripts.sandbox.runner import Limits, run

WRAPPER = """\
import json, os, sys
result = os.fdopen(os.dup(1), 'w')
os.dup2(2, 1)  # LiveCodeBench prints progress
sys.path.insert(0, '/lcb')
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation import codegen_metrics, extract_instance_results
request = json.loads(sys.stdin.read())
problem = CodeGenerationProblem(**request['problem'], private_test_cases=open('/tests.json').read())
metrics, results, metadata = codegen_metrics([problem.get_evaluation_sample()], [request['code_list']], k_list=[1],
                                             num_process_evaluate=WORKERS, timeout=TIMEOUT)
result.write(json.dumps({'graded_list': extract_instance_results(results)[0], 'metadata': metadata[0]}))
result.close()
"""
GIB = 1 << 30


def problem_limits(samples: int, tests: int, timeout: int, workers: int, memory_gb: int) -> Limits:
    """A wall limit above LiveCodeBench's own per-sample cap, (timeout + 1) * tests + 5 s, so it never binds first."""
    wall = samples * ((timeout + 1) * tests + 5) / workers + 300
    return Limits(wall_seconds=wall, cpu_seconds=int(wall * workers), cpus=workers, memory_bytes=memory_gb * GIB,
                  processes=1024, file_bytes=GIB, tmpfs_bytes=2 * GIB, output_bytes=64 << 20, open_files=4096)


class SandboxLimit(Exception):
    """The sandbox killed the run for memory or time."""


def _sandboxed(problem: dict, code_list: list[str], tests: Path, count: int, lcb: Path, venv: Path, timeout: int,
               workers: int, memory_gb: int) -> dict:
    interpreter_root = Path(os.path.realpath(venv / "bin" / "python")).parents[1]
    mounts = {str(venv): venv, str(interpreter_root): interpreter_root, "/lcb": lcb, "/tests.json": tests}
    wrapper = WRAPPER.replace("WORKERS", str(workers)).replace("TIMEOUT", str(timeout))
    result = run({"evaluate.py": wrapper}, entry="evaluate.py", limits=problem_limits(len(code_list), count, timeout,
                                                                                         workers, memory_gb),
                 stdin=json.dumps({"problem": problem, "code_list": code_list}), mounts=mounts,
                 python=str(venv / "bin" / "python"))
    if result.oom_killed or result.timed_out:
        raise SandboxLimit("out of memory" if result.oom_killed else "timed out")
    if result.exit_code != 0 or result.stdout_truncated:
        raise RuntimeError(f"{problem['question_id']}: sandboxed evaluation failed (exit {result.exit_code}):\n"
                           f"{result.stderr[-4000:]}")
    verdict = json.loads(result.stdout)
    if len(verdict["graded_list"]) != len(code_list):
        raise RuntimeError(f"{problem['question_id']}: {len(verdict['graded_list'])} verdicts for {len(code_list)} samples")
    return verdict


def evaluate_problem(problem: dict, code_list: list[str], tests: Path, lcb: Path, venv: Path, timeout: int,
                     workers: int, memory_gb: int) -> dict:
    """LiveCodeBench's verdicts for one problem's samples, graded together in one sandbox run.

    LiveCodeBench sets no memory limit, so a sample that exhausts the sandbox's memory (or, failing that, its
    time) is a failed sample. If the shared run hits a limit, each sample is graded in its own run; a sample
    that hits a limit alone fails, with the limit in its metadata.
    """
    count = len(json.loads(tests.read_text())) + len(json.loads(problem["public_test_cases"]))
    arguments = (tests, count, lcb, venv, timeout, workers, memory_gb)
    try:
        return _sandboxed(problem, code_list, *arguments)
    except SandboxLimit:
        pass
    verdict = {"graded_list": [], "metadata": []}
    for code in code_list:
        try:
            single = _sandboxed(problem, [code], *arguments)
        except SandboxLimit as limit:
            single = {"graded_list": [False], "metadata": [json.dumps({"error_message": f"sandbox: {limit}"})]}
        verdict["graded_list"] += single["graded_list"]
        verdict["metadata"] += single["metadata"]
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lcb", type=Path, required=True, help="LiveCodeBench checkout")
    parser.add_argument("--problems", type=Path, required=True, help="output of scripts.codeprobe.livecodebench_data")
    parser.add_argument("--generations", type=Path, required=True, help="output of livecodebench_codegen")
    parser.add_argument("--output", type=Path, required=True, help="eval_all.json to write")
    parser.add_argument("--venv", type=Path, default=Path(".venv-probe").absolute())
    parser.add_argument("--timeout", type=int, default=6, help="LiveCodeBench's per-test limit in seconds (its default)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new output")
    problems = {json.loads(line)["question_id"]: json.loads(line)
                for line in (args.problems / "problems.jsonl").read_text().splitlines()}
    generations = json.loads(args.generations.read_text())
    missing = [row["question_id"] for row in generations if row["question_id"] not in problems]
    if missing:
        parser.error(f"generations for problems not in {args.problems}: {missing[:5]}")
    console = Console()
    rows = []
    for index, row in enumerate(generations, start=1):
        problem = problems[row["question_id"]]
        verdict = evaluate_problem(problem, row["code_list"], args.problems / "tests" / f"{row['question_id']}.json",
                                   args.lcb.absolute(), args.venv, args.timeout, args.workers, args.memory_gb)
        rows.append({**row, **verdict, "pass@1": sum(verdict["graded_list"]) / len(verdict["graded_list"])})
        console.print(f"{index}/{len(generations)} {row['question_id']} ({problem['difficulty']}): "
                      f"{sum(verdict['graded_list'])} of {len(verdict['graded_list'])} pass", markup=False)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

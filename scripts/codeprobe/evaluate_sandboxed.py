"""Run the official `evalplus.evaluate` inside the bubblewrap sandbox.

EvalPlus executes every sample, so it never runs on the host. Generation
(`evalplus.codegen`) executes nothing and runs outside the sandbox on the GPU.

Inside the sandbox: the probe environment and its interpreter, the samples
and the EvalPlus dataset cache are read-only at their host paths, `/samples.jsonl`
and `/evalplus-cache`. EvalPlus works in `/work` (it recomputes ground truth
there), and the results file comes back on stdout.

The verdicts are computed inside the sandbox by EvalPlus, alongside the code
being judged, so a deliberately adversarial sample could forge them. That is
acceptable for base-model numbers meant to be comparable with published ones;
scripts/sandbox/grader.py is the tamper-proof path used for training.
"""
import argparse
import os
from pathlib import Path

from scripts.sandbox.runner import Limits, run

WRAPPER = """\
import shutil, subprocess, sys
shutil.copytree('/evalplus-cache', '/work/.cache/evalplus')  # HOME is /work, so EvalPlus finds it here
shutil.copy('/samples.jsonl', '/work/samples.jsonl')
command = [sys.executable, '-m', 'evalplus.evaluate', '--dataset', DATASET, '--samples', '/work/samples.jsonl',
           '--parallel', PARALLEL]
code = subprocess.run(command, stdout=sys.stderr).returncode
if code:
    sys.exit(f'evalplus.evaluate exited with {code}')
sys.stdout.write(open('/work/samples_eval_results.json').read())
"""


def evaluate(samples: Path, dataset: str, output: Path, venv: Path, cache: Path, parallel: int, limits: Limits) -> None:
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new output")
    interpreter_root = Path(os.path.realpath(venv / "bin" / "python")).parents[1]
    mounts = {str(venv): venv, str(interpreter_root): interpreter_root, "/samples.jsonl": samples,
              "/evalplus-cache": cache}
    wrapper = WRAPPER.replace("DATASET", repr(dataset)).replace("PARALLEL", repr(str(parallel)))
    result = run({"evaluate.py": wrapper}, entry="evaluate.py", limits=limits, mounts=mounts,
                 python=str(venv / "bin" / "python"))
    if result.exit_code != 0 or result.stdout_truncated:
        raise RuntimeError(f"sandboxed evaluation failed (exit {result.exit_code}, timed out {result.timed_out}, "
                           f"OOM {result.oom_killed}, truncated {result.stdout_truncated}):\n{result.stderr[-4000:]}")
    output.write_text(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=Path, required=True, help="the .jsonl written by evalplus.codegen")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="where to write eval_results.json")
    parser.add_argument("--venv", type=Path, default=Path(".venv-probe").absolute())
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "evalplus")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--wall-hours", type=float, default=2.0)
    parser.add_argument("--memory-gb", type=int, default=8)
    args = parser.parse_args()
    gib = 1 << 30
    limits = Limits(wall_seconds=args.wall_hours * 3600, cpu_seconds=int(args.wall_hours * 3600), cpus=args.parallel,
                    memory_bytes=args.memory_gb * gib, processes=1024, file_bytes=gib, tmpfs_bytes=4 * gib,
                    output_bytes=gib, open_files=4096)
    evaluate(args.samples.absolute(), args.dataset, args.output, args.venv, args.cache, args.parallel, limits)


if __name__ == "__main__":
    main()

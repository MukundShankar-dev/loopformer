"""Build, print or run the headroom-probe commands from configs/probe_headroom.json.

Generation runs the official tools on the host, which execute no model code:
EvalPlus codegen through scripts/codeprobe/codegen.py and LiveCodeBench's
runner through scripts/codeprobe/livecodebench_codegen.py, both with vLLM's GPU
share capped. Everything that executes samples runs inside the sandbox:
EvalPlus's evaluator (scripts/codeprobe/evaluate_sandboxed.py) and
LiveCodeBench's (scripts/codeprobe/evaluate_livecodebench.py). LiveCodeBench
uses only problems dated before `livecodebench.before`, so the confirmation
window is never decoded or graded. Models load from pinned local snapshots.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.codeprobe.livecodebench_data import RELEASE_FILES

LCB_KEYS = ("release_version", "dataset", "dataset_revision", "raw", "checkout", "commit", "before", "problems",
            "published_window")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    for key in ("output", "models", "samples", "temperature", "evalplus_datasets", "livecodebench", "threads"):
        if key not in config:
            raise ValueError(f"{path}: missing {key}")
    missing = [key for key in LCB_KEYS if key not in config["livecodebench"]]
    if missing:
        raise ValueError(f"{path}: livecodebench lacks {missing}")
    if config["samples"] < 16:
        raise ValueError("Use at least 16 samples so pass@16 and the sometimes-solved split are defined")
    return config


def snapshot(model: dict) -> Path:
    org, name = model["id"].split("/")
    return Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "snapshots" / str(model["revision"])


def build_commands(config: dict) -> list[dict]:
    """Ordered steps. `network` downloads, `data` prepares problems, `gpu` generates, `sandbox` executes samples,
    `summary` reads results. A step is skipped when its `creates` path exists: always for `data`, and with --resume
    for the rest. EvalPlus codegen writes samples as it goes, so it counts as done only once they are evaluated."""
    out, n, t, python = Path(config["output"]), str(config["samples"]), str(config["temperature"]), sys.executable
    hf = str(Path(python).parent / "hf")  # the Hub CLI of the same environment
    lcb = config["livecodebench"]
    raw, checkout, problems = (str(Path(lcb[key]).expanduser()) for key in ("raw", "checkout", "problems"))
    steps = [
        {"name": "download livecodebench", "kind": "network",
         "argv": [hf, "download", lcb["dataset"], *RELEASE_FILES[lcb["release_version"]], "--repo-type", "dataset",
                  "--revision", lcb["dataset_revision"], "--local-dir", raw]},
        {"name": f"livecodebench problems before {lcb['before']}", "kind": "data", "creates": problems,
         "argv": [python, "-m", "scripts.codeprobe.livecodebench_data", "--raw", raw, "--release",
                  lcb["release_version"], "--before", lcb["before"], "--output", problems]},
    ]
    evalplus_results, lcb_results = [], []
    for model in config["models"]:
        revision = ["--revision", model["revision"]] if model["revision"] else []
        steps.append({"name": f"download {model['name']}", "kind": "network",
                      "argv": [hf, "download", model["id"], *revision]})
        path = str(snapshot(model))
        for dataset in config["evalplus_datasets"]:
            for label, flags in [("greedy", ["--greedy"]), (f"n{n}", ["--n_samples", n, "--temperature", t, "--bs", n])]:
                root = out / "evalplus" / model["name"] / dataset / label
                results = str(root / "eval_results.json")
                steps.append({"name": f"evalplus codegen {dataset} {label} {model['name']}", "kind": "gpu",
                              "creates": results, "argv": [python, "-m", "scripts.codeprobe.codegen", "--model", path, "--dataset", dataset,
                                       "--backend", "vllm", *flags, "--root", str(root)]})
                steps.append({"name": f"evalplus evaluate {dataset} {label} {model['name']}", "kind": "sandbox",
                              "creates": results, "argv": [python, "-m", "scripts.codeprobe.evaluate_sandboxed", "--dataset", dataset,
                                       "--samples", f"{root}/{dataset}/*_temp_*[0-9].jsonl",
                                       "--output", results, "--parallel", "2", "--memory-gb", "10"]})
                evalplus_results.append(f"{model['name']}/{dataset}-{label}={results}")
        for label, flags in [("greedy", ["--n", "1", "--temperature", "0"]), (f"n{n}", ["--n", n, "--temperature", t])]:
            root = out / "livecodebench" / model["name"] / label
            steps.append({"name": f"livecodebench codegen {label} {model['name']}", "kind": "gpu",
                          "creates": str(root / "generations.json"), "argv": [python, "-m", "scripts.codeprobe.livecodebench_codegen", "--lcb", checkout,
                                   "--problems", problems, "--output", str(root / "generations.json"), "--",
                                   "--model", model["id"], "--local_model_path", path,
                                   "--release_version", lcb["release_version"], *flags]})
            steps.append({"name": f"livecodebench evaluate {label} {model['name']}", "kind": "sandbox",
                          "creates": str(root / "eval_all.json"), "argv": [python, "-m", "scripts.codeprobe.evaluate_livecodebench", "--lcb", checkout,
                                   "--problems", problems, "--generations", str(root / "generations.json"),
                                   "--output", str(root / "eval_all.json")]})
            lcb_results.append(f"{model['name']}/livecodebench-{label}={root / 'eval_all.json'}")
    summary = [python, "-m", "scripts.codeprobe.headroom", *(f"--evalplus={spec}" for spec in evalplus_results),
               *(f"--livecodebench={spec}" for spec in lcb_results)]
    window = ":".join(lcb["published_window"])
    steps.append({"name": "headroom summary", "kind": "summary", "creates": str(out / "summary.json"),
                  "argv": [*summary, "--output", str(out / "summary.json")]})
    steps.append({"name": f"headroom summary, LiveCodeBench {window}", "kind": "summary",
                  "creates": str(out / "summary-livecodebench-published-window.json"), "argv": [python, "-m", "scripts.codeprobe.headroom",
                           *(f"--livecodebench={spec}" for spec in lcb_results), "--dates", window,
                           "--output", str(out / "summary-livecodebench-published-window.json")]})
    return steps


def resolve(argument: str) -> str:
    """Expand the samples glob, which names codegen's output file, once that file exists."""
    if "*" not in argument:
        return argument
    matches = sorted(Path("/").glob(argument.lstrip("/")) if argument.startswith("/") else Path().glob(argument))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one file matching {argument}, found {len(matches)}")
    return str(matches[0])


def check_checkout(lcb: dict) -> None:
    head = subprocess.run(["git", "-C", str(Path(lcb["checkout"]).expanduser()), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != lcb["commit"]:
        raise SystemExit(f"LiveCodeBench checkout is at {head}, config pins {lcb['commit']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/probe_headroom.json"))
    parser.add_argument("--run", action="store_true", help="execute; default prints the commands only")
    parser.add_argument("--resume", action="store_true", help="with --run: continue an existing output directory")
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume needs --run")
    config = load_config(args.config)
    steps = build_commands(config)
    if args.run:
        unpinned = [m["id"] for m in config["models"] if not m["revision"]]
        if unpinned:
            parser.error(f"Pin model revisions in {args.config} before running: {unpinned}")
        if Path(config["output"]).exists() != args.resume:
            parser.error(f"{config['output']} exists; choose a new output or pass --resume" if not args.resume
                         else f"--resume: {config['output']} does not exist")
        check_checkout(config["livecodebench"])
    threads = str(config["threads"])
    environment = {**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                   "VLLM_USE_FLASHINFER_SAMPLER": "0"}  # its JIT needs nvcc, which the desktop lacks
    for step in steps:
        print(f"# {step['name']} [{step['kind']}]\n{' '.join(step['argv'])}", flush=True)
        if not args.run:
            continue
        if "creates" in step and (step["kind"] == "data" or args.resume) and Path(step["creates"]).exists():
            print(f"# skipped: {step['creates']} exists", flush=True)
            continue
        offline = {} if step["kind"] == "network" else {"HF_HUB_OFFLINE": "1"}
        subprocess.run([resolve(argument) for argument in step["argv"]], check=True, env={**environment, **offline},
                       preexec_fn=lambda: os.nice(10))  # the desktop is shared


if __name__ == "__main__":
    main()

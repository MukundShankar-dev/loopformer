"""Build, print or run the headroom-probe commands from configs/probe_headroom.json.

Generation uses the official tools on the host (it executes no model code):
EvalPlus codegen through scripts/codeprobe/codegen.py, which caps vLLM's GPU
share, and LiveCodeBench's runner without `--evaluate`. EvalPlus evaluation,
which executes the samples, runs inside the sandbox
(scripts/codeprobe/evaluate_sandboxed.py). LiveCodeBench evaluation also
executes code and has no sandboxed wrapper yet, so it is not among the steps.
Models load from pinned local snapshots. The LiveCodeBench flags follow its
documentation and are unverified here; check the printed commands first.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    for key in ("output", "models", "samples", "temperature", "evalplus_datasets", "livecodebench", "threads"):
        if key not in config:
            raise ValueError(f"{path}: missing {key}")
    if config["samples"] < 16:
        raise ValueError("Use at least 16 samples so pass@16 and the sometimes-solved split are defined")
    return config


def snapshot(model: dict) -> Path:
    org, name = model["id"].split("/")
    return Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "snapshots" / str(model["revision"])


def build_commands(config: dict) -> list[dict]:
    """Ordered steps; `network` steps download, `gpu` steps generate, the rest execute samples in the sandbox."""
    out, n, t = Path(config["output"]), str(config["samples"]), str(config["temperature"])
    steps = []
    for model in config["models"]:
        revision = ["--revision", model["revision"]] if model["revision"] else []
        steps.append({"name": f"download {model['name']}", "kind": "network",
                      "argv": ["hf", "download", model["id"], *revision]})
        path = str(snapshot(model))
        for dataset in config["evalplus_datasets"]:
            for label, flags in [("greedy", ["--greedy"]), (f"n{n}", ["--n_samples", n, "--temperature", t, "--bs", n])]:
                root = out / "evalplus" / model["name"] / dataset / label
                steps.append({"name": f"evalplus codegen {dataset} {label} {model['name']}", "kind": "gpu",
                              "argv": ["python", "-m", "scripts.codeprobe.codegen", "--model", path, "--dataset", dataset,
                                       "--backend", "vllm", *flags, "--root", str(root)]})
                steps.append({"name": f"evalplus evaluate {dataset} {label} {model['name']}", "kind": "sandbox",
                              "argv": ["python", "-m", "scripts.codeprobe.evaluate_sandboxed", "--dataset", dataset,
                                       "--samples", f"{root}/{dataset}/*_temp_*[0-9].jsonl",
                                       "--output", str(root / "eval_results.json"), "--parallel", "2", "--memory-gb", "10"]})
        steps.append({"name": f"livecodebench codegen n={n} {model['name']}", "kind": "gpu",
                      "argv": ["python", "-m", "lcb_runner.runner.main", "--model", model["id"], "--local_model_path", path,
                               "--scenario", "codegeneration", "--release_version", config["livecodebench"]["release_version"],
                               "--n", n, "--temperature", t]})
    return steps


def resolve(argument: str) -> str:
    """Expand the samples glob, which names codegen's output file, once that file exists."""
    if "*" not in argument:
        return argument
    matches = sorted(Path("/").glob(argument.lstrip("/")) if argument.startswith("/") else Path().glob(argument))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one file matching {argument}, found {len(matches)}")
    return str(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/probe_headroom.json"))
    parser.add_argument("--run", action="store_true", help="execute; default prints the commands only")
    args = parser.parse_args()
    config = load_config(args.config)
    steps = build_commands(config)
    if args.run:
        unpinned = [m["id"] for m in config["models"] if not m["revision"]]
        if unpinned:
            parser.error(f"Pin model revisions in {args.config} before running: {unpinned}")
        if Path(config["output"]).exists():
            parser.error(f"{config['output']} exists; choose a new output")
    threads = str(config["threads"])
    environment = {**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                   "VLLM_USE_FLASHINFER_SAMPLER": "0"}  # its JIT needs nvcc, which the desktop lacks
    for step in steps:
        print(f"# {step['name']} [{step['kind']}]\n{' '.join(step['argv'])}")
        if args.run:
            offline = {} if step["kind"] == "network" else {"HF_HUB_OFFLINE": "1"}
            subprocess.run([resolve(argument) for argument in step["argv"]], check=True, env={**environment, **offline})


if __name__ == "__main__":
    main()

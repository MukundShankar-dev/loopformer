"""Build, print or run the headroom-probe commands from configs/probe_headroom.json.

Generation and grading use the official EvalPlus and LiveCodeBench tools, so
base-model numbers are comparable with published ones. Model weights are
downloaded first; tool runs then use HF_HUB_OFFLINE=1 inside `sandbox_prefix`
(for example a no-network container), because both tools execute model code.

The tool flags follow their documentation as of writing and are unverified
here; check `--dry-run` output against the installed versions before `--run`.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    for key in ("output", "models", "samples", "temperature", "evalplus_datasets", "livecodebench", "sandbox_prefix", "threads"):
        if key not in config:
            raise ValueError(f"{path}: missing {key}")
    if config["samples"] < 16:
        raise ValueError("Use at least 16 samples so pass@16 and the sometimes-solved split are defined")
    return config


def build_commands(config: dict) -> list[dict]:
    """Ordered steps: download (network), then sampled and greedy runs (sandboxed)."""
    out, n, t = Path(config["output"]), str(config["samples"]), str(config["temperature"])
    lcb = config["livecodebench"]
    steps = []
    for model in config["models"]:
        revision = ["--revision", model["revision"]] if model["revision"] else []
        steps.append({"name": f"download {model['name']}", "sandbox": False,
                      "argv": ["hf", "download", model["id"], *revision]})
        for dataset in config["evalplus_datasets"]:
            root = str(out / "evalplus" / model["name"])
            base = ["evalplus.evaluate", "--model", model["id"], "--dataset", dataset, "--backend", "hf", "--root", root]
            steps.append({"name": f"evalplus {dataset} greedy {model['name']}", "sandbox": True, "argv": [*base, "--greedy"]})
            steps.append({"name": f"evalplus {dataset} n={n} {model['name']}", "sandbox": True,
                          "argv": [*base, "--n_samples", n, "--temperature", t]})
        steps.append({"name": f"livecodebench n={n} {model['name']}", "sandbox": True,
                      "argv": ["python", "-m", "lcb_runner.runner.main", "--model", model["id"], "--scenario", "codegeneration",
                               "--evaluate", "--release_version", lcb["release_version"], "--n", n, "--temperature", t]})
    return steps


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
        if not config["sandbox_prefix"]:
            parser.error("Set sandbox_prefix: both tools execute model-generated code")
        if Path(config["output"]).exists():
            parser.error(f"{config['output']} exists; choose a new output")
    env = {**os.environ, "OMP_NUM_THREADS": str(config["threads"]), "MKL_NUM_THREADS": str(config["threads"])}
    for step in steps:
        argv = [*config["sandbox_prefix"], *step["argv"]] if step["sandbox"] else step["argv"]
        print(f"# {step['name']}\n{' '.join(argv)}")
        if args.run:
            subprocess.run(argv, check=True, env={**env, **({"HF_HUB_OFFLINE": "1"} if step["sandbox"] else {})})


if __name__ == "__main__":
    main()

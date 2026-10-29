"""Run the frozen, matched original-Qwen / one-loop recurrent MMLU comparison."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from rich.console import Console
from transformers import AutoTokenizer

from scripts.eval.loading import load_model
from scripts.eval.mmlu_backend import SingleTokenLM
from scripts.eval.paper_baseline import atomic_json
from scripts.eval.pointer_task import sha256_file
from scripts.eval.single_token import encode_pair, last_logits, padded_inputs
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.model import RecurrentQwen


def task_manager(root: Path, config: dict) -> TaskManager:
    """Copy standard harness YAMLs and pin only their dataset revision."""
    destination = root / "task_configs"
    if not destination.exists():
        source = Path(lm_eval.__file__).parent / "tasks/mmlu/default"
        shutil.copytree(source, destination)
        template = destination / "_default_template_yaml"
        template.write_text(template.read_text() + "\ndataset_kwargs:\n  revision: " + config["dataset_revision"] + "\n")
    return TaskManager(include_defaults=False, include_path=str(destination))


def preflight(model, tokenizer, manager: TaskManager, config: dict, directory: Path) -> dict:
    """Check wrapper equivalence and adapter likelihoods against standard HFLM."""
    prepared = manager.load([config["task"]])
    tasks = prepared["tasks"]
    counts = {name: len(task.test_docs()) for name, task in tasks.items()}
    if len(counts) != config["expected_subjects"] or sum(counts.values()) != config["expected_questions"]:
        raise ValueError("MMLU subject/question counts differ from the frozen protocol")
    requests = []
    for name, task in list(sorted(tasks.items()))[:3]:
        task.set_fewshot_seed(config["seed"])
        doc = task.test_docs()[0]
        context = task.fewshot_context(doc, num_fewshot=config["num_fewshot"])
        requests.extend(task.construct_requests(doc, context, metadata=(name, 0, 1)))
    adapter = SingleTokenLM(model, tokenizer, recurrent=False, batch_size=config["batch_size"])
    compact = adapter.loglikelihood(requests)
    standard = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, backend="causal", max_length=model.config.max_position_embeddings)
    reference = standard.loglikelihood(requests)
    differences = [abs(a[0] - b[0]) for a, b in zip(compact, reference, strict=True)]
    if max(differences) > 1e-4 or any(a[1] != b[1] for a, b in zip(compact, reference, strict=True)):
        raise ValueError("Compact likelihood adapter differs from the standard HF harness")
    predictions = [max(range(4), key=lambda i: compact[start + i][0]) for start in range(0, len(compact), 4)]
    reference_predictions = [max(range(4), key=lambda i: reference[start + i][0]) for start in range(0, len(reference), 4)]
    if predictions != reference_predictions:
        raise ValueError("Choice decisions differ from the standard HF harness")
    contexts = [encode_pair(tokenizer, *request.args)[0] for request in requests[::4]]
    ids, mask = padded_inputs(contexts, pad_token_id=tokenizer.pad_token_id, device=next(model.parameters()).device)
    with torch.inference_mode():
        original = last_logits(model, ids, mask, recurrent=False)
        fresh = RecurrentQwen(model, recurrence_mode="fixed_prompt").eval()
        wrapped = last_logits(fresh, ids, mask, recurrent=True)
        torch.testing.assert_close(wrapped, original, atol=1e-5, rtol=1e-5)
    result = {"passed": True, "subjects": len(counts), "questions": sum(counts.values()),
              "subject_counts": counts, "reference_requests": len(requests),
              "max_likelihood_difference": max(differences),
              "choice_predictions": predictions, "reference_choice_predictions": reference_predictions,
              "fresh_one_loop_max_logit_difference": float((wrapped - original).abs().max()),
              "dataset_fingerprints": {name: task.dataset["test"]._fingerprint for name, task in tasks.items()}}
    atomic_json(directory / "preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mmlu_retention.json"))
    parser.add_argument("--arm", choices=("base", "recurrent"), required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != config:
        raise ValueError("Existing run has another frozen protocol")
    atomic_json(protocol_path, config)
    if importlib.metadata.version("lm_eval") != "0.4.13":
        raise ValueError("Use the pinned lm_eval 0.4.13 harness")
    if config["num_loops"] != 1 or config["dtype"] != "float32" or config["attention"] != "sdpa":
        raise ValueError("This comparison requires exactly one loop, FP32, and SDPA")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    checkpoint = Path(config["checkpoint"])
    if sha256_file(checkpoint / "adapter_model.pt") != config["checkpoint_sha256"]:
        raise ValueError("Final recurrent checkpoint weight hash differs")
    manager = task_manager(root, config)
    directory = root / ("preflight" if args.preflight else args.arm)
    directory.mkdir(exist_ok=True)
    if args.preflight and args.arm != "base":
        parser.error("Preflight uses the independent base checkpoint")
    if args.arm == "base":
        model, tokenizer, _ = load_model(config["base_model"], device="cuda", dtype="float32",
                                         revision=config["base_revision"], tokenizer_source=str(checkpoint),
                                         base_model=None, download=False)
    else:
        model, tokenizer, spec = load_recurrent_checkpoint(checkpoint, device="cuda", dtype="float32", download=False)
        if spec["base_model"] != config["base_model"] or spec["revision"] != config["base_revision"]:
            raise ValueError("Recurrent and base initialization identities differ")
        model.router = None
        model.completion_head = None
    model.config._attn_implementation = "sdpa"
    model.requires_grad_(False)
    original_tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"], local_files_only=True)
    if tokenizer.get_vocab() != original_tokenizer.get_vocab():
        raise ValueError("Checkpoint tokenizer vocabulary differs from the pinned base")
    tokenization = json.loads(tokenizer.backend_tokenizer.to_str())
    original_tokenization = json.loads(original_tokenizer.backend_tokenizer.to_str())
    for key in ("padding", "truncation"):
        tokenization.pop(key, None)
        original_tokenization.pop(key, None)
    if tokenization != original_tokenization or tokenizer.eos_token_id != original_tokenizer.eos_token_id:
        raise ValueError("Checkpoint tokenization rules differ from the pinned base")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
    tokenizer.padding_side = "left"
    if args.preflight:
        Console().print(preflight(model, tokenizer, manager, config, directory))
        return
    if not json.loads((root / "preflight/preflight.json").read_text())["passed"]:
        raise ValueError("Complete the reference/equivalence preflight first")
    identity = {"protocol": config, "arm": args.arm, "checkpoint_sha256": config["checkpoint_sha256"],
                "source_sha256": {str(p): sha256_file(p) for p in sorted(Path("scripts/recurrent_qwen").glob("*.py"))},
                "evaluation_sha256": {str(p): sha256_file(p) for p in [Path(__file__), Path("scripts/eval/mmlu_backend.py"), Path("scripts/eval/single_token.py")]},
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "tokenizers", "lm_eval", "datasets", "numpy", "huggingface-hub"]},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "worktree": subprocess.check_output(["git", "status", "--short"], text=True),
                "device": torch.cuda.get_device_name(), "router_and_timer_bypassed": args.arm == "recurrent"}
    freeze_path = directory / "freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != identity:
        raise ValueError("Cached run identity changed; use a new run directory")
    atomic_json(freeze_path, identity)
    backend = SingleTokenLM(model, tokenizer, recurrent=args.arm == "recurrent", batch_size=config["batch_size"],
                            progress_path=directory / "progress.json")
    started = perf_counter()
    results = lm_eval.simple_evaluate(model=backend, tasks=[config["task"]], task_manager=manager,
              num_fewshot=config["num_fewshot"], apply_chat_template=config["apply_chat_template"],
              use_cache=str(directory / "likelihood_cache"), log_samples=True, bootstrap_iters=1000,
              random_seed=config["seed"], numpy_random_seed=config["seed"],
              torch_random_seed=config["seed"], fewshot_random_seed=config["seed"])
    samples = results.pop("samples")
    for name, rows in samples.items():
        path = directory / (name + ".jsonl")
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        temporary.replace(path)
    atomic_json(directory / "results.json", results)
    atomic_json(directory / "progress.json", {"status": "complete", "seconds": perf_counter() - started,
                 "questions": sum(len(rows) for rows in samples.values()), "subjects": len(samples),
                 "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "timing_scope": "harness dataset/task preparation, likelihood evaluation and raw-sample export; excludes model loading"})
    Console().print(results["groups"]["mmlu"])


if __name__ == "__main__":
    main()

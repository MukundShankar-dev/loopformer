"""Inventory saved evidence without loading models or evaluating reserved data."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console


SPAN_FIELDS = {"step", "depth", "steps", "task_depth", "loop", "loop_index", "requested_steps"}
ID_FIELDS = {"example_id", "graph_id", "graph_index"}
BINARY_SUFFIXES = {".pt", ".bin", ".safetensors", ".png", ".pdf", ".svg", ".npy", ".npz"}
METADATA_NAMES = {"summary.json", "config.json", "run.json", "manifest.json", "freeze.json",
                  "best_checkpoint.json", "last_checkpoint.json", "recurrent_config.json",
                  "saved_arm_reuse.json", "executor_identity.json"}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def table_profile(path: Path) -> dict[str, Any]:
    """Read actual CSV rows, including compressed files; IDs are counted per file."""
    opener = gzip.open if path.name.endswith(".gz") else open
    spans: dict[str, list[float]] = {}
    identifiers: dict[str, set[str]] = {}
    rows = 0
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for column in ID_FIELDS.intersection(columns):
            identifiers[column] = set()
        for row in reader:
            rows += 1
            for column in SPAN_FIELDS.intersection(columns):
                value = row[column]
                if value in (None, ""):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue  # E.g. the descriptive "overall" row, not a depth.
                span = spans.setdefault(column, [number, number])
                span[0], span[1] = min(span[0], number), max(span[1], number)
            for column, values in identifiers.items():
                if row[column] not in (None, ""):
                    values.add(row[column])
    return {"columns": columns, "rows": rows, "numeric_spans": spans,
            "distinct_ids": {k: len(v) for k, v in identifiers.items()}}


def jsonl_profile(path: Path) -> dict[str, Any]:
    """Inspect recorded schemas/events, without copying records into the inventory."""
    keys: set[str] = set()
    events: Counter[str] = Counter()
    spans: dict[str, list[float]] = {}
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            rows += 1
            if not isinstance(value, dict):
                continue
            keys.update(value)
            event = value.get("event", value.get("phase", value.get("type")))
            if isinstance(event, str):
                events[event] += 1
            for key in SPAN_FIELDS.intersection(value):
                number = value[key]
                if isinstance(number, (int, float)) and not isinstance(number, bool):
                    span = spans.setdefault(key, [number, number])
                    span[0], span[1] = min(span[0], number), max(span[1], number)
    return {"keys": sorted(keys), "rows": rows, "numeric_spans": spans, "events": dict(events)}


def protected_dataset(path: Path, root: Path) -> bool:
    """Do not read unopened test records, or any seed-29 dataset contents."""
    relative = path.relative_to(root)
    if relative.parts[0] != "data":
        return False
    if any(part.startswith("seed-29") for part in relative.parts):
        return True
    return path.name == "test.jsonl" and "seed-61-independent" not in relative.parts


def inventory(root: Path, label: str) -> dict[str, Any]:
    """Capture availability, schemas, row coverage and small provenance metadata."""
    files: list[dict[str, Any]] = []
    groups: dict[str, dict[str, int]] = {}
    for folder in ("data", "models", "eval"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_symlink():
                files.append({"path": str(path.relative_to(root)), "symlink": True})
                continue
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            # Generated inventory evidence is excluded so reruns do not inventory themselves.
            if str(relative).startswith("eval/pointer_analysis/"):
                continue
            record: dict[str, Any] = {"path": str(relative), "bytes": path.stat().st_size}
            group = str(Path(*relative.parts[:3]))
            total = groups.setdefault(group, {"files": 0, "bytes": 0, "binary_bytes": 0})
            total["files"] += 1
            total["bytes"] += record["bytes"]
            if protected_dataset(path, root):
                record["content_inspected"] = False
                record["reason"] = "Reserved dataset: presence/size only"
            elif path.name.endswith((".csv", ".csv.gz")):
                record["table"] = table_profile(path)
            elif path.suffix == ".jsonl":
                record["records"] = jsonl_profile(path)
            elif path.suffix == ".json" and path.name not in {"tokenizer.json", "tokenizer_config.json"}:
                data = json.loads(path.read_text())
                record["json_keys"] = sorted(data) if isinstance(data, dict) else []
                if path.name in METADATA_NAMES:
                    record["metadata"] = data
            if path.suffix in BINARY_SUFFIXES:
                total["binary_bytes"] += record["bytes"]
                record["binary_contents_loaded"] = False
            elif not protected_dataset(path, root):
                with path.open("rb") as handle:
                    record["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
            files.append(record)
    checkpoints = []
    for record in files:
        if Path(record["path"]).name != "recurrent_config.json":
            continue
        directory = str(Path(record["path"]).parent)
        artifacts = [f for f in files if str(Path(f["path"]).parent) == directory]
        checkpoints.append({"path": directory, "config": record.get("metadata"),
                            "inference_weights_present": (root / directory / "adapter_model.pt").is_file(),
                            "weight_files": [f["path"] for f in artifacts
                                             if Path(f["path"]).suffix in {".pt", ".bin", ".safetensors"}]})
    configured = {checkpoint["path"] for checkpoint in checkpoints}
    orphan_weights = sorted(str(Path(record["path"]).parent) for record in files
                            if Path(record["path"]).name == "adapter_model.pt"
                            and str(Path(record["path"]).parent) not in configured)
    return {"schema_version": 1, "label": label, "root": str(root),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_output(root, "rev-parse", "HEAD"),
            "git_status": git_output(root, "status", "--short"),
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "All present data/models/eval files; closed datasets presence only; no model execution",
            "groups": groups, "checkpoints": checkpoints,
            "weight_directories_without_config": orphan_weights, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh inventory path")
    console = Console()
    console.print(f"[cyan]Reading saved evidence · {args.label} · no inference[/cyan]")
    with console.status("Inspecting artifact schemas and coverage…"):
        result = inventory(args.root.resolve(), args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if args.output.suffix == ".gz":
        args.output.write_bytes(gzip.compress(payload, mtime=0))
    else:
        args.output.write_bytes(payload)
    console.print(f"[green]{len(result['files']):,} files · {len(result['checkpoints'])} checkpoint configurations[/green]")
    console.print(str(args.output.resolve()))


if __name__ == "__main__":
    main()

"""Freeze the LiveCodeBench confirmation set as a hashed list of problem IDs.

    .venv-probe/bin/python -m scripts.codetasks.confirmation --raw ~/src/lcb-data --cutoff 2024-11-12 \\
        --output configs/confirmation_livecodebench.json

Confirmation is every problem dated on or after the cutoff (the base model's
public release), so the model cannot have seen it. This reads only each
problem's ID, date, platform and difficulty; hidden tests are not decoded. The
SHA-256 of the sorted IDs is recorded so later changes to the set are visible.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rich.console import Console

from scripts.codeprobe.livecodebench_data import RELEASE_FILES


def id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="directory with the release's JSONL files")
    parser.add_argument("--release", choices=sorted(RELEASE_FILES), default="release_v6")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD; problems on or after it are confirmation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; the confirmation set is written once")
    rows = []
    for name in RELEASE_FILES[args.release]:
        for line in (args.raw / name).open():
            record = json.loads(line)
            if record["contest_date"][:10] >= args.cutoff:
                rows.append({key: record[key] for key in ("question_id", "contest_date", "platform", "difficulty")})
    ids = sorted(row["question_id"] for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question IDs in the confirmation set")
    dates = sorted(row["contest_date"][:10] for row in rows)
    args.output.write_text(json.dumps({
        "dataset": "livecodebench/code_generation_lite", "release": args.release,
        "dataset_revision": args.dataset_revision, "cutoff": args.cutoff, "first_date": dates[0],
        "last_date": dates[-1], "difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "platform": dict(sorted(Counter(row["platform"] for row in rows).items())),
        "sha256": id_hash(ids), "question_ids": ids}, indent=2) + "\n")
    Console().print(f"{len(ids)} confirmation problems from {dates[0]} to {dates[-1]} written to {args.output}")


if __name__ == "__main__":
    main()

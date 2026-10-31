"""Split a downloaded LiveCodeBench release into per-problem files, development problems only.

    python -m scripts.codeprobe.livecodebench_data --raw ~/src/lcb-data --before 2024-11-12 \\
        --output data/livecodebench/release_v6-0fe84c3-before-2024-11-12

LiveCodeBench's own loader runs the dataset's remote loading script and
unpickles the hidden tests, either of which can execute code from the
download. This reads the raw JSONL files instead and decodes the hidden tests
with an unpickler that refuses every class, so decoding can only build plain
strings. Problems dated on or after --before are not written: the confirmation
set comes from after the cutoff, which is no earlier than the base model's
release, so nothing on or after that date may be decoded or graded before
confirmation.

Output: `problems.jsonl` (LiveCodeBench's fields without the hidden tests,
which codegen does not need) and `tests/<question_id>.json` (the hidden tests
as LiveCodeBench's JSON text).
"""
import argparse
import base64
import io
import json
import pickle
import re
import zlib
from pathlib import Path

from rich.console import Console

RELEASE_FILES = {"release_v6": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")}
FIELDS = ("question_title", "question_content", "platform", "question_id", "contest_id", "contest_date",
          "starter_code", "difficulty", "public_test_cases", "private_test_cases", "metadata")
SAFE_ID = re.compile(r"\A[A-Za-z0-9_-]+\Z")


class _NoClasses(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def decode_private_tests(text: str) -> str:
    """The hidden tests as JSON text: plain JSON, or base64(zlib(pickle(JSON text))) as LiveCodeBench stores them."""
    try:
        json.loads(text)
        return text
    except ValueError:
        pass
    value = _NoClasses(io.BytesIO(zlib.decompress(base64.b64decode(text)))).load()
    if not isinstance(value, str):
        raise ValueError(f"expected pickled JSON text, got {type(value).__name__}")
    json.loads(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="directory with the release's JSONL files")
    parser.add_argument("--release", choices=sorted(RELEASE_FILES), default="release_v6")
    parser.add_argument("--before", required=True, help="keep problems dated before this YYYY-MM-DD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new output")
    (args.output / "tests").mkdir(parents=True)
    kept = skipped = 0
    with (args.output / "problems.jsonl").open("w") as problems:
        for name in RELEASE_FILES[args.release]:
            for line in (args.raw / name).open():
                record = json.loads(line)
                if set(record) != set(FIELDS):
                    raise ValueError(f"{name}: unexpected fields {sorted(set(record) ^ set(FIELDS))}")
                if record["contest_date"][:10] >= args.before:
                    skipped += 1
                    continue
                if not SAFE_ID.match(record["question_id"]):
                    raise ValueError(f"{name}: question_id {record['question_id']!r} is not a safe file name")
                path = args.output / "tests" / f"{record['question_id']}.json"
                if path.exists():
                    raise ValueError(f"{name}: duplicate question_id {record['question_id']!r}")
                path.write_text(decode_private_tests(record.pop("private_test_cases")))
                problems.write(json.dumps(record) + "\n")
                kept += 1
    Console().print(f"{kept} problems before {args.before} written to {args.output}; {skipped} later ones not decoded")


if __name__ == "__main__":
    main()

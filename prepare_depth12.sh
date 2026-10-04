#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo 'Usage: bash prepare_depth12.sh [--dry-run]' >&2
  exit 2
fi
output=data/pointer/seed-47-depth12-gaps
if [[ $# == 0 && -e "$output" ]]; then
  python -m scripts.dataset --verify "$output"
  python - "$output" <<'CHECK'
import json
import sys
from pathlib import Path
config = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())["config"]
expected = {"seed": 47, "train_count": 30000, "validation_count": 1500,
            "test_count": 1500, "depth_test_count": 1000, "min_depth": 1,
            "max_train_depth": 12, "max_eval_depth": 20,
            "train_depths": [1, 2, 3, 4, 5, 6, 8, 10, 12]}
if config != expected:
    raise SystemExit("Existing dataset is valid but does not match the depth-12 recipe")
CHECK
  exit 0
fi
python -m scripts.dataset --seed 47 --train-count 30000 \
  --validation-count 1500 --test-count 1500 --depth-test-count 1000 \
  --max-train-depth 12 --max-eval-depth 20 \
  --train-depths 1 2 3 4 5 6 8 10 12 --output "$output" "$@"

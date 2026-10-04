#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing virtual environment: $PWD/.venv" >&2
  exit 1
fi
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash eval_ckpts.sh [completion-training-run-directory]' >&2
  exit 2
fi

run_dir="${1:-models/stage1_pointer/depth6-completion-seed37}"
if [[ ! -f "$run_dir/best_checkpoint.json" ]]; then
  echo "Missing completed training run: $run_dir/best_checkpoint.json" >&2
  exit 1
fi

checkpoint="$(python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]).resolve()
name = json.loads((run / "best_checkpoint.json").read_text())["path"]
checkpoint = (run / name).resolve()
if checkpoint.parent != run or not checkpoint.is_dir():
    raise SystemExit("best_checkpoint.json must name a step directory inside the run")
print(checkpoint)
PY
)"

for file in recurrent_config.json adapter_model.pt tokenizer.json; do
  if [[ ! -f "$checkpoint/$file" ]]; then
    echo "Missing checkpoint file: $checkpoint/$file" >&2
    exit 1
  fi
done
python - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

spec = json.loads((Path(sys.argv[1]) / "recurrent_config.json").read_text())
if spec.get("format") != "loopformer-stage1-completion-v1" or "completion_head" not in spec:
    raise SystemExit("Selected checkpoint has no trained completion head")
PY
for split in validation depth_test; do
  if [[ ! -f "data/pointer/seed-17/$split.jsonl" ]]; then
    echo "Missing dataset: data/pointer/seed-17/$split.jsonl" >&2
    exit 1
  fi
done

output="eval/pointer_loops/completion-best-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$output"

run_evals() {
  echo "Checkpoint: $checkpoint"
  echo "Output: $output"
  for split in validation depth_test; do
    python -u -m scripts.eval.loop_test \
      --model "$checkpoint" \
      --data "data/pointer/seed-17/$split.jsonl" \
      --device cuda --batch-size 16 --loops 20 \
      --output "$output/$split" || return $?
  done
  python - "$output" <<'PY' || return $?
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for split in ("validation", "depth_test"):
    summary = json.loads((root / split / "summary.json").read_text())
    stop = summary["completion"]
    print(f"{split}: trajectory {summary['trajectory_accuracy']:.1%}; "
          f"final {summary['final_accuracy']:.1%}; "
          f"stop exact {stop['first_stop_exact_rate']:.1%}; "
          f"early {stop['first_stop_early_rate']:.1%}; "
          f"late/missing {stop['first_stop_late_or_missing_rate']:.1%}")
PY
}

run_evals 2>&1 | tee "$output/run.log"

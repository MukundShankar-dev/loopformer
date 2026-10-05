#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 ]]; then echo 'Usage: bash eval_controller.sh [training-run-directory]' >&2; exit 2; fi
source .venv/bin/activate
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-loopformer}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run="${1:-models/stage1_pointer/controller-seed61}"
checkpoint="$(python - "$run" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]).resolve()
checkpoint = (run / json.loads((run / 'best_checkpoint.json').read_text())['path']).resolve()
if checkpoint.parent != run or not (checkpoint / 'adapter_model.pt').is_file():
    raise SystemExit('Need the complete selected controller/executor checkpoint')
print(checkpoint)
PY
)"
output="eval/pointer_diagnostics/controller-eval-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$output"
run_evals() {
  for split in validation depth_test; do
    python -u -m scripts.eval.loop_test --model "$checkpoint" --device cuda \
      --data "data/pointer/seed-61-independent/$split.jsonl" --loops 64 \
      --stop-policy completion --stop-threshold 0.5 --attention sdpa \
      --output "$output/$split-stopped" || return $?
  done
  python -u -m scripts.eval.controller_diagnostic --model "$checkpoint" --device cuda \
    --output "$output/diagnostic" || return $?
}
run_evals 2>&1 | tee "$output/run.log"

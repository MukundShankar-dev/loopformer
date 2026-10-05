#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 ]]; then echo 'Usage: bash diagnose_controller.sh [training-run-directory]' >&2; exit 2; fi
source .venv/bin/activate
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-loopformer}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run="${1:-models/stage1_pointer/executor_r-seed61}"
checkpoint="$(python - "$run" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]).resolve()
checkpoint = (run / json.loads((run / 'best_checkpoint.json').read_text())['path']).resolve()
if checkpoint.parent != run or not (checkpoint / 'adapter_model.pt').is_file():
    raise SystemExit('Need the complete selected executor checkpoint on this device')
print(checkpoint)
PY
)"
output="eval/pointer_diagnostics/controller-$(basename "$run")-$(date -u +%Y%m%dT%H%M%SZ)-$$"
# Progress bars stay interactive; tee retains only bounded noninteractive output.
# Keep the log next to the output because the CLI creates its directory exclusively.
mkdir -p -- "$(dirname -- "$output")"
python -u -m scripts.eval.controller_diagnostic --model "$checkpoint" --device cuda \
  --output "$output" 2>&1 | tee "${output}.log"

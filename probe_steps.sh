#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo 'Usage: bash probe_steps.sh [--dry-run]' >&2
  exit 2
fi
checkpoint=models/stage1_pointer/depth6-fixed-prompt-seed37/step-003250
if [[ ${1:-} == --dry-run ]]; then
  exec python -m scripts.eval.paired_steps --model "$checkpoint" --dry-run
fi
output="eval/pointer_probes/paired-steps-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p eval/pointer_probes
python -u -m scripts.eval.paired_steps --model "$checkpoint" --output "$output" 2>&1 | tee "${output}.log"

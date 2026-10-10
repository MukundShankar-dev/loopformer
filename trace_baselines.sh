#!/usr/bin/env bash
# One ordinary SFT run, two trace baselines, independent audits and new figures.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TRAIN=models/trace_sft/seed61-full
OUT=eval/pointer_traces/seed61-20261010
mkdir -p "$OUT"
stage() {
    "$PYTHON" - "$OUT" "$1" <<'PY'
import sys
from pathlib import Path
from scripts.eval.paper_baseline import atomic_json
stage=sys.argv[2]
atomic_json(Path(sys.argv[1])/'pipeline.json',dict(status=stage if stage in ('failed','complete') else 'running',stage=stage))
PY
}
trap 'stage failed' ERR
if [[ "${1:-}" == --dry-run ]]; then
    "$PYTHON" -m scripts.training.train_trace --dry-run
    echo "Queue: ordinary SFT -> original trace benchmark -> SFT trace benchmark -> audits/plots"
    exit 0
fi
if [[ $# -gt 0 ]]; then echo "Usage: bash trace_baselines.sh [--dry-run]" >&2;exit 2;fi
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo "Trace pipeline already running" >&2;exit 1; }
stage training
if [[ ! -f "$TRAIN/summary.json" ]]; then
    resume=()
    if [[ -f "$TRAIN/last/training_state.pt" ]]; then resume=(--resume "$TRAIN/last")
    elif [[ -f "$TRAIN/best/training_state.pt" ]]; then resume=(--resume "$TRAIN/best");fi
    "$PYTHON" -m scripts.training.train_trace "${resume[@]}" --wandb-mode "$WANDB_MODE"
fi
stage original-trace-benchmark
"$PYTHON" -m scripts.eval.trace_baselines --arm original --wandb-mode "$WANDB_MODE"
stage sft-trace-benchmark
"$PYTHON" -m scripts.eval.trace_baselines --arm sft --wandb-mode "$WANDB_MODE"
stage independent-audits-and-plots
"$PYTHON" -m scripts.eval.trace_analysis --input "$OUT"
stage complete
echo "Complete: $OUT/plots/index.md"

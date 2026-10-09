#!/usr/bin/env bash
# Frozen population analysis. Safe to rerun: extraction resumes identical chunks.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/loopformer-paper-mpl"
output="${1:-eval/pointer_analysis/paper-20261009}"
mkdir -p "$output"
rm -f "$output/COMPLETE" "$output/FAILED"
touch "$output/RUNNING"
trap 'code=$?; printf "Suite failed (exit %s). See run.log; retained chunks are resumable.\n" "$code"; touch "$output/FAILED"; rm -f "$output/RUNNING"; exit "$code"' ERR
exec > >(tee -a "$output/run.log") 2>&1
printf 'Pointer paper suite: %s\nFrozen weights; 1350 graphs × all depths 1–256; FP32/SDPA.\n' "$output"
python -m scripts.eval.paper_reuse_audit --input "$output"
python -m scripts.eval.paper_suite --output "$output"
python -m scripts.eval.paper_baseline --input "$output"
python -m scripts.eval.paper_snapshots --output "$output"
python -m scripts.eval.paper_metrics --input "$output"
python -m scripts.eval.paper_structure --input "$output"
python -m scripts.eval.audit_paper_analysis --input "$output"
python -m scripts.eval.plot_paper_analysis --input "$output" --replace
python -m scripts.eval.retire_pointer_plots --input "$output"
python -m scripts.eval.paper_report --input "$output"
printf 'Complete. Read %s/plots/index.md\n' "$output"
touch "$output/COMPLETE"
rm -f "$output/RUNNING"

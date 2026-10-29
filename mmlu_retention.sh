#!/usr/bin/env bash
# Frozen evaluation only: standard MMLU, pinned original Qwen and recurrent T=1.
set -euo pipefail
cd "$(dirname "$0")"
run_dir=eval/general/mmlu-retention-20261012
mkdir -p "$run_dir"
exec 9>"$run_dir/launcher.lock"
flock -n 9 || { printf 'MMLU launcher is already running.\n' >&2; exit 1; }
trap 'printf "failed\n" > "$run_dir/STAGE"' ERR
printf 'preflight\n' > "$run_dir/STAGE"
.venv/bin/python -m scripts.eval.mmlu --arm base --preflight > "$run_dir/preflight.log" 2>&1
printf 'base\n' > "$run_dir/STAGE"
.venv/bin/python -m scripts.eval.mmlu --arm base > "$run_dir/base.log" 2>&1
printf 'recurrent\n' > "$run_dir/STAGE"
.venv/bin/python -m scripts.eval.mmlu --arm recurrent > "$run_dir/recurrent.log" 2>&1
printf 'audit\n' > "$run_dir/STAGE"
.venv/bin/python -m scripts.eval.mmlu_analysis > "$run_dir/audit.log" 2>&1
printf 'complete\n' > "$run_dir/STAGE"

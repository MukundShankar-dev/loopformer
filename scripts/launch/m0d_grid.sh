#!/usr/bin/env bash
# M0d base-model grid on the dev sets: single turn, then repair and tools at 2 and 4 turns, 16 samples at T=0.8.
# Selected@k and pass@k for k <= 16 come from the same 16 samples (scripts/agent/report.py). Every run shares dtype,
# rows and cache size, since bf16 results depend on batch settings (docs/history.md, M0c). Completed runs are skipped,
# so the script can be re-run after an interruption; a partial run must be moved aside first.
#   bash scripts/launch/m0d_grid.sh [set ...]    (default: mbppplus humanevalplus livecodebench)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
model="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
output=eval/agent/m0d-grid-20261031
declare -A problems=(
  [mbppplus]=data/agent/mbppplus-v0.2.0
  [humanevalplus]=data/agent/humanevalplus-v0.1.10
  [livecodebench]=data/agent/livecodebench-release_v6-0fe84c3-before-2024-11-12
)
conditions=("single 1" "repair 2" "repair 4" "interaction 2" "interaction 4")
python="$HOME/loopformer/.venv/bin/python"
sets=("$@")
[[ ${#sets[@]} -gt 0 ]] || sets=(mbppplus humanevalplus livecodebench)
mkdir -p "$output"
for set in "${sets[@]}"; do
  [[ -n "${problems[$set]:-}" ]] || { echo "unknown set $set" >&2; exit 2; }
  for condition in "${conditions[@]}"; do
    read -r name turns <<< "$condition"
    run="$output/$set-$name-$turns"
    if [[ -f "$run/run.json" ]] && grep -q '"throughput"' "$run/run.json"; then
      echo "skip $run (complete)"
      continue
    fi
    if [[ -e "$run" ]]; then
      echo "$run is partial; move it aside and re-run" >&2
      exit 1
    fi
    echo "$(date -u +%FT%TZ) start $run"
    OMP_NUM_THREADS=4 nice "$python" -m scripts.agent.run --model "$model" --problems "${problems[$set]}" \
      --condition "$name" --max-turns "$turns" --seeds $(seq 0 15) --temperature 0.8 --max-turn-tokens 1024 \
      --dtype bfloat16 --rows 48 --capacity 6144 --first-failure --output "$run/episodes.jsonl" > "$run.log" 2>&1
    by=(); [[ $set == livecodebench ]] && by=(--by difficulty)
    "$python" -m scripts.agent.report --episodes "$run/episodes.jsonl" --problems "${problems[$set]}" "${by[@]}" > /dev/null
    echo "$(date -u +%FT%TZ) done $run: $(tail -1 "$run.log")"
  done
done

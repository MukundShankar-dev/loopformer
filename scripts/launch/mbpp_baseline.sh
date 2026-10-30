#!/usr/bin/env bash
# MBPP+ baseline of pinned Qwen2.5-Coder-0.5B-Instruct: official EvalPlus codegen
# (vLLM, GPU share capped) on the host, official evaluation inside the sandbox.
# Usage: scripts/launch/mbpp_baseline.sh [run_dir]. Needs .venv-probe and a systemd
# user session for the sandbox (see docs/setup.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
model=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25
run_dir=${1:-eval/codeprobe/mbpp-coder-0.5b-20261030}
python=.venv-probe/bin/python
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VLLM_GPU_MEMORY_UTILIZATION=0.3 VLLM_USE_FLASHINFER_SAMPLER=0  # no nvcc here for its JIT
mkdir -p "$run_dir"
printf 'greedy codegen\n' > "$run_dir/STAGE"
nice -n 5 $python -m scripts.codeprobe.codegen --model "$model" --dataset mbpp --backend vllm --greedy \
  --root "$run_dir/greedy" > "$run_dir/codegen-greedy.log" 2>&1
printf 'sampled codegen\n' > "$run_dir/STAGE"
nice -n 5 $python -m scripts.codeprobe.codegen --model "$model" --dataset mbpp --backend vllm --n_samples 16 \
  --temperature 0.8 --bs 16 --root "$run_dir/sampled" > "$run_dir/codegen-sampled.log" 2>&1
for kind in greedy sampled; do
  printf '%s evaluation\n' "$kind" > "$run_dir/STAGE"
  samples=$(ls "$run_dir/$kind"/mbpp/*_temp_*[0-9].jsonl)
  nice -n 10 $python -m scripts.codeprobe.evaluate_sandboxed --dataset mbpp --samples "$samples" \
    --output "$run_dir/$kind/eval_results.json" --parallel 2 --memory-gb 10 > "$run_dir/evaluate-$kind.log" 2>&1
done
printf 'summary\n' > "$run_dir/STAGE"
$python -m scripts.codeprobe.headroom --evalplus "coder-0.5b/mbpp-greedy=$run_dir/greedy/eval_results.json" \
  --evalplus "coder-0.5b/mbpp-n16-t0.8=$run_dir/sampled/eval_results.json" --output "$run_dir/summary.json"
printf 'complete\n' > "$run_dir/STAGE"

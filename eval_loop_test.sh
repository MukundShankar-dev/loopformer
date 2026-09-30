source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096

for batch in 1 4 8 16; do
		python -m scripts.eval.loop_test --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 --data data/pointer/seed-17/depth_test.jsonl --device cuda --loops 16 --limit 128 --batch-size "$batch" --output "eval/pointer_loops/perf-baseline2500-batch-$batch" || break
done


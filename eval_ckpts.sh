source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for step in 002500 003250 003750; do
  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/depth6-loopbalanced-seed37/step-$step" \
    --data data/pointer/seed-17/validation.jsonl \
    --device cuda --loops 8 || break

  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/depth6-loopbalanced-seed37/step-$step" \
    --data data/pointer/seed-17/depth_test.jsonl \
    --device cuda --loops 16 || break
done
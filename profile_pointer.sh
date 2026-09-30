PROFILE_POINTER=models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500

for batch in 4 8; do
		python -m scripts.eval.profile_pointer --model "$POINTER_CHECKPOINT" --data data/pointer/seed-37-depth6-30k/train.jsonl --device cuda --mode train --batch-size "$batch" --effective-batch 8 --train-max-depth 6 --ouput "eval/pointer_profiles/baseline2500-train-b$batch" ||break
done

python -m scripts.eval.probe_pointer --model "$POINTER_CHECKPOINT" --data data/pointer/seed-17/depth_test.jsonl --device cuda --limit 32 --restart-after 6 --output eval/pointer_probes/baseline2500


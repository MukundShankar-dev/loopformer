for step in 000500 000800 000938; do
	python -m scripts.eval.loop_test --model "models/stage1_pointer/depth6-seed17/step-$step" --data data/pointer/seed-17/validation.jsonl --device cuda --loops 8 || break
	python -m scripts.eval.loop_test --model "models/stage1_pointer/depth6-seed17/step-$step" --data data/pointer/seed-17/depth_test.jsonl --device cuda --loops 16 || breakwq

done

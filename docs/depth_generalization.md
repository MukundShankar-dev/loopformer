# Depth-6 training and outward OOD evaluation

Status: setup implemented; pretrained depth-6 training and paired sweeps have not run. This is the next Stage 1 experiment. Overscaling and knowledge-retention benchmarks remain deferred.

## Question and comparison

The depth-4 model at update 1875 achieves 99.4% complete trajectories at validation depths 1–4, 82.4% at depth 5, 20.8% at depth 6, 1.6% at depth 7, and 0% at depth 8. We now test whether training through depth 6 supports execution farther beyond training, rather than only solving depths newly included in training.

| Role | Training exposure | In-range evaluation | Nearby OOD | Farther evaluation |
| --- | --- | --- | --- | --- |
| Reference | Depths 1–4, three epochs | 1–4 | 5–6 | 7–16 |
| New stage | Reference adapters plus depths 1–6 | 1–6 | 7–8 | 9–16 |

Both checkpoints are evaluated on the **same examples at every absolute depth**, using the existing seed-17 `validation.jsonl` (1–8) and `depth_test.jsonl` (9–16). These become development diagnostics when used to decide further training. A separate seeded confirmation dataset remains reserved for after the setup is frozen.

Report absolute depth and `steps_beyond_training = task_depth - train_max_depth`. Compare equal positive offsets too: reference depth 6 versus candidate depth 8 are both +2. Equal offsets involve different-depth tasks, not paired examples. Improvements at candidate depths 5–6 are in-range results, not OOD gains.

The reference is the existing checkpoint, not an equal-compute control. The candidate receives additional training examples, updates, and recurrent computation, and resets its optimizer. This is a curriculum extension experiment; it does not isolate training depth as the sole cause. Record the extra budget rather than attributing every improvement to depth alone.

## Preview, then train on the desktop

From the repository root, with complete checkpoint files still present:

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6.json --device cuda \
  --init-from models/stage1_pointer/20260911T003442.178029Z/step-001875 \
  --dry-run
```

Preview checks config, data, tokenizer, and source metadata without loading weights or writing files. It cannot verify adapter tensor contents or actual GPU memory. After reviewing it:

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6.json --device cuda \
  --init-from models/stage1_pointer/20260911T003442.178029Z/step-001875 \
  --output models/stage1_pointer/depth6-seed17
```

The output directory must not already exist. The config selects **7,500 mappings**, 1,250 at each depth 1–6, for **one new epoch / 938 optimizer updates** (the last update has four examples). Each example receives equal total nominal CE weight, as before. Batch 1, accumulation 8, float32, rank-8 q/v LoRA, learning rate 0.0002 and ten warmup updates remain explicit. Original Qwen parameters stay frozen. Validation monitoring remains eight examples per depth 1–8; the probe has four examples per trained depth, now 24 total.

`--init-from` loads only compatible adapters and records source checkpoint hashes and source training depth. It starts a new optimizer, seed-driven RNG sequence, warmup schedule, and update/epoch counters. The source has already seen 15,000 example presentations; this stage adds 7,500. It is **not** `--resume`, which restores optimizer/RNG/cursor and rejects changed training/data settings. These flags are mutually exclusive. A completed warm-start stage can subsequently be resumed with its own matching config and checkpoint.

Base revision, recurrent split, LoRA settings, symbol IDs, prompt/loss format, and tokenizer must match the source. The new recorded training maximum cannot be below the source maximum. Checkpoint lineage persists into saved metadata and through later exact-resume attempts. Inference remains compatible with `naive_test` and `loop_test`.

Six-loop backward resource use is not yet measured on the desktop. The dashboard records actual memory and ETA; no fallback device/dtype change is made. The existing startup gate checks architecture and a short gradient path, not six-loop peak memory. This one-epoch stage is a bounded pilot; review its results before extending it.

## Evaluate both models through depth 16

After training, resolve the new run's **validation-selected best checkpoint** automatically. Selection uses nominal loss on trained depths 1–6; depths 7+ do not enter selection. The new run has its own selection history and does not compare its loss against the previous depth-4 run's differently defined selection loss.

Preview the paired sweep first:

```bash
python -m scripts.eval.depth_generalization \
  --reference-model models/stage1_pointer/20260911T003442.178029Z/step-001875 \
  --model-run models/stage1_pointer/depth6-seed17 \
  --device cuda --dry-run
```

Then run:

```bash
python -m scripts.eval.depth_generalization \
  --reference-model models/stage1_pointer/20260911T003442.178029Z/step-001875 \
  --model-run models/stage1_pointer/depth6-seed17 \
  --device cuda
```

Alternatively, `--model` accepts an explicit saved step directory. Dry-run is plan-only and still requires model metadata (or the run's best-checkpoint pointer); it performs no inference or writes. The full command launches four ordinary `loop_test` sweeps with Rich progress: each checkpoint on all 1,000 validation examples through eight loops and all 1,000 deeper examples through sixteen loops. No terminal transformation or retention scoring occurs.

Outputs go under `eval/pointer_depth_comparison/<timestamp>/`. Each child directory has the normal full-loop artifacts. The parent `comparison.csv` and `summary.json` report both checkpoints by absolute depth, offset beyond training, trained/near-OOD/far-OOD region, nominal final accuracy, complete-trajectory accuracy, CE, and count. Comparisons require identical source-data hashes and inference settings within each pair, consistent checkpoint hashes across files, and disjoint depth ranges. An interrupted or failed run remains `status: running`; only complete exports get `status: complete`. Choose a new output directory to rerun.

Use `--data` repeatedly to override the two source files and `--output` for a new explicit parent directory. Relative-depth regions are model-specific: +1/+2 is near OOD; +3 or more is far OOD. Do not pool them with trained depths or describe final-only accidental matches as valid trajectories. File-level labels such as `depth_test` are not substitutes for these exposure-based definitions.

## Reserved confirmation set

Reserve master seed **29** and do not generate, inspect, or evaluate its questions during this development cycle. After selecting training settings and checkpoints and fixing the claims/metrics, generate the independent dataset with the pinned tokenizer:

```bash
python -m scripts.dataset --seed 29 \
  --train-count 10000 --validation-count 1000 --test-count 1000 --depth-test-count 1000 \
  --min-depth 1 --max-train-depth 8 --max-eval-depth 16 \
  --output data/pointer/seed-29
```

Use only `test.jsonl` and `depth_test.jsonl` for that final assessment, evaluating both frozen checkpoints with repeated `--data` arguments. The generator also creates train/validation files; do not train on them as part of the confirmation experiment. Check exact whole-table overlap with development/training data before interpreting confirmation. A seed alone is not a substitute for overlap checks or a predeclared protocol. If confirmation informs another training change, reserve another untouched set rather than continuing to call seed 29 final confirmation.

## Validation

Implementation tests cover initialization compatibility and exposure tracking, preservation of loaded tensors before the first update, a new optimizer/counter without parent optimizer files, subsequent resume of the initialized stage, matching-dataset comparison, relative-depth classification, and a plan-only evaluation preview. No pretrained depth-6 training, OOD sweep, or seed-29 generation was launched by the assistant. The Mac full suite passed 94 tests in 64.94 seconds; the nine focused cases passed in 27.96 seconds. The real-data depth-6 preview, CLI help, and local documentation links passed. See [status](status.md).

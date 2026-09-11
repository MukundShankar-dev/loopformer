# LoopFormer

This project studies whether a recurrent block in `Qwen/Qwen2.5-0.5B-Instruct` can learn one pointer transition per loop, generalize to deeper tasks, and eventually repair mistakes while preserving correct answers.

The current focus is **further pointer training and depth generalization**. Full-loop tests show useful execution on unseen mappings at trained depths 1–4, with weak extension beyond them. Overscaling tools are prepared for later; those experiments have not run. See [current status](docs/status.md) and the [research plan](docs/project_plan.md).

## Setup

Use Python 3.11 and run commands from the repository root. Windows/NVIDIA users should follow the [WSL2/CUDA setup guide](docs/windows_cuda_setup.md) first; other environment details are in [setup](docs/setup.md).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Download the base revision used for the reproducible experiments:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

Check ordinary model loading and generation:

```bash
python scripts/smoke_test_qwen.py
```

Activate `.venv` in each new terminal. The commands below default to CPU unless a device is supplied. Use `--device mps` on Apple Silicon or `--device cuda` on the configured NVIDIA desktop. For deterministic CUDA runs, set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Pointer dataset: preview and reproduce

Dataset code lives in [`scripts/dataset/`](scripts/dataset/). Preview five examples without writing files:

```bash
python -m scripts.dataset --seed 17 --dry-run
```

Use `--dry-run 10` for ten examples. Reproduce the existing dataset with master seed **17**, the pinned tokenizer above, and this complete configuration:

```bash
python -m scripts.dataset \
  --seed 17 \
  --train-count 10000 \
  --validation-count 1000 \
  --test-count 1000 \
  --depth-test-count 1000 \
  --min-depth 1 \
  --max-train-depth 8 \
  --max-eval-depth 16 \
  --output data/pointer/seed-17

python -m scripts.dataset --verify data/pointer/seed-17
```

The existing seed-17 dataset **already reaches depth 16; no regeneration is needed for depth-6 training**. Generated files are excluded from Git, so use the reproduction command above only if the dataset is missing on a new machine.

| File under `data/pointer/seed-17/` | Examples | Depths | Examples per depth |
| --- | ---: | --- | ---: |
| `train.jsonl` | 10,000 | 1–8 | 1,250 |
| `validation.jsonl` | 1,000 | 1–8 | 125 |
| `test.jsonl` | 1,000 | 1–8 | 125 |
| `depth_test.jsonl` | 1,000 | 9–16 | 125 |

The depth-6 training config selects only depths 1–6 from `train.jsonl`. Its `validation_max_depth: 8` controls training-time monitoring; the paired depth evaluator separately reads `depth_test.jsonl` and runs through depth 16. See [dataset details](docs/phases/stage1_pointer.md) and [paired evaluation](docs/depth_generalization.md).

## Evaluate a model

Inspect three ordinary-model questions before running the full test:

```bash
python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test
python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct
```

The ordinary baseline uses **three-shot examples at depths 1, 2, and 3**, defined in [`prompts/pointer_task.txt`](prompts/pointer_task.txt). `--model` also accepts a saved model directory. Models load locally by default; `--download` permits missing downloads.

Progress and throughput appear in the terminal. Predictions and summaries go to `eval/pointer_task/`. See [baseline evaluation](docs/naive_pointer_eval.md) for loading and scoring options.

## Train pointer execution

Preview the initial training configuration before launching it:

```bash
python -m scripts.training.train_pointer --config configs/stage1_pointer.json --dry-run
python -m scripts.training.train_pointer --config configs/stage1_pointer.json --device cuda
```

Configs live in [`configs/`](configs/). The trainer uses raw dataset prompts and exact per-loop supervision. A compact Rich dashboard shows progress, ETA, losses, accuracy, and memory. Checkpoints and logs go to `models/stage1_pointer/`; model binaries are excluded from Git.

The three-epoch depth-4 run is complete. The next experiment uses `configs/stage1_pointer_depth6.json` and adapter-only initialization, with OOD evaluation moved outward through depth 16. Follow the [depth-6 setup and paired evaluation commands](docs/depth_generalization.md), starting with its dry-run. The [training guide](docs/training_pointer.md) covers checkpoint selection and resume.

## Inspect recurrent checkpoints

Pass a complete saved step directory, including its adapter weights and tokenizer:

```bash
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/<run>/step-000625 \
  --device cuda --loops 8 --test
```

Remove `--test` for all 1,000 test examples. Outputs go to `eval/pointer_loops/`. This reads the model after every recurrent loop; the ordinary three-shot prompt is not used. See [full-loop evaluation](docs/loop_pointer_eval.md) for commands and metrics.

## Deferred overscaling experiments

Preview the separate absorbing-terminal task variant without loading a model or writing files:

```bash
python -m scripts.eval.overscaling_test --dry-run
```

Checkpoint sweeps, repair/damage scoring, and survival exports are available but **running them is deferred until further Stage 1 training and review**. See [overscaling usage](docs/overscaling_eval.md).

## Documentation

- [Documentation index](docs/README.md)
- [Current status and next experiment](docs/status.md)
- [Research plan](docs/project_plan.md)
- [Audited training and full-loop results](docs/experiments/stage1_cuda_5k.md)

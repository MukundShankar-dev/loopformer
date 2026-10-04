# LoopFormer

LoopFormer studies **recurrent computation and adaptive inference depth** in `Qwen/Qwen2.5-0.5B-Instruct`.

> Can a recurrent language model learn useful iterative computation, and can inference determine when additional recurrent computation is useful, unnecessary, or harmful?

Reusing a learned block lets inference vary computational depth without adding a separate set of weights for every step. The project first tests whether recurrence executes an interpretable algorithm, then studies depth generalization, overscaling dynamics, and eventually compute-quality tradeoffs through stopping. Pointer chasing is the controlled starting environment: fresh rules live in each prompt, and exact intermediate targets make errors and progress checkable.

```text
prompt
  ↓
frozen prelude
  ↓
shared recurrent block × T
  ↓
frozen coda
  ↓
readout
```

One recurrent block and its LoRA adapters are reused across all loops. Original pretrained weights stay frozen. The prelude runs once; the frozen coda reads each loop's state for supervision and evaluation. The next loop consumes the recurrent hidden state, rather than the coda output or a decoded answer. See [architecture](docs/architecture.md).

## Verified so far

| Evidence | Scope and limits |
| --- | --- |
| [One-loop equivalence](docs/experiments/stage0_validation.md), shared-weight and gradient-scope tests | Validated model surgery; useful task execution needs separate evidence. |
| [Fresh 30k depth-6 experiment](docs/experiments/stage1_fresh30k.md) | Strong unseen-mapping trajectories and bounded extension beyond training depth; one training run on development evaluation sets. |
| [Joint learned-completion ablation](docs/experiments/stage1_learned_completion.md) | Stop timing and pointer execution do not extend as well as the CE-only reference; one development run. |
| [Evaluation and reproducibility](docs/evaluation.md) | Seeded data, exact intermediate supervision, full per-loop exports, source/data/checkpoint provenance, selection discipline, synchronized timing and throughput. |
| [Recorded contract-test validation](docs/status.md) | Architecture, data, training and metric contracts; tests do not establish empirical research gates. |

For the fresh 30k run, **step 2500** achieved the following complete-trajectory accuracy (every nominal intermediate state correct):

| Task depth | 1–6 | 7 | 8 | 9 | 10 | 11 | 12 | 13–16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 98.0% | 94.4% | 94.4% | 73.6% | 41.6% | 14.4% | 1.6% | 0% |

Depths 1–6 aggregate 750 examples; each individual depth has 125. Step 2500 is the working depth-generalization checkpoint among three fully evaluated candidates; step 3250 remains the trained-loss-selected reference. Seed-17 evaluations are development diagnostics, and seed 29 remains reserved for confirmation. This supports execution beyond the training depth followed by degradation at greater task depths. It does not establish arbitrary-depth execution, general reasoning, or post-completion overscaling damage.

The completed [mechanism diagnostics](docs/experiments/stage1_integrated_mechanism.md) and [learned-completion ablation](docs/experiments/stage1_learned_completion.md) narrow the failure but do not identify its internal cause. [Current status](docs/status.md) records the evidence and remaining gates.

## What comes next

The [loop-balanced loss ablation](docs/experiments/stage1_loopbalanced.md) and joint completion training did not improve farther-depth generalization. Further experiments should target a specific unresolved mechanism. [Terminal overscaling evaluation](docs/evaluation.md) already provides repair/damage and continuous-survival metrics, but pretrained terminal sweeps remain deferred. Asymmetric dynamics and transfer remain planned.

The next major direction is **adaptive inference compute**. Offline oracle/heuristic analysis, opt-in stopped inference, synchronized latency recording, and a gated lightweight head trainer now have code and tests. [The adaptive-compute guide](docs/adaptive_compute.md) is the single source for methods and usage; no pretrained adaptive policy or speedup result has been verified. The original continuing-pointer tasks need explicit completion semantics before answer changes can be interpreted as damage.

Start with the [research plan](docs/project_plan.md), [current status](docs/status.md), [evaluation guide](docs/evaluation.md), and [documentation index](docs/README.md).

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

The existing seed-17 dataset **already reaches depth 16** for development evaluation. The current fresh-training experiment uses a separate [30k seed-37 dataset](docs/training_pointer.md#loop-balanced-loss-experiment). Generated files are excluded from Git, so use the reproduction command above only if the dataset is missing on a new machine.

| File under `data/pointer/seed-17/` | Examples | Depths | Examples per depth |
| --- | ---: | --- | ---: |
| `train.jsonl` | 10,000 | 1–8 | 1,250 |
| `validation.jsonl` | 1,000 | 1–8 | 125 |
| `test.jsonl` | 1,000 | 1–8 | 125 |
| `depth_test.jsonl` | 1,000 | 9–16 | 125 |

The depth-6 training config selects only depths 1–6 from `train.jsonl`. Its `validation_max_depth: 8` controls training-time monitoring; the paired depth evaluator separately reads `depth_test.jsonl` and runs through depth 16. See [dataset details](docs/training_pointer.md) and [paired evaluation](docs/training_pointer.md).

## Evaluate a model

Inspect three ordinary-model questions before running the full test:

```bash
python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test
python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct
```

The ordinary baseline uses **three-shot examples at depths 1, 2, and 3**, defined in [`prompts/pointer_task.txt`](prompts/pointer_task.txt). `--model` also accepts a saved model directory. Models load locally by default; `--download` permits missing downloads.

Progress and throughput appear in the terminal. Predictions and summaries go to `eval/pointer_task/`. See [baseline evaluation](docs/evaluation.md) for loading and scoring options.

## Train pointer execution

Preview the initial training configuration before launching it:

```bash
python -m scripts.training.train_pointer --config configs/stage1_pointer.json --dry-run
python -m scripts.training.train_pointer --config configs/stage1_pointer.json --device cuda
```

Configs live in [`configs/`](configs/). The trainer uses raw dataset prompts and exact per-loop supervision. A compact Rich dashboard shows progress, ETA, losses, accuracy, and memory. Checkpoints and logs go to `models/stage1_pointer/`; model binaries are excluded from Git.

The next experiment uses `configs/stage1_pointer_depth6_loopbalanced.json`: fresh adapters on the existing 30,000 mappings, with equal loss weight per loop position. Follow the [loop-balanced training commands](docs/training_pointer.md#loop-balanced-loss-experiment), starting with the dry-run. The [training guide](docs/training_pointer.md) also covers checkpoint selection, resume, tmux, and [old-artifact cleanup](docs/training_pointer.md#artifact-cleanup).

## Inspect recurrent checkpoints

Pass a complete saved step directory, including its adapter weights and tokenizer:

```bash
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 \
  --device cuda --loops 8 --test
```

This requires the complete checkpoint on the training desktop; local metadata alone is insufficient. Remove `--test` for all 1,000 test examples. Outputs go to `eval/pointer_loops/`. This reads the model after every recurrent loop; the ordinary three-shot prompt is not used. See [full-loop evaluation](docs/evaluation.md) for commands and metrics.

## Deferred overscaling experiments

Preview the separate absorbing-terminal task variant without loading a model or writing files:

```bash
python -m scripts.eval.overscaling_test --dry-run
```

Checkpoint sweeps, repair/damage scoring, and survival exports are available but **running them is deferred until further Stage 1 training and review**. See [overscaling usage](docs/evaluation.md).

## Documentation

- [Documentation index](docs/README.md)
- [Current status and next experiment](docs/status.md)
- [Research plan](docs/project_plan.md)
- [Fresh 30k training and full-loop results](docs/experiments/stage1_fresh30k.md)
- [Adaptive-compute guide](docs/adaptive_compute.md)

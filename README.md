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

One recurrent block is reused across all loops. The executor experiment trained all weights in that block while keeping the prelude and coda frozen; older experiments used LoRA. The executor’s prompt context stays fixed across loops; the frozen coda reads each loop's state for supervision and evaluation. The next loop consumes the recurrent hidden state, rather than the coda output or a decoded answer. See [architecture](docs/architecture.md).

## Current result and diagnostics

The frozen model passes the [full benchmark](docs/experiments/pointer_final_benchmark.md):
100% on the 1,536-query existing test, **97.81% answer-plus-exact-stop success**
across 1,350 new graphs at every depth 1–256, and **96.89% complete trajectories
through 256**. Training exposure stayed unchanged: R unrolled through 12;
controller count labels through 63, with held-out values. Exact stopping is 100%
through 256. All graphs have 26 states and long executions can cycle. The earlier
[frozen benchmark](docs/experiments/pointer_frozen_benchmark.md) records the
number-reading failure that motivated this repair.

See [repair commands and data](docs/number_reader_repair.md),
[controller architecture](docs/architecture.md), and the saved
[figures](docs/experiments/pointer_final_benchmark.md#commands-figures-and-artifacts).
Weights remain on the desktop; Git carries metrics, audits and metadata.

The [repair runbook](docs/number_reader_repair.md) fits a shared number reader,
then tightens its two countdown parameters on the same training labels.
`bash repair_number_reader.sh --dry-run` and `bash refine_countdown.sh --dry-run`
preview the two stages. The completed final checkpoint is
`models/stage1_pointer/controller-shared-number-precision-seed83/best`.
`bash benchmark_pointer.sh --full --dry-run` previews the full benchmark;
see the [report](docs/experiments/pointer_final_benchmark.md) for reproduction
and plotting commands. Controller-only stopping through 8,192 is a separate
numerical result; whole-model pointer quality above 256 remains untested.

Start with the [research plan](docs/project_plan.md), [current status](docs/status.md),
and [documentation index](docs/README.md). Historical experiments remain in the
linked reports; development results are not independent confirmation.

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

## Original pointer dataset: preview and reproduce

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

The existing seed-17 dataset **already reaches depth 16** for development evaluation. The current executor uses the separate seed-61 dataset created by `bash prepare_executor.sh`; see [its run instructions](docs/training_pointer.md#isolated-executor-upgrade--current-desktop-run). Generated files are excluded from Git, so use the reproduction command above only if the dataset is missing on a new machine.

| File under `data/pointer/seed-17/` | Examples | Depths | Examples per depth |
| --- | ---: | --- | ---: |
| `train.jsonl` | 10,000 | 1–8 | 1,250 |
| `validation.jsonl` | 1,000 | 1–8 | 125 |
| `test.jsonl` | 1,000 | 1–8 | 125 |
| `depth_test.jsonl` | 1,000 | 9–16 | 125 |

The current seed-61 run has 36,000 training questions at requested counts 1–6, 8, 10 and 12; its deep development queries span 13–64. See [dataset and training details](docs/training_pointer.md).

## Evaluate a model

Inspect three ordinary-model questions before running the full test:

```bash
python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test
python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct
```

The ordinary baseline uses **three-shot examples at depths 1, 2, and 3**, defined in [`prompts/pointer_task.txt`](prompts/pointer_task.txt). `--model` also accepts a saved model directory. Models load locally by default; `--download` permits missing downloads.

Progress and throughput appear in the terminal. Predictions and summaries go to `eval/pointer_task/`. See [baseline evaluation](docs/evaluation.md) for loading and scoring options.

## Train pointer execution

On the CUDA desktop, prepare the new depth-independent dataset and check the
isolated executor pipeline:

```bash
bash prepare_executor.sh --dry-run
bash prepare_executor.sh
bash train_executor.sh --dry-run
bash train_executor.sh --smoke-test
```

After inspecting the smoke results, run `bash train_executor.sh`, then
`bash eval_executor.sh`. The default trains all recurrent-block weights with an
isolated controller and per-loop supervision. Full-model and LoRA controls are
also available. See the [current run instructions](docs/training_pointer.md#isolated-executor-upgrade--current-desktop-run)
for configuration, exact data reproduction and output locations.

A compact Rich dashboard shows progress, ETA, losses and memory. W&B uses project
`loopformer`. Checkpoints and logs go to `models/stage1_pointer/`; model binaries
are excluded from Git. The completed desktop results and measured resource use are recorded in the linked reports.

## Inspect recurrent checkpoints

`bash eval_executor.sh` runs matched-count/precision diagnostics, full-loop evaluation and actual learned stopping on development data. Results and terminal logs go under `eval/pointer_diagnostics/`. The older depth-12 scripts remain available for historical runs.

Pass a complete saved step directory, including its adapter weights and tokenizer:

```bash
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/controller-shared-number-precision-seed83/best \
  --data data/pointer/seed-61-independent/validation.jsonl \
  --device cuda --loops 272 --stop-policy completion --stop-threshold 0.5 --test
```

This requires the complete checkpoint on the training desktop; local metadata alone is insufficient. Remove `--test` for all 1,536 validation queries. Outputs go to `eval/pointer_loops/`. This reads the model after every recurrent loop; the ordinary three-shot prompt is not used. See [full-loop evaluation](docs/evaluation.md) for commands and metrics.

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

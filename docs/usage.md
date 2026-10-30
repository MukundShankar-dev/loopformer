# Usage

Run everything from the repository root with `.venv` active ([setup](setup.md)). Launchers in `scripts/launch/` change to the repository root themselves, so run them as `bash scripts/launch/<name>.sh`. Most launchers activate `.venv`, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, default `WANDB_MODE=online`, and assume CUDA. Python CLIs default to `--wandb-mode disabled` (or `$WANDB_MODE`). Models load with `local_files_only` unless you pass `--download`. Every output directory must be new; nothing is overwritten.

Final model lineage: `data/pointer/seed-61-independent` → `executor_r-seed61/step-002250` → GRU controller → affine controller → shared number reader → `controller-shared-number-precision-seed83/best`, all under `models/stage1_pointer/`. Architecture is described in [architecture](architecture.md).

## Dataset

```bash
python -m scripts.dataset --dry-run                     # preview 5 examples; no writes
python -m scripts.dataset --verify data/pointer/seed-61-independent
bash scripts/launch/prepare_executor.sh                 # verifies the directory if it exists, otherwise generates:
python -m scripts.dataset --seed 61 --graph-mode mixture --paired-horizons \
  --train-count 36000 --validation-count 1536 --test-count 1536 --depth-test-count 1664 \
  --max-train-depth 12 --train-depths 1 2 3 4 5 6 8 10 12 --max-eval-depth 64 \
  --output data/pointer/seed-61-independent
```

Other flags: `--graph-mode {unique_path,mixture,random_function,permutation,full_cycle}`, `--min-depth`, `--dry-run 10`. The split files are `train`, `validation`, `test` and `depth_test` `.jsonl`, plus `manifest.json`. Generation needs only the cached tokenizer. `data/` is gitignored and can be regenerated from the seeds.

## Stage 0 gate

```bash
python -m scripts.validate_stage0 --device cpu --output artifacts/stage0/cpu.json   # or --device mps
```

Flags: `--recurrent-start`, `--recurrent-end`, `--rank`, `--alpha`, `--loops`, `--seed`, `--threads`, `--revision`, `--download`.

## Executor training

```bash
bash scripts/launch/train_executor.sh --dry-run      # config/data/tokenizer check; no weights loaded
bash scripts/launch/train_executor.sh --smoke-test   # 10 updates into a separate *-smoke-* run
bash scripts/launch/train_executor.sh                # configs/executor_r.json → models/stage1_pointer/executor_r-seed61
EXECUTOR_CONFIG=configs/executor_no_bridge.json bash scripts/launch/train_executor.sh
```

The launcher needs Linux or WSL with util-linux `script`, which keeps a log through a PTY. Ablation configs: `executor_full`, `executor_lora`, `executor_no_bridge`, `executor_no_direct`. To run the trainer directly:

```bash
python -m scripts.training.train_pointer --config configs/executor_r.json --output RUN [--device cuda]
```

| Flag | Effect |
| --- | --- |
| `--resume STEP_DIR` | Resume the optimizer, RNG and data cursor. The identity must match. |
| `--allow-batch-change` | With `--resume`, allow a change of `--batch-size`/accumulation if their product is unchanged |
| `--init-from STEP_DIR` | Load adapters only. Accepts only legacy `loopformer-stage1-v1` LoRA checkpoints. |
| `--batch-size`, `--download`, `--wandb-{mode,project,entity,name}` | Microbatch override, allow downloads, W&B settings |

A run directory holds `config.json`, `run.json`, `metrics.jsonl`, `validation-step-*.csv`, `train-probe-step-*.csv`, `step-XXXXXX/`, `best_checkpoint.json`, `last_checkpoint.json` and `summary.json`.

Executor diagnostics: `bash scripts/launch/eval_executor.sh [RUN]`. It runs `executor_diagnostic --precision-check` and forced and stopped `loop_test` on the seed-61 validation and depth_test splits.

## Controller training (frozen executor)

Run in order. Each step reads the previous step's outputs.

| Step | Command | Output |
| --- | --- | --- |
| 1. GRU controller, plus the feature cache `features.pt` | `bash scripts/launch/train_controller.sh [--dry-run\|--smoke-test]` (`configs/controller_seed61.json`) | `controller-seed61/` |
| 2. Remaining-work arms; writes `deep_features.pt` | `bash scripts/launch/train_controller_remaining.sh` (`configs/controller_remaining.json`) | `controller-remaining-seed61/`, `eval/pointer_diagnostics/controller-remaining-seed61/` |
| 3. Affine countdown: `prepare_controller_prefix`, then `train_controller`, then `controller_candidate --native-check` | `bash scripts/launch/repair_controller_affine.sh [--dry-run]` (`configs/controller_affine.json`) | `controller-affine-seed83/best` |
| 4. Shared number reader fit, then its own frozen benchmark (seeds 281/283/293) | `bash scripts/launch/repair_number_reader.sh [--dry-run]` (`configs/controller_number_reader.json`) | `controller-shared-number-seed83/best` |
| 5. Countdown precision refit, then `countdown_precision_test` | `bash scripts/launch/refine_countdown.sh [--dry-run]` (`configs/controller_number_precision.json`) | `controller-shared-number-precision-seed83/best` (final) |

Steps 4 and 5 call `python` without activating `.venv`, so activate it first. Direct CLI: `python -m scripts.training.train_controller --config CFG --output DIR [--features-cache PT] [--dry-run|--smoke-test]`. A controller-only audit: `bash scripts/launch/audit_controller.sh` (`scripts.eval.controller_training_audit`). Superseded repair arms: `repair_controller.sh` (initialization loss) and `repair_controller_prefix.sh` (GRU prefix supervision).

## Evaluating one checkpoint

```bash
python -m scripts.eval.loop_test --model CKPT --data data/pointer/seed-61-independent/test.jsonl \
  --device cuda --attention sdpa --loops 272 --stop-policy completion --stop-threshold .5 --output OUT
```

Forced-depth runs omit `--stop-policy`. They accept `--batch-size`, `--precision bf16` (CUDA only) and `--record-states`. A stopped run requires batch 1, FP32 and an explicit `--loops` cap. Other policies are `requested_depth`, `stability` (with `--stop-k`), `margin`, `entropy` and `learned` (with `--head`). Use `--limit N`, or `--test` to print 2–3 examples. The output directory holds `summary.json`, `decisions.csv` and `trajectories.csv`. For offline stopping analysis of a completed run: `python -m scripts.eval.adaptive_analyze RUN --output DIR`.

## Frozen benchmark and audit

```bash
bash scripts/launch/benchmark_pointer.sh --full [--dry-run]
```

`--full` uses `configs/pointer_full_benchmark.json`: the final checkpoint, seeds 307/311/313, and 1,350 new graphs × N=1..256. Without `--full`, the launcher reruns the earlier affine confirmation (`configs/pointer_benchmark.json`). The launcher runs these steps:

1. `freeze_pointer_benchmark`: hashes the checkpoint and protocol.
2. Native `loop_test` on the seed-61 test split, then `--verify-result` against the freeze.
3. `frozen_pointer_benchmark --freeze … --output … --dataset-output …`. It generates the graphs and runs one count-free executor rollout per graph with controller replay for every N. It then checks a native stopped panel for exact decision fidelity.
4. `audit_pointer_benchmark` with `--results/--graphs` and `--native-results/--native-data`.

Output for `--full`: `eval/pointer_benchmark/final-full-20261008/`. `frozen_pointer_benchmark` accepts only isolated executors with `affine_suffix` or `shared_number` controllers.

## Population analysis

```bash
bash scripts/launch/analyze_pointer.sh [OUT]          # default eval/pointer_analysis/paper-20261009
python -m scripts.eval.paper_status [--input OUT] [--json]
```

The launcher runs `paper_reuse_audit`, `paper_suite` (six registry models from `configs/pointer_analysis_protocol.json`, 1,350 graphs × N=1..256, FP32/SDPA, resumable chunks), `paper_baseline` (ordinary Qwen), `paper_snapshots`, `paper_metrics`, `paper_structure`, `audit_paper_analysis`, `plot_paper_analysis --replace`, `retire_pointer_plots` and `paper_report`. It writes `RUNNING`, `COMPLETE` or `FAILED` markers and `run.log`. `paper_report` writes `OUT/report.md`; figures are indexed in `OUT/plots/index.md`. Plotting needs matplotlib.

## Trace baselines

```bash
bash scripts/launch/trace_baselines.sh [--dry-run]
```

The launcher runs these steps:

1. `scripts.training.train_trace` (`configs/trace_sft.json`): full ordinary-Qwen SFT on written traces using `prompts/pointer_trace.txt`. Output: `models/trace_sft/seed61-full`. It resumes from `last/` or `best/` if present.
2. `scripts.eval.trace_baselines --arm original`, then `--arm sft` (`configs/pointer_trace_baselines.json`). These accept `--smoke-test`, `--profile-only` and `--output`.
3. `scripts.eval.trace_analysis --input eval/pointer_traces/seed61-20261010`.

## MMLU retention

```bash
python -m pip install -r requirements-mmlu.txt --constraint requirements.txt
bash scripts/launch/mmlu_retention.sh
```

The launcher needs CUDA and `flock`. It runs `scripts.eval.mmlu --arm base --preflight`, `--arm base`, `--arm recurrent`, then `scripts.eval.mmlu_analysis` (`configs/mmlu_retention.json`: 5-shot, FP32/SDPA, 14,042 questions, checkpoint SHA-256 enforced). The recurrent arm removes the router and controller and runs T=1. Output: `eval/general/mmlu-retention-20261012/`, with the current stage in `STAGE`.

## Demo

```bash
python -m pip install -r requirements-demo.txt --constraint requirements.txt
python -m scripts.demo --checkpoint models/stage1_pointer/controller-shared-number-precision-seed83/best \
  --request request.json [--device cpu|cuda|mps] [--loop-cap 272] [--max-steps 256]
python -m scripts.demo --model HF_REPO_ID [--revision REV] --request request.json --download
```

`request.json` contains `{"mapping": [[src,dst] ×26], "start": "A", "steps": N}`. The CLI prints one JSON frame per loop.

| Task | Command |
| --- | --- |
| Export playback examples from the full benchmark's native panel | `python -m scripts.demo.export_examples` (writes `demo/web/examples.json`) |
| Check streamed output against native inference | `python -m scripts.demo.verify_checkpoint --checkpoint CKPT --examples demo/web/examples.json --device DEV --output OUT` |
| Publish the model repo and a ZeroGPU Space | `python -m scripts.demo.deploy --checkpoint CKPT --model-id USER/MODEL --space-id USER/SPACE [--max-steps N]` |
| Verify hosted runs | `python -m scripts.demo.verify_hosted --space USER/SPACE --examples … --output … [--authenticated]` and `python -m scripts.demo.verify_proxy --url URL --output OUT` |
| Test the web client | `node --test demo/test_core.mjs` |

`deploy` refuses any weights whose hash differs from the final hash. It uploads the checkpoint, `scripts/recurrent_qwen`, `scripts/dataset`, the runtime parts of `scripts/demo`, `demo/requirements-inference.txt` and [the model card](../demo/hf_model_card.md). The Space runs `demo/hf_space/app.py`. The static client is `demo/web/` (`config.json` names the Space and the proxy).

## Code headroom probe (v2 M0a, prepared, not run)

See [plan milestones](plan.md#milestones-and-gates) (M0a) for the gate, and check the `--dry-run` flags against the installed tool versions before running.

```sh
python -m scripts.codeprobe.commands                  # print every step from configs/probe_headroom.json
.venv-probe/bin/python -m scripts.codeprobe.commands --run
python -m scripts.codeprobe.headroom --since 2025-01-01 \
  --evalplus coder-0.5b/humaneval=<eval_results.json> \
  --livecodebench coder-0.5b=<eval_all.json> --output eval/codeprobe/headroom-v1/summary.json
```

`--run` refuses unpinned model revisions, an empty `sandbox_prefix` and an existing output directory. The summary reports, per benchmark, the always/sometimes/never-solved fractions and unbiased pass@1/4/16.

## Tests

```bash
python -m pytest -q
```

- `tests/test_demo_proxy.py` imports `fastapi`, so install `requirements-demo.txt` first. Without it, collection fails.
- `tests/test_paper_baseline.py` reads the gitignored `data/pointer/benchmark-seeds307-311-313/graphs.jsonl`. Regenerate it with the full benchmark.
- Several tests read tracked `eval/` and `models/` metadata.
- `test_depth12_compute.py` skips without CUDA.
- `tests/test_sandbox.py` is the sandbox's adversarial suite. Most of it skips off Linux. Run it on any machine before that machine executes model-written code: `python -m pytest -q tests/test_sandbox.py` (about 12 s).
- Most architecture tests use a tiny random `Qwen2Config` ([tests/conftest.py](../tests/conftest.py)).

## Historical launchers

These reproduce superseded experiments; see [history](history.md). Their default checkpoints are earlier runs.

| Launcher | Purpose |
| --- | --- |
| `train_fixed_prompt.sh`, `eval_ckpts.sh` | depth-6 fixed-prompt LoRA + completion head, then forced and stopped evaluation |
| `prepare_depth12.sh`, `train_depth12.sh`, `eval_depth12.sh`, `compare_depth12.sh`, `probe_depth12.sh` | depth-12 gap dataset (seed 47), training, evaluation, checkpoint comparison, paired-step probe |
| `probe_steps.sh`, `probe_controls.sh`, `probe_rule_edits.sh` | paired Steps, control and rule-edit probes on depth-6 runs |
| `profile_pointer.sh` | training/probe profiling of the depth-6 baseline |
| `diagnose_mechanism.sh`, `diagnose_controller.sh`, `eval_controller.sh` | mechanism diagnostics, GRU controller diagnosis and evaluation |
| `repair_controller.sh`, `repair_controller_prefix.sh` | superseded controller repair arms |

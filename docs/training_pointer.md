# Train recurrent pointer execution

Status: implemented and tested with random tiny Qwen models; pretrained CUDA training and resource measurements are recorded in the [5,000-mapping report](experiments/stage1_cuda_5k.md). Full-test trajectories for updates 500 and 625 are now audited in the same report; trained-depth complete-trajectory accuracy is 86.2% and 92.4%, respectively. The current priority is further training for depth extension, before overscaling experiments. See [implementation validation](experiments/stage1_training_implementation.md) for the checks performed. The [Stage 1 guide](phases/stage1_pointer.md) owns the research objective and acceptance gate.

## Preview, then run

Complete the [repository setup and seed-17 dataset generation](../README.md) first. Run from the repository root with the virtual environment activated. The configuration is JSON under `configs/`; actual checkpoints and logs go under `models/`.

Start with the small overfit configuration:

```bash
python -m scripts.training.train_pointer --config configs/stage1_pointer_overfit.json --dry-run
python -m scripts.training.train_pointer --config configs/stage1_pointer_overfit.json
```

`--dry-run` checks the configuration, manifest checksums, train/validation table separation, selected examples, tokenizer symbol contexts, and prompt lengths. It prints the budget without loading model weights, training, downloading, or writing output. It requires the tokenizer already in the local cache.

The training command defaults to CPU float32. Add `--device mps` for Apple Silicon or `--device cuda` for CUDA; device selection never silently falls back. Missing model files require explicit `--download`. Each real run checks fresh one-loop equivalence, shared recurrent weights, and the two-loop loss gradient path before any optimizer update.

| Configuration | Training data | Budget | Fixed monitoring subsets |
| --- | --- | --- | --- |
| `configs/stage1_pointer_overfit.json` | 32 examples, eight per depth 1–4 | 20 epochs, 160 optimizer updates; batch 1, accumulate 4 | All 32 training examples; 16 validation examples at depths 1–4 |
| `configs/stage1_pointer.json` | 5,000 examples, 1,250 per depth 1–4 | One epoch, 625 updates; batch 1, accumulate 8 | 16 training examples; 64 validation examples at depths 1–8 |

The overfit run checks whether the learning setup can fit a small set; 20 epochs is a starting budget, not a success guarantee. Inspect the fixed train-probe trajectory accuracy, losses, and measured memory before scaling. Once that check is useful, start a fresh larger run:

```bash
python -m scripts.training.train_pointer --config configs/stage1_pointer.json
```

One epoch is an initial experiment budget. Judge progress using held-out intermediate and trajectory accuracy and the gap from the fixed training probe. The script does not declare Gate 1 passed. Research acceptance thresholds remain unresolved; choose them before interpreting the experiment. Training deeper than four loops requires an explicit config change and new resource measurements.

## Objective and data flow

Training reads the dataset's raw `Rules`/`Start`/`Steps`/`Answer:` prompt, without a chat template or demonstrations. The three-shot prompt in `prompts/pointer_task.txt` belongs to the ordinary-model baseline.

AdamW uses the configured learning rate after a linear warmup (default 10 updates; none in the overfit config), with gradient clipping at norm 1 and no weight decay by default.

All original Qwen weights remain frozen. The default shared R is layers 6–17, with q/v LoRA rank 8, alpha 16, zero dropout, and no bridge. Only these adapters are optimized. Coda readouts stay differentiable; later losses propagate through earlier recurrent states. Targets and predicted symbols are never fed back into the recurrent state.

At loop `t`, cross-entropy is over the 26 validated space-prefixed symbol tokens, targeting the exact state after `t` pointer transitions. For example, `A → F → C → Q` receives targets F, C, Q at loops 1, 2, 3. Loss is averaged across valid loops within each example, then across examples. Gradient accumulation uses that same example weighting, including partial final updates. This avoids implicitly weighting longer tasks more heavily. Per-loop diagnostic losses have their own valid-example denominators.

Inputs are right-padded, answer readout uses the last unmasked input token, and no prompt is truncated. Each training microbatch unrolls to its maximum task depth. Shorter examples receive no loss after completion. There is no final-answer retention, asymmetric objective, anchor loss, hidden-state detachment, or mixed precision in this first experiment. See [D06](decisions.md#stage-1-training-resolutions--2026-09-10).

Training subsets are balanced by depth using seed 17. Each epoch uses a deterministic shuffle derived from the seed and epoch index. Only the train and validation splits are used; test files do not influence optimization or checkpoint selection. Validation at depths 5–8 in the default run is a monitored depth extension, not an untouched test set.

## Progress and saved signals

Rich updates a single dashboard with the current phase, accumulation progress, optimizer-update bar, estimated remaining time, loss, step accuracy, whole-trajectory accuracy, per-loop losses, learning rate, gradient norm, throughput, and process peak memory. Validation and checkpoint phases also update in place. No training prompts or responses are printed. ETA becomes available after the first optimizer update and estimates validation overhead; it is approximate, particularly early in a run.

Every completed optimizer update is flushed to `metrics.jsonl`. Each validation event records both fixed validation and fixed train-probe metrics. Useful signals include:

- **Per-loop loss and intermediate accuracy:** whether individual transitions improve; denominators differ because only examples with `d >= t` contribute.
- **Whole-trajectory accuracy:** fraction with every nominal step correct, a stricter check than getting only the final answer right.
- **Fixed train/validation gap:** separates fitting the small training set from execution on unseen rule tables.
- **Final accuracy by depth and loop budget:** `depth_by_loop` records final-target accuracy at every observed loop. Each validation CSV also includes intermediate targets/correctness, final correctness, and the final-symbol margin against the strongest other allowed symbol.
- **Optimization and resources:** gradient norm before clipping, learning rate, examples/s, valid supervised transitions/s, process lifetime peak RSS, and MPS allocated/driver memory or CUDA allocated/peak memory when applicable. Non-finite symbolic logits and gradient norms fail the run.

Validation runs before training, every configured `eval_every` updates, at epoch boundaries, and at the final update. It sweeps every fixed validation example through `validation_max_depth` loops. Extra-loop final readouts are observational: the current task permits continued pointer execution, so a changed final answer after completion is not automatically overscaling damage. Repair/damage definitions remain a later-stage decision.

The default validation subset has only eight examples per depth (four in the overfit run). These inexpensive monitoring curves are not the final test result. `best_checkpoint.json` selects the lowest validation loss among depths seen in training, excluding monitored deeper depths. The initial untrained adapter checkpoint is eligible, so degradation remains visible.

## Artifacts and checkpoint evaluation

Each new run creates `models/stage1_pointer/<UTC timestamp>/`, or the new directory passed with `--output`. Existing output directories are rejected. Small JSON/JSONL metadata, metrics, CSV trajectories, and best/last pointers are trackable in Git. Binary model/optimizer state and repeated tokenizer payloads remain ignored; keep full checkpoint files separately for evaluation or resume. See the [first CUDA overfit report](experiments/stage1_cuda_overfit.md) for a completed run.

| Artifact | Contents |
| --- | --- |
| `config.json`, `run.json` | Effective config, command, model/data/tokenizer identity, package and code provenance, startup checks, run status |
| `metrics.jsonl`, `summary.json` | Update/validation/checkpoint events and the completed-run summary |
| `validation-step-XXXXXX.csv`, `train-probe-step-XXXXXX.csv` | Per-example, per-loop predictions, targets, correctness, and final-symbol margins |
| `last_checkpoint.json`, `best_checkpoint.json` | Paths relative to the run directory; checkpoints themselves are directories |
| `step-XXXXXX/recurrent_config.json` | Recurrent architecture, base-model revision, symbol IDs, prompt/loss format; written last as the completion marker |
| `step-XXXXXX/adapter_model.pt` and tokenizer files | Trainable recurrent LoRA tensors and the tokenizer |
| `step-XXXXXX/training_state.pt` | Optimizer, RNG, completed update count, next epoch/offset, best loss, and resume identity |

Checkpoints are saved initially, at the configured cadence, on validation improvement, at epoch boundaries, and at completion. Interrupted runs retain previously completed checkpoints; partially written directories lacking the completion marker cannot be resumed.

Use the **step directory** with the existing evaluator:

```bash
python -m scripts.eval.naive_test --model models/stage1_pointer/<run>/step-000160 --test
python -m scripts.eval.naive_test --model models/stage1_pointer/<run>/step-000160
```

Choose the actual path in `last_checkpoint.json` or `best_checkpoint.json`. Add `--device mps` to evaluate there. Results still go to `eval/pointer_task/<run>/`.

The evaluator recognizes recurrent metadata, restores the frozen base and adapters, and predicts the allowed symbol with highest logit at loop `d` for a depth-`d` question. It uses the saved tokenizer and raw dataset prompt. In a mixed-depth batch it executes `max(d)` loops and reads each example at its own depth; CSV records both counts. Throughput is example-loops/s and questions/s, with zero generated tokens. Prompt, tokenizer, revision, base-model, and generation-budget overrides are rejected for these checkpoints.

These are adapter checkpoints requiring the recorded frozen base in the collaborator's local cache, or `--download`. They are not standalone Hugging Face causal-LM directories or merged ordinary models. Keep the entire step directory when sharing for evaluation. The recurrent readout uses a restricted A–Z vocabulary and raw prompt, so it must be distinguished from the ordinary three-shot unconstrained-generation baseline.

## Resume

Resume from a completed step directory into a new run directory:

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_overfit.json \
  --resume models/stage1_pointer/<run>/step-000080
```

This restores adapters, optimizer, Torch RNG, epoch shuffle position, and completed-step count. The configured epoch/step budget is the total, not an additional budget. To continue a finished run, increase `epochs` or `max_steps` in a copied JSON config. Those fields and reporting/checkpoint cadence may change; model, device, data, tokenization, subset selection, and optimization settings must match the saved identity. The resumed directory only records a new best checkpoint if it beats the previously saved best loss; consult the parent run for an earlier best.

Exact resume was tested on a fixed CPU toy model, including a short final accumulation group and a mid-epoch interruption. Cross-device/version bitwise reproducibility is not claimed. Keep the pinned environment, unchanged code, and original artifacts to reproduce a scientific run.


## Continue the current desktop run

Current priority: improve held-out mapping execution and depth extension before launching the deferred terminal overscaling experiment. We already have instance generalization; broader depth generalization is weak. `configs/stage1_pointer_continue.json` copies the first full-run settings and changes only `epochs` from 1 to 3. Training still uses the same 5,000 depth-1–4 mappings, intermediate supervision, and learning rate. Depths 5–8 stay untrained. If we later train through depth 8, success there would become in-range execution rather than depth extrapolation.

From the configured WSL repository with its complete checkpoints:

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_continue.json --device cuda --dry-run

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_continue.json --device cuda \
  --resume models/stage1_pointer/20260910T220130.926886Z/step-000625
```

The preview loads data/tokenization but no model weights, writes nothing, and is not a full resume-integrity check. Actual resume requires `training_state.pt`, adapter weights, and tokenizer files on the desktop; Git metadata alone is insufficient. The trainer validates saved configuration/data identity when resuming and writes a new run directory.

**Three total epochs** means two additional epochs from update 625: 1,250 additional optimizer updates, ending at update 1,875. This is a bounded pilot budget, not a guarantee of generalization. Do not omit `--resume` unless a fresh three-epoch run is intended. Resume uses the last optimizer state for a continuous training trajectory; it does not reselect update 625 as the historical primary evaluation checkpoint. The learning-rate schedule uses warmup then the existing constant rate; this config does not restart warmup or introduce a new decay schedule.

Use the fixed validation curves during the run, then evaluate the validation file across all depths using `scripts.eval.loop_test --data data/pointer/seed-17/validation.jsonl`. Keep nominal loss, complete trajectories, first failures, and depths 1–4 versus 5–8 separate. The small monitoring subset is only eight examples per depth, so do not rely on it alone to conclude generalization. Checkpoints continue to be selected by the existing trained-depth validation loss rule; do not silently switch selection to test accuracy.

Review this bounded run before adding epochs, changing architecture, or increasing training depth. If in-range execution improves but depth 5–8 stays poor, record that failure and choose the next isolated change. Set measurable acceptance criteria and a confirmation seed before the next evaluation intended to establish a gate. Previously inspected test sets are now development diagnostics for decisions informed by them, not an untouched confirmation set.

The desktop has a recorded tiny exact-resume discrepancy in a toy test; byte-for-byte resume equivalence on CUDA is **not established**. Preserve parent/resumed run metadata and report this limitation. The continuation configuration does not fix that separate issue. No pretrained continuation has been launched by the assistant.

Validation of the continuation config on the Mac: data/tokenizer-only dry run passed with 5,000 training examples, 64 validation examples, 16 probe examples, and 1,875 total planned updates. No checkpoint restoration or pretrained training was performed.


## Loop-balanced loss experiment

The [completed 30k baseline](experiments/stage1_fresh30k.md) used per-example mean CE. The new [loop-balanced config](../configs/stage1_pointer_depth6_loopbalanced.json) changes only `loss_reduction` to `loop_mean` relative to the batch-4 baseline. Start fresh adapters with the same existing seed-37 dataset and training seed 17; no regeneration, `--resume`, or `--init-from` is needed. Training remains depths 1–6, batch 4/accumulation 2, one epoch/3,750 updates, and learning rate 0.0002. Keep the desktop in tmux as described below.

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_loopbalanced.json --device cuda \
  --output models/stage1_pointer/depth6-loopbalanced-seed37 --dry-run

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_loopbalanced.json --device cuda \
  --output models/stage1_pointer/depth6-loopbalanced-seed37
```

For the selected training dataset, let N be the example count, D the maximum depth, and N_t the number of examples reaching loop t. Give each valid CE term weight N/(D*N_t), sum within an example, then average examples. Over the dataset this equals the mean of the D per-loop mean losses. The 30k balanced-depth dataset gives weights `[1/6, 1/5, 1/4, 1/3, 1/2, 1]`: each loop receives 1/6 of the total direct coefficient weight. These weights stay fixed across microbatches and accumulation, including batches without a deep example. Losses beyond an example's depth remain zero. Minibatches estimate this dataset objective; equal loss coefficients do not imply equal gradient norms.

The dashboard shows optimized objective loss separately from legacy example-mean CE. Existing evaluation loss fields retain their meaning for comparison. For this run only, automatic checkpoint selection uses equal-loop mean CE over validation tasks within training depths 1–6; it excludes depth-7/8 examples even from early-loop means. Monitor full trajectories and compare matching update numbers against the baseline. A different objective-selected checkpoint is not by itself proof of improved generalization. Depths 7–16 remain development evaluation; seed 29 stays reserved.

Historical configs default to `example_mean`. Resume cannot change the objective, including with `--allow-batch-change`. The new reduction is recorded in config, run identity, checkpoint metadata, and logs. No explicit loop counter, prompt modification, longer training horizon, or asymmetric retention is introduced. The pretrained experiment remains unrun by the assistant.

## Artifact cleanup

Old recurrent runs and redundant baseline checkpoint directories were removed locally at the user's request. Their tracked evidence remains in Git history; historical report paths can refer to removed files. Git pull will not remove ignored weights on the desktop. With no affected run active, preview and then apply the same bounded cleanup there:

```bash
python -m scripts.training.cleanup_pointer_runs
python -m scripts.training.cleanup_pointer_runs --apply
```

The script targets only named pre-30k runs, their recurrent evaluations, and redundant checkpoints of `depth6-fresh30k-seed37-batch4`. It keeps that run's metrics and evaluations, checkpoints 2250/2500/2750/3000/3250/3750, ordinary-Qwen baseline results, datasets, Stage 0 evidence, and all new or unknown runs. Retained checkpoints include optimizer state for future continuation; deleted directories include both tracked metadata and ignored binaries. This cleanup does not rewrite Git history or remove the shared pretrained-model cache.

## Fresh depth-6 run with 30,000 mappings

The current experiment uses [stage1_pointer_depth6_fresh30k.json](../configs/stage1_pointer_depth6_fresh30k.json): fresh recurrent adapters on the pinned pretrained Qwen base, with no `--init-from` or `--resume`. All original weights stay frozen and intermediate supervision is unchanged. This directly trains depths 1–6 without the earlier depth-4 curriculum; comparison with that curriculum does not isolate data diversity alone.

The user generated the new dataset on the desktop. Its reproduction command is below; run it only when that directory is absent (append `--dry-run 5` to preview without writing):

```bash
python -m scripts.dataset \
  --seed 37 --train-count 30000 --validation-count 600 \
  --test-count 600 --depth-test-count 1000 \
  --min-depth 1 --max-train-depth 6 --max-eval-depth 16 \
  --output data/pointer/seed-37-depth6-30k
```

Training uses all 30,000 new mappings (5,000 per depth). Monitoring uses the existing seed-17 validation file, with 32 examples per depth at 1–8 (256 total), plus 24 training probes. The generator seed is 37; adapter initialization and training order use seed 17. Seed 29 remains reserved. The trainer checks train/validation rule-table overlap before loading weights. The new dataset's own evaluation files are not used by this config.

From the repository root in WSL:

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_fresh30k.json --device cuda \
  --output models/stage1_pointer/depth6-fresh30k-seed37 --dry-run

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_fresh30k.json --device cuda \
  --output models/stage1_pointer/depth6-fresh30k-seed37
```

Budget: one epoch, 3,750 updates, batch 1 with accumulation 8, float32, learning rate 0.0002 with ten warmup updates. Validation and checkpoint saving occur every 250 updates; the Rich dashboard provides progress and ETA. Automatic best-checkpoint selection still uses trained-depth validation loss; inspect trajectory metrics before full evaluation. Evaluate generalization at depths 7–16 separately from trained depths 1–6. Overscaling and shortcut diagnostics remain deferred.

The config passes schema validation. The new dataset is absent on the Mac, so its training dry-run and pretrained training have not been run by the assistant; preview on the desktop before launching.

Startup troubleshooting: the desktop fresh run reported one vocabulary logit just outside the T=1 tolerance (approximately `1.3e-5` absolute difference). The startup reference now projects only the answer position, matching the recurrent readout's matrix shape, with the original tolerance unchanged. Transfer the updated `scripts/training/gates.py` and rerun the same training command. This failure occurs before the run directory is created or any optimizer update. The CUDA retry remains unverified; if the comparison still fails, retain the complete traceback rather than bypassing the gate.

## Resume with larger microbatches

The dashboard's process/host RAM peak is not GPU memory. CUDA runs now display current and peak allocated tensor memory separately; these exclude CUDA context and other non-tensor allocations. Use `nvidia-smi` for device-wide memory usage and utilization. GPU speedup and peak memory for larger batches must be measured on the desktop.

[stage1_pointer_depth6_fresh30k_batch2.json](../configs/stage1_pointer_depth6_fresh30k_batch2.json) changes only batch size to 2 and accumulation to 4. It retains eight examples per optimizer update and the 3,750-update budget. Mixed-depth batches execute to their maximum depth, with shorter examples' extra losses masked; batching can add unused computation, so more occupied memory does not guarantee higher throughput.

[stage1_pointer_depth6_fresh30k_batch4.json](../configs/stage1_pointer_depth6_fresh30k_batch4.json) is the next candidate: batch 4, accumulation 2, with all other settings unchanged. The user reports approximately 5,623 MiB out of 16,303 MiB device memory during early batch-2 training and an ETA near one hour; these are observations, not a completed throughput benchmark. Batch-4 GPU memory and speed remain unmeasured. To switch from batch 2, wait for a completed checkpoint save, retain that run, resolve `last_checkpoint.json` under `models/stage1_pointer/depth6-fresh30k-seed37-batch2`, and use the batch-4 config with `--resume <checkpoint> --allow-batch-change --output models/stage1_pointer/depth6-fresh30k-seed37-batch4`. Compare examples/s over multiple updates at both shallow and deep batches; keep the batch-2 checkpoint available if batch 4 is slower or does not fit.

Changing batch size on resume requires explicit `--allow-batch-change`. It permits only batch/accumulation changes with an unchanged product, while retaining strict dataset, selection, model, seed, optimizer-hyperparameter, validation, and device identities. Optimizer state, RNG state, epoch, next-example offset, and update count are restored. The existing budget/reporting-cadence exceptions still apply. `run.json` records the change and parent checkpoint, and subsequent checkpoints carry the new identity. The objective and example groups per update stay the same, but batching and floating-point accumulation can change the numerical trajectory; this is not bitwise-equivalent resume.

Transfer the updated code and config before switching. Wait until a checkpoint save finishes and training resumes, then interrupt the old process with Ctrl+C. Any work after the last saved checkpoint will be repeated. Keep the old run directory. In the repository root with `.venv` active:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CHECKPOINT=$(python - <<'PY'
import json
from pathlib import Path
run = Path("models/stage1_pointer/depth6-fresh30k-seed37")
print(run / json.loads((run / "last_checkpoint.json").read_text())["path"])
PY
)

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_fresh30k_batch2.json --device cuda \
  --resume "$CHECKPOINT" --allow-batch-change \
  --output models/stage1_pointer/depth6-fresh30k-seed37-batch2
```

The new output directory must not exist. Do not use `--init-from`: that resets optimizer and sample progress. A dry-run still checks data/config/tokenization only; actual resume validates the saved state. Compare examples/s over several updates and watch GPU memory before increasing the batch further. The original batch-1 run can still be resumed from its original checkpoint with its original config if the new batch is slower or does not fit.

### Keep training in tmux on WSL

In a separate Ubuntu terminal, install tmux if needed and start a session:

```bash
sudo apt update && sudo apt install -y tmux
cd ~/loopformer
tmux new -s pointer
```

Inside tmux, activate `.venv` and run the resume command above. To add a GPU monitor, press Ctrl+B, release, then `%`; run `watch -n 1 nvidia-smi` in the new pane. Ctrl+B then `o` switches panes; Ctrl+B then `z` zooms the selected pane. Ctrl+B then `d` detaches while training continues. Reattach with `tmux attach -t pointer`; list sessions with `tmux ls`. Ctrl+B then `[` enters scrollback (press `q` to leave in the default key mode). These are [standard tmux bindings](https://man.openbsd.org/tmux). Ctrl+C interrupts the foreground process in the selected pane; it does not detach. Keep Windows awake and do not shut down WSL. An already-running ordinary terminal process does not automatically move into tmux; resume it from a checkpoint inside the new session.

## Earlier depth stage: adapter-only initialization

The three-epoch depth-4 run and full validation evaluation are complete. The next [depth-6 experiment](depth_generalization.md) expands OOD evaluation through depths 9–16, keeps the depth-4 checkpoint as a paired reference, and reserves seed 29 for later confirmation.

Use `--init-from <step-directory>` to load compatible adapters into a **new** training stage. It is mutually exclusive with `--resume`. Base revision, recurrent architecture, LoRA settings, vocabulary, prompt/loss format, and tokenizer must match; training depth may increase. A decrease in the recorded maximum below source exposure is rejected. Original weights stay frozen and per-loop supervision is unchanged. The new optimizer, warmup schedule, seeded RNG sequence, and counters start afresh; parent optimizer state is not required. Source hashes/depth are recorded in `run.json` and saved recurrent metadata. Subsequent resume of this new stage retains lineage and uses the normal strict identity contract.

Preview validates source metadata but does not load adapter tensors. Actual initialization additionally checks tokenizer identity, hashes the adapter/metadata/tokenizer files, and validates tensor names, shapes, and finite values. Exact commands, budgets, checkpoint selection, OOD comparisons, and confirmation policy live in the [depth experiment guide](depth_generalization.md).

# Train recurrent pointer execution

Status: pointer training and full-loop checkpoint evaluation are implemented. The [fresh 30k baseline](experiments/stage1_fresh30k.md), [loop-balanced ablation](experiments/stage1_loopbalanced.md), and [learned-completion ablation](experiments/stage1_learned_completion.md) have completed. The latter two did not improve the depth frontier. Dated commands below document completed protocols, not current instructions. See [status](status.md) for the latest evidence and [implementation validation](experiments/stage1_training_implementation.md) for earlier toy-model checks.

## Fixed prompt memory: current desktop run

The [new config](../configs/stage1_pointer_depth6_fixed_prompt.json) differs from joint completion only by `recurrence_mode: fixed_prompt`. It retains fresh adapters, 30,000 seed-37 depth-1–6 examples, batch 4/accumulation 2, 3,750 updates, per-pass pointer CE, and completion BCE weight 0.1. See [the architecture and supervision contract](learned_loop_completion.md#fixed-prompt-memory-and-recurrent-working-state). The pretrained result is pending.

From the repository root in WSL/Linux:

```bash
bash train_fixed_prompt.sh --dry-run
bash train_fixed_prompt.sh

# After training completes:
bash eval_ckpts.sh
```

The preview validates existing data/tokenizer and writes nothing. The training launcher requires the existing `.venv`, cached pinned model, generated datasets, and Linux `script` utility (normally installed by util-linux). A PTY preserves the Rich dashboard/ETA while recording stdout/stderr to `models/stage1_pointer/depth6-fixed-prompt-seed37-launch-<UTC>-<pid>.log`. The new run directory is `models/stage1_pointer/depth6-fixed-prompt-seed37/`; an existing directory is an error, never an implicit resume or overwrite. Each run performs the pretrained equivalence/gradient gate before updates. The reference implementation uses dense full-sequence layer kernels; speed or lower memory use has not been measured.

`eval_ckpts.sh` selects `best_checkpoint.json` from that run. It evaluates validation and depth-test with full 20-loop traces (batch 16) and actual head-controlled stopping (cap 20, diagnostic threshold 0.5, batch 1), with failure propagation and one combined `run.log`. Outputs go to a timestamped `eval/pointer_loops/depth6-fixed-prompt-seed37-best-*/` directory. Weights/tokenizers must be on the evaluation device; metadata alone cannot load a checkpoint. Selection remains trained-depth pointer CE, not unseen-depth or stopping performance.

## Learned completion training: completed protocol

The [config](../configs/stage1_pointer_depth6_completion.json) matches the fresh 30k depth-6 batch-4 baseline's data selection, seed, optimizer, effective batch, and update budget. It adds a 128-wide hidden-state stop head and weight `0.1` on the continue/stop loss. The exact intermediate A–Z target is still decoded by frozen C after **every** recurrent pass and supervised by the existing 26-symbol CE. The head receives only the recurrent answer-position state. `Steps` appears in the raw prompt; the trainer's label mask uses task depth, but no loop/depth scalar is passed to R or the head. The first run must use fresh adapters; do not pass `--init-from` or `--resume`.

On the CUDA desktop, from the repository root, first verify that `data/pointer/seed-37-depth6-30k/train.jsonl` and `data/pointer/seed-17/validation.jsonl` are present, then preview and run:

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_completion.json --device cuda \
  --output models/stage1_pointer/depth6-completion-seed37 --dry-run

python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_completion.json --device cuda \
  --output models/stage1_pointer/depth6-completion-seed37
```

The output directory must be new. The live dashboard shows pointer objective/accuracy and completion loss, exact-stop rate, and early-stop rate at the diagnostic probability threshold 0.5. `metrics.jsonl` keeps each update's two loss terms, validation summaries, and selection loss; validation CSVs append stop logits, probabilities, nominal labels, and BCE through each task's depth. These historical checkpoints use `loopformer-stage1-completion-v1` and include both LoRA and head weights in `adapter_model.pt`. `naive_test` and default `loop_test` perform forced-depth pointer evaluation; the latter also records head probabilities. Actual stopping is now available through `--stop-policy completion`, while threshold selection and pretrained stopped results remain pending. The 30k training data are absent from this Mac checkout.

After training, resolve the pointer-CE-selected checkpoint and run full development sweeps on the same validation and depth-test sets used for the baseline. In bash under WSL:

```bash
RUN=models/stage1_pointer/depth6-completion-seed37
BEST=$(python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p/json.loads((p/"best_checkpoint.json").read_text())["path"])' "$RUN")
python -m scripts.eval.loop_test --model "$BEST" \
  --data data/pointer/seed-17/validation.jsonl --device cuda --batch-size 16 \
  --output eval/pointer_loops/completion-best-validation
python -m scripts.eval.loop_test --model "$BEST" \
  --data data/pointer/seed-17/depth_test.jsonl --device cuda --batch-size 16 --loops 20 \
  --output eval/pointer_loops/completion-best-depth-test
```

These commands run the model through the full budget and record every pass. The validation sweep covers requested depths 1–8; the depth-test sweep covers 9–16 and runs to 20 to distinguish a late head signal from no signal within the budget. `summary.json` reports pointer trajectory/final accuracy and the head's 0.5-threshold nominal exact/early/late stop rates. `trajectories.csv` includes per-pass stop probabilities, so thresholds can be studied on development data without rerunning the model. This is an **offline stopping diagnostic**, not measured compute saving or an actual self-stopped answer. Do not use the untouched confirmation split for threshold or checkpoint selection.

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

**Completed and retired.** These commands preserve the protocol, not instructions for another run. See the [results and cleanup](experiments/stage1_loopbalanced.md). The next priority is [diagnosis and profiling](diagnostics_and_performance.md).

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

Historical configs default to `example_mean`. Resume cannot change the objective, including with `--allow-batch-change`. The new reduction is recorded in config, run identity, checkpoint metadata, and logs. No explicit loop counter, prompt modification, longer training horizon, or asymmetric retention is introduced. The user completed the pretrained experiment; its artifacts were audited and subsequently removed at their request.

## Artifact cleanup

For the retired loop-balanced run only, preview with `python -m scripts.training.cleanup_pointer_runs --loopbalanced-only`, then add `--apply` on the desktop to remove the run and six exact evaluation directories, including ignored weights. This cleanup was completed locally; it leaves the baseline and any unknown/new run untouched.

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

The three-epoch depth-4 run and full validation evaluation are complete. The next [depth-6 experiment](training_pointer.md) expands OOD evaluation through depths 9–16, keeps the depth-4 checkpoint as a paired reference, and reserves seed 29 for later confirmation.

Use `--init-from <step-directory>` to load compatible adapters into a **new** training stage. It is mutually exclusive with `--resume`. Base revision, recurrent architecture, LoRA settings, vocabulary, prompt/loss format, and tokenizer must match; training depth may increase. A decrease in the recorded maximum below source exposure is rejected. Original weights stay frozen and per-loop supervision is unchanged. The new optimizer, warmup schedule, seeded RNG sequence, and counters start afresh; parent optimizer state is not required. Source hashes/depth are recorded in `run.json` and saved recurrent metadata. Subsequent resume of this new stage retains lineage and uses the normal strict identity contract.

Preview validates source metadata but does not load adapter tensors. Actual initialization additionally checks tokenizer identity, hashes the adapter/metadata/tokenizer files, and validates tensor names, shapes, and finite values. Exact commands, budgets, checkpoint selection, OOD comparisons, and confirmation policy live in the [depth experiment guide](training_pointer.md).

---

## Stage 1 data and execution contract

The dated milestones below record the original implementation protocol. [Status](status.md) and the loop-balanced section above own the current priority; historical setup language below is not a new instruction to rerun completed experiments.

Status: data milestone implemented and validated on 2026-09-10; pretrained CUDA training and final-answer checkpoint evaluation have run. Full-dataset per-loop evaluations of updates 500 and 625 are complete and audited in the [run report](experiments/stage1_cuda_5k.md). Further training for depth extension is the current priority. Gate 1 is not established. Source: project plan sections 5, 7, and 32. See [current evidence](status.md) and the [data validation report](experiments/stage1_data_validation.md).

### Purpose

Teach one recurrent loop to execute one pointer transition using rules supplied in the prompt. Separate data validation from training so target errors cannot masquerade as model failures.

### Milestone 1: data only

1. Define task records with family, depth, initial state, exact intermediate states, final state, example ID, seed, and rendered prompt.
2. Validate a fixed symbolic vocabulary with the actual tokenizer, including prompt and answer contexts. Store the answer token IDs explicitly.
3. Generate a fresh random mapping and start state per example. Compute targets with an exact reference interpreter independent of model predictions.
4. Define prompt rendering and the answer readout position. Resolve completion, cycles, and repeated-state semantics in [decisions](decisions.md) before finalizing labels.
5. Build reproducible instance and depth splits. Start with training depths at most eight and reserve deeper compositions for evaluation.
6. Test single-token symbols, valid mappings, exact trajectories, deterministic regeneration, and the depth/split boundaries.
7. Stop and report data validation before training.

Randomize mappings so the weights cannot solve examples by memorizing a global symbol-to-symbol lookup. Document split construction and any example-overlap checks.

### Milestone 2: training and evaluation

Implement the Stage 1 training loop using differentiable unrolls and intermediate supervision. For a trajectory `A -> F -> C -> Q`, target `F` after loop 1, `C` after loop 2, and `Q` after loop 3. Do not replace these with the final answer at every loop.

Specify masking for mixed-depth batches and loss reduction explicitly. Choose the cross-entropy vocabulary convention and record it; symbolic evaluation is restricted to the validated answer set regardless. Document any supervision beyond nominal completion separately.

Begin with tiny batches, short prompts, gradient accumulation, and recurrent depths around 4–8. Save reproducible configurations and checkpoints. Build the depth-by-loop evaluator alongside training using [evaluation conventions](evaluation.md).

The implementation is in `scripts/training/`, composed by `python -m scripts.training.train_pointer`. JSON configs live in `configs/`; checkpoint directories and metrics live in `models/`. See [training usage](training_pointer.md) for preview/run/resume commands, the 32-example overfit configuration, the initial depth-1–4 epoch, and exact logging/selection semantics. Loss is 26-symbol CE, averaged over nominal loops per example and then examples, with no post-completion labels. Validation emits per-loop trajectories and a depth-by-loop final-readout matrix.

Evaluate complete datasets from saved checkpoints with `python -m scripts.eval.loop_test`. It reuses training's evaluation path and adds per-example first-error summaries and an exportable depth-by-loop matrix. See [full-loop commands](evaluation.md); no retraining or optimizer state is required.

### Acceptance gate

- Meaningful intermediate accuracy on unseen random mappings.
- Deeper tasks generally require more recurrent computation.
- Additional loops can extend execution, including evaluation beyond trained depths.
- Final-answer accuracy does not conceal shallow shortcuts or broken intermediate execution.

Use the depth-by-loop heatmap and intermediate trajectories to assess the gate. Establish quantitative thresholds, confirmation seeds, and evaluation sizes before a future gate decision. Current measurements are exploratory: trained-depth complete-trajectory accuracy is 86.2%/92.4% at updates 500/625, with weak execution beyond depth 5.

### Implementation record

#### Data generation and verification

Run from the repository root with the [configured environment](setup.md):

All current dataset code lives in `scripts/dataset/`. Preview before choosing to write a full dataset. The [root README](../README.md#pointer-dataset-preview-and-reproduce) provides the complete explicit seed-17 configuration and runtime requirements for reproducing the existing JSONL files byte for byte.

```bash
# Preview five examples, with Rich tables of per-loop targets; no dataset writes.
.venv/bin/python -m scripts.dataset --dry-run
.venv/bin/python -m scripts.dataset --dry-run 10

# Generate the default 13,000 examples and verify the files after writing.
.venv/bin/python -m scripts.dataset --seed 17

# Independently verify an existing dataset, including replay from seeds.
.venv/bin/python -m scripts.dataset --verify data/pointer/seed-17

# Example customization (illustrative; this particular command was not run).
.venv/bin/python -m scripts.dataset --seed 42 --train-count 20000 --max-train-depth 4 --max-eval-depth 12
```

The default output is `data/pointer/seed-<seed>/`; `--output` selects another new directory. Existing directories are never overwritten. `--dry-run [5|10]` selects actual records spread across the available depths, using the same seeds and IDs as a full run with the same configuration. With default settings, the five-example preview shows depths 1, 4, 8, 12, and 16. It prints the full configured count per depth separately; the preview is a depth showcase, not a random frequency sample. If fewer than the requested number of records are configured, all available records are shown. It does not generate the full dataset, create output directories, or write a manifest. All modes use the cached, pinned Qwen tokenizer with `local_files_only=True`; there is no implicit download or model loading.

| File | Default examples | Depths | Purpose |
| --- | ---: | --- | --- |
| `train.jsonl` | 10,000 | 1–8 | Adapter training (initial config filters depths 1–4) |
| `validation.jsonl` | 1,000 | 1–8 | Validation and model selection |
| `test.jsonl` | 1,000 | 1–8 | Unseen-instance evaluation |
| `depth_test.jsonl` | 1,000 | 9–16 | Held-out-depth evaluation |
| `manifest.json` | — | — | Configuration, tokenizer IDs/revision, provenance, counts, checksums |

Counts are configurable with `--train-count`, `--validation-count`, `--test-count`, and `--depth-test-count`. Depth boundaries use `--min-depth`, `--max-train-depth`, and `--max-eval-depth`. Require `1 <= min_depth <= max_train_depth < max_eval_depth <= 25`, with positive split counts. Defaults are an initial data budget, not a measured training requirement or statistically justified gate threshold.

Depths cycle in increasing order within each split; counts per depth differ by at most one. Training explicitly shuffles selected records each epoch. A SHA-256 derivation of master seed, split, and index produces an independent 64-bit example seed; local `random.Random(seed)` samples each table. Increasing a split count preserves its existing prefix and does not change the other splits. Changing depth configuration can change the examples. Python version and generator source hashes are recorded; seeds alone are not a promise of byte identity across future code or runtime changes.

The generator samples `d+1` distinct symbols for the nominal path and fills unused sources with independently random destinations. Every example contains exactly 26 rules (all A–Z sources) in shuffled order. This is a random mapping **conditioned on no repeat along the requested path**, not an unrestricted random-function distribution or a permutation. This avoids early/repeated final-state visits; random unused edges can still create cycles outside the nominal path. Depth is limited to 25 by the 26-symbol pool. Longer unique paths need a separately validated larger vocabulary.

Mapping fingerprints sort all pairs and ignore presentation order, start, and depth. Duplicate tables anywhere within/across generated splits cause an error. Individual edges and shorter subpaths may recur, as expected with a shared finite vocabulary; this is not a compositional-subpath-disjoint split or a cross-family test.

#### Prompt, tokens, and record schema

The raw-text prompt has this format (abbreviated table):

```text
Rules: ( A, C) ( C, D) ( D, E)
Start: A
Steps: 2
Answer:
```

The actual prompt includes all 26 shuffled pairs. No intermediate states or final answer appear as extra labels in the prompt. No chat template or special tokens are added. The leading space **inside** each pair is intentional: compact `(A,C)` can merge letters with punctuation in Qwen's tokenizer. Each symbolic token encodes `" " + symbol`, e.g. `" A"`; the logical state remains `"A"`. The prompt ends at `Answer:` without a trailing space, and a valid answer continuation is `" D"`. Validation covers all 676 source/target contexts, all start/answer symbols, and every generated prompt's exact symbol spans.

Each JSONL line stores:

- Identity: `schema_version=1`, `example_id`, `family="pointer"`, `split`, and per-example `seed`.
- Program: `mapping` as a list of two-symbol lists in prompt order, `mapping_sha256`, `initial_state`, and `task_depth` (the explicit requested step count).
- Targets: `intermediate_states[t-1]` is the exact state after transition `t`, for `1 <= t <= task_depth`; `final_state` is the last target. `intermediate_token_ids` and `final_token_id` are their actual tokenizer IDs.
- Input: `prompt`, `prompt_token_count`, and zero-based `answer_position`, the last input token's index. This is a next-token readout position for an unpadded example; a future batching layer must adjust it for left padding.

`h_0` is still an architectural hidden state; no assertion or label requires the frozen coda to decode it to `initial_state`. There is no target after the requested depth and no absorbing-terminal requirement. If another pointer transition leaves the final answer, that alone is not evidence of model damage. Post-completion semantics must be settled separately before Stage 2/3 retention analysis.

#### Library interfaces

- [pointer.py](../scripts/dataset/pointer.py): `generate_example(seed, depth, split, index)`, `parse_mapping(text)`, `execute(mapping, start, steps)`, `validate_example(example)`, and `check_predictions(mapping, start, predictions, steps)`.
- [symbols.py](../scripts/dataset/symbols.py): `validate_symbols(tokenizer)` returns the logical-symbol-to-token-ID table; `validate_prompt_tokens(...)` checks exact spans and returns the answer readout index.
- [dataset.py](../scripts/dataset/dataset.py): `DatasetConfig`, deterministic split generation, JSONL/manifest writing, and `verify_dataset(path, tokenizer)`.
- [CLI](../scripts/dataset/cli.py): argument parsing, explicit cached tokenizer loading, Rich preview/summary, and provenance capture.

The reference parser also accepts the user's compact format, independent of model tokenization:

```python
from scripts.dataset.pointer import parse_mapping, execute, check_predictions

rules = parse_mapping("(A,C) (C,D) (D, E)")
assert execute(rules, start="A", steps=2) == ["C", "D"]
assert check_predictions(rules, "A", [" C", "D"], steps=2) == [True, True]
assert check_predictions(rules, "A", ["D", "E"], steps=2) == [False, False]
```

The checker accepts a prediction prefix, strips surrounding whitespace, and requires an exact symbol rather than extracting an answer from prose. Predictions beyond nominal depth are rejected. Its general reference interpreter supports partial tables and cycles, while the dataset sampler enforces full tables and non-repeating nominal paths.

Verification checks file hashes, counts and depth histograms, unique mappings, prompt/table consistency, all labels by reparsing and executing the prompt, actual token contexts, and exact per-record seed replay. Source hashes and runtime provenance live in the manifest; generated data is excluded from Git by `data/` in `.gitignore`. The [report](experiments/stage1_data_validation.md) records the default artifact hashes as durable evidence.

The separate [ordinary-model final-answer baseline](evaluation.md) is implemented and tested with toy models; the user's first full three-shot run reached 6.00% accuracy and is documented in the [baseline report](experiments/naive_pointer_baseline.md). It shares the dataset and uses `prompts/pointer_task.txt` instructions, without changing the nominal targets or introducing recurrent training. Training code, resumable adapter checkpoints, and per-loop validation are implemented and tested on random tiny models. Pretrained CUDA training and checkpoint final-answer results are recorded in the [5,000-mapping report](experiments/stage1_cuda_5k.md); full-test per-loop evaluation is now complete and audited. The depth-1–4 continuation is historical; [status](status.md) identifies the current bounded experiment. Terminal overscaling execution remains deferred. Passing data validation or final-answer baseline tests does not establish Gate 1.


### Current training experiment

The fresh 30k baseline and full evaluations are complete; see [results](experiments/stage1_fresh30k.md). The equal-loop CE ablation is now complete and retired; see its [report](experiments/stage1_loopbalanced.md). The current priority is failure diagnostics and profiling, not another training run.

The subsequent fresh-adapter 30k run is complete; its [report](experiments/stage1_fresh30k.md) supersedes the setup notes for that run.

### Earlier depth extension

The depth-4 continuation reached update 1875 and full validation now shows 99.4% trained-depth complete trajectories, 82.4% at depth 5, and 20.8% at depth 6. The subsequent depth-6 adapter-only initialization and outward OOD comparison are documented below. It keeps intermediate supervision, evaluates the depth-4 reference on the same data, reports extrapolation distance, and reserves a new confirmation seed. Extending the training range does not turn depth-5/6 results into OOD evidence. Overscaling execution stays deferred.

---

## Historical depth-6 curriculum and paired evaluation

These commands describe the completed curriculum comparison. The current priority is [diagnosis and profiling](diagnostics_and_performance.md); see [status](status.md).

Status: this document records the earlier depth-6 curriculum setup and paired-evaluation protocol. The later [fresh 30k depth-6 baseline and full evaluations](experiments/stage1_fresh30k.md) are complete; the loop-balanced ablation is also complete. See [status](status.md) for the latest evidence. Commands and resource expectations below describe that earlier protocol, not the current run instructions. Overscaling and knowledge-retention benchmarks remain deferred.

### Question and comparison

The depth-4 model at update 1875 achieves 99.4% complete trajectories at validation depths 1–4, 82.4% at depth 5, 20.8% at depth 6, 1.6% at depth 7, and 0% at depth 8. The historical comparison tested whether training through depth 6 supported execution farther beyond training, rather than only solving depths newly included in training.

| Role | Training exposure | In-range evaluation | Nearby OOD | Farther evaluation |
| --- | --- | --- | --- | --- |
| Reference | Depths 1–4, three epochs | 1–4 | 5–6 | 7–16 |
| New stage | Reference adapters plus depths 1–6 | 1–6 | 7–8 | 9–16 |

Both checkpoints are evaluated on the **same examples at every absolute depth**, using the existing seed-17 `validation.jsonl` (1–8) and `depth_test.jsonl` (9–16). These become development diagnostics when used to decide further training. A separate seeded confirmation dataset remains reserved for after the setup is frozen.

Report absolute depth and `steps_beyond_training = task_depth - train_max_depth`. Compare equal positive offsets too: reference depth 6 versus candidate depth 8 are both +2. Equal offsets involve different-depth tasks, not paired examples. Improvements at candidate depths 5–6 are in-range results, not OOD gains.

The reference is the existing checkpoint, not an equal-compute control. The candidate receives additional training examples, updates, and recurrent computation, and resets its optimizer. This is a curriculum extension experiment; it does not isolate training depth as the sole cause. Record the extra budget rather than attributing every improvement to depth alone.

### Existing data: no regeneration required

The seed-17 files already contain the required depths. `train.jsonl` has 1,250 examples per depth at 1–8; `validation.jsonl` and `test.jsonl` each have 125 per depth at 1–8; `depth_test.jsonl` has 125 per depth at 9–16. A local read-only inventory confirmed these counts and all four manifest hashes. No examples or seeds were changed.

The new training config deliberately keeps `train_max_depth: 6` and `validation_max_depth: 8`. The latter controls the small training-time monitoring subset, not the paired evaluator's maximum depth. Do not change it to 16 while pointing at the existing depth-1–8 validation file. After training, `depth_generalization` defaults to both validation and deeper files; each `loop_test` child automatically uses its file's maximum depth (8 or 16).

To verify your local desktop copy without generating data:

```bash
python -m scripts.dataset --verify data/pointer/seed-17
```

If the dataset is absent on a new machine, use the [README's complete seed-17 reproduction command](../README.md#pointer-dataset-preview-and-reproduce). Existing output directories are never overwritten. Seed 29 remains reserved; do not regenerate with a different seed merely to enable depth-16 evaluation.

### Preview, then train on the desktop

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

### Evaluate both models through depth 16

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

### Reserved confirmation set

Reserve master seed **29** and do not generate, inspect, or evaluate its questions during this development cycle. After selecting training settings and checkpoints and fixing the claims/metrics, generate the independent dataset with the pinned tokenizer:

```bash
python -m scripts.dataset --seed 29 \
  --train-count 10000 --validation-count 1000 --test-count 1000 --depth-test-count 1000 \
  --min-depth 1 --max-train-depth 8 --max-eval-depth 16 \
  --output data/pointer/seed-29
```

Use only `test.jsonl` and `depth_test.jsonl` for that final assessment, evaluating both frozen checkpoints with repeated `--data` arguments. The generator also creates train/validation files; do not train on them as part of the confirmation experiment. Check exact whole-table overlap with development/training data before interpreting confirmation. A seed alone is not a substitute for overlap checks or a predeclared protocol. If confirmation informs another training change, reserve another untouched set rather than continuing to call seed 29 final confirmation.

### Validation

Implementation tests cover initialization compatibility and exposure tracking, preservation of loaded tensors before the first update, a new optimizer/counter without parent optimizer files, subsequent resume of the initialized stage, matching-dataset comparison, relative-depth classification, and a plan-only evaluation preview. No pretrained depth-6 training, OOD sweep, or seed-29 generation was launched by the assistant. The Mac full suite passed 94 tests in 64.94 seconds; the nine focused cases passed in 27.96 seconds. The real-data depth-6 preview, CLI help, and local documentation links passed. See [status](status.md).

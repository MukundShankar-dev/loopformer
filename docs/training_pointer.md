# Train recurrent pointer execution

Status: implemented and tested with random tiny Qwen models; pretrained CUDA training and resource measurements are recorded in the [5,000-mapping report](experiments/stage1_cuda_5k.md). Full-test intermediate execution remains to be measured with the [loop evaluator](loop_pointer_eval.md). See [implementation validation](experiments/stage1_training_implementation.md) for the checks performed. The [Stage 1 guide](phases/stage1_pointer.md) owns the research objective and acceptance gate.

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

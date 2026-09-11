# Full-loop pointer evaluation

Evaluate a saved recurrent checkpoint without training or loading optimizer state. The CLI reuses the trainer's per-loop evaluation and exact nominal targets. It records every frozen-coda readout; predictions are never fed back as input tokens. This measures latent recurrent execution, not a generated explanation.

## Run

From the repository root on the configured desktop:

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Inspect three examples through eight loops first.
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 \
  --device cuda --loops 8 --test

# Full test set, both previously identified checkpoints; separate outputs.
for step in 000500 000625; do
  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/20260910T220130.926886Z/step-$step" \
    --device cuda --loops 8
done
```

Use the actual **step directory**, including `adapter_model.pt`, `recurrent_config.json`, and saved tokenizer files. The metadata-only Git checkout cannot reconstruct the trained model. Run on the desktop where the complete checkpoints remain, or copy them separately. The pinned frozen base loads from the local cache; `--download` explicitly permits a missing base download. `training_state.pt` is not required. No checkpoint is modified.

Default input is all 1,000 examples in `data/pointer/seed-17/test.jsonl`. `--data` selects another file, including `depth_test.jsonl`. `--loops` defaults to the deepest selected task and applies equally to every example. It must cover every selected nominal target: an insufficient budget is rejected, not truncated. `--limit N` selects the first N examples; `--test [2|3]` selects a small prefix and prints prompts and per-loop RIGHT/WRONG decisions. Both use the same scoring path and write normal result files. To compare a preview with the full run, specify the same loop budget explicitly.

Defaults are CPU, float32, batch 1, seed 17, four CPU threads, eager attention, and strict deterministic algorithms. Use `--device cuda` or `--device mps` explicitly; there is no fallback. A compact Rich progress bar shows ETA, questions/s, and executed example-loops/s. Timing includes evaluation, scoring, and trajectory CSV writing, but excludes checkpoint loading, input encoding, and final diagnostic exports. There is no text generation or tokens/s measurement.

## Artifacts

Each run writes a new directory under `eval/pointer_loops/<timestamp>-<training-run>-<step>/`. `--output` selects another new directory; existing directories are rejected. The repository's current ignore rules allow these small CSV/JSON artifacts to be committed.

| File | Contents |
| --- | --- |
| `trajectories.csv` | One row per example per loop: ID, split, seed, family, depth, initial state, prediction, intermediate and final targets/correctness, intermediate CE, intermediate/final raw-logit margins, and post-completion flag |
| `examples.csv` | One row per example: target and predicted sequences, complete-trajectory correctness, nominal final correctness, first erroneous loop, correct-prefix length, first loop matching the final answer, and repeated adjacent nominal predictions |
| `depth_by_loop.csv` | Task depth × loop matrix in long form: counts, final accuracy, and nominal intermediate accuracy/loss |
| `summary.json` | Overall and per-depth/per-loop metrics, diagnostic counts, model/checkpoint/data/source hashes, command, settings, package versions, Git provenance, timing, and completion status |

`summary.json` begins with `status: running` and is marked `complete` only after all exports finish. Interrupted runs are incomplete and cannot be resumed; choose a new output directory. Trajectory CSV writing occurs after the evaluation sweep. No model weights, optimizer state, hidden-state tensors, or full vocabulary logits are exported.

## Interpretation

- Loop t is scored against the exact state after t transitions for **t <= task depth**. Complete-trajectory accuracy requires every nominal step to be correct, even if the final answer is right after an earlier error.
- First-error indices are 1-based and blank for perfect trajectories. Correct-prefix length ends at the first error; recovery cannot lengthen it. First-final-correct indices search the full observed sweep and are blank if never correct; an early final match does not prove correct execution.
- Loss is 26-symbol CE, averaged across nominal loops within each example, then across examples. Per-loop loss/accuracy use only examples whose depth reaches that loop; their denominators differ. These are the same definitions as training validation.
- A margin is the target symbol's raw logit minus the largest other allowed-symbol logit. A–Z argmax uses the first symbol for ties; a correct argmax can therefore have zero margin.
- All examples are swept through the same loop count, so the final-target depth × loop matrix includes post-completion observations. Intermediate target/loss/margin/correctness are blank after completion and excluded from supervised metrics. This does not define a retention target or classify valid continuing pointer moves as damage.
- Repeated adjacent predictions can indicate a stalled decoded answer, but do not establish frozen hidden states. The count considers adjacent nominal loops only, where the reference path does not repeat.

For the original one-epoch experiment, update 500 remains the validation-loss-selected primary checkpoint. Update 625 is its already inspected secondary comparison; do not silently reselect a primary checkpoint using test performance. Separate depths 1–4 (trained), 5–8 (untrained depths already monitored during training), and the deeper test file. Both historical checkpoints have now been evaluated and audited in the [run report](experiments/stage1_cuda_5k.md). Gate 1 still requires explicit empirical criteria; the CLI does not declare a gate passed. The next priority is [further training](training_pointer.md#continue-the-current-desktop-run), not repeating these completed sweeps.

## Validation

Tests use prescribed logits and a random tiny Qwen checkpoint with a local synthetic tokenizer. They cover first-error/recovery semantics, nominal masking, margins, mixed depths/padding, checkpoint loading without optimizer state, agreement with `naive_test` final readouts, exports, inspection mode, and rejection of insufficient loops or existing output paths. The user subsequently completed pretrained full-loop execution on CUDA for both checkpoints; see the [audited report](experiments/stage1_cuda_5k.md). Implementation tests alone make no performance or gate claim.

Validation on the Mac: `.venv/bin/python -m pytest tests/test_loop_eval.py tests/test_pointer_training.py -q` passed all nine focused cases; `.venv/bin/python -m pytest -q` passed all 89 cases in 45.95 seconds. The CLI help command passed. No pretrained checkpoint evaluation was launched by the assistant; the user-run evidence is recorded separately.


For later overscaling experiments with a well-defined terminal state, use the separate [terminal evaluator](overscaling_eval.md). The ordinary `loop_test` command and existing dataset retain their nominal-only semantics. Terminal execution is currently deferred in favor of further Stage 1 training.

For the current outward-depth experiment, update 1875 from the completed depth-4 continuation is the reference. The [paired depth evaluator](depth_generalization.md#evaluate-both-models-through-depth-16) evaluates it and the new depth-6 checkpoint on identical files and exports absolute and relative depth metrics.

# Stage 1 training implementation validation

Date: 2026-09-10. This is software validation using random tiny Qwen models, not a pretrained pointer-learning result.

## Implemented behavior

The [training guide](../training_pointer.md) owns commands, configurations, loss semantics, monitoring subsets, and artifacts. The implementation trains only shared recurrent LoRA with per-loop symbol CE, saves adapters/tokenizer/optimizer/RNG/cursor state under `models/`, and supports checkpoint evaluation through the existing naive-test CLI. No pretrained training was launched.

## Commands and results

Run from the repository root:

```bash
.venv/bin/python -m scripts.training.train_pointer --config configs/stage1_pointer_overfit.json --dry-run
.venv/bin/python -m scripts.training.train_pointer --dry-run
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

Both dry runs passed against the existing seed-17 data and cached pinned Qwen tokenizer. They wrote no output and loaded no model weights. The overfit preview reported 32 training, 16 validation, and 32 train-probe examples with 160 planned updates. The larger preview reported 5,000 training, 64 validation, and 16 train-probe examples with 625 planned updates.

The complete suite passed **87 tests in 34.70 seconds**. `pip check` found no broken requirements. Local Markdown file links and `git diff --check` passed.

The seven new cases in [test_pointer_training.py](../../tests/test_pointer_training.py) check:

- Exact CE weighting across unequal depths, zero masked gradients, unequal microbatch accumulation, and metric denominators.
- Deterministic balanced subset selection and invalid configuration rejection.
- Startup equivalence and two-loop gradients, frozen-weight preservation after an optimizer step, exact checkpoint-logit restoration, and per-example depth readout in mixed batches.
- Exclusion of post-completion steps from supervised metrics, while retaining extra-loop observations in CSV.
- Exact CPU adapter and validation-metric agreement between an uninterrupted six-update run and a run resumed after update two, including short final accumulation groups and epoch boundaries.
- Recurrent checkpoint loading, raw inputs, restricted-symbol scoring, and CSV/JSON output through `scripts.eval.naive_test --test`.
- The complete training CLI: data/tokenizer dry run without output creation, a toy optimizer update, checkpoint save, and resume through the next epoch boundary. This case uses the real cached tokenizer with a random three-layer, hidden-size-16 model, not pretrained Qwen weights.

Toy checkpoints and logs were written only inside pytest temporary directories. They are disposable implementation evidence, not checkpoints to use for the research experiment. No new research run was written under `models/`.

## Limits and next step

The existing pretrained Stage 0 CPU/MPS architecture gate is separate evidence. These new checks establish CPU implementation behavior, not pretrained training success or MPS training memory. Use the small overfit configuration first, inspect its measured resource use and fixed training-trajectory accuracy, and choose empirical Gate 1 criteria before interpreting held-out results. A one-epoch budget is not an assertion of sufficient learning. Stage 2/3 repair, damage, and retention remain unimplemented and require settled post-completion semantics.

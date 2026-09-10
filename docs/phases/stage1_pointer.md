# Stage 1: pointer data and stepwise execution

Status: data milestone implemented and validated on 2026-09-10; pretrained CUDA training and final-answer checkpoint evaluation have run. The [full-loop CLI](../loop_pointer_eval.md) is implemented; pretrained full-dataset per-loop evaluation remains pending. Gate 1 is not established. Source: project plan sections 5, 7, and 32. See [current evidence](../status.md) and the [data validation report](../experiments/stage1_data_validation.md).

## Purpose

Teach one recurrent loop to execute one pointer transition using rules supplied in the prompt. Separate data validation from training so target errors cannot masquerade as model failures.

## Milestone 1: data only

1. Define task records with family, depth, initial state, exact intermediate states, final state, example ID, seed, and rendered prompt.
2. Validate a fixed symbolic vocabulary with the actual tokenizer, including prompt and answer contexts. Store the answer token IDs explicitly.
3. Generate a fresh random mapping and start state per example. Compute targets with an exact reference interpreter independent of model predictions.
4. Define prompt rendering and the answer readout position. Resolve completion, cycles, and repeated-state semantics in [decisions](../decisions.md) before finalizing labels.
5. Build reproducible instance and depth splits. Start with training depths at most eight and reserve deeper compositions for evaluation.
6. Test single-token symbols, valid mappings, exact trajectories, deterministic regeneration, and the depth/split boundaries.
7. Stop and report data validation before training.

Randomize mappings so the weights cannot solve examples by memorizing a global symbol-to-symbol lookup. Document split construction and any example-overlap checks.

## Milestone 2: training and evaluation

Implement the Stage 1 training loop using differentiable unrolls and intermediate supervision. For a trajectory `A -> F -> C -> Q`, target `F` after loop 1, `C` after loop 2, and `Q` after loop 3. Do not replace these with the final answer at every loop.

Specify masking for mixed-depth batches and loss reduction explicitly. Choose the cross-entropy vocabulary convention and record it; symbolic evaluation is restricted to the validated answer set regardless. Document any supervision beyond nominal completion separately.

Begin with tiny batches, short prompts, gradient accumulation, and recurrent depths around 4–8. Save reproducible configurations and checkpoints. Build the depth-by-loop evaluator alongside training using [evaluation conventions](../evaluation.md).

The implementation is in `scripts/training/`, composed by `python -m scripts.training.train_pointer`. JSON configs live in `configs/`; checkpoint directories and metrics live in `models/`. See [training usage](../training_pointer.md) for preview/run/resume commands, the 32-example overfit configuration, the initial depth-1–4 epoch, and exact logging/selection semantics. Loss is 26-symbol CE, averaged over nominal loops per example and then examples, with no post-completion labels. Validation emits per-loop trajectories and a depth-by-loop final-readout matrix.

Evaluate complete datasets from saved checkpoints with `python -m scripts.eval.loop_test`. It reuses training's evaluation path and adds per-example first-error summaries and an exportable depth-by-loop matrix. See [full-loop commands](../loop_pointer_eval.md); no retraining or optimizer state is required.

## Acceptance gate

- Meaningful intermediate accuracy on unseen random mappings.
- Deeper tasks generally require more recurrent computation.
- Additional loops can extend execution, including evaluation beyond trained depths.
- Final-answer accuracy does not conceal shallow shortcuts or broken intermediate execution.

Use the depth-by-loop heatmap and intermediate trajectories to assess the gate. Establish quantitative thresholds and evaluation sizes before judging success; none have been measured yet.

## Implementation record

### Data generation and verification

Run from the repository root with the [configured environment](../setup.md):

All current dataset code lives in `scripts/dataset/`. Preview before choosing to write a full dataset. The [root README](../../README.md#pointer-dataset-preview-and-reproduce) provides the complete explicit seed-17 configuration and runtime requirements for reproducing the existing JSONL files byte for byte.

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

### Prompt, tokens, and record schema

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

### Library interfaces

- [pointer.py](../../scripts/dataset/pointer.py): `generate_example(seed, depth, split, index)`, `parse_mapping(text)`, `execute(mapping, start, steps)`, `validate_example(example)`, and `check_predictions(mapping, start, predictions, steps)`.
- [symbols.py](../../scripts/dataset/symbols.py): `validate_symbols(tokenizer)` returns the logical-symbol-to-token-ID table; `validate_prompt_tokens(...)` checks exact spans and returns the answer readout index.
- [dataset.py](../../scripts/dataset/dataset.py): `DatasetConfig`, deterministic split generation, JSONL/manifest writing, and `verify_dataset(path, tokenizer)`.
- [CLI](../../scripts/dataset/cli.py): argument parsing, explicit cached tokenizer loading, Rich preview/summary, and provenance capture.

The reference parser also accepts the user's compact format, independent of model tokenization:

```python
from scripts.dataset.pointer import parse_mapping, execute, check_predictions

rules = parse_mapping("(A,C) (C,D) (D, E)")
assert execute(rules, start="A", steps=2) == ["C", "D"]
assert check_predictions(rules, "A", [" C", "D"], steps=2) == [True, True]
assert check_predictions(rules, "A", ["D", "E"], steps=2) == [False, False]
```

The checker accepts a prediction prefix, strips surrounding whitespace, and requires an exact symbol rather than extracting an answer from prose. Predictions beyond nominal depth are rejected. Its general reference interpreter supports partial tables and cycles, while the dataset sampler enforces full tables and non-repeating nominal paths.

Verification checks file hashes, counts and depth histograms, unique mappings, prompt/table consistency, all labels by reparsing and executing the prompt, actual token contexts, and exact per-record seed replay. Source hashes and runtime provenance live in the manifest; generated data is excluded from Git by `data/` in `.gitignore`. The [report](../experiments/stage1_data_validation.md) records the default artifact hashes as durable evidence.

The separate [ordinary-model final-answer baseline](../naive_pointer_eval.md) is implemented and tested with toy models; the user's first full three-shot run reached 6.00% accuracy and is documented in the [baseline report](../experiments/naive_pointer_baseline.md). It shares the dataset and uses `prompts/pointer_task.txt` instructions, without changing the nominal targets or introducing recurrent training. Training code, resumable adapter checkpoints, and per-loop validation are implemented and tested on random tiny models. Pretrained CUDA training and checkpoint final-answer results are recorded in the [5,000-mapping report](../experiments/stage1_cuda_5k.md); full-test per-loop evaluation is pending. Passing data validation or final-answer baseline tests does not establish Gate 1.

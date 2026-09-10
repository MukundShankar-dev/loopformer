# Ordinary-model pointer evaluation

Status: implemented and tested with local random toy models. The user completed a full three-shot pretrained run with **6.00% accuracy (60/1,000)**; the saved results were audited without rerunning inference. See the [baseline report](experiments/naive_pointer_baseline.md) for evidence and limitations. This is a final-answer baseline, separate from recurrent execution and training.

## Usage

From the repository root with the configured environment and generated dataset:

```bash
# Inspect three actual model inputs, responses, and right/wrong decisions.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test

# Inspect two instead.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test 2

# All 1,000 same-depth test examples, default CPU / float32 / batch 1.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct

# Short first run on Apple Silicon, if MPS is available.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --device mps --limit 8

# Held-out depths 9–16.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --data data/pointer/seed-17/depth_test.jsonl

# A directory written with Hugging Face save_pretrained, including its tokenizer.
.venv/bin/python -m scripts.eval.naive_test --model path/to/saved-model

# A full PyTorch state dict, using the specified base for architecture/tokenizer.
.venv/bin/python -m scripts.eval.naive_test --model path/to/weights.pt --base-model Qwen/Qwen2.5-0.5B-Instruct

.venv/bin/python -m scripts.eval.naive_test --help
```

These commands are for user execution; the completed default full-test run is documented in the [baseline report](experiments/naive_pointer_baseline.md). Other configurations shown here are usage examples, not measured results. `--limit N` selects the first N rows; the generated file cycles through depths, so a tiny prefix need not cover every depth. The summary reports the depths actually evaluated.

`--test` runs only the first three examples; `--test 2` runs two (or all available if the file is smaller). It prints the exact input passed to the tokenizer, including chat formatting, the raw decoded response as a Python `repr` so whitespace is visible, the expected symbol, parsed symbol or format failure, RIGHT/WRONG, stop reason, and generated token IDs. It uses the same generation/scoring/output path as the full evaluator and writes the normal artifacts with `test_mode: true`; default output directory names include `test-`. `--test` and `--limit` are mutually exclusive. Test mode is an actual model run, not a dry run.

`--model` supports a Hugging Face repository ID, a local full-model directory, or a standalone `.safetensors`, `.pt`, `.pth`, or `.bin` file. PyTorch files must contain a tensor state dict or `{"state_dict": tensor_dict}`; full pickled model objects are not loaded. A standalone file requires `--base-model` unless its directory contains `config.json` and a usable tokenizer. `--tokenizer` can override the tokenizer source. Missing/unexpected model weights fail; tied parameter aliases may be omitted in standalone files when their shared tensor is present. A single shard is not a complete checkpoint. Adapter-only checkpoints and the custom recurrent wrapper are not supported by this ordinary-model evaluator.

Loading uses cached/local files unless `--download` is explicitly supplied. The default revision for the project's Qwen model is `7ae557604adf67be50417f59c2c2f167def9a775`; other Hub IDs default to `main`. `--revision` selects a different model/base revision. The resolved model commit is saved. An overridden tokenizer uses its own source's default revision; use a local tokenizer snapshot to pin a separate tokenizer exactly.

`--device cpu|mps|cuda` and `--dtype float32|float16|bfloat16` are explicit; there is no fallback. Defaults are CPU and float32. Set `--batch-size` to change the default batch of one. Left padding is used, and tokenization uses the evaluated model's tokenizer rather than the token IDs stored in the Qwen dataset. Unsupported device/dtype operations fail rather than silently switching settings.

## Prompt and score

The shared [prompt template](../prompts/pointer_task.txt) explains pointer pairs, asks for exactly the specified number of transitions, and requests only a final uppercase letter. It includes three hand-written demonstrations at depths 1, 2, and 3, each with an independent shuffled six-symbol rule table and a single-letter final answer. Their exact paths are A→F, A→D→E, and E→B→A→C; only final answers appear in the demonstrations. They are not selected from the test set. The actual task follows a clear boundary and uses its own rules.

Placeholders are limited to `{rules}`, `{start}`, and `{steps}`; the evaluated example's final/intermediate targets are never supplied to the template. `--prompt FILE` selects another template with that same interface. This three-shot prompt replaces the initial zero-shot instructions after the user's 0/3 report; compare runs using the saved prompt text/hash, since they represent different prompting conditions. The full three-shot run reached 6.00%; no paired full zero-shot run was collected, so an improvement from demonstrations has not been established.

Default `--prompt-format chat` wraps the task text in one user message with the model tokenizer's chat template and generation prefix. For a model without a chat template, explicitly choose `--prompt-format raw`. The task text stays shared; the chat wrapper is model-specific. This adds natural-language task instructions to the dataset's bare rule table and is not a silent change to the future recurrent training prompt.

Generation is greedy, one beam, with no answer-vocabulary restriction, `use_cache=False`, strict deterministic algorithms, seed 17, four CPU threads, and an eight-token output budget by default. `--max-new-tokens` changes the budget. Token outputs are ordinary autoregressive generation, not recurrent loops. Inputs are never truncated; a prompt plus output budget exceeding the model's declared context limit fails explicitly.

Decode only newly generated tokens, skip special tokens, and strip surrounding whitespace. A prediction must equal the expected single uppercase symbol to count as correct. For target D, `" D\n"` passes; `"d"`, `"D."`, `"Answer: D"`, empty text, and prose fail. Invalid answers stay in the accuracy denominator. The CSV preserves the response, and the summary separately counts invalid answers and token-budget stops. A valid answer at the token limit can still be correct; the stop reason remains visible.

This strict score measures both task correctness and requested-format compliance. It does not implement lenient answer extraction or compare only A–Z logits. Comparisons with a later restricted-logit recurrent readout must label that distinction.

## Outputs and timing

Default output: `eval/pointer_task/<UTC timestamp>-<model>/`, rooted at the repository even when invoked from another working directory. `--output DIRECTORY` selects another new directory. Existing output directories are refused. Generated evaluation artifacts are excluded from Git.

- `predictions.csv`: example ID, split, seed, task depth, target, raw response, parsed prediction, valid/correct flags, prompt/generated token counts, stop reason, generated token IDs, rendered task prompt, and complete model input.
- `summary.json`: run configuration, prompt/data/source hashes, model and tokenizer identity, local checkpoint hashes when applicable, package versions, overall and per-depth accuracy, invalid-answer/token-limit counts, timings, and throughput.

CSV rows flush after each batch. The summary initially has `status: running` and changes to `complete` only after all selected examples finish. If interrupted or failed, partial CSV rows and a running summary are not a completed evaluation; start a new output directory to rerun. There is no resume mode.

Rich displays a progress bar, completed questions, elapsed/remaining time, running accuracy, generated tokens/s, and questions/s. Normal runs save individual prompts and responses only to CSV; `--test` also prints them for inspection.

Generated tokens/s divides generated token count by synchronized `model.generate` time, including prompt prefill. Generated token counts include the first EOS but exclude subsequent batch padding. Questions/s uses evaluation-loop wall time including tokenization, decoding, CSV writes, and progress updates, but excludes loading and initial metadata hashing. These are measured batch-run rates, not standalone decode latency. Memory use remains unmeasured; the first CPU pretrained throughput measurements are in the [baseline report](experiments/naive_pointer_baseline.md).

## Stepwise testing later

The dataset already stores the state after every pointer transition. A recurrent evaluator can read the frozen coda after loop t and compare its decoded symbol against `intermediate_states[t-1]`. That measures whether one recurrent loop corresponds to one transition and belongs to Stage 1's recurrent evaluator.

Ordinary Qwen could instead be prompted to print a sequence of intermediate symbols, but that would measure a generated textual trace, not internal recurrent depth. Neither stepwise baseline is implemented in this change. Final-answer accuracy alone does not pass the Stage 1 mechanism gate or establish repair/damage dynamics.

## Implementation and checks

- [naive_test.py](../scripts/eval/naive_test.py): CLI, explicit configuration, progress, CSV and JSON output.
- [pointer_task.py](../scripts/eval/pointer_task.py): dataset verification, input-only prompting, generation batches, scoring, and metrics.
- [loading.py](../scripts/eval/loading.py): Hub/directory/state-file model loading with explicit tokenizer, device, and dtype.
- [Focused tests](../tests/test_naive_pointer_eval.py): strict scoring, prompt field restrictions, target validation, EOS/padding token counts, per-depth denominators, batch continuation slicing, saved-directory/PyTorch/Safetensors weight fidelity, partial-state rejection, and CLI output/overwrite behavior using local random toy models.

Focused validation: `.venv/bin/python -m pytest tests/test_naive_pointer_eval.py -q` passed all 17 cases. CLI tests include normal mode and both `--test` sizes, asserting that the printed RIGHT/WRONG counts agree with the persisted summary. No pretrained checkpoint was loaded during these checks.

Final regression check: `.venv/bin/python -m pytest -q` passed all 80 tests in 19.33 seconds, with no skips.

No new dependencies are required. The [first full-run report](experiments/naive_pointer_baseline.md) supersedes the initial 0/3 inspection as the aggregate baseline; its source summary/CSV are retained under `eval/pointer_task/`.

## Saved recurrent training checkpoints

The same CLI also accepts the step directories produced by [Stage 1 training](training_pointer.md#artifacts-and-checkpoint-evaluation):

```bash
python -m scripts.eval.naive_test --model models/stage1_pointer/<run>/step-000160 --test
```

The checkpoint's `recurrent_config.json` selects recurrent loading. The evaluator uses its frozen base revision, saved tokenizer, raw dataset prompt, and A–Z argmax at loop equal to requested depth. It prints and records this scoring convention separately from ordinary-model three-shot generation. In a mixed-depth batch, each example is scored at its own depth even though the batch executes its maximum depth. Throughput is example-loops/s and questions/s. Omit generation-budget, prompt, base-model, revision, and tokenizer overrides for this format. Use the whole step directory, not its adapter tensor file; the recorded base must be cached or explicitly downloaded with `--download`.

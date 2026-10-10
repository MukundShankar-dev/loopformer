# Ordinary Qwen trace baselines

User-authorized experiment, 2026-10-10. These baselines ask whether explicit
autoregressive traces, and then task-specific ordinary SFT, explain the benefit
otherwise attributed to latent recurrence. No existing recurrent weights,
controller thresholds, data, or completed population figures change.

## Two arms

1. **Original Qwen with trace prompting.** Pinned original Qwen2.5-0.5B-Instruct,
   one three-shot chat prompt requesting all successive states.
2. **Ordinary Qwen with trace SFT.** The same pretrained initialization, data and
   inference prompt; standard autoregressive response supervision. Adapt **all Qwen parameters**: embeddings, all 24 decoder layers,
   final normalization and tied LM head. There is no LoRA or recurrent reuse.
   All **494,032,768 parameters** are trainable. This follows the user's explicit
   correction; it is not a
   capacity-matched middle-layer ablation.

Each query provides rules, start and requested count **once**, using
[`pointer_trace.txt`](../prompts/pointer_trace.txt). A response is only successive
uppercase states separated by whitespace, excluding the start. The model decides
its states and EOS. Evaluation never supplies intermediate targets, reprompts a
step, repairs a prefix, restricts logits to A–Z, or forces the requested length.
The three examples cover one, two and three transitions. Intermediate targets
appear as gold previous tokens during teacher-forced SFT only.

## Matched training and remaining differences

[`trace_sft.json`](../configs/trace_sft.json) checks the important fields against
[`executor_r.json`](../configs/executor_r.json). Use the exact seed-61 independent
training file: **36,000 examples**, requested counts **1–6, 8, 10, 12**, one epoch,
seed 17, effective batch 16, **2,250 updates**, AdamW, LR 2e-5, 100-update warmup,
cosine schedule to 0.1 of peak LR, weight decay 0.01 and gradient clip 1. Counts
7, 9 and 11 remain held out from training. Data byte hashes and selected ID/order
hashes are recorded. No new tables or longer training traces are generated.

The ordinary SFT objective is full-vocabulary next-token CE on **every response
state and EOS**, averaging response-token loss within each example, then examples
within the batch. Prompt tokens have no loss. Projecting only supervised hidden
positions avoids creating prompt-vocabulary logits while preserving the same
loss and gradients. The whole trace is processed in parallel during training;
inference generates it autoregressively with a KV cache.

Use microbatch **16**, accumulation **1**, gradient checkpointing, BF16 autocast
with FP32 parameters and optimizer state, SDPA and fused CUDA AdamW. Ten-update
full-SFT comparisons measured 0.85/0.60/0.51 seconds per update at microbatches
4/8/16 respectively, with allocated peaks 8.61/8.83/8.03 GiB. All kept effective
batch 16. The noncheckpointed earlier middle-layer smoke is not a full-SFT
resource measurement. The bounded profile does not establish full-run runtime.

Select `best/` using minimum **trained-count validation response CE**, checking
every 250 updates; save `last/` every 750 updates. Validation includes 64 examples
at each count 1–12, but untrained counts do not contribute to checkpoint selection.
Teacher-forced token/whole-response accuracy is labelled as such, never reported
as free-generation accuracy. A fixed four-query-per-count validation panel also
generates traces freely, with a 32-token budget, at initialization and each
validation; it is logged but does not select the checkpoint.
Keep `best/` and `last/` as complete Hugging Face
checkpoints under `models/trace_sft/seed61-full/`, including optimizer/RNG state.
The ordinary model loader accepts these directories. The old final-only prompt
is a different inference protocol; use the trace evaluator for this benchmark.

This controls examples, epoch, effective batch and schedule; ordinary full SFT
has substantially more trainable capacity than R. It does **not** make the
objectives or prompts identical to R. R used symbol CE,
latent intermediate supervision and a re-entry bridge. Its final controller also
saw separate count labels through 63. Ordinary trace SFT sees requested counts
only through 12. Consequently stopping is not an exposure-matched architecture
ablation. The original final-only baseline uses FP32, a different prompt and an
eight-token budget; it remains contextual evidence, not a precision-controlled
prompt ablation. Do not infer matched FLOPs from generated tokens versus R passes.

## Benchmark and scoring

[`pointer_trace_baselines.json`](../configs/pointer_trace_baselines.json) fixes
the existing 1,350 graphs (seeds 307/311/313, three graph types) and **every request
1–256: 345,600 independent generations per arm**. Also rerun the previously opened
1,536-query seed-61 test, counts 1–12. These are retrospective comparisons on
opened panels, not fresh confirmation. Every graph has 26 states; cycles remain.

Both arms use greedy BF16/SDPA inference and a maximum **272 generated tokens**
per query, including EOS. This budget provides room for a 256-state trace without
forcing termination at any count. Requested count is part of the input, so a
single long trace cannot substitute for independent queries at other counts.
Original and SFT outputs are strict whole-response parses; explanations,
punctuation and non-EOS special tokens are malformed.

Report separately:

- Returned final-letter accuracy, regardless of the number of emitted states.
- State at requested position N, when emitted, and correct entire prefix through N.
- Exact termination: a well-formed response with exactly N states **and EOS**.
- Final letter plus exact termination, and **strict success**: every transition
  correct plus exact termination. Cyclic correct letters at wrong lengths fail.
- Early EOS, late EOS, no EOS before the token cap, malformed output, emitted
  length, first wrong symbol and first prefix failure, including missing states.
- Per-count and seed/type/transient/cycle strata; whole-graph stratified bootstrap
  intervals and paired differences from the frozen final recurrent system.

Autoregressive tokens are not recurrent passes. A transition position is a parsed
state position, even when a malformed response uses several tokens per state.
First-error comparisons fix the input request at N=256; they do not blend prompts
at different requested counts. Plot malformed/budget/early/late/exact categories
as a mutually exclusive partition.

## Speed, provenance and running

`trace_baselines.sh` runs training, original trace evaluation, SFT trace evaluation,
then independent audits and four separate figure families. It uses an exclusive
launcher lock, resumes ordinary training from `last/` or `best/`, and resumes
unchanged inference chunks. It fails on incompatible data, weight, prompt, package
or source identities. It never reruns the completed recurrent GPU suite.

```bash
bash trace_baselines.sh --dry-run
bash trace_baselines.sh
```

W&B defaults to the logged-in `loopformer` project for training and both eval
arms; set `WANDB_MODE=offline` or `disabled` explicitly if needed. Local JSON,
JSONL and CSV evidence remains authoritative. Training progress gives ETA, CE and
examples/sec; evaluation gives coverage, strict success, tokens/sec and remaining
time. Track it from the Mac:

```bash
python -m scripts.eval.trace_status --remote desktop --watch
```

Cached generation profiles batch sizes 64/128/256/512 on depth-12 **validation** queries,
selecting throughput alone. A preflight static-cache compilation attempt was
slower and changed continuations, so **production compilation is disabled**.
The optional profiler can reconsider it only with exact eager-continuation
agreement and higher measured throughput. GPU peaks and candidates are saved.
A separate 54-query opened-panel audit records batch/compile sensitivity versus
batch-one eager calls at counts 1/6/12/32/128/256. Any differences remain reported;
the implemented BF16 batch policy defines the baseline, rather than silently
claiming bitwise equivalence to FP32 or batch one. Synchronized generation timing
includes prefill and decode but excludes model loading, profiling and audit time.

For separate tests or a manual continuation:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.training.train_trace --smoke-test --output /tmp/trace-smoke
python -m scripts.training.train_trace --resume models/trace_sft/seed61-full/last
python -m scripts.eval.trace_baselines --arm original
python -m scripts.eval.trace_baselines --arm sft
python -m scripts.eval.trace_analysis --input eval/pointer_traces/seed61-20261010
```

Raw IDs, decoded outputs, input hashes and scores live in atomic compressed chunks
under `eval/pointer_traces/seed61-20261010/{original,sft}/chunks/`. The independent
audit reconstructs every input, target and response from those raw files, checking
coverage, EOS, hashes and metrics before rendering figures. Smaller JSON summaries,
NPZ outcomes, CSV strata, provenance and plots are shareable without model weights
or the large raw response payloads. Preserve the desktop raw files for re-audits.

Four figure families live under this run's `plots/`: depth quality, execution and
termination failures, graph-stratum heatmaps, and SFT learning curves. Every panel
has labels/legends; line styles, sparse offset markers and heatmaps distinguish
overlap. The original 25-family bundle remains unchanged.

Implementation: [`trace_task.py`](../scripts/trace_task.py) owns trace targets,
loss and scoring; [`trace_runner.py`](../scripts/training/trace_runner.py) owns
optimization; [`trace_baselines.py`](../scripts/eval/trace_baselines.py) composes
the existing ordinary model loader/evaluator; [`trace_analysis.py`](../scripts/eval/trace_analysis.py)
independently audits outputs and reuses existing clustered metrics.
[`cached_generation.py`](../scripts/eval/cached_generation.py) supplies explicit
cache settings and trace scoring through that evaluator's interface. The
historical evaluator stays byte-identical so completed baseline source freezes
remain valid.

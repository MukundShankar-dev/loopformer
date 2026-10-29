# MMLU retention comparison — 2026-10-12

## Completed result

The full comparison is complete and independently audited. At one loop, the
trained recurrent pointer executor substantially underperforms the pinned
original checkpoint on standard five-shot MMLU.

| Model | Correct / questions | Question-weighted accuracy |
| --- | ---: | ---: |
| Original Qwen2.5-0.5B-Instruct | 6,589 / 14,042 | 46.92% |
| Final recurrent pointer checkpoint, one loop | 3,497 / 14,042 | 24.90% |

The paired difference is **−22.02 percentage points**, with subject-stratified
paired-bootstrap 95% interval **−23.07 to −20.96 points** (10,000 replicates,
seed 0). The recurrent score is near the 25% uniform-random four-choice level.
The two models agree on **3,800 / 14,042 predictions (27.06%)**.
Of 6,589 base-correct items, **4,880** become wrong; **1,788** base-wrong items
become correct. Net loss: 3,092 correct answers. Macro-average subject accuracy
also drops, from 48.52% to 24.87%.

| Standard MMLU group | Questions | Base | Recurrent, T=1 |
| --- | ---: | ---: | ---: |
| Humanities | 4,705 | 42.87% | 24.95% |
| Other | 3,107 | 52.20% | 24.65% |
| Social sciences | 3,077 | 53.75% | 24.02% |
| STEM | 3,153 | 41.10% | 25.94% |

The recurrent model selects A/B/C/D 6,400/3,036/3,481/1,125 times;
the base selects them 3,273/3,769/4,136/2,864 times. This documents changed
answer preferences; it is not an intervention that identifies a cause.

This is evidence of substantial loss of functional MMLU performance after
pointer adaptation, even at one pass. It does not establish that every general
capability is lost or distinguish changed representations from changed answer
readout. The earlier successful sky answer at one loop did not establish broad
preservation. No additional fitting or attempted repair follows this evaluation.

## Authorization and frozen protocol

The user requested full MMLU on the SSH desktop, comparing original Qwen to the
trained recurrent pointer architecture at exactly one loop. This is a bounded
general-capability evaluation of frozen weights. It authorizes no fitting,
checkpoint selection, controller repair, or multi-family training. MMLU does
not establish a natural-language meaning for individual recurrent loops.

The protocol is frozen in [mmlu_retention.json](../../configs/mmlu_retention.json)
before scoring benchmark questions:

- Original `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Final recurrent checkpoint
  `models/stage1_pointer/controller-shared-number-precision-seed83/best`,
  adapter SHA-256
  `0f8a3f2572ec2caf4fafbcdd074b5283bd8c610602502763b3c04487ce1f123d`.
- EleutherAI harness 0.4.13, standard `mmlu` task definitions;
  `cais/mmlu` at `c30699e8356da336a370243923dbaf21066bb9fe`.
  All 57 subjects and 14,042 test questions; five first-in-order development
  examples per subject, plain standard MMLU prompt, no chat template.
- The same checkpoint tokenizer for both arms, with equality to the pinned
  original tokenizer checked. Score the four standard A–D continuation
  log-likelihoods; choose the first maximum on ties. No free generation.
- FP32, SDPA, disabled TF32, deterministic algorithms, disabled KV caching,
  fixed batch 4, seed 0. No prompt truncation. Both arms append no intermediate
  answers. The standard prompt includes only the five development answers.
- Force one recurrent pass. Bypass only the task-specific router/timer in
  memory because MMLU lacks the pointer Steps field. Keep trained R, frozen
  P/C, and the trained bridge; the bridge is unused on loop one.

This measures the trained executor's one-pass general capability, not native
pointer-controller stopping. A frozen controller cannot initialize from a
general MMLU prompt under its current input contract. No inference checkpoint
files are rewritten.

Primary benchmark definitions: [EleutherAI MMLU task template](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml),
[pinned MMLU dataset revision](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe).
The actual installed task YAMLs are copied into the run's `task_configs/`.

## Scoring and validation requirements

Use the harness's own prompts, documents, few-shot selection and accuracy
aggregation. A narrow single-token likelihood adapter computes only the final
full-vocabulary logits, grouping the four choices for an identical context.
It rejects non-single-token targets or tokenization boundary mismatches rather
than silently scoring another benchmark. This avoids a prompt-length × full
vocabulary logit allocation. Compare its scores with the standard HF harness
on a bounded pilot, and check fresh one-loop recurrence against original Qwen
before using the adapted checkpoint.

Retain per-question likelihoods, IDs, targets, prompts and input hashes for
both arms; verify paired inputs and independently recompute all accuracies.
Report question-weighted overall accuracy, per-subject accuracy, prediction
agreement, base-correct/recurrent-wrong and the reverse. Use a paired bootstrap
stratified within subject for the accuracy difference. No formal equivalence
margin is selected; describe the measured difference and uncertainty.

## Environment and artifacts

Host: `desktop`, `/home/mukund/loopformer`, NVIDIA GeForce RTX 5070 Ti.
Python 3.11.8, Torch 2.14.0+cu130, Transformers 5.17.0.
Missing evaluation dependencies were installed by requesting
`lm_eval==0.4.13` constrained by `requirements.txt`; the resolved dataset version
is also pinned in `requirements-mmlu.txt` for reproduction. Existing main
research pins were retained; harness dependencies downgraded transitive
`chardet` from 7.6.0 to 6.0.0.post1. Before-install freeze and install logs are
under desktop `/tmp/loopformer-before-mmlu.txt` and
`/tmp/loopformer-mmlu-install{,-plan}.log`.

Run artifacts belong in `eval/general/mmlu-retention-20261012/` on the desktop.
Install/setup commands:

```bash
.venv/bin/python -m pip install lm_eval==0.4.13 --constraint requirements.txt
.venv/bin/python -m pytest -q tests/test_single_token.py tests/test_mmlu_analysis.py
.venv/bin/python -m scripts.eval.mmlu --arm base --preflight
```

The full run was launched from the Mac in a persistent desktop session:

```bash
ssh desktop 'tmux new-session -d -s mmlu-retention "cd /home/mukund/loopformer && bash mmlu_retention.sh"'
```

`mmlu_retention.sh` runs preflight, both arms, and the independent audit.
`STAGE`, arm `progress.json` files and logs under the run directory expose
progress without interrupting inference. Response caches permit resumption
only with unchanged frozen identities. Final raw per-subject JSONL files retain
the four likelihoods and full input/target records.

Both inference arms completed once. The first audit attempt failed because
Python `str.splitlines()` split Unicode line separators embedded inside JSON
strings. The reader now iterates physical file lines, with a regression check.
The completed independent audit was rerun without repeating model inference:

```bash
.venv/bin/python -m scripts.eval.mmlu_analysis > eval/general/mmlu-retention-20261012/audit.log 2>&1
```

It reconstructs every prompt from the pinned test/development rows and copied
subject description, verifies all 14,042 question IDs and targets for each arm,
Independent argmax scoring of the four raw likelihoods agrees with every saved
question result and all 57 subject/four category/overall harness aggregates.
There are 14,015 distinct encoded prompts; grouping repeated prompts preserves
all 14,042 separately scored questions. The final adapter SHA-256 is unchanged.

An additional unadapted-R control uses the **exact saved executor settings**,
including `prefix_reuse=True` and the loaded architecture's other switches,
with the same router/timer bypass. Its checked full-vocabulary one-loop logits
also match original Qwen exactly. The control ran as:

```bash
PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python /tmp/loopformer_architecture_control.py
```

The source is preserved as `architecture_control.py` in the copied run artifacts,
with results in `architecture_control.json`. This control and the standard-HF
scorer comparison limit an execution-path explanation for the observed gap;
they do not identify which trained representations or readout preferences cause it.

Measured harness durations are 602.9 seconds (base) and 675.7 seconds (recurrent).
They include task preparation, likelihood scoring and export, and exclude model
loading. They are run-cost records rather than a model-only latency measurement.
Peak allocated CUDA memory: 2.75 GiB and 3.75 GiB, respectively.

An audited snapshot is also copied to the Mac at the same repository-relative
location, including all raw subject records and checksums. Desktop response
caches and logs remain on the desktop. Principal artifacts:

- `comparison.json`: accuracy, paired changes, intervals and group summaries.
- `paired.csv`: every question's two predictions and target.
- `subjects.csv`: all 57 subject comparisons.
- `independent_audit.json`: coverage/input/target/metric checks and raw-file hashes.
- `base/` and `recurrent/`: per-subject raw JSONL, harness results, frozen
  identities and completed progress/resource records.
- `protocol.json`, `task_configs/`, `environment.txt`, `preflight/preflight.json`,
  `architecture_control.json`, `choice_distribution.json`.

The Mac copy matches all 114 subject-file hashes in the desktop audit.

`pip check` also found an OpenTelemetry SDK/exporter mismatch already present
in the saved before-install environment. This unrelated transitive mismatch
was preserved; the evaluator imports and scientific contract checks pass.

## Status

Six focused tests pass on the Mac and desktop. The installed dataset resolves
to all 57 subjects/14,042 questions. Twelve standard-HFLM likelihood requests
agree with the compact scorer to a maximum absolute residual of
0.00001680851; checked A–D decisions agree. Fresh fixed-prompt recurrence at one
loop matches original Qwen's full-vocabulary final-position logits exactly
(maximum absolute residual zero on checked inputs).

Both complete model arms and the independent audit pass. `STAGE` is `complete`.
No training, checkpoint changes, or post-result checkpoint/prompt selection occurred.

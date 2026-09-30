# Offline failure review and checkpoint comparison

Reviewed 2026-09-29 using synced artifacts dated through 2026-09-30 UTC. This extends the [initial probe/profile report](baseline2500_diagnostics.md). No pretrained model execution or training was performed. All results are development diagnostics; seed 29 remains reserved.

## Inputs and reproducibility

The review uses all six retained fresh-30k evaluations at steps 2500, 3250 and 3750, covering seed-17 validation (depths 1–8) and depth_test (9–16), plus the 32-example paired probe at step 2500. It reuses the existing reference execution and first-failure validator, verifies dataset checksums, full fixed-depth coverage, targets and correctness, and pairs checkpoints by example ID. All 32 probe-original prediction sequences match their historical step-2500 traces exactly.

Run from the repository root:

```bash
.venv/bin/python eval/pointer_diagnostics/offline-review/analyze.py
```

The [analysis source](../../eval/pointer_diagnostics/offline-review/analyze.py) records input hashes in [summary.json](../../eval/pointer_diagnostics/offline-review/summary.json). The same directory contains per-checkpoint first errors/risk sets, paired checkpoint outcomes, failure-aligned margins and paired probe measurements as CSVs. Rerunning replaces these derived analysis files. It does not load model weights or change source artifacts.

Historical full evaluations contain target margins but not entropy, target rank or hidden summaries. Those richer measurements exist only for the small new probe; they cannot be reconstructed for later checkpoints from the saved CSVs.

## What precedes late failure?

For the same 22 original probe examples whose first error occurs after loop six:

| Measurement, median | Three loops before | Two before | One before | First error |
| --- | ---: | ---: | ---: | ---: |
| Correct-answer logit margin | 10.31 | 8.76 | 5.18 | -2.51 |
| Answer entropy, nats, over 26 symbols | 0.0019 | 0.0056 | 0.182 | 2.151 |
| Correct-answer rank | 1 | 1 | 1 | 9.5 |
| Answer-position relative update norm | 0.676 | 0.596 | 0.477 | 0.315 |

The preceding margin decreases between offsets -3 and -1 in 20/22 examples. Only six first errors retain the correct answer in second place; fifteen rank it below third. Thus many failures involve substantial deterioration of the symbolic readout, rather than a tiny argmax reversal. First-error categories in this cohort are nine other earlier-path states, seven off-path symbols, four previous-state repeats and two future-path states.

This is a failure-aligned description, not a validated early-warning detector. Every first-error margin is nonpositive by definition. We have not compared a held-out predictor against matched successful controls or established an entropy threshold.

The margin pattern is not confined to 32 examples. Among all 783 step-2500 depth-test examples whose first error occurs after loop six, median margins at offsets -3/-2/-1/0 are 9.78/8.45/5.31/-2.64. This cohort has a fixed denominator at every offset. Early failures differ: two of the four early failures in the small probe have wrong-answer margins above 11 with very low entropy. Confidence alone would miss those errors.

## Matching the same lookup after restart

At the 22 aligned first-error transitions, the fresh suffix gets every answer right. Its median correct-answer margin is 11.56 versus -2.51 uninterrupted, entropy 0.00055 versus 2.15, and answer-position relative update norm 0.707 versus 0.315. These are within-sequence update norms, not distances between original and suffix hidden states.

The suffix gives the correct intermediate Start, reruns P, reduces requested Steps, and resets recurrent history. It is therefore an oracle-assisted diagnostic, not a deployable repair method. Its remaining depths span 3–10, compared with original depths 9–16. The saved scalar summaries cannot distinguish a degraded rule representation, an inadequate current-state representation, a coda/readout mismatch, or prompt/depth conditioning. Raw hidden states and token-region measurements were not exported.

## What changes with further training?

| Checkpoint | Complete at trained depths 1–6 | Complete at depths 9–16 | Deep first errors repeating previous reference state |
| --- | ---: | ---: | ---: |
| 2500 | 735/750 (98.0%) | 164/1000 (16.4%) | 191/836 |
| 3250 | 744/750 (99.2%) | 92/1000 (9.2%) | 318/908 |
| 3750 | 745/750 (99.3%) | 121/1000 (12.1%) | 380/879 |

On identical deep examples, relative to step 2500:

- Step 3250 loses 96 previously complete trajectories and gains 24. Its correct prefix becomes shorter for 496 examples and longer for 190; 314 are unchanged.
- Step 3750 loses 85 and gains 42. Its prefix becomes shorter for 451 and longer for 238; 311 are unchanged.

There is a measurable tradeoff: trained-depth execution improves while farther-depth execution regresses. This is consistent with specialization to the training horizon, but it does not establish its optimizer or representational cause. Step 3750 also recovers some performance relative to 3250, so deterioration is not monotonic.

The failure boundary moves inward even after conditioning on a fully correct prefix. At loop eight, first failures are 46/937 (4.9%) at 2500, 121/951 (12.7%) at 3250, and 122/943 (12.9%) at 3750. At loop nine, they are 131/891 (14.7%), 267/830 (32.2%), and 231/821 (28.1%). Survivor cohorts differ across checkpoints; the paired prefix counts above provide the matched-example comparison.

Later first failures increasingly repeat the previous reference state (22.8%, 35.0%, 43.2% of failures). This describes decoded stalling, not proof that R becomes an identity function. Historical later-checkpoint traces lack hidden-state measurements. Across late-failure cohorts, median first-error target margins become more negative: -2.64, -3.61, -5.20. Those cohorts also differ, so this is not a paired confidence-change estimate.

## Recurrence and supervision audit

Inspected `scripts/recurrent_qwen/model.py`, `scripts/training/data.py`, `scripts/training/objective.py`, and `scripts/training/runner.py`:

- Inputs are encoded once. P runs once, and the same R updates every token position at every loop. No immutable copy of the rule representations is reintroduced.
- C reads each recurrent state, but its output and decoded symbols never enter the next R call. This is continuous hidden-state recurrence, not teacher-forced answer-token chaining.
- Loop t is supervised against reference transition t. Targets are supplied to the loss, not inserted into the recurrent input. The baseline averages valid per-loop CE within each example and then across examples.
- There is no detach/no-grad around training recurrence or C. Later losses backpropagate through earlier recurrent iterations; only R's LoRA parameters are optimized. Loops beyond an example's nominal depth are masked.
- There is no explicit loop-index input. The prompt supplies requested Steps and the evolving state can carry implicit iteration information; neither guarantees learned counting.
- Because attention is causal and Rules precedes Start/Steps, rule-prefix positions cannot receive information from those later tokens. Their representations nevertheless evolve through R each loop. The answer position can attend to the preceding context.

This makes rule-memory evolution a concrete candidate to separate from answer-state evolution. It does not establish that rules are being forgotten. Per-loop loss is present; it does not constrain a unique hidden representation or guarantee stable extrapolation beyond six training loops.

## Bounded desktop follow-up, proposed only

First separate prompt conditioning from recurrence age: compare the existing reference suffix restart with a version retaining the original Steps text, running the same remaining number of loops and scoring against the same suffix targets. This deliberately decouples the displayed request from the diagnostic horizon; it must be labeled as such. Record token-position/length changes, since they are another possible confound. Keep the original uninterrupted and current suffix variants as controls.

Then consider a separately implemented intervention refreshing only the rule-prefix representations from P while preserving the current recurrent answer state. Compare with untouched recurrence and a no-op intervention, including healthy trained-depth examples. A benefit would implicate access to persistent rule context; harm would be inconclusive because overwriting representations also changes the distribution seen by the trained R/C. No oracle intermediate answer should enter this particular intervention.

These are diagnostic designs, not instructions to retrain or an adopted architecture. The evidence does not yet select a new loss, a loop counter, a halting head, or a bridge. Halting before a required transition cannot solve a still-unfinished pointer program. Keep numerical performance experiments separate from these scientific interventions.

## Validation

The analysis completed with reference/coverage/checksum/pairing assertions passing. Existing focused tests for gradient scope, shared weights, observability and pointer diagnostics passed: **25 tests in 15.46 seconds**. These use tiny local models and do not rerun pretrained CUDA inference. Edited documentation file links resolve and `git diff --check` passes.

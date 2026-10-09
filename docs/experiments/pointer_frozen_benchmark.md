# Frozen pointer benchmark — 2026-10-08

The frozen model passes the current reserved test perfectly, but **does not pass
the larger end-to-end generalization benchmark**. Exact stopping fails starting
at requested count 70, before the first recurrent transition: the initializer
reads 70 as approximately 49.26. No model parameter, threshold or benchmark panel
was changed in response. Executor and controller evidence are reported separately.

## Frozen identity and predeclaration

Checkpoint: `models/stage1_pointer/controller-affine-seed83/best`, the first
predeclared repaired seed. Its weight SHA-256 is
`8dfdaee726c0f6413ed2f15478abad68574eb63d8401bb3463bbdc09ab27ff7a`.
Stop threshold is 0.5 and the common safety cap is 272. All inference-file hashes
(tokenizer/config/weights) were recorded before opening the test and verified
afterward. P/R/C, bridge, controller initializer, recurrence and stop readout are
all frozen; benchmark extraction disables every parameter's gradients and uses
inference mode. Native stopped evaluation also uses inference mode. No fitting,
threshold search, checkpoint selection or target-fed memory occurs.

The [protocol](../pointer_benchmark.md) and
`configs/pointer_benchmark.json` were committed at `e71f516` before the existing
test ran. The larger suite ran implementation `d44f602`, with source/config hashes
saved before model execution. Its quality thresholds were 95% overall joint and
complete-trajectory success, 99% exact stopping, plus 90% joint/trajectory and
95% stopping at every individual depth. **Overall gate: failed.** The observed
1–69 boundary is a post-result description, not a newly selected acceptance gate.
Seed 29 and unrelated research stages remain unopened.

## Current reserved test

All 1,536 seed-61 reserved questions on 128 independent graphs, balanced over
counts 1–12, ran through actual native learned-stop inference. There are no
filtered questions or truncation.

- Exact stopping: **1,536/1,536 (100%)**.
- Complete nominal trajectories: **1,536/1,536 (100%)**.
- Joint correct answer and exact timing: **1,536/1,536 (100%)**.
- Early, late and safety-cap fallback stops: **zero**.
- All **9,984** intermediate predictions/targets match independent raw-rule execution.

Synchronized inference took 305.66 seconds, with mean
6.5 executed loops and p95 per-question latency 0.4388s.
This timing includes model forward, readouts and stop checks, excluding loading,
encoding and export. At 128 graph-level successes, the marginal 95% Wilson lower
bound is 97.09%; repeated depths do not make 1,536 independent graph samples.
The raw audit and native evaluator metadata are saved under `existing_test/`.

## Larger independent benchmark

Three fresh data seeds 211/223/227 each supply 150 graphs of each kind: random
function, random permutation and full cycle. This is **1,350 distinct graphs**, 450
per mode, and **345,600 graph/count queries** at every integer depth 1–256.
Sampling fixes graph/start/rule order without consulting depth. All generated
records reproduce from seeds and reference execution. Zero graphs overlap any
old split (36,288 distinct old rule tables excluded) or another new graph.

The training controller saw requests 1–63 excluding 9/17/29/41/53, but was unrolled
only through 12. Thus 13–63 is mostly seen numeric values with unseen full recurrent
rollouts; 64–99 adds unseen values/roles; 100–256 tests three-digit layouts. Do not
call all requests above 12 unseen numeric values. Five withheld values are
reported separately in the summary.

| Requested count | Executor: every nominal step correct | Executor: final letter at N | Controller: exact stop | Whole model: letter + exact stop |
| --- | ---: | ---: | ---: | ---: |
| 1–12 | 99.14% | 99.29% | 100.00% | 99.29% |
| 13–63 | 97.17% | 97.94% | 100.00% | 97.94% |
| 64–99 | 96.96% | 97.85% | 16.67% | 16.31% |
| 100–256 | 96.96% | 97.85% | 0.00% | 0.00% |

Across all 256 counts, complete trajectories average
**97.11%**, nominal final accuracy
**97.93%**, exact stopping **26.95%**,
and joint answer/timing **26.46%**.
At count 256 specifically, executor complete trajectories are
**96.96%** and nominal final accuracy is
**97.85%**. Exact stopping is zero there. At depth 256, **1,309/1,350** graphs have every
transition correct (marginal Wilson 95% interval 95.91%–97.75%). All 41 imperfect
graphs first fail between loops 3 and 31; no additional first failures occur
at loops 32–256 on this panel. This is finite evidence of durable transition
execution, rather than a claim of universal or arbitrary-depth correctness.

| Graph mode | Independent graphs | Complete trajectory through 256 | Final letter at 256 |
| --- | ---: | ---: | ---: |
| random_function | 450 | 97.33% | 97.78% |
| permutation | 450 | 97.56% | 98.00% |
| full_cycle | 450 | 96.00% | 97.78% |

Range summaries use a 2,000-repeat graph-cluster bootstrap within seed/mode strata
(seed 239). For overall complete trajectories, the graph-bootstrap 95% interval is
96.21%–97.95%.
Per-count/stratum rates also include marginal graph-level Wilson intervals. These
are not simultaneous confidence bounds over 256 counts. The controller's suffix
input is graph invariant, so its 345,600 repeated outcomes represent just **256
distinct numeric inputs**; graph intervals do not certify unseen-integer behavior.
Every seed/mode/depth/orbit stratum is exported, with exact denominators.

## What fails in the controller

Every count 1–69 stops exactly; every count 70–256 stops early. No late or missing
stops occur. Initial count MAE is approximately 1e-6 within the trained numeric
range, 33.81 across 70–99, and 151.46 across 100–256.

| Requested steps | Initial memory | Actual stop loop |
| --- | ---: | ---: |
| 69 | 69.000000 | 69 |
| 70 | 49.256649 | 49 |
| 80 | 46.231937 | 46 |
| 90 | 43.075726 | 43 |
| 100 | 17.992903 | 18 |
| 128 | 15.498024 | 15 |
| 256 | 43.283772 | 43 |

The frozen tokenizer/input audit shows that count 70 puts digit 7 into suffix
position 3, where training had no digit 7. Counts 80–99 similarly introduce
unseen tens-role digits. Count 100 introduces previously unseen tokens in suffix
positions 0/1/2/3; 128 and 256 introduce new positions 0/1/2. The initializer is
linear on independently weighted token positions, with no sharing that enforces a
common digit value across roles. Those architecture/coverage facts and measured
pre-loop errors support an initialization failure, not an explanation based only
on losing track of elapsed loops. They do not prove every possible cause of any
executor error. The countdown itself starts from an incorrect number.

Correct letters at wrong times remain explicit failures. Across the full panel,
**11.78%** of queries have a correct stopped
letter despite incorrect timing, while stopped-letter accuracy is
**38.24%**. Those cyclic coincidences never raise
exact-stop, joint or strict-success scores.

## Efficient execution and native fidelity

The isolated executor never receives Steps, and this controller ignores executor
observations. Therefore the suite runs each graph once through 272 forced loops,
then applies native suffix initialization/recurrent head updates for all counts.
Shared `forward_symbols`, `stop_rows`, model loading and reference execution are
reused. This is an executor-quality baseline plus controller replay; it is not
345,600 separately executed native stopped requests or measured adaptive latency.

The predeclared native panel has the first graph in each seed/mode stratum at
counts 1/9/12/17/29/41/53/64/80/99/100/101/128/256. All **126/126** actual exported
learned-stop calls match reused predictions, executed loops, exact/early/late
stops, joint success and missing/cap semantics, including failed requests. The
benchmark refuses acceptance on disagreement. The full forced executor pass
executed **367,200 example-loops** in 794.12 seconds
(462.40 example-loops/s). This includes batched
forward/readout, CPU prediction transfer and progress overhead, excluding loading,
encoding, scoring and export. No speedup or compute/quality tradeoff claim is made.

## Commands, artifacts and checks

The existing test ran with:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.eval.loop_test \
  --model models/stage1_pointer/controller-affine-seed83/best \
  --data data/pointer/seed-61-independent/test.jsonl \
  --device cuda --attention sdpa --loops 272 \
  --stop-policy completion --stop-threshold .5 \
  --output eval/pointer_benchmark/frozen-20261008/existing_test \
  --wandb-mode disabled
```

Then `bash benchmark_pointer.sh` verified/reused that completed test and ran the
larger suite. The reusable [runbook](../pointer_benchmark.md) includes dry run,
freeze checks and optional standalone figures. Output paths refuse overwrite.
A later guard additionally checks the completed test's data/model/policy hashes
and full coverage before reuse; it was tested against these actual artifacts.

All results are under `eval/pointer_benchmark/frozen-20261008/`. The independent
panel saves `summary.json`, `controller_counts.csv`, `graphs.csv`,
`per_count_and_stratum.csv`, `per_loop.csv`, compressed full `decisions.csv.gz`,
native tasks/trajectories/decisions, dataset/reference/suffix audits and provenance.
Generated graphs are under `data/pointer/benchmark-seeds211-223-227/`, excluded from
Git; its manifest and seed-replay evidence are tracked with evaluation results.
No checkpoint weights are pushed or changed. Logs and plots are artifacts, not
replacement documentation.

Eleven focused benchmark/affine tests passed in 22.01 seconds. Contracts cover
disjoint seed replay, no depth-conditioned graphs, wrong-time cyclic answers,
missing-cap failures, first-error recovery versus strict trajectories, graph
bootstrap boundaries, unchanged weights, tiny native/reuse agreement on failures,
and refusing partial/mismatched reserved-test reuse. All 345,600 compressed decision rows and every all-graph per-depth aggregate were
independently checked against reference targets and saved model predictions. All
3,771 native intermediate rows were reference-checked; inference-file hashes still
match the pre-test freeze. Local links, shell syntax, whitespace and standalone
figure rendering were checked.

Original render exports are superseded by the [organized population suite](../analysis.md#runbook-outputs-and-rendering-contract).
The replacement is generated after full-suite audits pass. Prior renders remain
recoverable from Git history at `b49030e`; raw metrics and numeric audits are retained.
compares execution, stopping and pre-loop initialization. Dotted boundaries mark
12 supervised recurrent loops, maximum training request 63 (five exclusions),
and the three-digit boundary at 100. Plot provenance records Matplotlib 3.11.2,
source and input hashes. Plot dependencies were isolated in `/tmp`; the main
training/evaluation environment was unchanged.

Automatic approval review rejected the attempted reserved-test W&B upload because
it lacked payload-specific authorization. All runs used W&B disabled; outputs are
local/repository artifacts, with no external SaaS upload or model changes.

## Scope and next decision

This benchmark strengthens finite pointer-instance and recurrent-depth evidence,
and rejects a claim that the complete model/controller generalizes across 1–256.
The successful short test does not override the numeric-input boundary. Review
the number reader before another training recipe; these evaluated panels are now
opened, so any later tuning requires a new held-out confirmation design.

All graphs have 26 states, so a 256-step execution necessarily cycles; these are
not nonrepeating 256-state paths. Fixed prompt syntax, one frozen model seed,
restricted vocabulary, graph modes and finite count range limit the claim.
There is no natural-language/cross-family result, generic correctness evaluator,
terminal repair/damage evidence or arbitrary-depth guarantee. The raw per-loop
and first-error exports support investigation without changing the frozen model.

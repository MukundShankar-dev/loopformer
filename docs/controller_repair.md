# Controller repair — 2026-10-08

The user authorized working through the controller failure using the desktop.
The successful seed-61 executor at step 2,250 remains frozen. The completed
[training audit](experiments/controller_training_audit_seed61.md) does not establish
a unique cause: initialization errors, imperfect fitting and optimization
sensitivity coexist. A new run must answer a specific question before the next
change is selected.

## First controlled intervention

`configs/controller_initialization.json` keeps the original graph panel, requested
counts 1–6/8/10/12, 3,000 updates, optimizer seed 83, controller initialization,
GRU architecture and threshold 0.5. Its only objective change relative to the
seed-83 remaining-work arm is an additional loop-zero regression loss with
weight 12. The loss is mean ((predicted initial work − N)/12)^2 across examples,
without dividing by trajectory length. Existing stop BCE and weight-one trajectory
regression remain. Twelve is a declared exploratory coefficient, not a measured
optimum. The historical arm is the retained control; no identical rerun is needed.

The readout remains training-only. Targets are never forward inputs, and neither
readout predictions nor a programmed decrement update the controller. Full BPTT
and original learned stopping remain. Source executor weights cannot change.

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.training.train_controller \
  --config configs/controller_initialization.json \
  --features-cache models/stage1_pointer/controller-seed61/features.pt \
  --output models/stage1_pointer/controller-initialization-seed83
```

Evaluate the selected and final controllers using the existing deep cache:

```bash
python -m scripts.eval.controller_candidate \
  --run models/stage1_pointer/controller-initialization-seed83 \
  --deep-features-cache eval/pointer_diagnostics/controller-remaining-seed61/deep_features.pt
```

This composes the existing replay and exact-stop scorer; it does not extract new
Qwen features, tune thresholds or replace native inference validation.
Outputs use the existing portable checkpoint format and training artifacts.
Evaluate initial error, recurrent error and exact first-crossing stops separately,
including every individual count and best versus final checkpoints. Improving
familiar-count performance alone does not repair generalization. A count-nine
failure despite improved initialization rejects loss dilution as a sufficient
explanation. Confirmation data remain closed.

## Acceptance scope

Before claiming a controller repair, require at least 95% exact stopping overall
and at least 90% at each count on graph-disjoint development queries through 64,
with the same 0.5 threshold. Report joint answer-and-timing success, full executor
trajectory correctness, early/late/missing stops, and correct letters at wrong
times separately. Native exported inference must agree with cached replay on a
predeclared panel including short, failed and long requests. Repeat a successful
recipe across optimizer seeds 83/89/97 without choosing a winning seed.

Training coverage and claim scope must be explicit. Counts newly supervised in
initialization are seen requested values even if recurrent training never unrolls
that far. Separate held-out requested values from seen-value longer rollouts.
These are finite development acceptance criteria, not proof of arbitrary-depth
counting or independent confirmation. Reserved test data and seed 29 remain
untouched; no natural-language or cross-family claim is opened.

## Broader count exposure with short prefixes

The first intervention completed in 88.55 seconds. Selected step 2,800 reached
95.92% familiar-count stopping, 38.02% held-out 7/9/11 and 1.80% deep stopping.
Count nine remains 0%; familiar initial MAE improved to 0.212 but count-nine MAE
was 1.164. This rejects independent initialization weighting as a sufficient fix.
See [the report](experiments/controller_initialization_seed61.md).

The next declared exploratory config, `configs/controller_prefix.json`, uses 256
of the same training graphs and requests 1–63 excluding 9/17/29/41/53. Counts
7/11 are now seen. Counts 9/17/29/41/53 remain held out; 64 exceeds the largest
seen request. Train the same GRU and auxiliary readout from the same original
controller for 6,000 updates, batch 256, seed 83. Full prompt features are frozen
P outputs; existing count-free R features provide the first 12 observations.
A no-update native check validates reuse on four graphs at short/middle/long
requests and cached validation prompts before training.

Stop labels are continue at every observed t<N and stop only if t=N is actually
observed. A request longer than 12 has **no positive stop label** in its prefix;
the training cap is never mislabeled as completion. Numerical targets are N−t
through min(N,12), including initialization, with existing scale 12 and independent
initial weight 12. No decoded number is fed back, no memory is reset, and the GRU
must generate each next memory itself. This tests broader state/count coverage
while keeping actual recurrent training unrolls short. It changes graph count,
request coverage and update budget together, so it is a repair pilot, not a causal
ablation attributing any gain to a single factor.

Select on exact stopping for **seen requests at most 12**, then their BCE. Larger
requests and held-out numeric values cannot select checkpoints. Evaluations
separately report `selection`, `seen_count_long_rollout`, and
`unseen_requested_count`; count-level rows retain exact denominators. The original
32-graph deep cache remains development evidence, not confirmation.

```bash
bash repair_controller_prefix.sh --dry-run
bash repair_controller_prefix.sh
```

The launcher prepares `models/stage1_pointer/controller-prefix-features.pt`, trains
`models/stage1_pointer/controller-prefix-seed83/`, then evaluates selected/final
heads on the immutable development caches. Output paths refuse overwrite. Frozen
prompt extraction executes P only, instead of all 64 executor loops per request.
Model export remains the existing format, with every non-controller tensor intact.
Implementation tests and actual desktop effectiveness must be recorded separately.

## Suffix initializer and learned affine memory

The broader GRU pilot completed in 144.38 seconds but still failed: selected
step 5,800 had 60.65% short seen-count stopping; deep stopping was 8.18% for seen
values and 5.63% for unseen values. Short initialization MAE was 1.743. A frozen
summary-feature ridge probe still made about 1.5-step validation errors. Merely
standardizing that summary did not resolve the initialization bottleneck.

A no-update probe instead fitted numeric values from the last eight frozen token
embeddings. It recovered all 1–64 values, including held-out 9/17/29/41/53/64,
to double precision. For this raw prompt template those suffix tokens contain
Steps and Answer, independent of graph/start. This identifies an available
compositional input route; it does not establish any controller result.

The next explicit architectural pilot (`controller_kind: affine_suffix`) has:

- A linear initializer on flattened final-eight token embeddings. Supervised
  least squares fits its weights on the same training requests only; the learned
  weights are frozen afterwards. No number parser, integer lookup or gold count
  executes at inference.
- An unbounded one-dimensional learned memory. Each executor transition invokes
  the shared trainable affine cell `m_next = a*m + b`. It starts at identity
  (a=1, b=0), without a decrement. Free-running N−t supervision trains a and b;
  targets are never fed into the cell. Numeric measurement is memory itself,
  implemented through a fixed identity readout for existing metric interfaces.
- A learned affine stop readout with ordinary sigmoid/0.5 first crossing. BCE
  trains this readout on detached memory, so it cannot distort numerical dynamics.
  It intentionally ignores R's symbol/state for timing: this controller counts
  requested transitions and does not judge whether execution needs repair.

This is a strong counting inductive bias, distinct from a generic GRU controller
or adaptive confidence-based stopping. It changes both initialization access and
memory dynamics; success cannot establish which change alone was necessary. The
raw public prompt, R, P/C execution, bridge and intermediate pointer targets stay
unchanged. Only the controller branch uses the suffix embeddings rather than P's
single answer-position summary. No new prompt appears between loops.

Training reuses the 12-loop prefixes and broader counts above. After initializer
fitting, AdamW trains four scalar recurrence/stop parameters for 3,000 updates.
Base LR is .01 with the same warmup/decay; the gain parameter's LR is .0001
(1/100 of the base) because gain errors grow with memory magnitude. Initialization
is analytically supervised, not described as SGD or ordinary full-model SFT.
Every learned parameter and its fit method is recorded. Stop threshold and
short-seen-count selection remain fixed; no held-out value selects a checkpoint.

```bash
bash repair_controller_affine.sh --dry-run
bash repair_controller_affine.sh
```

New caches and the complete portable model are under
`models/stage1_pointer/controller-affine*`; selected/final development evaluation
uses the existing replay and exact-stop metrics. Loader/export preserve every
non-controller tensor and explicitly save the controller kind. Historical GRU
checkpoints remain compatible. Native inference and three-seed results are now complete, as recorded below. The suffix length and training coverage restrict
claims to the declared numeric/template scope; arbitrary integers or alternate
prompt wording are not established.

The first pretrained affine pilot has completed. It reached 100% exact stopping
at every evaluated count through 64, for both selected and final heads. All 44
native checks matched replay. The recipe was frozen before the two optimizer-order repetitions:

```bash
python -m scripts.training.repeat_affine_controller \
  --pilot-evaluation eval/pointer_diagnostics/controller-affine-seed83-20261008T225656.737071Z
```

This reuses the completed seed 83 and its frozen caches, trains seeds 89/97,
performs native checks for each, reports all seeds and fails unless each seed
meets the declared exact-timing thresholds. It does not select a winning seed or
open the reserved confirmation split.

## Completed repair and current use

All three seeds 83/89/97 now pass: exact stopping is 100% at every evaluated count
1–64, with zero early, late or missing stops. Selected and final checkpoints agree
on these outcomes. All 132 predeclared native checks match replay. Validation
joint success and complete nominal trajectories are 2,022/2,048 (98.73%); deep
joint success and trajectories are 1,664/1,664. See the
[report](experiments/controller_affine_seed61.md) for provenance, limits and audits.
This completes the declared development repair, without opening confirmation.

The three complete checkpoints are `models/stage1_pointer/controller-affine-seed83/best`
(and seeds 89/97) on the desktop. Weights and caches are excluded from Git; metrics
and metadata are tracked. Existing outputs refuse overwrite, so the launchers
above reproduce the protocol only in a fresh output location/workspace with its
required source weights and caches.

To inspect actual learned stopping on the existing validation dataset:

```bash
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/controller-affine-seed83/best \
  --data data/pointer/seed-61-independent/validation.jsonl \
  --device cuda --loops 64 --stop-policy completion --stop-threshold 0.5 --test
```

Remove `--test` for all original validation queries. `naive_test` on recurrent
checkpoints forces the requested depth and does not test this controller.
`controller_candidate --native-check` supplies the expanded count panel; historical
GRU-specific fitting/audit commands are not advertised for the affine architecture.
The controller is a learned timer, not an execution-quality evaluator: it receives
neither R's current symbol nor confidence. A future adaptive-quality controller
would be a separate research design.

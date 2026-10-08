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

## Contingent next investigation

If independent initialization supervision is insufficient, broaden count exposure
in frozen prompt features and supervise short recurrent prefixes on those larger
requests. This deliberately changes count coverage while retaining short training
rollouts; it must be documented as such before implementation. Do not substitute
a parsed-count clock, gold memory re-entry, numerical feedback or hand-coded
countdown. An architecture change requires a separate recorded rationale and
contract tests. No such change has been implemented or validated yet.

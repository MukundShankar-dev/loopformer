# Learned affine controller repair — 2026-10-08

The declared development controller repair passes across optimizer seeds 83/89/97:
**100% exact stopping at every evaluated requested count 1–64**, with 132/132
native exported-model checks matching replay. This succeeds after two failed GRU
pilots. It changes the architecture into an explicitly supervised learned timer;
it is not evidence that the previous GRU learned generic adaptive reasoning.

## Frozen executor and data

All runs retain `models/stage1_pointer/executor_r-seed61/step-002250`.
Its adapter payload SHA-256 is
`5a53372639924222266e457e1b1c586aeaa8db7340192c449cb7f8fd288c4e95`.
Qwen2.5-0.5B-Instruct uses pinned revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Prelude layers 0–5, recurrent
layers 6–17, coda layers 18–23, prompt routing and normalized re-entry remain
unchanged. The executor reads Rules/Start with no Steps line. Only the controller
is replaced; 148 non-controller checkpoint tensors were checked bitwise against
the actual source payload for each exported checkpoint.

The dataset is `data/pointer/seed-61-independent`, generator v2: 26-state functional
graphs sampled independently of requested depth, with possible cycles. The original
executor training set has 36,000 independent graphs and requests 1–6/8/10/12.
Controller graph selection uses seed 83 and 256 of its training graphs, paired
with **58 requested counts: 1–63 excluding 9/17/29/41/53** (14,848 training queries).
No validation or deep graph enters the training panel. The initializer fits only
the 58 distinct training-count suffixes; their embeddings are verified bitwise
identical across graph/start variants, so repeating graphs adds no initializer
information. No held-out numeric labels enter the fit.

Controller recurrent training supplies at most **12 loops** per query. Numerical
targets are N−t at t=0..min(N,12); no targets beyond completion are included.
Continue/stop labels are continue at t<N and stop at t=N when observed. Requests
above 12 receive no positive stop label. There is no invented stop at the prefix
cap and no teacher-forced memory reset. The native model never receives these
labels, a parsed count, a loop index or computed remaining count as inputs.

Development validation has 128 independent graphs at counts 1–16 (2,048 queries).
Deep development has 32 different graphs at counts 13–64 (1,664 queries). Counts
13–16 therefore have two graph panels. The 32 deep graphs are repeated at 52
horizons; these are not 1,664 independent graphs. Most deep numeric values were
seen in initialization/prefix training although their full recurrent rollouts were
not. Counts 9/17/29/41/53 are unseen values; 64 is one above the largest seen value.
Reserved test data and seed 29 were not used.

## Why change the controller

The matched [initialization-only pilot](controller_initialization_seed61.md) failed
to repair count-nine or deep timing. The broader-count GRU pilot used the same
58 counts with short prefixes, 256 graphs and 6,000 updates. Selected step 5,800
still achieved only 60.65% short seen-count timing, 8.18% seen deep-value timing,
and 5.63% unseen deep-value timing. Short initialization MAE was 1.743.
It took 144.38 seconds. Its artifact root is
`eval/pointer_diagnostics/controller-prefix-seed83-20261008T224026.497535Z`.

Two no-weight-update probes informed the architecture selection:

- Training-count ridge probes of the frozen P summary, with centered or
  standardized features and ridge coefficient one, still had about 1.5-step
  validation MAE. Feature scaling alone did not resolve this numerical fit.
  Results are in `eval/pointer_diagnostics/controller-prefix-feature-conditioning/summary.json`.
- A training-only least-squares probe of the last eight frozen token embeddings
  recovered counts 1–64, including held-out values, to double precision. It used
  centered features, the first 40 SVD coordinates and rcond 1e-8; numerical feature
  rank was 15. Maximum absolute error was 7.82e-14. Results are in
  `eval/pointer_diagnostics/controller-suffix-initializer-probe/summary.json`.

These exploratory probes ran as desktop stdin Python analyses, not a formal probe
CLI; their JSON method/results are retained. The production initializer below
uses direct least squares and float32 exported weights, rather than treating the
double-precision probe as a native controller result. A failed summary fit does
not prove that the summary contains no useful count information. Changing both
input access and recurrence class does not isolate which change was necessary.

## Architecture and optimization

One raw Rules/Start/Steps/Answer prompt remains the public input. The controller
branch embeds its final eight tokens with frozen Qwen embeddings, flattens them
into 7,168 features, and applies a learned linear initializer to obtain one scalar
memory m0. For the tested one/two-digit template this suffix includes the requested
number and Answer marker, excluding Rules/Start. It does not use P's contextual
summary. It intentionally ignores executor hidden-state observations.

The 7,169 initializer parameters are learned by training-only float64 CPU least
squares (`gelsd`, rcond 1e-8), copied to float32, then frozen. Feature rank including
the intercept is 16; training maximum absolute error is 3.81e-6. This analytic
supervision is not ordinary full-model SFT. Four more learned scalars define:

```text
m_next = a*m + b
stop_probability = sigmoid(w*m_next + c)
```

The shared cell starts at a=1, b=0. No subtraction is programmed. Free-running
remaining-work MSE trains a/b with full BPTT; stop BCE trains w/c on detached
memory and cannot distort the numerical transition. No decoded number/state is
fed back. A fixed identity numerical readout supports existing metric interfaces.
Independent initial MSE remains in the objective but cannot change the already
frozen initializer. Stop threshold stays 0.5.

`configs/controller_affine.json` declares 3,000 AdamW updates, batch 256,
LR .01 for cell offset and stop readout, .0001 for cell gain, warmup 100 and cosine
minimum ratio .1. Remaining-work weight is one, scale 12, initial-loss weight 12.
Checkpoint selection uses exact timing on **seen requests at most 12**, then BCE;
held-out and larger counts do not select checkpoints. Seeds 89/97 change optimizer
sampling order only; data, initializer and recipe are shared. All seeds are
reported rather than selecting a winning seed.

## Exact launch commands

The seed-83 launcher ran from the desktop repository with its active `.venv`:

```bash
bash repair_controller_affine.sh
```

It prepares suffix caches from the existing frozen training/deep caches, trains,
exports and evaluates with `controller_candidate --native-check`. The command
and source hashes are recorded in each `run.json`; the launcher sets deterministic
CUDA workspace configuration and online W&B project `loopformer`.

After the first seed passed, the frozen recipe was repeated:

```bash
source .venv/bin/activate
WANDB_MODE=online CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m scripts.training.repeat_affine_controller \
  --pilot-evaluation eval/pointer_diagnostics/controller-affine-seed83-20261008T225656.737071Z
```

These output paths now exist and refuse overwrite; do not rerun into them.
Reproduction needs the source weights, dataset and previous frozen caches on the
desktop. No model weight or feature cache is committed to Git.

## Results and fidelity

| Optimizer seed | Selected update | Validation exact timing | Deep exact timing | Native matches | Run wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 83 | 3,000 | 2,048/2,048 | 1,664/1,664 | 44/44 | 51.84 s |
| 89 | 3,000 | 2,048/2,048 | 1,664/1,664 | 44/44 | 38.53 s |
| 97 | 3,000 | 2,048/2,048 | 1,664/1,664 | 44/44 | 48.65 s |

Every count passes the predeclared 90% timing threshold; each seed exceeds the
95% overall requirement. Selected and final checkpoints both score 100% timing.
There are zero early, late or missing stops and zero correct-letter/wrong-time
outcomes. Cap fallbacks are excluded from exact-stop success.

All seeds have validation joint success and complete nominal trajectories of
2,022/2,048 (98.7305%); deep joint success and trajectories are 1,664/1,664.
Thus a perfect timer does not erase the frozen executor's 26 validation failures.
Initial count MAE is 9.54e-7 on validation and 1.54e-6 on deep queries. Mean
per-transition countdown error is approximately 1.6e-5 and 9.7e-6 respectively.
Learned gain is 0.999999404 for every seed; offsets are approximately −0.999981.
For seed 83, stop slope/intercept are −3.833384 and 2.678321. These are learned
values, not assigned countdown coefficients.

Native checks were predeclared: first four validation graphs at counts 1/6/9/12/16
and first four deep graphs at 13/17/29/41/53/64. Actual learned-stop exported
inference agrees with replay on prediction, loop count, exact stop, joint success
and missing/cap semantics for every one of the 132 checks. Cached full-panel
quality is not a full native timing benchmark. Reported run times include cached
training/export, not recurrent Qwen backpropagation or a measured adaptive speedup.

Offline saved-decision checks revalidated exact-stop/joint arithmetic on all
22,272 selected/final rows. All 3,132 native transition targets and correctness
flags match independent execution of raw saved rule tables. The audit is at
`eval/pointer_diagnostics/controller-affine-repeats/offline_audit.json`.

Implementation contracts passed: 50 focused tests in 120.55 seconds, including
identity initialization, numerical full-BPTT targets, stop-gradient isolation,
no fabricated prefix completion, checkpoint reload, frozen executor export and
native/cache agreement. Shell syntax and documentation/whitespace checks passed.

## Artifact locations and interpretation

Complete desktop checkpoints are `models/stage1_pointer/controller-affine-seed{83,89,97}/best`.
Each run retains config, commands/source hashes, initializer provenance, metrics,
selection history, remaining-work traces, W&B metadata and `frozen_executor_check.json`.
Caches are `models/stage1_pointer/controller-affine-features.pt` and
`controller-affine-features-deep.pt`, with tracked JSON metadata.

Seed-83 evaluation is
`eval/pointer_diagnostics/controller-affine-seed83-20261008T225656.737071Z`.
The aggregate result, per-count rows and seed-89/97 evaluations are under
`eval/pointer_diagnostics/controller-affine-repeats`.
The [W&B aggregate](https://wandb.ai/mukunds/loopformer/runs/ksuvm8db) records the
three-seed gate; [seed-83 training](https://wandb.ai/mukunds/loopformer/runs/sjw2bi6c)
and [native evaluation](https://wandb.ai/mukunds/loopformer/runs/u0xbrs68) provide
linked run artifacts. No new dependencies were required.

The finite timer repair is complete. Arbitrary integers, alternate wording or
suffix layouts, independent data-seed confirmation, nonrepeating 50-state paths,
confidence-based repair allocation and natural-language transfer remain untested.
A 64-step query over 26 states necessarily revisits states. This controller
learns how many transitions to invoke; it does not inspect whether they are useful
or correct. Review that distinction before selecting another research phase.

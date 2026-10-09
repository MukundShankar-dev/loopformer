# Saved pointer evidence inventory — 2026-10-09

## Purpose and capture

The user requested a full evidence inventory and a paper-style analysis design,
with population coverage for earlier selected architectures and checkpoint
progression only for the successful model. [The active handoff](../analysis.md)
defines model choices, metrics, denominators, figures and pending work.

Mac and desktop both reported Git `b49030e`. Mac tracked work was clean before
this change; desktop had untracked controller weights/logs and one audit CSV.
Those files were preserved. No model was loaded, no training/inference ran, and
no unopened test records were inspected. Inventory reads existing schemas,
actual CSV row counts, JSONL event coverage, metadata and non-binary hashes.
It records binary presence/size without loading or hashing their contents.
This is an availability audit, not another checkpoint-integrity gate.

The implemented collector is [`evidence_inventory.py`](../../scripts/eval/evidence_inventory.py).
Capture artifacts are in [`eval/pointer_analysis/evidence-20261009/`](../../eval/pointer_analysis/evidence-20261009/).
Compressed JSON uses standard gzip and can be read with Python's `gzip`/`json`.
The catalog groups every artifact family; complete snapshots retain individual
paths, schemas, counts and small run/checkpoint metadata. No inference tensors
or raw hidden vectors were copied into Git.
`collector_at_capture.py.gz` preserves the exact source matching both snapshots'
collector hashes. The current utility additionally reports orphan weight
directories; those paths were independently derived into `coverage.json`.

## Availability

| Captured host | Present files under data/models/eval | Checkpoints with config and inference weights | Data/eval/model bytes |
| --- | ---: | ---: | --- |
| Mac | 1,314 | 0 (78 configurations without weights) | 12,412,849 / 291,525,283 / 321,984,749 |
| Desktop | 2,383 | 83 | 127,877,713 / 956,981,884 / 37,137,368,309 |

Counts exclude inventory artifacts themselves and include duplicate exports,
W&B caches and tokenizer payloads. They are not sample sizes. After capture,
the original full benchmark graph JSONL was restored to its normal ignored
data path on the Mac; `coverage.json` records that post-capture file and hash.

The desktop additionally has **65 old adapter-weight directories with no
`recurrent_config.json`** after earlier metadata pruning. The complete path list
is in `coverage.json`. Their mere binary presence does not make them runnable by
the current checkpoint loader; metadata would need recovery from historical Git
before using them. They are not additional selected architectures. No files were
deleted or reconstructed as part of this inventory.

The only differing hash among shared inspected non-binary paths is
`data/pointer/seed-17/manifest.json`: its provenance differs, but split counts
and all recorded split-file hashes match. Do not overwrite it to make Git states
look identical. Desktop-only `controller-training-audit-20261008T221058.031281Z/
predictions.csv` contains 245,760 rows. Its derived tables are already local;
the larger raw CSV remains on the desktop and can be accessed if needed.

## Evidence families: what they establish and what they do not

| Family | Present evidence and scope | Gap or interpretation limit |
| --- | --- | --- |
| Ordinary Qwen baseline | Three saved generation attempts, including completed 60/1,000 run; predictions/summary. | Retry/smoke exports are not additional independent baselines; generation protocol differs from recurrent restricted-symbol readout. |
| Fresh 30k CE model | Training history and monitor traces; six original full evaluations at 2500/3250/3750; diagnostics and recorded causal interventions. | No full 1,350-graph benchmark for the chosen step 2500. Earlier matched plots instead use 3250. |
| Joint completion model | Full training/monitor history, selected-3250 full validation/deep traces and completion logits; small paired population subset. | Population architecture characterization still missing. |
| Fixed-prompt depth 6 | Training monitors, selected-3250 forced/stopped validation/deep calls, fixed-prompt failure report and paired-Step probe. | Hidden norms are not saved in every original full evaluation; subset probe internals do not extend population coverage. |
| Fixed-prompt depth 12 | Training monitors, selected-5000 forced/stopped traces, 4500/5000/5500/7500 development progression and paired-count probes. | Same architecture/different recipe; those 1,000-query depth examples are not paired with seed-17 evaluations or the new cyclic benchmark. |
| Successful isolated R | Executor history, direct-R/C readouts, matched precision/count panel, original validation/deep/stopped evaluations and later full populations. | One executor training seed, several upgrade components changed together. Snapshot population extrapolation has not been measured. |
| Frozen controller development | Original GRU, stop/remaining-work three-seed comparisons, initializer/prefix pilots, gradient/countdown fitting audit, trained/development histories. | Lots of outputs reuse the same frozen R graphs/features. They are not independent executor replications. |
| Positional affine benchmark | 1,350 new graphs × every count 1–256, native regression/fidelity, independent raw-reference audit. | Negative number-reading result from 70; must not erase it or attribute failure to R. |
| Shared-reader benchmark | Separate 1,350 graphs × every count 1–256; native checks, numeric reader/countdown stress. | Distinct graph seeds: direct score deltas across population panels are not paired model effects. |
| Final reader/cell checkpoint | Full reserved-test regression, fresh 1,350 graphs × 256 requests, all intermediate C/direct-R letters, 189 native fidelity calls and audits; numeric timer through 8192. | Numeric timer stress does not execute R. Independent sample size is 1,350 graphs, not 345,600 independent questions. |
| Architecture comparisons | Audited 27-graph/30-count paired results, actual stopping, forced-only N/A semantics, plot-cell audits. | Useful implementation/exploratory evidence; not the main full-population comparison requested now. |
| Rule/cue/restart interventions | Small frozen same-map cohorts with exact modified inputs, reference outputs, attention/linear-probe and some hidden-state records. | Intervention effects are cohort-specific; representational correlations do not prove causal mechanisms. |
| Performance profiles | Baseline timing/operator summaries, batch profiles, saved training resource/throughput logs. | Different scope/device/batches; cannot infer adaptive speedup from reused quality evaluation. |
| Retired/pruned runs | Reports and historical tracked evidence remain in Git; orphan old binaries recorded separately. | The completed loop-balanced ablation was retired on user request. Do not regenerate it or fabricate missing model metadata. |

All families' individual run directories, row counts and schemas are in
`catalog.json.gz`. The raw snapshots additionally inventory local W&B caches;
remote W&B API records and unpushed cloud-only media were not queried. The omitted
124 MB executor launcher log is present on the desktop but is unnecessary for
structured metrics/reference analysis. Non-CSV opaque W&B records are availability
entries, not parsed metric history.

## Dataset inventory and boundaries

Seven dataset roots are present on the desktop: seed 17, seed-37 depth-6 30k,
seed-47 depth-12 gaps, seed-61 independent graphs, and benchmark seeds
211/223/227, 281/283/293, 307/311/313. Each benchmark has 1,350 graph records.
The seed-61 data has 36,000 train examples, 1,536 validation/test queries each
on 128 graphs, and 1,664 deep queries on 32 graphs. These paired queries must
not be counted as independently sampled graphs. Original depth-conditioned
datasets and the later depth-independent cyclic generator have different support.

The collector does not read `test.jsonl` outside the already-opened seed-61 root,
or records under seed-29 roots. File presence/size alone may be recorded. No
seed-29 dataset was found under the captured data roots; its reserved status
does not change. Existing final benchmark graphs are opened evidence. More plots
or historical evaluations on them are retrospective, not fresh confirmation.

## Independent offline observations from the complete final panel

The extra analysis reparsed every graph, checked identity against `graphs.csv`,
executed dictionary transitions through 256, and compared saved C/direct-R letters.
All first-error annotations match. Results are in `full_final_observations.json`:

- 42/1,350 graphs first fail; all first errors are at loops 2–30.
- Full cycles fail on 18/450 graphs (4%; pointwise Wilson 95% interval 2.54–6.23%).
  Permutations and random functions each fail on 12/450 (2.67%; 1.53–4.60%).
  Overlap and only 42 events limit structural interpretation.
- Graph seeds 307/311/313 have 15/15/12 failures, each over 450 graphs.
- At first C error, direct R is correct on 5 graphs; on the other 37 it agrees
  with the same wrong C letter. This localizes readout disagreement on five
  examples; agreement is not a proof of latent-state correctness or error.
- Thirteen first errors are one reference transition ahead; none repeats the
  immediately previous reference state. This differs from a generic "stuck" story.
- Twelve failed-prefix graphs nevertheless give the correct letter at 256.
  Final-letter success does not imply uninterrupted correct execution.
- Mean correct prefix truncated at 256 is 248.365 transitions. This large mean
  is dominated by the 1,308 unfailed graphs; always accompany it with the failure
  distribution and censoring horizon.

Exact-value period/transient strata are retained with denominators, not selected
according to a favorable association. This descriptive reference check neither
fits a model nor proves whether wrong readouts arise from counting, state
transition, memory access or C's projection.

## Reproduction and validation

From each repository root, with its environment active, choose fresh paths:

```bash
python -m scripts.eval.evidence_inventory \
  --label mac --output /tmp/loopformer-mac-evidence-new.json.gz
# On the desktop, use --label desktop with another fresh output path.
```

The desktop capture used the identical collector via a temporary SSH copy,
explicit `--root /home/mukund/loopformer`, and its installed `.venv`. No source
checkout changes were needed for collection. Raw snapshot collector hashes,
Git status, capture times, file hashes and metadata are retained. The grouped
catalog and selected registry were composed from those snapshots; the offline
reference summary used the original benchmark graph bytes.

Schema/coverage capture completed on both hosts. Focused collection tests cover
protected-split non-reading, CSV multiline/gzip coverage and orphan weight reporting.
This turn's final validation is recorded in the active handoff/status after checks.
All **11 focused inventory/comparison/failure tests passed**; compilation and
319 local file/directory documentation links passed, with no whitespace errors.
The full-population architecture inference, expanded scoring and new paper
figures remain **pending**. [Analysis](../analysis.md) is the next bounded work
contract; no new training or later research stage is selected.

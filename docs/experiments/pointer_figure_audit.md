# All pointer figures: independent rescoring and reproduction

Date: 2026-10-08. Status: completed. The user challenged the flat green curve and
the relationship between first intermediate errors and final-answer/exact-stop
success. This audit uses the existing frozen observations. No training, threshold
selection, checkpoint selection or new model inference took place.

## Metric definitions and the disputed curve

For the same graph g evaluated at each requested count N:

- Final-answer/exact-stop success: the C letter at the actual stopping loop equals
  the reference letter after N transitions, **and** the actual first stop is N.
  Earlier incorrect letters can recover without failing this criterion.
- Complete trajectory: every C letter at loops 1 through N equals its corresponding
  reference letter. One earlier error permanently fails this prefix criterion.
- First intermediate error: the first loop whose C letter differs from that loop's
  reference. This is not a separate measurement of first final-answer/timing failure.

At each N, the denominator is **1,350 graphs**, including the 42 graphs with errors.
It is not N, 1,350×N, or only the surviving graphs. The plot is not mean step accuracy
or an aggregate over all requested counts up to N. The controller stops at N on
every graph throughout this panel, so blue joint success equals endpoint letter
accuracy here; this equality need not hold for earlier controllers.

| Requested N | Error-free prefixes | Correct final answer + exact stop | Exact stops |
| --- | --- | --- | --- |
| 12 | 1,323 / 1,350 | 1,330 / 1,350 | 1,350 / 1,350 |
| 30 | 1,308 / 1,350 | 1,320 / 1,350 | 1,350 / 1,350 |
| 64 | 1,308 / 1,350 | 1,319 / 1,350 | 1,350 / 1,350 |
| 128 | 1,308 / 1,350 | 1,320 / 1,350 | 1,350 / 1,350 |
| 256 | 1,308 / 1,350 | 1,320 / 1,350 | 1,350 / 1,350 |

The plateau is therefore supported by independently recomputed numerator counts:
1,308/1,350 = 96.8889% at every N from 30 to 256. The 42 first intermediate errors
occur **across loops 2–30**, not all at loop 30. Already-failed graphs keep executing,
sometimes recovering their endpoint and sometimes failing again. That explains
why endpoint success can fluctuate while strict-prefix success remains fixed.

No discrepancy was found in the original metric data. The presentation needed
more explicit definitions and denominators; the old explanation applied to the
green prefix metric, not the blue final-answer/exact-stop metric.

## Independent validation coverage

The new `scripts/eval/audit_pointer_figures.py` computes raw graph references
without calling the inference scorer or reference executor. It checks:

- 345,600 individual benchmark decision flags, every count and graph identity,
  controller stop flags, and overall rates.
- All **11,776** stratum/count rows, including graph type, seed, transient length
  and cycle period; every denominator and pointwise Wilson bound.
- All **256** loop rows: C/R accuracy, R/C agreement, strict prefixes, conditional
  next-step accuracy and the changing conditional risk-set denominator.
- All **4,050** five-model comparison decisions through the independent raw-rule
  comparison auditor; the native/reuse records are rechecked separately.
- All **12,288** numeric timing decisions (4,096 before precision, 8,192 after),
  flags, first failure and summary totals, plus both 10,000-request reader tables.
- Every exported failure diagnostic: 42 first-error records, 676 confusion cells,
  26 final-letter rates, 6,656 cycle/loop cells, 264 error-episode entries,
  12 category cells and nine post-error dynamics cells.

The original benchmark auditor was also rerun on the desktop with the original
dataset and exclusion bytes: 345,600 decisions pass, with zero overlap against
38,988 excluded rule tables. Native record rechecks pass all **2,751 calls and
45,791 transition rows**: 1,536 reserved-test calls, 189 main fidelity calls and
1,026 comparison calls. These are saved-call audits, not rerun inference.

## Every figure reviewed

All thirteen figures were regenerated as PNG/PDF/SVG and inspected individually.
Existing numerical CSV exports are byte-identical. The raw inference inputs and
observations remain unchanged; fresh hashes and audit receipts accompany the plots.

| Figure | What is checked / clarified |
| --- | --- |
| `quality_over_depth` | Same graphs at each N; blue endpoint+stop, green strict prefix, black exact stop; both scales use the same audited rates |
| `quality_by_graph_type` | Same 450 graphs per type at every N; all type rates and uncertainty bounds independently recomputed |
| `trajectory_failures` | Strict-prefix survival and one first intermediate C error per failing graph; no first-joint-failure interpretation |
| `controller_diagnostics` | Count-reading residual and actual first-stop residual; one controller observation per integer |
| `intermediate_readouts` | Pointwise C/R accuracy and agreement; conditional panel explicitly excludes already-failed graphs |
| `failed_trajectory_matrix` | All 42 failed graphs, correct/wrong C letters at every loop; recovered letters remain visible |
| `symbol_confusion` | One contribution per first error; final-letter denominators use all graphs at loop 256 |
| `failure_signatures` | Mutually exclusive categories, direct R/C first-error agreement, and exhaustive post-error transition counts |
| `cycle_period_matrix` | Raw reachable cycle periods, denominator per row and wrong-letter fraction at every loop |
| `architecture_metric_matrices` | Identical paired graph/count coverage; forced answer, forced prefix, actual stopping and joint success kept separate |
| `requested_actual_stop_matrices` | Actual first stops, distinct missing-stop row, discrete request columns and labeled logarithmic color scale |
| `cyclic_coincidence_and_early_stops` | Wrong-time letters fail joint success; sparse requested counts shown as measured points without interpolation |
| `numeric_precision_comparison` | Left panel explicitly aggregates integer requests 1…N; right panel is pointwise timing drift; neither executes R |

The remaining limitations are unchanged: 26-state graphs eventually cycle,
comparison rates use only 27 graphs, repeated counts are paired, confidence bands
are pointwise, decoded-state observations are not causal hidden-state evidence,
and controller-only numerical stress is not whole-model pointer evaluation.

## Artifacts and reproduction

Updated figures remain in the original locations:

- `eval/pointer_benchmark/final-full-20261008/independent/plots/`
- `eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/`

Each directory now contains `plot_input_audit.json`. Additional source/native
audits and the original/new export hashes are under
`eval/pointer_benchmark/final-full-20261008/figure_audit/`.

On the desktop, from the repository root, render fresh copies:

```bash
python -m scripts.eval.plot_pointer_benchmark \
  --results eval/pointer_benchmark/final-full-20261008/independent \
  --graphs data/pointer/benchmark-seeds307-311-313/graphs.jsonl \
  --output /tmp/pointer-reviewed-overview

python -m scripts.eval.plot_pointer_failures \
  --graphs data/pointer/benchmark-seeds307-311-313/graphs.jsonl \
  --comparison eval/pointer_benchmark/checkpoint-comparison-20261008 \
  --numeric-results eval/pointer_benchmark/shared-number-20261008/numeric \
                    eval/pointer_benchmark/shared-number-precision-20261008/numeric \
  --output /tmp/pointer-reviewed-failures
```

Output directories must be fresh. The Mac reproduction used the same graph bytes
from `/tmp/loopformer-benchmark-graphs/graphs.jsonl`, Matplotlib 3.11.2 and
`MPLCONFIGDIR=/tmp/loopformer-mpl-cache`, initially rendering into
`/tmp/loopformer-figure-final-export/` before replacing only the derived figure exports.

Validation: six new audit tests and fifteen relevant existing tests pass. The
new tests cover recovered endpoint versus permanently failed prefix, corrupt
stratum rates, Wilson bounds, conditional denominators, pointwise loop rates
and individual numeric timing flags. Audit/source hashes, local documentation
links and `git diff --check` pass. Original numerical outcomes are unchanged.

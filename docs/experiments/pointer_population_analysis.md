# Full pointer population analysis

Status: **resumed after CE completed; final population results and replacement figures pending**.

The selected CE checkpoint passed 54 native comparisons with exact decoded
agreement (maximum absolute logit difference 0.0000496). Full extraction is
complete for all 1,350 CE graphs. After the requested pause, the user authorized
resumption: CE was skipped and joint-full native checking is running. See the
[pause/resume handoff](../analysis.md#pause-and-resume-handoff--2026-10-09).
Remaining models and nine successful R snapshots follow on resume, then
independent scoring/audits and figure replacement. No new training
or checkpoint/threshold selection occurs.

```bash
bash analyze_pointer.sh
python -m scripts.eval.paper_status
```

The default output is `eval/pointer_analysis/paper-20261009/`. The launcher resumes
unchanged chunks; do not launch a second instance while it is running. `run.log`,
`RUNNING`, `FAILED`, and `COMPLETE` record pipeline state. The completed runner
rewrites this report from audited outcomes and generates `plots/index.md`,
PNG/PDF/SVG, plotted numerical arrays, hashes and export-retirement records.

See [the analysis contract](../analysis.md) for models, architecture/source maps,
metrics, uncertainties and interpretive limits. The panel remains retrospective:
all 1,350 graphs are opened, and finite 26-state cyclic execution does not prove
unlimited-depth, larger-state or cross-family generalization.

Validation: 28 focused tests pass. A separate synthetic reporting fixture
independently checked 2,073,600 decisions and 15,360 aggregate cells and rendered
twenty-one figure families. This is reporting validation, not historical-model
population evidence. Layout inspection found and fixed redundant cohort curves,
overlong overlap labels and clipped sparse-bin uncertainty intervals.

The queued plot bundle additionally includes two observed-policy Pareto figures,
with exact-stop quality versus logical recurrent passes. This addition uses
saved audited outcomes, not another GPU run; it does not estimate comparable
latency from extraction timings. [Output map](../analysis.md#queued-output-organization-and-pareto-comparisons).

The ordinary base checkpoint is now queued on all 345,600 current questions,
under a separate frozen three-shot generation configuration. Its full-response
independent audit, paired final-letter comparison and graph-stratum summaries
will be included before final rendering/reporting. Results are pending. The old
seed-17 6% baseline remains historical context. Ordinary generation has no
recurrent stopping-loop/trajectory or R-pass Pareto metric. The clean bundle now
requires 25 families. [Baseline contract](../analysis.md#ordinary-qwen-on-the-current-benchmark--2026-10-09).

The new baseline/reporting contracts passed 37 focused tests on both hosts,
including actual tiny-model FP32/SDPA generation, model-free resume and corrupted
raw-evidence rejection. Both added layouts were inspected in a synthetic
25-family rendering fixture; this does not supply pretrained baseline metrics.

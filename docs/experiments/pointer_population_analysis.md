# Full pointer population analysis

Status: **running; population results and final replacement figures pending**.

The selected CE checkpoint passed 54 native comparisons with exact decoded
agreement (maximum absolute logit difference 0.0000496). Full extraction is
running on the CUDA desktop; remaining models and nine successful R snapshots
follow, then independent scoring/audits and figure replacement. No new training
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

Validation: 24 focused tests pass. A separate synthetic reporting fixture
independently checked 2,073,600 decisions and 15,360 aggregate cells and rendered
fifteen figure families. This is reporting validation, not historical-model
population evidence. Layout inspection found and fixed redundant cohort curves,
overlong overlap labels and clipped sparse-bin uncertainty intervals.

# Data

The raw dataset is **not committed** (it is ~618 MB and lives under the gitignored
`data/raw/`). To reproduce the pipeline, place the dataset file here:

```
data/raw/multi-smell-dataset-v1_2.csv
```

## Source

- Dataset: `multi-smell-dataset` (v1.2), a multi-label code-smell dataset.
- Source / licence / exact citation: **TBD - filled during the Phase 1 audit.**
- Granularity, languages, label source, and how smell labels were produced
  (manual vs tool/rule-based) are documented in `ml/reports/dataset_audit.md`
  after Phase 1, and summarised in `CLAUDE.md` Section 2.

## Circularity note

If the smell labels were produced by a rule/metric-threshold tool, models trained
on the same metrics can trivially rediscover the rule (label leakage). The Phase 1
audit records the verdict and, if needed, Phase 5 runs an ablation dropping the
rule-defining metrics. See `CLAUDE.md` Section 2.2.

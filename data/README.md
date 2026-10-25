# Data

The raw dataset is **not committed** (it is ~618 MB and lives under the gitignored
`data/raw/`). To reproduce the pipeline, place the dataset file here:

```
data/raw/multi-smell-dataset-v1_2.csv
```

## Source

- Dataset: **SmellyCode++** (v1.2), a multi-label code-smell dataset for Java.
- Citation: *SmellyCode++: Multi-Label Dataset for Code Smell Detection*, KFUPM,
  published in *Scientific Data* (July 2025).
- Landing pages:
  - <https://pmc.ncbi.nlm.nih.gov/articles/PMC12255726/>
  - <https://pure.kfupm.edu.sa/en/publications/smellycode-multi-label-dataset-for-code-smell-detection/>
- Licence: check the Scientific Data article (CC-BY likely); confirm before any redistribution.
- Content: 107,554 rows over 89 Java OSS projects; 14 numeric metrics (Halstead +
  cyclomatic + logical lines), 4 binary smell labels (Long method, God class,
  Feature envy, Data class), plus `File`/`Project`/`Class` identifiers and a raw
  `Code` source column.
- Full profile: `ml/reports/dataset_audit.md`; summary in `CLAUDE.md` Section 2.

## Circularity note

If the smell labels were produced by a rule/metric-threshold tool, models trained
on the same metrics can trivially rediscover the rule (label leakage). The Phase 1
audit records the verdict and, if needed, Phase 5 runs an ablation dropping the
rule-defining metrics. See `CLAUDE.md` Section 2.2.

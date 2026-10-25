"""Phase 4 - RQ2 statistics and thresholds.

Per consensus metric: per-class descriptive stats, density + box plots (with
correct axis labels), Mann-Whitney U (balanced undersample + full data),
rank-biserial effect size, Holm/Bonferroni correction, and warning/critical
thresholds (smelly median / smelly Q3). Fix track: evaluate each threshold as a
classifier on CV test folds and add a Youden-J threshold.

Writes rq2_stats.json, rq2_tests.json, rq2_thresholds.json.

Run:  python -m ml.src.rq2_statistics
"""


def main() -> None:
    raise NotImplementedError("Implemented in Phase 4.")


if __name__ == "__main__":
    main()

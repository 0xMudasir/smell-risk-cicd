"""Phase 1 - Dataset audit.

Reads data/raw/<dataset>, writes ml/reports/dataset_audit.md with: shape, dtypes,
missing values, duplicates, per-smell positive rate, smell co-occurrence matrix,
per-project counts, date range, the metric list mapped to the paper's 25
Understand metrics, the label source, and the circularity verdict (Section 2.2).

Run:  python -m ml.src.audit
"""


def main() -> None:
    raise NotImplementedError("Implemented in Phase 1.")


if __name__ == "__main__":
    main()

"""Phase 2 - Preprocessing.

Cleans the raw dataset (dedupe exact duplicates, build the `has_smell` target,
separate identifiers / categoricals / numeric metrics) and saves a tidy parquet
for the downstream RQ scripts. Leakage rules from CLAUDE.md Section 4.3 apply.

Run:  python -m ml.src.preprocess
"""


def main() -> None:
    raise NotImplementedError("Implemented in Phase 2.")


if __name__ == "__main__":
    main()

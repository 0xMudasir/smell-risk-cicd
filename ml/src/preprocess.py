"""Phase 2 - Preprocessing.

Loads the raw SmellyCode++ CSV (without the `Code` source column), builds the
`has_smell` target, validates columns, and saves a tidy parquet for the RQ
scripts. Per D1.5 no rows are dropped (exact duplicates are genuinely distinct
tiny methods; grouped-by-project CV handles leakage). Leakage rules from
CLAUDE.md Section 4.3 apply downstream (SMOTE only inside training folds, etc.).

Run:  python -m ml.src.preprocess
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pandas as pd

from .config import ROOT, load_config, p

CFG = load_config()


def _read_header(path) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def load_raw() -> pd.DataFrame:
    raw = p(CFG["paths"]["raw"])
    source_col = CFG["dataset"]["source_column_not_loaded"]
    header = _read_header(raw)
    usecols = [c for c in header if c != source_col]
    return pd.read_csv(raw, usecols=usecols, low_memory=False)


def build(df: pd.DataFrame) -> pd.DataFrame:
    cols = CFG["columns"]
    target = CFG["target"]
    smells = target["smell_label_columns"]
    metrics = cols["numeric_metrics"]
    ids = cols["identifiers"]

    # --- validate expected columns exist ---
    missing_cols = [c for c in (ids + metrics + smells) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected columns missing from dataset: {missing_cols}")

    # --- coerce types ---
    for s in smells:
        df[s] = df[s].astype("int8")
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="raise")

    # --- integrity checks (fail loudly rather than silently) ---
    if df[metrics].isna().any().any():
        raise ValueError("NaNs found in metric columns; audit reported none - investigate.")
    neg = {m: int((df[m] < 0).sum()) for m in metrics}
    neg = {m: n for m, n in neg.items() if n}
    if neg:
        raise ValueError(f"Negative values in metrics (unexpected for size/count metrics): {neg}")

    # --- primary target ---
    df[target["name"]] = (df[smells].sum(axis=1) > 0).astype("int8")

    # --- column order: identifiers, metrics, smells, target ---
    ordered = ids + metrics + smells + [target["name"]]
    return df[ordered].reset_index(drop=True)


def save(df: pd.DataFrame) -> dict:
    out = p(CFG["paths"]["clean_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    cols = CFG["columns"]
    target = CFG["target"]["name"]
    smells = CFG["target"]["smell_label_columns"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clean_parquet": CFG["paths"]["clean_parquet"],
        "rows": int(len(df)),
        "deduplicated": CFG["preprocess"]["deduplicate"],
        "n_features": len(cols["numeric_metrics"]),
        "feature_columns": cols["numeric_metrics"],
        "identifier_columns": cols["identifiers"],
        "smell_label_columns": smells,
        "target": target,
        "target_positive": int(df[target].sum()),
        "target_rate": float(df[target].mean()),
        "per_smell_rate": {s: float(df[s].mean()) for s in smells},
        "n_projects": int(df[CFG["columns"]["project_column"]].nunique()),
    }
    summ_path = p(CFG["paths"]["artifacts"]) / "preprocess_summary.json"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    summ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print("Loading raw dataset (Code column skipped) ...")
    df = load_raw()
    df = build(df)
    summary = save(df)
    print(f"Saved {summary['rows']:,} rows x {summary['n_features']} features -> "
          f"{summary['clean_parquet']}")
    print(f"Target '{summary['target']}' positive rate: {summary['target_rate']*100:.2f}% "
          f"over {summary['n_projects']} projects")


if __name__ == "__main__":
    main()

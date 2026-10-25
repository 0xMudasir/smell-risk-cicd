"""Phase 2 - Cross-validation split factory.

Per CLAUDE.md Section 4.2, this dataset has no date column but has a Project
column, so the chosen strategy is:

    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), groups=Project

so that no project appears in both train and test (realistic generalisation) and
the class ratio is preserved per fold. Recorded in docs/DECISIONS.md (D2.*).

Also provides `split_sanity()` which prints per-fold sizes, class ratios and
confirms zero project overlap between train and test.

Run:  python -m ml.src.cv            # loads clean parquet and prints the sanity report
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from .config import CONFIG_PATH, load_config, p

CFG = load_config()


def make_splitter():
    """Return (splitter, strategy_name) chosen from config (Section 4.2)."""
    cv = CFG["cv"]
    n = cv["n_splits"]
    seed = CFG["seed"]
    strategy = cv["strategy"]
    if strategy == "time_series":
        return TimeSeriesSplit(n_splits=n), "TimeSeriesSplit"
    if strategy == "grouped":
        return (
            StratifiedGroupKFold(n_splits=n, shuffle=cv.get("shuffle", True), random_state=seed),
            "StratifiedGroupKFold",
        )
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=seed), "StratifiedKFold"
    raise ValueError(f"Unknown cv.strategy: {strategy!r}")


def get_groups(df: pd.DataFrame):
    """Return the group array (Project) or None, per strategy."""
    if CFG["cv"]["strategy"] == "grouped":
        return df[CFG["cv"]["group_column"]].to_numpy()
    return None


def split_indices(df: pd.DataFrame):
    """Yield (fold, train_idx, test_idx) using the configured splitter."""
    splitter, _ = make_splitter()
    y = df[CFG["target"]["name"]].to_numpy()
    groups = get_groups(df)
    X = np.zeros((len(df), 1))  # splitter only needs shape + y (+ groups)
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        yield fold, tr, te


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(p(CFG["paths"]["clean_parquet"]))


def split_sanity(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = load_clean()
    target = CFG["target"]["name"]
    group_col = CFG["cv"].get("group_column")
    _, name = make_splitter()
    overall_rate = df[target].mean()

    print(f"CV strategy: {name} (n_splits={CFG['cv']['n_splits']}, seed={CFG['seed']})")
    print(f"Rows: {len(df):,}   overall smelly rate: {overall_rate*100:.2f}%")
    print(f"{'fold':>4} {'train':>9} {'test':>9} {'tr_pos%':>8} {'te_pos%':>8} "
          f"{'tr_proj':>8} {'te_proj':>8} {'overlap':>8}")
    all_ok = True
    for fold, tr, te in split_indices(df):
        tr_rate = df.iloc[tr][target].mean() * 100
        te_rate = df.iloc[te][target].mean() * 100
        if group_col:
            tr_g = set(df.iloc[tr][group_col].unique())
            te_g = set(df.iloc[te][group_col].unique())
            overlap = len(tr_g & te_g)
            tr_ng, te_ng = len(tr_g), len(te_g)
        else:
            overlap, tr_ng, te_ng = 0, "-", "-"
        all_ok &= (overlap == 0)
        print(f"{fold:>4} {len(tr):>9,} {len(te):>9,} {tr_rate:>8.2f} {te_rate:>8.2f} "
              f"{tr_ng:>8} {te_ng:>8} {overlap:>8}")
    if group_col:
        print("No project appears in both train and test:" , "PASS" if all_ok else "FAIL")


def main() -> None:
    split_sanity()


if __name__ == "__main__":
    main()

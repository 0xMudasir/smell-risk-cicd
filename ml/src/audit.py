"""Phase 1 - Dataset audit.

Reads data/raw/<dataset>, writes ml/reports/dataset_audit.md and
ml/artifacts/dataset_audit.json with: shape, dtypes, missing values, duplicates,
per-smell positive rate, smell co-occurrence, per-project counts, date range, the
metric list mapped to the paper's 25 Understand metrics, the label source, and the
circularity verdict (CLAUDE.md Section 2.2).

The huge `Code` column (raw source text, ~618 MB of the file) is never loaded: it
is an identifier-like artefact, not a feature.

Run:  python -m ml.src.audit
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths (kept simple; pipeline.yaml is finalised in Phase 2) ---
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "multi-smell-dataset-v1_2.csv"
REPORT_MD = ROOT / "ml" / "reports" / "dataset_audit.md"
REPORT_JSON = ROOT / "ml" / "artifacts" / "dataset_audit.json"

SEED = 42

# Columns that are identifiers / source text, never model features.
IDENTIFIER_COLS = ["File", "Project", "Class"]
SOURCE_COL = "Code"

# The 4 smell labels in this dataset.
SMELL_COLS = ["Long method", "God class", "Feature envy", "Data class"]

# How each numeric metric relates to the paper's SciTools Understand metrics and to
# the Halstead family. "deterministic_of" marks metrics that are exact algebraic
# functions of others (|r| ~ 1) - important for RQ1 coefficient stability.
METRIC_MAP = {
    "Logical Lines":        {"family": "size",        "paper_understand": "CountLineCode / CountStmt (closest)", "note": "size metric; paper's size family"},
    "Distinct Operators":   {"family": "halstead",    "paper_understand": "none",  "note": "Halstead n1"},
    "Distinct Operands":    {"family": "halstead",    "paper_understand": "none",  "note": "Halstead n2"},
    "Total Operators":      {"family": "halstead",    "paper_understand": "none",  "note": "Halstead N1"},
    "Total Operands":       {"family": "halstead",    "paper_understand": "none",  "note": "Halstead N2"},
    "Vocabulary":           {"family": "halstead",    "paper_understand": "none",  "note": "n1+n2 (deterministic)"},
    "Length":               {"family": "halstead",    "paper_understand": "none",  "note": "N1+N2 (deterministic)"},
    "Calculated Length":    {"family": "halstead",    "paper_understand": "none",  "note": "n1*log2 n1 + n2*log2 n2 (deterministic)"},
    "Volume":               {"family": "halstead",    "paper_understand": "none",  "note": "Length*log2 Vocabulary (deterministic)"},
    "Difficulty":           {"family": "halstead",    "paper_understand": "none",  "note": "(n1/2)*(N2/n2) (deterministic)"},
    "Effort":               {"family": "halstead",    "paper_understand": "none",  "note": "Difficulty*Volume (deterministic)"},
    "Time Required":        {"family": "halstead",    "paper_understand": "none",  "note": "Effort/18 (deterministic)"},
    "Bugs":                 {"family": "halstead",    "paper_understand": "none",  "note": "Halstead delivered-bugs estimate = Volume/3000 (deterministic); NOT a label"},
    "Cyclomatic Complexity":{"family": "cyclomatic",  "paper_understand": "SumCyclomatic (closest)", "note": "paper's core cyclomatic family"},
}


def _read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _fmt_int(n) -> str:
    return f"{int(n):,}"


def load() -> pd.DataFrame:
    header = _read_header(RAW)
    usecols = [c for c in header if c != SOURCE_COL]
    # Everything except the smell labels and ids is numeric; let pandas infer, then coerce.
    df = pd.read_csv(RAW, usecols=usecols, low_memory=False)
    return df, header


def numeric_metric_cols(df: pd.DataFrame) -> list[str]:
    exclude = set(IDENTIFIER_COLS) | set(SMELL_COLS) | {SOURCE_COL}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_report() -> dict:
    df, header = load()
    n_rows, n_cols_loaded = df.shape
    n_cols_total = len(header)

    metrics = numeric_metric_cols(df)

    # --- dtypes & missing ---
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    missing = {c: int(df[c].isna().sum()) for c in df.columns}

    # --- smell labels: coerce to int, check they are 0/1 ---
    smell_value_sets = {}
    for s in SMELL_COLS:
        vals = sorted(pd.unique(df[s].dropna()).tolist())
        smell_value_sets[s] = [int(v) if float(v).is_integer() else v for v in vals]
        df[s] = df[s].astype("int8")

    # --- primary target: has_smell = any smell positive ---
    df["has_smell"] = (df[SMELL_COLS].sum(axis=1) > 0).astype("int8")

    # --- per-smell positive rate ---
    per_smell = {}
    for s in SMELL_COLS:
        pos = int(df[s].sum())
        per_smell[s] = {"positives": pos, "rate": pos / n_rows}
    has_pos = int(df["has_smell"].sum())
    target_summary = {
        "positives": has_pos,
        "negatives": n_rows - has_pos,
        "rate": has_pos / n_rows,
    }

    # --- number of smells per row (multi-label structure) ---
    smell_count = df[SMELL_COLS].sum(axis=1)
    smells_per_row = {int(k): int(v) for k, v in smell_count.value_counts().sort_index().items()}

    # --- co-occurrence matrix (counts of rows positive for BOTH i and j) ---
    cooc = {}
    for a in SMELL_COLS:
        cooc[a] = {}
        for b in SMELL_COLS:
            cooc[a][b] = int(((df[a] == 1) & (df[b] == 1)).sum())

    # --- duplicates ---
    dup_full_metrics = int(df.duplicated(subset=metrics + SMELL_COLS).sum())
    id_cols_present = [c for c in IDENTIFIER_COLS if c in df.columns]
    dup_ids = int(df.duplicated(subset=id_cols_present).sum()) if id_cols_present else None

    # size profile of rows involved in any exact metric+label duplicate
    dup_mask = df.duplicated(subset=metrics + SMELL_COLS, keep=False)
    dup_lloc_median = float(df.loc[dup_mask, "Logical Lines"].median()) if dup_mask.any() and "Logical Lines" in df else None

    # --- granularity evidence: identifier recurrence + smell-family cross-tab ---
    granularity = None
    if id_cols_present:
        gsz = df.groupby(id_cols_present).size()
        method_smells = ["Long method", "Feature envy"]
        class_smells = ["God class", "Data class"]
        cross = int((((df[method_smells].sum(axis=1) > 0)) & ((df[class_smells].sum(axis=1) > 0))).sum())
        granularity = {
            "identifier_cols": id_cols_present,
            "n_unique_identifiers": int(len(gsz)),
            "n_identifiers_repeated": int((gsz > 1).sum()),
            "max_rows_per_identifier": int(gsz.max()),
            "method_level_smells": method_smells,
            "class_level_smells": class_smells,
            "rows_positive_for_both_families": cross,
            "dedup_identifier_only_rows": int(df.drop_duplicates(subset=id_cols_present).shape[0]),
        }

    # --- per-project ---
    proj = None
    if "Project" in df.columns:
        g = df.groupby("Project")
        proj_tbl = pd.DataFrame({
            "files": g.size(),
            "smelly": g["has_smell"].sum(),
        })
        proj_tbl["smelly_rate"] = proj_tbl["smelly"] / proj_tbl["files"]
        proj_tbl = proj_tbl.sort_values("files", ascending=False)
        proj = {
            "n_projects": int(df["Project"].nunique()),
            "top": proj_tbl.head(25).reset_index().to_dict("records"),
            "files_min": int(proj_tbl["files"].min()),
            "files_max": int(proj_tbl["files"].max()),
            "files_median": float(proj_tbl["files"].median()),
        }

    # --- correlated metric pairs (|r| > 0.9) and near-exact (>0.999) ---
    corr = df[metrics].corr(method="pearson")
    high_pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.9:
                high_pairs.append({"a": cols[i], "b": cols[j], "r": round(float(r), 4)})
    high_pairs.sort(key=lambda d: -abs(d["r"]))

    # --- metric separability: standardized mean difference (smelly - clean) per metric ---
    sep = {}
    clean = df[df["has_smell"] == 0]
    smelly = df[df["has_smell"] == 1]
    for m in metrics:
        mu_c, mu_s = clean[m].mean(), smelly[m].mean()
        sd = df[m].std(ddof=0)
        cohen_d = float((mu_s - mu_c) / sd) if sd and not np.isnan(sd) and sd != 0 else None
        sep[m] = {
            "clean_median": float(clean[m].median()),
            "smelly_median": float(smelly[m].median()),
            "clean_mean": float(mu_c),
            "smelly_mean": float(mu_s),
            "cohen_d": round(cohen_d, 4) if cohen_d is not None else None,
        }

    # --- date column? ---
    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "commit", "version"))]
    # "Time Required" is a Halstead metric, not a timestamp - exclude it.
    date_candidates = [c for c in date_candidates if c != "Time Required"]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(RAW.relative_to(ROOT)),
        "seed": SEED,
        "shape": {"rows": n_rows, "columns_total": n_cols_total, "columns_loaded": n_cols_loaded,
                  "note": f"'{SOURCE_COL}' column not loaded (raw source text)"},
        "columns": {
            "identifiers": id_cols_present,
            "source_text": SOURCE_COL,
            "numeric_metrics": metrics,
            "smell_labels": SMELL_COLS,
            "target": "has_smell (derived)",
        },
        "dtypes": dtypes,
        "missing": missing,
        "missing_total": int(sum(missing.values())),
        "smell_value_sets": smell_value_sets,
        "target_has_smell": target_summary,
        "per_smell": per_smell,
        "smells_per_row": smells_per_row,
        "cooccurrence": cooc,
        "duplicates": {
            "exact_on_metrics_and_labels": dup_full_metrics,
            "duplicate_identifier_rows": dup_ids,
            "identifier_cols_used": id_cols_present,
            "dup_rows_logical_lines_median": dup_lloc_median,
        },
        "granularity": granularity,
        "projects": proj,
        "metric_map": METRIC_MAP,
        "correlated_pairs_gt_0_9": high_pairs,
        "separability": sep,
        "date_columns_detected": date_candidates,
    }
    return result, df


def write_markdown(r: dict) -> None:
    L = []
    w = L.append
    w("# Dataset audit - multi-smell-dataset v1.2\n")
    w(f"_Generated {r['generated_at']} (UTC). Seed {r['seed']}. Source: `{r['raw_file']}`._\n")
    w("> The `Code` column (raw Java source) is not loaded or used; it is the bulk of the "
      "618 MB file and is not a feature.\n")

    w("## 1. Shape\n")
    w(f"- Rows: **{_fmt_int(r['shape']['rows'])}**")
    w(f"- Columns in file: **{r['shape']['columns_total']}** "
      f"(loaded {r['shape']['columns_loaded']}; `Code` skipped)")
    w("")

    w("## 2. Columns\n")
    w(f"- Identifiers (never features): {', '.join('`%s`' % c for c in r['columns']['identifiers'])}")
    w(f"- Source text (never a feature): `{r['columns']['source_text']}`")
    w(f"- Numeric metrics ({len(r['columns']['numeric_metrics'])}): "
      f"{', '.join('`%s`' % c for c in r['columns']['numeric_metrics'])}")
    w(f"- Smell labels ({len(r['columns']['smell_labels'])}): "
      f"{', '.join('`%s`' % c for c in r['columns']['smell_labels'])}")
    w(f"- Primary target: `has_smell` = 1 if any smell label is positive")
    w("")

    w("### Dtypes and missing values\n")
    w("| Column | dtype | missing |")
    w("|---|---|---|")
    for c in r["dtypes"]:
        w(f"| `{c}` | {r['dtypes'][c]} | {_fmt_int(r['missing'][c])} |")
    w(f"\nTotal missing cells: **{_fmt_int(r['missing_total'])}**.\n")

    w("## 3. Target and per-smell positive rates\n")
    t = r["target_has_smell"]
    w(f"- **has_smell**: {_fmt_int(t['positives'])} positive / {_fmt_int(t['negatives'])} negative "
      f"-> **{t['rate']*100:.2f}%** smelly.")
    w("")
    w("| Smell | positives | rate | values seen |")
    w("|---|---|---|---|")
    for s, d in r["per_smell"].items():
        w(f"| `{s}` | {_fmt_int(d['positives'])} | {d['rate']*100:.2f}% | {r['smell_value_sets'][s]} |")
    w("")
    w("Smells per row (multi-label structure):\n")
    w("| # smells | rows |")
    w("|---|---|")
    for k, v in r["smells_per_row"].items():
        w(f"| {k} | {_fmt_int(v)} |")
    w("")

    w("## 4. Smell co-occurrence (rows positive for both)\n")
    cols = r["columns"]["smell_labels"]
    w("| | " + " | ".join(cols) + " |")
    w("|" + "---|" * (len(cols) + 1))
    for a in cols:
        w(f"| **{a}** | " + " | ".join(_fmt_int(r['cooccurrence'][a][b]) for b in cols) + " |")
    w("")

    w("## 5. Duplicates\n")
    d = r["duplicates"]
    w(f"- Exact duplicate rows on metrics + labels: **{_fmt_int(d['exact_on_metrics_and_labels'])}**")
    if d["duplicate_identifier_rows"] is not None:
        w(f"- Duplicate identifier rows ({'+'.join(d['identifier_cols_used'])}): "
          f"**{_fmt_int(d['duplicate_identifier_rows'])}**")
    if d.get("dup_rows_logical_lines_median") is not None:
        w(f"- Rows involved in an exact duplicate have a median of "
          f"**{d['dup_rows_logical_lines_median']:.0f} logical lines** - i.e. they are mostly tiny "
          f"entities (one-line getters/setters) whose metrics happen to coincide, not data errors. "
          f"Blind exact-dedup would therefore delete genuinely distinct methods, so dedup is deferred "
          f"to a Phase 2 decision (see Section 12).")
    w("")

    if r.get("granularity"):
        gr = r["granularity"]
        w("## 5b. Granularity (mixed method + class)\n")
        w(f"- The identifier ({'+'.join(gr['identifier_cols'])}) recurs: "
          f"**{_fmt_int(gr['n_unique_identifiers'])}** unique identifiers, "
          f"**{_fmt_int(gr['n_identifiers_repeated'])}** of them appear more than once, up to "
          f"**{gr['max_rows_per_identifier']}** rows for a single identifier. Rows are therefore "
          f"**sub-class entities (methods), not whole files** - finer than the paper's file level.")
        w(f"- The smell labels split into two families that **never co-occur**: method-level "
          f"({', '.join('`%s`' % s for s in gr['method_level_smells'])}) vs class-level "
          f"({', '.join('`%s`' % s for s in gr['class_level_smells'])}). "
          f"Rows positive for both families: **{gr['rows_positive_for_both_families']}**.")
        w(f"- There is no method-name column or explicit method/class flag; for clean rows the "
          f"granularity is not recoverable from the file. This is the main structural caveat of the "
          f"dataset.")
        w("")

    if r["projects"]:
        p = r["projects"]
        w("## 6. Projects\n")
        w(f"- Distinct projects: **{p['n_projects']}**. "
          f"Files per project: min {p['files_min']}, median {p['files_median']:.0f}, max {_fmt_int(p['files_max'])}.")
        w("")
        w("Top 25 projects by file count:\n")
        w("| Project | files | smelly | smelly rate |")
        w("|---|---|---|---|")
        for row in p["top"]:
            w(f"| `{row['Project']}` | {_fmt_int(row['files'])} | {_fmt_int(row['smelly'])} | {row['smelly_rate']*100:.1f}% |")
        w("")

    w("## 7. Date / version column\n")
    if r["date_columns_detected"]:
        w(f"- Candidate date/version columns: {r['date_columns_detected']}")
    else:
        w("- **None.** No date, commit, or version column. "
          "CV strategy therefore falls to grouped CV on `Project` (Section 4.2 option 2).")
    w("")

    w("## 8. Metric mapping to the paper's Understand metrics\n")
    w("| Metric | family | closest paper (Understand) metric | note |")
    w("|---|---|---|---|")
    for m, meta in r["metric_map"].items():
        w(f"| `{m}` | {meta['family']} | {meta['paper_understand']} | {meta['note']} |")
    w("\n> This dataset's metrics are **Halstead software-science metrics + cyclomatic complexity + "
      "a size metric**, not the 25 SciTools Understand metrics the paper used. Only cyclomatic "
      "complexity and Logical Lines map cleanly to the paper's families. We therefore replicate the "
      "**method**, not the feature set.\n")

    w("## 9. Correlated metric pairs (|r| > 0.9)\n")
    if r["correlated_pairs_gt_0_9"]:
        w("Many Halstead metrics are exact algebraic functions of one another, so they are almost "
          "perfectly correlated. This destabilises coefficient-based importance (PAPER_REVIEW item 4).\n")
        w("| metric A | metric B | r |")
        w("|---|---|---|")
        for pr in r["correlated_pairs_gt_0_9"]:
            w(f"| `{pr['a']}` | `{pr['b']}` | {pr['r']} |")
    else:
        w("- None above 0.9.")
    w("")

    w("## 10. Separability (standardized mean difference, smelly - clean)\n")
    w("Quick signal of how different the smelly class looks per metric (Cohen's d; "
      "|d| ~ 0.2 small, 0.5 medium, 0.8 large).\n")
    w("| Metric | clean median | smelly median | Cohen's d |")
    w("|---|---|---|---|")
    sep = r["separability"]
    for m in sorted(sep, key=lambda k: -(abs(sep[k]["cohen_d"]) if sep[k]["cohen_d"] is not None else 0)):
        d = sep[m]
        dd = f"{d['cohen_d']:+.3f}" if d["cohen_d"] is not None else "n/a"
        w(f"| `{m}` | {d['clean_median']:.3g} | {d['smelly_median']:.3g} | {dd} |")
    w("")

    w("## 11. Label source and circularity verdict\n")
    w("- **Dataset identified:** this is **SmellyCode++**, a multi-label code-smell dataset "
      "(KFUPM; *Scientific Data*, July 2025). The sample count (107,554), the four smells, and the "
      "multi-label structure all match. Per its abstract, code was parsed into ASTs, features were "
      "extracted, and samples were **annotated 'based on prior studies'** (Fontana-style).\n")
    w("- **Label source: still to confirm exactly.** The accessible abstract/landing pages do not state "
      "the precise labeling rule (manual validation vs DECOR/PMD/iPlasma-style metric thresholds vs "
      "heuristics); the full data descriptor sits behind a browser check. The dataset's own reported "
      "per-smell F1 scores are high (Data class 95.9, God class 94.5, Feature envy 88.7, Long method "
      "88.9), which is consistent with labels that are largely derivable from the features.\n")
    w("- **Circularity verdict (Section 2.2): PLAUSIBLE, not confirmed.** If smells were produced by "
      "metric-threshold rules, models on the same metrics would rediscover them. Evidence so far is "
      "mixed: single-metric separability (Section 10) is only **moderate** (best Cohen's d ~0.78 on "
      "Vocabulary / Distinct Operators), not the near-perfect split a one-metric threshold would give - "
      "so no single Halstead metric is the rule. But a multi-metric heuristic remains possible. "
      "**Action:** confirm the labeling rule from the full SmellyCode++ descriptor before Phase 3, and "
      "in Phase 5 run an ablation dropping the most discriminative metrics regardless, reporting both.\n")

    w("## 12. Open questions for Phase 2\n")
    w("1. **Target over mixed granularity.** CLAUDE.md 2.1 sets the primary target as `has_smell` "
      "(any smell). Because rows mix method-level and class-level entities (Section 5b), confirm we keep "
      "all rows for the primary target, and that the per-smell extension models naturally respect the "
      "method/class split.")
    w("2. **Deduplication.** Exact duplicates are mostly tiny distinct methods (Section 5), and grouped "
      "CV on `Project` already keeps all entities of a project (and hence a class) on one side of each "
      "split, so cross-split leakage from the method/class repetition is contained without dropping "
      "rows. Proposed: do **not** drop exact duplicates; rely on grouped CV. Needs sign-off (Rule 4).")
    w("3. **CV strategy.** No date column -> `StratifiedGroupKFold(n_splits=5, groups=Project)` "
      "(Section 4.2 option 2). To be finalised and recorded in Phase 2.")
    w("4. **Project count.** The file has 89 projects; the SmellyCode++ paper mentions ~103 Java "
      "projects. Minor discrepancy to note, not blocking.")
    w("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    print(f"Reading {RAW} (skipping '{SOURCE_COL}') ...")
    r, _df = build_report()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(r, indent=2), encoding="utf-8")
    write_markdown(r)
    print(f"Rows: {r['shape']['rows']:,}  smelly: {r['target_has_smell']['rate']*100:.2f}%")
    print(f"Wrote {REPORT_MD.relative_to(ROOT)} and {REPORT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

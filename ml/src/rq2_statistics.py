"""Phase 4 - RQ2 statistics and thresholds.

For each consensus metric (the >=4/5 RQ1 consensus), replicates the paper's RQ2
and adds the fixes it skipped.

Replicate (CLAUDE.md 4.5):
  - per-class descriptive stats (count, mean, std, min, Q1, median, Q3, max) and
    smelly/clean median ratio (undefined if clean median is 0) -> the "3x rule"
  - density + box plots per metric, labelled from ONE function (fixes the paper's
    mislabelled x-axes, PAPER_REVIEW item 2)
  - Mann-Whitney U on a balanced random undersample (paper method) AND full data;
    rank-biserial effect size with bands (0.1 small, 0.3 medium, 0.5 large)
  - Holm correction across metrics
  - thresholds: warning = smelly median, critical = smelly Q3

Fixes (labelled "Ours"):
  - evaluate each threshold AS A CLASSIFIER on grouped-CV test folds (precision,
    recall, F1 of "metric > threshold"), threshold fit on the train fold only
  - add a Youden-J (ROC) threshold per metric

Outputs: ml/artifacts/rq2_stats.json, rq2_tests.json, rq2_thresholds.json + figures.

Run:  python -m ml.src.rq2_statistics
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import precision_recall_fscore_support, roc_curve

from .config import load_config, p
from . import cv as cvmod

CFG = load_config()
SEED = CFG["seed"]
TARGET = CFG["target"]["name"]
ARTIFACTS = p(CFG["paths"]["artifacts"])
FIGDIR = ARTIFACTS / "figures" / "rq2"

C_SMELLY = "#A8322D"   # --crit
C_CLEAN = "#2F6B4F"    # --ok


# ----------------------------- helpers -----------------------------
def consensus_metrics() -> list[str]:
    rq1 = json.loads((ARTIFACTS / "rq1_importance.json").read_text(encoding="utf-8"))
    metrics = rq1["consensus"]["at_least_4of5"]
    # keep config order for stable output
    order = CFG["columns"]["numeric_metrics"]
    return [m for m in order if m in metrics]


def describe(series: pd.Series) -> dict:
    q1, med, q3 = series.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(series.max()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, pv) in enumerate(items):
        running = max(running, (m - i) * pv)
        adj[k] = min(1.0, running)
    return adj


def effect_band(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


def mwu(smelly: np.ndarray, clean: np.ndarray) -> dict:
    res = mannwhitneyu(smelly, clean, alternative="two-sided")
    n_s, n_c = len(smelly), len(clean)
    cles = res.statistic / (n_s * n_c)      # P(random smelly > random clean)
    r_rb = 2 * cles - 1                      # rank-biserial correlation
    return {
        "U": float(res.statistic),
        "p": float(res.pvalue),
        "n_smelly": int(n_s),
        "n_clean": int(n_c),
        "cles": float(cles),
        "rank_biserial": float(r_rb),
        "effect_magnitude": effect_band(r_rb),
    }


# ----------------------------- figures -----------------------------
def plot_metric(metric: str, clean: pd.Series, smelly: pd.Series) -> None:
    """Density + box for one metric. Axis labels come from `metric` once
    (fixing the paper's mislabelled figures). Log1p x/y for the long tails."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (axd, axb) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]})

    lc = np.log1p(clean.to_numpy())
    ls = np.log1p(smelly.to_numpy())
    bins = np.linspace(0, max(lc.max(), ls.max()), 60)
    axd.hist(lc, bins=bins, density=True, alpha=0.55, color=C_CLEAN, label="clean")
    axd.hist(ls, bins=bins, density=True, alpha=0.55, color=C_SMELLY, label="smelly")
    axd.set_xlabel(f"log(1 + {metric})")
    axd.set_ylabel("density")
    axd.set_title(f"{metric}: distribution by class")
    axd.legend()

    axb.boxplot([clean.to_numpy(), smelly.to_numpy()], tick_labels=["clean", "smelly"],
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="#E8ECEA"), medianprops=dict(color="#18202A"))
    axb.set_yscale("symlog")
    axb.set_ylabel(metric)
    axb.set_title("median / IQR")

    fig.suptitle(f"RQ2 - {metric} (clean vs smelly)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGDIR / f"{metric.replace(' ', '_')}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- threshold-as-classifier (fix) -----------------------------
def threshold_scores_cv(df: pd.DataFrame, metric: str) -> dict:
    """For each grouped-CV fold, fit thresholds on train, score 'metric > t' on test.
    Thresholds: warning (train smelly median), critical (train smelly Q3), youden (train ROC)."""
    kinds = ["warning", "critical", "youden"]
    acc = {k: {"precision": [], "recall": [], "f1": [], "threshold": []} for k in kinds}
    for fold, tr, te in cvmod.split_indices(df):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        tr_smelly = tr_df[tr_df[TARGET] == 1][metric]
        t_warn = float(tr_smelly.median())
        t_crit = float(tr_smelly.quantile(0.75))
        # Youden J on train using the metric as the score
        fpr, tpr, thr = roc_curve(tr_df[TARGET].to_numpy(), tr_df[metric].to_numpy())
        j = tpr - fpr
        t_youden = float(thr[int(np.argmax(j))])
        for k, t in [("warning", t_warn), ("critical", t_crit), ("youden", t_youden)]:
            pred = (te_df[metric].to_numpy() > t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(
                te_df[TARGET].to_numpy(), pred, average="binary", zero_division=0)
            acc[k]["precision"].append(float(pr))
            acc[k]["recall"].append(float(rc))
            acc[k]["f1"].append(float(f1))
            acc[k]["threshold"].append(float(t))
    out = {}
    for k in kinds:
        out[k] = {
            "threshold_mean": float(np.mean(acc[k]["threshold"])),
            "threshold_std": float(np.std(acc[k]["threshold"])),
            "precision_mean": float(np.mean(acc[k]["precision"])),
            "recall_mean": float(np.mean(acc[k]["recall"])),
            "f1_mean": float(np.mean(acc[k]["f1"])),
            "f1_std": float(np.std(acc[k]["f1"])),
        }
    return out


# ----------------------------- main -----------------------------
# loose mapping to the paper's Table 14 thresholds (different metric + target; context only)
PAPER_EQUIV = {
    "Cyclomatic Complexity": "SumCyclomatic",
    "Logical Lines": "CountStmtExe",
}


def main() -> None:
    df = cvmod.load_clean()
    metrics = consensus_metrics()
    clean_df = df[df[TARGET] == 0]
    smelly_df = df[df[TARGET] == 1]
    rng = np.random.RandomState(SEED)

    paper_ref = json.loads(p("ml/config/paper_reference.json").read_text(encoding="utf-8"))
    paper_thr = paper_ref["rq2_thresholds_table14"]

    stats, tests, thresholds = {}, {}, {}
    p_full, p_bal = {}, {}

    # balanced undersample of clean to match smelly count (paper method), once, reused per metric
    n_smelly = len(smelly_df)
    clean_bal = clean_df.sample(n=n_smelly, random_state=SEED)

    for m in metrics:
        c, s = clean_df[m], smelly_df[m]
        cs, ss = describe(c), describe(s)
        ratio = (ss["median"] / cs["median"]) if cs["median"] != 0 else None
        stats[m] = {
            "clean": cs,
            "smelly": ss,
            "median_ratio_smelly_over_clean": (round(ratio, 3) if ratio is not None else "undefined"),
            "approx_3x_rule": (ratio is not None and 2.0 <= ratio <= 4.0),
        }

        full = mwu(s.to_numpy(), c.to_numpy())
        bal = mwu(s.to_numpy(), clean_bal[m].to_numpy())
        tests[m] = {"full": full, "balanced_undersample": bal}
        p_full[m] = full["p"]
        p_bal[m] = bal["p"]

        thresholds[m] = {
            "warning": float(ss["median"]),        # paper rule: smelly median
            "critical": float(ss["q3"]),           # paper rule: smelly Q3
            "clean_median": float(cs["median"]),   # for the ruler marker
            "ruler_quantiles": {                   # for the UI threshold ruler
                "min": cs["min"], "clean_median": cs["median"],
                "warning": ss["median"], "critical": ss["q3"],
                "p95_smelly": ss["p95"], "max": max(cs["max"], ss["max"]),
            },
            "paper_equivalent": (
                {"metric": PAPER_EQUIV[m], **paper_thr[PAPER_EQUIV[m]],
                 "note": "paper's Table 14 value for a DIFFERENT metric and target (security bugs); "
                         "context only, not a like-for-like comparison"}
                if m in PAPER_EQUIV else None
            ),
            "as_classifier_cv": threshold_scores_cv(df, m),  # the fix the paper omits
        }

        plot_metric(m, c, s)

    # Holm correction across metrics (separately for full and balanced)
    holm_full = holm(p_full)
    holm_bal = holm(p_bal)
    for m in metrics:
        tests[m]["full"]["p_holm"] = holm_full[m]
        tests[m]["balanced_undersample"]["p_holm"] = holm_bal[m]

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "consensus_metrics": metrics,
        "target": TARGET,
        "n_clean": int(len(clean_df)),
        "n_smelly": int(len(smelly_df)),
        "balanced_undersample_n_per_class": int(n_smelly),
        "notes": {
            "effect_bands": "rank-biserial |r|: <0.1 negligible, 0.1 small, 0.3 medium, 0.5 large",
            "thresholds": "warning = smelly median, critical = smelly Q3 (paper rule). "
                          "as_classifier_cv evaluates 'metric > threshold' on grouped-CV test folds "
                          "with thresholds fit on the train fold (the paper never validates thresholds).",
            "mann_whitney": "two-sided; full data and a balanced random undersample (seed 42).",
        },
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "rq2_stats.json").write_text(json.dumps({**meta, "stats": stats}, indent=2), encoding="utf-8")
    (ARTIFACTS / "rq2_tests.json").write_text(json.dumps({**meta, "tests": tests}, indent=2), encoding="utf-8")
    (ARTIFACTS / "rq2_thresholds.json").write_text(json.dumps({**meta, "thresholds": thresholds}, indent=2), encoding="utf-8")

    # console summary
    print(f"RQ2 over {len(metrics)} consensus metrics: {metrics}\n")
    print(f"{'metric':24s} {'clean~':>8} {'smelly~':>8} {'ratio':>6} {'r_rb':>6} {'band':>10} "
          f"{'crit':>8} {'critF1':>7}")
    for m in metrics:
        st, te, th = stats[m], tests[m]["full"], thresholds[m]
        ratio = st["median_ratio_smelly_over_clean"]
        ratio_s = f"{ratio}" if ratio == "undefined" else f"{ratio:.2f}"
        print(f"{m:24s} {st['clean']['median']:>8.3g} {st['smelly']['median']:>8.3g} {ratio_s:>6} "
              f"{te['rank_biserial']:>6.3f} {te['effect_magnitude']:>10} "
              f"{th['critical']:>8.3g} {th['as_classifier_cv']['critical']['f1_mean']:>7.3f}")
    print(f"\nWrote rq2_stats.json, rq2_tests.json, rq2_thresholds.json and figures/rq2/ to {ARTIFACTS}")


if __name__ == "__main__":
    main()

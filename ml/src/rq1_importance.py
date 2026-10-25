"""Phase 3 - RQ1 feature importance.

Replicates the paper's five importance methods and builds consensus lists, then
adds the fix track (permutation importance on CV test folds, SHAP, correlated-pair
flags). RQ1 importance is descriptive and runs on the full data (CLAUDE.md 4.3);
Boruta and SHAP use a stratified subsample for speed, which is noted in the output.

Replicate (CLAUDE.md 4.4):
  1. Linear Regression |coef|   on standardized features
  2. Logistic Regression |coef| on standardized features
  3. XGBoost gain importance
  4. ANOVA F-score (f_classif)
  5. Boruta (BorutaPy over RandomForest)
  + Pearson metric-metric matrix and metric-target vector
  + consensus: strict intersection of the 5 top-10s, and "appears in >= 4 of 5"

Fixes (labelled "Ours, extra"):
  + permutation importance on grouped-CV test folds (mean +/- std)
  + SHAP summary (XGBoost stand-in; regenerated for the saved model in Phase 5)
  + flag correlated pairs (|r| > 0.9) - coef rankings are unstable for them

Outputs: ml/artifacts/rq1_importance.json, rq1_correlation.json, figures/.

Run:  python -m ml.src.rq1_importance
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import load_config, p
from . import cv as cvmod

CFG = load_config()
SEED = CFG["seed"]
FEATURES = CFG["columns"]["numeric_metrics"]
TARGET = CFG["target"]["name"]

ARTIFACTS = p(CFG["paths"]["artifacts"])
FIGDIR = ARTIFACTS / "figures"
TOPK = 10
BORUTA_SUBSAMPLE = 25000
SHAP_SUBSAMPLE = 5000


# ----------------------------- helpers -----------------------------
def _ranked(scores: dict[str, float], higher_is_better: bool = True) -> dict:
    """Turn {feature: score} into a ranked list and a top-K name list."""
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    ranked = [{"feature": f, "score": float(s), "rank": i + 1} for i, (f, s) in enumerate(items)]
    top = [r["feature"] for r in ranked[:TOPK]]
    return {"ranked": ranked, "top10": top}


def _subsample(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    if len(df) <= n:
        return df
    # stratified by target
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0]
    n_pos = int(round(n * len(pos) / len(df)))
    n_neg = n - n_pos
    s = pd.concat([
        pos.sample(min(n_pos, len(pos)), random_state=rng.randint(1 << 30)),
        neg.sample(min(n_neg, len(neg)), random_state=rng.randint(1 << 30)),
    ])
    return s.sample(frac=1.0, random_state=rng.randint(1 << 30)).reset_index(drop=True)


# ----------------------------- methods -----------------------------
def linear_coef(Xz: pd.DataFrame, y: np.ndarray) -> dict:
    m = LinearRegression().fit(Xz, y)
    return _ranked({f: abs(c) for f, c in zip(FEATURES, m.coef_)})


def logistic_coef(Xz: pd.DataFrame, y: np.ndarray) -> dict:
    m = LogisticRegression(max_iter=1000, solver="lbfgs").fit(Xz, y)
    return _ranked({f: abs(c) for f, c in zip(FEATURES, m.coef_[0])})


def xgb_gain(X: pd.DataFrame, y: np.ndarray) -> dict:
    m = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                      tree_method="hist", verbosity=0)
    m.fit(X, y)
    gain = m.get_booster().get_score(importance_type="gain")  # keyed by column name
    scores = {f: float(gain.get(f, 0.0)) for f in FEATURES}
    return _ranked(scores)


def anova_f(X: pd.DataFrame, y: np.ndarray) -> dict:
    F, _pv = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0)
    return _ranked({f: float(v) for f, v in zip(FEATURES, F)})


def boruta_rank(X: pd.DataFrame, y: np.ndarray) -> dict:
    # BorutaPy needs numpy; it fixes a RandomForest internally.
    from boruta import BorutaPy
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=SEED, class_weight="balanced")
    bor = BorutaPy(rf, n_estimators="auto", max_iter=60, random_state=SEED, verbose=0)
    bor.fit(X.to_numpy(), y)
    rank = bor.ranking_  # 1 = confirmed best
    max_r = int(rank.max())
    # Boruta often confirms every feature when all carry signal and are collinear, leaving its
    # ranking fully tied. Break ties with a RandomForest importance so the ordering (and the
    # consensus) is meaningful rather than column-order.
    rf_tie = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1,
                                    random_state=SEED, class_weight="balanced")
    rf_tie.fit(X, y)
    imp = {f: float(v) for f, v in zip(FEATURES, rf_tie.feature_importances_)}
    # score = confirmed-tier (dominant) + normalised RF importance (tie-break)
    scores = {f: (max_r - int(r) + 1) + imp[f] for f, r in zip(FEATURES, rank)}
    out = _ranked(scores)
    out["boruta_support"] = {f: bool(s) for f, s in zip(FEATURES, bor.support_)}
    out["boruta_ranking"] = {f: int(r) for f, r in zip(FEATURES, rank)}
    out["rf_importance_tiebreak"] = imp
    out["note"] = ("Boruta confirmed/rejected shown in boruta_support; ties (Boruta confirms all "
                   "relevant features here) are broken by a RandomForest importance so the top-10 is "
                   "discriminative.")
    return out


# ----------------------------- fixes -----------------------------
def permutation_cv(df: pd.DataFrame) -> dict:
    """Permutation importance on each grouped-CV test fold, averaged (ROC-AUC drop)."""
    per_fold = {f: [] for f in FEATURES}
    for fold, tr, te in cvmod.split_indices(df):
        Xtr, ytr = df.iloc[tr][FEATURES], df.iloc[tr][TARGET].to_numpy()
        Xte, yte = df.iloc[te][FEATURES], df.iloc[te][TARGET].to_numpy()
        model = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                              tree_method="hist", verbosity=0)
        model.fit(Xtr, ytr)
        r = permutation_importance(model, Xte, yte, scoring="roc_auc",
                                   n_repeats=5, random_state=SEED, n_jobs=-1)
        for f, imp in zip(FEATURES, r.importances_mean):
            per_fold[f].append(float(imp))
    agg = {f: {"mean": float(np.mean(v)), "std": float(np.std(v))} for f, v in per_fold.items()}
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    return {
        "note": "Ours, extra. Mean ROC-AUC drop when a feature is permuted, over the 5 grouped-CV "
                "test folds (XGBoost fit per fold). Computed on held-out data, unlike the replication "
                "methods which are descriptive on full data.",
        "ranked": [{"feature": f, "mean": d["mean"], "std": d["std"]} for f, d in ranked],
        "top10": [f for f, _ in ranked[:TOPK]],
    }


def shap_summary(df: pd.DataFrame, rng: np.random.RandomState) -> dict:
    import shap
    sample = _subsample(df, SHAP_SUBSAMPLE, rng)
    X = sample[FEATURES]
    y = sample[TARGET].to_numpy()
    model = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                          tree_method="hist", verbosity=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    scores = {f: float(v) for f, v in zip(FEATURES, mean_abs)}
    out = _ranked(scores)
    # figure
    try:
        shap.summary_plot(sv, X, show=False, plot_type="bar", max_display=len(FEATURES))
        plt.tight_layout()
        FIGDIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGDIR / "shap_summary.png", dpi=130, bbox_inches="tight")
        plt.close("all")
    except Exception as e:  # plotting must never break the artifact
        print("SHAP plot skipped:", e)
    out["note"] = (f"Ours, extra. mean(|SHAP|) from an XGBoost stand-in fit on a stratified "
                   f"{len(sample):,}-row sample. The SHAP summary for the actual saved model is "
                   f"regenerated in Phase 5.")
    out["n_sample"] = int(len(sample))
    return out


# ----------------------------- figures -----------------------------
def fig_rank_matrix(methods: dict) -> None:
    method_keys = ["linear_regression_coef", "logistic_regression_coef",
                   "xgboost_gain", "anova_f", "boruta"]
    labels = ["LinReg |coef|", "LogReg |coef|", "XGB gain", "ANOVA F", "Boruta"]
    # rank (1..) per feature per method, blank if outside top-10
    M = np.full((len(FEATURES), len(method_keys)), np.nan)
    for j, mk in enumerate(method_keys):
        rank_of = {r["feature"]: r["rank"] for r in methods[mk]["ranked"]}
        for i, f in enumerate(FEATURES):
            rk = rank_of.get(f)
            if rk is not None and rk <= TOPK:
                M[i, j] = rk
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto", vmin=1, vmax=TOPK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(method_keys)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > 5 else "black", fontsize=8)
    ax.set_title("RQ1 rank matrix (top-10 per method; 1 = most important)")
    fig.colorbar(im, ax=ax, label="rank")
    plt.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGDIR / "rank_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_corr(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=7)
    ax.set_title("Metric-metric Pearson correlation")
    fig.colorbar(im, ax=ax, label="r")
    plt.tight_layout()
    plt.savefig(FIGDIR / "correlation_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_permutation(perm: dict) -> None:
    feats = [d["feature"] for d in perm["ranked"]][::-1]
    means = [d["mean"] for d in perm["ranked"]][::-1]
    stds = [d["std"] for d in perm["ranked"]][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(feats, means, xerr=stds, color="#1D5C7A")
    ax.set_xlabel("mean ROC-AUC drop when permuted")
    ax.set_title("Permutation importance (grouped-CV test folds) - Ours, extra")
    plt.tight_layout()
    plt.savefig(FIGDIR / "permutation_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- main -----------------------------
def main() -> None:
    rng = np.random.RandomState(SEED)
    df = cvmod.load_clean()
    X = df[FEATURES]
    y = df[TARGET].to_numpy()
    Xz = pd.DataFrame(StandardScaler().fit_transform(X), columns=FEATURES)

    print("Running 5 replication importance methods ...")
    methods = {
        "linear_regression_coef": linear_coef(Xz, y),
        "logistic_regression_coef": logistic_coef(Xz, y),
        "xgboost_gain": xgb_gain(X, y),
        "anova_f": anova_f(X, y),
    }
    print("  Boruta (subsampled) ...")
    bor_sample = _subsample(df, BORUTA_SUBSAMPLE, rng)
    methods["boruta"] = boruta_rank(bor_sample[FEATURES], bor_sample[TARGET].to_numpy())
    methods["boruta"]["n_sample"] = int(len(bor_sample))

    # --- Pearson ---
    corr = X.corr(method="pearson")
    pearson_target = {f: float(np.corrcoef(X[f], y)[0, 1]) for f in FEATURES}
    high_pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.9:
                high_pairs.append({"a": FEATURES[i], "b": FEATURES[j], "r": round(float(r), 4)})
    high_pairs.sort(key=lambda d: -abs(d["r"]))

    # --- consensus over the 5 methods ---
    method_keys = ["linear_regression_coef", "logistic_regression_coef",
                   "xgboost_gain", "anova_f", "boruta"]
    counts = {f: 0 for f in FEATURES}
    for mk in method_keys:
        for f in methods[mk]["top10"]:
            counts[f] += 1
    strict = sorted([f for f, c in counts.items() if c == 5], key=lambda f: -counts[f])
    soft = sorted([f for f, c in counts.items() if c >= 4], key=lambda f: -counts[f])

    print("Permutation importance on CV folds ...")
    perm = permutation_cv(df)
    print("SHAP summary (subsampled) ...")
    shap_out = shap_summary(df, rng)

    # --- figures ---
    print("Rendering figures ...")
    fig_rank_matrix(methods)
    fig_corr(corr)
    fig_permutation(perm)

    # --- paper comparison (families, not exact names) ---
    paper_ref = json.loads(p("ml/config/paper_reference.json").read_text(encoding="utf-8"))
    paper_core = paper_ref["rq1_core_metrics"]["metrics"]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "topk": TOPK,
        "methods": methods,
        "pearson_target": pearson_target,
        "consensus": {
            "appears_in_top10_counts": counts,
            "strict_intersection_5of5": strict,
            "at_least_4of5": soft,
        },
        "fixes": {
            "permutation_importance": perm,
            "shap": shap_out,
            "correlated_pairs_gt_0_9": high_pairs,
            "coef_caveat": "Linear/Logistic coef rankings are unstable here: nearly all metrics are "
                           "|r|>0.9 collinear (several exactly 1.0), so coefficient mass is split "
                           "arbitrarily among near-duplicates. Prefer the tree/permutation/SHAP "
                           "rankings for interpretation.",
        },
        "paper_comparison": {
            "paper_core_metrics": paper_core,
            "note": "The paper's core metrics are SciTools Understand metrics (size + cyclomatic "
                    "families) that do not exist in SmellyCode++, so a name-level intersection is "
                    "empty by construction. We compare at the family level: whether size/cyclomatic "
                    "metrics dominate our consensus, mirroring the paper's size+cyclomatic core.",
        },
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "rq1_importance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ARTIFACTS / "rq1_correlation.json").write_text(json.dumps({
        "features": FEATURES,
        "matrix": corr.round(4).values.tolist(),
        "pearson_target": pearson_target,
        "pairs_gt_0_9": high_pairs,
    }, indent=2), encoding="utf-8")

    # --- console summary ---
    print("\n=== RQ1 top-10 per method ===")
    for mk in method_keys:
        print(f"{mk:26s}: {', '.join(methods[mk]['top10'][:6])} ...")
    print(f"\nConsensus strict (5/5): {strict}")
    print(f"Consensus >=4/5:        {soft}")
    print(f"Permutation top: {perm['top10'][:6]}")
    print(f"SHAP top:        {shap_out['top10'][:6]}")
    print(f"\nWrote rq1_importance.json, rq1_correlation.json, and figures/ to {ARTIFACTS}")


if __name__ == "__main__":
    main()

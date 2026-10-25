"""Phase 5 - RQ3 classifiers (replication + improvement).

Replication track: the 11 models with Table 4 hyperparameters, inside a
leakage-safe imbalanced-learn Pipeline ([scaler if scale-sensitive] -> SMOTE ->
clf), evaluated with the Section 4.2 grouped CV. Per-fold and mean +/- std:
accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC. Friedman + Nemenyi on
ROC-AUC. Paper value shown side by side.

Improvement track ("Ours, tuned"): nested RandomizedSearchCV for XGBoost / Random
Forest / AdaBoost, a SMOTE vs scale_pos_weight comparison for XGBoost, decision-
threshold tuning for a high-recall operating point, and a size-metric ablation
(circularity proxy, since the exact SmellyCode++ labeling rule is unconfirmed).

Outputs: rq3_results.json, rq3_roc.json, rq3_confusion.json, rq3_nemenyi.json,
rq3_improvements.json.

Scalability note: SVMs and MLP cap the per-fold training rows (libsvm is ~O(n^2));
caps are recorded in the output. Set SMOKE=1 for a fast subsampled dry run.

Run:  python -m ml.src.rq3_models
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import friedmanchisquare
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import load_config, p
from . import cv as cvmod

CFG = load_config()
SEED = CFG["seed"]
FEATURES = CFG["columns"]["numeric_metrics"]
TARGET = CFG["target"]["name"]
ARTIFACTS = p(CFG["paths"]["artifacts"])
SMOKE = os.environ.get("SMOKE") == "1"

warnings.filterwarnings("ignore")  # QDA collinearity + LR non-convergence are expected (paper settings)

# model name -> (factory, scale_sensitive, train_cap). Caps bound libsvm/MLP cost.
TRAIN_CAP_SVM = 3000 if SMOKE else 12000
TRAIN_CAP_MLP = 5000 if SMOKE else 40000
TRAIN_CAP_KNN = 5000 if SMOKE else 60000


def _models() -> dict:
    return {
        "KNN": (lambda: KNeighborsClassifier(n_neighbors=3), True, TRAIN_CAP_KNN),
        "Decision Tree": (lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED), False, None),
        "Random Forest": (lambda: RandomForestClassifier(max_depth=3, n_estimators=10, max_features=1,
                                                         random_state=SEED, n_jobs=-1), False, None),
        "MLP": (lambda: MLPClassifier(activation="relu", solver="adam", alpha=1, batch_size="auto",
                                      random_state=SEED, max_iter=100), True, TRAIN_CAP_MLP),
        "AdaBoost": (lambda: AdaBoostClassifier(n_estimators=50, random_state=SEED), False, None),
        "XGBoost": (lambda: XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                                          tree_method="hist", verbosity=0), False, None),
        "Naive Bayes": (lambda: GaussianNB(priors=None, var_smoothing=1e-9), False, None),
        "QDA": (lambda: QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                                                      store_covariance=False, tol=1e-4), True, None),
        "Linear SVM": (lambda: SVC(kernel="linear", random_state=SEED), True, TRAIN_CAP_SVM),
        "RBF SVM": (lambda: SVC(kernel="rbf", random_state=SEED), True, TRAIN_CAP_SVM),
        "Logistic Regression": (lambda: LogisticRegression(random_state=0, solver="lbfgs",
                                                           max_iter=100), True, None),
    }


def _pipeline(factory, scale: bool) -> ImbPipeline:
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", factory()))
    return ImbPipeline(steps)


def _fit_pipe(name, factory, scale, Xtr, ytr):
    """Fit a pipeline; QDA with reg_param=0 (Table 4) is singular on the collinear
    Halstead metrics, so escalate reg_param as a forced, flagged deviation."""
    try:
        pipe = _pipeline(factory, scale)
        pipe.fit(Xtr, ytr)
        return pipe, None
    except np.linalg.LinAlgError:
        if name == "QDA":
            for rp in (1e-3, 1e-2, 1e-1):
                try:
                    pipe = _pipeline(
                        lambda rp=rp: QuadraticDiscriminantAnalysis(reg_param=rp, tol=1e-4), scale)
                    pipe.fit(Xtr, ytr)
                    return pipe, rp
                except np.linalg.LinAlgError:
                    continue
        raise


def _scores(pipe, X) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except Exception:
            pass
    return pipe.decision_function(X)


def _metrics(y, pred, score) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


def _cap(idx: np.ndarray, y: np.ndarray, cap, seed: int) -> np.ndarray:
    if cap is None or len(idx) <= cap:
        return idx
    keep, _ = train_test_split(idx, train_size=cap, stratify=y[idx], random_state=seed)
    return keep


# ----------------------------- replication track -----------------------------
def run_replication(df: pd.DataFrame) -> dict:
    y_all = df[TARGET].to_numpy()
    folds = list(cvmod.split_indices(df))
    models = _models()

    per_model = {}
    oof_scores = {m: np.full(len(df), np.nan) for m in models}
    oof_pred = {m: np.full(len(df), -1) for m in models}

    for name, (factory, scale, cap) in models.items():
        fold_metrics = []
        qda_reg = set()
        for fold, tr, te in folds:
            tr_c = _cap(tr, y_all, cap, SEED + fold)
            Xtr, ytr = df.iloc[tr_c][FEATURES], y_all[tr_c]
            Xte, yte = df.iloc[te][FEATURES], y_all[te]
            pipe, note = _fit_pipe(name, factory, scale, Xtr, ytr)
            if note is not None:
                qda_reg.add(note)
            pred = pipe.predict(Xte)
            sc = _scores(pipe, Xte)
            fold_metrics.append(_metrics(yte, pred, sc))
            oof_scores[name][te] = sc
            oof_pred[name][te] = pred
        keys = fold_metrics[0].keys()
        per_model[name] = {
            "per_fold": fold_metrics,
            "mean": {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys},
            "std": {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys},
            "train_cap": cap,
            "scale_sensitive": scale,
        }
        if qda_reg:
            per_model[name]["deviation"] = (
                f"Table 4 reg_param=0.0 is singular on the collinear Halstead metrics; "
                f"used reg_param in {sorted(qda_reg)} instead (forced deviation, PAPER_REVIEW item 4).")
        mm = per_model[name]["mean"]
        print(f"  {name:20s} AUC {mm['roc_auc']:.3f}  PR {mm['pr_auc']:.3f}  "
              f"F1 {mm['f1']:.3f}  MCC {mm['mcc']:.3f}  Rec {mm['recall']:.3f}")
    return per_model, oof_scores, oof_pred, [f for f, _, _ in folds]


def friedman_nemenyi(per_model: dict, metric: str = "roc_auc") -> dict:
    import scikit_posthocs as sp
    names = list(per_model.keys())
    n_folds = len(per_model[names[0]]["per_fold"])
    mat = np.array([[per_model[m]["per_fold"][f][metric] for m in names] for f in range(n_folds)])
    stat, pval = friedmanchisquare(*[mat[:, j] for j in range(len(names))])
    # mean ranks (higher metric = better -> rank 1). rank per fold descending.
    ranks = np.zeros((n_folds, len(names)))
    for f in range(n_folds):
        order = pd.Series(mat[f], index=names).rank(ascending=False, method="average")
        ranks[f] = order.values
    mean_ranks = {names[j]: float(ranks[:, j].mean()) for j in range(len(names))}
    nemenyi = sp.posthoc_nemenyi_friedman(mat)
    nemenyi.index = names
    nemenyi.columns = names
    return {
        "metric": metric,
        "friedman_stat": float(stat),
        "friedman_p": float(pval),
        "mean_ranks": dict(sorted(mean_ranks.items(), key=lambda kv: kv[1])),
        "nemenyi_p_matrix": {a: {b: float(nemenyi.loc[a, b]) for b in names} for a in names},
    }


def _downsample_curve(x, y, n=120):
    if len(x) <= n:
        return x.tolist(), y.tolist()
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return np.asarray(x)[idx].tolist(), np.asarray(y)[idx].tolist()


# ----------------------------- improvement track -----------------------------
def tuned_track(df: pd.DataFrame) -> dict:
    y_all = df[TARGET].to_numpy()
    folds = list(cvmod.split_indices(df))
    tune_cap = 6000 if SMOKE else 30000
    n_iter = 2 if SMOKE else 6
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    spaces = {
        "XGBoost": (lambda: XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                                          tree_method="hist", verbosity=0), {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 4, 6, 8],
            "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
        }),
        "Random Forest": (lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1), {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 8, 16],
            "clf__max_features": ["sqrt", 0.5],
            "clf__min_samples_leaf": [1, 5, 20],
        }),
        "AdaBoost": (lambda: AdaBoostClassifier(random_state=SEED), {
            "clf__n_estimators": [100, 200, 400],
            "clf__learning_rate": [0.1, 0.5, 1.0],
        }),
    }

    out = {}
    for name, (factory, space) in spaces.items():
        fold_metrics, best_params = [], []
        for fold, tr, te in folds:
            tr_c = _cap(tr, y_all, tune_cap, SEED + fold)
            Xtr, ytr = df.iloc[tr_c][FEATURES], y_all[tr_c]
            Xte, yte = df.iloc[te][FEATURES], y_all[te]
            pipe = ImbPipeline([("smote", SMOTE(random_state=SEED)), ("clf", factory())])
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision",
                                        cv=inner, random_state=SEED, n_jobs=-1, refit=True)
            search.fit(Xtr, ytr)
            best = search.best_estimator_
            pred = best.predict(Xte)
            sc = _scores(best, Xte)
            fold_metrics.append(_metrics(yte, pred, sc))
            best_params.append({k: v for k, v in search.best_params_.items()})
        keys = fold_metrics[0].keys()
        out[name] = {
            "mean": {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys},
            "std": {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys},
            "best_params_per_fold": best_params,
            "tune_train_cap": tune_cap, "n_iter": n_iter, "inner_cv": 3,
        }
        print(f"  tuned {name:16s} AUC {out[name]['mean']['roc_auc']:.3f}  "
              f"PR {out[name]['mean']['pr_auc']:.3f}  F1 {out[name]['mean']['f1']:.3f}")
    return out


def smote_vs_weight(df: pd.DataFrame) -> dict:
    """XGBoost: SMOTE pipeline vs scale_pos_weight (no SMOTE)."""
    y_all = df[TARGET].to_numpy()
    folds = list(cvmod.split_indices(df))
    spw = float((y_all == 0).sum() / (y_all == 1).sum())
    variants = {
        "smote": lambda: ImbPipeline([("smote", SMOTE(random_state=SEED)),
                                      ("clf", XGBClassifier(random_state=SEED, n_jobs=-1,
                                       eval_metric="logloss", tree_method="hist", verbosity=0))]),
        "scale_pos_weight": lambda: ImbPipeline([("clf", XGBClassifier(
            random_state=SEED, n_jobs=-1, eval_metric="logloss", tree_method="hist",
            verbosity=0, scale_pos_weight=spw))]),
    }
    res = {}
    for vname, make in variants.items():
        fm = []
        for fold, tr, te in folds:
            pipe = make()
            pipe.fit(df.iloc[tr][FEATURES], y_all[tr])
            pred = pipe.predict(df.iloc[te][FEATURES])
            sc = _scores(pipe, df.iloc[te][FEATURES])
            fm.append(_metrics(y_all[te], pred, sc))
        keys = fm[0].keys()
        res[vname] = {k: float(np.mean([f[k] for f in fm])) for k in keys}
    res["scale_pos_weight_value"] = spw
    return res


def threshold_tuning(oof_scores: np.ndarray, y: np.ndarray, target_recall=0.80) -> dict:
    """Operating points from pooled out-of-fold XGBoost probabilities."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, oof_scores)
    # precision_recall_curve returns prec/rec of len n+1, thr len n
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best_f1_i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    # highest-precision point that still reaches target recall
    ok = np.where(rec[:-1] >= target_recall)[0]
    hr_i = int(ok[np.argmax(prec[:-1][ok])]) if len(ok) else best_f1_i
    return {
        "default_0.5": _op(y, oof_scores, 0.5),
        "max_f1": {"threshold": float(thr[best_f1_i]), **_op(y, oof_scores, float(thr[best_f1_i]))},
        f"recall>={target_recall}": {"threshold": float(thr[hr_i]), **_op(y, oof_scores, float(thr[hr_i]))},
    }


def _op(y, score, t) -> dict:
    pred = (score >= t).astype(int)
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


def ablation(df: pd.DataFrame) -> dict:
    """Circularity proxy: drop the size-dominant strict-consensus metrics, re-run XGBoost."""
    rq1 = json.loads((ARTIFACTS / "rq1_importance.json").read_text(encoding="utf-8"))
    drop = rq1["consensus"]["strict_intersection_5of5"]  # size-scaled metrics
    kept = [f for f in FEATURES if f not in drop]
    y_all = df[TARGET].to_numpy()
    folds = list(cvmod.split_indices(df))

    def run(feats):
        fm = []
        for fold, tr, te in folds:
            pipe = ImbPipeline([("smote", SMOTE(random_state=SEED)),
                                ("clf", XGBClassifier(random_state=SEED, n_jobs=-1,
                                 eval_metric="logloss", tree_method="hist", verbosity=0))])
            pipe.fit(df.iloc[tr][feats], y_all[tr])
            pred = pipe.predict(df.iloc[te][feats])
            sc = _scores(pipe, df.iloc[te][feats])
            fm.append(_metrics(y_all[te], pred, sc))
        keys = fm[0].keys()
        return {k: float(np.mean([f[k] for f in fm])) for k in keys}

    return {
        "dropped_metrics": drop,
        "kept_metrics": kept,
        "full_features": run(FEATURES),
        "ablated": run(kept),
        "note": "Size-ablation as a circularity proxy: the exact SmellyCode++ labeling rule is "
                "unconfirmed, so we drop the size-dominant strict-consensus metrics and measure the "
                "XGBoost performance drop. A small drop would mean the signal is not purely size.",
    }


# ----------------------------- main -----------------------------
def main() -> None:
    df = cvmod.load_clean()
    paper = json.loads(p("ml/config/paper_reference.json").read_text(encoding="utf-8"))["rq3_table15"]

    print(f"RQ3 replication ({'SMOKE' if SMOKE else 'full'}) - 11 models, grouped CV ...")
    per_model, oof_scores, oof_pred, fold_ids = run_replication(df)

    print("Friedman + Nemenyi on ROC-AUC ...")
    nem = friedman_nemenyi(per_model, "roc_auc")

    # paper side-by-side
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "smoke": SMOKE,
        "cv": {"strategy": CFG["cv"]["strategy"], "n_splits": CFG["cv"]["n_splits"],
               "group": CFG["cv"]["group_column"]},
        "metrics": ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"],
        "models": {},
    }
    for name, r in per_model.items():
        results["models"][name] = {
            "ours_mean": r["mean"], "ours_std": r["std"],
            "paper": paper.get(name),
            "train_cap": r["train_cap"], "scale_sensitive": r["scale_sensitive"],
            "per_fold": r["per_fold"],
        }

    # ROC + confusion from pooled OOF (top-3 by AUC for ROC)
    y = df[TARGET].to_numpy()
    top3 = sorted(per_model, key=lambda m: -per_model[m]["mean"]["roc_auc"])[:3]
    roc = {}
    for name in top3:
        mask = ~np.isnan(oof_scores[name])
        fpr, tpr, _ = roc_curve(y[mask], oof_scores[name][mask])
        fx, fy = _downsample_curve(fpr, tpr)
        roc[name] = {"fpr": fx, "tpr": fy, "auc_oof": float(roc_auc_score(y[mask], oof_scores[name][mask]))}
    confusion = {}
    for name in per_model:
        mask = oof_pred[name] >= 0
        cm = confusion_matrix(y[mask], oof_pred[name][mask])
        confusion[name] = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                           "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}

    print("Improvement track: tuned models ...")
    tuned = tuned_track(df)
    print("Improvement track: SMOTE vs scale_pos_weight, threshold tuning, ablation ...")
    svw = smote_vs_weight(df)
    xgb_oof = oof_scores["XGBoost"]
    thr_tune = threshold_tuning(xgb_oof[~np.isnan(xgb_oof)], y[~np.isnan(xgb_oof)])
    abl = ablation(df)

    improvements = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "seed": SEED, "smoke": SMOKE,
        "tuned_nested_cv": tuned,
        "smote_vs_scale_pos_weight_xgboost": svw,
        "threshold_tuning_xgboost_oof": thr_tune,
        "size_ablation_xgboost": abl,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "rq3_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ARTIFACTS / "rq3_nemenyi.json").write_text(json.dumps(nem, indent=2), encoding="utf-8")
    (ARTIFACTS / "rq3_roc.json").write_text(json.dumps({"top3": top3, "curves": roc}, indent=2), encoding="utf-8")
    (ARTIFACTS / "rq3_confusion.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    (ARTIFACTS / "rq3_improvements.json").write_text(json.dumps(improvements, indent=2), encoding="utf-8")

    best = min(nem["mean_ranks"], key=nem["mean_ranks"].get)
    print(f"\nBest mean-rank (ROC-AUC): {best}")
    print(f"Friedman p = {nem['friedman_p']:.3g}")
    print(f"Ablation XGB AUC: full {abl['full_features']['roc_auc']:.3f} -> "
          f"ablated {abl['ablated']['roc_auc']:.3f}")
    print(f"Wrote rq3_results/nemenyi/roc/confusion/improvements.json to {ARTIFACTS}")


if __name__ == "__main__":
    main()

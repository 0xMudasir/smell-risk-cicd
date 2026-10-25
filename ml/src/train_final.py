"""Phase 5 - Train and save the deployable model.

Picks the best model from rq3_results.json (by ROC-AUC, recall tie-breaker),
refits it on ALL data with the same leakage-safe pipeline, and saves to
backend/model/: model.joblib, feature_schema.json, thresholds.json,
model_card.json. Smoke-tests that it loads and predicts one row in < 1 s.

Run:  python -m ml.src.train_final
"""
from __future__ import annotations

import importlib.metadata as md
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .config import ROOT, load_config, p
from . import cv as cvmod
from .rq3_models import _models, _pipeline

CFG = load_config()
SEED = CFG["seed"]
FEATURES = CFG["columns"]["numeric_metrics"]
TARGET = CFG["target"]["name"]
ARTIFACTS = p(CFG["paths"]["artifacts"])
MODEL_DIR = ROOT / "backend" / "model"
VERSION = "1.0.0"

FEATURE_DESC = {
    "Logical Lines": "Logical lines of code in the entity (size).",
    "Distinct Operators": "Halstead n1: number of distinct operators.",
    "Distinct Operands": "Halstead n2: number of distinct operands.",
    "Total Operators": "Halstead N1: total operator occurrences.",
    "Total Operands": "Halstead N2: total operand occurrences.",
    "Vocabulary": "Halstead vocabulary n = n1 + n2.",
    "Length": "Halstead length N = N1 + N2.",
    "Calculated Length": "Halstead estimated length n1*log2 n1 + n2*log2 n2.",
    "Volume": "Halstead volume = Length * log2(Vocabulary).",
    "Difficulty": "Halstead difficulty = (n1/2) * (N2/n2).",
    "Effort": "Halstead effort = Difficulty * Volume.",
    "Time Required": "Halstead time = Effort / 18 (seconds).",
    "Bugs": "Halstead delivered-bugs estimate = Volume / 3000.",
    "Cyclomatic Complexity": "McCabe cyclomatic complexity of the entity.",
}

LIB_PINS = ["numpy", "scikit-learn", "xgboost", "imbalanced-learn", "pandas", "scipy", "joblib"]


def pick_best() -> tuple[str, dict]:
    res = json.loads((ARTIFACTS / "rq3_results.json").read_text(encoding="utf-8"))["models"]
    ranked = sorted(res.items(),
                    key=lambda kv: (kv[1]["ours_mean"]["roc_auc"], kv[1]["ours_mean"]["recall"]),
                    reverse=True)
    name, info = ranked[0]
    return name, info


def build_schema(df: pd.DataFrame) -> dict:
    schema = {"features": []}
    for f in FEATURES:
        s = df[f]
        schema["features"].append({
            "name": f,
            "dtype": "number",
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
            "description": FEATURE_DESC.get(f, ""),
        })
    schema["order"] = FEATURES
    return schema


def build_thresholds() -> dict:
    rq2 = json.loads((ARTIFACTS / "rq2_thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    per_metric = {m: {"warning": d["warning"], "critical": d["critical"],
                      "clean_median": d["clean_median"], "ruler_quantiles": d["ruler_quantiles"]}
                  for m, d in rq2.items()}
    return {
        "risk_tiers": {
            "low": {"max_probability": 0.4, "rule": "probability < 0.4 and no critical flag"},
            "critical": {"min_probability": 0.7, "min_critical_flags": 2,
                         "rule": "probability >= 0.7 OR two or more critical flags"},
            "elevated": {"rule": "anything between low and critical"},
        },
        "metric_thresholds": per_metric,
    }


def build_card(name: str, info: dict, df: pd.DataFrame) -> dict:
    return {
        "model": {"name": name, "version": VERSION},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": {"name": TARGET, "definition": "has_smell = 1 if any of "
                   f"{CFG['target']['smell_label_columns']} is positive"},
        "dataset": {"name": CFG["dataset"]["name"], "rows": int(len(df)),
                    "smelly_rate": float(df[TARGET].mean()),
                    "n_projects": int(df[CFG["columns"]["project_column"]].nunique())},
        "cv": {"strategy": CFG["cv"]["splitter"], "n_splits": CFG["cv"]["n_splits"],
               "group": CFG["cv"]["group_column"]},
        "cv_metrics_mean": info["ours_mean"],
        "cv_metrics_std": info["ours_std"],
        "paper_reference_same_model": info.get("paper"),
        "features": FEATURES,
        "n_features": len(FEATURES),
        "library_versions": {lib: md.version(lib) for lib in LIB_PINS},
        "notes": "Replicates Diouf et al. (2026) method on SmellyCode++ (code smells, not security "
                 "bugs). Risk tiers and metric thresholds live in thresholds.json. SMOTE is applied "
                 "only during fit; prediction applies scaler (if any) + classifier.",
    }


def main() -> None:
    df = cvmod.load_clean()
    name, info = pick_best()
    print(f"Best model: {name}  (CV ROC-AUC {info['ours_mean']['roc_auc']:.3f}, "
          f"recall {info['ours_mean']['recall']:.3f})")

    factory, scale, _cap = _models()[name]
    pipe = _pipeline(factory, scale)
    print("Refitting on all data ...")
    pipe.fit(df[FEATURES], df[TARGET].to_numpy())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps(build_schema(df), indent=2), encoding="utf-8")
    (MODEL_DIR / "thresholds.json").write_text(json.dumps(build_thresholds(), indent=2), encoding="utf-8")
    (MODEL_DIR / "model_card.json").write_text(json.dumps(build_card(name, info, df), indent=2), encoding="utf-8")

    # --- smoke test: reload + predict one median row in < 1s ---
    loaded = joblib.load(MODEL_DIR / "model.joblib")
    row = pd.DataFrame([{f: float(df[f].median()) for f in FEATURES}])
    t0 = time.perf_counter()
    proba = float(loaded.predict_proba(row)[0, 1])
    dt = time.perf_counter() - t0
    assert 0.0 <= proba <= 1.0, f"probability out of range: {proba}"
    assert dt < 1.0, f"prediction too slow: {dt:.3f}s"
    print(f"Smoke test OK: median-row probability {proba:.3f} in {dt*1000:.1f} ms")
    print(f"Saved model + schema + thresholds + card to {MODEL_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

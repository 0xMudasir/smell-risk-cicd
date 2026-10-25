"""Model loading, validation and prediction (Phase 7).

Loads model.joblib + feature_schema.json + thresholds.json + model_card.json once,
validates input rows, runs the prediction, and assembles the response:
probability, label, risk_tier, threshold_flags, top_contributors. Risk-tier
cutoffs come from thresholds.json, not code (CLAUDE.md Section 6).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model"
TOP_CONTRIBUTORS = 5


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "model.joblib")
        self.schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
        self.thresholds = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
        self.features: list[str] = self.schema["order"]
        self.model_name = self.card["model"]["name"]
        self.model_version = self.card["model"]["version"]
        self._tiers = self.thresholds["risk_tiers"]
        self._metric_thr = self.thresholds["metric_thresholds"]

    # ------------------------- validation -------------------------
    def validate(self, metrics: dict) -> tuple[dict, dict, list]:
        """Return (clean_values, field_errors, ignored_keys)."""
        if not isinstance(metrics, dict):
            return {}, {"metrics": "must be an object of feature -> number"}, []
        errors, clean = {}, {}
        for f in self.features:
            if f not in metrics or metrics[f] is None or metrics[f] == "":
                errors[f] = "required"
                continue
            try:
                v = float(metrics[f])
            except (TypeError, ValueError):
                errors[f] = "must be a non-negative number"
                continue
            if not math.isfinite(v):
                errors[f] = "must be finite"
            elif v < 0:
                errors[f] = "must be a non-negative number"
            else:
                clean[f] = v
        ignored = [k for k in metrics if k not in self.features]
        return clean, errors, ignored

    # ------------------------- prediction -------------------------
    def _probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def _contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature SHAP contributions for XGBoost; else an importance proxy."""
        clf = self.model.named_steps.get("clf", self.model)
        try:
            import xgboost as xgb
            booster = clf.get_booster()
            contribs = booster.predict(xgb.DMatrix(X, feature_names=self.features),
                                       pred_contribs=True)
            return contribs[:, :-1]  # drop bias column
        except Exception:
            if hasattr(clf, "feature_importances_"):
                imp = np.asarray(clf.feature_importances_)
                return np.tile(imp, (len(X), 1))
            if hasattr(clf, "coef_"):
                return X.to_numpy() * np.ravel(clf.coef_)
            return np.zeros((len(X), len(self.features)))

    def _threshold_flags(self, values: dict) -> list:
        flags = []
        for m, thr in self._metric_thr.items():
            if m not in values:
                continue
            v = values[m]
            level = None
            if v > thr["critical"]:
                level = "critical"
            elif v > thr["warning"]:
                level = "warning"
            if level:
                flags.append({"metric": m, "value": v, "warning": thr["warning"],
                              "critical": thr["critical"], "level": level})
        return flags

    def _risk_tier(self, prob: float, flags: list) -> str:
        n_crit = sum(1 for f in flags if f["level"] == "critical")
        low = self._tiers["low"]
        crit = self._tiers["critical"]
        if prob >= crit["min_probability"] or n_crit >= crit["min_critical_flags"]:
            return "critical"
        if prob < low["max_probability"] and n_crit == 0:
            return "low"
        return "elevated"

    def _response(self, values: dict, prob: float, contrib_row: np.ndarray) -> dict:
        flags = self._threshold_flags(values)
        order = np.argsort(-np.abs(contrib_row))[:TOP_CONTRIBUTORS]
        top = [{"metric": self.features[i], "value": values[self.features[i]],
                "impact": float(contrib_row[i])} for i in order]
        return {
            "probability": round(float(prob), 4),
            "label": int(prob >= 0.5),
            "risk_tier": self._risk_tier(prob, flags),
            "threshold_flags": flags,
            "top_contributors": top,
            "model": {"name": self.model_name, "version": self.model_version},
        }

    def predict_one(self, metrics: dict) -> dict:
        clean = {f: metrics[f] for f in self.features}
        X = pd.DataFrame([[clean[f] for f in self.features]], columns=self.features)
        prob = float(self._probability(X)[0])
        contrib = self._contributions(X)[0]
        return self._response(clean, prob, contrib)

    def predict_batch(self, df: pd.DataFrame) -> list:
        X = df[self.features].astype(float)
        probs = self._probability(X)
        contribs = self._contributions(X)
        rows = []
        for i in range(len(X)):
            values = {f: float(X.iloc[i][f]) for f in self.features}
            r = self._response(values, probs[i], contribs[i])
            r["row"] = int(i)
            rows.append(r)
        rows.sort(key=lambda r: -r["probability"])
        return rows

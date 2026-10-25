"""Flask ML API (Phase 7).

Endpoints (CLAUDE.md Section 6):
  GET  /api/health         -> {status, model_loaded, version}
  GET  /api/model-info     -> model_card.json + feature list
  GET  /api/schema         -> feature_schema.json
  POST /api/predict        -> single prediction
  POST /api/predict/batch  -> CSV upload (<=5 MB, <=5000 rows)

/api/analyze-code is intentionally not implemented (D7.1): the model needs
Halstead operator/operand counts that open-source extractors do not produce.

Model, schema and thresholds load once at startup. CORS allows
http://localhost:4200 and any https://*.github.io origin.
"""
from __future__ import annotations

import io

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from predictor import Predictor

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5000


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    CORS(app, resources={r"/api/*": {"origins": [
        "http://localhost:4200",
        r"https://.*\.github\.io",
    ]}})

    predictor = Predictor()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": True, "version": predictor.model_version})

    @app.get("/api/model-info")
    def model_info():
        return jsonify({**predictor.card, "features": predictor.features})

    @app.get("/api/schema")
    def schema():
        return jsonify(predictor.schema)

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        # accept {"metrics": {...}} or a bare {feature: value} object
        metrics = body["metrics"] if isinstance(body, dict) and "metrics" in body else body
        clean, errors, ignored = predictor.validate(metrics)
        if errors:
            return jsonify({"error": "validation_error", "fields": errors}), 400
        result = predictor.predict_one(clean)
        if ignored:
            result["ignored"] = ignored
        return jsonify(result)

    @app.post("/api/predict/batch")
    def predict_batch():
        if "file" not in request.files:
            return jsonify({"error": "validation_error",
                            "fields": {"file": "a CSV file is required (multipart field 'file')"}}), 400
        raw = request.files["file"].read()
        if len(raw) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "file_too_large", "max_bytes": MAX_UPLOAD_BYTES}), 400
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            return jsonify({"error": "parse_error", "message": str(e)[:200]}), 400
        if len(df) > MAX_ROWS:
            return jsonify({"error": "too_many_rows", "max_rows": MAX_ROWS, "rows": int(len(df))}), 400
        missing = [f for f in predictor.features if f not in df.columns]
        if missing:
            return jsonify({"error": "validation_error",
                            "fields": {m: "column required" for m in missing}}), 400
        try:
            num = df[predictor.features].apply(pd.to_numeric, errors="coerce")
        except Exception:
            return jsonify({"error": "validation_error",
                            "fields": {"csv": "all feature columns must be numeric"}}), 400
        if num.isna().any().any() or (num < 0).any().any():
            return jsonify({"error": "validation_error",
                            "fields": {"csv": "feature values must be finite and non-negative"}}), 400
        results = predictor.predict_batch(num)
        return jsonify({"count": len(results), "results": results})

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "file_too_large", "max_bytes": MAX_UPLOAD_BYTES}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

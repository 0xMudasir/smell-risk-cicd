"""Phase 6 - Export artifacts to the app.

Copies the RQ result JSONs from ml/artifacts/ (+ paper_reference and the model
card) to frontend/src/assets/results/, and verifies the model files the API
needs are present in backend/model/. The frontend reads these static JSONs so the
GitHub Pages site works without the API running (CLAUDE.md 7.6). Rule 2: the UI
shows only numbers produced by the pipeline.

Run:  python -m ml.src.export_artifacts
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from .config import ROOT, load_config, p

CFG = load_config()
ARTIFACTS = p(CFG["paths"]["artifacts"])
FRONTEND_RESULTS = ROOT / "frontend" / "src" / "assets" / "results"
BACKEND_MODEL = ROOT / "backend" / "model"

# result JSONs the research pages consume
RESULT_FILES = [
    "dataset_audit.json",
    "preprocess_summary.json",
    "rq1_importance.json",
    "rq1_correlation.json",
    "rq2_stats.json",
    "rq2_tests.json",
    "rq2_thresholds.json",
    "rq3_results.json",
    "rq3_nemenyi.json",
    "rq3_roc.json",
    "rq3_confusion.json",
    "rq3_improvements.json",
]
# extra static files the shell/overview need
EXTRA = [
    (p("ml/config/paper_reference.json"), "paper_reference.json"),
    (BACKEND_MODEL / "model_card.json", "model_card.json"),
    (BACKEND_MODEL / "feature_schema.json", "feature_schema.json"),
    (BACKEND_MODEL / "thresholds.json", "thresholds.json"),
]
MODEL_FILES = ["model.joblib", "feature_schema.json", "thresholds.json", "model_card.json"]


def main() -> None:
    FRONTEND_RESULTS.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []

    for name in RESULT_FILES:
        src = ARTIFACTS / name
        if src.exists():
            shutil.copy2(src, FRONTEND_RESULTS / name)
            copied.append(name)
        else:
            missing.append(name)

    for src, dest_name in EXTRA:
        if src.exists():
            shutil.copy2(src, FRONTEND_RESULTS / dest_name)
            copied.append(dest_name)
        else:
            missing.append(dest_name)

    # manifest for the frontend empty-state check + version banner
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(copied),
        "model": None,
    }
    card = BACKEND_MODEL / "model_card.json"
    if card.exists():
        mc = json.loads(card.read_text(encoding="utf-8"))
        manifest["model"] = {"name": mc["model"]["name"], "version": mc["model"]["version"]}
    (FRONTEND_RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # verify backend model files
    model_missing = [f for f in MODEL_FILES if not (BACKEND_MODEL / f).exists()]

    print(f"Copied {len(copied)} files to {FRONTEND_RESULTS.relative_to(ROOT)}")
    if missing:
        print(f"  WARNING missing source artifacts (run the RQ phases): {missing}")
    if model_missing:
        print(f"  WARNING backend model files missing (run train_final): {model_missing}")
    else:
        print(f"Backend model files present in {BACKEND_MODEL.relative_to(ROOT)}: {MODEL_FILES}")
    print("Wrote manifest.json")


if __name__ == "__main__":
    main()

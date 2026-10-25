"""Flask ML API (Phase 7).

Endpoints (CLAUDE.md Section 6):
  GET  /api/health         -> {status, model_loaded, version}
  GET  /api/model-info     -> model_card.json + feature list
  GET  /api/schema         -> feature_schema.json
  POST /api/predict        -> single prediction
  POST /api/predict/batch  -> CSV upload (<=5 MB, <=5000 rows)
  POST /api/analyze-code   -> optional, only if feasible after Phase 1

Model, schema and thresholds load once at startup. CORS allows
http://localhost:4200 and the GitHub Pages origin. Implemented in Phase 7.
"""


def create_app():
    raise NotImplementedError("Implemented in Phase 7.")


if __name__ == "__main__":
    raise NotImplementedError("Implemented in Phase 7.")

"""Model loading, validation and prediction (Phase 7).

Loads model.joblib + feature_schema.json + thresholds.json once, validates input
rows (all schema features required, numeric, finite, non-negative), runs the
prediction, and assembles the response: probability, label, risk_tier,
threshold_flags, top_contributors. Risk-tier cutoffs come from thresholds.json,
not code (CLAUDE.md Section 6).
"""


class Predictor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 7.")

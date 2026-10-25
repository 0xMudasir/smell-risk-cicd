"""Phase 5 - Train and save the deployable model.

Picks the best model by ROC-AUC (recall tie-breaker), refits on all data with the
same leakage-safe pipeline, and saves to backend/model/: model.joblib,
feature_schema.json, thresholds.json, model_card.json. Must load and predict one
row in under 1 second.

Run:  python -m ml.src.train_final
"""


def main() -> None:
    raise NotImplementedError("Implemented in Phase 5.")


if __name__ == "__main__":
    main()

"""Phase 5 - RQ3 classifiers.

Replication track: the 11 models with Table 4 hyperparameters inside a
leakage-safe imblearn Pipeline ([scaler] -> SMOTE -> clf), using the Section 4.2
CV. Reports accuracy, precision, recall, F1, MCC, ROC-AUC and PR-AUC per fold and
mean +/- std. Friedman + Nemenyi on ROC-AUC. Improvement track ("Ours, tuned"):
nested RandomizedSearchCV for XGB/RF/AdaBoost, SMOTE vs class_weight, decision-
threshold tuning, and the circularity ablation if Section 2.2 applies.

Writes rq3_results.json, rq3_roc.json, rq3_confusion.json, rq3_nemenyi.json.

Run:  python -m ml.src.rq3_models
"""


def main() -> None:
    raise NotImplementedError("Implemented in Phase 5.")


if __name__ == "__main__":
    main()

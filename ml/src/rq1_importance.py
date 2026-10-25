"""Phase 3 - RQ1 feature importance.

Replicates the paper's 5 importance methods (LR coef, LogReg coef, XGBoost gain,
ANOVA F-score, Boruta) + Pearson correlation, builds strict and 4-of-5 consensus
lists, and adds the fix track (permutation importance on CV test folds, SHAP,
correlated-pair flags). Writes ml/artifacts/rq1_importance.json and
rq1_correlation.json plus figures.

Run:  python -m ml.src.rq1_importance
"""


def main() -> None:
    raise NotImplementedError("Implemented in Phase 3.")


if __name__ == "__main__":
    main()

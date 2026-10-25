# Paper review

Shortcomings found in Diouf et al. (2026), *Predicting Software Security Bugs
Using Machine Learning and Quality Metrics* (CMC 87(3), DOI
10.32604/cmc.2026.077139), and what we did instead. Paper numbers are from
`ml/config/paper_reference.json` (transcribed from the article); our numbers come
from `ml/artifacts/`.

> Framing: we replicate the paper's **method** on a code-smell dataset
> (SmellyCode++), not its security-bug data. "Better/worse" below is about method
> rigour, not about beating their scores.

| # | Status |
|---|---|
| 1 Inconsistent Table 15 F1 | confirmed (arithmetic) |
| 2 Mislabelled density-plot axes | addressed in our figures |
| 3 Categorical features inside SMOTE | not applicable to us; avoided by design |
| 4 Coef importance without scaling, on correlated metrics | confirmed and handled |
| 5 Near-duplicate rows leak across splits | handled by grouped CV |
| 6 Thresholds never validated | validated in RQ2 |
| 7 Weak, untuned hyperparameters | kept for replication + tuned track added |
| 8 Accuracy-led under imbalance, no PR-AUC | PR-AUC + mean/std added |
| 9 Minor reporting slips | noted |

---

## 1. Inconsistent F1 values in Table 15 (SVMs)

- **Paper:** Linear SVM precision 0.35, recall 0.30, **F1 0.52**; RBF SVM
  precision 0.29, recall 0.36, **F1 0.47**.
- **Problem:** F1 is the harmonic mean of precision and recall, so those inputs
  force F1 = 0.323 (linear) and 0.321 (RBF). The reported 0.52 / 0.47 are
  impossible from the stated P/R; they look transcribed or computed by hand.
- **Us:** every metric is computed from predictions with scikit-learn, per fold,
  then averaged. Our XGBoost mean P 0.284 / R 0.641 / F1 0.393 is internally
  consistent. (`ml/src/rq3_models.py` -> `rq3_results.json`.)

## 2. Mislabelled figures

- **Paper:** the density plots for SumCyclomatic, SumCyclomaticModified and
  AvgCyclomaticModified carry other metric names on the x-axis (MaxEssential,
  CountStmtDecl, CountLineComment).
- **Problem:** a reader cannot trust which distribution belongs to which metric.
- **Us:** every RQ2 density + box figure is drawn by one function
  (`plot_metric(metric, ...)` in `ml/src/rq2_statistics.py`) that receives the
  metric name once and uses it for the title and both axis labels, so a label can
  never drift from its data. Figures in `ml/artifacts/figures/rq2/`.

## 3. Categorical features inside SMOTE

- **Paper:** states all predictors are numeric, yet `extension` and `Ecosystem`
  appear in its importance tables; SMOTE then interpolates encoded categories,
  which invents fractional category codes that correspond to no real category.
- **Problem:** synthetic minority rows with impossible categorical values.
- **Us:** not applicable - SmellyCode++ has no categorical predictor (`Project`
  is a grouping key, never a feature; D2.3). We therefore run plain SMOTE safely.
  Had a categorical existed, the plan was one-hot + SMOTENC (CLAUDE.md 4.3).

## 4. Coefficient importance without stated scaling, on correlated metrics

- **Paper:** reports Linear/Logistic Regression coefficient importance without
  saying it standardized, and over metrics that are highly correlated.
- **Problem:** unscaled coefficients are not comparable across metrics with
  different units, and with collinear metrics the coefficient mass splits
  arbitrarily among near-duplicates, so the ranking is unstable.
- **Us:** we standardize before taking |coef| (`ml/src/rq1_importance.py`), and we
  flag the collinearity directly: **39 metric pairs with |r| > 0.9**, several
  exactly 1.0 (`Bugs` = Volume/3000, `Time Required` = Effort/18). We tell readers
  to prefer the tree / permutation / SHAP rankings. The same collinearity makes
  QDA's covariance singular at the paper's `reg_param=0` (see RQ3), a concrete
  demonstration of the hazard. (`rq1_correlation.json`, `rq1_summary.md`.)

## 5. Pre-patch / post-patch near-duplicate rows leak across splits

- **Paper:** keeps pre- and post-patch versions of the same file with opposite
  labels; under random or poorly grouped CV, near-identical rows land on both
  sides of a split and inflate scores.
- **Problem:** optimistic, leaked evaluation.
- **Us:** our analogue is that SmellyCode++ mixes a class row with its method
  rows under one `File+Project+Class` identifier (audit Section 5b). We evaluate
  with `StratifiedGroupKFold` grouped on `Project`, so every entity of a project
  (hence every method of a class) stays on one side of each fold - verified zero
  train/test project overlap in all 5 folds (`ml/src/cv.py`). No row dropping was
  needed (D1.5/D2.0).

## 6. Thresholds are never validated

- **Paper:** defines warning = buggy-class median and critical = buggy-class Q3
  (Table 14) but never reports how well "metric > threshold" actually detects
  bugs.
- **Problem:** descriptive quantiles are presented with the authority of a
  detection rule.
- **Us:** we evaluate each threshold **as a classifier** on grouped-CV test folds
  (fit on train, score on test) and add a Youden-J ROC threshold
  (`rq2_thresholds.json`, `as_classifier_cv`). Result: single-metric thresholds
  are weak detectors (best F1 ~0.37; Youden reaches recall 0.63-0.89 but precision
  0.15-0.20). This is why RQ3 uses multivariate models. (`rq2_summary.md`.)

## 7. Weak, untuned hyperparameters

- **Paper:** e.g. Random Forest with 10 trees, depth 3, max_features 1 - clearly
  under-powered - yet reports it as the model's capability.
- **Problem:** conclusions about a model class rest on an untuned instance.
- **Us:** we keep Table 4 exactly for the replication track (our RF scores
  AUC 0.794), then add a nested RandomizedSearchCV "tuned" track: tuned RF reaches
  **0.843** and tuned XGBoost **0.847** vs 0.835 untuned. The gap quantifies the
  cost of not tuning. (`rq3_improvements.json`, `rq3_summary.md`.)

## 8. Accuracy-led reporting under imbalance, no PR-AUC

- **Paper:** headlines accuracy (e.g. 0.98) on a 9.84%-positive problem, where a
  trivial all-negative classifier already scores ~0.90, and reports no PR-AUC and
  no per-fold spread.
- **Problem:** accuracy flatters models under class imbalance.
- **Us:** we report **PR-AUC** (more honest under imbalance) alongside the six
  paper metrics, and give **mean +/- std over the 5 folds** for every metric
  (`rq3_results.json`). We lead with ROC-AUC / PR-AUC / MCC, not accuracy.

## 9. Minor reporting slips

- **Paper:** the non-buggy count is given as 305,148 in Table 11, while
  338,442 - 33,294 = 305,148 exactly, and other tables imply 305,149; the
  multiple-comparison correction is mentioned only in the threats-to-validity
  section rather than applied throughout.
- **Us:** we apply a Holm correction across metrics in RQ2 explicitly
  (`rq2_tests.json`, `p_holm`), and our own counts reconcile exactly (107,554
  rows = 97,636 clean + 9,918 smelly, `dataset_audit.json`).

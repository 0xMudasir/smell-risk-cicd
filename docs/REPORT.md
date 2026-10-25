# Smell Risk Dashboard — final report

**Author:** Mudasir Ali (BSSE-F23-A, Air University)
**Repo:** https://github.com/0xMudasir/smell-risk-cicd · **Live site:** https://0xmudasir.github.io/smell-risk-cicd/

Every number in this report comes from a file in `ml/artifacts/`, produced by the
pipeline with seed 42. Paper numbers come from `ml/config/paper_reference.json`
and are always labelled "Paper".

---

## 1. Goal

Replicate the **methodology** of Diouf, Toe, Grichi, Nakouri and Jaafar (2026),
*Predicting Software Security Bugs Using Machine Learning and Quality Metrics*
(CMC 87(3), DOI 10.32604/cmc.2026.077139), on a **code-smell** dataset, and carry
it through a full software-engineering lab workflow: VS Code → GitHub → GitHub
Actions CI → CD to GitHub Pages → an Angular dashboard talking to a Flask ML API.

The paper predicts security bugs; our target is code smells. We therefore
replicate the **method** (RQ1 importance, RQ2 statistics/thresholds, RQ3
classifiers with grouped CV) and compare **patterns** — do the same metric
families matter, does a "3× rule" appear, do tree ensembles win — rather than
exact numbers.

## 2. Dataset

**SmellyCode++** (KFUPM, *Scientific Data*, 2025): 107,554 Java entities from 89
projects, 4 binary smell labels. The primary target `has_smell` (any smell) is
positive for **9.22%** of rows — close to the paper's 9.84% bug rate.

Key structural findings from the Phase 1 audit (`ml/reports/dataset_audit.md`):

- **Granularity is mixed** method/class: one `File+Project+Class` identifier recurs
  up to 17× (a class row plus its method rows). The method-family {Long method,
  Feature envy} and class-family {God class, Data class} never co-occur.
- **Metrics are Halstead** (operators/operands, volume, difficulty, effort, …) plus
  cyclomatic complexity and logical lines — **not** the paper's 25 SciTools
  Understand metrics. Only cyclomatic complexity and logical lines map across.
- **Severe collinearity**: 39 metric pairs with |r| > 0.9, several exactly 1.0
  (`Bugs` = Volume/3000, `Time Required` = Effort/18).
- **No date column** → cross-validation uses project grouping, not time.
- **Circularity (label leakage) verdict: plausible but not trivial.** SmellyCode++
  annotates "based on prior studies"; the exact rule is unconfirmed. Single-metric
  separability is only moderate (best Cohen's d ≈ 0.78), so no one metric is the
  label rule.

## 3. Method and the leakage-safe pipeline

- **Target:** `has_smell` = 1 if any of the 4 smells is positive (per-smell models
  are an extension).
- **Cross-validation:** `StratifiedGroupKFold(n_splits=5, groups=Project)` (Section
  4.2 option 2). Verified: per-fold smelly rate 9.13–9.46%, **zero** train/test
  project overlap. Grouping keeps every method of a class on one side of each fold,
  so no row dropping was needed.
- **Leakage controls:** every model is an `imblearn` pipeline
  `[StandardScaler if scale-sensitive] → SMOTE → classifier`, so SMOTE and scaling
  are fit on training folds only; test folds keep the real 9.22% ratio. Identifiers
  are never features. SmellyCode++ has no categorical predictor, so plain SMOTE is
  safe (no SMOTENC needed).

## 4. RQ1 — feature importance

Five methods (Linear/Logistic Regression |coef| on standardized features, XGBoost
gain, ANOVA F, Boruta) plus Pearson, with a fix track (permutation importance on
CV test folds, SHAP). Details in `ml/reports/rq1_summary.md`.

- **`Logical Lines` is rank 1 in every method**, and in permutation and SHAP.
- **Strict consensus (5/5):** Logical Lines, Total Operands, Vocabulary, Length,
  Calculated Length. **≥4/5** adds Volume, Bugs and **Cyclomatic Complexity**.
- **Pattern vs paper:** the paper's core is the size + cyclomatic families; ours is
  too (whole strict consensus is size-scaled, cyclomatic enters at 4/5). Metric
  *names* cannot overlap (Halstead vs Understand), so the comparison is at the
  family level — **and it matches.**
- **Caveat:** coefficient rankings are unstable under the heavy collinearity; the
  tree/permutation/SHAP rankings are preferred.

## 5. RQ2 — statistics and thresholds

For the 8 consensus metrics: per-class stats, Mann-Whitney U (full + balanced
undersample), rank-biserial effect size, Holm correction, and warning/critical
thresholds (smelly median / Q3). Fix track: thresholds validated as classifiers on
CV test folds, plus Youden-J. Details in `ml/reports/rq2_summary.md`.

- **All 8 metrics separate the classes** with medium-to-large effect (rank-biserial
  0.49–0.61; all p survive Holm).
- **The "3× rule" appears, usually stronger:** smelly medians are 3.6× (Vocabulary)
  to 11× (Volume) the clean median; the paper reported ≈3× for bugs.
- **Honest finding the paper omits:** single-metric thresholds are weak detectors —
  best F1 ≈ 0.37 (Logical Lines warning); Youden reaches recall 0.63–0.89 but
  precision 0.15–0.20, so F1 falls. Thresholds are descriptive, not deployable
  rules, which is why RQ3 uses multivariate models.

## 6. RQ3 — classifiers

11 models with the paper's Table 4 hyperparameters in the leakage-safe pipeline on
grouped CV; 7 metrics including PR-AUC; Friedman + Nemenyi on ROC-AUC. Improvement
track: nested tuning, SMOTE vs `scale_pos_weight`, threshold tuning, size ablation.
Details in `ml/reports/rq3_summary.md`.

| Model | ROC-AUC (ours) | Paper AUC | F1 | MCC |
|---|---|---|---|---|
| **XGBoost** | **0.835** | 0.91 | 0.393 | 0.339 |
| AdaBoost | 0.809 | 0.89 | 0.302 | 0.254 |
| Decision Tree | 0.803 | 0.90 | 0.303 | 0.249 |
| Random Forest | 0.794 | 0.90 | 0.271 | 0.235 |
| … | … | … | … | … |
| Naive Bayes | 0.744 | 0.53 | 0.168 | -0.042 |

- **XGBoost wins** (Nemenyi mean rank 1.0, Friedman p = 2e-5); tree ensembles
  dominate — **the same pattern as the paper.** Our absolute AUCs are lower (0.835
  vs 0.91), expected given a different target, a smaller Halstead feature set, and
  realistic grouped CV.
- **Honest divergence:** our linear models beat the paper's (Logistic Regression
  0.79 vs 0.51; both SVMs ≈ 0.80 vs 0.69–0.71). Naive Bayes collapses
  (recall 0.99, MCC −0.04), as in the paper.
- **Improvement track:** nested tuning lifts XGBoost to AUC 0.847 and the weak
  Table-4 RF from 0.794 to 0.843; `scale_pos_weight` marginally beats SMOTE; a
  recall-0.80 operating point is available via threshold tuning.
- **Circularity probe:** dropping the 5 size-dominant metrics lowers XGBoost AUC only
  0.835 → 0.816 — the signal is not purely raw size, so the labels are not a trivial
  size threshold.
- **Deployable model:** XGBoost refit on all data → `backend/model/model.joblib`
  (2.8 MB), median-row prediction ≈ 6 ms.

## 7. Paper review

Nine shortcomings documented and addressed in `docs/PAPER_REVIEW.md`, each tied to a
file in `ml/artifacts/` and the quantitative ones verified by computation — e.g.
the paper's SVM F1 values (0.52 / 0.47) are arithmetically impossible from their
P/R (should be 0.32), the density-plot axes are mislabelled, and the thresholds are
never validated. We fix these by computing every metric from predictions,
generating figures from one labelled function, validating thresholds on held-out
folds, adding PR-AUC and mean ± std, and applying a Holm correction.

## 8. The CI/CD lab workflow

- **CI** (`.github/workflows/ci.yml`, on push/PR to `main`): backend `compileall` +
  `pytest` (10 tests), a model-smoke job that loads `model.joblib` and predicts a
  row, and an Angular production build. Runs on **Python 3.13** — the trained stack
  (numpy 2.5.x) requires ≥3.12, which the first CI run surfaced.
- **CD** (`deploy.yml`): on green CI it generates `build-info.ts`, builds with
  `--base-href "/<repo>/"` (read from the event), and deploys the browser bundle to
  GitHub Pages. The live top-bar build tag shows the commit SHA + run number —
  the proof that CD ran.
- **Backend hosting:** Flask runs locally; the live Predict page calls
  `http://localhost:5000` and shows an offline state elsewhere.

## 9. Conclusions

Applying the paper's method to a different target and a different metric toolkit,
the **patterns replicate**: size and cyclomatic complexity dominate importance, a
multiplicative "amplification" (3×+) separates smelly from clean, and tree
ensembles — XGBoost first — win. The absolute scores are lower and more honest
(grouped CV, PR-AUC, validated thresholds), and several of the paper's reporting
problems do not survive re-computation. The whole result is reproducible from the
raw dataset with one command per phase and is served through a complete
VS Code → GitHub → CI → CD → Pages + API pipeline.

# CLAUDE.md: Smell Risk Dashboard (paper replication + full CI/CD)

> Working title: **Smell Risk Dashboard**. Suggested repo name: `smell-risk-cicd`.
> Owner: Mudasir Ali (BSSE-F23-A, Air University). Read this whole file before doing anything.

---

## 0. How to work in this repo (rules for Claude Code)

1. **Phased and stop-gated.** Work one phase at a time (Section 5). At the end of each phase: run its checks, print a short summary (what was done, files created, key numbers, open questions), then **STOP and wait for "continue"**. Never start the next phase on your own.
2. **Never invent numbers.** Every number shown in the UI, README or report must come from a file in `ml/artifacts/` produced by the pipeline. Paper numbers live only in `ml/config/paper_reference.json` and are always labelled "Paper".
3. **Explain the why.** For every non-trivial decision (target definition, CV strategy, encoding, hyperparameters, thresholds), append one entry to `docs/DECISIONS.md`: *decision, alternatives, reason, effect*. The instructor grades understanding of what was done, how, and why.
4. **Ask before** changing the label definition, dropping columns or rows beyond exact duplicates, or changing the CV strategy chosen in Phase 2.
5. **Never commit** `data/raw/`, any file > 50 MB, `.env`, or virtualenvs.
6. **Pin versions.** `scikit-learn`, `xgboost`, `imbalanced-learn`, `numpy` must be the *same versions* in `ml/requirements.txt` and `backend/requirements.txt`, otherwise the saved model will not load in the API.
7. Prefer small, readable modules over notebooks. Notebooks are allowed only in `ml/notebooks/` for exploration; the reproducible pipeline is the `ml/src/` scripts.
8. Windows-friendly commands where possible (the owner works in VS Code on Windows). Use `python -m ...` forms.

---

## 1. Project context

### 1.1 The reference paper
Diouf, Toe, Grichi, Nakouri, Jaafar. *Predicting Software Security Bugs Using Machine Learning and Quality Metrics: An Empirical Study.* Computers, Materials & Continua 87(3), 2026. DOI 10.32604/cmc.2026.077139. Companion repo: https://github.com/MdioufDataScientist/PredictSecBugs (study its scripts before Phase 3).

What the paper does, in one pass:

| Step | Paper | Output |
|---|---|---|
| Data | 338,442 files from 7 OSV ecosystems, 33,294 buggy (9.84%) | binary label `buggy` |
| Features | 25 static metrics from SciTools Understand (size, cyclomatic, structure) + `extension` | feature matrix |
| RQ1 | 5 importance methods: Linear Regression coef, Logistic Regression coef, XGBoost gain, ANOVA F-score, Boruta; plus Pearson correlation; intersect top-10s | 6 core metrics |
| RQ2 | Per-class descriptive stats, density + box plots, Mann-Whitney U, rank-biserial effect size | "3x rule", warning (buggy median) and critical (buggy Q3) thresholds |
| RQ3 | 11 classifiers, SMOTE inside training folds only, 5-fold time-series CV, Friedman + Nemenyi | XGBoost best: Acc 0.98, P 0.95, R 0.82, F1 0.88, MCC 0.87, AUC 0.91 |

### 1.2 Our task
Apply the **same methodology** to `multi-smell-dataset` and carry it through the full lab workflow (VS Code -> GitHub -> GitHub Actions CI -> CD to GitHub Pages -> Angular frontend talking to a Flask ML API).

**Important framing.** Our dataset labels *code smells*, not *security bugs*. So we replicate the **method** and compare the **patterns** (do the same metric families matter? does a "3x rule" appear? do tree ensembles win?). We do not expect to match the paper's exact numbers, and the UI must never imply we did.

### 1.3 Paper reference values (store in `ml/config/paper_reference.json`)

Table 15 (RQ3, mean over 5 time-series folds):

| Model | Acc | Prec | Rec | F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.98 | 0.95 | 0.82 | 0.88 | 0.87 | 0.91 |
| Random Forest | 0.97 | 0.89 | 0.82 | 0.85 | 0.84 | 0.90 |
| AdaBoost | 0.97 | 0.90 | 0.79 | 0.84 | 0.83 | 0.89 |
| MLP | 0.97 | 0.90 | 0.77 | 0.83 | 0.82 | 0.88 |
| Decision Tree | 0.96 | 0.78 | 0.83 | 0.80 | 0.78 | 0.90 |
| KNN | 0.94 | 0.77 | 0.48 | 0.59 | 0.58 | 0.73 |
| Logistic Regression | 0.90 | 0.31 | 0.03 | 0.05 | 0.06 | 0.51 |
| Linear SVM | 0.86 | 0.35 | 0.30 | 0.52 | 0.25 | 0.71 |
| RBF SVM | 0.87 | 0.29 | 0.36 | 0.47 | 0.22 | 0.69 |
| Naive Bayes | 0.84 | 0.16 | 0.15 | 0.15 | 0.07 | 0.53 |
| QDA | 0.62 | 0.15 | 0.61 | 0.24 | 0.15 | 0.62 |

Table 14 thresholds: CountStmtExe >50 / >130, SumCyclomatic >24 / >57, SumCyclomaticModified >23 / >55, AvgCyclomaticModified >3 / >5, CountDeclClass >2 / >3 (warning / critical).

Table 4 hyperparameters (use exactly in the replication track):

| Model | Hyperparameters |
|---|---|
| KNN | n_neighbors=3 |
| Decision Tree | max_depth=3 |
| Random Forest | max_depth=3, n_estimators=10, max_features=1 |
| MLP | activation="relu", solver="adam", alpha=1, batch_size="auto" |
| AdaBoost | n_estimators=50 |
| XGBoost | defaults |
| Gaussian Naive Bayes | priors=None, var_smoothing=1e-9 |
| QDA | priors=None, reg_param=0.0, store_covariance=False, tol=1e-4 |
| SVM linear | kernel="linear" |
| SVM RBF | kernel="rbf" |
| Logistic Regression | random_state=0, solver="lbfgs", max_iter=100 |

---

## 2. Dataset (fill this section in during Phase 1)

```
Dataset:             SmellyCode++ (KFUPM; Scientific Data, Jul 2025). File: multi-smell-dataset-v1_2.csv
Location:            data/raw/multi-smell-dataset-v1_2.csv        (gitignored, 618 MB)
Rows x columns:      107,554 rows x 22 columns (21 loaded; 'Code' source text never loaded)
Granularity:         MIXED method-level + class-level (one row per sub-class entity). An
                     identifier recurs up to 17x (a class's row + its methods' rows). Finer
                     than the paper's file level. No method-name / granularity-flag column.
Language(s):         Java (OSS projects: jena, hive, hadoop, hbase, nifi, activemq, ...)
Smell label columns: Long method (1.46%), God class (4.03%), Feature envy (1.86%), Data class (3.05%).
                     has_smell (any) = 9.22% smelly. Method-family {Long method, Feature envy} and
                     class-family {God class, Data class} NEVER co-occur (0 rows) -> granularity split.
Metric columns (14): Halstead family -> Distinct/Total Operators, Distinct/Total Operands, Vocabulary,
                     Length, Calculated Length, Volume, Difficulty, Effort, Time Required, Bugs
                     (Halstead delivered-bugs estimate, NOT a label); plus Logical Lines (size) and
                     Cyclomatic Complexity. Only Cyclomatic Complexity (~SumCyclomatic) and Logical
                     Lines (~CountLine) map to the paper's 25 Understand metrics; the Halstead set does
                     not. We replicate the METHOD, not the feature set. Near-all metrics |r|>0.9
                     (several exactly 1.0: Bugs=Volume/3000, Time=Effort/18) -> coef importance unstable.
Identifier columns:  File, Project, Class (+ Code source text). Never used as features.
Date / commit col:   NONE. -> CV falls to grouped CV on Project (Section 4.2 option 2).
Project column:      Project (89 distinct; paper cites ~103 - minor discrepancy). Used for grouped CV.
Label source:        SmellyCode++: AST parse + feature extraction, annotated "based on prior studies"
                     (Fontana-style). Exact rule (manual vs DECOR/PMD/iPlasma thresholds) NOT confirmed
                     from accessible pages (full descriptor behind a browser check). Circularity:
                     PLAUSIBLE but single-metric separability is only moderate (best Cohen's d ~0.78),
                     so no one-metric threshold rule. Confirm before P3; run ablation in P5 regardless.
```

### 2.1 Target definition
- **Primary target (mirrors the paper):** `has_smell = 1` if any smell label is positive, else `0`.
- **Extension:** one binary task per smell type (same pipeline, looped). Report per-smell results in a separate section.
- Multi-label methods (classifier chains etc.) are optional stretch work, not part of the replication.

### 2.2 Circularity check (mandatory, report in Phase 1)
If smell labels were produced by a tool that applies **metric thresholds** (very common: e.g. God Class = high WMC + low cohesion), then models trained on those same metrics will simply rediscover the rule and score near 1.0. That is label leakage, not good modelling. In Phase 1:
- Find out how labels were produced (dataset README / paper).
- If rule-based, list which metrics define each smell, and in Phase 5 run an **ablation** that drops those metrics. Report both.

---

## 3. Repository structure

```
smell-risk-cicd/
├─ CLAUDE.md
├─ README.md
├─ .gitignore
├─ data/
│  ├─ raw/                      # dataset lives here, gitignored
│  └─ README.md                 # source, licence, how to download
├─ ml/
│  ├─ requirements.txt
│  ├─ config/
│  │  ├─ pipeline.yaml          # paths, target, CV strategy, seeds
│  │  └─ paper_reference.json   # Section 1.3 numbers
│  ├─ src/
│  │  ├─ audit.py               # Phase 1
│  │  ├─ preprocess.py          # Phase 2
│  │  ├─ cv.py                  # Phase 2: split factory
│  │  ├─ rq1_importance.py      # Phase 3
│  │  ├─ rq2_statistics.py      # Phase 4
│  │  ├─ rq3_models.py          # Phase 5
│  │  ├─ train_final.py         # Phase 5: fits + saves deployable model
│  │  └─ export_artifacts.py    # Phase 6: copies JSON to frontend + backend
│  ├─ notebooks/                # exploration only
│  ├─ artifacts/                # JSON results, model, figures (generated)
│  └─ reports/                  # dataset_audit.md, results tables (generated)
├─ backend/
│  ├─ app.py
│  ├─ predictor.py              # loads model + schema, validates, predicts
│  ├─ model/                    # model.joblib, feature_schema.json, model_card.json, thresholds.json
│  ├─ requirements.txt
│  └─ tests/                    # pytest
├─ frontend/                    # Angular app (Section 7)
│  └─ src/assets/results/       # static JSON exported from ml/artifacts
├─ docs/
│  ├─ DECISIONS.md
│  ├─ PAPER_REVIEW.md           # shortcomings found + fixes applied
│  └─ REPORT.md                 # final write-up
└─ .github/workflows/
   ├─ ci.yml
   └─ deploy.yml
```

`.gitignore` must include: `data/raw/`, `*.parquet` in data, `.venv/`, `__pycache__/`, `node_modules/`, `frontend/dist/`, `.angular/`, `.env`.

---

## 4. ML pipeline specification

### 4.1 Libraries (`ml/requirements.txt`)
pandas, numpy, pyarrow, pyyaml, scikit-learn, imbalanced-learn, xgboost, Boruta (BorutaPy), scipy, scikit-posthocs, shap, matplotlib, seaborn, joblib. Pin all versions.

### 4.2 Cross-validation strategy (decided in Phase 2)
Choose in this order and record the choice in `DECISIONS.md`:
1. **Date or commit column exists** -> sort by date, `TimeSeriesSplit(n_splits=5)` (same as the paper).
2. **No date, project column exists** -> `StratifiedGroupKFold(n_splits=5, groups=project)` so the same project never sits in train and test (stronger realism than random CV).
3. **Neither** -> `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, flagged as the weakest option.

Global seed: 42 everywhere.

### 4.3 Leakage rules (non-negotiable)
- Every model is an `imblearn.pipeline.Pipeline`: `[scaler if needed] -> SMOTE -> classifier`. SMOTE therefore runs on training folds only; test folds keep the real class ratio.
- Scaling (StandardScaler) only for scale-sensitive models: LR, SVMs, KNN, MLP, QDA. Trees get raw values.
- Categorical columns (e.g. language/extension): one-hot. If SMOTE must see them, use `SMOTENC`, never SMOTE on label-encoded integers.
- Identifiers (names, paths, project) are never features.
- RQ1 importance on the full dataset is **descriptive only**. If any feature selection feeds RQ3, it must be redone inside each training fold.

### 4.4 RQ1: feature importance (replicate + fix)
Replicate:
- Linear Regression |coef| and Logistic Regression |coef| on **standardized** features (the paper does not say it standardized; unscaled coefficients are not comparable).
- XGBoost gain importance, ANOVA F-score (`f_classif`), Boruta (BorutaPy over RandomForest).
- Pearson correlation: metric-metric matrix and metric-target vector.
- Consensus: strict intersection of the five top-10 lists, plus a softer "appears in at least 4 of 5" list (strict intersection can be empty).

Fixes (extra track, clearly labelled):
- Permutation importance computed on CV test folds.
- SHAP summary for the final model.
- Note correlated pairs (|r| > 0.9) because coefficient-based rankings are unstable for them.

Outputs: `ml/artifacts/rq1_importance.json` (per method ranked list + scores, consensus lists), `rq1_correlation.json`, figures.

### 4.5 RQ2: statistics and thresholds (replicate + fix)
Replicate, for each consensus metric:
- Descriptive stats per class: count, mean, std, min, Q1, median, Q3, max, smelly/clean ratio (handle zero denominators by reporting "undefined", as the paper does).
- Density plot and box plot per metric, with **correct axis labels** (the paper's figures have mismatched x-axis labels; we fix that).
- Mann-Whitney U on a balanced random undersample (paper method) and on the full data; rank-biserial effect size with the paper's magnitude bands (0.1 small, 0.3 medium, 0.5 large).
- Holm or Bonferroni correction across metrics.
- Thresholds: warning = smelly-class median, critical = smelly-class Q3 (paper rule).

Fixes:
- Evaluate each threshold **as a classifier** on CV test folds (precision, recall, F1 of "metric > threshold"). The paper never validates its thresholds.
- Add an ROC-derived threshold (Youden's J) per metric for comparison.

Outputs: `rq2_stats.json`, `rq2_tests.json`, `rq2_thresholds.json`, plus the raw quantiles needed to draw the threshold ruler in the UI.

### 4.6 RQ3: classifiers (replicate + improve)
Replication track: the 11 models with Table 4 hyperparameters, inside the leakage-safe pipeline, using the CV from 4.2.
Metrics per fold and mean ± std: accuracy, precision, recall, F1, MCC, ROC-AUC, **plus PR-AUC** (more honest under imbalance).
Statistical comparison: Friedman test, then Nemenyi post-hoc (`scikit-posthocs`) on ROC-AUC; produce mean ranks and groups like the paper's Table 16.

Improvement track (labelled "Ours, tuned"):
- RandomizedSearchCV **inside** each training fold (nested) for XGBoost, Random Forest, AdaBoost.
- SMOTE vs `class_weight` / `scale_pos_weight` comparison.
- Decision-threshold tuning for a high-recall operating point.
- Ablation without rule-defining metrics if the circularity check (2.2) applies.

Outputs: `rq3_results.json` (per model, per fold, mean, std, paper value side by side), `rq3_roc.json` (FPR/TPR points for top models), `rq3_confusion.json`, `rq3_nemenyi.json`.

### 4.7 Final deployable model
- Best model by ROC-AUC with recall as tie-breaker, refit on all data with the same pipeline.
- Save `model.joblib`, `feature_schema.json` (ordered feature names, dtype, min, median, max, description), `thresholds.json`, `model_card.json` (model name, version, trained_at, CV strategy, CV metrics, dataset rows, target definition, library versions).
- Must load and predict in under 1 second on one row.

---

## 5. Phases (stop after each)

| Phase | Goal | Done when |
|---|---|---|
| **P0** Scaffold | Create the structure in Section 3, `.gitignore`, empty requirement files, `git init`, first commit | Tree matches, nothing large tracked |
| **P1** Dataset audit | `audit.py` writes `ml/reports/dataset_audit.md`: shape, dtypes, missing, duplicates, per-smell positive rate, smell co-occurrence matrix, per-project counts, date range, metric list mapped to the paper's 25, label source, circularity verdict. Fill Section 2 of this file | Section 2 has no TBD left; open questions listed |
| **P2** Preprocess + CV | `preprocess.py`, `cv.py`, `pipeline.yaml`; choose CV per 4.2 | Clean dataset saved as parquet; split sanity printed (class ratio per fold, no group overlap) |
| **P3** RQ1 | `rq1_importance.py` | JSON + figures; consensus list printed and compared to the paper's 6 core metrics |
| **P4** RQ2 | `rq2_statistics.py` | Stats, tests, effect sizes, thresholds, threshold-as-classifier scores |
| **P5** RQ3 | `rq3_models.py`, `train_final.py` | Replication table + tuned table + Nemenyi; final model saved and smoke-tested |
| **P6** Review + export | Write `docs/PAPER_REVIEW.md` (Section 9) and `export_artifacts.py` (copies JSON to `frontend/src/assets/results/` and model files to `backend/model/`) | Both destinations populated |
| **P7** Flask API | Section 6 | `pytest` green, manual curl of `/api/predict` works |
| **P8** Angular UI | Section 7 | All pages render from static JSON; Predict works against local API; offline state works |
| **P9** CI/CD | Section 8 | CI green on push; Pages site live; version banner changes after a push |
| **P10** Docs + demo | `README.md`, `docs/REPORT.md`, demo script | A new person can clone, run, and reproduce |

---

## 6. Flask backend specification

Stack: Flask, flask-cors, joblib, pandas, numpy, model libraries (pinned to training versions), gunicorn (for optional hosting), pytest.

Load model, schema and thresholds **once at startup**.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{ "status": "ok", "model_loaded": true, "version": "..." }` |
| GET | `/api/model-info` | contents of `model_card.json` + feature list |
| GET | `/api/schema` | `feature_schema.json` (frontend builds the form from this) |
| POST | `/api/predict` | single prediction |
| POST | `/api/predict/batch` | CSV upload (multipart, max 5 MB, max 5,000 rows) |
| POST | `/api/analyze-code` | **only if feasible after P1**: extract metrics from pasted source with an open-source tool (e.g. `lizard`), then predict. Mark the response `"approximated": true` because the metric definitions differ from the dataset's tool |

`POST /api/predict` request:
```json
{ "metrics": { "<feature_name>": 42, "...": 3 } }
```
Response:
```json
{
  "probability": 0.81,
  "label": 1,
  "risk_tier": "critical",
  "threshold_flags": [
    { "metric": "SumCyclomatic", "value": 71, "warning": 24, "critical": 57, "level": "critical" }
  ],
  "top_contributors": [
    { "metric": "CountStmtExe", "value": 160, "impact": 0.22 }
  ],
  "model": { "name": "XGBoost", "version": "1.0.0" }
}
```
Risk tier: `low` if probability < 0.4 and no critical flag; `critical` if probability >= 0.7 or two or more critical flags; else `elevated`. Put the cutoffs in `thresholds.json`, not in code.

Validation: all schema features required, numeric, finite, non-negative. On failure return 400 `{ "error": "validation_error", "fields": { "<name>": "must be a non-negative number" } }`. Unknown extra keys are ignored and listed under `"ignored"`.

CORS: allow `http://localhost:4200` and `https://<github-username>.github.io`.

Tests (`backend/tests/`): health, schema, valid predict, missing field 400, negative value 400, batch with 3 rows, model loads with pinned versions.

---

## 7. Frontend design and UI specification

### 7.1 Design read
**Reading this as:** a research dashboard plus prediction tool for an instructor and developers, with a precise, instrument-like visual language (closer to a static-analysis report than a marketing site), built in Angular with SCSS tokens and Apache ECharts.

Dials: `DESIGN_VARIANCE 3`, `MOTION_INTENSITY 2`, `VISUAL_DENSITY 7`. This is an app screen, not a landing page: no hero, no gradients, no glassmorphism, no feature-card grids.

If a `ui-ux-pro-max` design-system generator is available in this environment, it may be run as a cross-check before P8, but the tokens below win unless the owner agrees to change them.

### 7.2 Stack
- Angular (current stable, standalone components, signals). Check `ng version` and use what is installed.
- Routing with **hash location** (`withHashLocation()`) so refreshing a page on GitHub Pages never 404s.
- Charts: Apache ECharts via `ngx-echarts` (native box plots, ROC lines, heatmaps).
- Icons: one family only, Lucide (`lucide-angular`).
- Styling: plain SCSS + CSS custom properties. No UI kit; Angular CDK allowed for a11y helpers.
- HTTP: `HttpClient` with a small `ApiService`; static results via `ResultsService` reading `assets/results/*.json`.

### 7.3 Design tokens (`frontend/src/styles/_tokens.scss`)

Light (default):
```scss
:root {
  --paper:      #F3F5F4;  // page background, cool off-white
  --surface:    #FFFFFF;  // panels
  --ink:        #18202A;  // primary text
  --ink-muted:  #5B6672;  // secondary text, axis labels
  --rule:       #D8DEDB;  // borders, dividers, grid lines
  --accent:     #1D5C7A;  // petrol blue: links, active nav, focus ring, selected rows

  --ok:         #2F6B4F;  // clean class, low risk
  --warn:       #B7791F;  // warning threshold, elevated risk
  --crit:       #A8322D;  // smelly class, critical risk
}
```
Dark (under `@media (prefers-color-scheme: dark)` and `[data-theme="dark"]`):
`--paper #121619, --surface #1A2025, --ink #E6EAEC, --ink-muted #9AA5AE, --rule #2B333A, --accent #6AAFCF, --ok #5DAF86, --warn #E0A73E, --crit #E06A62`.

Chart convention (mirrors the paper's red/green figures, but never color-only): smelly = `--crit` solid line/fill; clean = `--ok` dashed line, lighter fill. Legends always present.

Typography:
- **IBM Plex Sans** for all UI text (400, 500, 600). Chosen because it was designed for engineering tools and stays legible at dense sizes.
- **IBM Plex Mono** only for metric identifiers (`CountStmtExe`, `SumCyclomatic`) and code. Metric names are literally code identifiers, so mono is meaningful here, not decoration.
- Numbers in tables: Plex Sans with `font-variant-numeric: tabular-nums`, right-aligned.
- Scale (px): 12 / 14 / 16 / 20 / 24 / 32. Body 14, table 13, page title 24.
- Sentence case everywhere. No all-caps labels, no letter-spaced eyebrows.

Shape and spacing: 4 px spacing base. Radius has two tiers only: 4 px (inputs, chips, buttons) and 8 px (panels). Borders (`1px solid var(--rule)`) instead of shadows. Content max width 1280 px, left-aligned.

Motion: only responsive motion (panel expand, result reveal 150 ms). Respect `prefers-reduced-motion`. No entrance animations.

### 7.4 Signature component: the threshold ruler
The one memorable element. Used on Thresholds and Predict pages. A horizontal scale per metric showing where a value sits against the evidence:

```
SumCyclomatic                                      value 71  ■ critical
|----------●-------------|============|############|------------>
0        8               24           57
     clean median      warning      critical
                     (smelly median) (smelly Q3)
```
- Track segments: neutral up to warning, `--warn` tint warning to critical, `--crit` tint above critical.
- Tick marks: clean median (dashed), warning, critical, with values.
- Marker: the current value (Predict) or the smelly median (Thresholds).
- Log scale toggle for long-tailed metrics; values beyond the axis clamp with an arrow.
- Fully accessible: `role="img"` with an `aria-label` sentence ("SumCyclomatic is 71, above the critical threshold of 57").

### 7.5 App shell
```
┌──────────────────────────────────────────────────────────────────────┐
│ Smell Risk Dashboard        Model XGBoost v1.0   ● API online  Build a1b2c3 #14 │
├────────────┬─────────────────────────────────────────────────────────┤
│ Overview   │                                                         │
│ Dataset    │                 page content                            │
│ Features   │                                                         │
│ Thresholds │                                                         │
│ Models     │                                                         │
│ Predict    │                                                         │
│ Pipeline   │                                                         │
└────────────┴─────────────────────────────────────────────────────────┘
```
- Left rail 220 px; below 900 px it becomes a horizontal scrollable tab bar under the top bar.
- Top bar: title, model name + version (from static `model_card.json`), API status pill (polls `/api/health` every 30 s: "API online" / "API offline"), build SHA + run number (injected by CD, Section 8.3). The build tag is how the demo proves CD worked.

### 7.6 Pages (all research pages read static JSON, so the Pages site works without the API)

**Overview**
```
Replicating Diouf et al. (2026) on a code smell dataset
[ one-paragraph summary of the method, plain language ]

┌ Dataset ─────────┐ ┌ Best model ──────┐ ┌ Core metrics ───────────┐
│ N rows, P% smelly│ │ XGBoost AUC 0.xx │ │ CountStmtExe, ...  (k)  │
└──────────────────┘ └──────────────────┘ └─────────────────────────┘

Paper vs ours (best model)          Acc  Prec  Rec  F1  MCC  AUC
  Paper (security bugs)             0.98 0.95 0.82 ...
  Ours, replication                 ...
  Ours, tuned                       ...
Note: different target (code smells), so compare patterns, not exact numbers.
```

**Dataset**: class balance bar, per-smell positive rate bars, smell co-occurrence heatmap, per-project table (sortable), metric distribution explorer (pick a metric -> density + box plot, smelly vs clean). Every chart has a "View as table" toggle.

**Features (RQ1)**: a rank matrix heatmap (rows = metrics, columns = 5 methods, cell = rank 1..10, empty if outside top-10), consensus metrics highlighted, toggle for strict vs 4-of-5 consensus, correlation heatmap below, permutation/SHAP panel labelled "Ours, extra".

**Thresholds (RQ2)**: table of stats per metric (clean vs smelly: median, Q1, Q3, ratio), Mann-Whitney U, p (corrected), effect size with magnitude chip; under it one threshold ruler per consensus metric; a small table "Threshold as a rule" (precision, recall, F1) next to the paper-style thresholds and the Youden thresholds.

**Models (RQ3)**: leaderboard table of 11 models (mean ± std per metric, sortable, paper value in a muted column next to ours); track switch "Replication / Tuned"; ROC curves for top 3; confusion matrix for selected model; Nemenyi mean-rank chart with group letters.

**Predict**
```
Input   ( Enter metrics | Upload CSV | Paste code [experimental] )

[ form generated from /api/schema: one numeric field per feature,
  placeholder = dataset median, helper text = feature description ]
[ Load typical clean example ] [ Load typical smelly example ]   [ Run prediction ]

Result (aria-live)
  Smell probability 81%     Risk: Critical
  Why: top contributors list (metric, value, impact bar)
  Threshold rulers for each consensus metric with this file's value
```
CSV mode: drag-and-drop, preview first 5 rows, results table sorted by probability with a risk chip per row, "Download results CSV".

**Pipeline**: the lab workflow as a simple step list (VS Code, commit + push, GitHub repo, CI, CD, Pages, Flask API, end to end) with links to the repo, the latest Actions run, and the live site; current build SHA and time.

### 7.7 States and copy
- Loading: skeleton rows/blocks matching the final layout, never spinners alone.
- Empty (no results JSON yet): "No results yet. Run the ML pipeline, then export artifacts."
- API offline on Predict: "The prediction service isn't reachable at {apiBaseUrl}. Start it with `python backend/app.py`, then select Retry." Research pages stay fully usable.
- Validation: inline under the field, e.g. "Enter a number of 0 or more."
- Buttons say exactly what happens: "Run prediction", "Upload CSV", "Download results CSV". Result toast: "Prediction ready".
- No em dashes in any visible UI text. No filler words (seamless, powerful, unleash). Sentence case.

### 7.8 Accessibility and quality floor
WCAG AA contrast in both themes; visible focus ring (`2px solid var(--accent)`, 2 px offset); full keyboard navigation; every chart has a table fallback and an `aria-label`; layout works at 360 px width with no horizontal page scroll; Lighthouse accessibility >= 95.

### 7.9 Frontend structure
```
frontend/src/app/
├─ core/        api.service.ts, results.service.ts, health.service.ts, models.ts (typed interfaces for every JSON)
├─ shared/      threshold-ruler/, metric-name/, risk-chip/, stat-table/, chart-panel/ (chart + "View as table"), skeleton/
├─ layout/      shell/, top-bar/, side-nav/
└─ pages/       overview/, dataset/, features/, thresholds/, models/, predict/, pipeline/
environments/   environment.ts (apiBaseUrl http://localhost:5000), environment.prod.ts, build-info.ts (generated)
styles/         _tokens.scss, _typography.scss, _base.scss
```

---

## 8. CI/CD specification

### 8.1 `ci.yml` (on push and pull_request to `main`)
Jobs:
1. **backend**: checkout, setup-python 3.11 with pip cache, `pip install -r backend/requirements.txt`, `python -m compileall backend ml/src` (syntax check, as in the lab), `pytest backend/tests`.
2. **model-smoke**: load `backend/model/model.joblib` + schema, predict one row built from schema medians, assert probability in [0, 1].
3. **frontend**: setup-node 20 with npm cache, `npm ci`, `npm run build -- --configuration production`.
CI must not need the raw dataset.

### 8.2 `deploy.yml` (CD to GitHub Pages)
- Trigger: `workflow_run` of "CI" with `types: [completed]` on `main`; job runs only `if: github.event.workflow_run.conclusion == 'success'`. Check out `github.event.workflow_run.head_sha`.
- Permissions: `contents: read`, `pages: write`, `id-token: write`.
- Steps: generate `build-info.ts` (SHA, run number, UTC time), `npm ci`, `ng build --configuration production --base-href /<repo-name>/`, upload the browser output folder (check `angular.json` outputPath, usually `dist/<project>/browser`) with `actions/upload-pages-artifact`, then `actions/deploy-pages`. Use the latest major versions of these actions.
- Repo setting: Pages source = GitHub Actions.

### 8.3 Demonstrating CD (step 9 of the lab diagram)
Change something visible (e.g. bump the version label), commit, push; CI runs, CD deploys, the top bar shows the new build SHA and run number on the live site.

### 8.4 Backend hosting
Default (matches the lab): Flask runs locally; the live Pages site calls `http://localhost:5000`, so Predict works on the presenter's machine and shows the offline state elsewhere. Optional: deploy the API to a free host (Render / Hugging Face Spaces) with gunicorn and point `environment.prod.ts` at it. Ask the owner before doing this.

---

## 9. Paper review: shortcomings to document in `docs/PAPER_REVIEW.md`

For each: what the paper says, why it is a problem, what we did instead. Verify each against the PDF before writing.

1. **Inconsistent F1 values in Table 15.** Linear SVM P 0.35 / R 0.30 gives F1 about 0.32, not 0.52; RBF SVM P 0.29 / R 0.36 gives about 0.32, not 0.47. We compute F1 from predictions, never by hand.
2. **Mislabelled figures.** Density plots for SumCyclomatic, SumCyclomaticModified and AvgCyclomaticModified show other metric names on the x-axis (MaxEssential, CountStmtDecl, CountLineComment). Our plots are generated from one function with the metric name passed once.
3. **Categorical features inside SMOTE.** The paper says all predictors are numeric, yet `extension` and `Ecosystem` appear in its importance tables. SMOTE interpolating encoded categories creates invalid values. We use one-hot + SMOTENC where needed.
4. **Coefficient importance without stated scaling, on highly correlated metrics.** We standardize and flag correlated pairs.
5. **Pre-patch and post-patch versions of the same file** are both in the dataset with opposite labels, which makes near-duplicate rows; random or poorly grouped splits leak. We use grouped / time-based CV and deduplicate.
6. **Thresholds are never validated.** Median and Q3 of the buggy class are descriptive; the paper does not report how well "metric > threshold" actually detects bugs. We evaluate them as classifiers and add ROC-based thresholds.
7. **Weak, untuned hyperparameters** (Random Forest with 10 trees, depth 3, max_features 1). We keep them for replication and add a nested-CV tuned track.
8. **Accuracy-led reporting under imbalance and no PR-AUC.** We add PR-AUC and report mean ± std per fold.
9. **Minor reporting slips:** non-buggy count 305,148 vs 305,149 in Table 11; multiple-comparison correction mentioned only in the threats section.

---

## 10. Definition of done (whole project)
- `python -m ml.src.<phase script>` reproduces every artifact from the raw dataset with seed 42.
- `docs/DECISIONS.md`, `docs/PAPER_REVIEW.md`, `docs/REPORT.md` complete.
- CI green, Pages site live, build tag visible, Predict works with the local API, offline state correct.
- README covers: setup (Python venv + Node), dataset placement, run pipeline, run API, run frontend, deploy, and a 5-minute demo script.

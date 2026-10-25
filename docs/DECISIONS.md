# Decisions

One entry per non-trivial decision: *decision, alternatives, reason, effect*
(CLAUDE.md Rule 3). Newest at the bottom of each phase.

---

## Phase 0 - Scaffold

### D0.1 Move the 618 MB dataset into `data/raw/`
- **Decision:** Moved `multi-smell-dataset-v1_2.csv` from the repo root to `data/raw/`.
- **Alternatives:** Leave it in root; keep it outside the repo entirely.
- **Reason:** `data/raw/` is gitignored, so the file cannot be committed by accident (Rule 5: never commit `data/raw/` or files > 50 MB). Keeping it inside the repo tree (but ignored) keeps paths stable for the pipeline.
- **Effect:** `ml/config/pipeline.yaml` points `paths.raw` here. Anyone cloning the repo must place the dataset at this path (documented in `data/README.md`).

### D0.2 Gitignore the reference paper PDF
- **Decision:** Added `TSP_CMC_77139.pdf` and `*.pdf` to `.gitignore`.
- **Alternatives:** Commit the PDF (it is ~4 MB, under the 50 MB limit).
- **Reason:** It is a copyrighted published article; it should not be redistributed via a public GitHub repo. It stays on the owner's machine for reference.
- **Effect:** `docs/PAPER_REVIEW.md` must cite the paper by DOI, not bundle it.

### D0.3 Starter requirement files, pin exact versions in Phase 2
- **Decision:** `ml/requirements.txt` and `backend/requirements.txt` list libraries unpinned for now, with the four load-critical ones (`scikit-learn`, `xgboost`, `imbalanced-learn`, `numpy`) grouped and flagged; exact `==` pins are added in Phase 2 against a verified install.
- **Alternatives:** Pin guessed versions now.
- **Reason:** Guessing versions risks pinning a combination that does not resolve on both local Python 3.13 and CI Python 3.11, or that breaks `Boruta` (older releases use removed numpy aliases). Rule 6 (identical pins in both files) matters at model save/load time (Phase 5-7), so it is satisfied when the venv is actually built.
- **Effect:** Phase 2 produces a verified, pinned lockset; until then `requirements.txt` is a manifest, not a lockfile.

### D0.4 Python version split: local 3.13, CI 3.11
- **Decision:** Noted that the owner's machine runs Python 3.13.12 while CI (CLAUDE.md Section 8.1) pins Python 3.11.
- **Alternatives:** Force CI to 3.13; force local to 3.11.
- **Reason:** 3.11 is the lab's stated CI version. A model pickled under one minor version can usually load under another if library versions match, but this is a known compatibility risk to watch.
- **Effect:** In Phase 2 we verify the pinned versions install on both; in Phase 5/CI model-smoke we confirm the saved model loads under 3.11.

---

## Phase 1 - Dataset audit

### D1.1 Do not load or use the `Code` column
- **Decision:** The audit (and the whole pipeline) reads the CSV with `usecols` excluding `Code`.
- **Alternatives:** Load the full file; use source text as an NLP feature.
- **Reason:** `Code` is the raw Java source and is the bulk of the 618 MB file. It is an identifier-like artefact, not a quality metric, and loading it balloons memory for no modelling value. (A future "paste code" API path would re-derive metrics with a tool, not read this column.)
- **Effect:** Audit runs in seconds on ~107k x 21. `Code` is listed as an identifier-class column everywhere.

### D1.2 Primary target stays `has_smell` over mixed granularity
- **Decision:** Keep CLAUDE.md 2.1's primary target (`has_smell` = any of the 4 smells), despite the dataset mixing method-level and class-level rows.
- **Alternatives:** Split into separate method-level and class-level tasks; keep only class rows.
- **Reason:** The audit showed rows are sub-class entities (an identifier recurs up to 17x) and the method-family {Long method, Feature envy} never co-occurs with the class-family {God class, Data class} (0 rows). There is no granularity flag to split cleanly on. `has_smell` mirrors the paper's binary any-label target; the per-smell extension models naturally respect the method/class split.
- **Effect:** One primary binary task on all rows; per-smell models reported separately (extension). Granularity caveat documented in the report and REPORT.md.

### D1.3 Metrics are Halstead, not the paper's Understand set - replicate method only
- **Decision:** Treat the 14 numeric columns (Halstead family + cyclomatic + logical lines) as the feature set, and frame the project as replicating the paper's *method*, not its features.
- **Alternatives:** Try to recompute the paper's 25 Understand metrics from the source (needs SciTools Understand, licensed, and the full source tree).
- **Reason:** Only Cyclomatic Complexity (~SumCyclomatic) and Logical Lines (~CountLine) map to the paper's families; the rest are Halstead software-science metrics the paper did not use. Recomputing Understand metrics is out of scope and not reproducible without the tool.
- **Effect:** RQ1/RQ2/RQ3 run on these 14; comparisons to the paper are about *patterns* (families, a 3x rule, tree ensembles winning), never exact metric overlap. Strong collinearity (near-all |r|>0.9, several exactly 1.0) is flagged for RQ1.

### D1.4 Circularity verdict: PLAUSIBLE, confirm later; ablation planned regardless
- **Decision:** Record circularity as plausible-but-unconfirmed and proceed; verify the SmellyCode++ labeling rule before Phase 3 and run a metric-ablation in Phase 5 either way.
- **Alternatives:** Assume leakage and drop metrics now; ignore the risk.
- **Reason:** SmellyCode++ annotates "based on prior studies" (rule not stated in accessible pages; full descriptor behind a browser check). Measured single-metric separability is only moderate (best Cohen's d ~0.78), so no single Halstead metric is the label rule - but a multi-metric heuristic is still possible, and the dataset's own high reported F1s are consistent with learnable labels.
- **Effect:** CLAUDE.md Section 2 and the audit record the verdict; PAPER_REVIEW item 5 / Phase 5 ablation track this.

### D1.5 Proposed: no deduplication; rely on grouped-by-project CV (needs sign-off)
- **Decision (proposed, pending Rule 4 sign-off in Phase 2):** Do not drop the ~42.9k exact metric+label duplicates.
- **Alternatives:** Drop exact duplicates; dedup on identifier.
- **Reason:** Exact-duplicate rows are mostly tiny distinct methods (median 2 logical lines) - e.g. many identical one-line getters - so they are genuinely different entities, not errors; there is no method-name column to tell them apart. `StratifiedGroupKFold(groups=Project)` already keeps every entity of a project (hence every method of a class) on one side of each split, so the method/class repetition cannot leak across folds without any dropping.
- **Effect:** If approved, preprocessing keeps all rows and leans on grouped CV for the leakage guarantee (paper-review item 5). Revisit if grouped CV proves infeasible.

---

## Phase 2 - Preprocessing + CV

### D2.0 Owner sign-off on D1.2 and D1.5
- **Decision:** Owner approved both Phase-1 proposals ("do same for both"): keep `has_smell` (any smell) as the primary target over the mixed method/class rows, and do **not** deduplicate.
- **Effect:** `pipeline.yaml` sets `preprocess.deduplicate: false`; the primary task is the single binary `has_smell`, per-smell models are the extension. No further sign-off needed for these.

### D2.1 CV strategy: StratifiedGroupKFold grouped on Project
- **Decision:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=Project` (Section 4.2 option 2).
- **Alternatives:** TimeSeriesSplit (needs a date column - none exists); plain StratifiedKFold (weakest; leaks because methods/classes of one project would straddle folds).
- **Reason:** No date/commit column, but 89 projects. Grouping by project keeps every entity of a project (hence every method of a class) on one side of each split, which both gives realistic cross-project generalisation and neutralises the within-class method-repetition leakage - so no row dropping is needed (D1.5). Stratification preserves the 9.22% class ratio per fold.
- **Effect:** Verified in `cv.py` split sanity: per-fold test rate 9.13-9.46%, projects split ~70/19, **zero** train/test project overlap in all 5 folds. SMOTE/scaling go inside each training fold in RQ3 (Section 4.3), never here.

### D2.2 Pinned library versions (resolves D0.3)
- **Decision:** Pinned exact versions against a fresh `.venv` (Python 3.13.12): numpy 2.5.4, scikit-learn 1.9.1, xgboost 3.4.1, imbalanced-learn 0.14.2 (the four load-critical, identical in both requirement files), plus pandas 3.1.0, pyarrow 26.0.0, scipy 1.18.1, Boruta 0.4.3, scikit-posthocs 0.17.1, shap 0.54.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0.
- **Alternatives:** Older "known-stable" pins; leave unpinned.
- **Reason:** Verified they install and import on the owner's machine; Boruta 0.4.3 (unlike 0.3) is numpy-2 compatible, avoiding the removed `np.int` aliases. CI pins Python 3.11, which has cp311 wheels for all of these; the CI model-smoke job (Phase 9) confirms the model loads there.
- **Effect:** `ml/requirements.txt` fully pinned; `backend/requirements.txt` pins the four load-critical + pandas/joblib now, web/test deps in Phase 7. Rule 6 satisfied.

### D2.3 No categorical features -> no one-hot / SMOTENC
- **Decision:** Treat all 14 metric columns as numeric; no one-hot encoding, and plain SMOTE (not SMOTENC) in RQ3.
- **Alternatives:** The paper used categorical `extension`/`Ecosystem` and needed SMOTENC.
- **Reason:** SmellyCode++ has no categorical predictor; `Project` is an identifier/grouping key, not a feature (using it would leak project identity). So the PAPER_REVIEW item 3 concern (SMOTE interpolating encoded categories) does not arise for us - we note it as avoided.
- **Effect:** Simpler pipeline; `columns.categorical: []` in config.

---

## Phase 3 - RQ1 feature importance

### D3.1 Break Boruta ties with a RandomForest importance
- **Decision:** When Boruta confirms every feature (it does here - all 14 are "relevant"), order its top-10 by a separate RandomForest `feature_importances_` instead of by Boruta's tied ranking.
- **Alternatives:** Use Boruta's raw ranking (all confirmed features tie at rank 1, so the "top-10" degenerates to dataframe column order); use Boruta importance history.
- **Reason:** A column-order top-10 is an artefact, not a result, and it silently distorted the strict consensus (it dropped Cyclomatic Complexity purely by position). An RF importance tie-break makes Boruta's contribution meaningful.
- **Effect:** `boruta.rf_importance_tiebreak` recorded; `boruta_support` still shows the confirm/reject verdict (all confirmed). Strict consensus unchanged; the >=4/5 list now correctly includes Cyclomatic Complexity.

### D3.2 Compare to the paper at the family level, not by metric name
- **Decision:** Report RQ1 agreement with the paper as size/cyclomatic family overlap, not a name-level intersection with its 6 Understand core metrics.
- **Alternatives:** Intersect metric names (empty by construction - different toolkits).
- **Reason:** SmellyCode++ has Halstead metrics, the paper has SciTools Understand metrics; no names overlap. The scientific question (CLAUDE.md 1.2) is whether the same *families* matter.
- **Effect:** `rq1_summary.md` records the verdict: size dominates (whole strict consensus) and cyclomatic enters at 4/5 - the paper's pattern replicates despite a different target and toolkit.

### D3.3 Subsample Boruta and SHAP; keep the other methods on full data
- **Decision:** Run Boruta on a 25k stratified subsample and SHAP on 5k; LinReg/LogReg/XGB/ANOVA/permutation use all 107k rows (permutation per CV fold).
- **Alternatives:** Full data for all (Boruta fits many RF ensembles - minutes per run).
- **Reason:** RQ1 importance is descriptive (CLAUDE.md 4.3), so a stratified subsample for the expensive methods is acceptable and keeps the script re-runnable in under a minute. Sample sizes are recorded in the JSON.
- **Effect:** Reproducible with seed 42; sample sizes noted in `rq1_importance.json` and `rq1_summary.md`.

---

## Phase 4 - RQ2 statistics and thresholds

### D4.1 RQ2 metric set = the >=4/5 RQ1 consensus (8 metrics)
- **Decision:** Run RQ2 over Logical Lines, Total Operands, Vocabulary, Length, Calculated Length, Volume, Bugs, Cyclomatic Complexity.
- **Alternatives:** Strict 5/5 consensus only (5 metrics); all 14.
- **Reason:** The strict set omits Cyclomatic Complexity, which is the one metric that maps to the paper's cyclomatic core and is needed for the family comparison. The >=4/5 set keeps it while staying focused. Volume and Bugs are collinear (Bugs = Volume/3000) and give identical stats - kept for completeness, flagged as redundant.
- **Effect:** 8 metrics in all RQ2 outputs and the UI threshold rulers.

### D4.2 Validate thresholds leakage-safely (fit on train fold, score on test fold)
- **Decision:** For the threshold-as-classifier fix, compute warning/critical/Youden thresholds on each CV **train** fold and score "metric > threshold" on the held-out **test** fold; report mean +/- std.
- **Alternatives:** Compute the threshold on all data and score on all data (what a naive reading of the paper's rule would do).
- **Reason:** Scoring a threshold on the same data it was derived from is optimistic. Fitting on train and scoring on test gives an honest estimate and matches the Section 4.3 leakage rules.
- **Effect:** `as_classifier_cv` block per metric. Shows single-metric F1 ~0.3, i.e. thresholds are descriptive, not detectors.

### D4.3 Add Youden-J threshold; report that it trades precision for recall
- **Decision:** Add a ROC Youden-J threshold per metric alongside the paper's median/Q3 rule.
- **Alternatives:** Report only the paper's thresholds.
- **Reason:** CLAUDE.md 4.5 asks for an ROC-derived comparison. Youden maximises TPR-FPR, which at a 9% base rate picks a low threshold with high recall (0.63-0.89) but poor precision (0.15-0.20) and lower F1 than the median/Q3 rules - a useful, honest contrast documented in `rq2_summary.md`.
- **Effect:** Three threshold types per metric in `rq2_thresholds.json`; the UI can offer the trade-off.

### D4.4 "3x rule" reported as a pattern, with our actual ratios
- **Decision:** Report the smelly/clean median ratio per metric and flag approximate-3x (2-4x), rather than asserting the paper's exact 3x.
- **Alternatives:** Claim we reproduced "the 3x rule".
- **Reason:** Rule 2 (never invent numbers). Our ratios are 3.6x-11x, usually larger than the paper's 3x; the amplification *pattern* holds but the magnitude differs (different target).
- **Effect:** `median_ratio_smelly_over_clean` + `approx_3x_rule` per metric; narrative in `rq2_summary.md`.

---

## Phase 5 - RQ3 classifiers + final model

### D5.1 Cap training rows for SVM / MLP / KNN
- **Decision:** Subsample each CV train fold (stratified) before SMOTE for the kernel SVMs (12,000), MLP (40,000) and KNN (60,000); all other models use the full train fold.
- **Alternatives:** Full data for all (the paper's scale).
- **Reason:** libsvm SVC is ~O(n^2)-O(n^3); on ~86k train (x1.8 after SMOTE) it does not finish in a practical time. The caps keep the whole replication reproducible in minutes. Caps are recorded per model in `rq3_results.json`.
- **Effect:** SVM/MLP/KNN numbers are on capped folds (noted). Given they are mid-pack, the cap does not change the headline (XGBoost wins).

### D5.2 QDA reg_param escalation (forced deviation from Table 4)
- **Decision:** Try QDA with Table 4 `reg_param=0.0`; on a singular-covariance `LinAlgError`, escalate to 1e-3 / 1e-2 / 1e-1 and record the value used.
- **Alternatives:** Drop QDA; set reg_param>0 unconditionally.
- **Reason:** The Halstead metrics are exactly collinear (e.g. Bugs = Volume/3000), so class covariance is singular and `reg_param=0` raises rather than runs. A minimal regularisation yields a comparable QDA number. This is a concrete instance of PAPER_REVIEW item 4.
- **Effect:** QDA runs (AUC 0.765); the `deviation` field documents the reg_param used.

### D5.3 Deploy the replication XGBoost; document the stronger variants
- **Decision:** The deployable model is XGBoost (replication, SMOTE pipeline) refit on all data.
- **Alternatives:** Deploy the tuned XGBoost (AUC 0.847) or the scale_pos_weight variant.
- **Reason:** XGBoost is the clear best by ROC-AUC and Nemenyi rank (1.0). The replication model is a single fixed, reproducible configuration; the tuned/scale_pos_weight variants are marginally better (AUC +0.01) but vary per fold or need a separate refit. Deploying the clean model keeps the artifact reproducible; the alternatives are recorded in `rq3_improvements.json` and `rq3_summary.md`.
- **Effect:** `backend/model/model.joblib` (2.8 MB) + schema/thresholds/card; median-row predict ~6 ms.

### D5.4 Size-ablation as the circularity probe
- **Decision:** Implement the Section 2.2 ablation as "drop the 5 size-dominant strict-consensus metrics, re-run XGBoost", since the exact SmellyCode++ labeling rule is unconfirmed.
- **Alternatives:** Drop the specific rule-defining metrics (unknown); skip the ablation.
- **Reason:** We cannot drop the exact rule metrics without the labeling spec, but size is the dominant predictor, so removing it is the strongest available circularity stress test.
- **Effect:** AUC 0.835 -> 0.816 (small drop) recorded in `rq3_improvements.json`; supports "circularity plausible but not trivial".

---

## Phase 6 - Paper review + export

### D6.1 Export static result JSONs + a manifest to the frontend
- **Decision:** `export_artifacts.py` copies the 12 RQ JSONs plus `paper_reference.json`, `model_card.json`, `feature_schema.json`, `thresholds.json` into `frontend/src/assets/results/`, and writes a `manifest.json` (file list + model name/version + timestamp).
- **Alternatives:** Have the frontend fetch everything from the API; commit nothing static.
- **Reason:** CLAUDE.md 7.6 - the GitHub Pages site must render all research pages without the API. Static JSONs make that work and keep Rule 2 (UI shows only pipeline numbers). The manifest drives the empty-state check and the version banner.
- **Effect:** 16 files in `frontend/src/assets/results/`; the Predict page still needs the live API, everything else works offline.

### D6.2 PAPER_REVIEW grounded in our artifacts, arithmetic verified
- **Decision:** Write all 9 review items with each claim tied to a file in `ml/artifacts/`, and verify the quantitative ones by computation (item 1 SVM F1s: 0.323/0.321 not 0.52/0.47; item 9 count 338,442-33,294=305,148).
- **Reason:** Rule 2 + Section 9 ask for verified, evidenced criticism, not assertions.
- **Effect:** `docs/PAPER_REVIEW.md` complete with a status table.

---

## Phase 7 - Flask API

### D7.1 Do not implement /api/analyze-code
- **Decision:** Skip the optional pasted-code endpoint.
- **Alternatives:** Use `lizard` to extract metrics from pasted source, then predict.
- **Reason:** The model's 14 features include Halstead distinct/total operator and operand counts. `lizard` (and similar open-source tools) produce cyclomatic complexity, NLOC and token counts, but not the Halstead operator/operand decomposition, so the extracted feature vector could not be built faithfully. A "mark approximated: true" response would be misleading here.
- **Effect:** Endpoint omitted; documented in `backend/app.py` and `backend/requirements.txt`. Enter-metrics and CSV-upload cover the use case.

### D7.2 Per-prediction explanations via XGBoost SHAP contributions
- **Decision:** `top_contributors` uses the XGBoost booster's exact `pred_contribs` (per-row SHAP), with a feature-importance / coef proxy fallback for other model types.
- **Alternatives:** Ship the `shap` package in the backend (heavy); global feature importances only.
- **Reason:** `pred_contribs` gives exact per-row attributions with no extra dependency (xgboost is already pinned). Falls back gracefully if a non-XGBoost model is ever deployed.
- **Effect:** Signed per-feature impacts in every prediction response.

### D7.3 Risk tier combines probability and threshold flags
- **Decision:** Implement Section 6 exactly: critical if probability >= 0.7 OR >= 2 critical flags; low if probability < 0.4 and no critical flag; else elevated. Cutoffs read from `thresholds.json`, not hard-coded.
- **Reason:** The two signals catch different failure modes - a confident model score and an extreme-metric outlier the model may under-weight (seen in the live smoke test: an out-of-range row scored prob 0.18 but was flagged critical by 8 metrics).
- **Effect:** `risk_tier` logic lives in `predictor.py`; tunable via `thresholds.json`.

---

## Phase 8 - Angular UI

### D8.1 Angular 20 (not latest 22)
- **Decision:** Pin Angular 20; chart/icon deps echarts 5.6.0, ngx-echarts 20.0.2, lucide-angular 1.0.0.
- **Alternatives:** Angular 22 (latest).
- **Reason:** Angular CLI 22 requires Node >= 24.15 (local is 24.13) and drops Node 20, but CI is specced on Node 20 (8.1). Angular 20 supports Node 20.19+ and Node 24, so it builds on both; ngx-echarts 22 needs Angular 22, hence the 20.x line.
- **Effect:** Builds clean locally and will on CI Node 20. Recorded so a future Node bump can revisit.

### D8.2 Static JSON first; API only for Predict; committed build-info
- **Decision:** Research pages read `assets/results/*.json` via `ResultsService`; only Predict calls the API, with a static `feature_schema.json` fallback and the offline banner. A default `build-info.ts` is committed (CD overwrites it) rather than gitignored.
- **Reason:** CLAUDE.md 7.6 - the Pages site must work with no API. Committing build-info means CI can build the tree (a gitignored generated import would break the frontend CI job).
- **Effect:** All pages render offline (verified with Playwright: Overview, Dataset, Features, Thresholds, Models render; Predict shows the offline state and a schema-driven form). `.gitignore` no longer ignores build-info.ts.

### D8.3 Hash routing, assets mapping, and a page-level overflow guard
- **Decision:** `withHashLocation()`; map `src/assets` -> `assets` in angular.json; `html, body { overflow-x: hidden }` plus per-table `.scroll-x` wrappers.
- **Reason:** Hash routing avoids 404s on deep-link refresh on Pages (7.2). The overflow guard + table wrappers meet the "no horizontal page scroll at 360px" floor (7.8) while the nav tab bar and wide tables keep their own internal scroll (verified: pageScrollX = 0 on every page at 360px).
- **Effect:** Output at `dist/smell-risk-dashboard/browser` (for CD). Initial bundle 332 kB raw / 96 kB transfer; echarts lazy-loaded.

---

## Phase 9 - CI/CD

### D9.1 Repo-name-free deploy (base-href from the event)
- **Decision:** `deploy.yml` builds with `--base-href "/${{ github.event.repository.name }}/"` instead of a hardcoded repo name.
- **Alternatives:** Hardcode `/smell-risk-cicd/`.
- **Reason:** The workflow then works for any repository name the owner picks, with no edit. The Pages URL is `https://<owner>.github.io/<repo>/`, which the backend CORS already allows (`*.github.io`).
- **Effect:** No repo name needs to be committed anywhere.

### D9.2 CI Python 3.11 with a model-smoke cross-version guard
- **Decision:** CI runs Python 3.11 (lab spec) though the model was trained on 3.13; the `model-smoke` job loads `model.joblib` and predicts a row to prove it deserializes under 3.11.
- **Alternatives:** Set CI to 3.13 to match local exactly.
- **Reason:** Follow the lab's stated CI version. All pinned libraries publish cp311 wheels and the pickle is protocol 5 (readable on 3.11), so the load should succeed; the smoke job is the explicit check. If it ever fails, bump CI to 3.13 (one line).
- **Effect:** A red model-smoke job would be the early warning D0.4/D2.2 anticipated.

### D9.3 CD gated on CI success via workflow_run
- **Decision:** `deploy.yml` triggers on `workflow_run` of "CI" completed on main, runs only if `conclusion == 'success'`, checks out `head_sha`, and generates `build-info.ts` (short SHA, run number, UTC time) before building.
- **Reason:** Matches CLAUDE.md 8.2/8.3; the build tag on the live top bar changes after each push, which is how the demo proves CD worked.
- **Effect:** Pages deploys only from green builds; manual steps (create repo, enable Pages = GitHub Actions) are the owner's, documented in the README.

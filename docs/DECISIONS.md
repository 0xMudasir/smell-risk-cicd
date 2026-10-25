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

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

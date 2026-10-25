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

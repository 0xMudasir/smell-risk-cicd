# Smell Risk Dashboard

Replicating the **method** of Diouf et al. (2026), *Predicting Software Security
Bugs Using Machine Learning and Quality Metrics* (CMC 87(3), DOI
10.32604/cmc.2026.077139), on a multi-label **code-smell** dataset, and carrying
it through a full lab workflow: VS Code -> GitHub -> GitHub Actions CI -> CD to
GitHub Pages -> an Angular frontend talking to a Flask ML API.

> Different target. The paper predicts security bugs; we predict code smells. We
> replicate the methodology and compare **patterns** (do the same metric families
> matter? does a "3x rule" appear? do tree ensembles win?), not exact numbers.

Owner: Mudasir Ali (BSSE-F23-A, Air University).

## Status

**Live site:** https://0xmudasir.github.io/smell-risk-cicd/ (deployed by CD; the
top-bar build tag shows the commit SHA + Actions run number). The Predict page
needs a local API (`python backend/app.py`); other pages work from static JSON.

Built in stop-gated phases (see `CLAUDE.md` Section 5). **Phase 3 (RQ1) done.**

Dataset identified as **SmellyCode++** (107,554 rows, 89 Java projects, 4 smells,
9.22% smelly). Full profile in `ml/reports/dataset_audit.md`. CV =
`StratifiedGroupKFold` grouped on `Project` (zero train/test project overlap).
RQ1: size + cyclomatic families dominate, mirroring the paper (`ml/reports/rq1_summary.md`).
RQ2: large effect sizes, smelly medians 3.6-11x clean; single-metric thresholds
are weak classifiers (F1 ~0.3) (`ml/reports/rq2_summary.md`).
RQ3: XGBoost wins (ROC-AUC 0.835, Nemenyi rank 1.0), tree ensembles dominate -
same pattern as the paper (`ml/reports/rq3_summary.md`). Model saved to `backend/model/`.

| Phase | Goal | State |
|---|---|---|
| P0 | Scaffold + git | done |
| P1 | Dataset audit | done |
| P2 | Preprocess + CV | done |
| P3 | RQ1 feature importance | done |
| P4 | RQ2 statistics + thresholds | done |
| P5 | RQ3 models + final model | done |
| P6 | Paper review + export | done |
| P7 | Flask API | done |
| P8 | Angular UI | done |
| P9 | CI/CD + Pages | done (live) |
| P10 | Docs + demo | done |

## Layout

```
ml/        reproducible ML pipeline (src scripts, config, artifacts, reports)
backend/   Flask prediction API
frontend/  Angular dashboard (set up in Phase 8)
docs/      DECISIONS.md, PAPER_REVIEW.md, REPORT.md
data/      raw dataset (gitignored) + data/README.md
.github/   CI and CD workflows (set up in Phase 9)
```

## Setup

Prerequisites: **Python 3.12+** (dev used 3.13; the stack needs ≥3.12) and
**Node 20+**.

```powershell
# 1. Python env for the ML pipeline + API
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r ml/requirements.txt  # pipeline deps
pip install -r backend/requirements.txt  # API deps

# 2. Place the dataset (gitignored, ~618 MB) — see data/README.md
#    data/raw/multi-smell-dataset-v1_2.csv

# 3. Frontend deps
cd frontend && npm ci && cd ..
```

## Reproduce the pipeline (seed 42)

Run the phases in order; each writes to `ml/artifacts/` and `ml/reports/`:

```powershell
python -m ml.src.audit            # dataset audit
python -m ml.src.preprocess       # clean parquet
python -m ml.src.rq1_importance   # RQ1
python -m ml.src.rq2_statistics   # RQ2
python -m ml.src.rq3_models       # RQ3 (heaviest; several minutes)
python -m ml.src.train_final      # save backend/model/*
python -m ml.src.export_artifacts # copy JSON to frontend + verify model
```

## Run locally

```powershell
# API (terminal 1)
python backend/app.py             # http://localhost:5000
python -m pytest backend/tests -q # 10 tests

# Frontend (terminal 2)
cd frontend && npm start          # http://localhost:4200
```

The research pages work with no API; the Predict page calls the local API.

## 5-minute demo script

1. **Live site** — open https://0xmudasir.github.io/smell-risk-cicd/. Point out the
   top-bar **build tag** (commit SHA + run number) — that is CD's fingerprint.
2. **Overview** — 107,554 rows, 9.2% smelly; Paper vs ours table (XGBoost 0.835 AUC
   vs paper 0.91) — "same pattern, different target".
3. **Features** — `Logical Lines` is rank 1 across all methods; size + cyclomatic
   dominate, as in the paper.
4. **Thresholds** — the signature ruler; note the "3× rule" (smelly medians 3.6–11×)
   and that single-metric thresholds score only ~0.3 F1 (the paper never checked).
5. **Models** — XGBoost tops the leaderboard and the Nemenyi ranks; tree ensembles win.
6. **Predict** — start `python backend/app.py`; the API pill flips to online. Load
   the smelly example, Run prediction → high risk with threshold rulers lit.
7. **CD proof** — bump a visible label, commit, push; CI runs, CD deploys, the live
   build tag changes.

See `docs/REPORT.md` for the full write-up and `docs/PAPER_REVIEW.md` for the
nine paper issues we fixed.

## CI/CD and GitHub Pages (Phase 9)

Workflows are in `.github/workflows/`:

- **ci.yml** (push / PR to `main`): backend `compileall` + `pytest`, a model-smoke
  job that loads `model.joblib` and predicts a row, and an Angular production build.
- **deploy.yml**: on a successful CI run on `main`, generates `build-info.ts`,
  builds with `--base-href "/<repo>/"` (read from the event, so no edit needed),
  and deploys `dist/smell-risk-dashboard/browser` to GitHub Pages.

To make the site live (one-time, needs your GitHub account):

```bash
# 1. create a repo (e.g. smell-risk-cicd) and add it as the remote
git remote add origin https://github.com/<you>/smell-risk-cicd.git
git push -u origin main

# 2. GitHub → Settings → Pages → Source: "GitHub Actions"
```

CI runs on push; on green, CD deploys and the top-bar build tag shows the new SHA
and run number on the live site (this is how the demo proves CD worked). The live
Predict page calls a local Flask API (`python backend/app.py`); elsewhere it shows
the offline state.

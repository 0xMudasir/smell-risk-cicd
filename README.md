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
| P7 | Flask API | next |
| P7 | Flask API | |
| P8 | Angular UI | |
| P9 | CI/CD + Pages | |
| P10 | Docs + demo | |

## Layout

```
ml/        reproducible ML pipeline (src scripts, config, artifacts, reports)
backend/   Flask prediction API
frontend/  Angular dashboard (set up in Phase 8)
docs/      DECISIONS.md, PAPER_REVIEW.md, REPORT.md
data/      raw dataset (gitignored) + data/README.md
.github/   CI and CD workflows (set up in Phase 9)
```

## Setup (will be expanded in Phase 10)

```bash
# Python ML + API
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r ml/requirements.txt
```

Place the dataset at `data/raw/multi-smell-dataset-v1_2.csv` (see `data/README.md`).
Full setup, run, and 5-minute demo instructions land in Phase 10.

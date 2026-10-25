"""Phase 2 - Cross-validation split factory.

Returns the CV splitter chosen per CLAUDE.md Section 4.2:
  1. date/commit column  -> TimeSeriesSplit(n_splits=5)
  2. project column      -> StratifiedGroupKFold(n_splits=5, groups=project)
  3. neither             -> StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

The chosen strategy is recorded in docs/DECISIONS.md. Global seed = 42.
"""


def make_splitter(*args, **kwargs):
    raise NotImplementedError("Implemented in Phase 2.")

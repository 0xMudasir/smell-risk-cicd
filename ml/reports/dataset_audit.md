# Dataset audit - multi-smell-dataset v1.2

_Generated 2026-10-25T09:30:56.300293+00:00 (UTC). Seed 42. Source: `data\raw\multi-smell-dataset-v1_2.csv`._

> The `Code` column (raw Java source) is not loaded or used; it is the bulk of the 618 MB file and is not a feature.

## 1. Shape

- Rows: **107,554**
- Columns in file: **22** (loaded 21; `Code` skipped)

## 2. Columns

- Identifiers (never features): `File`, `Project`, `Class`
- Source text (never a feature): `Code`
- Numeric metrics (14): `Logical Lines`, `Distinct Operators`, `Distinct Operands`, `Total Operators`, `Total Operands`, `Vocabulary`, `Length`, `Calculated Length`, `Volume`, `Difficulty`, `Effort`, `Time Required`, `Bugs`, `Cyclomatic Complexity`
- Smell labels (4): `Long method`, `God class`, `Feature envy`, `Data class`
- Primary target: `has_smell` = 1 if any smell label is positive

### Dtypes and missing values

| Column | dtype | missing |
|---|---|---|
| `File` | str | 0 |
| `Project` | str | 0 |
| `Logical Lines` | int64 | 0 |
| `Distinct Operators` | int64 | 0 |
| `Distinct Operands` | int64 | 0 |
| `Total Operators` | int64 | 0 |
| `Total Operands` | int64 | 0 |
| `Vocabulary` | int64 | 0 |
| `Length` | int64 | 0 |
| `Calculated Length` | float64 | 0 |
| `Volume` | float64 | 0 |
| `Difficulty` | float64 | 0 |
| `Effort` | float64 | 0 |
| `Time Required` | float64 | 0 |
| `Bugs` | float64 | 0 |
| `Cyclomatic Complexity` | int64 | 0 |
| `Class` | str | 0 |
| `Long method` | int64 | 0 |
| `God class` | int64 | 0 |
| `Feature envy` | int64 | 0 |
| `Data class` | int64 | 0 |

Total missing cells: **0**.

## 3. Target and per-smell positive rates

- **has_smell**: 9,918 positive / 97,636 negative -> **9.22%** smelly.

| Smell | positives | rate | values seen |
|---|---|---|---|
| `Long method` | 1,566 | 1.46% | [0, 1] |
| `God class` | 4,333 | 4.03% | [0, 1] |
| `Feature envy` | 1,996 | 1.86% | [0, 1] |
| `Data class` | 3,284 | 3.05% | [0, 1] |

Smells per row (multi-label structure):

| # smells | rows |
|---|---|
| 0 | 97,636 |
| 1 | 8,657 |
| 2 | 1,261 |

## 4. Smell co-occurrence (rows positive for both)

| | Long method | God class | Feature envy | Data class |
|---|---|---|---|---|
| **Long method** | 1,566 | 0 | 1,153 | 0 |
| **God class** | 0 | 4,333 | 0 | 108 |
| **Feature envy** | 1,153 | 0 | 1,996 | 0 |
| **Data class** | 0 | 108 | 0 | 3,284 |

## 5. Duplicates

- Exact duplicate rows on metrics + labels: **42,883**
- Duplicate identifier rows (File+Project+Class): **25,068**
- Rows involved in an exact duplicate have a median of **2 logical lines** - i.e. they are mostly tiny entities (one-line getters/setters) whose metrics happen to coincide, not data errors. Blind exact-dedup would therefore delete genuinely distinct methods, so dedup is deferred to a Phase 2 decision (see Section 12).

## 5b. Granularity (mixed method + class)

- The identifier (File+Project+Class) recurs: **82,486** unique identifiers, **25,013** of them appear more than once, up to **17** rows for a single identifier. Rows are therefore **sub-class entities (methods), not whole files** - finer than the paper's file level.
- The smell labels split into two families that **never co-occur**: method-level (`Long method`, `Feature envy`) vs class-level (`God class`, `Data class`). Rows positive for both families: **0**.
- There is no method-name column or explicit method/class flag; for clean rows the granularity is not recoverable from the file. This is the main structural caveat of the dataset.

## 6. Projects

- Distinct projects: **89**. Files per project: min 1, median 735, max 4,979.

Top 25 projects by file count:

| Project | files | smelly | smelly rate |
|---|---|---|---|
| `jena` | 4,979 | 244 | 4.9% |
| `hive` | 4,851 | 444 | 9.2% |
| `hadoop` | 4,336 | 196 | 4.5% |
| `hbase` | 4,288 | 226 | 5.3% |
| `nifi` | 3,873 | 417 | 10.8% |
| `activemq` | 3,802 | 289 | 7.6% |
| `accumulo` | 3,786 | 346 | 9.1% |
| `beam` | 3,726 | 147 | 3.9% |
| `openjpa` | 3,604 | 433 | 12.0% |
| `cxf` | 3,454 | 405 | 11.7% |
| `cayenne` | 3,031 | 298 | 9.8% |
| `airavata` | 2,893 | 604 | 20.9% |
| `drill` | 2,419 | 89 | 3.7% |
| `calcite` | 2,416 | 102 | 4.2% |
| `cocoon` | 2,011 | 184 | 9.1% |
| `fop` | 2,008 | 160 | 8.0% |
| `kafka` | 1,964 | 102 | 5.2% |
| `wicket` | 1,929 | 158 | 8.2% |
| `jackrabbit` | 1,908 | 209 | 11.0% |
| `isis` | 1,868 | 170 | 9.1% |
| `tomcat` | 1,692 | 204 | 12.1% |
| `pulsar` | 1,677 | 90 | 5.4% |
| `tajo` | 1,576 | 135 | 8.6% |
| `storm` | 1,520 | 108 | 7.1% |
| `syncope` | 1,508 | 194 | 12.9% |

## 7. Date / version column

- **None.** No date, commit, or version column. CV strategy therefore falls to grouped CV on `Project` (Section 4.2 option 2).

## 8. Metric mapping to the paper's Understand metrics

| Metric | family | closest paper (Understand) metric | note |
|---|---|---|---|
| `Logical Lines` | size | CountLineCode / CountStmt (closest) | size metric; paper's size family |
| `Distinct Operators` | halstead | none | Halstead n1 |
| `Distinct Operands` | halstead | none | Halstead n2 |
| `Total Operators` | halstead | none | Halstead N1 |
| `Total Operands` | halstead | none | Halstead N2 |
| `Vocabulary` | halstead | none | n1+n2 (deterministic) |
| `Length` | halstead | none | N1+N2 (deterministic) |
| `Calculated Length` | halstead | none | n1*log2 n1 + n2*log2 n2 (deterministic) |
| `Volume` | halstead | none | Length*log2 Vocabulary (deterministic) |
| `Difficulty` | halstead | none | (n1/2)*(N2/n2) (deterministic) |
| `Effort` | halstead | none | Difficulty*Volume (deterministic) |
| `Time Required` | halstead | none | Effort/18 (deterministic) |
| `Bugs` | halstead | none | Halstead delivered-bugs estimate = Volume/3000 (deterministic); NOT a label |
| `Cyclomatic Complexity` | cyclomatic | SumCyclomatic (closest) | paper's core cyclomatic family |

> This dataset's metrics are **Halstead software-science metrics + cyclomatic complexity + a size metric**, not the 25 SciTools Understand metrics the paper used. Only cyclomatic complexity and Logical Lines map cleanly to the paper's families. We therefore replicate the **method**, not the feature set.

## 9. Correlated metric pairs (|r| > 0.9)

Many Halstead metrics are exact algebraic functions of one another, so they are almost perfectly correlated. This destabilises coefficient-based importance (PAPER_REVIEW item 4).

| metric A | metric B | r |
|---|---|---|
| `Total Operands` | `Length` | 1.0 |
| `Volume` | `Bugs` | 1.0 |
| `Effort` | `Time Required` | 1.0 |
| `Distinct Operands` | `Vocabulary` | 0.9996 |
| `Logical Lines` | `Length` | 0.9992 |
| `Logical Lines` | `Total Operands` | 0.999 |
| `Total Operands` | `Volume` | 0.999 |
| `Total Operands` | `Bugs` | 0.999 |
| `Length` | `Volume` | 0.999 |
| `Length` | `Bugs` | 0.999 |
| `Total Operators` | `Length` | 0.9983 |
| `Logical Lines` | `Total Operators` | 0.9982 |
| `Logical Lines` | `Volume` | 0.9982 |
| `Logical Lines` | `Bugs` | 0.9982 |
| `Logical Lines` | `Cyclomatic Complexity` | 0.998 |
| `Total Operators` | `Total Operands` | 0.9977 |
| `Total Operators` | `Volume` | 0.9969 |
| `Total Operators` | `Bugs` | 0.9969 |
| `Length` | `Cyclomatic Complexity` | 0.9969 |
| `Total Operands` | `Cyclomatic Complexity` | 0.9968 |
| `Volume` | `Cyclomatic Complexity` | 0.996 |
| `Bugs` | `Cyclomatic Complexity` | 0.996 |
| `Total Operators` | `Cyclomatic Complexity` | 0.9955 |
| `Distinct Operands` | `Calculated Length` | 0.987 |
| `Vocabulary` | `Calculated Length` | 0.9838 |
| `Volume` | `Effort` | 0.9817 |
| `Volume` | `Time Required` | 0.9817 |
| `Effort` | `Bugs` | 0.9817 |
| `Time Required` | `Bugs` | 0.9817 |
| `Total Operands` | `Effort` | 0.9753 |
| `Total Operands` | `Time Required` | 0.9753 |
| `Length` | `Effort` | 0.9748 |
| `Length` | `Time Required` | 0.9748 |
| `Logical Lines` | `Effort` | 0.9716 |
| `Logical Lines` | `Time Required` | 0.9716 |
| `Total Operators` | `Effort` | 0.9704 |
| `Total Operators` | `Time Required` | 0.9704 |
| `Effort` | `Cyclomatic Complexity` | 0.9682 |
| `Time Required` | `Cyclomatic Complexity` | 0.9682 |

## 10. Separability (standardized mean difference, smelly - clean)

Quick signal of how different the smelly class looks per metric (Cohen's d; |d| ~ 0.2 small, 0.5 medium, 0.8 large).

| Metric | clean median | smelly median | Cohen's d |
|---|---|---|---|
| `Vocabulary` | 20 | 71 | +0.779 |
| `Distinct Operators` | 2 | 5 | +0.776 |
| `Distinct Operands` | 18 | 66 | +0.766 |
| `Calculated Length` | 75.6 | 411 | +0.621 |
| `Difficulty` | 2.4 | 10 | +0.565 |
| `Total Operators` | 6 | 52 | +0.080 |
| `Length` | 40 | 314 | +0.068 |
| `Total Operands` | 33 | 252 | +0.066 |
| `Logical Lines` | 7 | 56 | +0.065 |
| `Volume` | 172 | 1.93e+03 | +0.042 |
| `Bugs` | 0.0572 | 0.644 | +0.042 |
| `Effort` | 368 | 1.89e+04 | -0.018 |
| `Time Required` | 20.4 | 1.05e+03 | -0.018 |
| `Cyclomatic Complexity` | 2 | 12 | +0.006 |

## 11. Label source and circularity verdict

- **Dataset identified:** this is **SmellyCode++**, a multi-label code-smell dataset (KFUPM; *Scientific Data*, July 2025). The sample count (107,554), the four smells, and the multi-label structure all match. Per its abstract, code was parsed into ASTs, features were extracted, and samples were **annotated 'based on prior studies'** (Fontana-style).

- **Label source: still to confirm exactly.** The accessible abstract/landing pages do not state the precise labeling rule (manual validation vs DECOR/PMD/iPlasma-style metric thresholds vs heuristics); the full data descriptor sits behind a browser check. The dataset's own reported per-smell F1 scores are high (Data class 95.9, God class 94.5, Feature envy 88.7, Long method 88.9), which is consistent with labels that are largely derivable from the features.

- **Circularity verdict (Section 2.2): PLAUSIBLE, not confirmed.** If smells were produced by metric-threshold rules, models on the same metrics would rediscover them. Evidence so far is mixed: single-metric separability (Section 10) is only **moderate** (best Cohen's d ~0.78 on Vocabulary / Distinct Operators), not the near-perfect split a one-metric threshold would give - so no single Halstead metric is the rule. But a multi-metric heuristic remains possible. **Action:** confirm the labeling rule from the full SmellyCode++ descriptor before Phase 3, and in Phase 5 run an ablation dropping the most discriminative metrics regardless, reporting both.

## 12. Open questions for Phase 2

1. **Target over mixed granularity.** CLAUDE.md 2.1 sets the primary target as `has_smell` (any smell). Because rows mix method-level and class-level entities (Section 5b), confirm we keep all rows for the primary target, and that the per-smell extension models naturally respect the method/class split.
2. **Deduplication.** Exact duplicates are mostly tiny distinct methods (Section 5), and grouped CV on `Project` already keeps all entities of a project (and hence a class) on one side of each split, so cross-split leakage from the method/class repetition is contained without dropping rows. Proposed: do **not** drop exact duplicates; rely on grouped CV. Needs sign-off (Rule 4).
3. **CV strategy.** No date column -> `StratifiedGroupKFold(n_splits=5, groups=Project)` (Section 4.2 option 2). To be finalised and recorded in Phase 2.
4. **Project count.** The file has 89 projects; the SmellyCode++ paper mentions ~103 Java projects. Minor discrepancy to note, not blocking.

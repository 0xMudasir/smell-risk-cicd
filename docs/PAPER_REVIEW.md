# Paper review

Shortcomings found in Diouf et al. (2026) and what we did instead. Each item must
be verified against the PDF before it is written up (CLAUDE.md Section 9). Filled
in Phase 6; the list of candidate items is below as a checklist.

- [ ] 1. Inconsistent F1 values in Table 15 (SVMs).
- [ ] 2. Mislabelled density-plot x-axes.
- [ ] 3. Categorical features inside SMOTE.
- [ ] 4. Coefficient importance without stated scaling, on correlated metrics.
- [ ] 5. Pre-/post-patch versions of the same file (near-duplicate leakage).
- [ ] 6. Thresholds never validated as classifiers.
- [ ] 7. Weak, untuned hyperparameters.
- [ ] 8. Accuracy-led reporting under imbalance, no PR-AUC.
- [ ] 9. Minor reporting slips (305,148 vs 305,149; correction only in threats).

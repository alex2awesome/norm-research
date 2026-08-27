---
source_url: https://medium.com/@ptannor/checklist-for-data-science-research-review-8a817b50697b
title: Checklist for Data Science Research Review
source_type: medium_article
author: Philip Tannor
venue: Medium
fetched: 2026-05-09
---

# Checklist for Data Science Research Review (Tannor / Medium)

A practitioner's deep checklist for self- and peer-review of data-science work before deployment.

## 9 Categories

### 1. Dataset assumptions
- Sampling method validity for production.
- Feature availability and consistency.
- Dataset shifts over time.
- Sampling-procedure limitations.
- Low-frequency phenomena coverage.

### 2. Preprocessing
- Consistent procedures across train / test.
- Per-algorithm normalization.
- Parameters fit on training data only.
- Anomaly sensitivity in scaling methods.
- Delta-like distributions needing non-linear scaling.

### 3. Leakage and bias
- Index leakage.
- Feature-importance anomalies as leakage signal.
- Generated-data consistency across classes.
- Sample-collection consistency.
- Over-experimentation creating "leaderboard likelihood."
- Train-test segment differences.
- Reasonableness checks on performance.

### 4. Causality
- Label-timing alignment.
- Future information in training data.
- Forward-looking features (e.g., bi-directional moving averages).
- Rolling-model consistency during training.

### 5. Loss / metric
- Loss function correctness.
- Loss-metric alignment and monotonicity.
- Business-metric correspondence.
- Ensemble optimization validity.
- NN training behavior smoothness.
- Custom-loss anomaly sensitivity.

### 6. Overfit detection
- Random-seed impact.
- Test-set sampling consistency.
- Multiple-fold evaluation.
- Hyperparameter tuning on train only.
- Complexity-parameter graphs.
- Parameter-reduction feasibility.

### 7. Runtime
- Feature-reduction options.
- Feature computation time.
- Hardware assumptions.
- Ensemble-improvement quantification.

### 8. "Stupid bugs"
- Index-column deletion.
- Column-name preservation.
- Label-feature index matching.
- Merge / join effects.
- Dictionary version control.
- Correct model-weight files.

### 9. Trivial questions
- Untested libraries due to technical issues.
- Redundant features.
- Non-ML intelligent benchmarks comparison.

## Key Insight

"There are a million things you should be checking." Hire quality people and invest in training rather than seek shortcuts.

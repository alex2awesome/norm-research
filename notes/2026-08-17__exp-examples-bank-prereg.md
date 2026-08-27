# EXP-EXAMPLES-BANK-1 — prereg: examples-vs-definition at bank scale, task x category

Status: PREREGISTERED 2026-08-17 before any confirmatory run. User request (session
paper-2-writing): scale the examples analysis beyond the 38 slate metrics; separate TASK
dependency from CATEGORY dependency (existing winners table is humor-dominated because flip
machinery only ran on the zxa slates).

## Design
Per metric: the flip_functional protocol (v2 code path, osl_staged_20260808/flip_functional_v2.py
lineage) with three changes, all declared:
1. REFERENCE at bank scale = consensus of llama70b + qwen25-72b labelings under the metric's own
   bank rubric (crowd panels mbar2_<task>_<exec>.npz), ties/disagreements excluded (label -1).
   (The slate version used the 4-model dossier-arm consensus; dossier panels do not exist at
   bank scale. Weaker 2-voter reference disclosed; metrics with <60 decided items are skipped.)
2. SELECTION EXECUTOR = llama8b (never an evaluator) — removes the judge-self-selection bias
   documented in 4.2E-b. Same three-way stable-hash split (train-A select at theta=.01 /
   train-B confirm / holdout touched once), exemplar items masked, 500-char exemplar truncation,
   MAX_SET 12, selection-null control on every 3rd metric.
3. EVALUATION = holdout balanced acc at llama70b AND qwen25-72b (reported separately; no
   best-cell max in the headline — the per-judge table is primary, max-cell relegated to a
   labeled exhibit).

## Sample
Stratified: for each (task x 6-category) cell, up to 6 metrics by seeded random (seed 0) from
the fitted 1,270 (taxonomy labels osl_metric_types_20260728.json, M6 merge), 8 tasks. Cells
with fewer metrics take all. Expected ~180-220 metrics.

## Preregistered readouts (numbers first, no verdicts inside the runner)
- Primary: delta = functional - definition per (metric x judge); decomposition table
  mean delta by task (within-category) and by category (within-task); a metric-level OLS of
  delta on task dummies + category dummies (both-in model), reported with author-level (metric)
  bootstrap CIs. The task-vs-category question = which factor's dummies carry the variance.
- Secondaries: name-arm baseline; null-control holdouts (must sit ~name level, else leak);
  per-cell n.
- Exclusions: gemma2 anywhere; news probe-universe joins (standing landmine) — news metrics
  ARE included but flagged, their reference built only from the same mbar2 panel (no fresh
  joins); metrics with degenerate reference (<60 decided) skipped and counted.

## Gates
- L1 leak: pooled null-control holdout within .03 of the name arm.
- L2 reference sanity: per task, median reference agreement rate between the 2 voters >= .70
  on decided items; tasks failing L2 are reported but excluded from the headline decomposition.
No optional stopping; the stratified sample is fixed before the first selection call.

## Not yet run
No selection or evaluation calls made as of freeze. Runs on sk3 (selection llama8b single GPU
stacked; evaluation llama70b/qwen72b BF16 TP=2 per the established recipe).

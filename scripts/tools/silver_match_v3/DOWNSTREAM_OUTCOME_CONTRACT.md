# Downstream outcome contract after silver matching freezes

The silver matcher is an evaluate-only measurement instrument.  Outcome labels
may neither alter the metric bank nor train/select the matcher.  Global joins
begin after all 1,732,515 norm decisions freeze. A completed task may be
released earlier only through `freeze_task_analysis_release.py`, after its
complete exact-output audit and independent blind MATCH/abstention audit. From
that release onward the task matcher is immutable, and its MI/outcome results
may not tune or otherwise inform any remaining task's retriever, prompt,
verifier, threshold, or bank. This staggered task firewall permits prompt
analysis without outcome leakage.

Every staggered release also carries a hash-bound union of all canonical norm
UIDs used for retriever/LoRA/GEPA/verifier train, dev, or test. Those rows retain
final decisions for complete-dataset accounting but are excluded from MI and
outcome estimation. Precision and false-abstention claims for staggered
analysis use a separate uniform blind sample drawn only from the never-labeled
remainder; the all-row blind audit remains a distinct dataset-quality report.

## Required three-leg design

Every task is reported separately on:

1. **Expert-verdict:** a qualified authority's stated quality judgment.
2. **Expert-revealed:** a gatekeeper's action, such as acceptance or curation.
3. **Community-revealed:** crowd behavior, such as votes, use, sales, or
   citations.

A task with an unavailable leg is reported as missing that leg; another leg is
not silently substituted.  Pooled conclusions require a label-type interaction
and heterogeneity analysis, not one outcome collapsed across tasks.

## Already pinned sources and caveats

| Task | Artifact | Current role/caveat |
|---|---|---|
| peer review | `datasets/peer-review/splits/train.csv.gz`, fixed binary `judgement` | Canonical labeled panel; confirm stated-score versus accept-action provenance before assigning its leg |
| peer review | `datasets/peer-review/s2_citations_2024_25.jsonl` | Community leg must use accepted-only as primary because citation availability is outcome-correlated (98.8% accepted versus 56.2% rejected coverage); never naive full-set complete-case |
| creative writing | `datasets/creative-writing/litbench-to-train.csv.gz`, fixed `judgement` | Community/upvote-derived leg; do not re-threshold the stored label |
| creative writing | Wigleaf and Royal Road expert/market panels referenced in project memory | Locate and hash the built artifacts before use; preserve curation and market outcomes as separate legs |
| press releases | `datasets/press-releases/press_release_modeling_dataset.csv.gz`, fixed `judgement` | News-pickup/gatekeeper leg; retain venue, company, domain, exposure, and length controls |
| all tasks | `outputs/v2_db/cells_v1/task=<task>/judge=<judge>/data.{parquet,csv.gz}` | Canonical judge-cell source of truth; never reconstruct counts from raw response directories |

Code review, Humor, Math, Notice-and-Comment, and Legal still require an
append-only outcome-source inventory assigning each available label to one of
the three legs.  That inventory must record path, row count, label definition,
coverage/missingness, grouping key, and SHA-256 before analysis.

The first verified candidate inventory now lives in
`OUTCOME_SOURCE_INVENTORY.md`.  Rows marked `MISSING` remain genuine missing
legs; candidate status does not authorize a join before silver assignments
freeze or waive the listed unit, provenance, and confound audits.

## Reporting and confound rules

- Use canonical `judgement` columns as stored.  Do not re-threshold raw votes or
  pickup counts to improve an effect.
- Report direct quality judgment, gatekeeping, and popularity separately.  A
  popularity null is not a task-level validity null.
- Reuse the existing length, venue, exposure, year, source/community, and
  publisher/company controls where applicable; grouped splits must block the
  relevant source identity.
- Report missingness and range restriction by outcome and class.  Do not use
  coverage itself as an accidental predictor.
- For every task × label leg, report the exact joined denominator, match and
  typed-abstention inclusion policy, metric coverage, univariate effects,
  controlled effects, uncertainty, and sign.  Then report equal-task and
  random-effects summaries with heterogeneity and label-type interactions.
- Preserve both exact-ID and family-level sensitivity results.  Family rollups
  use only the separately versioned relation graph and never rewrite the frozen
  exact assignments.

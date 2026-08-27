# Outcome-source inventory for broad silver-match claims

This is an append-only inventory of candidate downstream outcomes.  It was
checked against the current workspace and the relevant Claude-Code memories on
2026-07-12.  It does **not** authorize outcome leakage into retrieval,
adjudication, verification, bank construction, or threshold selection.
Global outcome joins begin only after all exact silver decisions are frozen.
Task-local joins may begin after a hash-bound task analysis release freezes
that task's complete decisions and blind risk audits; released results are
strictly prohibited from influencing unfinished matchers.

The three legs are distinct:

1. `expert-verdict`: a qualified evaluator's stated judgment;
2. `expert-revealed`: a gatekeeper's action;
3. `community-revealed`: crowd attention, use, or uptake.

A single label may not silently fill two legs.  `MISSING` means the task-level
claim for that leg remains unavailable; another task or label type cannot
substitute for it.

## Verified candidates

| Task | Leg | Frozen candidate | Rows | Join / label | Status and primary caveat |
|---|---|---|---:|---|---|
| peer-review | expert-verdict | `datasets/peer-review/unified_reviews.csv.gz` (`69799a29…`) | 299,961 reviews | `paper_id`; `recommendation_numeric` / normalized recommendation | READY-CANDIDATE; reviewer scores must remain separate from acceptance |
| peer-review | expert-revealed | `datasets/peer-review/splits/train.csv.gz` (`6d7811ae…`) plus its fixed eval/test siblings | 56,153 train papers | `paper_id`; stored binary `judgement` | READY-CANDIDATE; hash and audit all three fixed splits before the join |
| peer-review | community-revealed | `datasets/peer-review/openalex_citations/openalex_citations_v2.csv.gz` (`1ba7e34e…`) | 27,233 papers | paper ID/title bridge; `cited_by_count` / stored percentile | READY-CANDIDATE; accepted-only primary because citation availability is outcome-correlated |
| creative-writing | expert-verdict | — | — | — | MISSING; no direct qualified rating panel is currently pinned |
| creative-writing | expert-revealed | sk3 `datasets/creative-writing/wigleaf/built/{train,eval,test}.csv.gz` (`c01f6e64…`, `b48feb2c…`, `c9e954c4…`) | 1,568 stories | story identity; editor Top-50/longlist `judgement` | READY-CANDIDATE; preserve magazine/year grouping and fetch-source audit |
| creative-writing | expert-revealed sensitivity | sk3 `datasets/creative-writing/royalroad_stubs/built/royalroad_v2_fiction_topicstrat.csv.gz` (`de9cceb7…`) | 1,274 fictions | `fiction_id`; frozen market/gatekeeper `judgement` | READY-CANDIDATE; report separately from literary curation and retain Wayback-era controls |
| creative-writing | community-revealed | `datasets/creative-writing/litbench-to-train.csv.gz` (`3c0e9af4…`) | 87,654 texts | row/text bridge; stored `judgement`, raw `upvotes` diagnostic only | READY-CANDIDATE; do not re-threshold upvotes |
| press-releases | expert-verdict | — | — | — | MISSING; pickup is an action, not a stated quality verdict |
| press-releases | expert-revealed | `datasets/press-releases/press_release_deconfounded.parquet` (`91e0299a…`) | 72,315 releases | `id`; frozen `judgement` | READY-CANDIDATE; company-grouped, topic/exposure controls required; do not silently change pickup threshold |
| press-releases | community-revealed | — | — | — | MISSING; media pickup is gatekeeping, not crowd uptake |
| code-review | expert-verdict | — | — | — | MISSING as an independent outcome; review comments are the silver source, not a hidden verdict label |
| code-review | expert-revealed | `datasets/code-review/pr_merge_status.csv.gz` (`148a96f4…`) | 152,300 PRs | owner/repo/PR; `pr_merged`, timestamps | READY-CANDIDATE; group by repository and distinguish merge from speed-to-merge |
| code-review | community-revealed | — | — | — | MISSING; comment count and latency are process measures, not yet validated crowd-quality outcomes |
| code-review | objective-verifier (non-triad) | `datasets/code-review/pr_test_execution/outputs/consolidated_verdicts_ALL_final.parquet` (`217d41aa…`) | 85,146 PR executions | repo/PR; test `verdict`, `veracity_tier` | READY-CANDIDATE sensitivity only; never relabel as an expert/community leg |
| math-stackexchange | expert-verdict | — | — | — | MISSING; formal expert grades are not yet pinned to the canonical math units |
| math-stackexchange | expert-revealed | `datasets/math/mathlib/friction_dataset.csv.gz` (`27210fd6…`) | 19,356 PRs | PR number; merge `y`, review process fields | READY-CANDIDATE for the Mathlib corpus only; repository gatekeeping is not interchangeable with Math.SE answer acceptance |
| math-stackexchange | community-revealed | `datasets/math/stackexchange/math_se_v3_3_propensity_balanced.csv.gz` (`451a0492…`) | 99,722 answers | question/answer IDs; `score`, stored `judgement`, `accepted` | READY-CANDIDATE; use the frozen propensity/position design and report score separately from asker acceptance |
| humor | expert-verdict | — | — | — | MISSING; McSweeney's responses are qualitative anchors and lack submitted text |
| humor | expert-revealed | sk3 `datasets/humor/contest_corpus/contest_corpus_clean.jsonl` (`82ec69da…`) | 513 entries with text | contest entry; expert-curation label | READY-CANDIDATE after a fresh unit/label audit; small and source-heterogeneous |
| humor | community-revealed | `datasets/humor/reddit_humor_modeling_dedup.csv.gz` (`34ef7c68…`) | 383,786 jokes | canonical text row; stored stratified `judgement` | READY-CANDIDATE; MinHash/topic/length design must be retained |
| notice-and-comment | expert-verdict | `datasets/notice-and-comment/v4/nc_v42_balanced.jsonl.gz` (`772d6b0b…`) | 109,761 comments | `doc_id`; agency `response_type` and response text | READY-CANDIDATE; response labels are model-extracted and need an independent audit |
| notice-and-comment | expert-revealed | same frozen v4 artifact | 109,761 comments | `doc_id`; `outcome_collapsed` / rule-change outcome | READY-CANDIDATE; do not count the same response field as both legs; preserve docket/agency grouping |
| notice-and-comment | community-revealed | — | — | — | MISSING; organization identity or duplicate-comment count is not a validated uptake label |
| legal-outcome-prediction | expert-verdict | sk3 `datasets/legal-outcome-prediction/nlrb/assembled_pairs.v2.jsonl.gz` (`fc3c3f55…`) and CAVC/BVA lower decisions | 3,242 NLRB records | lower adjudicator recommendation / findings | READY-CANDIDATE; freeze a clean lower-decision field rather than reusing the appellate outcome |
| legal-outcome-prediction | expert-revealed | sk3 `datasets/legal-outcome-prediction/cavc/data/modeling_pool_cavc_ge006.jsonl.gz` (`9ee59e5f…`) | 7,602 appeals | docket/case; affirm versus disturbed `y` | READY-CANDIDATE; margin≥.06 linkage floor and docket-grouped splits are mandatory |
| legal-outcome-prediction | community-revealed | sk3 `datasets/legal-outcome-prediction/citation_percentile/built/labels_binary_v2.csv.gz` (`9afcd41e…`) | 17,994 opinions | opinion/docket; citation count/within-cell percentile | READY-CANDIDATE; citation digitization/source restrictions and court×year cells must be retained |

## Explicit exclusions and safeguards

- New Yorker caption ratings remain excluded for Humor: they are paid crowd
  annotations and the task README explicitly rules them out.
- The 30 McSweeney's rejection messages (`a869cc04…`) remain qualitative
  norm anchors only because every submitted piece is missing.
- `judgement` columns are used as stored.  Raw votes, citations, pickup counts,
  or timing variables are diagnostics unless a separately frozen outcome leg
  explicitly declares them.
- Every downstream table must record exact joined denominator, unit/group key,
  missingness, outcome definition, inclusion policy for typed abstentions,
  source hash, and label-type interaction.
- Cross-task conclusions require equal-task and random-effects summaries plus
  task and corpus ranges.  A strong community or gatekeeping result cannot
  repair a missing expert-verdict leg.

This inventory extends `DOWNSTREAM_OUTCOME_CONTRACT.md`; it does not supersede
the canonical judge-cell source in `outputs/v2_db/cells_v1/`.

## Full source digests

The shortened hashes in the table are display labels only.  These full
SHA-256 digests are the provenance values to validate before a downstream
join:

| Artifact | SHA-256 |
|---|---|
| `datasets/peer-review/unified_reviews.csv.gz` | `69799a295fb05e5e7ae8a387ee32c1c8282524a77f34bb8c191660ad3f0bfba9` |
| `datasets/peer-review/splits/train.csv.gz` | `6d7811ae56dde1763a619aababeca00afdbf596da0eb2a107e3ac43f7f590c76` |
| `datasets/peer-review/openalex_citations/openalex_citations_v2.csv.gz` | `1ba7e34eaa1f5f519f061b533ae4d333557e6b386e2f7718d9ae89c8971e0bbd` |
| sk3 `datasets/creative-writing/wigleaf/built/train.csv.gz` | `c01f6e648c6ec24da309ebbc354ab342a9989cf6bd6ae73558ebc3ea7486ba7a` |
| sk3 `datasets/creative-writing/wigleaf/built/eval.csv.gz` | `b48feb2c7e042861b5f073ff4a00b65a908831e75b58a3a2f465afc3724e2e86` |
| sk3 `datasets/creative-writing/wigleaf/built/test.csv.gz` | `c9e954c4560965ccababc70392748c3eb923589de030f987ca2676d0cc86c9e9` |
| sk3 `datasets/creative-writing/royalroad_stubs/built/royalroad_v2_fiction_topicstrat.csv.gz` | `de9cceb734ef877d473f79deb6ed26eab9ee9152dc506b5b4319611c0e2eef56` |
| `datasets/creative-writing/litbench-to-train.csv.gz` | `3c0e9af49dadae13f5385b65e0fbeb117011ca05a5d65f7b86c4710a431240c9` |
| `datasets/press-releases/press_release_deconfounded.parquet` | `91e0299a0d43591045022b513f8241c271ae217e68a574d04d40e800685c3436` |
| `datasets/code-review/pr_merge_status.csv.gz` | `148a96f429e17a76d8546ec21b29123723e51d0d941b8572a7f6ece9dcd6f186` |
| `datasets/code-review/pr_test_execution/outputs/consolidated_verdicts_ALL_final.parquet` | `217d41aa2d84535870fcb27abcc492833221ce86fe1743eaccae99362119b6c1` |
| `datasets/math/mathlib/friction_dataset.csv.gz` | `27210fd6355caf24303d34db310233dcfd0b36aa0b0d4d5a3380b9bb76cb56ff` |
| `datasets/math/stackexchange/math_se_v3_3_propensity_balanced.csv.gz` | `451a049238b623f5ab531ddc8c09dbe791aa721078435aa09499ded4194a5550` |
| sk3 `datasets/humor/contest_corpus/contest_corpus_clean.jsonl` | `82ec69daf52d7b8b0b1f16976e76b53e5e89207bd643e9a92ebd573a58edb6e0` |
| `datasets/humor/reddit_humor_modeling_dedup.csv.gz` | `34ef7c68df44cf07e5d57669f8e7332e51a050b93f28985cfe6c1f162de2acd6` |
| `datasets/humor/mcsweeneys_rejections/pairs.jsonl` | `a869cc0459263c1d3d10ee09befc3b6a65ab4e958d219e008ef9102af9948b33` |
| `datasets/notice-and-comment/v4/nc_v42_balanced.jsonl.gz` | `772d6b0bfe6bc45cf6bdedff23451be047ea6662e35d5f692747485d60f9d50c` |
| sk3 `datasets/legal-outcome-prediction/nlrb/assembled_pairs.v2.jsonl.gz` | `fc3c3f55f5c93b41df59e515197961d0c7459791f208d148718aa2b085564d31` |
| sk3 `datasets/legal-outcome-prediction/cavc/data/modeling_pool_cavc_ge006.jsonl.gz` | `9ee59e5f220b30367a64c7c1484eb7f6956c9d312eb2010a14cdc576290cde22` |
| sk3 `datasets/legal-outcome-prediction/citation_percentile/built/labels_binary_v2.csv.gz` | `9afcd41e8f545009744304be218e8fa2c61366e97986174d2a54698e11d5556a` |

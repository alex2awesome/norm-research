# N&C V4 — all-agency long-comment corpus with three outcome-label axes (2026-07-09)

Built for two uses: (a) outcome-label (y-variable) exploration across
`response_engagement_type` / `type_of_response` / `rule_change_outcome`, and (b) per-agency
deep dives. Complements — does not replace — the earlier corpora: v1 claims (108-char parsed
claims), V2 (agency-paraphrase pairs, ~220 chars), v3.1 (full submissions, 16 agencies,
panel-science probe pool).

## Sources & join (verified)

- **Comments (full text)**: `regulations-demo/data/bulk_downloads/all_comments_with_matches.csv.gz`
  (6.4 GB gz; 11,600,599 comments, ~50 agencies incl. EPA/SEC/NOAA/FCC that the 16-agency
  `bulk_downloads_package` lacks). Carries `matched_response_keys` from the upstream
  cross-encoder matching pipeline.
- **Labels**: per-agency `responses_combined.csv.gz` (342 files, 40 agencies; 205,622 response
  rows = 140,956 `llama_v2` + 64,666 `gpt5` extractions). **This — not
  `comment_responses_V2.jsonl` — is the matcher's actual response universe**: joining against
  V2 alone hit ~24% of matched keys; against `responses_combined`, **93.7%** (verified on an
  800K-row sample).
- **Join key**: `matched_response_keys` = `AGENCY|docket_id|content_of_comment` (content
  truncated at 200 chars). We join on `(agency, docket, whitespace-normalized content[:120])`.

## Build (`sk3:outputs/nc_v4/build_v4.py`)

1. Label index from all `responses_combined` files (108,032 unique keys).
2. Stream all comments; keep `len >= 700` chars (matched rows kept down to 200 — labels are
   scarce); attach ALL matching label triples (a comment can match several responses); write
   full + per-agency files.
3. Balanced view: reservoir sample per (agency × has-labels) cell, cap 4,000/cell.
4. Clean pass: strip `<<COMMENT n>>`/`<<PAGE n>>` assembly markers + leaked booleans; add
   `outcome_collapsed` ∈ {MADE, CONSIDERED, NONE, OTHER}.

**Splits**: `split` column = docket-level stable hash (70/30 train/test) — group split by
docket to avoid rule-level leakage (the flaw flagged in the v1 per-row splits).

## Files

| file | where | contents |
|---|---|---|
| `nc_v4_balanced_clean.jsonl.gz` | here (493 MB) | 163,363 rows: per-(agency × labeled) reservoir, cleaned, `outcome_collapsed` added |
| `sample_200.jsonl` | here | first 200 rows for quick inspection |
| `v4_stats.json` | here | per-agency counts + label distributions |
| `nc_v4_full.jsonl.gz` | sk3 `outputs/nc_v4/` (5.7 GB) | all 7,921,326 long comments |
| `by_agency/<AG>.jsonl.gz` | sk3 `outputs/nc_v4/` | same rows, one file per agency (38) |

## Row schema

```json
{"doc_id": "...", "agency": "EPA", "docket": "EPA-...", "org": "...",
 "is_matched": true, "n_matched": 2,
 "labels": [{"engagement": "substantive_response", "response_type": "disagree",
             "outcome": "change_considered_but_not_made", "outcome_collapsed": "CONSIDERED",
             "response": "<agency response text, 400c>", "label_source": "gpt5|llama_v2"}],
 "split": "train", "text": "<full comment text, cleaned>"}
```

## Headline stats

- 7,921,326 long comments; **35,533 with full three-axis labels** across 38 agencies.
- Top labeled agencies: EPA 9,207 · CMS 5,381 · FWS 3,979 · FAA 3,616 · FDA 2,790 ·
  AMS 2,424 · USCIS 2,097 · APHIS 1,536.
- Collapsed outcomes (balanced file): **MADE 13,869 / CONSIDERED 10,513 / NONE 16,949 /
  OTHER 514** (98.8% covered by the collapse taxonomy).

## Caveats

1. `is_matched` base rate is ~0.6% of all comments — matching recall upstream is unknown;
   treat unmatched ≠ ignored-by-agency (it may be matcher miss). For "ignored" analyses,
   prefer engagement labels within the matched subset.
2. Labels are LLM-extracted (gpt5/llama_v2) from RTC sections — silver, not gold;
   `label_source` kept for source-effect checks.
3. Raw `rule_change_outcome` is free text with a 400+-variant tail; use `outcome_collapsed`
   or your own collapse.
4. Comment text is the regulations.gov `comment_text` field; attachments-only submissions
   ("see attached") can still be thin.
5. A comment matching multiple responses carries multiple label triples — decide an
   aggregation rule per analysis (e.g., any-MADE, or majority engagement).

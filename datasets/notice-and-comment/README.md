# Notice-and-Comment (N&C)

Federal rulemaking. Public submits comments on a proposed rule; the issuing agency publishes a final rule that may (or may not) respond substantively to each comment. We predict whether an individual comment (or comment-claim) receives a substantive agency response.

**→ For the current VAT (Verifiability/Articulability/Taste) modeling campaign on the v4.2 corpus, see [§9](#9-vat-campaign-2026-07-15-v42-corpus) below.** Sections 1–8 describe the original v1 claims corpus and are historical; §9 is where the live numbers are.

## 1. Task

Binary label `judgement`:
- `1` — the comment cluster was "matched yes" in the regulations-demo matching pipeline (agency engaged with the substance of this comment in the response-to-comments section of the final rule).
- `0` — "matched no" (comment received no substantive engagement: ignored, lumped into a boilerplate acknowledgement, or addressed only procedurally).

Unit of supervision is the **claim** (one parsed argument from a public submission), not the whole comment letter. Multiple claims from the same submission inherit the submission-level label and appear as separate rows.

Source label definition: `matched` column of `all_cluster_labels.csv.gz` from the upstream `rfi-research/regulations-demo` pipeline (see `build_agency_splits.py:33` and `:56`).

## 2. Sources

- **Comments + claims**: `regulations-demo/data/to_upload/<agency>/<agency>_YYYY_YYYY/public_submission_all_text__claims.csv.gz` — public submissions scraped from regulations.gov, parsed into discrete claims by an upstream LLM step.
- **Labels (matched yes/no)**: `regulations-demo/data/bulk_downloads/match_results_summary/all_cluster_labels.csv.gz` — produced by the upstream matching pipeline between comment clusters and response-to-comments sections of final rules.
- **Final-rule HTML** (for Response-to-Comments / RTC extraction): `regulations-demo/data/bulk_downloads/` — used by `rtc_extracted/extract_rtc.py` to pull the agency's verbatim response text.
- **V2 paraphrased pairs**: `comment_responses_V2.jsonl` (254 MB, 113,485 rule docs, ~112K (comment, response) pairs) — LLM-summarized comment+response pairs produced upstream in regulations-demo; mirrored at `v2_existing/comment_responses_V2.jsonl`.

Both label source and comment text originate ultimately from regulations.gov.

## 3. Collection scripts

In this directory:

- `build_agency_splits.py` — joins claims with cluster labels per agency, explodes claims to one-row-per-claim, dedups exact text, writes 70/15/15 random split to `agencies/<agency>/{train,eval,test}.csv.gz`. Tier-1 (12 agencies) + Tier-2 (6 agencies) hardcoded at top of file.
- `rebuild_nc_agency_balanced.py` — per-agency rebalancing pass to remove the agency-identity confound surfaced by `analyze_nc_leakage.py` (per-agency positive rate ranges 49% → 89%, MI(agency, label) = 5.1% of H(L)). Caps each agency's contribution at 100K balanced rows. Outputs `notice_and_comment_agency_balanced.csv.gz` on sk3.
- `rtc_extracted/extract_rtc.py` — Phase A pure-regex extractor that locates the "Response to Comments" section of final-rule HTML and emits `rtc_sections.parquet` (3,644 docs). Strict-match.
- `rtc_extracted/extract_rtc_v2backfill.py` — permissive variant that adds more header patterns and a ±5K-char fallback window around comment-trigger phrases; used to backfill verbatim RTC text for documents flagged by the V2 pipeline that the strict regex missed.
- `v2_existing/analyze_v2.py` — coverage + quality audit of the V2 paraphrased corpus.

In `/Users/spangher/Projects/stanford-research/norm-research/scripts/`:

- `analyze_nc_leakage.py` — audits surface-length, exact dupes, label-discriminative trigrams, agency-as-confound, near-dup Jaccard for the length-balanced N&C file.
- `queue_nc_extractions.sh` — sequential STaR-Local feature extraction for nc_overall and 5 target agencies (CDC → USCIS → NOAA → FWS → EPA → overall).
- `queue_nc_clustering.sh` — Steps 1–6 of the clustering pipeline for the same task set (gpt-5-mini + BGE-large, no Llama needed).
- `queue_nc_agency_runs.sh` — full end-to-end local-explanations runs for the 5 smallest agencies (fsis, fs, irs, blm, osha) with Qwen3.5-122B-FP8 + gpt-5-mini proposer, on GPU 7.
- `link_comments_to_code_review.py` — cross-task linker (out of scope here).

## 4. File layout

```
datasets/notice-and-comment/
  notice_and_comment.csv.gz                # canonical pooled file (text, judgement)
  build_agency_splits.py                   # per-agency split builder
  rebuild_nc_agency_balanced.py            # per-agency rebalance (deconfound)
  agencies/                                # 18 per-agency dirs
    build_summary.csv                      # totals + pos/neg per agency
    <agency>/{train,eval,test}.csv.gz      # 70/15/15 split, text + judgement
  rtc_extracted/                           # verbatim agency response sections
    extract_rtc.py                         # strict regex extractor
    extract_rtc_v2backfill.py              # permissive backfill extractor
    rtc_sections.parquet                   # 3,644 RTC docs (strict)
    v2_backfill_rtc_sections.parquet       # backfill from V2 doc list
    rtc_per_agency.csv
    rubrics.jsonl
    samples.txt
    run.log
  v2_existing/                             # V2 LLM-paraphrased pairs corpus
    comment_responses_V2.jsonl             # 113,485 docs, ~112K pairs
    analyze_v2.py
    analysis_output.json
    quality_assessment.md                  # written 2026-06-01
    smoke_v3/                              # smoke-test jsonl in/out + errors
  online-rubrics/                          # external "how to write a good comment"
    raw/                                   # crawled HTML (federal-register guides,
                                           #   agency commenting pages, ABA/ACUS/AILA
                                           #   guidance, scholarly PDFs)
    claude-parsed/                         # 171 markdown rubric extractions
    gpt-parsed/gpt-5-mini/
    urls-visited.csv, *_log.csv, *_seen.txt
```

### Agencies covered (18)

Tier 1 (12): cms, epa, fws, fda, faa, aphis, noaa, ed, ams, nhtsa, uscis, dot.
Tier 2 (6): cdc, blm, fs, osha, fsis, irs.

Per `agencies/build_summary.csv`, totals range from CMS (607K rows, 70% pos) to FSIS (12K rows, 81% pos). Heavy class imbalance per agency — see leakage notes below.

## 5. Canonical dataset file

- **Pooled, unbalanced**: `notice_and_comment.csv.gz` — 235,294 rows, schema `text, judgement`. Each row is one claim. Built upstream by pooling all agencies.
- **Per-agency splits**: `agencies/<agency>/{train,eval,test}.csv.gz` — same schema, 70/15/15 random split keyed by row index (NOT by rule_id; group-leakage by rule is possible — flagged for fixing).
- **Per-agency rebalanced (sk3 only)**: `/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/notice_and_comment_agency_balanced.csv.gz` — 50/50 within each agency, agency cap 100K, carries an `agency` column for downstream `GroupShuffleSplit`.

## 6. Modeling state

Per `project_nc_pipeline_state.md` (49 days stale — verify against current outputs before relying on numbers):

| Subset | Train rows | Extraction | LoRA model | Canonical features |
|---|---:|---|---|---|
| nc_overall | 142,464 | done | running | running |
| EPA | 278,059 | done | OOM, fell back to base BGE | 30 |
| FWS | 86,138 | done | BGE-large | 30 |
| NOAA | 47,129 | OOM (retry) | — | — |
| USCIS | 29,643 | done | BGE-large | 30 |
| CDC | 21,654 | done | mpnet (pre-fix) | 30 |

Optuna sweeps on this corpus are blocked on a free GPU big enough for Llama-3.3-70B-FP8 (~135 GB). Llama-8B sweeps exist at `runs/notice_and_comment_sweep_llama8b/subset_0_{1..9}/trial_{0000..}`.

No reliable AUC headline on N&C yet at the date of this README. Peer-review comparator under the same recipe runs ~0.59 baseline / ~0.61 best.

### Verification_library / V+A+Taste extraction state

- **RTC parquet (strict)**: 3,644 docs of verbatim agency response sections.
- **V2 paraphrased pairs**: ~112K (comment, response) pairs across 15,061 responsive docs. LLM-summarized, two abstractions from raw rule text.
- **Online rubrics corpus**: 171 Claude-parsed markdown rubrics + a gpt-5-mini-parsed mirror, sourced from federal-register/EPA/agency commenting guides, ABA/ACUS/AILA professional guidance, and scholarship (Yackee, Shulman, Wagner). Indexed by `urls-visited.csv` + per-wave `waveh*_log.csv` crawler logs. Intended as external normative priors on what a "good comment" looks like.
- **No per-aspect Python predict-programs yet for N&C** — `runs/validity_full/v2/<task>/codegen_claude/` is populated only for peer_review.

## 7. Key decisions

**Structural mismatch with the response-as-feedback frame.** Documented in `project_nc_structural_mismatch.md` (2026-06-02). For peer review, code review, or jokes, the supervising text critiques the input. For N&C, agency responses overwhelmingly argue the **regulatory substance** (e.g. "we disagree because § 304(e) requires X"), not the **comment quality** ("the commenter's argument fails because…"). So response-extracted norms describe regulatory rationale, not comment-quality dimensions, and the supervision link from agency text to y=1 is weak.

**Usable subset ≈ 15–25%.** A minority of responses do critique comment quality ("the commenter provided robust empirical data on…", "the commenter failed to consider Y"). Expected yield: 17K–28K high-signal comment-critique pairs from V2's 112K. Path forward (deferred): train a classifier that splits (comment, response) pairs into COMMENT-CRITIQUE vs SUBSTANTIVE and keep only the former as supervision.

**Filtering rules proposed in V2 quality assessment** (`v2_existing/quality_assessment.md`):
1. `n_responses >= 2` to drop trivial docs.
2. Exclude `response_engagement_type == "acknowledgement"` and `rule_change_outcome == "no_change_made"` when response length < 200 chars.
3. Bias agency mix toward EPA, FAA, FCC, NOAA, FWS, CMS, IRS (high per-doc richness). Use SEC sparingly (V2 catches only 15% of SEC RTCs).
4. Backfill verbatim text from the strict RTC parquet for the ~1,156 RTC docs V2 missed.

**Binarization**: `judgement = 1` iff upstream `matched == "yes"`. Cluster-level label; same `cluster_uid → document_id` mapping applied to every claim from that submission.

**Dedup is exact-text only**, applied per agency in `build_agency_splits.py:86`. Near-duplicate templated comments (e.g. mass-comment campaigns) are not collapsed.

**Splitting**: random 70/15/15 over claim rows (not group-aware over rule_id, submission_id, or agency). This is the main known leakage risk. Per `analyze_nc_leakage.py`, per-agency positive rate ranges 49–89% and MI(agency, label) = 5.1% of H(L), so any pooled model that learns "guess the agency" gets a free lift. The agency-balanced rebuild + GroupShuffleSplit on agency or rule_id is the intended remediation.

**External rubrics are kept separate** in `online-rubrics/` and are not part of the labeled supervision pipeline — they support normative priors and clustering anchors only.

## 8. Open questions / next steps

1. **Group-aware splits.** Re-derive `rule_id` (and ideally `submission_id`) per claim and split with `GroupShuffleSplit(group=rule_id)` to eliminate same-rule and same-submission leakage between train/eval/test.
2. **Comment-critique classifier.** Build the (comment, response) → {COMMENT-CRITIQUE, SUBSTANTIVE} classifier flagged in `project_nc_structural_mismatch.md`. Train on a few hundred hand-labelled V2 pairs; apply to V2 to extract the 17–28K high-signal subset; treat this subset as the actual N&C "feedback corpus".
3. **Verbatim backfill from RTC parquet.** For the ~32% of strict-RTC docs where V2 has no responses (esp. SEC / CMS / FWS / HHS), run a focused per-doc extraction so downstream STaR / judge-rationale work has verbatim agency language.
4. **NOAA extraction retry** (47K docs, ~50 min on Llama-70B) — currently OOM-blocked.
5. **EPA LoRA retry** — first attempt OOM'd and fell back to base BGE; canonical features exist but could be improved.
6. **Optuna sweeps per task** (nc_overall first, then CDC / USCIS / FWS / EPA, NOAA last) — blocked on GPU big enough for Llama-3.3-70B-FP8 for the binary scoring step.
7. **Whether N&C remains a main task at all.** Per the structural-mismatch note, peer_review and code_review are structurally cleaner for the articulability-via-feedback story; N&C may belong as a small filtered supplement rather than a headline task.
8. **Per-aspect Python predict-programs for N&C** — Tier 1–4 deterministic ladder + codegen artifacts are missing for this task (currently exist only for peer_review).

## Related references

- `project_nc_pipeline_state.md` — pipeline state April 17 2026 (extraction / clustering / sweep status, blockers).
- `project_nc_structural_mismatch.md` — June 2 2026 structural-mismatch finding.
- `v2_existing/quality_assessment.md` — coverage + quality audit of V2 paraphrased pairs (2026-06-01).
- `scripts/analyze_nc_leakage.py` — leakage audit of the length-balanced pooled file.

---

## 9. VAT Campaign (2026-07-15, v4.2 corpus)

Supersedes §6's "no reliable AUC headline" — this section is the current source of truth for
N&C modeling results. Corpus: `v4/nc_v42_training.jsonl.gz` (per `v4/README.md`: 163K-row
reservoir, three outcome-label axes, docket-level stable-hash split). All artifacts referenced
below live in `datasets/notice-and-comment/v4/` unless noted. Memory pointers:
`project_nc_vat_run.md`, `project_nc_agency_size_throughput.md` (in the auto-memory store).

### 9.1 Sample and y-variable definitions

`v4/build_vat_sample.py` → `v4/nc_vat_sample.jsonl`: 7,482 comments across 22 agencies
(16 agencies at 400 balanced rows on the primary y, 6 smaller balanced sets), exact-text
dedup, per-docket cap 10/class. Full per-label rejoin (all engagement/response_type/outcome
values, not just the collapsed axis) in `v4/nc_vat_sample_labels_full.json`.

Three y-variables were tried, in order of what actually carries signal:

| y | definition | best instrument (AUC) | where the signal lives |
|---|---|---|---|
| **outcome-majority** (original primary) | majority vote of a comment's `outcome_collapsed` labels, MADE(1) vs NONE(0), ties dropped | dense .588 / V_deep .615 / 8B .602 | **between-docket only** (within-docket OOF concordance .498 = chance) — this y is mostly a rule-level property, not comment-level |
| **agree-vs-disagree** | majority vote of `response_type`, accepted/agree(1) vs disagree(0) | dense .671 (TF-IDF) / 8B .647 (docket-disjoint) | the only y with real within-docket, comment-level signal (.558) |
| **responded-or-not** ("true substantivity") | matched-with-any-label(1) vs genuinely unmatched, ≥700-char comment, same 22 agencies(0) | **dense .712 [.703,.726]** (campaign high) / V+A .646 [.635,.657] / A .636 [.625,.647] | strongest A-rubric signal of any y AND the only y with a real tacit gap (T̂ = dense − V+A = **+.066**); not a length artifact (unmatched comments are actually *longer*, char_len-alone AUC .422) |

Retired: **any-MADE union y** (a comment counted MADE if *any* of its matched-response labels
said MADE) — confounded by `n_labels` (comments matched to more responses mechanically get
more chances at MADE: n_labels-alone AUC .695 any-MADE vs .589 majority-vote). Never use
any-MADE as primary. `MADE vs CONSIDERED` and `engagement subst-vs-proc` were also audited
(`v4/y_audit_nc.py` → `v4/nc_y_audit.json`) and are weaker than the three above.

**Responded-or-not caveat**: "unmatched" ≠ confirmed-ignored — upstream matching recall
between comments and RTC sections is unknown (v4 README: `is_matched` base rate ~0.6% of
all comments). Read this y as "matched-in-our-pipeline-or-not," not "the agency truly never
engaged." Unmatched sample: `v4/nc_unmatched_sample.jsonl` (2,116 usable rows, same 22
agencies, ≥700 chars, docket-capped, drawn from `nc_v4_balanced_clean.jsonl.gz` cells with
`is_matched=False` and no labels).

### 9.2 A-lane: rubric bank + GEPA optimization

**A instrument**: 198 merged rubrics (`v4/nc_rubrics.jsonl`, sourced from
`rtc_extracted/rubrics.jsonl`) scored 1.0/0.5/0.0/NA by Gemma-4-31B offline batch vLLM
(`v4/score_va_gemma_nc.py`). 3 synthetic anchor comments (strong/mid/weak) ride along every
shard for judge-sanity checks — separated .872/.523/.032 identically across every shard run
this campaign, NA rate consistently ~0.63–0.64.

**GEPA optimization** (per the repo's label-free-fidelity GEPA convention, e.g.
`datasets/peer-review/gepa_revive_dead.py`): Sonnet is the frozen construct reference —
`v4/gepa_nc/sonnet_ref.py` scores all 198 rubrics against a fixed 63-item dev set
(60 comments, round-robin-sampled across all 22 agencies by hash order, label-free selection,
+ 3 anchors) via `claude -p --model sonnet`, one call per rubric scoring all dev items at
once. `v4/gepa_nc/diagnose_propose.py` computes a label-free fidelity score per rubric
(0.5·categorical-agreement + 0.5·max(Spearman,0) between Sonnet and Gemma on the dev set)
and has Sonnet rewrite the Gemma-facing description for any rubric below a 0.75 threshold,
targeting the specific disagreement patterns. `v4/gepa_nc/run_round.sh` chains scoring
(remote, sk2 GPU) → fetch → diagnose/propose (local) into one round.

Fidelity progression over 4 rounds (mean across 198 rubrics):

| round | mean fidelity | rubrics <0.5 |
|---|---|---|
| r0 (original bank) | .485 | 131/198 |
| r1 | .516 | 99/198 |
| r2 | .532 | 82/198 |
| r3 | .548 | 68/198 |
| r4 | .544 (plateau — stopped here) | 71/198 |
| **best-of-all-rounds** (`v4/gepa_nc/bank_best.jsonl`) | **.577** | 45/198 |

Full per-rubric round history in `v4/gepa_nc/history.json`.

**Post-GEPA full re-score result (7,439 rows, `v4/nc_scores_gepa_shard{0..4}.npz`) — an
honest NEGATIVE for prediction, with an interesting shape:**

| | outcome-y (n=7,084) | agree-y (n=5,046) |
|---|---|---|
| A pre-GEPA | .592 [.580,.608] | .612 [.596,.629] |
| **A post-GEPA** | **.578 [.564,.593]** | **.595 [.583,.610]** |
| V+A pre-GEPA | .593 | .625 [.611,.641] |
| **V+A post-GEPA** | **.574** | **.605** |
| within-docket (V+A) pre → post | .498 → .484 | **.558 → .501** |

GEPA **succeeded at its own objective** — construct fidelity to the Sonnet reference rose
.485→.577, anchor separation sharpened (.872/.523/.032 → .911/.575/.057), and 98/140
rubrics improved their univariate |AUC−.5| — **yet multivariate predictive AUC dropped on
both y's, and the agree-y within-docket signal (the genuinely comment-level part) collapsed
from .558 to .501.** Descriptive reading (not a settled conclusion): fidelity-optimization
made the rubric bank more construct-faithful and more internally redundant (all rubrics
pulled toward Sonnet's reading; NA rate rose .64→.69), stripping out judge-idiosyncratic
variance that the original noisy bank was exploiting for prediction. In other words, part
of the pre-GEPA A-bank's predictive power was NOT the articulated constructs themselves —
it lived in the unarticulated residue of how the judge happened to read the original
descriptions. This is directly relevant to the articulability thesis and worth its own
follow-up (e.g., is the same true for peer-review's A-bank?).

**Quoting rule: the headline A row for N&C remains the PRE-GEPA bank** (.612 agree /
.592 outcome). Post-GEPA numbers are reported alongside as the fidelity-vs-prediction
tradeoff finding, never silently substituted.

### 9.3 Deep-V lane: metric-seam hybrid programs

Per the metric-seam hybrid convention (`methods/metric_seam/hybrids/ops.py`,
`programs_peer_review/a214_h0.py`): each program optionally declares `LLM_FIELDS` (≤3
one-line extraction instructions filled by Gemma), then `def score(text, extracted, ops) ->
[0,1]` computes the metric deterministically, blended 0.65·code + 0.35·llm (defensive
try/except → 0.5). Extraction driver `v4/extract_nc_fields_gemma.py`, CPU scorer
`v4/score_nc_code_metrics.py` (auto-discovers `*_h0.py`/`*_h1.py`/`*_h2.py` in
`methods/metric_seam/hybrids/programs_notice_and_comment/`).

14 programs across 3 waves, in `methods/metric_seam/hybrids/programs_notice_and_comment/`:

**Wave 1 (`_h0`, outcome-oriented, comment-quality surface):**
| program | what it does |
|---|---|
| `citation_validity_h0.py` | real CFR (title 1–50) / USC (title 1–54) / Federal Register citation grammar; section- vs part-level precision; quoted-regulatory-text detection |
| `redline_ask_h0.py` | ready-to-adopt-edit detection ("strike X insert Y", "revise § N to read…") |
| `cba_rigor_h0.py` | magnitude-aware dollar/% parsing windowed near econ vocabulary; RIA critique |
| `evidence_provenance_h0.py` | named studies/Author-(Year)/DOI/URL counting + claimed credentials |
| `alternatives_analysis_h0.py` | proposed policy alternatives + trade-off language |
| `structure_org_h0.py` | code-only: headers, numbered lists, intro/conclusion, paragraph-length regularity |
| `legal_argument_h0.py` | sentence-level grammar: authority citation AND conflict verb in the same sentence |
| `stake_specificity_h0.py` | "representing N members", quantified impact, geographic specificity |

**Wave 2 (`_h1`, agree-y-targeted, persuasion-oriented, authored after the y-audit surfaced
agree-vs-disagree as the better y):**
| program | what it does |
|---|---|
| `stance_alignment_h1.py` | graded position parser (support/oppose, negation, "support…but", withdraw-override); best univariate on agree-y (.627, beats a crude stance-count baseline's .603) |
| `ask_modesty_h1.py` | ask-verb taxonomy by weight: clarification/phase-in (modest) vs overhaul/withdrawal (maximal) |
| `deference_tone_h1.py` | code-only: politeness vs accusatory-2nd-person/imperative/insult/caps/exclamation |
| `technical_precision_h1.py` | engagement with the rule's own apparatus (proposed §, preamble, agency's own data) |

**Wave 3 (`_h2`, verification-tier — LLM extracts, program *decides correctness*, the
strongest metric-seam case):**
| program | what it does |
|---|---|
| `numeric_consistency_h2.py` | parses all quantities (magnitude/unit-normalized $, %, counts, rates; excludes years/citations/page/docket numbers), checks component-sum-to-total / part-whole / rate×count arithmetic coherence within sentence-adjacency windows, penalizes internal contradictions |
| `authority_lookup_h2.py` | parses CFR/USC/FR citations, looks the LOAD-BEARING one (LLM-identified) up against a real eCFR index (`v4/cfr_parts_index.json.gz`, built by `v4/build_cfr_index.py` from the live eCFR API, 49/50 titles, 9,700 parts) — score rewards citing authorities that actually exist |

Smoke-tested (strong > weak, [0,1], `ops=None`-safe) for all 14; wave 3 additionally verified
real-vs-fabricated CFR cites (7 CFR 56 exists → scores higher than a fabricated 7 CFR 9999).
**Wave 3 has NOT yet been scored on the full VAT sample** (extraction only ran through wave
2's aspects, `v4/nc_fields_v2.jsonl`) — open item, re-run `extract_nc_fields_gemma.py` with
all 14 aspects then `score_nc_code_metrics.py` for the complete Deep-V table.

**Results (docket-grouped CV, waves 1+2 = 12 programs, `v4/nc_deepv2_scores.npz`):**

| | outcome-y | agree-y |
|---|---|---|
| V_deep (12 programs) | **.615** | .595 |
| V_deep, wave-2-only (4 programs) | .596 | .600 |
| V regex (27 features) | .595 | .612 |
| V regex + V_deep | .609 | .610 |
| dense (TF-IDF) | .588 | .671 |
| Llama-8B (docket-disjoint LoRA) | .602 | .647 |

**Headline: V_deep (.615) beats both regex-V (.595) and the fine-tuned 8B (.602) on
outcome-y** — the 8 interpretable outcome-tier hybrid programs match or exceed a black-box
dense model trained on 4,891 docket-disjoint rows. On agree-y, `stance_alignment_h1` alone
(.627) beats the full V_deep bundle (.595) — the graded stance parser is doing most of the
work; the other 3 wave-2 aspects add little on top of it.

### 9.4 Dense (TF-IDF) and 8B LoRA bounds

Matched-protocol dense: TF-IDF word(1–2) + char_wb(3–5), vectorizer refit per fold
(`v4/aggregate_vat_nc.py`), same docket-grouped CV as everything else.

Llama-3.1-8B-Instruct LoRA (`methods/dense/train_reward_model.py`, selection on eval split,
test scored exactly once via `eval_test_once.py`), trained on a **docket-disjoint** pool from
`nc_v42_training.jsonl.gz` (dockets excluded if they appear in the VAT sample; exact-text +
prefix-collision guarded against the VAT sample too) and evaluated on the VAT sample rows:

| y | train pool | test rows | internal-split AUC (leaky, sanity check) | **honest docket-disjoint test AUC** |
|---|---|---|---|---|
| outcome-majority | 4,891 rows | 7,124 | .893 | **.602** |
| agree-vs-disagree | 6,729 rows | 5,086 | (not re-checked) | **.647** |

The internal-split number (.893 for outcome-y) is included specifically to show *why* it must
not be quoted — it demonstrates the model can trivially separate rows sharing a docket
(rule-level shortcut), collapsing to .602 once dockets are made disjoint. **Interesting
reversal**: the 8B beats TF-IDF-dense on outcome-y (.602 > .588) but loses to it on agree-y
(.647 < .671) — worth investigating further, not yet explained.

### 9.5 Per-agency ladder (agree-y, docket-grouped, sorted by dense AUC)

`v4/nc_agree_per_agency.json`. Reliability rule (per the original outcome-y run,
`project_nc_vat_run.md`): ≥20 dockets and ≥8 dockets/class in-sample or don't trust the
readout — grouped-CV on very few dockets gives artifact AUCs dominated by rule-level outcome
clustering (63% of multi-comment dockets are outcome-pure).

| agency | n | dense | V | A (pre-GEPA) | V+A | V_deep |
|---|---|---|---|---|---|---|
| FAA | 278 | .773 | .702 | .586 | .595 | .723 |
| DOT | 159 | .711 | .518 | .587 | .532 | .678 |
| BLM | 100 | .666 | .558 | .315 | .361 | .408 |
| FSIS | 273 | .638 | .514 | .611 | .612 | .507 |
| USCBP | 327 | .633 | .568 | .400 | .451 | .632 |
| IRS | 95 | .626 | .593 | .448 | .494 | .265 |
| CDC | 292 | .619 | .631 | .621 | .625 | .706 |
| AMS | 302 | .613 | .628 | .494 | .542 | .478 |
| FDA | 262 | .601 | .610 | .522 | .580 | .601 |
| EPA | 285 | .584 | .492 | .520 | .530 | .617 |
| NOAA | 236 | .550 | .467 | .543 | .513 | .523 |
| ED | 265 | .521 | .508 | .581 | .526 | .666 |
| APHIS | 223 | .502 | .432 | .435 | .419 | .394 |
| FWS | 315 | .493 | .459 | .527 | .517 | .535 |
| CMS | 254 | .482 | .598 | .527 | .540 | .589 |
| FEMA | 112 | .476 | .513 | .443 | .443 | .347 |
| USCIS | 295 | .391 | .614 | .456 | .474 | .512 |
| MSHA | 85 | .354 | .437 | .389 | .388 | .453 |
| NHTSA | 221 | .315 | .302 | .474 | .431 | .374 |
| FS | 279 | .269 | .686 | .526 | .567 | .572 |
| ICEB | 328 | .191 | .579 | .287 | .510 | .239 |

Note the tail (ICEB, NHTSA, FS-dense) is genuinely bad, not smoothed over — worth a
fold-stability check (esp. FS's below-chance dense .269 on n=279) before quoting individually.

### 9.6 Agency size / throughput vs VAT signal (Collins Relational Tacit Knowledge test)

Tests the hypothesis that larger agencies build more structured, mechanically-legible
comment-handling routines (Harry Collins' Relational Tacit Knowledge, made procedural within
an organization). Three data sources merged in `v4/nc_size_vat_merged.json`:

1. **FTE headcount** — FedScope on-board civilian headcount, Sept 2024, via USAFacts.org
   (one consistent snapshot methodology across all agencies; web research, 2026-07-15,
   `v4/nc_agency_fte.json`; per-agency confidence flags — NHTSA's 902 FTE is single-source,
   uncross-validated; FEMA/IRS/AMS/FS/BLM have an alternate budget-FTE figure noted where it
   diverges materially).
2. **Rulemaking volume** (dockets, proposed/finalized rules, response counts) — **already
   sitting in the sibling project**, `/Users/spangher/Projects/stanford-research/rfi-research/
   regulations-demo/data/derived/paper_stats/{per_agency_table.csv,per_agency_volume.csv}`,
   built from the regulations.gov bulk download. Check there first before re-deriving agency
   volume for any future task. **19/21 agencies matched cleanly; USCBP and ICEB fell back to
   comments-only** because that table uses lowercase `cbp`/`ice` codes that don't join to our
   `USCBP`/`ICEB` v42 codes (`v4/nc_agency_volume_official.json`, `source: FALLBACK` flag).
3. **VAT signal** — the agree-y per-agency table above.

**Raw correlations (Spearman, n=19–21):**

| | dense | V_deep | A | V |
|---|---|---|---|---|
| FTE (headcount) | +.325 (p=.15) | +.204 (p=.38) | −.025 (p=.92) | **+.496 (p=.022)** |
| dockets (lifetime total, throughput proxy) | +.242 (p=.32) | **+.635 (p=.003)** | **+.507 (p=.027)** | +.023 (p=.93) |

**Headline: raw headcount does NOT predict mechanizable signal** (FTE vs A/V_deep ≈ 0) —
it only correlates with the shallow regex-V. **Docket throughput predicts V_deep and A far
better than headcount does.** Reframe of the hypothesis: it isn't organizational headcount
that builds legible structure, it's *case throughput* — agencies running many separate
rulemakings (FAA 8,056 dockets, EPA 5,268, FDA 3,111) look like they've proceduralized
comment-handling into codifiable routines; low-docket high-volume-per-docket agencies (BLM 43
dockets/540K comments, IRS 415, FS 104) show it least.

**Confound check (user-flagged, important): does "more dockets" just mean "more distinct
dockets land in the fixed-size sample," which mechanically improves grouped-CV reliability
regardless of any real organizational effect?** Yes, partially. `n_dockets_sample` (distinct
dockets actually appearing in an agency's VAT rows) correlates ρ=+.891 with `dockets_total` —
almost fully mechanical, since a bigger docket pool spreads a capped sample across more
dockets, which reduces AUC-estimate noise independent of any true effect.
**Partial correlation of `dockets_total` vs signal, controlling for `n_dockets_sample`**
(`v4/nc_agency_docket_diversity.json`):

| | raw ρ | partial ρ (net of sample docket-diversity) |
|---|---|---|
| V_deep | +.635 (p=.003) | **+.576 (p=.010) — survives, modestly attenuated** |
| A | +.507 (p=.027) | +.396 (p=.093) — drops to marginal, don't trust as-is |
| dense | +.242 (p=.32, already weak) | **−.062 (p=.80) — vanishes entirely, was pure artifact** |

**Bottom line**: the dense/A correlations with throughput were mostly or entirely a sample-
composition artifact. **V_deep is the one result that holds up** after netting out the
confound — real signal that agencies processing more distinct rulemakings show more
mechanically-legible comment structure, but this is not proof of the Collins mechanism
specifically ("dockets_total" is a lifetime count from the bulk download, not a true per-year
rate — no clean rulemaking-rate series exists, this is an approximation).

**FAA-outlier check (done):** FAA (8,056 dockets — the largest by a wide margin) could have
been single-handedly driving the whole correlation. Re-tested with FAA excluded (n=18):

| | rho with FAA (n=19) | rho w/o FAA (n=18) | rho w/o FAA, partial on sample-diversity |
|---|---|---|---|
| V_deep | +.635 (p=.003) | **+.571 (p=.013)** | **+.572 (p=.013)** |
| A | +.507 (p=.027) | +.482 (p=.043) | +.389 (p=.110) — marginal |
| dense | +.242 (p=.32) | +.108 (p=.67) | −.098 (p=.70) |

**V_deep's correlation with docket throughput survives both checks independently** — it is
not an FAA artifact and it is not (fully) a sample-diversity artifact. This is now the most
defensible version of the finding: agencies that run more separate rulemakings show
genuinely more mechanically-legible comment structure, net of the two most obvious
confounds. Still n=18-19 and still only two of many possible confounds checked.

### 9.6b Distributable metrics packet

All explicit metrics from this campaign are packaged for distribution in
**`metrics_packet/`** (built by `metrics_packet/build_packet.py`): the 198 A-rubrics
enriched with per-rubric agency provenance (which agencies' documents each was mined
from — 87 multi-agency / 53 single-agency / 58 no-agency-doc; sources.csv lists all
889 source documents; leaf_criteria.jsonl unpacks every rubric to the 3,063 original
per-source statements), the GEPA-optimized variants (with the quoting-rule warning),
the 27 V regex features as a standalone module, the 14 V_deep hybrid programs
(self-contained, eCFR index included), the judge protocol + anchors, and a
per-metric univariate performance summary.

### 9.7 Reproduction pointers

| result | script | output |
|---|---|---|
| VAT sample | `v4/build_vat_sample.py` | `v4/nc_vat_sample.jsonl` |
| A-scoring (Gemma rubrics) | `v4/score_va_gemma_nc.py` | `v4/nc_scores_shard{0..4}.npz` |
| y-audit | `v4/y_audit_nc.py` | `v4/nc_y_audit.json` |
| pooled + per-agency + by-year aggregation | `v4/aggregate_vat_nc.py` | `v4/../notebooks/data/nc_vat.json` |
| GEPA reference scoring | `v4/gepa_nc/sonnet_ref.py` | `v4/gepa_nc/ref_scores.json` |
| GEPA round driver | `v4/gepa_nc/run_round.sh`, `diagnose_propose.py` | `v4/gepa_nc/bank_r{N}.jsonl`, `history.json` |
| deep-V hybrid programs | `methods/metric_seam/hybrids/programs_notice_and_comment/*_h{0,1,2}.py` | — |
| deep-V extraction | `v4/extract_nc_fields_gemma.py` | `v4/nc_fields_v2.jsonl` |
| deep-V scoring | `v4/score_nc_code_metrics.py` | `v4/nc_deepv2_scores.npz` |
| eCFR authority index | `v4/build_cfr_index.py` | `v4/cfr_parts_index.json.gz` |
| 8B dense LoRA | `methods/dense/train_reward_model.py` + `eval_test_once.py` | sk2 `nc_vat/dense8b/run{,_agree}/` |
| agency FTE research | (web research, one-off) | `v4/nc_agency_fte.json` |
| agency volume (already local) | regulations-demo `data/derived/paper_stats/` | `v4/nc_agency_volume_official.json` |
| size/throughput correlation | inline analysis | `v4/nc_size_vat_merged.json`, `v4/nc_agency_docket_diversity.json` |
| responded-or-not | inline analysis + `v4/nc_unmatched_sample.jsonl` | `v4/nc_responded_or_not.json` |

### 9.8 Open items

1. ~~Full re-score with GEPA bank~~ **DONE** — see §9.2: fidelity up, predictive AUC down;
   pre-GEPA bank remains the quoted A row. Follow-up worth running: same test on the
   peer-review A-bank (does fidelity-optimization strip predictive judge-idiosyncrasy there
   too?).
2. ~~Wave-3 (`_h2`) deep-V programs~~ **DONE** — scored on the full sample
   (`v4/nc_deepv3_scores.npz`): authority_lookup .539 / numeric_consistency .540 univariate
   on outcome-y — the verification instruments work but are weakly discriminating at this y.
   **Consolidated notebook: `notebooks/2026-07-16__nc-vat-campaign.ipynb`** (executed, all
   tables + figures + the code-review RTK comparison). Also added late: responded-or-not
   dense = **.712 [.703,.726]** — campaign high; T̂ = dense − V+A = **+.066**, the first N&C
   y with a genuine tacit gap.
3. 8B outcome-vs-agree reversal (8B beats dense on outcome-y, loses on agree-y) — unexplained.
4. ~~FS's below-chance dense AUC (.269, n=279)~~ **RESOLVED**: confirmed fold-instability
   artifact, not a real finding — FS has only 6 total dockets (0% docket-purity among its
   4 multi-row dockets), and dense AUC swings .269→.421 across 6 different CV seeds
   (mean .355, std .069). Do not quote FS's per-agency AUC as a real below-chance result.
5. ~~Size/throughput finding needs a FAA-excluded re-test~~ **DONE**: V_deep survives
   FAA-exclusion (ρ=.571, p=.013) and FAA-exclusion + partial-diversity-control together
   (ρ=.572, p=.013) — see §9.6. Still only two confounds checked of many possible; a design
   that equalizes sample docket-diversity by construction (rather than partialling it out
   post hoc) remains the more rigorous next step if this becomes a headline claim.
6. USCBP/ICEB docket-volume fallback (comments-only) — could be fixed by joining
   regulations-demo's `ice`/`cbp`-coded rows properly instead of falling back.

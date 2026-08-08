# V7 — patents COMMUNITY/vote cell (y = forward citations)

Built 2026-08-08. Cell slug `patents_fwdcites`. Coordinator brief: build the
patents community/vote cell for the VAT decomposition program, on the same
document corpus whose VERDICT cell (claim-fell) was closed as a leak
post-mortem.

Everything below is sk3 unless stated; `export HOME=/lfs/skampere3/0/alexspan`
first, every time. Python `/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python`.

---

## 0. What was reused vs built (reuse-before-rebuild inventory)

| artifact | status |
|---|---|
| `datasets/patents/build_forward_citations.py` | **REUSED as the starting point**, then extended — the original discards `citation_category` and cannot window (see §1) |
| `datasets/patents/forward_citations.parquet` (9.05M cited patents) | **REUSED for the ground-truth check only.** Not used to build `y`: it is an all-time, all-category, duplicate-inclusive total |
| PatentsView raw tables (`g_patent`, `g_patent_abstract`, `g_cpc_current`, `g_us_patent_citation`) | REUSED |
| `processed/granted_patents_claim1_v2.parquet` (6.9M claim-1 texts) | REUSED |
| `stable_hash_bucket_map` (claim-fell) | **TRIED AND REJECTED** — degenerates on this cell, see §5 |
| claim-fell A bank (4 columns) | **NOT REUSED** — see §6 |
| claim-fell row construction (claim elements + candidate references) | **NOT REUSED** — see §2 |
| `datasets/va_gemma_banks/score_va_gemma_banks.py`, `score_scaleupC_banks.py` | REUSED verbatim (scoring loop, sharding, NA parsing, anchor battery) |
| `methods/dense/run_dense_standard_scaleupC.sh`, `train_reward_model.py`, `score_eval_dense_v4.py` | REUSED verbatim (frozen recipe) |
| `methods/taste_decomposition/scaleupC_layer1.py`, `layer1_gemma_cells.py` | REUSED as imports (no estimator reimplemented) |

Nothing the claim-fell campaign produced was re-run.

---

## 1. MANDATORY GROUND TRUTH — label channel verified by hand

`gt_fwdcites.py` → `/lfs/skampere3/0/alexspan/tmp/gt_fwdcites.json`.
21 patents (stratified across the whole count distribution, grant years
2000–2015) were recounted by a **full scan of all 151,140,729 raw citation
edges** and compared against `forward_citations.parquet`.

**Result: 21/21 exact matches.** The parquet's counts are correct.

Three properties of the raw channel were established from the data (not
assumed), and all three change the build:

**(a) `citation_date` is the CITED patent's grant date, not the citing date.**
2,721/3,000 sampled edges match the cited patent's grant month; **0/3,000**
match the citing patent's. A post-grant citation window therefore CANNOT be
computed from that column — the citing patent must be joined to `g_patent` for
its own grant date. (The original `build_forward_citations.py` comment
"year-bucketed counts if citation date present (for citation-age normalization
later)" would have produced a window on the wrong date.)

**(b) Patents with zero forward citations are ABSENT from the parquet**
(`min(n_forward_cites) = 1`). Any join must left-join and fill 0, or it silently
drops the entire bottom of the distribution.

**(c) `citation_category` is populated but ERA-DEPENDENT** — see §3.

---

## 2. Population: why NOT the claim-fell rows

The brief described forward citations as "a different y on the same document
population". The corpus is the same; **the row construction is not reused**, for
a specific reason. claim-fell's unit is a claim-element from an *application*,
paired with a candidate prior-art reference set. Its post-mortem
(`notes/2026-08-07__closure_patents.md`) found the two channels that killed it
are properties of exactly that construction:

* claim ORDINAL NUMBER, printed verbatim in the element text, .754 alone —
  85% of the eval gap;
* a LABEL-CONDITIONAL reference set (gold reference appended to positives
  99.66% / negatives 0.00%, last slot 86.6%).

Forward citations attach to **granted patents**, not to claim elements, so
reusing that construction would have imported both poisons for no benefit. The
V7 unit is the granted-patent document.

**Universe**: US utility patents (`patent_type == "utility"`, `withdrawn != 1`)
granted 2005–2015 = **2,374,167** patents; 2,374,102 carry an inventional CPC.
Grant year ≤ 2015 so that every patent's 5-year window closes by 2020 and is
fully observed in data that runs through 2025.

**Text = title + abstract + claim 1, and nothing else.**

---

## 3. Confound design — the four traps

### AGE — solved by a fixed window, and the fix is measurable

`y` counts only citations from patents granted within **[grant_year,
grant_year + 5]** (calendar-year resolution), then takes a **within-cohort**
split. The window alone already flattens the mechanical age gradient:

| grant year | all-time fwd cites (mean) | **5-year window (mean)** |
|---:|---:|---:|
| 2005 | 22.69 | **3.77** |
| 2008 | 16.94 | **3.84** |
| 2011 | 13.48 | **3.87** |
| 2013 | 10.98 | **3.72** |
| 2015 | 8.01 | **3.58** |

All-time counts fall 22.7 → 8.0 (a pure age artifact — older patents have had
longer to accumulate). The windowed count is flat at 3.58–4.03. The
within-cohort split then removes the residual era and technology gradients.
Confirmed on the final matrix: **grant-year-alone AUC .491, CPC-section-alone
.506** (§7).

### SELF / EXAMINER citations

`citation_category` exists, with this era structure (measured over all 151M
edges, by CITING grant year):

| citing year | examiner | applicant | "cited by other" | blank |
|---:|---:|---:|---:|---:|
| ≤2001 | 0.000 | 0.000 | 0.000 | 1.000 |
| 2002–2012 | .38→.25 | **0.000** | **.55→.76** | 0.000 |
| 2013+ | .23→.20 | **.74→.82** | 0.000 | 0.000 |

**"cited by other" IS the applicant bucket before 2013** — PatentsView simply
renamed it. Taking the literal `"cited by applicant"` string therefore yields
**identically zero** for every cohort granted before ~2008, which is exactly the
silent-null failure the brief warned about. The era-robust
examiner-independent measure is **`tot5 − exm5`**, and that is what is used.
Every citing year inside a 2005–2015 cohort's window falls in 2005–2020, i.e.
wholly inside the era where `"cited by examiner"` is reliably marked.

Both halves ship as never-merged secondary y's (`y_fwd5_examiner`,
`y_fwd5_nonexaminer`).

**Self-citation limitation (recorded, not solved):** the PatentsView tables
downloaded here carry **no assignee table**, so applicant self-citations cannot
be identified and are included in the primary y. Mitigation:
`y_fwd5_examiner` counts only examiner-added citations, which are third-party by
construction and therefore immune to the self-citation trap; it is reported
beside the primary y rather than instead of it.

Counts are of **DISTINCT citing patents** — 74,846 duplicate (citing, cited)
pairs were collapsed during the scan.

### METADATA — the claim-fell killer

No examiner field, art unit, assignee, inventor, attorney, filing or grant date,
patent number, CPC code, `num_claims`, or citation count reaches any instrument.
Cohort keys (grant year, CPC class) are used to *define* y and are then dropped.
This is asserted in code in three places, not merely intended:
`build_v7_population.py` (banned column names + an examiner-token scan of the
text), `score_v7_patents_bank.py` (banned-token scan of the assembled judge
context), and `build_dense_bundle.py` (banned substrings + an ISO-date regex over
the dense `text` column). The A-bank merge additionally drops any *criterion*
mentioning excluded metadata (§6).

### FAMILY / CONTINUATION duplicates

Grouping unit = a near-duplicate cluster: MinHash-LSH (64 permutations, 16×4
bands, Jaccard ≥ 0.6 over 5-gram shingles of title + claim 1), unioned with
exact normalised-title match **that also clears Jaccard ≥ 0.3 on claim
language**. The claim-language guard is load-bearing: a first pass unioned on
title alone and merged 26 unrelated patents sharing the title "semiconductor
device". Final: 15,973 groups over 16,000 rows, largest 8, 17 multi-member
groups. Within multi-member groups the y-agreement rate is **.730** vs a .5
baseline — near-duplicates really do share the outcome, so the grouping is doing
real work.

---

## 4. `y` definition

`y_fwd5` = **within-(grant_year × CPC class) median split** of the 5-year
distinct-citing-patent count; cohort-median ties dropped. This is the
`mathse_vote_score` / `so_votes` convention (within-group median split of the raw
vote score), which is the right precedent since this is the program's patents
*vote* cell.

Cohort thresholds are computed on the **full 2.16M-patent universe**, not on the
sample, so they are population-referenced and sampling cannot distort them.

* cohort = grant_year × CPC class, minimum 50 patents (1,233 cohorts); 3,174
  rows dropped in undersized cohorts.
* Degenerate-cohort guard: 178 cohorts whose untied rows were all one class (or
  had <20 untied rows) were dropped (211,371 rows). Such a cohort carries no
  within-cohort contrast and would smuggle cohort identity back in as signal.
* Tie loss on the primary y: 21.0% of the universe.
* Eligible universe: 1,780,093. Sampled to **n = 16,000** (884 cohorts).

Secondary y's, carried in the bank meta and **never merged**:
`y_fwd5_examiner`, `y_fwd5_nonexaminer`, `y_fwd_alltime` (the age-confounded
comparator, kept only to demonstrate the trap), `y_fwd5_topquartile`.

---

## 5. Splits — the imported bucketer had to be replaced

`stable_hash_bucket_map` (imported from `build_dense_standard_claimfell.py`)
produced **train 15,972 / eval 14 / test 14**.

Cause, worth recording because it will bite any other near-singleton-group cell:
the bucketer scores a candidate bucket as
`size_term + λ · Σ_b (pos_rate_b − overall_rate)²` with λ = 2.5. Adding one row
to a nearly-empty bucket swings that bucket's pos rate to 0 or 1 (penalty
≈ λ·0.25), while adding it to the large bucket barely moves the rate. With
15,973 groups of median size 1 the pos-rate term dominates the size term at
every greedy step and pours everything into one bucket. The term is load-bearing
on claim-fell (few large app groups, corr(group size, y) = +.30); here
corr(group size, y) = **+.013**, so it buys nothing and costs the split.

Replaced with a plain stable hash of the family group —
`sha1("v7-split|" + group) % 100` → 80/10/10 — which is what the build spec
asks for and satisfies the standing no-seeded-shuffle rule.

| split | n | pos rate | groups | cohorts |
|---|---:|---:|---:|---:|
| train | 12,787 | .5094 | 12,764 | 851 |
| eval | 1,573 | .5238 | 1,572 | 437 |
| test | 1,640 | .4951 | 1,637 | 439 |

0 families straddle a split. Split-identity-alone AUC **.4993**.

---

## 6. V and A

**V** — `datasets/patents/v7_community/v_features.py`, **58 deterministic
columns**, same module shape (`V_NAMES` / `v_features` / `vector`) and same
generic length-structure-register tail as the StackOverflow and math.SE cells,
so the surface channel stays comparable across cells. The domain block is
patent-specific: claim-1 grammar (transitional phrase, limitation count,
means-plus-function, "configured to" functional language, antecedent markers,
relative/degree terms), the textual claim-BREADTH proxies from the patent-scope
literature (claim-1 word count, limitation count, "plurality of" / "at least
one"), plus abstract, title and cross-field columns.

V matrix validated over all 16,000 rows (`v7_vcheck.py` → `v_matrix_check.json`):
58/58 columns finite, **zero degenerate columns**, and — the important negative
result — **no single V column exceeds AUC .552** (top: abstract sentence count
.5515, abstract mean sentence length .4516, title↔claim word overlap .4607).
Contrast claim-fell, where claim ordinal number alone reached .754: this cell
has no comparable structural killer channel in its surface block.

**A** — the claim-fell A bank was **not reused**. Its own audit found it is
*one* concept in four aggregations (pairwise ρ .853–1.000, two exactly 1.0), it
scores element-level disclosure by prior-art references, and the claim-fell
RUNBOOK's revival prerequisite #3 explicitly requires building a real
multi-criterion bank. Coverage on this population would in any case be zero: the
unit is different (granted patent vs claim element).

The V7 bank is mined fresh, label-blind, by the program's standard route: four
proposer subagents each read a disjoint batch of 14 real train-split patents
(no label attached, sha256 order), each given a different angle, each proposing
17 candidates self-labelled Track A / Track B. `build_rubrics.py` then applies a
hard metadata ban, a circularity ban (nothing may ask the judge to predict
importance/influence/citation), token-overlap dedup, and the GEPA phrasing
rewrite to the fixed two-sided `<property> / HIGH / LOW / NA` shape.

Scoring: Gemma-4-31B, offline batch vLLM, temperature 0, one token per
(item, criterion), 8 shards, `--battery 50`.

**Anchor deviation (deliberate).** so_votes anchored its battery on `y_accepted`,
an independent channel on the same rows. **This cell has no independent
channel** — every observed label on a granted patent here is a forward-citation
count, i.e. the quantity under test — so a y-based anchor would be circular.
The anchors are instead CONSTRUCTED and MATCHED: each trio is one real patent in
three states — intact / deterministically degraded (generic title, boilerplate
abstract, claim limitations replaced by purely functional shells, length
preserved) / word-scrambled. The known label is the degradation, so the
certificate is non-circular while remaining a known-label blinded anchor in
every judging batch.

---

## 7. Leak battery on the final matrix

Run inside `build_v7_population.py`; stored in `population_manifest.json`.

| probe | AUC | reading |
|---|---:|---|
| grant year alone | **.491** | age confound removed |
| CPC section alone | **.506** | technology confound removed |
| cohort identity, out-of-fold | **.532** | small residual, see below |
| cohort identity, in-sample | .633 | vs permutation null [.606, .617] |
| split identity alone | **.4993** | splits carry no signal |
| family within-group y agreement | .730 | (vs .5) grouping is doing real work |
| `num_claims` alone | **.596** | DECLARED NUISANCE — see below |
| full text char length | .525 | |
| claim-1 char length | .523 | |
| corr(group size, y) | **+.013** | the claim-fell splitter landmine is absent |

The **.532 out-of-fold cohort residual** is real but small and benign in origin:
cohort medians sit on small integers, so the share of untied rows falling above
the median varies slightly by cohort. No instrument is shown the cohort, the
year or the CPC code, so none can exploit it directly. Recorded, not headlined.

**`num_claims` at .596 is this cell's declared nuisance channel** — the analogue
of claim-fell's claim-ordinal killer, banked at round 0 exactly as that
post-mortem's RUNBOOK item 4 requires. It is *not* a leak: only claim 1 is shown
to any instrument, so `num_claims` is not recoverable from the text. It is
nonetheless banked as a STRUCT block and Δ is quoted over **V + A + STRUCT** as
well as over V + A.

---

## 8. Artifacts

All paths sk3-relative to `/lfs/skampere3/0/alexspan/norm-research/`.

| artifact | path |
|---|---|
| ground-truth report | `/lfs/skampere3/0/alexspan/tmp/gt_fwdcites.json` (+ `gt_fwdcites.py`) |
| citation aggregates (2.37M patents) | `datasets/patents/v7_community/v7_cite_aggregates.parquet` |
| category-by-era tally | `datasets/patents/v7_community/v7_cat_by_year.json` |
| population + y + splits | `datasets/patents/v7_community/population.csv.gz` |
| manifest + leak battery | `datasets/patents/v7_community/population_manifest.json` |
| V features (58 cols) | `datasets/patents/v7_community/v_features.py` |
| A-bank proposals (raw) | `datasets/patents/v7_community/proposals_b{0,1,2,3}.jsonl` |
| A bank | `datasets/patents/v7_community/rubrics.jsonl` (+ `rubrics_audit.json`) |
| A scoring driver | `datasets/patents/v7_community/score_v7_patents_bank.py` |
| A matrices | `outputs/va_gemma_banks_patents_fwdcites/patents_fwdcites_shard{0..7}.npz` + `_meta.json` |
| dense bundle | `datasets/patents/v7_community/dense_standard/{data.csv,split/,manifest.json}` |
| dense preds / T | `datasets/patents/v7_community/dense_standard/rm_out_seed{42,1,2}/`, `eval_pass_results.json` |
| layer-1 script | `methods/taste_decomposition/patents_fwdcites_layer1.py` |
| **layer-1 ledger** | `methods/taste_decomposition/results/patents_fwdcites_ledger.json` |
| OOF ids | `methods/taste_decomposition/results/patents_fwdcites_oof_ids.npy` |

Build scripts staged in `/lfs/skampere3/0/alexspan/tmp/` and committed to the
repo: `gt_fwdcites.py`, `v7_agg_cites.py`, `build_v7_population.py`,
`v7_fix_splits.py`, `v7_exemplars.py`, `v7_ydiag.py`.

---

## 8b. A-bank construction, in numbers

68 raw proposals → 66 after the metadata + circularity bans (2 dropped, both on
"influence") → 35 after the curated semantic merge, the lexical pass, and the
post-smoke drops. **Final bank: 35 criteria, 28 Track A / 7 Track B**, 560,000
judge calls at n = 16,000.

The lexical dedup fired **0 times** on 68 candidates: the four proposers were
given different angles and re-derived the same concepts in different *words*
(antecedent basis twice, functional-language-needs-structure three times,
element interdependency three times, claim-scope-vs-contribution three times,
causal mechanism at three loci). 20 merges were therefore made by reading all 66
survivors; each is recorded in `build_rubrics.py:CURATED_MERGE` with its
survivor and reason, and mirrored into `rubrics_audit.json`.

Post-smoke drops were made on **measured** judge behaviour, threshold NA > 0.70:
"Numeric Parameters Tied To Function" (NA .85), "Chemical Or Biological
Mechanism Disclosed" (NA .81 and every scored value 0.0), and "Numeric Threshold
Tied To Rationale" (its merge survivor was one of those). High-but-*meaningful*
NA was kept: "Method Step Causal Chain" sits at NA .63 because it is
domain-gated on method claims, and an NA that means "this document is not a
method" is information, not a failure.

Smoke health (27 rows × 37 criteria): overall NA .189, mean score .674.

**Anchor calibration.** The first constructed degradation replaced title,
abstract and claim 1 with boilerplate and scored **.190 against .192 for pure
word-scrambled nonsense** — it destroyed as much as scrambling, collapsing the
required pos > neg > scram ordering (`score_bank` retries such a shard four
times, then marks it invalid). Recalibrated to degrade **only claim 1**, keeping
the real title and abstract, so abstract-side criteria stay partially
satisfiable and the degraded document lands between intact and nonsense.

## 9. Ledger

**V legs (final, frozen layer-1 estimators, n = 16,000, 15,973 groups,
pos rate .509):**

| leg | AUC | 95% CI (group bootstrap) |
|---|---:|---|
| V_lin (58 surface cols) | **.5899** | [.5812, .5991] |
| V_nl (GBM, seeds 0/1/2) | **.5950** | seeds .5961 / .5949 / .5942 |
| V_interact | +.0051 | |
| STRUCT_lin (declared nuisance: num_claims + 3 lengths) | **.6018** | [.5932, .6105] |
| V+STRUCT_lin | **.6222** | [.6133, .6311] |
| V+STRUCT_nl | **.6218** | seeds .6222 / .6220 / .6213 |

The notable result here is that **the 4-column STRUCT block alone (.6018) beats
the entire 58-column V surface block (.5899)**, and almost all of that is
`num_claims` (.596 alone). Claim count is the single strongest observable on this
cell — and it is *not* visible to any instrument, because only claim 1 is shown.
That is precisely why it is banked as STRUCT and Δ is quoted over V + A + STRUCT
as well as over V + A.

**A legs, T and Δ_beyond: PENDING.** Both remaining jobs are launched and
resumable; the box is running 8/8 GPUs at 100% from other agents' campaigns, so
they are grinding rather than blocked:

| job | state |
|---|---|
| dense T (Llama-3.1-8B LoRA, seeds 42/1/2, GPU 2) | training, seed 42 at step ~150/1600 |
| A-bank full scoring (Gemma-4-31B, 560K calls) | bank frozen and smoke-validated; awaiting GPU headroom |

To finish, in order:

```bash
export HOME=/lfs/skampere3/0/alexspan
cd /lfs/skampere3/0/alexspan/norm-research
# 1. A bank (shard-checkpointed: re-running skips finished shards)
CUDA_VISIBLE_DEVICES=N /lfs/skampere3/0/alexspan/envs/gemma4/bin/python \
    datasets/patents/v7_community/score_v7_patents_bank.py --battery 50 --util 0.85
# 2. dense arm is already chained (RUN_DONE-sentinel resumable); it scores itself
# 3. the ledger
/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python \
    methods/taste_decomposition/patents_fwdcites_layer1.py
```

Step 3 asserts the OOF reproduction rule itself (reloads the saved OOF array and
ids from disk, reassembles, and requires the assembled-order AUC to match the
published figure to < 1e-9).

---

## 10. Deviations and limitations

1. **Row construction not reused** from claim-fell (§2) — the unit differs and
   both known leak channels are properties of that construction.
2. **A bank rebuilt, not reused** (§6) — the claim-fell bank is one concept at a
   different unit; its own RUNBOOK requires a rebuild before revival.
3. **Split bucketer replaced** (§5) — the imported one collapses on
   near-singleton groups.
4. **Anchor labels constructed, not observed** (§6) — no non-circular observed
   label exists on this population.
5. **Self-citations cannot be excluded** — no assignee table (§3). Mitigated by
   reporting the examiner-only y.
6. **Window is calendar-year resolution**, not day-exact: a citation counts if
   the citing patent's grant *year* is within grant_year..grant_year+5
   inclusive (so up to 6 calendar years). Day-exact windowing is possible from
   `g_patent.patent_date` if ever needed.
7. **`y` is a within-cohort split, so this cell measures relative standing
   inside a technology-and-era cohort**, not absolute citation impact. That is
   the intended construct for a community/vote cell, but it means the numbers
   are not comparable to raw-citation-count regressions in the patent
   literature.
8. `citation_category` cannot separate third-party or "imported from a related
   application" citations before 2013; both are <0.04% of edges and fall into
   the non-examiner bucket.

# Legal-outcome-prediction V/A audit — 2026-06-22

Deep honesty + coverage audit of the 12-domain legal V/A ladder. Triggered before sharing the
ladder with collaborators. Findings + corrections below.

## 1. Faithfulness audit (defects, ranked by impact)

1. **CAVC→BVA linkage contaminated in the low-margin tier (HIGH, fixable).** Hand-audit (subagents,
   60 stratified pairs, CAVC recital re-joined from `decisions_*.jsonl` by the `case` field): margin≥0.15
   ~95% correct; margin 0.06–0.15 ~100% correct; **margin<0.06 ~90% FALSE MATCH** (2/20 correct). Median
   **303 same-date BVA candidates** per CAVC case; templated language → TF-IDF top-1 below 0.06 is usually
   the wrong veteran. The 0.035 acceptance threshold was too loose: **1,538/9,140 pairs (16.8%) are <0.06
   → ~15% of CAVC is false matches** (corrupt input text + label). FIX: raise threshold to ≥0.06 →
   `modeling_pool_cavc_ge006.jsonl.gz` (7,602 pairs) built; re-score in flight. Silver lining: false
   matches attenuate toward chance, so CAVC's old 0.62 was partly noise-suppressed.

2. **CAVC level was era-inflated (HIGH, now measured).** DOL & CAVC were never run through `exante_scrub`
   (0/200 rows carried `<DATE>`/`<YR>` tokens vs scrubbed peers). Scrub-rescore (fix#1): **DOL genuine**
   (A 0.676→0.686 on scrubbing, no drop); **CAVC era-inflated** (A 0.620→0.563, −0.057 from decision-date
   signal). So CAVC's honest level is materially below 0.62 once cleaned (≥0.06) + scrubbed.

3. **FLSA usable-count overcount (reporting error — corrected).** `flsa_fullpool_v3` has 11,160 rows but
   **4,020 are labeled −1** (MIXED/PROCEDURAL/UNCLEAR, dropped before modeling). Binary-usable FLSA =
   **7,140**, not 11,160. (A/V figures unaffected — scored on the clean subset.) Title VII 6,410 and
   SS 52,560 are clean-balanced.

4. **Mixed-outcome handling inconsistent in CODE, ~0% in impact (LOW).** CAVC drops affirmed_in_part;
   DOL/MSPB force →0; court slices drop MIXED→−1. But the pool fractions are ~0% (DOL 0%, CAVC 0%, MSPB
   1.2%=7 cases), so cross-domain gaps are NOT materially affected. Code-hygiene item, not urgent.

5. **CAVC + DOL use a naive id-split, not an entity group-split (LOW–MED).** ptab/nlrb/ttab/mspb verified
   group-disjoint (0 entities cross splits); CAVC/DOL have empty `entity_key` (split by docket/id hash) →
   mild memorization risk for repeat veterans/respondents.

**CLEAN (verified):** metric banks — adversarial scan of all 780 metrics found **no tautological metric**,
principled thin/thick flagging, scoring prompt forbids inferring the ruling. Group-splits disjoint for
ptab/nlrb/ttab/mspb. Tier-1 de-confounding sound (pro-se/enforcer-plaintiff drop, posture-only-AUC≈0.5).
`exante_scrub` reversible + non-destructive. `fact-free-source` guard catches extraction hallucinations.

## 2. Coverage audit (are we scraping enough?)

| Source | Universe | Collected | Verdict |
|---|---|---|---|
| BVA | ~1.1M | ~1.13M | ✅ complete (sitemap bug fixed) |
| CAVC enumeration | 142,007 docketed | 20,770 decisions | ✅ complete |
| PTAB | ~19,300 trials | 15,345 pairs (~80%) | ✅ excellent |
| Trademark | ~5.11M eligible | **318,380** (was 79,936; 4× expansion done) | designed sample, now larger |
| TTAB | 283K opp/can | 4,894 (1.7%) | ⚠️ elite brief-filter slice — expansion pending label check |
| DOL | ~43K docs | 8,936 pairs (~20%) | ⚠️ partial (pre-1998 gap, 100K cap) |
| NLRB | 5,272 ALJ | 2,995 (57%) | ⚠️ pagination cap + join loss |
| MSPB | ~3,648 CAFC | 598 (16%) | ⚠️ FOIA-only structural wall |
| Court slices | unknown CL baseline | 6.4K/7.1K/52K/1.9K | ⚠️ SS drops consent-remands (pos 0.94 → pool harder than reality); ERISA ~1,888 likely <5% of LTD |

## 3. Web-research corroboration (is the 0.50→0.76 spread expected?)

**Yes.** Two axes explain the spread: (a) **political volatility / case-specific strategy** → low
predictability (NLRB 0.50: Board swings with admin, Semet 2016; PTAB 0.52: "death squad" + art-unit
specifics, Love & Ambwani 2014 / Helmers & Love 2023); (b) **deferential review of a fixed record** →
high (ERISA 0.76 Langbein 2007; DOL 0.73). SS 0.63 explained by documented **ALJ variance** (grant
rates 20–80% across ALJs, Krent & Morris 2013). Priest–Klein selection compounds the low end. The
chance domains are genuine idiosyncrasy, not harness failure.

## 4. In-flight / pending
- **CAVC cleaned (≥0.06) re-score** + **SS & Title-VII leakage probes** — `vat_part_ab.py` on GPU3.
- **T (dense ceiling)** — queued (ModernBERT Tier-2 `dense_ceiling_exante.py`, Llama-8B Tier-1
  `canonical_llama8b.py`); no legal-domain T results existed yet.
- **TTAB ~40× expansion** — pending unbriefed-merits label-quality check before the filter drop.
- **Fix #4** (unify mixed-outcome; CAVC/DOL entity split) — lower priority.
- Patents — handled in a separate thread.

## 5. Resolved results (2026-06-22, evening)

- **CAVC honest level = ~0.56** (was 0.620). Two independent corrections converge: scrubbing the
  contaminated pool (A 0.620→0.563) AND cleaning the false-match tier ≥0.06 (A→0.563). CAVC's 0.62 was
  inflated ~0.06 by decision-date/era signal plus false-match noise. Gap stays ~0 → genuine low-A/low-gap
  domain. **Shared table: CAVC row → A≈0.56.** Cleaned pool `modeling_pool_cavc_ge006.jsonl.gz` = 7,602.
- **Leakage-probe template applied to SS + Title VII (gaps are genuine, not framing):**
  - SS: A 0.583→0.577 (neutralized), gap +0.087→+0.029 — survives (V≈chance so gap is all articulable).
  - Title VII: A 0.611→0.644 (neutralized), gap +0.043→+0.059 — **grows** (cleanest).
  - Contrast ERISA: ~⅓ framing on decision-quality metrics. So Tier-1 gaps are real doctrine; ERISA is the
    framing outlier. Probe = reusable framing-detector.
- **T (dense ceiling) — two fast attempts INVALID, proper run in flight:**
  - ModernBERT-base (150M, fine-tuned): too weak — title_vii T=0.610 < A=0.637.
  - 70B few-shot direct: too weak (no calibration) + 2500-char truncation gutted Tier-2 long docs → near
    chance (0.40–0.57). Finding: even 70B raw prediction is near-chance; the doctrinal DECOMPOSITION (A) is
    what unlocks predictability (structured judgment > raw pattern-matching).
  - **Proper T = Llama-3.1-8B + LoRA** (repo's `fullpool_llama8b.py`), 2000 rows/domain, running GPU3 (~80min).
- **Trademark 4× expansion done** (`pool_v200k.jsonl.gz`, 318,380 pairs).

## 6. T (dense ceiling) LANDED — Llama-3.1-8B + LoRA, all 12 domains (2026-06-23)

`dense_T_llama8b_all.py` → `dense_T_llama8b_all_results.json`. 2000 balanced rows/domain (vs A's 600),
single 80/20 stratified split, scrubbed text. **MSPB skipped (n=162 < 200 threshold) — FOIA-walled
micro-domain, no dense ceiling possible.** 11/12 domains have T.

| Domain | V | A | T | A−V | T−A |
|---|---|---|---|---|---|
| erisa_ltd | 0.548 | 0.758 | 0.644 | **+0.210** | −0.114 |
| dol | 0.717 | 0.728 | 0.622 | +0.011 | −0.106 |
| trademark | 0.620 | 0.641 | 0.638 | +0.021 | −0.003 |
| title_vii | 0.576 | 0.637 | 0.654 | +0.061 | +0.017 |
| ss_disability | 0.529 | 0.626 | 0.671 | +0.097 | **+0.045** |
| ttab_dupont | 0.594 | 0.613 | 0.487 | +0.019 | −0.126 |
| flsa | 0.603 | 0.604 | 0.674 | +0.002 | **+0.070** |
| mspb_cafc | 0.582 | 0.596 | — | +0.014 | — |
| ttab_exa | 0.549 | 0.584 | 0.545 | +0.035 | −0.039 |
| cavc_review | 0.562 | 0.563 | 0.534 | +0.001 | −0.029 |
| ptab_aia | 0.529 | 0.523 | 0.475 | −0.006 | −0.048 |
| nlrb | 0.502 | 0.504 | 0.512 | +0.002 | +0.008 |

**Headline finding (T−A):** In **7/11 domains, articulated doctrine (A) MEETS OR BEATS the 8B dense
fine-tune (T)** — despite T having 3× the training rows. Doctrine wins most in ttab_dupont (−0.126),
erisa (−0.114), dol (−0.106). Dense wins only where **raw surface text carries signal the doctrinal
rubrics don't encode**: flsa (+0.070, factual hours/wage arithmetic), ss (+0.045, medical terminology),
title_vii (+0.017).

**Honest caveat (do not over-claim):** A and T are NOT the same model/protocol. A = 70B metric-extraction
+ LR (5-fold CV, n=600); T = 8B LoRA (single split, n=2000). **The 8B is a LOWER BOUND on the true dense
ceiling** — so "A > T" is a conservative statement about how much structured judgment adds; it does NOT
rule out a tacit residual a larger dense model would surface. The 4 "T > A" domains are the clean
dense-residual signal. To make T a true upper bound, a fine-tuned 70B (or ≥8B-fp8 stacked) ceiling would
be the next rung — not yet run.

## 7. Fix #4 refined — largely MOOT on inspection (2026-06-23)

Re-examining defect #4 (mixed-outcome) + #5 (CAVC/DOL entity split):

1. **Scorer is group-agnostic, not the pool split.** `run_vat.py` uses `StratifiedKFold(5, shuffle=True,
   random_state=0)` over ALL rows for ALL 12 domains — it **never reads the pool's `split` field**. So the
   audit's "CAVC/DOL naive id-split" concern is moot for the reported A/V: every domain is scored with the
   same random stratified CV. (The pool `split` field is only consumed by the dense-ceiling T's
   train_test_split — also random, also not the group split.)
2. **Mixed-outcome rows are already absent.** y_raw in CAVC/DOL/MSPB pools is clean 4-way
   (vacated_remanded / affirmed / reversed / remanded) — NO `affirmed_in_part` / `mixed` survived
   pool-build. The "inconsistency" was a source-stage artifact; nothing to unify.
3. **Repeat-entity leakage cannot inflate A/V.** A/V are an LR over doctrine-metric *values* (0/0.5/1);
   party identity is not a feature, so a repeat employer/veteran/applicant in both folds gives the LR
   nothing extra to memorize. Measured: DOL employer caption extractable in only 15% of rows, **4.0% of
   rows** belong to a repeat employer (and the top "repeat" is a misparsed OWCP-director caption, not an
   employer → real rate lower). Trademark pool carries no owner field. CAVC veteran is anonymized in the
   text ("the Veteran"). So identity leakage is bounded and A/V-irrelevant.
4. The only consumer where identity *could* leak is dense-T (sees raw text w/ names); T≈A in most domains
   bounds that, and a group-aware T re-run is available as a robustness check if ever needed.

**Conclusion: no code change warranted.** Defects #4/#5 were over-stated; the reported ladder is sound on
both axes. Documented here rather than "fixed," because there is nothing material to fix.

## 8. T⁺ (V+A+T) probe — COMPLETE (2026-06-23)

`build_Tplus.py` + `build_Tplus_oriented.py`. T⁺ = zero-shot FP8-70B predicts outcome from
`<facts>` + `<legal_factors>` (each metric's name + definition + per-case assessed value), single-token
logprob(1). Same 600-case sample as A. Scoring byte-identical to A (imports run_vat). Two passes because
the first run's metric directions were stated per-metric (mixed) → 4 domains inverted; the oriented re-run
flips each metric by sign(corr(value,y)) so "1 = favors moving party" uniformly. Final numbers are
**oriented** (build_Tplus_oriented_results.json). A from vat_ladder.json (CAVC honest 0.563); T from
dense_T_llama8b_all_results.json; T-direct (facts-only 70B) from dense_T_direct_results.json.

| Domain | V | A | T(8B) | T-direct | T⁺ | T⁺−A | rubric-gain (T⁺−T-direct) |
|---|---|---|---|---|---|---|---|
| erisa_ltd | 0.548 | 0.758 | 0.644 | 0.657 | **0.767** | +0.009 | **+0.110** |
| flsa | 0.603 | 0.604 | 0.674 | 0.560 | **0.664** | **+0.060** | **+0.104** |
| ss_disability | 0.529 | 0.626 | 0.671 | 0.563 | 0.616 | −0.010 | +0.053 |
| title_vii | 0.576 | 0.637 | 0.654 | 0.548 | 0.614 | −0.023 | +0.066 |
| ttab_dupont | 0.594 | 0.613 | 0.487 | 0.515 | 0.569 | −0.044 | +0.054 |
| ttab_exa | 0.549 | 0.584 | 0.545 | 0.512 | 0.546 | −0.038 | +0.034 |
| ptab_aia | 0.529 | 0.523 | 0.475 | 0.490 | 0.542 | +0.019 | +0.052 |
| dol | 0.717 | 0.728 | 0.622 | 0.522 | 0.501 | **−0.227** | −0.021 |
| cavc_review | 0.562 | 0.563 | 0.534 | 0.542* | 0.523 | −0.040 | (see ‡) |
| nlrb | 0.502 | 0.504 | 0.512 | 0.462 | 0.489 | −0.015 | +0.027 |
| trademark | 0.620 | 0.641 | 0.638 | 0.515 | 0.470 | **−0.171** | −0.045 |
| mspb_cafc | 0.582 | 0.596 | — | 0.599* | 0.613 | +0.017 | (see ‡) |

‡ **FIXED 2026-06-24** via MOV relabel (`build_Tplus_relabel.py`). cavc/mspb both have **y=1 = affirmed =
government/agency wins** (build_pairs.py L148; cavc confirmed), but the prompt's "moving party" named the
veteran/employee (backwards) → spurious inversion. Corrected MOV to the affirmed side + cache re-predict
(no re-score): **cavc 0.491→0.523, mspb 0.364→0.613** (mspb now ≈ A=0.596). *T-direct shown polarity-
flipped (1−original) since the original facts-only run used the same wrong MOV. dol & trademark remain
GENUINE collapses (T-direct≈chance even at correct polarity — the 70B can't read long ALJ orders / bare
mark-goods text zero-shot; needs fine-tuning, which T=0.62/0.64 provides).

**Findings:**
1. **Linear A is a strong ceiling.** T⁺ (zero-shot 70B + facts + rubric) beats A in only ~1–2 domains
   (flsa +0.06; erisa +0.01 within noise). The linear combination of doctrine scores is hard to exceed by
   reasoning over the same scores.
2. **The rubric genuinely helps the dense model** (rubric-gain = T⁺ − T-direct): positive in 8/12, median
   +0.05, up to **+0.11 (erisa, flsa)**. Handing the law explicitly adds real signal beyond raw facts.
3. **ERISA is the showcase for the user's hypothesis.** T⁺(rubric)=0.767 vs T(8B fine-tune)=0.644: where
   doctrine is a crisp structural legal test (discretionary-authority → deferential review), explicit-law-
   reasoning beats learning-from-2000-cases. The 8B cannot learn the standard-of-review nuance; the rule
   handed over works.
4. **The reverse holds for messy-facts domains** (dol, trademark): fine-tuning T=0.62/0.64 works but
   zero-shot rubric-reasoning collapses (T⁺≈0.50, rubric-gain ≤0, T-direct≈chance). There you MUST learn
   from data; the rule alone doesn't crack long ALJ orders / bare mark-goods text.
5. **T⁺ ≫ T where fine-tuning fails** (erisa +0.12, ttab_dupont +0.08, ptab +0.07): rubric-reasoning
   unlocks domains the 8B can't learn.

**Caveats:** T⁺ conflates facts + rubric (a no-facts variant would isolate pure non-linear-A signal);
±0.02 FP8 reproducibility floor (flsa +0.06 and erisa +0.01 straddle it — the robust rubric-gain is the
better-anchored number); cavc/mspb MOV relabel DONE 2026-06-24 (see ‡) — all 12 T⁺ values now clean.

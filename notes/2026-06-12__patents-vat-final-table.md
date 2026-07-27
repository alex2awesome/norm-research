# Patents VAT decomposition — final table (2026-06-12)

## V claim-level (true-cites testbed, independent claims)

- v0 max-cos v3: **0.502**
- v0 max-cos v6a: **0.569**
- v2 min_of_max (v6a retr + qwen judge): **0.574** ← claim-level V ceiling for the bi-encoder recipe

### v6.1a ablation (overnight 06-11→06-12): 7× data, no gain

```
v6a  +qwen122b: min_of_max pooled=0.5743 within=0.5747 (n=3273)
v6.1a+qwen122b: min_of_max pooled=0.5598 within=0.5513 (n=3273)
```

**⚠️ Within-doc eval caveat:** the post-chain log's EVAL 2 block (v6a MRR 0.595, v6.1a 0.207)
uses the CONTAMINATED split — `seed(7)+shuffle` reshuffled when the apps list grew, leaking
v6a train apps into the test split. Contamination-controlled numbers (round-2-only pairs,
`eval_v6_within_doc_honest.py`):

| model | n | MRR | top1 | top3 | top10 |
|---|---|---|---|---|---|
| v6a (honest) | 2,274 | 0.207 | 0.105 | 0.208 | 0.429 |
| v6.1a (honest) | 2,274 | 0.214 | 0.107 | 0.229 | 0.428 |

→ **bi-encoder plateaued**; v6a stays production retriever.

### xenc-v1 follow-through (06-12 early am): every remaining pipeline knob ruled out

xenc-v1 = bge-reranker-base distilled from the 71,685 Qwen judge labels
(`models/element-para-xenc-v1`; held-out AUC vs judge **0.844**, agree@0.5 0.77).

**1. Ranking improves** (honest round-2-only within-doc):

| ranker | MRR | top1 | top3 | top10 |
|---|---|---|---|---|
| v6a alone | 0.207 | 0.105 | 0.208 | 0.429 |
| xenc rerank v6a top-20 | 0.257 | 0.151 | 0.283 | 0.474 |
| xenc full ranking | **0.261** | 0.146 | 0.287 | 0.497 |

**2. Judge swap is free** (same v6a top-3 pairs, min_of_max recipe):
Qwen-122B 0.5743/0.5747 vs **xenc 0.5735/0.5782** (n=3273) — the 280M
cross-encoder replaces the 122B judge at pipeline level.

**3. But the better ranking does NOT move pipeline AUC** (wide-K, xenc over
v6a top-K from cited docs' full specs; K=3 reproduces the swap exactly):

| pool | pooled | within |
|---|---|---|
| top-3 | 0.5735 | 0.5782 |
| top-10 | 0.5707 | 0.5595 |
| top-30 | 0.5672 | 0.5801 |

Max-pooling noise eats the ranking gain (+26% rel MRR → −0.6pt AUC at K=30).

**Exclusion chain for the 0.574 V2 ceiling:** not judge-limited (swap test),
not retriever-data-limited (v6.1a 7× null), not retrieval-breadth-limited
(wide-K null). Remaining suspects are structural: element decomposition +
min/max aggregation, and the label itself — §103 combination-reasoning and
examiner discretion live outside disclosure lookup (i.e., taste).

### Gold-evidence + aggregation follow-through (06-12 pm): the structural suspect convicted

**Gold-evidence diagnostic** (examiner's own cited paragraphs as evidence):
gold pairs score "disclosed" only 37.8% (vs v6a-top1 30.7%; gold>top1 just
56.5% → low within-doc MRR is pessimism-by-construction, retrieval exonerated).
**Oracle recall: only 26.7%** of actually-rejected claims pass min-over-elements
with examiner evidence. Manual 20-pair audit: 9 DISCLOSES / 9 PARTIAL / 2 WRONG
→ extraction alignment sound; PARTIAL = limitation in adjacent paragraph.

Evidence-side repairs both FAIL discriminatively (recall up, AUC down — max-noise):
p±1 windows lift oracle recall 26.7→33.7% but v2 AUC drops to 0.5617.

**Aggregation-side repair WORKS**: sweep over claim-score aggregators
(same Qwen pair scores) — softmin beats strict min; split-half validated
(stable hash by record, T chosen on opposite half):

| aggregator | held-out AUC (half0/half1) |
|---|---|
| min (baseline) | 0.5834 / 0.5648 |
| **softmin (T≈0.05–0.15)** | **0.5922 / 0.5916** |

→ **claim-level V2 = ~0.59 with softmin** (was 0.574). Diagnosis: strict
min-over-elements is brittle under judge noise — one weak element vetoes the
claim. Substantive reading: examiner §102/103 judgment does not decompose into
independent per-element paragraph lookups; the ~66% of rejected claims that
fail even oracle-evidence verification quantify the gap between mechanical
element-wise verification and holistic examiner judgment.

## V app-level (3K main + 1K zero-resolved control apps)

- main: n=1880, AUC=0.5469 → **honest ≈ 0.53 after confound control** (see below)
- control (falsification): n=986, AUC=0.5551 — **did NOT collapse to 0.5**

### Aggregator sweep at app level (06-12 pm): softmin does NOT transfer — V signal ≈ ZERO

Same softmin sweep as claim level, on the 67,326 app-level pair judgments:

| aggregator | main AUC | control AUC |
|---|---|---|
| min (baseline) | 0.5469 | 0.5551 |
| softmin T=0.1 | 0.5354 | 0.5367 |
| softmin T=0.3 | 0.5295 | 0.5760 |
| mean | 0.5239 | **0.5885** |

Main ≤ control for EVERY aggregator; softer aggregation *raises* the
filler-only control (pure confound, no real art present) while main falls.
Split-half: held-out softmin 0.515/0.538 vs min 0.528/0.562 — strictly worse.
→ The claim-level softmin gain is real signal recovered from judge noise;
the app-level "signal" is the n_elements/claim-length artifact and nothing
else. **Honest app-level V = null.** Claim-level V2 softmin **0.59** is the
implemented-V number for the VAT table.

### Why the control "failed" — artifact identified (06-12 02:45 analysis)

1. **min-of-max ∝ −n_elements**: more elements → lower min, mechanically (corr v↔n_el = −0.14).
   Element count / claim-1 length predict ACCEPTANCE on their own (AUC toward rejected 0.45/0.39 —
   longer claim = narrower scope = easier allowance). V inherits "short broad claim → rejected"
   through its denominator — identically present in the filler-only control → 0.555.
2. **Citation-count confound**: n_real (resolved cited docs) alone = AUC **0.64**, but is
   leakage-adjacent (more rejection events mechanically produce more cited docs) — excluded.
   n_real-conditioned V-AUC: 0.530 / 0.529 / 0.534 (strata 1 / 2 / 3+). Logistic coefs →rejected:
   v 0.10, n_el 0.04, c_len −0.36, n_real 0.59.
3. **The judge is fine**: mean V 7.2 on real-art apps vs 4.0 filler-only (zero-rate 42% vs 58%) —
   real cited art IS detected as more disclosing. Pair-level signal exists; it doesn't aggregate
   into accept/reject with cited docs' claim-1 texts as evidence. Next granularity lever:
   spec paragraphs as evidence (#57c).

### V per rejection class (main apps, accepted vs class-rejected)

| class | n_rej | V AUC |
|---|---|---|
| rejected_101 | 158 | 0.6528 ⚠️ |
| rejected_102 | 815 | 0.5672 |
| rejected_103 | 1184 | 0.5449 |
| rejected_112a | 179 | 0.4967 |
| rejected_112b | 597 | 0.5294 |

⚠️ §101's V "lead" is suspect — likely the same n_el artifact (§101-rejected claims skew short/broad);
V should be *worst* on §101 by theory. Treat per-class V as descriptive only until n_el-conditioned.

## A per rejection class (qwen judge, pooled CV logistic over aspects)

| class | n_rej | A AUC | 95% CI |
|---|---|---|---|
| rejected_101 | 33 | **0.780** | [0.69, 0.861] |
| rejected_102 | 87 | 0.560 | [0.484, 0.625] |
| rejected_103 | 148 | 0.526 | [0.471, 0.577] |
| rejected_112a | 28 | 0.418 | [0.322, 0.525] |
| rejected_112b | 67 | 0.560 | [0.48, 0.632] |
| rejected_112d | 9 | 0.670 | [0.432, 0.869] |
| rejected_dp | 9 | 0.599 | [0.364, 0.824] |

Per-aspect: §101 rubrics hit a31=0.73/a53=0.88 vs §101 rejections; novelty/obviousness rubrics
(a26/a34/a35) are at chance (0.52–0.56) vs actual §102/103 rejections.

## Dense ceiling (Llama-8B reward model)

- first_draft: **0.7413**
- first_draft_cpc: 0.7349 (tech class not a confound)
- final_outcome: 0.7005
- +applicant cites: 0.7445 / +examiner cites: 0.712

## COMMON-COHORT VAT TABLE (06-12 pm) — same 803 dp for dense + A + V (#60)

Dense model (subset_0_7 best, the 0.73 checkpoint) scored on the SAME 803-dp
validity cohort the A matrix lives on (`scripts/dense_score_dp_cohort.py` →
`runs/validity_full/v2/patents/dense_scores_dp.json`). Loading verified: the
identical code path reproduces **0.7333** on the sweep's own test split.

- **dense pooled on cohort: 0.5826** (0.5789 excluding the 52 dp that leak
  into the dense train subsample; 709/803 dp are in NO sweep split — the
  validity pool ≠ the CPC/length-balanced year≤2021 sweep pool).
- → the 0.74 "dense ceiling" is pool-specific; on the common cohort, dense
  (0.58) sits at par with implemented claim-level V2 softmin (0.59).

Per-class, accepted (420) vs class-rejected, same rows as the A table:

| class | n_rej | dense AUC | A AUC | winner |
|---|---|---|---|---|
| rejected_101 | 33 | 0.671 | **0.780** | A — eligibility is articulable |
| rejected_102 | 87 | **0.658** | 0.560 | dense — prior-art (V's domain) |
| rejected_103 | 148 | **0.623** | 0.526 | dense — prior-art (V's domain) |
| rejected_112a | 28 | 0.513 | 0.418 | neither |
| rejected_112b | 67 | 0.579 | 0.560 | ≈ tie |

Reading: the rubric bank OUTPERFORMS the dense model on §101 — articulated
eligibility rules beat learned features. The dense model's edge is exactly
where theory predicts V should live (§102/103 prior-art), and its margin
there (0.62–0.66 vs A's 0.53) brackets what a full V implementation must
beat; our claim-level V2 (0.59 softmin, different unit) is within range.
V app-level on this cohort: null (aggregator sweep, see above).
Caveats: small n_rej for §101/§112a; class flags only for dp with OARD rows.

## #61 OA-scale V RESULT — mixing real examiner art improves verification (06-13)

Ran on a balanced 12K sample from the OACT-bulk testbed (3,000 §102-rejected +
3,000 §103-rejected claims, both with resolved examiner-cited gold paragraphs,
+ 6,000 non-rejected claims; stable-hash subsample of the 341K-unit
`gold_mixture_testbed_v1`). Pipeline: Qwen-122B element decompose → v6a top-10
paragraph retrieval per element → xenc-v1 disclosure score → **softmin** claim
aggregation (split-half T-tuned, T=1.0 selected). Claim score AUC vs whether the
claim fell to §102/§103:

| condition | rej | pooled AUC | within-doc |
|---|---|---|---|
| **retrieved-only** (v6a top-k, fair control) | all | **0.621** | 0.626 |
| retrieved-only | §102 | 0.578 | 0.597 |
| retrieved-only | §103 | 0.635 | 0.640 |
| **mixture** (top-k + real examiner-cited art) | all | **0.641** | 0.639 |
| mixture | §102 | 0.595 | 0.586 |
| mixture | §103 | 0.658 | 0.662 |

**Examiner-evidence advantage (mixture − retrieved, paired, 2000× bootstrap):
all +0.020 [+0.017,+0.023], §102 +0.017 [+0.014,+0.021], §103 +0.022
[+0.018,+0.026] — ALL significant (CIs exclude 0).** Mixing the examiner's actual
cited prior-art paragraphs into the retrieved evidence significantly improves
claim-level verification, and **§103 (obviousness / combination) benefits more
than §102 (anticipation)** (§103 CI barely overlaps §102's) — consistent with
§103 requiring references the bi-encoder retriever is more likely to miss.

AUC 95% bootstrap CIs: retrieved-only all 0.621 [0.610,0.632] / §102 0.578
[0.556,0.600] / §103 0.635 [0.620,0.649]; mixture all 0.641 [0.630,0.652] /
§102 0.595 [0.573,0.617] / §103 0.658 [0.643,0.672]. (`logs/gold_mixture_bootstrap.log`.)

OA-scale **retrieved-only V = 0.62** is the headline implemented-V number — it
*exceeds* the claim-level truecite V2 (0.59), because the larger fresh sample +
correctly T-tuned soft aggregation recover more signal than the earlier min-ish
recipe. This is the fully-fair number (both labels get retrieved evidence).

Caveats: (1) gold exists only for rejected claims (examiner cites no art against
a non-rejected claim), so part of the +0.020 mixture lift is a mere-presence
artifact — retrieved-only 0.62 is the clean number, +0.020 is an upper bound on
the pure examiner-evidence value. **But a per-element decomposition shows the lift
is mostly REAL, not artifact**: the examiner's gold paragraph beats *all* v6a-retrieved
candidates on **19.2% of positive elements** (22.8% of positive claims), with mean
disclosure-score lift 0.139 (median 0.078, max 0.98) where it wins. Mere-presence
would give tiny uniform boosts; instead the lift concentrates in ~1-in-5 elements
where retrieval missed an anticipatory paragraph the examiner found — i.e. the
examiner-evidence advantage measures a genuine retrieval gap. (`logs/gold_mixture_decompose_lift.log`.)
(2) gold-only condition is undefined (positives only → nan). Artifacts: `scripts/score_gold_mixture.py` (AUC-orientation bug
fixed: softmin is positively oriented to "fell", matching truecite/v2_aggregation
convention), element scores cached at `units_sample_elscores.json`.

### #61 DEEP DIVE — operational detection vs real examiner label (06-13)

Does V actually fire correctly on the claims we have ground-truth for? (logs:
`v_detection_deepdive.log`, `v_failure_modes.log`.)

- **Modest-lift detector, NOT a sharp tail.** Precision flat ~74% across V≥0.5→0.8;
  top-5% = top-20%. §103: fires 22%, precision 71% vs 55% base = **1.3× lift** (the
  real win). §102: 1.08× (base already 78%).
- **TPs are right-for-the-right-reason** (near-verbatim): claim "second resistance
  greater than first" → examiner paragraph "The second resistance may be greater
  than the first resistance." Multi-element precision-when-fired 82%.
- **Where V works best (corrected by slice AUC):** V's sweet spot is SHORT dependent
  claims with one crisp limitation that's verbatim in prior art (the verbatim-match TPs).
  Counterintuitively the multi-element *independent*-claim slice scores LOWER (retrieved
  0.591 / mixture 0.612 / §103-multi 0.626) than the full sample (0.621/0.641/0.658) —
  independent claims have many elements (softmin dilutes) and ~80–87% base reject (little
  discriminative room). So the headline 0.62 is NOT understated by dependent-claim
  "collapse"; those claims prop it up. (`v_cleanslice_auc.log`.)
- **Genuine depressors:** (1) **misaligned gold** on the misses — low-V *rejected* claims
  have claim↔gold token overlap 23% (22% near-zero) vs 35% (9% near-zero) for hits;
  resolved examiner paragraph often the wrong doc (cart-chair/sling claim's gold is about
  a semiconductor memory device) → V correctly scores low, the LABEL evidence is mis-resolved
  (can't cleanly filter for AUC — circular). (2) §103 combination reasoning. (3) extreme-high-V
  FPs are collapsed dependent blobs (xenc over-fires on a whole-claim fuzzy match), but a
  small enough fraction that the dependent slice still out-discriminates the independent one.
- **Implication:** V is a real but modest verbatim-disclosure detector (§103 1.3× lift),
  strongest on simple limitations, degrading on multi-element combination claims and
  mis-resolved evidence — i.e. the boundary of mechanizable verification vs holistic
  examiner judgment (taste). Cleanest lever to raise it: fix §102 location→paragraph
  resolution (cuts the misaligned-gold misses).

## #61 OA-scale gold-paragraph 102/103 test — pipeline staged, DESIGN FORK (06-12 eve)

Both halves staged + validated (no GPU yet): `scripts/build_gold_mixture_testbed.py`
(unit = (app,ifw,claim); joins on round-4 OACT-bulk extractions: extraction→OACT
100%, app→outcome 100%, anchor→paragraph 100% when doc present) and
`scripts/score_gold_mixture.py` (reuses v6a top-K + xenc + **softmin** aggregation;
LLM element-split pre-pass copied from v2_full_pipeline phase_decompose; dry-run green).
Gate = GP paragraph coverage of round-4 cited docs (cron scraper closing it;
7.3% → climbing; ~4,400 gold-eligible positives projected).

**DESIGN FORK (needs Alex):** examiner gold exists ONLY for rejected claims (the
examiner cites no art against a claim it didn't reject), so every NEGATIVE unit has
`gold=[]`. Consequence for the three conditions:
- **retrieved-only** (v6a paras for both labels) = the only fully-fair discriminative
  condition → this is the honest OA-scale implemented-V number (102/103 separately).
- **gold-only** AUC is undefined (positives only).
- **mixture** lift over control = part real disclosure signal, part mere-presence
  artifact ("has gold" ≈ label) → not a clean classifier.
Clean readout = retrieved-only AUC + a *paired* gold-vs-retrieved metric ON POSITIVES
(the #59 evidence-granularity ceiling, oracle recall 26.7%). To make gold SYMMETRIC,
the population must change to **rejected-art-stuck (abandoned) vs rejected-art-overcome
(later granted)** — both sides rejected, both carry gold, `app_granted` gives the split.
That's a better experiment but a different label. Fair retrieved-only + paired-gold run
needs no decision; the symmetric relabel does.

## Reading (final)

- **§101 is ARTICULABLE** (A=0.78, face-valid eligibility rubrics) — the headline structured finding.
- **§102/103 are A-blind** (~0.50–0.56): "does this read novel" degenerates without external prior
  art — they are **V's domain, and implemented paragraph-level V now delivers** (#61, OA scale):
  retrieved-only 0.62 / **§103 0.635**, and mixing real examiner-cited art lifts §103 to **0.658** —
  on par with the dense model's §103 (0.62) on the common cohort. The relational facts are
  real but expensive to operationalize — and the xenc follow-through ruled out every cheap
  pipeline knob (judge swap free at 280M, 7× retriever data null, wider evidence pool null):
  the ceiling is structural (decomposition/aggregation) or label-intrinsic (§103 combination
  reasoning, examiner discretion). For app-level V, evidence granularity (claim-1 text vs spec
  paragraphs) and the min-of-max element-count bias remain.
- **Gap to dense — REVISED by the common-cohort run**: the 0.74 dense number does not transfer
  to the validity cohort (dense 0.58 there, loading verified at 0.73 on its own pool). On common
  rows the story is per-class, not pooled: A wins §101 (0.78 vs dense 0.67), dense wins §102/103
  (0.62–0.66 vs A 0.53) — the remaining §102/103 margin over implemented V is the
  taste/unimplemented-V residual, and it is ~0.03–0.07 AUC, far smaller than the 0.74-vs-0.53
  framing suggested.
- Methodological exports: stable-hash splits (contamination incident), contamination-controlled
  eval pattern, control-set falsification catching aggregation artifacts.

---

## #62 — Bulk OACT §102/§103 parse COMPLETE (2026-06-15 19:26)

The full OACT bulk snapshot parse finished. Final state:
- Snapshot scanned: 614,566 OAs; §102/§103-bearing subset fully extracted (pending 0).
- Output: `datasets/patents/processed/oa_102_extractions_v2/` — 7 round files, headline round
  `extractions_round_20260613_022658.jsonl` (1.07 GB, the multi-day bulk run).
- **322,277 OA-level extraction records** total (≈4.04M individual §102/§103 citation sub-records).
- Extractor: Llama-3.3-70B BF16, vLLM offline batch, GPU 5, fp8 KV cache, ~1.0 OA/s, 24-way concurrency.
- Cron `oa_extraction_round.sh` (`*/30`) DISABLED (`#PAUSED-2026-06-15-bulk-OA-parse-complete`) so it
  no longer reloads the 70B every 30 min on an empty queue. GPU 5 released (0%, 4 MiB).

This was the last open item of the patents goal-hook (mixture VAT + V/A metrics already delivered in
#60/#61). The larger gold-positive pool (34,952 → now backed by the full 322K-record corpus) is
available if we ever want to scale the #61 mixture test beyond the 12K balanced sample.

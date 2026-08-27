# 2026-07-05 — Wave-2 isomorphism expansion: all remaining domains + calibration + band re-read

Continues `2026-07-04__crosstask-isomorphism.md` (goal thread: iso-morphism between pairs).
User directive: "do 1-5. Expand to all the other domains we have... Other model families, too."

## A. Wave-2 domain passes LAUNCHED (sk3, 2 GPUs)

Completes grids for ALL remaining matrix tasks. Chain per domain: R3 sweep (GLM-4.7 glm_a/b/c
proposers, Llama-8B executor, --text-first, orbit 4, forminv 12) → 75% ckpt gate → day-0 cert
(CPU, background) → decompression grid (grid_run_v2.sh, util 0.25). Driver:
`outputs/wave2_track.sh GPU TASK:SHORT:NGROUPS:RESERVE...`.

| track | GPU | domains (R3 groups) | launched |
|---|---|---|---|
| A | 5 | code-review (32) → legal-outcome-prediction (12) | 2026-07-05, verified live |
| B | 1 | peer-review (13) → grant-funding (16) | same, verified live |

Unlocks: transports for the 3 untested significant pairs (legal×math +5.1, grant×peer +3.7,
math×peer +3.65 — rigor/research-merit families), 9-domain curve table, pooled continuous-ρ.

Wiring (config.py TASK_PRESETS + manifest._FULL_DATASETS):
- `legal-outcome-prediction` = key-bridge of "law" (probes = title_vii facts, 360 loaded).
- `grant-funding` → open-source-grants/grants_labeled.csv.gz — **SMALL: 210 docs** (median 29K
  chars → 4K truncation); --gepa-reserve 10 so probes = texts[10:210] (n=200, others 300).
  Boilerplate scrape headers present in full_text (mild probe pollution; noted, not cleaned).
- `code-review` → code_review_dense_4096tok.csv.gz (sk3-only canonical), 360 texts.
- **Two framing bugs fixed pre-first-sweep** (the cross-task framing guard): peer-review corpus
  text is the PAPER (judgement=accept), preset said "REVIEW" → now PAPER; code-review default
  preset still said "competitive-programming solutions"/SOLUTION → now PULL_REQUEST. Neither
  preset had ever been sweep-exercised, so no prior result is contaminated.
- Deferred: patents (0 R3 groups) + notice-and-comment (5) — hierarchies too thin to sweep;
  need bank→clustering rebuilds first (wave-3 candidates).

## B. Band-mode (ε_form) re-read — ALL 5 cert domains (#20 DONE)

Extended `band_mode_reread.json` to pr/news/math (CW/humor recomputed → MATCH stored values;
same ε_form=0 caveat: day-0 certs carry only the form boolean, so band verdicts are the most
generous-to-CODIFIABLE case; per_metric detail now saved).

| domain | n | old verdicts | band verdicts | exFD → |
|---|---|---|---|---|
| creative-writing | 43 | 36 FD, 7 US | 2 COD, 41 US | 2 COD, 34 US |
| humor | 60 | 36 FD, 19 US, 5 COD | 8 COD, 52 US | 3 COD, 33 US |
| press-releases | 42 | 5 FD, 24 US, 9 DEEP, 4 COD | 4 COD, 10 DEEP, 28 US | 4 US, 1 DEEP |
| news-homepages | 25 | 1 FD, 12 US, 10 COD, 2 DEEP | 10 COD, 2 DEEP, 13 US | 1 US |
| math-SE | 21 | 16 FD, 4 US, 1 DEEP | 2 DEEP, 19 US | 15 US, 1 DEEP |

- **Math replicates the CW pattern**: 15/16 exFD land UNDERSAMPLED — the form cliff masks
  undersampling in expressive AND formal domains; probe count is the bottleneck, not a distinct
  form-dominated population. The "math 16/21 FD" raw quote is retired.
- **Scope-degeneracy census** (H_M<0.1): news 13/25, PR 6/42, humor 4/60, CW 0/43, math 0/21.
  PR prediction confirmed: its 4 band-CODIFIABLE are ALL degenerate apparatus criteria.
- ★ **Genuine verdicts (band + H_M≥0.1) INVERT by domain type**: CODIFIABLE only in expressive
  domains (CW 2: dialogue craft, setting-as-force; humor 7: game identification, callbacks,
  wit...); DEEP only in institutional/technical domains (PR 10, news 2, math 2: stepwise
  logical validity, reproducibility, explanatory journalism...). Expressive articulation
  resolves to short certified rubrics or stays undersampled; institutional/technical value has
  measured heavy tails. (DEEP still pending §12.6.7 recapture control — report as band-DEEP.)
  The 14 band-DEEP metrics = the target list for the GEPA rung (#14).

## C. Calibration judging (mid-z pairs) — DONE (#19)

Fills the z-gap between the z≈0.2 control and the z≥3.65 winners: code-review×peer-review
z=+1.17 (165 cands), grant×math z=+0.96 (123), humor×news z=+0.56 (324); same cascade (k=5
reciprocal TF-IDF on hierarchy-R3 texts → independent Sonnet judges, substitution 0/1/2 →
manual adjudication). Files: `crosstask/CALIB_*_{candidates,judge1..3}.json`,
`CALIB_midz_judged.json`, anchors `CALIB_anchor_{candidates,TRUTH,judge*}.json`.

**Anchor severity test (new protocol step, keep it):** 90 blinded pairs (44 known-SAME + 46
known-0 sampled from the verified winners) scored by every judge pass. It CAUGHT a bad
instrument: judge-1's *resumed* pass was degenerate (2/44 ≥1 on known-SAMEs, fewer RELATEDs on
anchors than on mid-z — an inconsistent standard), so judge-1 is excluded from SAME-level
claims (its original engaged pass kept for the RELATED gradient only). Validated instruments:
judge-2 (68% ≥1 recall on known-SAMEs, 0 false-2s) + fresh judge-3 (32% ≥1 recall, 0
false-2s) — both 100% SAME-precision, one severity notch stricter than the 07-04 batch (2-level
recalls 7%/16%), so ABSOLUTE rates are not comparable across batches; anchors make them
comparable.

**Result: 0 SAME / 612 across all three mid-z pairs.** All 4 single-judge 2s adjudicated to
RELATED: grant×math 6:0 (umbrella audience-targeting ⊃ non-specialist accessibility);
humor×news 18:3 (evidence+integrity compound ⊃ verification standards); humor×news 32:13 +
55:13 ("originality" lexical twins — humor = creative novelty vs news = newsgathering
provenance/anti-churnalism; compound-vs-strand).

| pair | z | n | SAME | both-validated ≥1 | ≥1 rate j1/j2/j3 (%) |
|---|---|---|---|---|---|
| code-review × peer-review | +1.17 | 165 | 0 | 4 (2.4%) | 19.4 / 12.1 / 3.6 |
| grant-funding × math-SE | +0.96 | 123 | 0 | 0 | 11.4 / 9.8 / 1.6 |
| humor × news-homepages | +0.56 | 324 | 0 | 2 (0.6%) | 4.3 / 8.3 / 1.2 |

- **RELATED density is z-monotone for all three judges independently** — the screen's z tracks
  shared-criteria density even below significance.
- **The non-significant band is SAME-free** under instruments that recover known matches with
  100% precision; mid-z candidates are 6–27× depleted in ≥1 verdicts vs known-SAMEs, and even
  BELOW the known-0 candidates from significant pairs (j2 ≥1: 22% on winners' non-matches vs
  8–12% mid-z) — significant pairs share denser vocabulary even off-match (the universal
  layer, again).
- Sporadic near-matches at low z are universal-layer strands (verification, originality) —
  never domain-layer. Matches the control's behavior.

**Protocol lesson (memorialized):** resumed subagents can produce degenerate follow-up
annotation passes; anchor-test EVERY pass, and never compare match rates across judging
batches without a shared anchor set.

## D. Wave-2 HARVEST (same day — all four sweeps landed by 11:58 PT)

All 4 sweeps 100% ckpts (cr 32, peer 13, legal 12, grant 16), 4 day-0 certs, 3/4 grids clean.
Failures found by verification (never trust exit codes / file-exists):
- **grant grid IndexError**: grid loaded probes with default reserve=60 (150 texts) while the
  sweep's exemplar indices refer to the reserve=10 window (200) → relaunched `--gepa-reserve 10`.
  LESSON: grid probe window must MATCH the sweep's.
- **Qwen panel raced its own teardown**: CW report.json = `{}` (score-phase engine died; report
  phase wrote empty; `-f report.json` success gate too weak), math got 1.5B only. Relaunched
  one-reader-per-process with free-mem gates + size-checked success (`wave2_fix.sh`).

**Face-2 (isomorphism_census_wave2.json + alldomains merge):**
- code-review (n=32): definition-peak (+.059 def−name, 21/32 best=definition) — jargon-rich but
  does NOT invert.
- peer-review (n=12): definition-peak (+.072).
- **legal (n=6 census-valid of 12): NAME-PEAK (−.058) — the math inversion REPLICATES in a
  second formalized-lexicon domain.** Overall recovery low (≤.17) — legal is hard for 1B/3B.
- 8-domain lexicalization gradient: humor .149 > news .103 > PR .082 > peer .072 > cr .059 >
  CW .051 > **math −.055 ≈ legal −.058**.

**Face-1 (band + H_M≥0.1, band_mode_reread.json now 9 domains):**
- Genuine DEEP stays EXCLUSIVELY institutional/technical: PR 10, cr 7 (naming, why-comments,
  atomic changes — the software-craft canon), legal 5 (BLUF, controlling theme, decision-maker
  register), news/math/peer 2 each, grant 1; still 0 in CW/humor.
- Genuine COD: CW 2 + humor 7 + grant 2 (grantsmanship clarity, partnership plan) — the one
  crack in "COD=expressive"; 13 total. Band-DEEP target list for GEPA (#14): **34 metrics**.
- Scope-degeneracy: cr 4/32, peer 4/13, legal 4/12, grant 3/16 (moderate everywhere except
  news's 13/25).

**Concept tags (concept_tags_wave2.json, 73 metrics):** 2 Sonnet taggers κ=.737 (78/88), 10
adjudicated; 15 blind anchors 10/15 both (2/2 MECH, 2/3 TASTE; misses = systematic craft→MECH
boundary shift vs the 07-03 batch — cross-batch label comparisons carry this caveat). legal
12/12 CRAFT; cr 27C/5M; grant 11C/5M; peer 7C/5M/1T. MECHANICAL cell now 28/340 metrics
across 9 domains.

**Transports (with tags):** legal×math 6/6 label-agree but **p_perm=1.0 — degenerate marginals**
(both sides ~all-craft; agreement carries no information); math×peer 2/3 n.s. Directionally
consistent (8/9) but unpowered — the rigor family shares ONE concept type, so type-transport is
only testable on type-diverse pairs. grant×peer (5M+11C grant side) = next powered test, pending
the repaired grant grid. Powered evidence remains CW×humor 25/30 + news×PR 11/11.

**Notebook (user-requested):** `notebooks/2026-07-05__crosstask-isomorphism-results.ipynb` —
7 sections, 6 figures, executes clean from the durable artifacts (screen, calibration+anchors,
layers, transport, 8-domain curves+gradient, 9-domain verdict inversion, roadmap).

## E. Model families (Stage D opening)

Per user ("Other model families, too?"): second family = REPLICATION PANEL, never pooled with
Llama (same-family-scaling doctrine). Plan: second-family reader ladder on the two pole
domains (CW definition-peak vs math name-peak) to test the lexicalization gradient
cross-family; then Gemma-4-31B family-top anchor for iso-performance chains (#12). Model
availability recon on sk3 pending (below).

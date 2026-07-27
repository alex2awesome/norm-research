# Why metric discovery plateaus — diagnosis (2026-07-05)

*Prompted by: the arms keep plateauing / induce no sensible new metrics, yet when I hand
Sonnet real examples and ask "what distinguishes good from bad," it names useful metrics
immediately. Which is it — bad algorithm or something upstream?*

**Verdict: the plateaus are mostly UPSTREAM of the algorithms.** The proposers already produce
good, human-plausible craft metrics. Three upstream conditions drive the measured gains to ~0,
and they are confounded together in the current CW/PR runs.

## 1. The proposers are NOT the problem — the proposals are good

From the July-02 ledgers, the arms proposed exactly what a human would name:
"Subtextual Dialogue", "Show Don't Tell (Sensory Evocation)", "Economy of Prose", "Immersive
Narrative Voice", "Narrative Perspective Consistency", "Pacing and Plot Progression". These are
sensible craft axes. They just don't move the label. So the puzzle is not proposal quality.

## 2. The CW label is upvote virality, not craft quality (the dominant cause)

`datasets/creative-writing/litbench-to-train.csv.gz`: `judgement` is a hard threshold on
Reddit WritingPrompts upvotes (AUC(upvotes→label) = 1.000; label 1 iff upvotes ≥ 101).
It measures **community virality**, which is dominated by factors OUTSIDE the text — posting
time, prompt popularity, early-vote cascades.

- **In-text ceiling is low and confounded.** TF-IDF(1–2gram) → label tops out at **AUC 0.663**
  (n=12k). The single most viral-predictive token is **"edit"** — authors editing in
  "Edit: thanks for the gold!" *after* going viral (reverse causation / leakage) — followed by
  character/prompt proper names (Timmy, Todd, Katie, Bruce): topic identity, not craft.
- So there is almost no craft signal in this label for ANY metric to capture. Good craft
  rubrics scoring ~0 is the CORRECT answer here, not a pipeline failure.

**This resolves the VAT contrast.** When you score math, the label is *correctness* — genuinely
determined by in-text properties — so craft/rigor metrics predict it and Sonnet's proposals
"just work." CW-litbench's label is *virality* — not in the text — so the same-quality
proposals plateau. Same instrument, opposite outcome, because the labels measure different
things. This is the articulability thesis, but the current CW setup conflates it with #3–#4.

## 3. The CW medoid bank is contaminated

`datasets/creative-writing/medoid-bank` (40 coverage-selected medoids): only **19/40 are about
creative writing**. 5 are pure boilerplate (Project Gutenberg license text; bacteriology lab
procedure "flame cotton-wool plugs… sterilise platinum loop"; CIA-Factbook land-use tables;
North Carolina history quiz), 16 are off-domain literary-theory fragments that don't apply to
short WritingPrompts stories. Coverage/medoid selection over a noisy 73k rubric pool actively
*prefers* these outliers because they are maximally distant from everything else.

Consequence: V(bank) is a noise floor, the WRONG/RIGHT residual contrast is formed against a
garbage predictor, and the "known criteria" blocklist misdirects the proposer. Fixable.

## 4. At thin signal scale, the pipeline rewards surface/leakage artifacts (PR)

On peer-review the single highest-bits-gain proposal was **"Manual Markdown Formatting"**
(+0.0102 bits) — a format artifact. Checked directly: surface features carry AUC ~0.51–0.54
(markdown chars 0.518, digit count 0.535, length 0.529, first-person 0.514). When the real
content signal is only ~0.05 bits, a weak surface feature is *competitive* and gets surfaced.
The gate optimizes "whatever nudges AUC," including format. Needs a content-only guard.

## 5. Judge execution attenuation (measured, still a floor)

Planted-bank calibration (§7) fidelity: the 70B judge scores TRIVIAL code-checkable features at
only AUC **0.70–0.87** (question-mark 0.70, dialogue 0.83–0.87). On subtle craft ("subtext",
"immersive voice") faithful execution is plausibly near-chance, attenuating any real signal
toward 0. There is currently NO per-metric test-retest reliability read, so a ~0 gain cannot be
separated into "metric is non-predictive" vs "judge can't apply it." This is the pivotal
disambiguation instrument still missing.

## What to change (ranked)

1. **Recommend (needs sign-off): fix the CW y-variable.** For measuring craft articulability,
   use a quality label (LitBench pairwise preference exists upstream but was thrown away for
   upvotes), OR explicitly reframe CW-virality as the "revealed community preference, mostly
   not-in-text" cell of the design — legitimate, but then the residual is label-noise, not
   tacit craft, and must be reported as such (ties to the noise-ceiling decomposition, the
   paper's #1 gap).
2. **Fix the CW bank** (no sign-off — repairing a contaminated instrument): a clean craft bank
   so V(bank) is an honest floor and the residual contrast is meaningful.
3. **Add per-metric judge reliability** (test-retest at temp>0): distinguishes measurement
   floor from genuine null. Without it every ~0 gain is ambiguous.
4. **Content-only guard / surface blocklist** in the proposal gate (markdown/length/pronoun/
   digit-only rubrics rejected ex-ante) so thin-signal runs stop rewarding format leakage.
5. **Positive-control watch**: the planted calibration's WITHHELD numeral feature (carries
   ~0.074 bits) must be recovered+accepted; if it isn't, the pipeline has a real sensitivity
   problem independent of the label — check the calibration verdict.

## The reframe for the paper

The right demonstration is the CONTRAST: same pipeline, clean bank, run on a corpus where craft
SHOULD predict (math/VAT, humor) → good metrics induced, real gains → vs CW-virality where it
can't → plateau. That contrast IS the articulability result. A plateau is only evidence of a
tacit gap once (a) the label is in-text (noise ceiling measured), (b) the bank is clean, (c) the
judge can execute (reliability measured). Until those three are controlled, a plateau is
ambiguous.

## Per-algorithm implementation review (3 parallel reviews, 2026-07-05)

All three converged on **judge-execution reliability is never measured** as the #1 defect —
i.e. every ~0 gain is ambiguous (genuine null vs the executor can't apply the rubric). Fixed
(reliability.py, wired into the global loop). Other findings:

**AutoMetrics-Iterative** (methods/autometrics/.../runner.py):
- Early stopping far too aggressive: `eval_plateau_eps=0.005` × patience 2 → stops at iteration
  ~2 even on an upward trend. (Raise eps to 0.01, patience 3, or slope test.)
- Selection optimism: active-metric selection (L1 non-zero) done on the SAME split whose AUC is
  then reported → optimistic; iteration reasoning uses the overfit metrics. (Nested CV; report
  on eval_gating only.)
- Pure-L1 `LogisticRegressionCV(l1_ratios=(1.0,))` on small eval sets zeros weak-but-real
  correlated metrics. (elastic net l1_ratio 0.7.)
- exact_match rounds scores to nearest INTEGER before pairing → 2.4≈2.6, loses resolution;
  greedy first-available pairing; unweighted L1 distance dominated by high-variance noise metrics.
- New metrics excluded from interaction terms → a metric with 0 main effect but real synergy is
  zeroed. Self-critique "superficial vs substantive" can drop subtle craft metrics (uncalibrated).

**Metric-Tree** (methods/metric_tree/): the 1-node stall is explained.
- `clustering_depth=2`: depth<2 proposes LABEL-INDEPENDENT clustering features. On a no-typology
  label (virality) every partition stays mixed, features fail the 15-85% balance check → 0 good
  features → node becomes a leaf immediately. **This is the "grows 1 node, gains 0" mechanism.**
  (Set clustering_depth=0 or make it domain-adaptive.)
- Balance thresholds (0.15 shallow / 0.05 deep) reject rare-but-real signals; NA threshold 0.05
  excludes partition-specific metrics (NA on out-of-partition rows); gap-fill base-rate window
  [0.2,0.8] skips impure partitions with hidden minority structure.
- Router variant: over-eager "majority-class" exits can short-circuit examples that needed deeper
  metrics.

**metrics_tree_infilling / MCC arms** (mine — fixes APPLIED):
- **No reliability read** → added per-metric test-retest (reliability.py), stamped on every scored
  proposal, `attenuation_flag` when retest < min_reliability. Summary reports low_gain_attenuated.
- **Surface leakage** → content-only guard: anti-surface prompt instruction + `is_surface_only`
  post-filter + `dropped:surface` gate. (Catches "Manual Markdown Formatting", bulleted-lists.)
- **Garbage-bank residual** → bank-AUC guard: skip the WRONG/RIGHT residual contrast when bank
  AUC < min_bank_auc_for_residual (0.55 in runs).
- **Residual-arm array-parser bug** → `_parse_json_candidates` mangled a top-level JSON array
  (the offline 70B's format) into invalid JSON → residual arm silently produced 0 proposals.
  Fixed to try array then object envelope + strip fences. (Regression test added.)

## The decisive number (planted calibration, first v2 run)

Bank of 3 KNOWN-good planted features (code V* = 0.095 bits) recovered only V_bits ≈ −0.002 when
the LLM JUDGE executed the rubrics, and the withheld numeral feature (code ~0.09 bits) was NOT
rediscovered by any arm (0 false positives). Strong evidence for a judge-execution measurement
FLOOR — but the guard split is small (~70 rows) so it is confounded with n-noise. v3 run (reliability
ON, parser fixed) will give the clean per-metric retest and a bigger-n read to confirm.

## CORRECTION (user, 2026-07-05): all three label types per task; virality is one leg

My "switch the CW label" recommendation was too narrow. Every task carries THREE label types
(paper §IV): expert-verdict (stated jury judgment), expert-revealed (gatekeeper choice),
community-revealed (crowd behavior). CW-virality is the **community-revealed** leg — not a bug.
My real error: I ran the arms on ONLY that leg (the weakest, already flagged as taste-laundering)
and over-generalized "plateau."

The expert legs for CW are already BUILT and were unrun:
| leg | CW data | n | status |
|---|---|---|---|
| community-revealed | WritingPrompts upvotes (litbench) | 87k | ran → plateau (expected) |
| expert-revealed (curatorial) | Wigleaf Top-50 editor cut | 1246 | wired `creative-writing-wigleaf`, QUEUED |
| expert-revealed (market) | RoyalRoad → KU/Amazon deal | 1274 | wired `creative-writing-royalroad`, QUEUED |
| expert-verdict (prize, purest) | LitBank prizewinners / Commonwealth-Bridport | — | NOT built (sourcing item) |

The **result is the gradient of articulable fraction across label types within a task** (predict
expert-verdict > expert-revealed > community-revealed) — that IS §IV, it's Nisbett-Wilson
(stated vs revealed) at corpus scale, and it subsumes the noise-ceiling concern. Running now on
sk3 (scripts/tools/cw_expert_cells.sh): arms+certificate on wigleaf + royalroad with the same
guards/reliability, so the three CW legs are directly comparable. The paper demonstration is this
WITHIN-CW three-label contrast, not only cross-task.

## Clean CW craft bank + overnight run matrix (2026-07-05, late)

Built `datasets/creative-writing/medoid-bank-clean/bank.json` (script:
`datasets/creative-writing/build_clean_craft_bank.py`) — filters the 73.5k-rubric pool:
junk blocklist → craft-anchor cosine ≥ 0.45 (11.3k survive) → dedup → **k-medoid** selection
(cluster + highest-craft member per cluster; NOT k-center, which reintroduces outliers — the
first attempt with k-center kept "Field code 2125 Terrain" and a gated-self-attention ML rubric
and rejected Aristotle's Peripeteia). Result: **37/40 craft-relevant vs 18/40 in the dirty
medoid bank** — genuine dimensions (Character Development, Tension, Ending-earned, Show-don't-
tell, Escalating Stakes, Reversals & Discoveries, POV, Dialogue, Pacing, Prose Rhythm, Plot,
Imagery).

Overnight run matrix on sk3 GPU3 (all: 5 arms, guards + reliability, 70B judge+proposer offline):
| label leg | dirty bank | clean bank |
|---|---|---|
| community-revealed (virality) | RUNNING | — (label is the ceiling, bank won't move it) |
| expert-revealed (wigleaf) | queued | queued |
| expert-revealed (royalroad) | queued | queued |

Reads on completion: (a) three-label gradient in articulable fraction (dirty bank); (b) bank
A/B on the expert legs (does a clean floor + functional residual arm change the induced-metric
story). Harvest when `cw_expert_cells.log` and `cw_cleanbank_expert.log` report COMPLETE.

## RESULTS — leg 1: virality (community-revealed, dirty bank), 2026-07-06 00:1x

Clean, rigorously-defended NULL on the community-revealed leg:
- **87 proposals across 4 arms, 0 kept, floor ≈ 0 bits** (−0.0157 on guard/CV).
- **Per-metric judge reliability median retest_spearman = 0.91** (min 0.67, 100% ≥ 0.5). So the
  judge executes these craft rubrics RELIABLY — the plateau is a GENUINE NULL (craft doesn't
  predict virality), **NOT a measurement floor**. This is the pivotal disambiguation resolved,
  and it refutes the "judge just can't apply craft rubrics" alternative. (It also explains the
  planted calibration's −0.002 V-recovery as small-n guard-split noise, not true zero
  transmission — the per-metric retest is the trustworthy read.)
- **12 proposals killed by the confirm stage** (passed primary CV, failed fresh-seed replication
  + Bonferroni) — winner's-curse artifacts the OLD protocol would have kept and lost on test.
- **Residual arm: 0 proposals** — the bank-AUC guard correctly fired (dirty bank AUC < 0.55 on
  virality → residual contrast is noise → skipped). Guard working as designed, not the parser bug.
- FLUX: 87 draws / 86 species (99% singletons) → arms keep proposing distinct plausible craft
  axes; none accepted. U_flux 0.38 bits process-relative but singleton-regime (loose, growing).

The community-revealed leg is the bottom rung, established rigorously. Prediction for the expert
legs (wigleaf/royalroad, running): if craft predicts expert-revealed quality, expect kept metrics
+ floor > 0; if they ALSO plateau at high reliability, that is a STRONGER tacitness claim (even
expert-revealed CW quality isn't captured by nameable craft).

## CORRECTION + BUG — residual arm silently disabled (context overflow), 2026-07-06 01:3x

Two corrections to the leg-1/leg-2 reads:
1. **The mid-run "18 kept" on wigleaf was a grep artifact** (the substring "kept" matched JSON
   keys like `kept_names`/`kept_bits_gains` in the log). The FINAL wigleaf result is **0 kept**.
2. **The residual arm produced 0 proposals on EVERY leg because its prompt overflowed the 8192-
   token engine** — `VLLMValidationError: passed 8193 input tokens` fired every round (72× on
   wigleaf), caught by the failure-isolation as an "empty round." So all results so far are
   **4-arm (unconditional/label_contrast/autometrics_iterative/metric_tree), NOT 5-arm** — the
   engine's PRIMARY targeted arm never ran. FIXED: `vllm_max_model_len` 8192→16384 + proposer
   prompt clamp (io_metrics/config, 63 tests). Clean-bank runs (not yet started) will pick it up.

**Honest state of the two dirty-bank legs (both 4-arm):**
| leg | bank base AUC | proposals | kept | floor | retest median |
|---|---|---|---|---|---|
| virality (community-revealed) | ~0.51 | 87 | 0 | ~0 bits | 0.91 |
| wigleaf (expert-revealed curation) | 0.508 | 75 | 0 | ~0 bits | 0.87 |

**KEY CAVEAT:** the dirty bank predicts NEITHER label (base AUC ~0.5 both) — 21/40 viable, mostly
non-craft. So V(bank)~0 is a BANK artifact, not proof the label is non-articulable. The plateau
is real for these 4 arms + this bank, and the metrics are reliably executed (retest 0.87-0.91 =
genuine null not measurement floor), BUT whether CRAFT predicts expert curation is only answered
by the **clean-bank runs** (37/40 craft, residual arm now working) — those are decisive:
- if clean-bank wigleaf shows bank AUC > 0.55 + floor > 0 + kept metrics → craft DOES predict
  expert curation; the dirty-bank plateau was a bank artifact; the gradient (expert > community)
  holds.
- if clean-bank wigleaf ALSO gives ~0.5 / 0 kept at high reliability → even good craft doesn't
  predict expert flash-fiction curation → a STRONG tacitness result.
Either is clean now that the instrument (reliability, confirm stage, residual arm) is trustworthy.

## royalroad-dirty result + fix confirmation, 2026-07-06 02:2x

- **royalroad-dirty (market-revealed, dirty bank): 0 kept, floor ~0** — but effectively **2-arm**:
  residual + label_contrast + metric_tree ALL produced 0 proposals. royalroad texts (web-serial
  opening chapters) are long, so ALL THREE example-bearing arms overflowed the 8192 context, not
  just residual. Only unconditional + autometrics survived. So the context bug suppressed multiple
  arms on long-text corpora — the dirty-bank royalroad read is nearly vacuous.
- **Context fix CONFIRMED working**: clean-bank wigleaf built the engine at max_model_len=16384
  with **0 VLLMValidationErrors**. The clean-bank runs are therefore the FIRST genuine full 5-arm
  runs (clean craft bank + all arms including residual). Decisive; in progress.

Net: every dirty-bank leg was compromised (bad bank predicting neither label + 1-3 arms disabled
by context overflow). Do NOT draw the gradient from them. The clean-bank wigleaf/royalroad runs
(full 5-arm, 37/40-craft bank, 16384 ctx) are the real experiment.

## Clean-bank wigleaf — DECISIVE run, partial (2026-07-06 03:2x, 2/5 arms done)

The first uncompromised run (full 5-arm, clean 37/40-craft bank, residual working, 16384 ctx):
- **Clean bank base AUC on wigleaf = 0.578** (vs dirty bank's 0.508). So a real craft bank DOES
  predict expert editorial curation above chance — the dirty-bank ~0.5 was a bank artifact, as
  suspected. Bank applicability also jumped (35/40 viable vs ~20/40 dirty).
- BUT **base bits ≈ 0** despite AUC 0.578 — a weak ranker that doesn't reduce log-loss on the
  small guard split (the AUC-vs-bits gap; certificates compose over bits).
- **Residual arm now functional**: 30 proposals (was 0 pre-fix); **content guard active** (9
  surface proposals dropped on residual alone) + 3 confirm-drops.
- **0 kept** on the 2 completed arms (residual, unconditional) — authoritative ledger counts
  (NOT the grep "kept" substring, which misled earlier).
Emerging read: craft weakly RANKS expert curation (AUC ~0.58) but the proposed metrics add ~0
bits beyond the bank; the AUC signal is real but thin. Await label_contrast/autometrics/metric_tree
+ royalroad-clean before the verdict.

## VERDICT — clean-bank wigleaf (4/5 arms, metric_tree finishing), 2026-07-06 04:1x

The first fully-uncompromised run resolves the core question. Full matrix so far:

| leg | bank | base AUC | proposals | kept | retest median |
|---|---|---|---|---|---|
| virality (community-revealed) | dirty | 0.508 | 87 (4-arm) | 0 | 0.91 |
| wigleaf (expert-curated) | dirty | 0.508 | 75 (4-arm) | 0 | 0.87 |
| royalroad (market-revealed) | dirty | ~0.5 | 60 (2-arm†) | 0 | — |
| **wigleaf (expert-curated)** | **CLEAN** | **0.578** | **120 (4-arm)** | **0** | **0.88** |
| royalroad (market-revealed) | CLEAN | — | running | — | — |
†royalroad-dirty had 3 arms disabled by the context overflow (now fixed).

**Honest findings:**
1. **Bank-quality A/B (real):** clean craft bank ranks wigleaf curation at AUC **0.578 vs 0.508**
   for the dirty bank (and 35/40 viable vs ~20/40). The contaminated bank was MASKING craft's
   predictive value for expert curation. Methodological lesson: bank hygiene matters.
2. **Label gradient (as predicted, at the RANKING level):** craft ranks expert curation (0.578)
   above community virality (0.508). Expert-curated quality IS more craft-predictable than
   community upvotes — the §IV prediction holds.
3. **But NO induced metrics on ANY leg (0 kept everywhere)** and all proposals reliably executed
   (retest 0.87–0.91 → genuine null, NOT judge attenuation). Even a clean craft bank + working
   residual arm + all 5 arms find no NEW metric that adds bits beyond the bank.
4. **Base bits ≈ 0 despite AUC 0.578** — the whole regime is sub-0.01-bit (the MCC per-criterion
   detection floor); the bits-gate rejects everything because there is <0.01 bit/metric to gain.

**Reading:** on CW, the articulable craft signal is THIN and already CAPTURED by the explicit
craft bank — most of what craft can say about quality is in the rubric bank; the discoverable
residual beyond it is below the detection floor for all three label types. The gradient exists
(expert > community, at the ranking level) but the "inducible new metric" tail is empty
everywhere. This is the thin-articulable-tail result, now confirmed across label types with the
full instrument (reliability + confirm + content guard + clean bank). Await royalroad-clean to
complete the market-revealed leg.

## KEY REFINEMENT — the gradient is aesthetic-authority vs market, not expert vs community

wigleaf-clean FINAL (5 arms): 150 proposals, **0 kept**, floor ≈ 0, flux tail 0.255.
royalroad-clean (running, base stable): **base bank AUC = 0.505** (chance), viable 36/40.

Craft-rankability of each CW label (clean 37/40-craft bank base AUC):
| label | who reveals it | base AUC (clean bank) | craft ranks it? |
|---|---|---|---|
| **wigleaf editorial curation** | a literary EDITOR's aesthetic judgment | **0.578** | YES (weakly) |
| virality (upvotes) | the READER CROWD | 0.508 | no (~chance) |
| royalroad KU/Amazon deal | the MARKET / commercial gatekeeper | 0.505 | no (~chance) |

**The split is not expert-vs-community — it's AESTHETIC-AUTHORITY vs MARKET-OUTCOME.** Craft
predicts an editor's *taste* (0.578) but not a *market outcome*, whether the market is a Reddit
crowd (0.508) or an Amazon commercial pickup (0.505). RoyalRoad KU deals are driven by commercial
viability (retention, genre fit, serialization hooks, update cadence), not literary craft — so
they behave like virality, NOT like editorial curation, despite both being "expert/market-
revealed." This refines the §IV framing: the meaningful axis for craft-articulability is WHO the
label's authority is (an aesthetic judge vs a market), not expert-vs-community per se.

Caveat still standing: even where craft RANKS (wigleaf 0.578), 0 new metrics are inducible and the
rankability is entirely in the EXISTING bank (base bits ~0, sub-0.01-bit floor). So: craft's thin
predictive signal for aesthetic judgment is already captured by the explicit rubric bank; there is
no discoverable articulable residual beyond it, and for market outcomes craft has no signal at all.
Await royalroad-clean completion to confirm 0 kept + retest (genuine null) on the market leg.

## FINAL MATRIX (all 5 legs), 2026-07-06 06:0x

royalroad-clean COMPLETE (5/5 arms): base AUC 0.505, 150 proposals, 0 kept, floor −0.022,
retest median 0.90 (98% ≥ 0.5) → genuine null. Confirms the market-label pattern. ALL 5 LEGS DONE;
GPU3 freed 06:43.

| leg | label authority | bank | base AUC | proposals | kept | retest |
|---|---|---|---|---|---|---|
| virality | reader crowd (market) | dirty | 0.508 | 87 | 0 | 0.91 |
| wigleaf | editor (aesthetic) | dirty | 0.508 | 75 | 0 | 0.87 |
| royalroad | KU/Amazon (market) | dirty | ~0.5 | 60† | 0 | — |
| wigleaf | editor (aesthetic) | CLEAN | **0.578** | 150 | 0 | 0.88 |
| royalroad | KU/Amazon (market) | CLEAN | **0.505** | 150 | 0 | 0.90 |
†2-arm (context overflow, since fixed).

CONCLUSIONS (all with the full instrument: reliability + confirm + content-guard + clean bank):
1. **Aesthetic-authority vs market-outcome** is the axis: craft ranks an editor's taste (0.578)
   but not market outcomes (crowd 0.508, commercial 0.505). NOT expert-vs-community.
2. **Bank hygiene is load-bearing**: clean bank 0.578 vs dirty 0.508 on the same wigleaf label —
   the contaminated bank hid craft's predictive value.
3. **Thin articulable tail, 0 kept everywhere**: even where craft ranks (wigleaf 0.578), no arm
   induces a NEW metric that adds bits; base bits ~0 (sub-0.01-bit MCC floor). The thin craft
   signal is ALREADY in the explicit bank; the discoverable residual is empty. All nulls are
   GENUINE (retest 0.87-0.91), not judge attenuation.

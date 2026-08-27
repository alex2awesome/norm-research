# Running Research Notes

Last rewritten 2026-06-05; last updated 2026-07-03 (two-faces paper push — see block below). The pre-2026-04 versions of this file were dataset-only;
this rewrite reorganizes around the current Verifiability + Articulability + Taste
framing, per-task modeling state, active method tracks, and open problems.

Each substantive claim cites the auto-memory file (under
`~/.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/`)
or the dated `notes/` file that backs it.

---

## 0a¹⁵. 2026-07-23 (eve) — PREREG'D humor four-universe readout MISSES its band; math levers land below band; the composition rule

**Humor R3 = the confirmatory pass, and it did not clear its own bar.** Prereg (`notes/2026-07-23__humor-r3-prereg.md`,
written and frozen before scoring; block `HUMOR_R3_PREREG_AND_SCORING_LAUNCH_20260723`) declared the
detectable stratum (auc≥.55) the headline in advance, with four predictions and a decision rule. Four
universes (old 5,343 / R1 7,996 / R2 8,702 / **R3 17,306**), ypos_v6 = 49,924 pairs / 190 metrics,
2,053 full-coverage channels, 4 GPUs × ~9h. Scorecard:

| prediction | outcome |
|---|---|
| detectable raw ∈ [.37,.45] | **MISS** — +0.344 |
| stratum rel_auc > .643 | HIT — **.727** |
| detectable corrected ∈ [.48,.58], factor ≤1.45 | **MISS** on level (.446), HIT on factor (1.30) |
| low tail < .10 | HIT — +0.074 |

Decision rule (predictions 1+3 must hold) → **NOT MET: humor does not enter the cross-task table as an
in-band result, and no further humor GPU spend.** Full strata: pooled-all +0.130 p=.0002 (n=106,
rel_auc .640), multi-universe +0.139 (n=101), triple-plus +0.158 (n=85, rel_auc .715), quad-support
+0.297 (n=16), **detectable +0.344 p=.0002 (n=21, rel_auc .727, corrected .446)**, low +0.074 (n=85).

**The methodological result is worth more than the number.** Humor's corrected estimate has now fallen
monotonically as its reliability got better-measured: **.709 (thin rel_auc .323) → .536 (3-universe
.643) → .446 (4-universe .727)**. Every increment of real data pulled corrected toward raw. This is
direct evidence that every remaining thin-rel_auc corrected value in the table — crx .581 (rel_auc .339),
pr .497 (.212), cw .82 (.48) — is inflated by the same mechanism and would shrink under equivalent
expansion. Humor is now the campaign's best-instrumented row (correction factor 1.30) and should be
read as the calibration standard, not the weakest entry. Also note the detectable stratum did NOT grow
(22→21 metrics): more data made AUC estimates sharper, tightening membership rather than expanding it —
the predicted 40–60 was wrong about the mechanism.

**Math: both levers pulled, both honest, still below band.** (a) Stratified readout of existing labels:
detectable n=5 pools **+0.628** vs low tail −0.016 — math obeys the humor concentration law. (b) Menu-wide
quote-gated GLM-5 direct annotation of all 2,056 docs (zero GPU; existing g4 scores re-read): 1,037 pos
pairs / 25 metrics ≥10-pos. gpt-5.6-sol blind audit (120 evidence-matched items, sealed key): **agreed
.85 precision / direct-only .625 / census-only .40** (a quarter of census-only positives are actually
NEG-state) → union **not defensible**, census y retired, `math_ypos_direct.json` canonical. Result
**+0.233 p=.0002 / corrected .431** (was +.181/.378).

**★ THE COMPOSITION RULE (third instance today).** On the 5 metrics eligible under all three label sets,
census +0.344 ≈ direct +0.345 — the headline gain is **metric composition (19→22 eligible metrics), not
per-metric label improvement**. Purity hint (agreed-only +0.446, 3/5 matched wins) is suggestive but
n=5 and not decisive; do NOT claim purity>quantity for math (peer found the opposite). **Standing rule:
re-run any pooled delta on a matched metric set before crediting the intervention** — today it caught
humor's dilution, the stratum conflation, and this. Blocks `HUMOR_R3_*`, `MATH_STRATIFIED_STATE_*`,
`MATH_DIRECT_ANNOTATION_*`, `MATH_AUDIT_INFORMED_20260723`.

**Infra:** codex `--background` handoffs can vanish silently (registry empty, no files); use `--wait` for
audits that gate a decision. Scorer wrappers now check exit codes (caught two GPU-collision deaths at launch).

## 0a¹⁴. 2026-07-23 — crx first honest ρ (+.284/.581), humor rel_auc convergence (corrections were inflated)

**crx, final, all 5,943 PRs:** first honest pos-state ρ = **+0.284, p=.0002, corrected .581** (strict
on-page R1 labels). The "structurally excluded"/"mention-mixture null" history was 100% instrument —
crx is a real, significant coupling, ≈ humor's detectable core. Neg channel ~null (+.050), descriptive.

**Humor three-universe readout (old 5,343 / R1 7,996 / R2 8,702; ypos_v5 = 22,567 pairs / 189 metrics).**
The data expansion nearly doubled core rel_auc (.323 → **.629**) — and with a trustworthy rel_auc, raw
and corrected COLLAPSED TOGETHER: multi-universe core **+0.158 raw / .220 corrected** (n=85). The old
[.364, .709] bracket's corrected upper end does NOT survive better instrumentation — vindicating the
peer-probe-pilot warning that √-corrections were upper-style. What DOES hold is the stratification:
triple-support-16 (strongest core, all 3 universes) **+0.341**, detectable-stratum (auc≥.55) **+0.388**,
low-detectability tail (n=79) +0.057. Honest humor = a detectable subset at ~.34–.39 raw (stable across
3 universes × 3 label rebuilds) plus a large near-zero weak tail; pooling gives ~.13–.16.

**Standing methods correction:** corrected columns divide by √(measured reliabilities); when rel_auc is
thin (few positives), the correction inflates. Report raw as the floor and corrected only with its
rel_auc stated; the trustworthy convergence comes from raising rel_auc via data, not from the divisor.

## 0a¹³. 2026-07-22 (PM) — STATE-VALIDITY DAY: crx old-y invalidated, math unlocked, humor core .364, tight-correction convention

Audit-driven afternoon; every claim in the canonical artifact (`*_20260722` blocks).

**crx old labels fail arbitration.** 24 disputed blind-anchor labels, 3 Opus judges: old match-based pos
labels are ~5% genuine pos assertions — 46% are NEGATIVE assertions (criticism recorded as praise), 37%
invoked-without-assertion. Old crx +0.128/.335 is hereby a mention-mixture readout, never a pos-state
coupling (flagged in place). New strict instruments (R1: 1,048 pos / 41,104 neg pairs; anchors 0-errors)
become canonical; 6-way g4 fleet scoring 5,943 new PRs lands overnight → first honest crx ρ.

**Math unlocked by the same audit.** math_y proved to be a mention-mixture too (census QC: 54% of its
"pos" are neg-state). Census state-filtered rebuild on existing scores: **POS-state +0.181 p=.0032
(n=19, corrected .378)**, NEG-state −0.120 p=.019 — mirror-signed channels = valence-sensitive coupling.
Spot-check (corrected evidence frame): 10/15 census pos calls confirmed (~67% precision, attenuating).
Doc-type confound ruled out. Lesson logged: audit judges must see the annotator's evidence.

**Humor core pushed to +0.364** (cross-universe averaged-AUC axis, 16 dual-support metrics; detectable
stratum +0.330 n=23). v4 re-quarantine (37% of new labels removed) left the coupling unchanged →
dilution is metric composition, not label noise; humor readouts are STRATIFIED from now on. Core
trajectory .015→.282→.320→.364; corrected (rel_auc_core16 .323, rel_MI .816) = **.709** → bracket
[.364, .71]. R2 launched to converge it: 12,000 threads annotated same-day (13,936 quote-verified pos
pairs, 102 metrics ≥10 pos; quote-requirement kills hallucinated support at source); 8,702-doc corpus
scoring staged behind the crx fleet.

**Peer probe-expansion pilot NEGATIVE (and useful).** +300 probes: rel_MI .673→.755 (Spearman-Brown ✓)
but observed ρ flat (.381→.348) — MI "unreliability" is partly probe-distribution-structured, so
√rel_MI corrections are upper-style. New convention (user-endorsed): quote corrected values from the
highest-reliability instrument → peer tight-corrected = .348/√(.755×.631) = **.504** (was .584).
Probe corpus stays at 300; pilot cost ~1 GPU-h instead of a wasted fleet.

## 0a¹². 2026-07-22 — FLEET PAYOFF: humor 3-cell verdict, first math ρ (null), pr corrected, crx R1 launched

The 5-job GPU fleet landed overnight (all `[DONE]`-verified). Four results, all in the canonical artifact
(`WITHIN_METRIC_MI_VALIDITY_PEER_20260716.json`, blocks `*_20260722`):

| readout | raw ρ | p | rel_MI | rel_auc | corrected |
|---|---|---|---|---|---|
| humor old-docs cell (2,053 ch × 5,343) | **+.320** | .0002 | .816 | .129 | (.99 — unstable) |
| humor new-docs cell (× 7,996, v3 labels) | **+.134** | .0002 | .816 | **.561** | **.198** |
| humor combined cell | +.014 | .61 | — | — | **INVALID** |
| math FIRST ρ (2,209 forms × 2,056) | +.029 | .48 | .738 | .310 | .060 — **NULL** |
| pr (corrected column completed) | +.203 | .0052 | .789 | .212 | **.497** |

**Humor verdict:** the old cell replicates and strengthens (+.282→+.320 with expanded forms), but the
better-instrumented new cell (97 metrics, rel_auc .561) says the true coupling is ~.20 corrected — the
earlier ~.77 divided by rel_auc .129 and does not survive. **Combined cell = pooling artifact**: old docs
score +.146 higher than new on 83% of channels (corpus-selection effect) and 95% of positives are new-docs,
so pooled AUC is cross-universe-dominated (grand AUC .470 < chance). New rule: never pool doc universes
with different score levels into one AUC (doc-universe analog of same-family/never-pool-panels). Open:
new-cell class split (salience-inversion set is back in the 97) + one-pass-annotation label-validity gap.

**Math:** first-ever ρ is a credible null (instrument healthy: rel_MI .738, rel_auc .310) — coherent with
math-as-orthogonal-floor. **pr:** corrected ~.50; bottleneck is rel_auc .212 (sparse positives).

**crx R1 launched** (user directive: raise crx .128/.335): upstream pool = 141,644 PRs
(`code_review_pr_aggregated.csv.gz`; the 2,000 corpus is 1.4% of it — but its `text` col contains review
comments, so x-texts were rebuilt diff-based to avoid leaking y-side material). 6,000 stable-hash new PRs
(5,943 usable) + 40 blind anchors; GLM-5 on-page assertion-only annotation (2 patient runners) + g4 scoring
on GPUs 5/6/7 (2,368 channels each ~1,981 docs). Readout: new-docs cell only, frozen estimator.

## 0a¹⁰. 2026-07-20 — FORM-SET EXPANSION: EQUIVALENCE-FILTER AUDIT CYCLE + HYBRID FREEZE (peer, code-review)

Continuation of the user-approved 24-36-form expansion (0a⁹ plan lever #3). GLM-5 generated 20 label-free
forms/metric across 8 predeclared axes (peer 1,018 candidates / crx 2,687; crx had ZERO forms before this —
its first x-axis). The equivalence filter went through a full blind-audit calibration cycle before anything
was scored:

| filter | agree w/ blind Opus auditor (n=50 sealed) | FP (drift passed) | FN (valid rejected) |
|---|---|---|---|
| v1 | .60 | 0/5 | 20/45 |
| v2 (recalibrated) | .88 | 3/5 | 3/45 |
| **hybrid = v2 on axes 1-7, v1 on axis 8** | .82 | **1/5** | 8/45 |

v1 was mis-calibrated by SURFACE FORM (rejected imperative mood 0/7, sub-aspects .09 vs auditor .64); v2
fixed that but over-corrected, re-admitting 2 of 3 axis-8 construct-drift cases. Axis 8 (sub-aspect
decomposition) is the only axis with genuine drift, so the frozen rule is hybrid: v2 everywhere, strict v1
on axis 8. Rationale: FN costs surplus forms; FP contaminates the within-metric ρ estimand — minimize FP.
Preregistered before any probe/corpus scoring. Frozen: **peer 834 new forms (~17/metric), crx 2,368
(~17.8/metric)**; axis-8 gate blocked 111/263 v2-passes. Artifact:
`outputs/silver_match_v3/WITHIN_METRIC_MI_VALIDITY_PEER_20260716.json` → `FORM_SET_EXPANSION_20260720`.

Also complete: peer exp2 annotation (2×9,541 GLM rows, both shards clean after the MENU-truncation fix);
g4 abstract scoring of the 5,000 exp2 papers running on GPU3, with the ~8,952-paper combined-ρ analysis
chained behind it, and new-form probe scoring queued behind that (one-GPU rule).

## 0a¹¹. 2026-07-20 (PM) — HUMOR SALIENCE-INVERSION: the mention channel isn't dead, it's argmax-sampled

User-prompted metric-level audit of humor's near-zero mention ρ (+0.015), given the surprising gap vs
CW (+0.517) despite parallel comment format. Chain of eliminations, each with its own instrument:

1. **Bimodality** (per-metric recompute): +0.015 is a cancellation — detectability≥.55 metrics pool
   +0.393 (punchline +.82, premise clarity +.64) while 30 low-detectability metrics pool −0.145
   (laugh-yield −.89, timing −.72). CW has no negative tail; the tail IS the cw–humor gap.
2. **Label quality refuted as separator** (132-item blind Opus audit, group-blind, anchored): clean
   rates top .72 vs bottom .67; sarcasm absent (1/120); construct fit high everywhere.
3. **Advice-as-praise contamination real (~27%) but not the mechanism** (GLM stance pass over all
   4,354 pos items, controlled paired test): removing advice moves pooled ρ −.058→−.019 only; tail
   metrics stay negative under sincere assertion-only labels.
4. **Compensation praise refuted**: tail positives are ABOVE-median audience outcome (.576).
5. **Salience mechanism CONFIRMED**: tail positives are BELOW-average on judge-visible text quality
   (G = −0.23) while top positives are above (+0.14). Praise marks a joke's comparative advantage;
   when the advantage is text-invisible (timing/persona/laugh-yield), praised jokes are text-weaker
   than peers → text-only judge scores positives below negatives → below-chance AUC → negative ρ.
6. **Blind construct-class gradient** (Opus classifies metric cards, data-blind): artifact-observable
   n=28 pooled +0.120 (p=.049 vs random subset) > mixed −0.077 > performance-dependent −0.342.

Verdict: humor's mention channel is valid on artifact-observable constructs and actively INVERTED on
performance constructs. Silver praise labels are argmax-sampled by comparative advantage — a
discourse-selection law predicting inversion wherever quality dimensions live off-artifact (next
candidates to check: peer "presentation/impact", crx process metrics). Also: outcome-channel
reliability decomposition (rel_AUC .90 / rel_MI .78, cross-fit +0.103 p=.0004) shows outcome true
value ~.12 — that estimand is ceiling-limited, not attenuated. Caveat: humor_ypos.json build rule
not reproducible from census files (34% overlap) — provenance gap flagged.
Artifact: WITHIN_METRIC json → HUMOR_METRIC_LEVEL_LABEL_AUDIT_20260720 + HUMOR_OUTCOME_RELIABILITY_20260720.

**(evening) ON-PAGE QUARANTINE — humor revived.** Per user directive, every one of the 5,753 positive
silver labels was classified per-item (GLM-5, comment text only): STANCE (assertion/advice/theory/
reaction) × TARGET (on-page/off-page/unclear). Blind Opus validation: leak 0/20, false-quarantine
3/20, agreement .92. Kept 1,865 assertion+on-page pairs; quarantined 3,832 with per-item log
(sk3:mention_auc/humor_quarantine_log.json) — top-quarantined metrics are all delivery/stage/persona
constructs (233 timing-delivery, 154 delivery-commitment, 139 persona…), face-validating the
salience-inversion mechanism. Result on the frozen estimator: **+0.015 → +0.282 (p=.0042)**, negative
tail 14→2 metrics, grand AUC .485→.545; corrected ~.77 but rel_auc collapsed to .170 (halved pool) so
raw is small-n-attenuated. Next lever launched: humor form expansion 12→29/metric (on-page-constrained
generation, hybrid filter), GPU scoring queued behind crx; math form chain (141 metrics) also running.
Block HUMOR_ONPAGE_QUARANTINE_20260720.

## 0a⁹. 2026-07-19 (PM) — PER-METRIC DIAGNOSTIC, SONNET-INSTRUMENT NEGATIVE, COVERAGE-HOLE ROUNDS

Goal-driven push (/goal: raw ρ into .5–.7). All artifacts in
`outputs/silver_match_v3/WITHIN_METRIC_MI_VALIDITY_PEER_20260716.json`, blocks
`PER_METRIC_DIAGNOSTIC_20260719` / `SONNET_INSTRUMENT_PILOT_20260719` /
`COVERAGE_HOLE_FULLDOC_ROUNDS_20260719` / `EXTERNAL_AUDIT_GPT56_20260719`.

**What separates good metrics from bad (user question):** per-metric ρ_m(MI_form, AUC_form)
correlates with GRAND DETECTABILITY (auc_mean: cw +.60 p=.001 / humor +.70 p=.02 / pr +.47 p=.06)
and with nothing else — n_pos null, label-blind-Sonnet-rated specificity REFUTED pooled (+.07 ns),
census-error weak. Negative-ρ tail = anti-correlated channels (a91: g4 rel .50, ρ −.75 — reliably
inverted, real structure). Detectability gate is DIAGNOSTIC only — post-hoc auc_mean gating is on
the p-hack blacklist (Codex plan).

**Sonnet-as-instrument pilot: NEGATIVE, decisive.** 350 cw stories × 94 forms, paired vs g4 on
identical subset/labels: Sonnet split-half form-AUC rel .075 vs g4 .499 (grand AUC .534 vs .552).
Batched (94-rubrics-per-story) design's within-story halo destroys between-form differentiation.
Never scale Sonnet measurement; g4 stays certified. (`sk3:mention_auc/pilot_compare_result.json`)

**Coverage-hole full-doc rounds (g4, GPUs 3/4):** humor had 72% of positive docs unscored
(capped 392-char texts, 1,600/5,343 docs); pr had 79% unscored (1,351/6,480).
- PR: +0.157 ns → **+0.203 p=.006, n=23** — first clean pos-channel significance. Pair-merge now
  FLAT (old .157→.185 gain was coverage-compensation, not semantics). Beyond this: label-source
  ceiling (news coverage ≠ PR critique).
- Humor: +0.015 p=.80 n=42 with HEALTHY rel (.41-.51) → mention/state channel dead definitively
  (not coverage, not capping, not reliability). Outcome-channel test preregistered and staged
  (`sk3:mention_auc/humor_outcome_within.py`), awaiting estimand sign-off.

**GPT-5.6 blind audit (140 items):** arbiter GLM-anchored (sided GLM 51% vs blind 29%); ambiguity
conclusion holds humor/math, pr = fixable near-duplicate taxonomy, crx = fixable matcher failure.
Codex maximization plan: peer true ~.56, path to raw .43-.46 = 8-10k papers + 24-36 forms/metric.

**Scoreboard: cw +0.517 (band) · peer +0.340 · pr +0.203 · humor mention-dead (outcome pending).**
Pending sign-off: peer 8-10k GLM expansion; humor outcome-channel run; 24-36-form expansion (peer+crx).

---

## 0a⁸. 2026-07-19 — SIX-TASK LABEL AUDIT + SONNET ARBITRATION CLOSEOUT

`outputs/silver_match_v3/PEER_SILVER_LABEL_AUDIT_20260717.json` → `SIX_TASK_LABEL_CLOSEOUT_20260719`.
**Census correctness (GLM-4.7, anchor-QC'd, ALL matches):** crx .858 > peer .824 > math .788 > pr .729
> cw .656 > humor .595 (sibling confusion = the complement, wrong ≤3% everywhere). **Sonnet
arbitration of 2,700 disputed labels:** pr/humor/math split near-evenly (orig 37-41% vs verifier
41-45% + 12-17% third options) = genuine construct-boundary ambiguity; **crx outlier: original right
only 24% on disputes + 14.7% NO fitting candidate** (retrieval/coverage holes — privacy, correctness).
**Fix outcomes:** cw +0.419→+0.517 raw (in .5-.7 band; census fixes + comment→story bridge from raw
WritingPrompts link_id); pr pos-channel UNPOPULATABLE (79% of arbitrated states neg — corpus is
journalist complaints); humor dead at −0.14 (third independent label construction) — outcome channel
is its home; math/crx blocked structurally (0 form-MI variance) with labels now certified good.
Lesson: small manual samples over-represent the disputed class (crx manual 4/8 vs census .86 —
reconciled: overall fine, disputed subset worst-adjudicated).

## 0a⁷. 2026-07-18 — CROSS-TASK SCALE-OUT COMPLETE: state-channel ρ table for every norm-feedback task

Artifact `WITHIN_METRIC_MI_VALIDITY_PEER_20260716.json` → `CROSS_TASK_SCALEOUT_20260718` (canonical).
Uniform recipe (state-pos label × g4 instrument × instrument-matched sup-tau MI × within-metric pooled
ρ, 5000-perm): **CW +0.419 (p=.0002, corrected 0.741) | PEER-expanded +0.340 (p=.0002, 3,952 papers,
FULL 43-metric set, corrected 0.565) | humor +0.184 ns (channel-dead on capped docs, grand AUC .500 —
instrument-limited) | PR +0.157 p=.08 (corrected 0.31) | math & code-review STRUCTURALLY EXCLUDED
(0 form-MI variance).** Peer expansion vindicated the attenuation roadmap exactly: +2,000 papers via
direct annotation → rel_AUC .365→.552 → observed +0.292→+0.340 at full coverage, with observed and
corrected CONVERGING (~0.55-0.65 true). Structural finding: within-metric MI-variance fraction orders
domains on the taste↔verifiable axis (humor 77% > cw 51% ≈ pr 37% ≫ math 0% = crx 0%) — MI-validity
is only askable where taste lives. Census verification complete for cw/pr/humor (pr 73% correct;
humor 60%, worst sibling degeneracy). ypos polarity separation positive everywhere (pr y_all −0.05 →
ypos +0.16). Refinement paths: humor full-doc scoring (5,343 docs), census-state label swap (small Δ).

## 0a⁶. 2026-07-18 — GOLD-LABEL PROGRAM: state channel takes observed ρ .19→.35, corrected validity .63-.68

Artifact `WITHIN_METRIC_MI_VALIDITY_PEER_20260716.json` → `GOLD_LABEL_PROGRAM_20260718` (canonical).
**Census verification of ALL 19,049 matches** (GLM-4.7; anchors 250/250 good kept, 239/250 wrong
flagged): 82% correct / 14.7% sibling-corrected / 2.9% wrong; 557 abstains recovered; **14,415 gold
per-(paper,metric) reviewer-asserted STATES** (9.1K neg / 5.3K pos). **Lane B direct annotation of
all 7,540 reviews × 88-menu (GLM-5.2)**: 34,611 pairs = 2.4× pipeline coverage; NOISE recheck: 59% of
dismissals actually evaluative → extraction has a big false-negative hole. **RESULTS LADDER (observed
within-metric ρ, g4 instrument):** y_all +0.189 → state-POS +0.292 → state-POS×fulltext-class +0.315
→ **union-pos(A∪B)×fulltext +0.346 (p=.0002, n=30)**; cross-instrument GLM replications +0.339/+0.306.
**Attenuation-corrected validity 0.63-0.68** (rel_MI .66, rel_AUC .33-.43 measured) — in the .5-.7
target band. ⚠ HONESTY: preregistered strict-intersection platinum cut MISSED (+0.137 ns — A∩B
compounds complementary false negatives); union is exploratory-but-replicated. y_gold mention
correction DROPPED ρ to +0.136 → part of y_all ρ was matcher-judge surface artifact. Mechanism:
praise-mentions assert readable SATISFIED state; complaints anti-correlate; polarity resolution is
what unlocked the channel. Dead levers: probe-ext (rel_MI structure-limited .66), rank-consensus AUC,
metric-level (intrinsically negative). Raw-observed .5 would need rel product ≥.59 — remaining lever
is paper count. Labels: sk3:mention_auc/peer_y_{state_pos,state_neg,union_pos,platinum_pos,gold}.json.

## 0a⁵. 2026-07-16 — peer-review + code-review silver matching COMPLETE; salience refresh weak/null

`outputs/silver_match_v3/PEER_CRX_MATCHING_COMPLETE_20260716.json` (canonical). Resumed both
matching runs from checkpoint and finished them via Workflow `wf_123cfe79-8a4` (186 remaining
batches, one Sonnet subagent each, 0 errors; 5 dropped peer items patched). **peer-review 541/541,
code-review 459/459 batches** judged. Choice names are the FAITHFUL bank names, so join to bank
`metric_id` is exact (100%). Anchor QC on the newly-judged batches matches the checkpoint (peer
good .967/abstain .967; crx good .969/abstain 1.000). **Leaf salience→OPT with complete matching:
peer +0.190 (p .070, borderline), code-review +0.069 (p .43, NULL).** ⚠ The crx +0.069 does NOT
reproduce `FINAL_R2units_R3families`'s crx leaf +0.274 — that used a DIFFERENT matching run
(`code_review-N` decisions + `cr_crosswalk` 252-name catalog) vs the faithful-bank `crx-NNNNN`
retrieval finished here; two catalogs, unreconciled — quote neither as settled yet.
**Primary mention-AUC test already ran for cw (+0.136 p.33) and math (+0.181 p.21) — both NULL.**
peer/crx mention-AUC + within-metric prompt-variant test still need the per-document certified-judge
p-scores (a fresh GPU sweep: peer 88 metrics × ~2200 docs, crx 133 × 2000; within-metric multiplies
by ~12 prompt forms). `y`, bank, OPT certs are ready (`mention_join_{peer,crx}_20260716.json`).
**MENTION-AUC (primary test) RAN for peer+crx** (`outputs/silver_match_v3/MENTION_AUC_PEER_CRX_20260716.json`):
scored the ARTIFACT (paper text / aggregated PR diff_hunks) against each metric's certified rubric
(8B YES/NO logprob), y = human reviewer invoked that metric; Spearman(mention-AUC, opt_omega_bits)
across metrics. Peer 1952 papers×88 (93% paper-text join); crx 2000 PRs×133 (100% join).
**Cross-metric result: NO task positive.** cw +0.14 (null), math +0.18 (null), **peer +0.00 (p.98
null)**, **crx −0.33 (p.0014 SIGNIFICANT NEGATIVE)**. Judges DO discriminate mentions (AUC med
.53-.56, most >.5) but WHICH metrics discriminate is uncorrelated (peer) or inversely correlated
(crx) with reconstructability. crx-negative read: high-MI crx metrics = mechanical/uniform-verdict
(naming/formatting) that reconstruct perfectly but whose mention is decoupled from PR code state.
**⚠ 2026-07-17 CORRECTION: the 8B +0.101 within-metric diagonal is TAU-CONVENTION-FRAGILE and is
downgraded** (scan-tau only; dies under median-tau −0.005 and sup-tau −0.000). **The ROBUST result is
the gemma-4 diagonal: ρ(g4-MI, g4-AUC) = +0.170 (median-tau) / +0.210 (sup-tau, perm p=.003, n=25
metrics so far)** — matching the +0.22 disattenuated prediction from the 8B reliability audit.
Corrected interpretation = AUC-side reliability gates detection (8B-AUC rel .27 too noisy to show the
effect from ANY MI source; g4-AUC rel .43 shows it from own MI +0.21, marginally from 8B-MI +0.12).
Convention going forward: sup-tau (binarized channel capacity). Fulltext pilot NULL for 8B (no lift,
bipolar +0.000); abstracts-vs-fulltext irrelevant for weak executor.
**FINAL GRID 2026-07-17 (full coverage, 3 instruments both sides, sup-tau, artifact
`FINAL_GRID_20260717`): the ONE defensible cell is ρ(g4-MI, g4-AUC) = +0.189, perm p<.001, n=45 —
survives conventions, coverage doubling, Bonferroni×12.**
**GLM CAMPAIGN CLOSEOUT 2026-07-17 (artifact `GLM_CAMPAIGN_CLOSEOUT_20260717`): mention-AUC channel
is at its CEILING; every lever negative/flat — (1) G1: GLM-5.2 as 4th instrument confirms
reliability-gating (rel .15/.23/.33/.40 → mean ρ +.02/+.08/+.16/+.13; 8B-MI→GLM +0.242 p=.012) but
grand AUC .529 no higher than local; (2) G2: fulltext × frontier judge NO LIFT (bad +0.002 —
evidence-visibility refuted); (3) G3: high-confidence filter HURTS (rel .365→.253, ρ→+0.105;
quantity>purity); (4) G4: sibling-merged y flat (+0.182) and does NOT clear matched-size random-merge
null (+0.152±.039). E2E LABEL AUDIT (120 items, 5 auditors, raw-review-context chain): 66% fully
valid (19% sibling / 15% wrong / 5% paper-summary leakage), precision uniform across good/bad
channels. CONCLUSION: binding constraint = label NATURE — mention-y is reviewer ATTENTION (7/88
metrics/paper, 62% complaints), predictable only via topic salience; ρ≈+0.18-0.24 with reliable
instruments is robust across ALL y perturbations and is THE defensible number. Beyond it needs a
criterion-STATE label (data collection), echoing the dose-response conclusion. ~173K GLM calls.** Structure: the most reliable AUC instrument
(g4 .365) is the only responsive column (draws +.07/+.19/+.10 from 8B/g4/70B MI); 8B dead both sides;
70B-AUC unresponsive despite rel .31; NOT a clean matched-instrument diagonal (70B diagonal null).
3-executor mean-p ensemble COLLAPSED reliability (.117) — executors' form rankings disagree; ensembling
is not a lever. 'Bigger executor' not monotone: gemma-4-31B beats llama-70B on reliability AND
MI-validity. Registry (3 configs × 637 rows): sk3:mention_auc/registry_peer_within.jsonl; grid local:
notebooks/data/silver_v2_20260711/peer_within_final_grid_20260717.json.

**WITHIN-METRIC prompt-variant test = POSITIVE & SIGNIFICANT for peer** (`outputs/silver_match_v3/
WITHIN_METRIC_MI_VALIDITY_PEER_20260716.json`) — [SUPERSEDED, see correction above] first silver-mention
MI-validity signal outside humor. Per metric, K~12 certified prompt FORMS stratified across their
reconstruction-MI; per form mention-AUC_form = AUC(form's 8B judge over papers, mention y); pooled
Spearman(MI_form, AUC_form). **Pooled Fisher-z +0.101, within-metric permutation p=0.031 (n=49
metrics, 28/49 positive)**; monotone with channel strength → **mean-AUC>0.55 subset: +0.293, p=0.019,
6/7 positive**. Contrast: cross-metric peer +0.00 (null), crx −0.33 (neg); **within-metric peer +0.10
(sig pos)**. Simpson-like resolution: cross-metric MI-vs-mention-AUC is dominated by a mechanical-vs-
taste confound (high-MI mechanical metrics don't discriminate mentions); holding the metric FIXED and
varying only the prompt form isolates MI as a valid measurement-quality signal — the level the recon
certificate operates at. crx within-metric NOT FEASIBLE (~0 form MI variance, uniform-verdict domain).
Caveats: modest effect (+0.10), strong-channel n=7; MI_form = i_binary reference-target (Sorensen
output-entropy MI_form robustness + y_pos not yet run). Infra: within-scorer needs `if __name__` guard
(spawn) + GPU util sized to free mem (contended node, 0.20 fits 42GB-free).

---

## 0a‴. 2026-07-16 — N&C VAT iteration 2: y-audit, deep-V, GEPA negative, size/throughput

Full campaign documented in `datasets/notice-and-comment/README.md` §9 (canonical). Highlights:
**(1) y-audit**: agree-vs-disagree (agency accepted vs disagreed) is the live y — dense .671,
only y with within-docket comment-level signal (.558); outcome-y is docket-level (within .498).
Responded-or-not (matched vs never-matched, length-guarded) gives the strongest A signal (.636).
**(2) Deep-V**: 14 metric-seam hybrid programs (`methods/metric_seam/hybrids/
programs_notice_and_comment/`) incl. verification tier (eCFR authority lookup vs real CFR index;
numeric arithmetic-consistency checking). V_deep .615 on outcome-y beats regex-V (.595) AND a
docket-disjoint fine-tuned Llama-8B (.602). **(3) GEPA negative result**: 4 Sonnet-driven rounds
raised construct fidelity .485→.577 but DROPPED predictive AUC (A .612→.595 on agree;
within-docket .558→.501) — part of the A-bank's predictive power is unarticulated judge residue,
not the constructs; pre-GEPA bank stays canonical. **(4) Collins size test**: agency FTE ⊥
mechanizable signal; docket THROUGHPUT predicts V_deep ρ=.635 (p=.003), survives FAA-exclusion
AND sample-diversity partialling (ρ=.57 both) — RTK-as-case-throughput, not headcount.
8B bounds: internal leaky splits give .89-.93, collapsing to .602/.647 docket-disjoint — never
quote the leaky numbers.

## 0a″. 2026-07-15 — N&C VAT row + per-agency size/time ladder (v42)

First VAT row for notice-and-comment, on the v4.2 corpus (details + tables:
`notes/2026-07-15__nc-vat-run.md`; results `notebooks/data/nc_vat.json`; raw A-scores
`datasets/notice-and-comment/v4/nc_scores_shard{0..4}.npz`). **Pooled, y = majority
rule-change outcome MADE vs NONE, docket-grouped CV, n=7,084: V .595 / A .592 / V+A .593 /
dense .588 — all arms ≈ char_len alone (.595); T̂ ≈ 0.** Engagement-y secondary: V .500 /
A .559 / dense .585 (only y where the judge beats regex). any-MADE union y RETIRED as
primary (n_labels confound, .695). Per-agency ladder over 22 agencies, 13 reliable
(≥20 dockets, ≥8/class): **agency size vs signal uncorrelated** (Spearman ρ=.05 VA /
−.27 dense, n.s.); FAA .671/.757 and NHTSA/AMS top; giants CMS/ED weak. Sub-.5 wild
AUCs in tiny-docket agencies (USCBP .167, MSHA .225) are rule-level outcome clustering
artifacts, not anti-signal — outcome-y is heavily docket-level (63% of multi-comment
dockets are outcome-pure), so within-docket comparative designs are the natural next step.

## 0a⁗. 2026-07-16 — ★ WITHIN-METRIC PROMPT-VARIANT VALIDATION: MI predicts measurement quality (CW)

**The decisive readout the mention-AUC program was building toward.** Design (user-specified): for a FIXED
metric, prompt variants φ (from the 98 stored rubric paraphrases per metric) each carry their own pipeline MI;
score each variant on the same source texts; AUC_φ against the positive-polarity-only mention label; correlate
MI_φ ↔ AUC_φ WITHIN the metric — all y-noise/polarity/sparsity confounds cancel.

**Creative writing: mean within-metric ρ = +0.393; 20/22 metrics positive; permutation p = 2×10⁻⁴; Fisher-z
p = 5×10⁻⁶** (22 metrics × K=10 forms × 1,040 stories; 8B YES/NO judge; per-form MI via the pipeline's own
`i_binary` estimator — declared fallback). Math: +0.106, 9/13 positive, ns — underpowered (its sigs store only
5–31 forms/metric).

**FOUR-DOMAIN FINAL (2026-07-16 later):** humor −0.049 (44 metrics) and press-releases −0.064 (57 metrics,
negative-polarity directional; direction assumption mildly inverted, null either way) are **floor-effect
nulls**: pooled ρ tracks measurement-channel strength (mean per-metric AUC: CW .609 → +.39; math .554 → +.11;
humor .503 → −.05; PR .481 → −.06). Where variants are at chance against y there is no validity gradient for
MI to track, and humor/PR per-form MI spreads are half of CW/math's. **Corrected claim: MI predicts
measurement quality where a measurement channel exists.** Within math, taste metrics are positive
(Explanatoriness +.61, Simplicity +.50) while verifiability-adjacent ones are negative (Elegance of proofs
−.72, rigor −.45) — the taste-vs-verifiable split recurs inside a domain. Top CW within-metric ρ: Narrative
Economy +.85, Character depth +.76, Dialogue craft +.71. Exec note: sk1 A100s ~20× slower than sk3 B200s on
long-text scoring. Between-metric mention-AUC, by contrast, is noise-bound everywhere (CW +.17 pos-only ns,
math +.09–.35 depending on polarity handling, PR +.09 ns) — the polarity audit showed complaint-invoked metrics
legitimately anti-discriminate (Garfinkel invoked-when-violated, quantified: "Standard English Conventions"
AUC=.40 with 170 positives).

Artifacts: `notebooks/data/silver_v2_20260711/{cw,math}_within_metric_result.json` (+variant MI/scores on sk3
/tmp/), plan+corrections `notes/2026-07-15__mention-auc-plan.md`. Levers to extend: more forms for math,
frontier judge, remaining corpora (peer-review needs matching; NC leak-blocked; code-review needs re-match;
humor needs standup source text). Infra: sk1 env rebuild in progress; v14 holds resident vLLM per stage.

## 0a‴. 2026-07-15 (late) — MENTION-AUC: the correct silver-label validation (design, data readiness, MI source)

**This supersedes the day's earlier salience and item-outcome analyses as THE silver-label test.** Full plan:
`notes/2026-07-15__mention-auc-plan.md`.

**The experiment (user-specified, = the long-planned "Leg B mention-AUC"):** per task, over texts `T` and R2
bank metrics `M`:
- `y[T,M]` = silver label = did a human comment on `T` invoke `M` (extraction→matching), aggregated to the
  parent text.
- `p[T,M]` = the CERTIFIED metric prompt's P(metric applies) scored on text `T`.
- `AUC_M = AUC(p[:,M], y[:,M])` per metric; then **`Spearman(AUC_M, MI(M))` across metrics** — do higher-MI
  metrics have judges that better predict where humans invoke them.

**What DIFFERS from earlier work (now corrected/struck in the artifacts):**
- ~~salience correlation (MI vs corpus-wide invocation *count*)~~ — no per-text `p`, no AUC; weaker channel.
- ~~item/outcome dose-response (MI vs metric-judgment AUC against the *upvote/merge* label on the
  `*_modeling` corpus)~~ — wrong label (outcome, not mention) AND wrong texts. The humor +0.47 there is
  outcome-prediction, NOT the silver-label result; do not quote it as such.

**Data readiness (inventoried 2026-07-15).** Per-task silver norms WITH `source_id` exist for **24 corpora**
(`data/silver_match_v3_20260712_faithful/norms/<corpus>.jsonl`). 7 tasks have MI certs. Metric matches join
to these norms (humor 100% by norm_uid; CW 100% by norm text). Source-text availability for the 7 cert tasks:

| task | norms+source_id | metric match | source TEXT in hand | MI cert |
|---|---|---|---|---|
| **creative-writing** | ✅ 4,929/2,431 src | ✅ | ✅ `wp_comments/input.jsonl` (unit_id→story) | ✅ |
| **code-review** | ✅ 200k/17k src | ✅ | ✅ 2636/2636 PRs join modeling text by paper_id | ✅ |
| peer-review | ✅ | ✅ | ⚠️ source_id `iclr_..._r0` wrapped — needs id-normalization | ✅ |
| press-releases | ✅ | ✅ | ⚠️ `pair_X` id mismatch — needs normalization | ✅ |
| humor | ✅ 77k/19k | ✅ (canonical) | ❌ standup source not in modeling corpus | ✅ |
| math-stackexchange | ✅ | ✅ | ❌ no direct join | ✅ |
| notice-and-comment | ✅ | ✅ | ❌ no direct join | ✅ |

**CORRECTION (2026-07-15 later): the matched-norm↔source-carrying-norm join is task-specific.** Our v2 matches
were over the `bge_pertask` norms; the source_id lives on the `faithful/norms` extraction. Text-join of bge→
faithful: **CW 100%, press-releases 100%, math 100%, but code-review only 1%** (its bge extraction differs).
So: CW/PR/math matched norms DO carry source_id (via 100% faithful join); **code-review needs its faithful
norms re-matched** (or match a fresh sample). Humor is matched on faithful (source_id) but its standup source
text is not stored. Revised readiness for the mention-AUC:

| task | matched norms carry source_id | source TEXT in hand | net |
|---|---|---|---|
| **creative-writing** | ✅ (100%) | ✅ wp_comments story text | **READY — launch now** |
| press-releases | ✅ (100%) | ⚠️ `pair_X` → 38% partial join; recover PR text | text-recovery |
| math | ✅ (100%) | ❌ `comment_X` not in modeling; needs raw math_se | text-recovery |
| code-review | ❌ (bge≠faithful, 1%) | ✅ PR text by paper_id | **re-match faithful norms** |
| humor | ✅ (canonical) | ❌ standup text absent | blocked on text |
| peer-review | ✅ | ❌ ICLR papers not in modeling | text-recovery |
| notice-and-comment | ✅ | ⚠️ likely in `rtc_sections` (docket ids) | text-recovery |

**IMMEDIATE plan (launched 2026-07-15, updated same night):** CW p-scoring RUNNING on sk3 GPU1 (sk1 env is
broken — no miniconda; sk3 GPUs 1/2 freed up). Source-text recovery COMPLETE for PR (15,067 — but norms were
extracted from the NEWS ARTICLE, so scoring uses the PRESS RELEASE side to avoid leakage; 2,217 scoreable),
math (6,492 answer bodies, clean), peer-review (23,398; 63% full text; clean but needs metric MATCHING first),
NC (recovered text = response-to-comments = the critique itself → LEAK-BLOCKED, needs underlying comments).
Math (2,056×50-metrics) + PR (2,217×73) queued behind CW on the same GPU. Prompt-variant (p,p′,p″ within-metric)
phase 2 design in the plan note §8b — infra supports it via OSL per_form. Code-review still needs faithful-norm
re-match. sk1 landmine: alexspan's miniconda3 wiped (dangling uv symlinks) — do not target sk1 until rebuilt.

→ **creative-writing is READY end-to-end now.** `y` built for CW (story-level: 1,040 labeled
stories, **51 metrics with ≥10 positive stories**; `/tmp/cw_story_y.json` + `cw_story_texts.jsonl` on sk3).
Remaining ingredient: **`p` (a certified-judge scoring run over the source texts)** — required regardless of
MI source. Existing `mbar`/`mbar2` panels are NOT reusable — they scored the `*_modeling` outcome corpus, not
these source texts.

**MI source (user: use the live v14 run).** v13/v14 MI is at **R3** (metric_key `<task>_R3_metric<N>`,
`achieved_value` bits, channels mcq+behavioral). v14 coverage is uneven: **creative-writing 50 metrics** (usable),
peer-review 8, math 4, humor 4, code-review 2. v14 behavioral lands overnight; only MCQ is readable now.
★ **Old opt cert vs v13 MI (rolled R2→R3, n=25): behavioral ρ=+0.65 (STRONG), MCQ ρ=+0.18 (weak).** So the
old opt cert (large n, 285/task) is a defensible proxy for the **behavioral** channel but not MCQ. Plan: wire
mention-AUC for BOTH MI sources (old-cert @ R2 large-n + v14 @ R3), lead with behavioral.

**Do we need another v14 run?** For creative-writing, NO — 50 v14 metrics suffice for a first correlation. For
the other tasks, v14 is too sparse (2–8) → either use the old cert (behavioral-correlated ρ.65) or wait for v14
to scale. The run that IS needed either way is the **`p` certified-judge scoring over source texts** (not a v14 rerun).

## 0a″. 2026-07-15 (night) — FIVE-TASK family-grain reconstruction–silver correlation

All five gold tasks now have full silver matching + salience↔OPT correlations
(master table `outputs/silver_match_v3/FIVE_TASK_FAMILY_CORRELATION_TABLE_20260715.json`;
decisions + replicated family maps in `notebooks/data/silver_v2_20260711/`). **At the
construct-FAMILY grain (sibling families from two independent Sonnet passes, pair Jaccard
.59–.70, consensus = intersection), the correlation is significant in 5/5 tasks:** humor .425,
CW .463, code-review .346, PR .267, math .274 (perm p .0005–.018, family-max OPT);
family-size partials survive at .16–.30. Leaf-level is significant only where sibling
degeneracy is low (code-review .274 — its anchors pass 68% exact vs CW's 32%; CW .231);
PR and math are leaf-null but family-significant. T_HM is a clean null discriminant in 4/5
tasks; in code-review it correlates too (OPT|H_M partial ≈ 0 — verdict-balance-mediated there,
flag when quoting). New silver completed this session: code-review 19,997 (364 anchored
batches) and math 3,133 (57 batches; the old "placeholder catalog" warning was wrong for
math — 135/208 names join the v3 bank). PR was rescued from the unharvested wf_e4731a71
journal (11,706 decisions) + 3 re-run batches. Detail: `memory/project_silver_matching_audit_v2.md`.

## 0a′. 2026-07-15 — Humor silver_match_v3 canonical release + reconstruction–silver MI

Humor canonical 77,378-row release and the 285-metric reconstruction–silver correlation are DONE
(sk2 `.../releases/humor_binary_typed_v1/`; local addendum
`outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/HUMOR_CANONICAL_RELEASE_EXECUTION_ADDENDUM_V2.json`).
Unblocked the failed-closed v1 handoff via two append-only repairs: (1) truth v2 confidence
backfill (31 legacy sonnet_audit rows had null confidence; 22 recovered from agreeing superseded
labels, 9 conservative "low"; UID set unchanged, v1 preserved); (2) risk-sampler v2 deriving audit
lanes from sealed `frozen_hybrid_evidence` (the sealed full-285 schema never carried
`provenance.hybrid_policy_lane`). **Primary MI result (22,090 labeled UIDs excluded; 8,010 exact
accepted matches; 15,370 source groups): source_presence.OPT ρ=.160, perm p=.0085, source-group
bootstrap 95% [.126,.179]; partial ρ|log-leaf+H_M=.064; silver split-half reliability .934;
117/285 metrics nonzero.** Old forced-top-1 ρ=.185 replicates in magnitude under the much cleaner
exact-accepted-only design. Claims firewall: advisory silver (blind 85.5%, Wilson LB .753);
precision claim NOT supported until the frozen 1,002-row risk sample
(`audits/humor_postblind_hybrid_risk_v1/sample_v2/`) is two-judge labeled. Key production
decomposition: of 43,748 "unstable" rows, 21,618 are both-orders-same-leaf CE-gate rejections vs
22,130 true order disagreements — the CE gate, not typed instability, is the biggest recall lever.
Detail: `memory/project_silver_matching_audit_v2.md`.

## 0a. 2026-07-09 — GitHub PR VAT audit + org-size↔formalization

**PR V/A/T ladder audit** (`memory/project_pr_vat_audit.md`): first-pass ladder (V .582/A .734/T
.749, "no taste residual") RETRACTED — in-sample fit + un-deduped join (25% dup keys) + repo
confound + leaky dense split (repo-rate lookup alone .794 > dense .749). **Definitive full-corpus
honest ladder** (44,751 sound PRs / 594 repos, 68K diffs × 127 metrics, CV'd): pooled V .576 / A
.615 / V+A .632 / repo-rate .706; GroupKFold V .549 / A .596; **A within-repo OOF .577 (571 repos,
69% >chance, p<1e-4)** — A genuinely generalizes cross-repo at scale (the 2.5K-cell "A=chance" was
POWER not confound). Top A features are real norms: test_presence, test_source_correspondence,
dependency_hygiene, import_organization, idiomatic_patterns = "ship tests + follow repo idioms."
P2F causally real but low-coverage (fires 1.8%; gate OR~1.5; rare-gate AUC ceiling .558 even at
perfect precision → report V as gates not AUC). Repo-disjoint dense T=.584 (not .749).

**Org-size ↔ formalization** (`notes/2026-07-09__org-size-formalization.md`): operationalized
"how institutionalized is a repo's accept/reject rule" = **within-repo predictability** (5-fold CV
AUC of our 82 VAT features — 64 A + 18 V — per repo). 249 repos, mean .579, range .07–.92. **Org
size predicts formalization ON PAPER but NOT IN PRACTICE**: stars→has-CONTRIBUTING ρ+.36 p<1e-4,
but stars→decision-consistency ρ+.04 (null), written-rules→consistency ρ+.00 (null). SMALL orgs
(≤5 members) MORE predictable (.611 vs .566, p=.035) — "one-man band = rigid rule; big org = docs
+ many reviewers + heterogeneity." Robust: replicates on independent TF-IDF-text predictability
(small .635 vs large .569, p=.020; VAT↔TF-IDF ρ+.34). Not a sample-size artifact.

**Infra scale-up** (target 50K signal rows): fixed silent 5-day collection death (gh-PATH +
disk-100%); 14 self-healing loops (sk1=4/sk2=6/sk3=6), 10-min watchdog (relaunch + disk-prune
>75%), queue 1,865→4,388 via relaxed cache re-triage + widened discovery, hourly laptop cron
(enqueue+harvest+consolidate). 8,426 signal rows and climbing.

## 0c. 2026-07-08 — claim-verification v2 (patents-shaped) + patents pipeline audit

**Peer-review claim-verification v2** (`methods/claim_verification/{paper_adapter,run_paper_pilot,
smoke_paper,rerun_pa_fixed}.py`, outputs sk3 `outputs/claimverify_paper/`): metric-space review
found old cv1-3 battery structurally weaker than patents (one claim/paper, regex-only body checks,
no planted controls, 18.5% EMPTY bodies from additive section whitelist). Rebuilt patents-shaped:
LLM claim decomposition (4.6 claims/paper) → retrieved body passages → localize-then-verify w/
verbatim-grounding demotion → prior-art leg (earlier-year ICLR pool + planted SELF + foreign) →
planted controls. Pilot n=300: **instrument VALID** (null-twin .719 vs .235 retrieval, verdicts
collapse on foreign body; number-perturbation kills FULL .069→.018; PA self-detect .702
echo-indexed / foreign-distinct .999). **Discrimination null CONFIRMED as real**: s_* arms ~.50,
old battery re-run on FIXED subtractive bodies (2399/2400 usable) still ~.50-.52 — claim
substantiation is necessary-but-not-sufficient for ICLR accept; three-way mechanization picture
stands. Best arm r_top1_overlap .600. PA-verdict positional-array misalignment bug → echo-indexed
verdicts (lesson: per-candidate arrays from one call are fragile). Gemma FULL-conservative (5.9%
FULL / 82% PARTIAL) — mirrors patents 31.4% gold-disclose.

**Patents audit (user: "surprised at low performance")**: V=.591/A=.616 were HARDCODED notebook
constants (claim-level `fell`, not doc-level); sk3 grouped-CV script + ALL option3 corpus builders
missing (parallel thread never checked in). `datasets/patents/audit_regroup_va.py` closes the hole:
CSV↔jsonl row-aligned 0/59,937 mismatch → grouped-by-app V=.601/A=.623; **dedup-stable** (12.2%
exact dups, 479 contradictory-label rows: V=.600/A=.624). +15.1pt disclosure gap REPLICATES
(+15.8/+16.9) but is 5× length-heterogeneous (+28.7 short → +5.3 long). **NEW: position leak
AUC=.87** (gold in slot 8/8 80.4% — reported leak probe only covered doc-id format); gap survives
within-position but identify-gold-among-K evals need shuffling. Verdict: low level is STRUCTURAL
(oracle-vs-retrieval evidence asymmetry, verifier conservatism, lit baselines .57-.64), not a bug.
Local `patents_final_outcome_balanced.csv.gz` = naive balance, orphan, never measure on it.

---

## 0b. 2026-07-07 — peer-review V/A decomposition (expert-verdict Y)

Source of truth: `project_peer_review_va.md`. Y = accept/reject on paper ABSTRACTS
(`datasets/peer-review/splits/*.csv.gz`, 69% accept). **Venue confound confirmed**
(venue-family alone → 0.819 AUC; matches other-agent's 0.827): NeurIPS/ICML are accepted-only
(ICML-2024 = 1.00), ICLR/OpenReview is the clean expert-verdict subpop, F1000 alien community.
PRIMARY corpus = ICLR-only, threshold-free AUC; pooled kept as confound reference. Controlling
venue costs only ~0.02 and A−V lift identical pooled vs ICLR (+0.056→+0.057), so articulable lift
is NOT a venue artifact.

**Baseline V/A (ICLR balanced n=2400, Gemma-4-31B, `notebooks/data/peer_review_va.json`):**
V=0.611 (17 inline regex feats) / A=0.676 (154 merged rubrics) / V+A=0.682 / A−V=+0.071, NA=0.65
(genuine — abstracts can't speak to replication detail). Top signal in BOTH layers =
reproducibility/asset-availability (best A "open data/code provided" 0.77; best V `v_kw_code`).
Scorer `datasets/peer-review/score_va_gemma.py` (raw npz cached) + `aggregate_va.py`.

**V thinning ladder (code-metric design, tiered mechanization):** T0 regex (have) → T1 agentic-hybrid
(LLM_FIELDS extract + `score(text,extracted,ops)`, metric_seam `programs_<task>/<aspect>_h0.py`
convention — peer-review currently has ZERO code-metrics; 92/154 aspects mechanizable) → T2 retrieval
(novelty=1−sim_to_nearest_prior_art: TF-IDF internal `Ops.retrieve_similar` free, external arXiv
bge-m3 FAISS = big reusable build mirroring patents pipeline; a214 URL-fetch; a45 HF/PWC catalog
lookup) → **T3 EXEC (implement baseline, run on data, check stated results) = OUT OF SCOPE but the
conceptual CEILING of the verifiability axis — name in write-up (tacitness/thinning-ladder + future work).**
Approved build: 5-aspect agentic-hybrid cluster a163/a130/a214/a25/a45 + cheap retrieval; GEPA on
A prompts (GLM-5.2 proposer, Gemma-4 judge, fidelity objective) on ~20-30 baseline-selected subset.

---

## 0a. 2026-07-04 — cross-task iso-morphism scale-out (GOAL thread)

User goal: scale tacit-knowledge analysis across many tasks, find significant iso-morphism
between pairs (all three readings staged: task pairs → form pairs → model pairs; 1 GPU OK).
Day-1 results (`notes/2026-07-04__crosstask-isomorphism.md`, data `notebooks/data/
two_faces_20260702/crosstask/`): **(1)** R3-level sharing matrix, 9 tasks × 36 pairs vs
size-matched other-tasks null: **5 pairs FDR-significant** (CW×humor z=+11.3; news×PR +5.7;
legal×math +5.1; grant×peer-review +3.7; math×peer-review +3.7) — interpretable criterion
families (narrative craft / news media / rigor / research merit); CW×PR ANTI-affine (−5.7).
**(2)** CW×humor judge-verified end-to-end (κ=.69, 30 same-criterion pairs): **concept TYPE
transports** (83% taste/craft agreement, perm p=.0017) — tacitness-class is concept-intrinsic;
continuous profile correlations n.s. at n=30 (needs more grid pairs). **(3)** Form-pair census
(`isomorphism_census.json`): definition = modal best rung; name recovers ~.3–.4 of own verdict;
exemplars/dossier REGRESSIVE at all reader sizes (construction audit owed — k=2/400ch).
**(4)** Split-half stability HARVESTED: OPT_Ω even/odd ρ = .70 CW / .96 humor, 0 determinate
flips (halves ~90% UNDERSAMPLED = n=150 censoring). In flight on sk3: press-releases R3 sweep
(GPU1, PID 2587370, 42 metrics, byte-comparable glm_a/b/c recipe) with auto-chain →
news-homepages (watcher PID 2596989). Math domain needs a task-key bridge
(`math-stackexchange` hierarchy vs `math` preset).

## 0. 2026-07-03 state — tacit-knowledge measurement → paper push

Roadmap approved by user (this date). Three doctrine resolutions now in theory doc
(`notes/2026-07-02__two-faces-theory.md`): (1) **reconstruction-only** — metrics never label-aware,
C_dense trains on own-verdicts (x, M̄_E(x)); no human studies (code verified compliant:
value_certificate.py "never the aggregate"); (2) **family-top anchor** adopted for iso-performance
only (biggest same-family member's M̄, a priori; executor-consistent stays PRIMARY for
decompression); (3) **ε_form band replaces FORM-DOMINATED verdict** (fragility charged against the
ceiling; binary flag → diagnostic).

**Seam-splitting night-1 (2026-07-03 eve, roadmap `notes/2026-07-03__metric-seam-paper-roadmap.md`
executed):** E-S1 planted kill-switch RUN end-to-end (pre-registered, clean-room after a caught
blinding breach): zero false certifications in 14 cells, op-type recovery 2/2, S1 ceiling formula
exact to 0.001, p903 evidence-op recovery 98%-of-ceiling once the contract's missing-dpid
artifact is excluded; near-misses = one-improver-round reaches 76-83% of ceiling (h1 follow-up).
v1 gates RESOLVED at n=500 (a110 certified .989, a80 scoped .99, a105 A-layer confirmed). CW
seam survey done: median rel .90 (highest) with median ρ/ceiling .128 (lowest) — taste pole
confirmed. Certificate lemmas drafted with 12 gaps (`notes/2026-07-03__seam-certificate-lemmas.md`).
Llama-3.3-70B second-family replication running overnight (PR+math, verbatim prompts). Code-PR
confound work HELD for the other agent. Details: `notes/2026-07-01__metric-seam-pilot-results.md`
(§GATE RESOLUTION, §E-S1, §CREATIVE WRITING).

In flight: 70B full native rescore GPU2 (metric ~44/67, native-complete ~Jul 10; pass-2 chain
armed — supersedes the Q1/Q2/Q3 fork). Shipped: expansion-chain v2 driver (self-readout, truncated
gold, balanced planted rules, compliance normalization; synced sk3). New finding:
`notes/2026-07-03__what-gets-decompressed.md` — taste = enculturated index (cheap decompression),
craft = expensive decompression; banks contain 0 mechanical criteria. Queued (GPU, after rescore):
humor 70B-orbit retarget → chain v2 stratified-by-concept runs → powered transitivity; Gemma-4
panel (needs smaller sizes downloaded); GEPA rung on DEEP metrics; new domains (news-homepages,
press-releases banks ready; math needs bank construction). Split-half certificate stability
running on sk3 CPU.

## 1. Overarching framing

### 1.1 The V + A + T decomposition

Source of truth: `project_verifiability_explainability_gaps.md` (Apr 2026, stable).

```
Outcome = f(Verifiable) + g(Articulable) + h(Taste)
```

Two named gaps we can actually measure:

- **Verifiability gap**: what `g + h` captures that `f` misses (programmatic verifiers fall short).
- **Articulability gap**: what `h` captures that `f + g` misses (LLM-as-judge + rubrics fall short).

`f` is split into nine operational sub-types (Computational / Factual / Consistency / Procedural / Statistical / Causal / Completeness / Pragmatic / Normative) — Tier 1-2 are mostly automatable; Tier 3 is judgment-requiring (the boundary with `g`). Each sub-type maps to a different program kind in the propose-and-test algorithm.

### 1.2 The three-AUC tacitness layering

Source: `project_tacitness_two_layers.md` (May 13, 2026).

For any rubric, measure three predictor AUCs against expert ground truth:

| AUC | Predictor | Reading |
|---|---|---|
| A | Code/program implementing the rubric | Fully verifiable |
| B | LLM-as-judge applying the rubric | Verifiable + language-tacit |
| C | Replicable-expert ceiling (best dense model) | Verifiable + language-tacit + fully-tacit |

| Gap | Formula | What it measures |
|---|---|---|
| Verifiability gap | C − A | Anything formal code can't reach within the replicable expert construct |
| LLM-over-code surplus (semi-tacit) | B − A | What language reasoning + world knowledge recovers beyond code |
| Articulability gap (fully tacit) | C − B | Polanyi's "we know more than we can tell"; even an LLM-judge with full world knowledge can't reach it |
| Out of scope | 1 − C | Personal taste + irreducible label noise. NOT what rubrics aim at. |

`C` is **not** 1.0. It is the dense-model ceiling per task. `1 − C` is taste residual and is explicitly out of scope for the articulability paper.

### 1.3 Noah framing (2026-05-28)

Source: `project_noah_meeting_2026_05_28.md`.

- Pitch as "the bound of explicitly or implicitly defined but **communicable, describable, community-agreeable** metrics" — NOT subjective vs objective. The target is **intersubjective**.
- Articulability ceiling is **operationalization-dependent** (tagger quality, prompt verbosity, rubric phrasing) — every "we've hit the ceiling" claim must carry that qualifier.
- Avoid the word "subjective" in writeups; use "community-agreeable."

### 1.4 Thin / thick rules, philosophy

Source: `project_thin_thick_rules_philosophy.md`.

- Daston, *Rules*: thin (algorithmic) vs thick (judgment-laden, with caveats). "Behind every thin rule is a thick rule, cleaning up after it."
- Wittgenstein §201 + Kripke: no rule fully self-applies; correctness lives in community practice. Implies no privileged metric-tree structure.
- Legal rules-vs-standards (Kennedy, Kaplow, Schauer): thinning is upfront investment, worth it for high-frequency evaluation.
- Dreyfus/Polanyi/Ryle/Aristotle: experts operate on internalized thick rules; bottom-up extraction captures pattern fragments, not hierarchies.

Mapping back: `g(Articulable) = g_thin + g_thick`. Thin g collapses into f when codified; thick g is rubric/LLM-judge territory; whatever neither captures is taste.

---

## 2. Per-task modeling state

Canonical clean dataset paths come from `reference_clean_datasets_per_task.md`
(2026-06-02). Dense reward model AUC numbers come from `project_dense_model_sweeps.md`
(2026-04-05, Llama-8B + LoRA, subset sweeps 0.1-1.0, 3-5 trials per subset).

### 2.1 Creative writing

- **Source**: WritingPrompts pipeline. Canonical file `creative-writing/writingprompts_modeling_clean.csv.gz` (135 MB, May 12) — 96,080 rows, 70,453 unique prompts. Mod-bot template drop + per-length-bucket rebalancing. Use `prompt_id` as group key.
- **Group-split done**: yes.
- **LitBench was tried and ruled out** (`project_creative_writing_dataset_search.md`) — same Reddit upvote family, taste-laundering. Don't re-suggest.
- **Dense AUC (Llama-8B)** on the legacy LitBench-shaped target: **median 0.868 / max 0.904 at subset 1.0** (~70K rows). **Still climbing** at full data — has not saturated. Variance high.
- **Articulation gap is large** (per `project_tacitness_two_layers.md` hypothesis: large B−A and large 1−B).
- **Active issue**: v2 judge cells were scored with a stale peer-review system prompt (`feedback_judge_prompt_cross_task_bug.md`) AND a narrow applicability interpretation. Re-scored as `judge=qwen_relaxed_v2_2026_06_01` per `project_cw_relaxed_appl_v2_2026_06_01.md`. Pre-fix RF AUC 0.541 → post-fix 0.636 on a fixed test split.
- **articulation_star smoke run** uses this task; see §3.1.

### 2.2 Peer review

- **Canonical file**: `peer-review/peer_review_modeling_dataset.csv.gz` + `_with_reasoning` + `_with_topics`. **NOT yet rebuilt with `paper_id` group key** — leakage flagged. Do NOT use for production runs until rebuild (per `reference_clean_datasets_per_task.md`).
- **Corpus scale**: unified dataset (Mar 2026 snapshot, pre-existing pre-rewrite content): 82,460 papers / 299,961 reviews / 43,332 accept / 21,880 reject across ICLR / NeurIPS / ICML / TMLR / eLife / F1000 / COLM / ACL / CoNLL.
- **Dense AUC (Llama-8B)**: **median 0.770 / max 0.783 at subset 1.0** (~56K rows). **Saturated** around subset 0.5-0.6 at ~0.77-0.78. Strongest signal of the four primary tasks.
- **Verification_library populated**: 571 per-aspect Python predict-programs at `runs/validity_full/v2/peer_review/codegen_claude/` (`reference_codegen_per_aspect_programs.md`). 2,618 thin / 15 thick / multiple thousand failures from prior Direction-2 pipeline (`project_verification_pipeline_recipe.md`).
- **Local explanations sweep** (Apr 17): test AUC 0.5892 baseline → 0.6141 sweep-best (Trial 11). See `project_nc_pipeline_state.md` for the active feature list; see `project_local_explanations_clustering_findings.md` for the locked operating point (tw=0, eps=0, LLM dedup).
- **Norm extraction in flight** on sk3 (Qwen-122B batch mode after killing OpenAI-server runs on 2026-06-04). Output: `/lfs/.../data/peer_review/norm_extracted/extracted_qwen.jsonl.gz`.

### 2.3 Code review (with LeetCode pivot)

State of play is layered — old `code_review_dense_4096tok` is being **demoted** (not deleted) in favor of LeetCode + CR.SE + LeetCode.

- **Old code_review_dense_4096tok**: 141K PRs from GitHub, 86/14 accept/reject (`project_code_review_modeling.md`). Demoted as primary because of three confirmed leakage issues (`project_code_dataset_pivot_2026_06_02.md`):
  - WIP / draft / "do not merge" title flags (1.6% near-deterministic).
  - Project bimodality: 71% from ≥85%-merge repos, 14% from <30%-merge repos.
  - Bot-merge mislabels: PRs whose title says "merged by Bors" labeled rejected.
- **Old dense AUC**: subset 0.5 = 0.780 (per `project_dense_model_sweeps.md`). Subset 1.0 never finished writing metrics. Now known to be partially leaky.
- **New primary code task**: LeetCode within-problem upvote rank, then editorial-similarity (pivot 2026-06-02).
  - **Phase 1**: top-25% / bottom-25% upvote rank within problem, 80/problem (LC v2 dataset). ~80K labeled rows from 1500-2000 problems. Build script `scripts/build_leetcode_balanced.py`. See `project_leetcode_push_2026_06_02.md`.
  - **Phase 2 pivot (2026-06-02)** to **editorial similarity** because three audits showed upvote-rank label is broken: 82% top-quartile = oldest-quartile by post age; 22% of "top" rows are not code (markdown leakage); linter metrics fight the label. See `project_lc_editorial_similarity_pivot_2026_06_02.md`.
  - Editorial corpus: 3,519 problems with approach + complexity + anti-pattern, from doocs/leetcode (`reference_leetcode_editorial_corpus_for_norm_commentary.md`).
  - Embedding: BAAI/bge-code-v1; per-candidate score = max cosine to that problem's editorials.
- **MI ladder finding (LC, 70K)**: MI = 0.632 ± 0.006 vs TF-IDF 0.566 within problem. Per-language: Python wins (MI 0.663 > TF-IDF 0.641); **C++ TF-IDF 0.663 > MI 0.575** — bank is silently Python-flavored. C++ metric rebuild a410-a417 added +0.003 only; full rebuild plan in `project_cpp_metric_rebuild_2026_06_02.md`.
- **V+A+T plan for code_review**: deterministic ladder (Tier 1-4: metadata → diff parsing → static analysis → tests). Per-aspect tool map in `project_code_review_verifiability_plan.md`. Pending: Python predict-programs for code_review's 394 aspects (the verification_library equivalent of peer_review). Not yet built.

### 2.4 Press releases

- **Canonical file**: `press-releases/press_release_modeling_dataset_clean.csv.gz` (118 MB, Apr 8). **No group-split rebuild yet**.
- **Pre-existing build**: 128,131 rows (53,780 positive / 74,351 negative), labeled by top-domain news pickup.
- **Dense AUC (Llama-8B)**: **median 0.711 / max 0.712 at subset 1.0** (~60K rows). Saturated around subset 0.6-0.7. Llama-70B at subset 0.1 = 0.649, **no improvement** over 8B. Bradley-Terry variant underperformed pointwise.
- **Rubric methods plateau at ~0.53-0.58** (`project_press_release_results.md`). Iterative Autometrics: 25 iters, best ~0.585, final ~0.534. Metric Tree: grew only 1 node. Big gap from rubrics (~0.58) to dense (~0.71) suggests label-quality or task-difficulty ceiling, not lack of data.
- **Norm extraction in flight**: input = PR↔article pairs; rubric vocab built from R2 hierarchy (`project_norm_extraction_overnight_2026_06_02.md`). 2,366 pair sample size.

### 2.5 News homepages (newsworthiness)

- **Canonical file**: `news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz` (173 MB, May 12). 21,951 unique snapshots, 50/50 balanced. Use `snapshot_id` group key. Group-split done.
- **Label is homepage SPATIAL LAYOUT** — which articles get placed in which positions on the outlet's homepage. NOT clicks, NOT engagement (`project_news_homepages_label_correction.md`, 2026-06-01). This is an editorial-prominence decision.
- **Three confounds found and deconfounded** (`project_homepage_newsworthiness.md`):
  - SOURCE/DATE in text → outlet identity leak (BBC 34% top, WashPost 73% top).
  - Target headline position in context → leaks label directly.
  - Topic → certain topics systematically top.
- **Deconfounded AUC ≈ 0.753** with topic-balanced (LDA-50) version.
- **Implication of label correction**: the existing norm library (editorial-process norms about how to produce a story) is structurally mismatched to a task about how prominence is *assigned*. Text-only approach may be structurally bounded similar to code_review.

### 2.6 Notice and comment

- **Per-agency status** (Apr 17, `project_nc_pipeline_state.md`): CDC, USCIS, FWS, EPA, NC-overall extraction done; NOAA blocked on GPU. Canonical features (30 per task) clustered for 5 of 6. Optuna sweeps need Llama-70B-FP8 (blocked at the time).
- **Local-explanation peer-review-like AUC reported on overall NC**: not directly stated; sweep work pending.
- **Structural mismatch finding** (`project_nc_structural_mismatch.md`, 2026-06-02): agency responses overwhelmingly articulate reasons about the regulatory outcome, not the comment quality. So response-as-feedback only works for ~15-25% of pairs (~17-28K of V2's 112K). The rest argue substantive positions, not comment quality. **N&C may belong as a small filtered supplement, not a main task.**
- **Norm extraction in flight**: 3,644 RTC sections × 218 rubric vocab (88 general + 130 specific) (`project_norm_extraction_overnight_2026_06_02.md`).

### 2.7 Patents

This is the active 2026-06-05 work; treat all dates as live unless contradicted.

- **Task A: rough-draft prediction** — `first_draft_approved` (granted with no office actions). Predict at filing. Allowed: draft text + applicant IDS cites. NEVER use examiner cites (leaky — only exist if there was an OA).
- **Task B: final-draft prediction** — granted vs abandoned. Examiner cites OK (happen before outcome), though absence is weakly leaky; use `--require-oa` for strict variant.
- **Dataset built 2026-04-14** (`project_patents_first_draft_prediction.md`): 4,693,870 apps with draft text from PatEx + PatentsView (filing 2010-2024). First-draft approved 501,523 (10.7%). Granted 2.91M, abandoned 1.07M, pending 707K. Pipeline scripts `01_*.sh` … `06_*.py` in `datasets/patents/scripts/`.
- **HUPD baselines to beat**: published 64% accuracy on 50/50 balanced (Suzgun 2022). Our advantages: 4M training rows, Llama-8B with full claim context, citation features (untried).
- **V3 codegen ensemble (2026-06-02)** broke past the 0.563 prior ceiling: **V3 + V0 / LR = 0.6040 CV AUC** (`project_patents_v3_rebuild_result_2026_06_02.md`). First time patents crossed 0.60 on any code-AUC. 23 v3 codegen files; novelty caches (bigram, embedding, TF-IDF) under `outputs/v2_analysis/patents_*`.
- **§102 anticipation retrieval pipeline (active 2026-06-04/05)** — see `notes/2026-06-04__patent_supervised_pairs_methodology.md`:
  - `oard_citations.csv` is the ground truth: per-OA-per-cite-per-rejection records (~60M rows). `action_type` field gives clean §102 vs §103 split (4% / 17% / 79% IDS).
  - **v2 mistakes documented**: loose join (all examiner cites for §102-rejecting apps) → ~80% noise; FAISS index only covered 4.7M docs while training pairs needed ~20M; OA PDFs are scanned images (OCR needed); MS_WORD downloads need redirect URL extraction.
  - **v3 plan**: clean §102 pair extraction filtered to `action_type='102'`; full-text spec corpus (PVGPATTXT/PVPGPUBTXT, ~95%+ coverage 1976-2025); limitation-level queries; OA-text validation.
  - **Decision 2026-06-05**: drop training pairs whose cited prior art isn't findable in our corpus (rather than padding). Residual unfindable mostly design patents (~7%) + foreign refs.
- **Cross-task judge prompt bug also affected patents**: rejected 0.914 / approved 0.922 / AUC 0.507 / 87% of applicable cells score ≥ 0.5 (`feedback_judge_prompt_cross_task_bug.md`). Must re-score with patent-correct framing before any V/A/T number.

### 2.8 Math (AoPS, math.SE, mathlib, ProofBench)

- **Unified 2026-06-10** under `datasets/math/{stackexchange,aops,mathlib,scripts}/` (laptop) and `/lfs/.../datasets/math/{stackexchange,mathlib,combined}/` (sk3, with back-compat symlinks at the old `math-stackexchange` / `lean_mathlib` / `combined_math` paths). See `datasets/math/README.md` for the portfolio map.
- **combined_math exists on sk3** (built 2026-06-02 by a worktree agent, script salvaged to `datasets/math/scripts/build_combined_math_dataset.py`): Math.SE proof-tag-focused 98,151 rows + ProofBench 848 + IMO-GradingBench 1,000, at `datasets/math/combined/combined.parquet`.
- **mathlib revisited 2026-06-10** (`datasets/math/mathlib/README.md`): merge-vs-not stays dead (reject class = 56%+ abandonment, <10% content rejections — within-repo balancing can't fix that). Reframed primary label = **review friction among merged PRs** (patents-Task-A analog: `easy`/zero-CHANGES_REQUESTED vs multi-round merged), plus within-PR revision pairs (first push vs merged state + the review comments that demanded the delta) as the A-track corpus, content-rejections as a small held-out eval only. sk3 has 5,000 PR stubs (`prs_list.jsonl`); per-PR review fetch is the missing piece (~35K closed PRs total). Hypothesis: mathlib = max-thinned community → smallest C−B gap; either outcome is informative for the rubric critique.
- **Confounds + V routes quantified 2026-06-10** (`project_math_unification_mathlib_revisit_2026_06_10.md`): Math.SE earlier-answer-wins = 67.3% of 412K pairs (76.4% at score_diff≥5), length only 52.9% — time-matching needed, mirrors LC. Mathlib: correctness is CI-gated-constant among merged → discriminative V must come from first-push state + first-push CI failures; easy-merged closes in 0.9d median vs 5.6d non-easy. AoPS: AMC/AIME wiki answer keys give code-checkable correctness; wiki solutions give LC-style editorial-similarity y; thanks = taste within verified-correct subset.
- **2026-06-10 convergence (evening)**: Math.SE **v3.3 canonical** (`math_se_v3_3_propensity_balanced.csv.gz`, propensity-decile×year balanced, question-floor 0.469=dead, answer-over-floor margin +0.15, audits in `v3_3_audit/`); pairwise later-wins companion 51,066 pairs; mathlib **friction_dataset_v2** canonical (16,884 rows, size×prefix×year×assoc×topic matched, title 0.627=task-type signal, metadata 0.509, banned-columns list in `friction_audit/REPORT.md`); sympy verification pipeline production-ready (`verification/`, 13/13 tests); GPU watcher armed for Qwen-122B extraction (all GPUs busy; N&C dense killed at user request, supervisor cron PAUSED — user decides resume). Full state: `project_math_unification_mathlib_revisit_2026_06_10.md`.
- **2026-06-10 execution wave** (state details in `project_math_unification_mathlib_revisit_2026_06_10.md`): Math.SE v3 position-matched 100K dataset BUILT on sk3 (P(pos earlier)=0.736 audited, position dists equalized, group-split); mathlib ~37K-PR review fetch RUNNING on sk3 (resume-safe GraphQL; mathlib4+mathlib3 ≈ 50K PRs total → permissive friction labels, not rare-y); AoPS wiki scrape RUNNING on laptop (5,332 contest pages — sk3 IP is CF-hard-challenged for fresh contexts, laptop works); sympy verification pilot DONE — 76.7% of Math.SE answers have ≥1 checkable claim, 2 real refutations caught, extraction-not-compute is the bottleneck, Lean autoformalization adds nothing today.
- **AoPS scrape active** (`project_aops_dataset_collection.md`, launched 2026-05-30): Playwright + stealth bypass for Cloudflare; 8 sk3 workers over topic_id range [1, 3.6M). Preserves `thanks_received`, `nothanks_received`, `num_edits`, edit reasons. Style proxy = thanks/nothanks ratio (AoPS has no "accepted answer"). Output `/lfs/skampere3/0/alexspan/aops/raw/shards/<shard>__w<worker>.jsonl.gz`. ~10 days projected for full crawl.
- **Math elegance research framing** (`project_math_elegance_research.md`): 4 measurable dimensions per academic lit — elegance, profundity, clarity, precision/intricacy/utility (Inglis-Aberdein 2015 PhilMath, Johnson-Steinerberger 2019, Sa et al. 2022). Consensus across cultures and expertise levels exists. Math.SE upvote may track exposition clarity more than elegance — filter by `proof-verification`/`proof-writing` tags to concentrate signal.
- **Math.SE dense sweep**: `runs/math_se_sweep_llama8b/` exists; numbers not in `project_dense_model_sweeps.md`. Not modeled here.
- **Datasets considered**: prhegde/preference-data-math-stack-exchange (19K pairs); HF stack-exchange-preferences math subset (~300-500K); ProofBench 435 expert-graded; IMO-GradingBench 1,000; Open Proof Corpus 5,062 step-annotated; MathNet 30K problems × 47 countries.

### 2.9 Humor

- **DO NOT use New Yorker caption ratings** (`feedback_no_newyorker_captions.md`): crowd-worker annotated, not genuine humor judgment.
- Old draft in pre-rewrite version of this file proposed mean rating ≥ 1.3 binarization; that pipeline is now dropped.
- **Active humor source**: standup-Reddit scrape on sk3 (`/lfs/.../data/humor/standup_reddit/filtered_threads.jsonl`). Used in the 2026-06-02 norm-extraction sweep with 150-rubric vocab limit (vocab was 364, too big for context).
- **Hypothesized layout** (`project_tacitness_two_layers.md`): large B−A AND large 1−B — timing/voice is taste, comedy context essential.

### 2.10 Grant funding

- **NIH RePORTER** is funded-only — within-dataset A0/A1 matching is impossible (`project_nih_a0_a1_investigation.md`). SUFFIX proxy on funded new-year-1 grants gives "had a rejected A0" label on 32.8% of 506,587 grants, but only the funded version's abstract text is available. Cannot train a rejected-vs-accepted text classifier from RePORTER alone.
- **Open Grants** (`datasets/grant-funding/open-source-grants/processed/`): ~12 MB voluntarily-shared proposals with labels.
- **Status**: not modeled. No active dense sweep. Future paths blocked on getting actual rejected text (FOIA likely blocked by Exemptions 4/5; PI partnership).

### 2.11 Legal outcome

- **Dense sweeps exist** at `runs/legal_outcome_facts_only_sweep_llama8b/` and `runs/legal_outcome_facts_statutes_sweep_llama8b/`. Numbers not summarized in any current memory.
- **V+A+T expectation** (`project_verifiability_explainability_gaps.md`): statutory interpretation is heavily rule-based — best target for the Explanation-Refiner (NL→FOL) deep dive. Non-formalizable remainder (balancing tests, reasonableness) = articulable-to-taste boundary.
- **Status**: not actively worked. Standing as a "Tier 2 / programmatic codification" candidate.

---

## 3. Active method tracks

### 3.0 metric_implementer — ★ FIRST PRIORITY (2026-06-10): applies to ALL metrics, not just the tree

`methods/metric_implementer/` (README + `2026-06-10__design.md`). Metric **validity,
improvement, and articulability scaling**, abstracted out of the metrics_tree_infilling
review because it applies to every metric in the project — explicit online-rubrics, code
metrics, autometrics, metric_tree, infilling discoveries, articulation_star rationales.

- **Invariant: evaluate, never gate.** Construct fidelity (reliability, counterfactual
  validity, reconstruction, consistency, silver alignment) is *optimizable*; predictive
  contribution is *evaluation only*; fidelity failures (reliable+predictive but
  reconstruction-failing = instrument-level tacit) are *findings*, not discards. Every
  Goodhart/circularity risk we found traces to violating this.
- **Scorecard per metric** (6 measures, ~1.3K offline-batch judge calls each); validity
  certificates are relative: "no articulable simpler reading survives decorrelation across
  model families."
- **GEPA-style prompt-improvement loop**: textual failure artifacts (reconstructor names
  the misreading; missed counterfactual pairs become few-shots) drive reflective mutation;
  fresh counterfactual batches per round; cross-family acceptance holdout; predictive
  performance NEVER in the objective.
- **Articulability scaling laws**: run the optimizer under budget caps → frontier
  `fidelity*(m;B)`; per-metric budget-to-articulation B* with right-censoring →
  Kaplan–Meier "fraction of metrics articulable at budget B" per axis; metrics classified
  articulated / climbing / resistant. Axes: instruction tokens, few-shots, data budget,
  model tier (judge vs optimizer varied SEPARATELY, numerically anchored), inference-time
  compute, optimizer rounds, structural complexity (thin-vs-thick made quantitative),
  interaction order. This turns the operationalization-dependent-ceiling caveat
  (Noah 2026-05-28) into a measured, falsifiable curve — the principled escape from the
  reconstruction-circularity objection.
- **Why the earlier scaling attempt was noisy** (each fixed in design §3): measured
  mean-over-prompts not the frontier (padding ≠ articulation effort; prompt-content
  variance swamps the budget trend — the frontier-under-budget-cap estimand is the single
  most important fix); pooled metrics with heterogeneous saturation points → flat averages
  (fit per-metric curves, aggregate as survival); raw-AUC y-axis (use disattenuated,
  ceiling-normalized fidelity vs log B); confounded axes; non-nested data samples; one seed
  per cell.
- Cross-axis deliverable: iso-fidelity contours (Chinchilla-style) — instruction/few-shot
  exchange rates at fixed token budget; whether data substitutes for model tier →
  "articulation-optimal budget allocation" reusable by every other pipeline.
- Build order: scorecard measures first (immediately useful to every pipeline) → optimizer
  on 2–3 weak rubrics → **minimal viable scaling experiment**: peer-review, ~20 metrics
  spanning the coverage range, cheap axes only (instruction tokens + few-shots +
  inference-time compute), 3 budget points × 3 optimizer seeds, frozen eval set — validate
  the protocol shows clean scaling before paying for the data/model-tier axes.
- **2026-06-11 — implemented + experimental plan formalized.** Trial built on competitive
  code (registry, scorecard, GEPA loop, judge-tier scaling grid; first live runs, ~$0.35).
  Two-stage novelty sweep (incl. 5 Sonnet subagents over 5 literatures): **zero kills**;
  closest = Alur et al. NeurIPS '23/'24 (label-needed dual). Headline reframe: the
  **mechanization floor** (weakest judge tier at which an optimized rubric still preserves
  the construct) as a label-free measurement of a criterion's rules-vs-standards position.
  Plan: `methods/metric_implementer/2026-06-11__experimental-plan.md` (E0 known-answer
  ladder → E1 frontier descent + KM curve over a 24-criterion bank → E2 words/reader 2×3 →
  E3 static thickness estimator → E4 validity battery → E5 cross-community comparison;
  ~$100–250 total, gated). References: design doc §8.
- **2026-06-25 — ALPHA-PROBE (§12.1a of prompt-optimality theory) implemented.** The minimal
  runnable front half of `B_E-ATLAS`: freeze → breadth-sample → collide → estimate Heaps
  exponent `α` + coverage → **GO** (run full ATLAS) / **NO-GO** (`T` is the only global optimality
  statement) / **AMBIGUOUS** (scale up). Decides whether the reachable behavior space is
  low-dimensional enough to cover (the go/no-go to run BEFORE the expensive full atlas).
  `experiments/alpha_probe.py` (species accounting B1–B7: signature, frozen breadth_sample,
  noise-floor τ, single-linkage collide + CMI cross-check, f_j spectrum, good-turing/chao1/
  rarefaction/heaps-α/diversity-gap/coverage-interval, decide rule) + `experiments/run_alpha_probe.py`
  (K diverse HTTP proposer families — GLM/Qwen/Llama/Haiku, one frozen vLLM executor, GEPA-disjoint
  probe set `texts[gepa_reserve:]`) + `tests/test_alpha_probe.py` (11 planted ground-truth tests,
  green). **One honest deviation:** the coverage lower bound is the assumption-free
  Berend–Kontorovich `C_lo` (valid under any family dependence) + a pairwise-Petersen sensitivity
  point, NOT the Fienberg/LP max-positive-dependence bound. **The Fienberg multi-list log-linear
  POINT estimators are now implemented** (`coverage_fienberg`: Poisson-GLM IRLS on the 2^K−1 observed
  cells, empty cell = exp(β_0); independence = K-list Petersen-generalization, pairwise =
  dependence-corrected, identified for K≥4) — but the §C Fréchet **max-positive-dependence** bound is
  provably **unbounded** (n_empty→∞ as interaction→∞), so the assumption-free `C_lo` remains the only
  valid certificate; Fienberg is a sensitivity ladder above it. Cost note: assumption-free `C_lo` needs
  N≈1200 draws to clear 0.95. Live run gated on GLM quota (resets ~06-30); one-command sk3 launcher at
  `scripts/tools/launch_alpha_probe_sk3.sh` (idle-GPU detect, HOME/ZAI pinned, nohup, PID-scoped kill).
  Dry-run (FakeVLLM + mock proposers) passes end-to-end. Theory:
  `notes/2026-06-18__prompt-optimality-theory.md` §12.1a.
- **2026-06-25 — ALPHA-PROBE review follow-ups (Opus).** (1) **B–K constant verified** against the source
  (Thm 1, arXiv:1210.3248): `ε=√(log(1/δ)/N)` is the EXACT upper-tail constant (`c=1`), so N≈1200 for 0.95
  stands; rider — Thm 1 bounds `M_0` around its MEAN, so add `+1/N` Good–Turing bias to `C_lo` for full
  rigor (≈0.0008, negligible). (2) **Novelty-fallback wired**: AMBIGUOUS now auto-adds ONE tail-tilted iid
  novelty list (`_novelty_generate`, temp 0.9, biased to rarely-stated criteria) and re-decides on the
  5-list sample before "crank M" (R9 corollary); `--no-novelty-fallback` to disable; tests green (13),
  dry-run fallback path smoked. (3) **Proposer freeze confirmed** — temp-0.7 iid, `existing=[]`, fixed
  prompt (G1 holds). (4) Report the `C_Fienberg − C_lo` GAP as the *price of the independence assumption*;
  never headline Fienberg. (5) **Scope of the certificate** (theory §12.2.4): `C_lo` covers WITHIN union
  support only; no LM-list independence closes the support-completeness gap (shared-support ceiling, not
  correlation) — only positivity (untestable) / Lipschitz-impact (bounds residual impact) / a non-LM expert
  list can, and none certifies.
- **2026-06-26 — ALPHA-PROBE smoke (peer-review×llama-8b, M=90) = NO-GO + value-census planned.** Smoke:
  α≈0.975 (robust across τ: 0.98/0.98/0.83), 87% singletons, f-spectrum {1:78,2:1,10:1}, capture
  {qwen:23,llama:27,haiku:29,all-3:1}; discrimination real (between-sig L1 0.237≫τ, 0% near-const, τ₀≈0) ⇒
  high α is NOT a collapse/noise artifact. C_lo=0 is singleton-driven (f_1/N=0.87), NOT just the N=90
  concentration term — M=1200 makes it precise & LOW (~0.13), it will NOT flip to GO. Chao1=3122 / CMI
  uninformative at N=90 (f_2≈0). Capture-disjointness partly an under-sampling artifact at 30 draws/family —
  M=1200 tests persistence. Verdict: B_E (proposer-reachable, executor-distinguishable) inexhaustible for
  peer-review ⇒ skip the census atlas, use T. Full M=1200 launched (K=3, GLM out ⇒ Fienberg pairwise NaN);
  driver persists `…_alpha_probe_sigs.npz` (`--from-checkpoint`).
  **New direction — the VALUE-CENSUS (theory §12.3):** α measures BREADTH not HEIGHT (inexhaustible behaviors
  ≠ unbounded improvement). Swap the counting measure for the value measure and re-run: α_V (value Heaps
  exponent), breadth gap **α−α_V**, value missing-mass **MV_0 = (Σ_singleton v_s)/N** (= expected
  objective-gain of the next unseen criterion; value-weighted Good–Turing). New structural ingredient =
  ordering/diminishing-returns ⇒ extreme-value + submodular (the iid break, fixed by submodularity). Runs
  CPU-only on the persisted sample + labels Y (sketch §12.3-D); reuses the back half (`submodular_tail_bound`,
  `_missing_impact`). Expect α≈0.97 ∧ α_V≪1 = "rich but saturated" → recovers the interpretable map NO-GO
  seemed to remove. R/T bracket the optimum (R≤OPT≤T); α/α_V describe the SHAPE of the climb between them.
- **2026-06-26 — α is METRIC-level, not task-level (course correction).** We optimize prompts to tag
  METRICS (one prompt per R2 cluster), so α must be measured per-metric: for each cluster, breadth-sample
  its Ω (atomic units/criteria) and estimate that metric's B_E. The task-level α sweep (peer-review,
  math, news-homepages, patents, law, notice-and-comment, creative-writing, humor — **all α≈0.89–0.96
  NO-GO**) was the WRONG level (the whole-task criteria universe is always vast); do not cite it as the
  metric-level answer. Memory `feedback_alpha_probe_is_metric_level`. Built the metric-scoped probe:
  `alpha_probe.breadth_sample_metric` (Ω = r2_children + GEPA-if-artifact + within-metric free-gen,
  scored over task probe items) + `experiments/run_alpha_probe_metric.py` (executor loaded ONCE, loops
  R2 clusters, α per metric; CMI skipped for speed) + `scripts/tools/launch_alpha_probe_metric_sk3.sh`.
  Dry-run already discriminates (α 0.41/0.59/1.00 across 3 clusters, one <0.5). CW per-metric sweep
  launched (30 clusters, largest-first; biggest clusters have 65–175 children). GEPA source deferred for
  R2 clusters (CW GEPA registry has only 3 standalone metrics not mapping to the 371 clusters; GEPA Phase
  A is GLM-gated ~06-30) — Ω = children + free-gen until then.
- **2026-06-26 — value census / "recovered %": reframed M_i-based + UNSUPERVISED (§12.3 rewrite).**
  Originally `recovered % = R_full / H(Y)` (Y-supervised, deferred). The §12.3 reframe makes it
  **anchor-free**: `recovered % = R_full / H(M_i)` = fraction of M_i recovered, where M_i = the metric's
  OWN verdict (`_pyes` on the cluster `merged_description`), never the aggregate Y. So the value census
  is now the canonical unsupervised per-metric census (`v(s)=I(M_i;σ)`, `run_value_census.py` per-metric,
  reads M_i from the checkpoint). The "is it standard?" answer (uncertainty coefficient / NMI of a
  submodular-greedy feature set, an info-theoretic R²) still holds — now against M_i. (The earlier Y-based
  task-level run, 4–19% recovered / mostly FLAT-LOW, is SUPERSEDED.) Notes:
  `notes/2026-06-26__value-census-recovered-pct-future-work.md` (update banner added).
- **2026-06-26 — code converted to metric-level + Y-free (per rewritten §12.3).** `run_alpha_probe.py` is
  now the per-metric driver (loops `r2_groups` → `breadth_sample_metric` per cluster, Ω_i=children+GEPA-
  if-artifact+within-metric free-gen, computes+saves **M_i** in each checkpoint). `value_census.py`:
  `v(s)=I(M_i;σ)` not `I(Y;σ)`, `load_probe_labels`/Y removed, GV5 (metric scope) added.
  `run_value_census.py`: per-metric loop, reads M_i from checkpoint (no Y). Folded
  `run_alpha_probe_metric.py`→`run_alpha_probe.py` + `launch_alpha_probe_metric_sk3.sh`→
  `launch_alpha_probe_sk3.sh`. 20/20 tests green (α-probe + value-census, now M_i-based); dry-runs
  confirm checkpoints carry M_i (not Y) and the value census reads it. NOTE: the pre-conversion CW metric
  checkpoints lack M_i — re-run the metric α-probe to embed M_i before the value census.
- **2026-06-26 — CW metric α sweep result + the iid/coverage-bound debate (dual read adopted).** The
  pre-conversion CW sweep (30 R2 clusters, general bucket, full Ω=children+free-gen, τ=0.02) finished:
  **α min=0.680 (Spectacle) / med=0.981 / max=1.000, α<0.5 = 0/30** — no CW metric is coverable at full
  Ω, τ=0.02. Second-agent review flagged ONE real gap: children+GEPA are curated/conditioned (NOT iid),
  so the Good–Turing/BK `C_lo` coverage bound doesn't formally hold over the pooled sample, and they bias
  α DOWN (the §D/curated-coherence effect — why children-only α was 0.248 vs full 0.68). **User pushback
  (correct): free-gen isn't perfectly iid either** (within-`per_call` correlation + proposer bias) — so
  freegen-only is the *least*-biased read, not a rigorous fix. **Resolution: report BOTH, favor full.**
  Added a dual read to `run_alpha_probe` — full-Ω α_i/C_lo (favored descriptor) AND freegen-only α_fg/C_lo
  (~iid coverage read). Dry-run already shows the bias concretely: Spectacle α_i=0.496 (full) vs α_fg=0.898
  (freegen); Primacy 0.729 vs 0.968 — the curated children drag full-pool α down ~0.4. Value census needs
  NO iid (reconstruction contribution) so it stays on the full pool. Re-running the converted code on CW
  (30 clusters, largest-first, --no-glm) to get full-vs-freegen α + M_i-bearing checkpoints for the value
  census. (Open: per_call=1 free-gen would be cleaner iid at 10× API cost; deferred.)

- **2026-06-27 — CW M_i VALUE CENSUS result (the Codex-priority reconstruction read).** The converted
  α sweep **crashed 06-26 17:58** (vLLM `EngineCore_DP0 died unexpectedly`) at **10/30 metrics**; the on-disk
  `summary.json` is the OLD pre-conversion run (α min 0.680). Hung parent reaped, GPU freed. 11 converted
  checkpoints carry M_i → ran `run_value_census` (CPU, anchor-free) on them. **Result: 10 TRACKS / 1
  LONG-TAIL / 0 SATURATED; rec%=R_full/H(M_i) = 48–82, median 69, mean 67.** α_V≈α_i everywhere (gap −0.03
  to +0.10) → value is DENSE (no small essential set; many criteria each chip in) and still CLIMBING
  (MV₀ 0.04–0.18 > 0). So 67% is a **current-scale floor, not a certified ceiling** — both censuses now
  agree these large CW metrics are inexhaustible at feasible Ω; still can't certify a tight bound. **One
  metric leans certifiable: Catharsis of Pity/Fear** (lone LONG-TAIL, α_V=0.71<α_i=0.81, lowest MV₀=0.04;
  crisp Aristotelian core + thin tail — candidate to push toward SATURATED). Top-value criteria are
  on-target (Show–Tell→"dialogue advances plot"; Catharsis→"pity/fear spring from the Plot, not
  spectacle"), so the map is meaningful even without saturation. **Circularity caveat:** M_i = executor
  verdict on the cluster's own `merged_description`, criteria share that source → 67% is *recovery of the
  compiled metric*, inflated; honest irreducible residual >33% until de-circularized (held-out items +
  cross-source criteria). Codex review (`task-mqvl111b-55qqqe`, relayed): α is just a shape diagnostic;
  make value census + headroom (T_i, R_full/H(M_i)) the main path; relevance restriction = value-census
  with ε against M_i, NOT raw `I(s;X|Ω)>0`. Next levers: push Ω scale (larger M_freegen, per_call=1) until
  α_V plateaus to find the true recovery ceiling; de-circularize M_i. Output:
  `outputs/value_census_cw/value_census_summary.json`.

- **2026-06-27 (cont.) — launched the CW multi-level SATURATION hunt (R3 + R1-sample + R2-all).** User
  asked to run the value-census experiment across ALL CW R2 + all R1 to find more SATURATED metrics.
  Scoping reality: CW hierarchy sizes are **R1=1700 families (median 1 leaf ≈ L0 atoms — DEGENERATE),
  R2=371 (med 10 leaves, the meaningful set), R3=70 (coarsest, med 29)**. User decision: **R2 all 371
  (full, resume from 11) + R1 sample ~150 + R3 all 70** ≈ 49 GPU-hr. CODE: added R1/R3 accessors to
  `mine_clusters` (`_expanded_groups` generalizes R2/R3; `r1_groups`/`r1_children`/`_cid2rep` for the
  `r1_families` format) + `--level {R1,R2,R3}` and `--skip-existing` to `run_alpha_probe` (level-encoded
  ckpt names `creative-writing_{level}_metric{gi}_sigs.npz`; `--skip-existing` recomputes α from the
  cached sigs → a re-run resumes cleanly after a crash). Synced R1 source files to sk3
  (`r1_families_creative-writing.json`, `clusters_creative-writing.json`, `canon_all_real_forms.jsonl`
  were local-only). Dry-run validated R1/R2/R3 E2E + the resume path. Preserved the 11 prior R2
  checkpoints (renamed M_i-bearing unprefixed → `R2`-prefixed so `--skip-existing` resumes them).
  **Supervisor `scripts/tools/cw_multilevel_alpha_supervisor.sh`** (PID 809673): sequential R3→R1→R2 on
  ONE GPU (excludes reserved GPU 0), per-attempt GPU re-detect (migrates if held GPU lost) +
  `--skip-existing` + `timeout --kill-after` (reclaims a hung/D-state engine) → crash-resilient; ends
  with a value census over ALL checkpoints → `outputs/value_census_cw_all/`. Currently waiting for a free
  GPU (cluster saturated). Milestone watcher `bkebqgqcl`. Hypothesis under test: SATURATED metrics (few
  criteria recover M_i) hide in SMALL-leaf clusters — the 11 largest had 0; the small-leaf R2 tail + R1
  singletons are where to look.

- **2026-06-28 — saturation hunt INTERIM result (231 metrics: R3 70 + R1 150 + R2-largest 11): 0 SATURATED
  at ANY granularity → hypothesis REFUTED for CW.** R1 singletons (smallest): 150/150 TRACKS, α_V med
  0.99, rec med 79% (a few 0%-recovery = degenerate near-constant M_i, noise). R3 broadest: 70/70 TRACKS,
  α_V med 0.98, rec med 71%. R2-largest: 10 TRACKS + 1 LONG-TAIL (Catharsis), rec med 69%. Behavior α
  also 0.87–1.0 everywhere (R1 α_fg≈α_i≈0.98 — the 1 child doesn't drag α). **No CW metric — singleton to
  broadest — has a small essential criterion set; value is always DENSE (TRACKS) and still climbing
  (α_V≈1).** Recovery stable ~70–79% across levels (slightly HIGHER for narrower metrics: R1 79% > R3 71%),
  ~25% residual. Implication: the value census yields **no certifiable saturation bound for CW at this Ω
  scale** — α_V≈1 means recovery hasn't plateaued. Path to a real bound = push Ω until α_V plateaus
  (Catharsis Ω-scale-up, deferred #1) OR accept ~25% as the practical residual floor (M_i-circularity
  caveat applies → honest residual larger). R2-all (360 small-leaf) still running but conclusion is robust
  (singletons already cover the small end). Outputs: `outputs/value_census_cw_partial/`. Watcher
  `bkebqgqcl` for R2-all completion.
- **2026-06-29 — saturation hunt FINAL (R2-all done): 591 metrics (R1 150 + R2 371 + R3 70), 0 GENUINE
  SATURATED.** The 1 flagged SATURATED = "Previously unpublished work only", an ELIGIBILITY cluster (not a
  quality dim) with near-constant M_i → rec=0%, α_V≈0, MV0=0 — the verdict's `α_V<0.3 ∧ MV0≤ε` rule
  mis-fired on degenerate M_i (0% recovery ≠ saturation). **BUG to fix:** SATURATED verdict must guard
  against degenerate M_i (rec≈0 / H(M_i)≈0). Real non-TRACKS: just Catharsis of Pity/Fear (LONG-TAIL,
  α_V=0.71, rec 56%) + Spectacle⊂Plot (α_V=0.71, rec 54%) — the closest-to-saturation pair, both still
  TRACKS-ish. **Definitive: no CW metric saturates at any granularity; saturation route to a bound is
  dead for CW.** Next: Catharsis Ω-scale-up (approved #1) — does α_V plateau at 10× Ω? Output:
  `outputs/value_census_cw_all/value_census_summary.json`.
- **2026-06-29 — Catharsis Ω-scale-up RESULT (the decisive bound test). M_freegen {60,300,600} on
  llama-8b, per_call=10.** α_i / α_V / rec%: 60→0.809/0.711/56%, 300→0.925/0.889/61%, 600→0.952/0.931/61%.
  (M=60 reproduces the 591-sweep exactly → apples-to-apples.) **TWO READS DIVERGE: (1) α_V does NOT
  plateau — climbs 0.71→0.89→0.93; value species inexhaustible; Catharsis transitions LONG-TAIL→TRACKS.
  So the species/saturation route yields NO bound. (2) BUT absolute recovery rec% DOES plateau at ~61%
  (56→61→61) across 10× Ω → a CANDIDATE recovery ceiling (~39% residual), the first convergent-bound
  signal.** Synthesis: the bound, if any, is on ABSOLUTE recovery (rec%=R_full/H(M_i)), NOT species-counting
  (α_V). For the closest-to-saturation metric, ~61% recoverable, no more, regardless of Ω. CAVEATS: 1
  metric (does rec% plateau generalize + at what level?); H(M_i) ceiling not true T_i=I(M_i;X); M_i
  circularity inflates the 61% → honest residual larger. Also fixed the degenerate-M_i SATURATED mis-fire
  (run_value_census guard: frac<0.05 or H_M_i<0.05 → DEGENERATE); canonical cw_all now 588 TRACKS / 1
  LONG-TAIL / 2 DEGENERATE / 0 SATURATED. Outputs: `outputs/catharsis_scaleup/M{60,300,600}/vc/`,
  `outputs/value_census_cw_all/`. Code: `--target-name` added to run_alpha_probe (isolate one metric).

- **2026-06-30 — ★ CAPTURE-RECAPTURE B_E NOW IDENTIFIABLE (the species route DOES yield a bound; the
  06-29 "no bound" was a species-definition artifact).** Pivot: stop all species counting on MARGINAL
  `collide` (f(x)=f(y), single-linkage at τ on mean-L1 P(YES) sig) → it splits paraphrases into new
  species (~95% singletons → Chao1 diverges → α≈0.9 "inexhaustible", B_E swings 4→627 over τ: knife-edge).
  Use CONDITIONAL / NON-OVERLAPPING species: a candidate e is the SAME species as core unit k iff
  `I(X_k;X_e) ≥ cmi_thresh·H(X_e)` (exact binary MI, closed-form CPU). Greedy → paraphrases collapse
  by construction. `alpha_probe.conditional_species` / `two_list_crc` / `crc_bootstrap` /
  `conditional_crc_report`. **Catharsis M600:** B_E=20.8±3.8 (15-split Lincoln-Petersen bootstrap),
  Chao1(within)=22.5, coverage=0.905 — Chao1≈LP (two independent estimators converge ⇒ saturation is
  real, NOT a clustering artifact). Across 611 CW ckpts (n=60–90): B_E median 8.0 (max 42, NONE >50),
  coverage median 0.914; heterogeneity real (within-metric noise/signal=0.39) but confounded w/ small-N
  LP variance (only Catharsis has n=600). **Upper bound = B_E (richness); lower = D_obs; coverage spans.**
  Two-list LP is NON-circular (cores on iid half A, recapture half B). Validity gates STANDARD:
  `order_stability` (permute greedy core-build order; Chao1 22±3, C_lo 0.930±0.001 despite core-identity
  Jaccard 0.13) + `form_invariance` (re-score criteria under question/boilerplate/reorder/suffix
  reformulations, drift vs τ₀). Joint-CMI cross-check: pairwise-max ≈ full-joint (D median 5 vs 5, ratio
  1.0, corr 0.69; joint merges more only at high-N: Catharsis 17→11). **Open knob:** cmi_thresh (0.15 =
  stable plateau) still hand-set → pin via irreducibility (decompose-then-re-orthogonalize), TBD.
  Analysis driver: `experiments/crc_analyze.py` (--dir / --scaling). Mem: project_capture_recapture_identifiable_BE.
- **2026-06-30 — SCALING-RUN LAUNCH: high-N (M=600, probes=300) capture-recapture across the open-weight
  executor ladder on CW R2 (top-12 metrics, --largest-first).** `scripts/tools/crc_scaling_supervisor.sh`.
  Ladder: llama-3.2-3B / llama-3.1-8B / gemma-4-31b / llama-3.3-70B-FP8 / qwen3.5-122B-A10B-FP8 (3B→122B,
  3 families). GPUs 1,2,3,7 (0=ahmedah, 4-6=animjha off-limits); 1 job/GPU, --skip-existing resume.
  gemma-2-27b DROPPED (cache stub, shared-cache write-locked). gemma-4 via gemma4 env (sklearn/scipy
  pip-installed into it). form_invariance default-on in run_alpha_probe (permutation test STANDARD).
  Own-freegen per model (criteria differ slightly across models — acceptable B_E noise at n=600; fixed-
  criteria --rescore-from is a future clean-criteria pass, deprioritized #84). Multi-task extension: only
  humor (285 R2) + patents (7) are R2-ready; news/math/coding/peer_review/PR/N&C have NO R2 hierarchy in
  any bucket → blocked on hierarchy-building.

### 3.1 articulation_star

`methods/articulation_star/`. Vanilla STaR + ranked fallbacks. Source: `project_articulation_star_*.md`.

- **2026-05-30 smoke green on creative_writing**: end-to-end loop works (generate 40 rationales → judge → train LoRA adapter); judge accuracy 40%, worse than always-negative baseline. **Judge sentiment bias surfaced as the v1 blocker** — judge predicted "upvoted" for 34/40 because rationales lean positive.
- **2026-05-30 overnight launch**: 3-iter STaR loop, logprob-scored contrastive filter, balanced 1500/label, 4-way DP gen.
- **2026-05-31 v2 launch**: weak judge dropped to Llama-1B for a real strong-weak gap; auto + LLM-judged leakage detection in master script.
- **2026-05-31 outcome**: experiment INVERTED — 1B at logprob mode hit 68.3% on rationale-only, while Qwen-122B at no-thinking max_tokens=1 only hit 53.2%. Weak > strong → contrastive design broke; kept set was 5576 all-y=1, run died at combine.
- **Leakage findings on v1**: test acc +8.2pp held-out, auto specificity 2×, sentiment vocab DOWN, template hits 5× UP (still small absolute). Composite LLM-judged leakage +0.04 across iters — driven by template growth, not sentiment.
- **Fallback ladder ordered by complexity** (`project_articulation_star_fallback_defenses.md`):
  1. Contrastive weak/strong judge filter; 1b. logprob-based scoring.
  2. Seed rationales from existing metric/rubric labels (use 434K v6 judge pair labels).
  3. Held-out evaluation judge.
  4. Counterfactual-swap probe.
  5. Process reward on groundedness/specificity (rStar-style; `project_articulation_star_rstar_followup.md`).
  6. Enrich the prediction target.
- **Next experiments proposed** (`project_articulation_star_v2_run.md` end): CoT-strong + logprob-weak contrastive; dedicated distilled leakage classifier; anti-template filter in keep rule.

### 3.2 metric_tree + metrics_tree_infilling

- **metric_tree** (`methods/metric_tree/`, algorithm 2 in `project_three_algorithms.md`): every example scored down entire tree; restructuring pipeline (generate → global ternary YES/NO/NA score → dedupe → rebuild → gap-fill → repeat). Prediction = base rate at leaf. Recent commit `6a7ba41`: BFS traversal, proposer retry, leaf regression models.
  - **Concern carried forward**: proposer generates generic metrics at all depths (`project_metric_specificity.md`); needs partition-specific steering.
  - **Restructuring 4-phase design**: robust metric generation → ternary scoring → rebuild over score matrix → gap-fill (`project_restructuring_pipeline.md`).
- **metrics_tree_infilling** (`methods/metrics_tree_infilling/`, built 2026-06-05, `project_metrics_tree_infilling.md`):
  - Gap-detecting MOB classification tree + LLM feature discovery.
  - Pure-Python port of R `partykit::glmtree` (M-fluctuation test, sup-LM/χ²). **Permutation null** replaces asymptotic Brownian-bridge p-value per user steer ("get as close to R as possible").
  - Metrics live in `datasets/<task>/online-rubrics/{gpt,claude}-parsed/**/*.json` → `extracted.rubrics_metrics`. ~75K raw entries for peer-review; caller caps to M.
  - Validation: `tests/validate_against_partykit.py` — 100 planted scenarios. Self-check: detection 0.81 / FP 0.0 / cutpoint err 0.008 on 20 scenarios. R parity runs iff Rscript+partykit present (not on laptop).
  - Honest 70/30 discover/test split; keep/drop guards on test side.
  - Documented limitation (§9): cannot find a missing interaction of absent features (root-level XOR).

### 3.3 autometrics (iterative)

`methods/autometrics/`. Algorithm 1 in `project_three_algorithms.md`. Flat feature generation, all examples scored on all metrics, LR / dense prediction. No tree. Architecture and bug history in `autometrics_architecture.md`. Pre-existing track; not the primary push as of June 2026.

### 3.4 verification_library

`methods/verification_library/`. Per-aspect Python codegen (`predict(text) -> float`).

- **Built for peer_review only**: 571 files at `runs/validity_full/v2/peer_review/codegen_claude/` named `a{ID}_v{0,1,2}_{flavor}.py` (`reference_codegen_per_aspect_programs.md`). Empty for all other tasks.
- **Direction-2 recipe** (`project_verification_pipeline_recipe.md`): Llama STaR → dedup (kmeans) → Qwen codes all features (multi-pass, code-stub forcing) → hierarchy → evaluate → annotate.
- **Status**: experimental, tuning. Llama self-assessment 28% thin / 72% thick. Qwen forced prompt got 42% success rate single-pass; multi-pass retry handles most degeneration.
- **code_review-specific need**: build per-aspect programs for code_review's 394 aspects (`project_code_review_verifiability_plan.md`); never run. Plus Tier 1-3 deterministic ladder (metadata → diff parsing → static analysis tools per norm category).

### 3.5 local_explanations

`methods/local_explanations/`. Two approaches + baselines (`project_local_explanations_design.md`).

- **A: Rationalization + prior calibration**. Tell model label, ask for features, subtract p(z|y) prior.
- **B: STaR-Local (amended)**. Blind extraction; incorrect predictions NOT discarded — used to downweight misleading features via 2×2 (correct/incorrect × winning/losing) weight matrix.
- **Required baselines per dataset**: no articulation (raw LLM zero-shot); rubric + datapoint in LLM (LLM predicts directly using rubric).
- **Peer-review locked operating point**: tw=0 / eps=0 + LLM dedup (`project_local_explanations_clustering_findings.md`). Supervised UMAP fragments same-concept-different-label.
- **Optuna sweep design** ready for execution once Llama-70B-FP8 free (`project_local_explanations_hyperparam_sweep.md`): scaling-law sweep, weight matrix, clustering variants, predictor variants. Step 1+2 extraction is cached, so Step 3+ trials are cheap.
- **Long-tail diversity follow-ups** (`project_local_explanations_followups.md`): two-pass extraction, anti-pattern few-shots, lift-based filtering, per-example targeted prompting.

### 3.6 Dense reward model sweeps

`methods/dense/` and `runs/<task>_sweep_llama8b/`. Llama-8B + LoRA. Source: `project_dense_model_sweeps.md`.

| Task | Full N | Saturation point | Plateau AUC | Notes |
|---|---:|---|---:|---|
| press_release | ~60K | 0.6-0.7 (~36-42K) | 0.71 max | Saturated. 70B no gain. BT underperforms pointwise. |
| peer_review | ~56K | 0.5-0.6 (~28-34K) | 0.77-0.78 | Saturated. Strongest of 4. |
| creative_writing / litbench | ~70K | not reached | 0.90 max at 1.0 | **Still climbing.** Variance high. |
| code_review | ~113K | (likely similar) | 0.78 at 0.5 | 1.0 run never finished writing metrics; needs rerun. Also leak-contaminated; see §2.3. |

**Operating rules from the memory**: always run 3-5 trials, report median + max; variance high; Llama-70B at subset 0.1 matched 8B on press_release; Bradley-Terry pairwise underperformed pointwise.

**Queue supervisor** keeps the dense sweep alive on sk3 across reboots via cron (`reference_sk3_queue_supervisor.md`).

### 3.7 Rubric clustering pipeline (leaf re-clustering)

Locked recipe (`project_rubric_clustering_pipeline.md`, 2026-05-18, tau-dropped 2026-05-19):

1. Canonicalize leaf rubrics with Llama-3.3-70B BF16 → 53,413 canonical forms (11 tasks × 3 buckets).
2. v6 graded judge (0/1/2 = unrelated / related-different / same) labels 434K pairs.
3. Distill per-task: LoRA bge-large (CoSENT) + ModernBERT-base cross-encoder.
4. Candidate net: LoRA-bge cosine top-200 kNN. 99.5% true-same coverage.
5. CE re-scores candidates. Hybrid affinity = **0.5·CE + 0.5·cos** on candidates, 1·cos elsewhere.
6. Average-linkage agglomerative, cut at **tau 0.825**.

Operating point: FP ~10.6% / FN ~9.8% at tau 0.825 (was tau 0.92 = 3.1% / 12.6% — tau dropped for more compression). FP is all judge=1, never judge=0.

### 3.8 Refactoring algorithm (unified V/A discovery)

`project_refactoring_algorithm_idea.md`. Library convergence as the measurement.

- Per-example programs → refactor into code library (verification side).
- Per-example z₊/z₋ rationales → refactor into principle library (articulation side).
- Natural measurements: library size at convergence; rate of stabilization; main() complexity at convergence (= taste residual); migration rate articulation→verification (= thin/thick boundary).
- Predicts the **verifiability cycle**: norm crystallization thins rules over time, but paradigm shifts / Goodhart pressure / novelty reset the cycle.

---

## 4. Active sweeps / overnight queues

- **Dense reward model sweeps** on sk3, cron-supervised. Press release, peer review saturated; creative writing still climbing per `project_dense_model_sweeps.md`. Code review 1.0 needs rerun.
- **Qwen-122B norm extraction** across peer_review → code_review → N&C → press_releases → humor. **Switched from OpenAI-compatible server to batch mode 2026-06-04** per `feedback_never_openai_server_for_bulk.md` (10-100× slower in HTTP mode). Sequential launches, one task at a time, in batch mode at `/lfs/.../scripts/llama_norm_extraction/run_sk3_batch.py`. ~48K rows preserved across resume. First batch (06-04 ~20:54): peer_review on GPU 1, batch_size=1000.
- **AoPS scrape**: 8 workers ongoing since 2026-05-30, ETA ~10 days. PIDs 787685..788900.
- **WritingPrompts comment scrape ongoing**, multi-day, cursor 2017-09 as of 2026-06-02.
- **AST scrape DONE** (2026-06-02): 40,026 files, 151,547 posts parsed. Integration into humor extraction is task #125.
- **Patents §102 v3**: spec-corpus indexing + clean pair extraction. As of 2026-06-05, 60 spec_chunks done per task scope; ParquetWriter fix landed; autorun resumed. (Detail per task brief; not yet captured in a memory file — the active state for this thread.)

---

## 5. Open problems

### 5.1 judge_0p5 noise (relaxed applicability marks ~69% of cells 0.5)

`project_judge_0p5_noise_filtering.md`, 2026-06-01.

- On creative_writing under relaxed applicability, mean score=0.5 rate across applicable cells = **68.7%**. 26 of 60 aspects covered so far have >70% 0.5 rate.
- A feature that's 0.5 for most artifacts dilutes signal — adds noise.
- Filtering strategies to test (none evaluated yet): hard threshold drop; judge-confidence weighting; treat 0.5 as missing + indicator; one-hot {0, 0.5, 1}; MI pre-select; binary re-prompt.
- May ultimately be subsumed by logprob scoring (fallback 1b in articulation_star).

### 5.2 v6 judge cross-task prompt bug

`feedback_judge_prompt_cross_task_bug.md`, 2026-06-02. Confirmed on creative_writing and patents.

- `runs/validity_full/v2/{task}/judge_system.txt` literally says "You are an expert peer-review evaluator scoring scientific papers..." regardless of task.
- Patents judge: rejected 0.914, approved 0.922 → AUC 0.507 (random).
- Rule: grep `judge_system.txt` for "peer-review"/"scientific papers"/"research articles" before trusting any v2 judge AUC. Affected tasks must be re-scored with task-correct framing.
- Sister issue: narrow vs relaxed applicability framing (§ 5.1).

### 5.3 Tacit knowledge as a cross-task constant

`project_tacit_knowledge_measurement.md`, 2026-05-29 (thinking stage; not active work).

- Want: a measure of tacit knowledge that's constant across tasks for comparison.
- Candidate directions: expert-LLM minus lay-LLM gap on task-specific factual probes; the C−B gap normalized by (tacit + articulable); benchmarks of task-specific knowledge.
- Tension with Noah's operationalization-dependence point: C depends on what counts as a replicable expert.

### 5.4 Tree structure not absolute

Wittgenstein consequence (`project_thin_thick_rules_philosophy.md`): no privileged decomposition of practice into rules. Implies metric-tree structure is *one possible articulation*, not *the* articulation. Don't seek the "correct" tree; optimize for thin-rule coverage given task frequency (Kaplow).

### 5.5 Bottom-up failing to structure

Dreyfus consequence: experts don't use hierarchical rules; their knowledge is holistic/situational. Forcing hierarchy is an approximation. Alternative: flat overlapping criteria with varying thickness (closer to legal standards).

### 5.6 Abstraction taxonomy unresolved

`project_abstraction_types_research.md`. Current working list: 8 types (Generalization, Composition, Condition/exception, Dependency, Scope constraint, Polarity pattern, Severity/weight, Default expectation). Potentially missing: causal/entailment, attack-type distinctions (ASPIC+), deontic modalities, compensatory/repair, temporal/ordering, analogical, epistemic status, belief dynamics (AGM). Not yet resolved how to use these in the refactoring clusterer.

---

## 6. Theory / philosophy threads (out of scope for current paper)

- **Norm emergence** — imposed vs emergent norms (`project_norm_emergence_future.md`). Thick emergent → thin imposed (e.g., "papers should be reproducible" → NeurIPS checklist). N&C is uniquely set up to study this (both top-down rules and bottom-up community responses). Saved for future paper.
- **Verifiability cycle** (within `project_refactoring_algorithm_idea.md`): the frontier between V/A/T isn't monotonic; paradigm shifts / Goodhart / community expansion revert verifiable items back to taste.
- **Faultless disagreement** (Kölbel, MacFarlane; Noah's recommendation): irreducible disagreement with no fact of the matter — theoretical grounding for the taste residual.
- **Tacit knowledge** as a separate measurement program (§ 5.3) — not active.

---

## 7. Recent meetings / decisions log (May-June 2026 timeline)

- **2026-04-09**: Local explanations design fixed (two approaches + baselines). `project_local_explanations_design.md`.
- **2026-04-10**: NIH A0/A1 ruled out as a rejection-text source. `project_nih_a0_a1_investigation.md`.
- **2026-04-14**: Patents dataset built from PatEx + PatentsView, 4.69M apps. `project_patents_first_draft_prediction.md`.
- **2026-04-17**: NC pipeline state — peer-review AUC 0.6141 (sweep best); 5/6 NC tasks extracted; NOAA blocked. `project_nc_pipeline_state.md`.
- **2026-04-20 to 04-25**: Homepage newsworthiness deconfounded; AUC drops from 0.96 to ~0.75 with topic-balancing. `project_homepage_newsworthiness.md`.
- **2026-05-12**: Clean group-split builds landed for creative_writing and news_homepages. `reference_clean_datasets_per_task.md`.
- **2026-05-18**: Rubric clustering pipeline locked (recipe + tau 0.92 operating point). `project_rubric_clustering_pipeline.md`.
- **2026-05-19**: Tau dropped to 0.825 for more compression at user request.
- **2026-05-28**: **Noah meeting**. Framing pivot to "communicable community-agreeable metrics," intersubjective rather than subjective, articulability ceiling is operationalization-dependent.
- **2026-05-30**: articulation_star v0 smoke green; judge sentiment bias surfaced; AoPS scrape launched.
- **2026-05-31**: articulation_star v2 launched with Llama-1B weak judge — design inverted (weak > strong on logprob). Leakage analysis on v1: +8.2pp test acc, +0.04 composite leakage from template growth.
- **2026-06-01**: v2relax re-scoring pipeline for creative_writing (`project_cw_relaxed_appl_v2_2026_06_01.md`). Discovered cross-task peer-review framing bug AND narrow-applicability bug. RF AUC moved 0.541 → 0.636 on fixed split. cells DB refresh 670 → 2611 datapoints (`project_cw_pump_2026_06_01.md`). news_homepages label corrected to "spatial layout" — not engagement.
- **2026-06-02**: **Code dataset pivot**. Demote code_review_dense_4096tok; build LeetCode + CR.SE combined. Within hours, LeetCode push pivots again from upvote-rank to editorial similarity (`project_lc_editorial_similarity_pivot_2026_06_02.md`). C++ metric rebuild planned. Patents V3 ensemble crosses 0.60 AUC for the first time. Overnight norm extraction across 5 tasks launched.
- **2026-06-04**: Patent supervised-pairs methodology documented after v2 retriever debugging (`notes/2026-06-04__patent_supervised_pairs_methodology.md`). All OpenAI-server bulk runs killed in favor of batch-mode runner.
- **2026-06-05**: `methods/metrics_tree_infilling/` built — gap-detecting MOB tree + permutation null + LLM feature discovery; all 6 unit/e2e tests pass. Patents v3 §102: drop unfindable training pairs decision.
- **2026-06-10**: metrics_tree_infilling review (10/10 tests, but all LLM seams scripted; never run on real data) → `methods/metrics_tree_infilling/2026-06-10__next-steps-plan.md`. Metric-validity framework abstracted into **`methods/metric_implementer/` — declared FIRST PRIORITY** (§3.0): scorecard (reliability / counterfactual / reconstruction / consistency / silver), evaluate-never-gate invariant, GEPA-style prompt improvement, articulability scaling laws (frontier + survival curves) as the answer to the reconstruction-circularity objection.
- **2026-06-11**: **NLRB ALJ→Board corpus acquisition launched** (`datasets/legal-outcome-prediction/nlrb/README.md`). x = ALJ decision text (pre-Board → exogenous), y = Board disposition; BVA→CAVC analog with Wright Line thin-rule/thick-input structure. nlrb.gov list pages enumerate both sides; case-number join validated 14/15 samples; ALJ list's own "Board Outcome" column is stale for ~40% of 2011-22 rows (board-list join is authoritative). 5,272 ALJ decisions 1997–2026, ~3,400+ joining Board decisions; ~85–90% affirm among reviewed → graded affirm_modify / affirm_in_part / reverse is the realistic target; "Adopted/No Exceptions Filed" is a separate stratum. Polite scraper running on sk3 (`nlrb/scrape.log`).
- **2026-06-12**: **FLSA v3 relabel + Scenario A ladder; NLRB Stages 1-3 + TTAB unified done.** Full FLSA posture-aware relabel (11,177 rows, Llama-70B, 100% parse) exposed v2 label noise (1,139 false positives, 164 flips); user chose Scenario A (SURVIVES→1) → balanced pool 7,140, ladder lexical 0.742 / BERT 0.691 / Llama-8B **0.797** — cleaner labels raised the ladder AND widened the tacit gap (0.047→0.055); BERT fell below lexical (truncation > contextual gains). NLRB: deduped 3,242→2,995 unified pairs (keep earliest Board response), Stage-3 span-grounded V-extraction (`nlrb/v_extract.py`), first ladder rungs V 0.575 / lexical 0.654 / Llama-8B 0.536@1024tok→0.630@4096tok (still below lexical — long-doc truncation artifact; 8192 probe running). TTAB re-emitted into unified schema (2,093 records; caught 116 docket-table leaks — "bd decision: sustained" inside complaint slots — + 2 mis-tagged decisions; quarantined via `x_leak_risk`). See `datasets/legal-outcome-prediction/EX_ANTE_PIPELINE.md`.
- **2026-06-13**: **Legal ex-ante expansion + V/A layer (ultracode session).** Three new corpora assembled & vetted dataset-first (trademark 79,936 balanced; DOL OALJ→ARB/BRB 8,840; MSPB→CAFC 598) + PTAB 19,239 / CAVC 20,770 / NLRB 2,995 / TTAB 2,093. **Uniform deconfounding** (`exante_scrub.py`+`build_modeling_pool.py`): strip temporal/identity/layout-number leakage (preserve §-refs), right-censor, entity-group split — caught a real leak on every pool (PTAB era-self-ID 0.759→0.625, NLRB transcript line-numbers, TTAB date fragments, trademark "(Based on Intent to Use)" posrate-0.00). **Doctrine workflow** (26 agents, adversarial): 75 statutes/cases/principles + **403 thin checkable metrics** → 13 banks (293→696 metrics) + `THIN_METRIC_REGISTRY.json`. **171 V-extractors wired+leak-verified** (`v_extractors/`, 12-agent codegen, 2 leaks quarantined). **V-rung ladder (deconfounded, group-split): the V→lexical gap orders by doctrinal thinness** — PTAB .041 (mechanical) < TTAB .082 < NLRB .107 < DOL-ARB .124 < DOL-BRB .134 (medical-weighing). A-rung (LLM-judge on doctrine banks) in progress. Full ledger: `datasets/legal-outcome-prediction/VERIFIABILITY_SCORECARD.md`; see [[project_exante_expansion_2026_06_12]]. Ops: regex-backtracking-vs-SIGALRM + pypdf-hang lessons banked.

---

## Appendix: where to look first

| Looking for | Go to |
|---|---|
| Metric validity / prompt improvement / scaling laws (FIRST PRIORITY) | `methods/metric_implementer/` (§3.0) |
| The V+A+T framework | `project_verifiability_explainability_gaps.md` |
| The three-AUC tacitness layering | `project_tacitness_two_layers.md` |
| Noah framing | `project_noah_meeting_2026_05_28.md` |
| Philosophical grounding (Daston / Wittgenstein / Dreyfus / Polanyi) | `project_thin_thick_rules_philosophy.md` |
| Canonical dataset paths | `reference_clean_datasets_per_task.md` |
| Dense AUC ceilings per task | `project_dense_model_sweeps.md` |
| sk3 paths and queue supervisor | `reference_sk3_paths.md`, `reference_sk3_queue_supervisor.md` |
| vLLM recipes (BF16 Llama-70B, FP8 Qwen-122B) | `reference_sk3_vllm_bf16.md`, `reference_qwen35_vllm_sk3.md`, `reference_fp8_vllm_sk3.md` |
| sk3 v2 task datasets | `reference_v2_task_datasets.md`, `reference_sk3_v2_datasets.md` |
| Cells DB | `reference_cells_db.md` |
| v6 judge pair labels (434K) | `reference_norm_embed_pair_labels.md` |
| Per-aspect Python predict-programs | `reference_codegen_per_aspect_programs.md` |

## 2026-06-12 — Taste taxonomy locked + collection sprint started

- **Taxonomy** (notes/2026-06-12__taste-taxonomy.md): (explicit|revealed) ×
  (expert|crowd) grid, citations→C in both law+academia (act-defined columns,
  credential as graded C annotation), B1/B2 split + circularity rule, master
  table §7 (11 in hand / 4 GREEN / 7 YELLOW), full-citation lit map §5 +
  novelty-threat sweep §5b (verdict: OPEN; cite Wang 2023, essay-scoring
  2025, Gong 2026 defensively).
- **Formalization saved**: methods/metric_implementer/2026-06-12__formalization.md
  (V-usable-information framing, gap(N)→gap(1), derived predictions T1–T6
  incl. Zipf-shared-exponent scaling law T2).
- **Chinchilla integration (§6 + T7 + E7)**: no executor-free bound exists
  (V-relativity; double limit degenerates to "the name suffices"), so the
  estimand is τ(E)/B\*(m), a function not a number. Chinchilla imports as the
  *separable two-resource null* fidelity\* = (1−τ) − A·L^−α − B·C^−β − I(L,C);
  I = lack-of-fit = thickness (E2 interaction generalized); α tied to T2's
  Zipf exponent. New E7 in the experimental plan: per-cell bounded bracket
  [best exhibited articulation on fresh holdout, disattenuation ceiling
  √(ρ_tier·ρ_anchor)], 3-cap L × tier grid, isotone projection (slack
  calibration), multi-form fits, three-estimator agreement rule; fitted τ̂
  descriptive-only per the 06-11 amendment.
- **E7 PILOT ran 2026-06-12** (law+CW+code, 3 metrics each, ~$2.23 OpenRouter):
  machinery validated end-to-end but **no interpretable measurement** — verified
  honest framing "instrument calibration, directional non-significant point
  estimates" (6-agent adversarial check, 4/5 candidate claims overclaimed).
  Writeup methods/metric_implementer/2026-06-12__e7-pilot-results.md, memory
  [[project_e7_pilot_results_2026_06_12]]. gemma-3-4b + llama-1b judges
  degenerate (score/applicability collapse) → only 1 informative tier;
  words_share ordering directionally right (voice 0.00<element 0.10<edge 0.17)
  but not statistically resolvable; "D-study 3×3→0.84" figure was wrong (real
  0.00–0.23). NOW SCALING → vLLM offline batch on sk3 (no more OpenRouter,
  [[feedback_metric_implementer_sk3_only]]): batch (metric×item) prompts, long
  table (judge, item, (prompt, iteration)) + GEPA operator labels, save all
  prompts/iterations.
- **Clustering instrument fixes**: notes/2026-06-12__structural-metrics-variance.md
  (page-level subsampling + grouped jackknife; with-replacement bootstrap
  biased for richness stats; between-subtask variance dominates 3–7×);
  notes/2026-06-12__tail-adoption-repair.md (realized FN 45–68% vs v6;
  star-adoption repair adopt_v2; judge score=2 not transitive).
- **New datasets built + protocol-checked**: datasets/peer-review/oral_spotlight/
  (33,890; TF-IDF floor 0.53 = chance) and sk3 caption_contest (227 contests,
  678 finalists; TF-IDF 0.573; bge+LR 0.730; crowd-implied ceiling 0.938;
  finalist-vs-hardneg bge 0.645 with crowd_mean anti-predictive 0.339 —
  the expert-vs-crowd boundary is partially learnable). CAUGHT: NYer
  typographic-quote leakage gave fake bge 0.996 → presentation
  normalization now mandatory for cross-source label joins.
- **Collection in flight on sk3**: Arctic Shift (news/worldnews/supremecourt/
  LegalAdviceUK/legaladviceofftopic), CourtListener bulk (2.8G), Law SE dump,
  caption repo. Queued: Wigleaf, humor contests, RoyalRoad stubs, Jeff Huang
  best-papers, OpenAlex.

### 2026-06-12 (later) — RoyalRoad stubs (cw-A) build launched
- Pipeline in `datasets/creative-writing/royalroad_stubs/` (scripts) →
  sk3 `/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/royalroad_stubs/`.
- y=1 = official STUB status (~1.5K); y=0 = followers-listing controls,
  greedy 1:1 metadata match. X = 3 lowest-chapter-id chapters for BOTH sides
  from Wayback (`id_` raw view) — symmetric archive-availability filter;
  display() normalization + raw_text kept; md5 fiction-id splits.
- Validation 10/10: real chapter-1 prose (Primal Hunter, Azarinth Healer,
  Chrysalis, ELLC, ...). Known loss mode: fictions archived only post-stub
  (Threadbare: 77-word remnants) — correctly dropped by ≥200-word filter.
- RoyalRoad anti-scrape watermark = hidden display:none/speak:never classes
  in inline <style>; extractor strips them. Wayback from sk3 ~50% timeouts,
  retries handle. Full run nohup on sk3, log `logs/full_run_2026-06-12.log`.

### 2026-06-12 (later) — Wigleaf Top 50 (cw-B) + humor contest corpus (humor-A) built & shipped
- **Wigleaf**: scripts `datasets/creative-writing/wigleaf/` → sk3
  `/lfs/.../datasets/creative-writing/wigleaf/`. 3,909 labels (905 top50 +
  3,004 longlist), all 18 years 2008-2025, both lists every year.
  Longlist entries are NEVER hyperlinked → full text recovery for longlist
  needs title+venue search later. Pilot 60 top50 texts: 35 live / 16 wayback /
  9 dead = 85% retrievable (dead = Flash/JS viewers + a few dead zines).
  Label spot-checks 10/10 + my independent 2008-page check pass. y=0 pool
  must be de-conflicted vs BSF/Best Microfiction/BotN (in README).
- **Humor contests**: scripts `datasets/humor/contest_corpus/` → sk3
  `/lfs/.../datasets/humor/contest_corpus/contest_corpus.jsonl`. 929 rows,
  513 with full text: Wergle Flomp 411 (378 texts, 2002-25), To Hull And Back
  383 (labels only — winner texts are paid-anthology-only, verified), Erma
  Bombeck 135 (135 texts, 2010-26 via live + Wayback index; pre-2010 .asp
  pages = known gap). Identical NFKC+quote normalization across all 3 sites,
  raw_text kept. CAUGHT in validation: winningwriters tombstone pages
  ("content no longer available") counted as poem text → tombstone filter +
  PDF fallback (recovered 2 PDF-only poems incl. a first-prize winner).
  Entry validation 11/11; 2024-25 tier renamed most_highly_commended.

### 2026-06-12 (later) — Academia datasets: best papers (acad-B2) + citation percentiles (acad-C)
- **OpenAlex API regime change (affects ALL future OpenAlex work)**: free
  tier is now $1/day = 10,000 credits PER IP, reset midnight UTC. Search-type
  requests (`title.search`, `search`, `display_name.search`) cost 10 credits;
  plain filters 1. Workaround discovered + verified: `title.search` accepts
  pipe-OR'd QUOTED phrases at flat 10 credits/request (~15-20 titles each) —
  `batch_title_join.py` implements this with unquoted singleton retries.
  Joins ran spread across sk1/sk2/sk3 (one budget each). 429+Remaining:0 →
  hard abort, never cache "no match" on a throttled response.
- **acad-B2 best papers**: `datasets/peer-review/best_papers/`. Jeff Huang
  page → 1,819 awards (page grew past the ~700-900 estimate; sum-of-rowspans
  verified), 32 venues, 1996-2025. OpenAlex join 82.7% (94.6% abstract
  coverage among matched); 141 awards still unqueried (budget), see
  finish_leftovers.sh. 10/10 manual DOI inspections pass. y=0 pools: OpenAlex
  conference mapping confirmed BROKEN at scale (famous CVPR/KDD/WWW papers
  are sourceless or arXiv-only; PLDI 2022+ lives in PACMPL which mixes
  POPL/OOPSLA/ICFP — excluded). Pools = honest-but-partial, per-venue-year
  coverage in pool_coverage.csv.
- **acad-C citation percentiles**: `datasets/peer-review/openalex_citations/`.
  DBLP defines membership: venue: search queries are term-level (workshop
  contamination in 2017-18 ICLR) → per-year TOC `toc:db/conf/iclr/iclr2017.bht:`
  queries exact; final source = dblp.xml.gz dump crossref keys (conf/iclr/2017
  vs 2017w separates main/workshop cleanly). 29,228 papers ICLR/NeurIPS/ICML
  2013-2023, every venue-year matches official accepted counts. OpenAlex
  source pull only covers ~9.2K (MAG-era, dead post-2021) — rest via batched
  title join, ~90-95% match rate.

## 2026-06-13 — ★ Taste-taxonomy grid COMPLETE: 12 new cells collected, deconfounded, bounded, V>0

All grid cells from the taste taxonomy (notes/2026-06-12__taste-taxonomy.md §13) are
built, deconfounded, dual-bounded (TF-IDF lower / bge upper), V>0-checked, and at
first-pass modeling state up-to-par with Math/Code.

**Cells (TF-IDF / bge / codegen-V):** oral_spotlight 0.505/0.597/0.549 · best_papers
0.583wvy/—/0.742 (DBLP+S2, 5.7× venue recovery, full build auto-running) · citations
0.790/—/0.756 (S2-fixed) · CL-citation 0.650/0.537/— (provenance-pinned) · law_se
0.624/0.603/0.532 (Math.SE 1:1) · scotus 0.562/0.596/0.539 (author-grouped) · reddit_news
0.738/0.754/0.636 · BBC 0.60-0.64/—/— · royalroad 0.749/0.604/0.560 · wigleaf
0.537wmag/—/0.653 · caption 0.546/0.629/0.606 · humor-A ~0.45/—/~0.

**V>0 on every cell.** Gradient confirms the thesis: high V/A where text-structure carries
the label; expert-curation = A>V (semantic, oral); creative taste = residual (humor-A V≈0,
Wigleaf within-venue 0.537).

**Method wins:** confound audit (workflow) caught + fixed leaks the per-cell builds missed —
caption typography (0.667→word 0.546), courtlistener citation-undercounting provenance
(0.951→0.508), scotus author-identity (0.722→0.500 author-grouped), reddit_news rare-domain
(0.662→0.523); OpenAlex $1/day blocker → Semantic Scholar pivot validated (match endpoint,
97% accept, no budget wall) for both acad-C and best_papers.

**Open (user decisions):** (1) full-29K acad-C S2 standardization overnight (~8h) for
source consistency; (2) NeurIPS-2022 0.989 cell leak review/exclude; (3) OpenAlex top-off
unblocks 17:00 PDT. **Still enriching (auto):** best_papers full 27K build (~20h),
BBC 2018-25 symmetric rebuild, RoyalRoad richer recovery.

## 2026-06-14 — EXA (TTAB ex parte appeals) ex-ante slice: assembled + deconfounded + doctrine compiled

TTAB ex parte appeals (applicant appeals an examiner's final refusal to register;
y = Board reverses[applicant wins] vs affirms). Download of 7,214 born-digital
proceedings (2008–2025) completed; assembled and run through the dataset-first gate.

**x** = appeal_brief + examiner_statement + reply_brief (the `decision` PDF excluded →
0 outcome-leak hits on samples). PyMuPDF + per-file SIGALRM extraction; exante_scrub +
identity strip (emails/examiner/filer names/signatures/docket codes; applicant name kept,
group-split isolates it). **Appellant-group split** (multi-pattern, 68% named; 0 entities
straddle splits). Balanced **1,688** (844/844) from natural 87.7/12.3.

**Confound scan PASSED:** TF-IDF+LR group-split **test AUC 0.676** (healthy band — real,
not saturated). length-only 0.493 (no length confound); identity-strip ΔAUC 0 (signal is
doctrinal, not identity); both classes present every year 2008–25 (no temporal partition).
Top features = interpretable doctrine — win: substitute specimen / disclaimer / specimen /
design elements (specimen- & disclaimer-type refusals are the winnable ones); lose: cited
mark / num 2d (§2(d) likelihood-of-confusion refusals hard to overturn).

**Doctrine compiled (goal-hook):** `online-rubrics/by-law/ttab_exa/refusal_grounds.json` —
12 ex-parte refusal grounds the opposition-focused `ttab_dupont` bank lacked (specimen use,
failure-to-function ornamental/informational, §2(e)(2) geographic, §2(e)(4) surname,
§2(e)(5) functionality, §6 disclaimer, §2(f) evidence, ID definiteness, mutilation,
third-party-reg evidence). **6 flagged thin/checkable** (substitute-specimen bool,
geographic gazetteer, surname registry, disclaimer-offered bool, 5-yr-use claim,
third-party-reg count). §2(d)/descriptiveness/genericness reused from ttab_dupont.

**Status:** dataset DONE + confound-clean (lexical rung 0.676); doctrine bank ready.
A-rung (ttab_dupont 2(d)/descr/generic + new ttab_exa grounds) + dense ceiling = next phase,
deferred (EXA loop instruction was "assemble" only). SS (~44K) + BVA still downloading.

## 2026-06-14 — SS ex-ante claimant-brief corpus: NEGATIVE RESULT (free-RECAP dead end)

Enumeration of SS disability §405(g) district-court appeals finished: **44,058 dockets**,
balanced (22,974 deny / 21,084 grant). But the download yielded only **57 usable briefs** —
not a bug: ~99% of the briefs RECAP *indexes* are PACER-paywalled. Only **47 claimant + 10
commissioner** briefs are is_available+filepath_local (free). 89% of dockets had no
brief-like entry detected at all (n_kept=0); of the 4,372 with a claimant-brief object,
only 47 free. **~0.13% free-availability ceiling → SS ex-ante claimant-brief track CLOSED**
(can't beat PACER paywall without purchase). enumeration.jsonl kept (reusable if PACER
access obtained). The ex-POST SS disability slice (ss_disability_balanced_v2) is separate
and unaffected. Note: ss_exante/RESULT_negative.md. Monitoring narrowed to BVA only.


## 2026-06-16 — metric_implementer: V-information first; scaling-law framing under review

**Decision:** ground metric articulability on the **unsupervised** objective
**I(x → m_recovered)** — V-usable information the datapoint carries about the metric *as recovered
through an articulation→re-execution bottleneck* (NOT label Y, NOT a strong-LLM holistic anchor;
this is the recovery loop the codability analysis was retracted for lacking). **Measure
V-information and trust the numbers first; only then attempt upper bounds.** Recovery channels to
experiment with (mixtures, hand-tuned): consistency (rubric re-applied, K passes) / genuine
reconstruction (induce rule from m's behavior → fresh executor re-applies) / synthetic (planted,
known I_V). Estimator + E0 calibration tests live in `methods/metric_implementer/vinfo.py` (+
`tests/test_vinfo.py`, 12/12 pass). E0 finding: I_V from finite passes is **upward-biased**
(Miller–Madow helps but can't correct cells where only one outcome was seen), so at **K=5** only
fairly large I_V *differences* resolve — small I_V needs more passes or a shrinkage estimator
(NSB/Beta-Binomial).

### The scaling-law breakdown (the term table)

```
fidelity*(m; L, C) = (1 − τ_m) − A_m·L^(−α_m) − B_m·C^(−β_m) − I_m(L, C)
```

| term | Chinchilla analog | what it means |
|---|---|---|
| `fidelity*(m;L,C)` | `L(N,D)` (maximized, not minimized) | best recovery quality of metric *m* at word-budget L, executor capability C |
| `L` | data `D` | **articulation budget** — words/clauses the rubric may use |
| `C` | params `N` | **executor capability** — strength of the reader (empirical, not nominal size) |
| `(1 − τ_m)` | irreducible loss `E` | the **ceiling** as L,C→∞; `τ_m` = tacit residual (Polanyi), the part no words+reader recover |
| `A_m·L^(−α_m)` | `A/N^α` | gain from **more words**; `α_m` = rate words close the gap |
| `B_m·C^(−β_m)` | `B/D^β` | gain from **more reader capability**; `β_m` = its rate |
| `I_m(L,C)` | *(none — Chinchilla assumes separable)* | the **interaction** = thin/thick test: ≈0 → words & reader substitutable (thin); large → words help only capable readers (thick) |

**T2 (predicted exponent):** criteria with Zipf-importance k^(−(α+1)), ~c words each → budget L
covers top L/c → residual tail ∝ **L^(−α)**, exponent = Zipf tail index = (claim) the same exponent
as dense data-scaling. Three independent measurements (norm-cluster Zipf tail, dense data-scaling
exponent, rubric-budget frontier exponent) should coincide → exponent *predicted*, not just fit.

### Open question: does Chinchilla even apply? (Alex, 2026-06-16) — likely NO as a literal law

Chinchilla is a **training** law (loss vs params N, data D — you *retrain* a model family at
controlled compute). **We do not train anything.** Our axes are (L = rubric words within one model's
context, C = which off-the-shelf model reads it). Neither is a training/compute axis, so the
Chinchilla *form* is at best loose intuition ("there's a floor + diminishing returns per resource"),
not the right model. Plan:

- **C (executor capability): use OBSERVATIONAL scaling laws (Ruan, Liu, Hashimoto 2024).** Place each
  judge model on a *measured* capability scalar — PCA of public benchmark scores across many models
  (`E_m ≈ h·σ(βᵀS + α)`), and/or capability recovered from our own response matrix — rather than
  nominal size. Caveat: we have only ~6 judge models, too few to fit a standalone observational law;
  options = borrow published benchmark scores for these models, and/or expand the ladder
  (EXPAND_TIERS 27B–80B, currently env-blocked), and/or co-estimate capability with IRT (below).
- **IRT is the natural OBSERVATIONAL estimator and is preferred over a parametric scaling fit.** It
  places executors on a latent ability θ and items on difficulty z directly from the judge×item
  response matrix — θ IS the observational capability axis, recovered from *our* data, co-estimated
  with item difficulty (so it controls for which items are hard). The **4PL upper asymptote d<1 is the
  articulability ceiling** read straight off the curve — a *free* parameter, not an assumed Chinchilla
  E. Identifiable only with a high-ability anchor in the panel (the strong/dense reader) and a crossed
  judge×rubric design. Priors: judge-IRT (Choi 2026 GRM; Cong 2026 grader-as-respondent); the
  formalization's κ_r articulability-discrimination factor; `irt_model.py` already fits 2PL+κ_r (4PL
  ceiling NOT yet coded).
- **What survives without the training analogy:** the **T2 Zipf-coverage argument is mechanistic, not
  a training story** — "budget L articulates the top L/c criteria, residual is the tail" needs no model
  retraining. So T2 (frontier exponent = norm-frequency tail index) stays a real, falsifiable
  prediction even though Chinchilla-the-law is the wrong frame.

**Net:** drop Chinchilla as the literal model; keep the additive-vs-interaction (thin/thick) idea but
fit it observationally — IRT θ for capability (4PL ceiling = the bound), an empirical within-model
length curve for L (expect an IFScale knee, not a clean power law), and the T2 Zipf check as the
mechanistic cross-validation of the exponent. All of this is the DEFERRED upper-bound phase; V-info
numbers come first.


### How the equation changes under IRT / observational scaling (first pass, 2026-06-16 — expand w/ derivation)

Moving from Chinchilla-style to IRT/observational is not a re-fit, it's a change of object: the
scaling law stops being "a curve fit to aggregate fidelity" and becomes "a **per-item response
model** we fit, from which the scaling behavior and the bound are READ OFF."

**Chinchilla-style** (assigned resources, aggregate fidelity, extrapolated floor):
```
fidelity*(m; L, C) = (1 − τ_m) − A_m·L^(−α_m) − B_m·C^(−β_m) − I_m(L,C)
```
**IRT/observational** (latent capability, per-item probability, estimated ceiling):
```
p_i(E, L) = P(verdict=1 | item i, executor E, budget L) = c + (d_i − c)·σ( κ(L)·a_i·(θ_E − b_i + g(L)) )
```

| IRT piece | replaces | meaning |
|---|---|---|
| `θ_E` | the C axis | reader capability as a **latent recovered** scalar (observational; Ruan/Tatsu), not assigned size |
| `b_i` | (nothing) | item **difficulty** explicit (Chinchilla averaged it away) = our difficulty-stratum axis |
| `d_i` | `(1−τ_m)` | ceiling as an **estimated 4PL upper asymptote** (the BOUND), not an extrapolated constant; needs a high-θ anchor (dense reader) to identify |
| `σ(·)` | `C^(−β)` | approach to ceiling is **logistic**, not power-law (Ruan: `h·σ(βᵀS+α)`) |
| `g(L)` / `κ(L)` | the L term | rubric budget as a difficulty **shift** (`g`) and/or discrimination **slope** (`κ`; `κ→0` = non-articulable) |
| shift vs slope | `I_m` | thin/thick = **additive-in-logit (shift, separable, thin)** vs **multiplicative (slope, interaction, thick)**; nested-model test |

**The subtlety that matters most (no ground truth → not vanilla IRT).** Standard IRT / judge-IRT
(Choi, Cong) model `P(correct)` — a right answer exists, and as θ→∞ everyone gets everything
"correct" (P→1). We rejected ground truth, and we do NOT want the recovered verdict to collapse to
all-YES — we want it to **track each item's true metric value**, which varies across items. So the
naïve `σ(θ−b)` import is wrong for the same reason the label-Y anchor was. The right family is
**IRT-without-an-answer-key**: Cultural Consensus Theory (Batchelder & Romney) / Dawid–Skene
latent-truth models. There, ability = **recovery quality**, not correctness: each item has a latent
recoverable verdict `q*_i`, reported through a noisy channel whose error shrinks with capability and
articulation:
```
p_i(θ_E, L) = (1 − e)·q*_i + e·(base rate),     e = e(θ_E, L) ∈ [0,1]
```
`e→0` (capable reader, rich rubric) ⇒ `p_i→q*_i` (perfect recovery); `e→1` ⇒ `p_i→base` (verdict
ignores the item). This is the GENERATIVE version of the `triad.py` G-theory decomposition
(σ²_item = the `q*` signal, σ²_interaction = reader×item coupling, σ²_residual = pass noise; Eρ² =
no-parametric cousin of IRT reliability).

**The payoff — our objective is a readout of this model:**
```
I_V(x→m_recovered)(θ_E, L) = H(p̄) − E_i[H(p_i)]      with p_i = p_i(θ_E,L)
```
As `e→0`: `I_V → H(q̄*) − E_i H(q*_i)` = the metric's intrinsic recoverable info = **the ceiling/bound**.
As `e→1`: `I_V → 0`. So the **scaling "law" is the recovery-error function `e(θ_E,L)`** (where a
logistic/power shape and the L-vs-θ additivity = thin/thick live), and **I_V is a deterministic
functional of it**; the bound is its `e→0` limit (set by the jointly-estimated consensus `q*`). We do
NOT fit fidelity to a power law — we fit a no-gold-key consensus/measurement model and read I_V off
it. Only T2 (Zipf exponent of the difficulty distribution) survives untouched, sitting on top
regardless of the link.

OPEN for tomorrow (derivation + Alex's questions): (1) θ identifiability with no answer key + why the
dense anchor identifies `d`; (2) does L enter as shift `g(L)` or slope `κ(L)` (which one IS thin/thick);
(3) exact bridge from fitted `p_i` to the 5-pass I_V estimator (smoothed vs plug-in).


### First real I_V numbers — consistency channel (2026-06-16, math + peer-review, 2 tiers)

Ran `vinfo.channel_consistency` on the existing 5-pass sampled long table (math + peer-review,
Qwen3-8B + Llama-3.1-8B, caps 120/1000, 8 metrics × 60 items × 5 passes; 127,800 rows; 426 cells).
SEED-rubric mean I_V (bits): math/Qwen 0.58(cap120)/0.44(cap1000), peer/Qwen 0.69/0.68;
math/Llama 0.22/0.27, peer/Llama 0.28/0.40.

**Finding 1 (the consistency channel is near-degenerate).** Pass-to-pass recovery noise is TINY:
per-item p_i are mostly ∈{0,1} (e.g. top math metric: 39 items at 0, 14 at 1, 7 at 0.2 → I_V 0.73 ≈
H(p̄) 0.82). So **I_V ≈ H(p̄): the consistency channel mostly measures whether the rubric is
balanced / uses its range, NOT articulability.** Re-running the same rubric is a trivial bottleneck.
This strongly validates the instinct that the **genuine reconstruction channel** (induce rule from
behavior → fresh executor) is the principled one — its bottleneck is real. ALSO: 3/8 math rubrics
collapse to always-NO (I_V=0, flagged), dragging math's mean.

**Finding 2 (clean capability ordering — promising for the θ axis).** Qwen3-8B >> Llama-3.1-8B on I_V
across both tasks/budgets (e.g. peer cap1000 0.68 vs 0.40). The stronger reader recovers more
item-dependent information — the C/θ signal we want — though with only 2 tiers it conflates
consistency with range-use; needs more tiers + the reconstruction channel.

**Not yet interpretable:** budget (L) effect is tier-dependent and confounded by base-rate shifts
(Qwen: cap120 > cap1000, i.e. inverted; Llama: opposite); GEPA "best-over-versions" gain is partly
max-selection winner's-curse. Both need the deferred controls. Minor cosmetic bug: driver used
`row.flags` (pandas attribute collision) instead of `row["flags"]` — fix in the notebook; I_V values
unaffected (computed via dict). vinfo.py + driver staged at sk3:/lfs/.../tmp_vinfo/.


### Reconstruction-channel first results — fidelity-to-m is the articulability signal (2026-06-16)

Built the GENUINE reconstruction bottleneck (`methods/metric_implementer/recon_channel.py`): m's
behavior on a train split -> reconstructor induces a rule (free-gen OR MCQ-pick-from-list) -> FRESH
executor re-applies to held-out x -> I_V over R=5 independent reconstructions. Forced YES/NO protocol
(`_YESNO_TEMPLATE` + `score_binary`), not the [0,1] float rubric (which collapses on lenient 8Bs).
Tiny 1-GPU runs (GPU6, VLLM_GPU_MEM_UTIL 0.3), math_se + peer_review, 60 items.

Bug found+fixed mid-run: `induce_*` called `generate()` (seed=0 default) -> all R reconstructions
identical -> no recovery diversity. Forwarding seed -> genuine diversity; this sharpened the result.

Two readouts per metric: **I_V_recon** (does the recovered rule discriminate items?) and
**fidelity-to-m** (does it agree with the ORIGINAL metric? = Pearson(recovered p_i, m_verdict),
proxy for I(m; m_recovered)).

| metric | Llama-8B I_V_recon / fid | Qwen3-8B I_V_recon / fid |
|---|---|---|
| math_se      | 0.47 / **0.68** | 0.57 / **0.49** |
| peer_review  | 0.10 / **0.03** | 0.32 / **0.09** |

**KEY FINDING.** Raw I_V_recon is **executor-confounded** — peer-review rises 0.10→0.32 from Llama to
Qwen (a stronger reader extracts more structure; 2/16 → 7/16 peer metrics discriminate). But
**fidelity-to-m is the ROBUST articulability signal**: math stays faithful on both executors
(0.49–0.68), peer-review stays ~0 on both (0.03/0.09). So peer-review is tacit NOT because the reader
is weak — a better reader recovers plenty of (different) item-info — but because what it recovers is
the WRONG criterion. Manual inspection confirms: reconstructions of "Adherence to ethical guidelines"
/ "Citation practices" induce "novel research / valuable contribution / reproducibility" — generic
quality, never the specific procedural criterion; the diverse guesses disagree with each other AND
with m. Math reconstructions stay in the math-quality family (accurate / clear / complete) and agree.

MCQ mode (pick from the seed list): math recon 0.40 / fid 0.63 (identifiable+applicable);
peer recon **0.000** (even the chosen peer rubric collapses to a constant verdict when applied —
only 4/16 peer metrics discriminate at all). Consistency channel is also pure capability
(IV_cons math 0.42→0.95 Llama→Qwen), confirming it is not articulability.

**Upshot for Goal 1:** the unsupervised articulability of a metric = how faithfully it is RECOVERED
(I(m; m_recovered)), NOT how much the recovered rule discriminates (I_V_recon, capability-bound).
First robust separation: **math = articulable, peer-review = tacit**, stable across 2 executors and
2 reconstruction modes. CAVEATS: 4 metrics/task (wide fid CIs); fidelity is Pearson not yet a proper
MI; 2 executors (8B-class) — the full E-axis (Goal 2) will test whether peer fidelity ever rises with
capability. NEXT: implement I(m; m_recovered) proper MI + CI; scale metrics/tasks; then the E-axis.
Artifacts: recon_channel.py + vinfo.py; results in sk3:/lfs/.../tmp_vinfo/recon_{results,mcq,qwen}.json.


### Goal 1 CONFIDENT: transmission I(m; m_recovered) is the articulability measure (2026-06-16)

Implemented the proper MI `vinfo.iv_transmission` = I(m(x); m_recovered(x)) (bits) with bootstrap CI;
E0-calibrated on a binary symmetric channel (recovers 1 - H_b(e); ~0 for independence; test_vinfo
17/17). This replaces the fidelity-Pearson proxy as THE articulability quantity. Across 2 executors,
free-gen reconstruction, 4 metrics/task, 60 items:

| task | TRANSMIT I(m;m^) Llama-8B / Qwen3-8B | I_V_recon Llama / Qwen |
|---|---|---|
| math_se     | 0.218 / 0.149 (CIs > 0) | 0.44 / 0.57 |
| peer_review | 0.006 / 0.008 (CIs incl 0) | 0.17 / 0.32 |

**Airtight diagnostic:** I_V_recon (the recovered rule's discrimination) RISES with executor strength
(capability-confounded; peer 0.17->0.32). TRANSMISSION I(m;m^) is capability-FLAT and cleanly
separates: math ~0.15-0.22 bits (per-metric CIs all above 0), peer ~0.006-0.008 (CIs include 0) on
BOTH executors. Peer transmission does NOT rise with capability despite I_V_recon doubling -> peer is
ROBUSTLY tacit, not a weak-reader artifact. So: **articulability = I(m; m_recovered), NOT
I(x->m_recovered).** math = articulable, peer-review = tacit, calibrated + capability-robust.

Note: math transmission is only ~0.15-0.22 of H(m)~1 bit, so even math is ~20% transmitted at 8B --
whether it climbs toward 1 with capability/budget is the Goal-2/3 question (articulability in the
limit of E, L). CAVEATS: 4 metrics/task; 2 executors (8B-class).

GOAL 1 STATUS: confident. Estimator calibrated, measure principled + capability-robust, interesting
separation, free+MCQ done. GOAL 2 (observational scaling: transmission vs E and L; the consensus/IRT
recovery-error fit) is gated on Alex's scaling-law derivation discussion (deferred to "tomorrow"); the
uncontroversial prep = collect transmission(E) over more tiers + transmission(L) over budgets, which
Goal 3 ("articulable in the limit of L,E") reads off.


### E-axis preview + a methodological catch (2026-06-16, Mixtral added)

Added Mixtral-8x7B as a strong tier. Per-task MEAN transmission across tiers: math 0.218(Llama-8B)/
0.149(Qwen3-8B)/0.033(Mixtral); peer 0.006/0.008/0.012. **DO NOT read the math numbers as a
transmission(E) trend** — each tier RE-SELECTS its own metrics by P(YES)-spread, so the three tiers
measured DIFFERENT metric subsets (metric-alignment confound). Two real lessons:
1. **The E-axis needs a FIXED metric set across tiers** (select once, measure the same metrics at
   every E). Add `--metric-ids` to recon_channel before any transmission(E) claim. This is a Goal-2
   design requirement.
2. **transmission-MI vs fid-Pearson diverge when the recovered channel collapses**: Mixtral math
   MI 0.03 vs fid 0.37 because some induced rubrics collapse on re-application (I_V_recon ~0.01) —
   transmission-MI correctly reports ~0 bits through a near-constant channel; Pearson over-credits
   weak directional alignment. **transmission-MI is the honest measure; prefer it.** (Possible real
   effect to study with fixed metrics: stronger models may write more abstract rubrics that
   discriminate less on re-application — but unconfirmed.)
ROBUST claim that survives: **peer-review transmission ~0 on all 3 executors** (tacit). The clean
transmission(E) curve + the scaling-law fit await the fixed-metric redesign + Alex's scaling
derivation discussion.

### Goal 3 (E-axis): valid fixed-metric transmission(E) + articulability classification (2026-06-16)

Added `--metric-ids` (fix the SAME metrics across tiers — resolves the metric-alignment confound) and
ran 5 math + 5 peer metrics on 4 executors. Per-task MEAN transmission I(m;m^) (bits):

| task | Llama-3.2-3B | Llama-3.1-8B | Qwen3-8B | Mixtral-8x7B |
|---|---|---|---|---|
| math_se     | 0.095 | 0.214 | 0.127 | 0.124 |
| peer_review | 0.016 | 0.001 | 0.008 | 0.009 |

(Fixed metrics put Mixtral math at 0.124, not the confounded 0.033 — the alignment fix mattered.)
Transmission is NOT monotone in model size (math peaks at Llama-8B, not Mixtral) -> these 4
heterogeneous models aren't a clean capability ladder; "limit of E" = MAX over tested executors, not
an E->inf extrapolation (needs the capability-axis + scaling-law fit = deferred).

**GOAL 3 (best-over-tiers transmission, CI_lo > 0.03 bits = articulable):**
- **math_se: 4/5 metrics ARTICULABLE** (best I(m;m^) 0.245-0.294 bits, CI_lo 0.11-0.15); 1/5 borderline-tacit (0.109).
- **peer_review: 0/5 articulable** (all CI_lo = 0.00; best 0.040). Robustly TACIT across all 4 tiers.

Caveat: even "articulable" math transmits only ~0.25 of ~0.8-1 bit H(m) -> ~25-30%; "articulable" =
reliably-nonzero, not fully-transmitted. 5 metrics/task, 2 domains.

### Goal 3 (L-axis): transmission saturates fast in articulation budget (2026-06-16)

Fixed metrics + fixed executor (Llama-8B), articulation budget L (token cap on induced rule):

| L (tokens) | math_se | peer_review |
|---|---|---|
| 40  | 0.000* | 0.000 |
| 150 | 0.212  | 0.000 |
| 450 | 0.214  | 0.001 |

*L=40 degenerate: too few tokens to emit the {rule,rubric} JSON -> inductions fail (a format
artifact, not a low-L signal; TODO: non-JSON induction or higher floor for a clean low-L point).
Substantive: **math transmission plateaus by L~150 tokens (~0.21 bits); no gain 150->450.**
Peer flat ~0 at every budget.

**COMBINED Goal 3 (in the limit of L and E, tested ranges):**
- **math_se = ARTICULABLE but SATURATED**: transmission ~0.21 bits, plateaued in BOTH L (by ~150
  tokens) and E (by ~8B). Only ~25-30% of H(m) transmitted -> a real tacit residual (~70%) survives
  more words AND a stronger reader.
- **peer_review = ROBUSTLY TACIT**: transmission ~0 at every L and every E. Neither articulation
  budget nor reader capability recovers the metric.

The EMPIRICAL saturation directly answers "articulable in the limit" at observed scales.

**POWERED L-axis re-test (2026-06-17, 49 metrics math+CW, R=5, 60 held, GEPA-detail to bind L) — results, not a verdict (still exploring):** at fixed forced length, transmission is flat L40→100
(paired sign-test p=0.84 math / 1.0 CW, null) then decreases forcing longer (math L600 median
Δ=−0.029, 3/24 positive, p=2e-4; CW L≥250 p=0.04). 75–80% of metrics peak at L≤100 tok. Nested
running-max (V_{≤L}, monotone) rises +0.02–0.03 bits, saturating ~L=100 (partly max-bias). NB the
earlier 6-metric "rise to a knee" read was max-selection bias — the unbiased paired test is
null-then-negative. Single executor (8B); E-axis and broader L ranges untested. Code
`analyze_lbig.py`; sk3 `tmp_vinfo/recon_lsweep_free_big.json`. Breadth join (broad/moderate/narrow
present; very_narrow/very_broad not auto-selected): all measured classes tap out ≤100 — stratified
GEPA-lineage run (running) is the proper breadth instrument.

### Goal 2 IMPLEMENTED: observational scaling-law fit (`scaling.py`), sane on the L-axis (2026-06-16)

Built `methods/metric_implementer/scaling.py`: fits transmission T to the discussed saturating
recovery-error form T(L)=T_inf*(1-exp(-L/Lc)) (L-axis) and T(S)=T_inf*sigmoid(a(S-b)) (E-axis,
Ruan-style), asymptote T_inf = articulability in the limit, with point-bootstrap CI + R^2. Fixed the
low-L degeneracy (plain-text induction, no JSON overhead) and ran a DENSE L-sweep (Llama-8B, fixed
metrics, L in {30,60,100,150,250,450}):

| L (tok) | 30 | 60 | 100 | 150 | 250 | 450 |
|---|---|---|---|---|---|---|
| math_se     | 0.233 | 0.251 | 0.251 | 0.251 | 0.251 | 0.251 |
| peer_review | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**L-AXIS FIT (sane, R^2=1.00):** math T_inf(L) = **0.251 bits**, knee Lc ~< 30 tok (saturates by
L=30-60; Lc=11 is an extrapolation below the data floor). peer T_inf(L) = **0.000**. So the
articulable part of a math metric is a ONE-LINE rule (more words add nothing), and its ceiling is
only ~0.25 of ~0.8-1 bit H(m) -> **math is "thin but shallow": quickly articulable, low ceiling, ~70%
tacit residual.** peer-review tacit at any budget.

**E-AXIS FIT: honest negative** — R^2 = nan/poor; the 4 tiers are NOT a clean capability ladder (math
peaks at Llama-8B, not Mixtral; size != task-capability). A real E-scaling law needs the capability
axis definition (Ruan-PCA vs IRT-theta) + more/cleaner tiers = the deferred derivation discussion.
(Minor: `rm fix_l*` also deleted fix_llama3b/llama8b.json; those E numbers are in the table above.)

GOAL 2 STATUS: scaling-law machinery implemented + iterated to a sane fit on the clean (L) axis +
reported. The E-axis parametric law remains provisional pending the capability-axis discussion.

### Cross-task articulability sweep — math is the lone articulable task; a 3-way split (2026-06-16)

Extended transmission I(m;m_recovered) from 2 domains (math, peer) to ALL 9 manifest tasks
(`recon_channel`, Llama-3.1-8B, free-gen reconstruction, L=150 tok, R=5, auto-selected
discriminating metrics, 60 items). Per-task best transmission (bits) + #metrics with CI_lo>0.03:

| task | n_disc | best I(m;m^) | best CI_lo | #artic | verdict |
|---|---|---|---|---|---|
| **math_se**      | 10/12 | 0.346 | 0.181 | **5/5** | **ARTICULABLE** |
| notice_and_comment | 5/12 | 0.051 | 0.003 | 0/5 | tacit |
| reddit_humor     | 11/12 | 0.047 | 0.000 | 0/5 | **tacit (sharp)** |
| news_homepages   | 7/12 | 0.004 | 0.000 | 0/5 | tacit |
| peer_review      | 2/12 | 0.004 | 0.000 | 0/2 | tacit |
| patents          | 1/12 | 0.038 | 0.001 | 0/1 | undetermined |
| title_vii        | 1/12 | 0.000 | — | 0/1 | undetermined |
| ss_disability    | 1/12 | 0.000 | — | 0/1 | undetermined |
| flsa             | 0/12 | — | — | — | undetermined |

**The result is a 3-way split, not a binary** — and the n=1 legal cells are a SECOND failure mode,
not weak sampling:
1. **ARTICULABLE — math only.** 5/5 metrics transmit (mean 0.24, best 0.35 bits, all CI_lo well
   above 0). Replicates the prior L-sweep ceiling (~0.25) cleanly at L=150.
2. **TACIT — humor, news, N&C, peer.** Metrics DO discriminate (binary verdict varies across items)
   yet transmission ~0. **reddit_humor is the sharpest case: 11/12 metrics fire strongly (P(YES)
   std 0.14–0.29, verdicts genuinely cross 0.5) but transmission stays ~0** — the firing pattern is
   real but cannot be reconstructed from m's behavior. This is the cleanest evidence yet that
   tacitness ≠ non-discrimination (a metric can be a reliable, high-variance function of x and STILL
   be unrecoverable through the articulation bottleneck).
3. **UNDETERMINED — legal (title_vii/flsa/ss_disability) + patents.** NOT tacit: Llama-8B applies
   these metrics **near-constantly** (P(YES) std mostly 0.04–0.12, mean 0.05–0.40 → binarized verdict
   is ~all-NO → H(m)≈0 → transmission ~0 TRIVIALLY). Lowering std-floor/widening the scan can't fix
   it — the binary verdict itself is degenerate. Cause is long-document truncation + 8B can't apply
   nuanced legal/patent doctrine with resolution (consistent with the legal-arm finding that
   long-doc tasks are truncation-masked; cf. trademark short-text being the lone dense>lexical gap).
   To measure articulability here we'd need either (a) soft/quantile-binarized transmission instead
   of the 0.5 cut (code change — `iv_transmission` currently hard-binarizes m), or (b) a stronger
   executor (70B) that produces a varied binary verdict — i.e. the E-axis, which is deferred.

Headline: **of the tasks we can measure, math is the only articulable one; humor/news/N&C/peer are
robustly tacit; legal/patents are unmeasurable at 8B (executor can't apply the metric, not shown
tacit).** Results: sk3 `tmp_vinfo/crosstask_l150_llama8b.json`. Caveats: single executor (8B),
auto-selected metrics, L=150, 60 items. The legal "undetermined" cells are the cleanest open
follow-up (soft-verdict transmission, or 70B executor).

### Recovery channel: synthesize (free) vs choose-from-list (MCQ) — MCQ transmission is confounded; identification accuracy is the real signal (2026-06-16)

Added behavioral distractor selection to the MCQ channel (`recon_channel.py`): distractors are
chosen by Cohen's-κ similarity of their *binarized verdict vector* to the target (not semantics,
because transmission is behavioral), with behavioral CLONES (κ≥0.97) excluded, base-rate-matching,
and a graded difficulty dial S = max distractor-κ. Ran free + mcq{random, hard, graded} on the
controls **math_se** (known articulable) and **reddit_humor** (known tacit), Llama-8B, R=5, 4 metrics,
4 options (chance=0.25). Analysis: `methods/metric_implementer/analyze_mvf.py`; results `tmp_vinfo/recon_{free,mcq_random,mcq_hard,mcq_graded}.json`.

| task | T_free | T_mcq_hard | id_hard | mcq-random pathology |
|---|---|---|---|---|
| math_se | 0.200 | 0.239 | 0.25 (=chance) | — |
| reddit_humor | 0.028 | 0.167 | 0.15 (<chance) | one metric S=0.05→id=1.0→**T=0.666** (inflated) |

**Three findings, the last one is the important one:**
1. **Controls reproduce in free mode** — math 0.20 (articulable), humor 0.03 (tacit). Instrument stable.
2. **The "too-easy distractor" artifact is real and now visible.** Under `random` distractors, the
   humor metric *Sentence-level incongruity* hit T=0.666 at S=0.05, id=1.00 — the model trivially
   recognizes the one true rule because the alternatives are behaviorally far. Pure artifact of the
   option set, not articulability. (Confirms the a-priori worry.)
3. **MCQ transmission is confounded BY CONSTRUCTION and is the wrong readout.** With hard negatives
   (κ-near-misses by design), *any* pick — right or wrong — transmits m's verdict pattern, because
   every option is behaviorally close to m. So MCQ transmission = an option-set-closeness FLOOR +
   an identification BONUS. The graded sweep proves it: as S rises, math transmission stays flat
   (~0.19→0.25) while identification drops (0.35→0.25) — transmission is invariant to whether the
   model actually recovered the metric. **The clean MCQ articulability signal is chance-corrected
   identification accuracy, not transmission.**

And under maximal (hard) difficulty, identification is **at/below chance for BOTH tasks** (math 0.25,
humor 0.15) — i.e. recognition adds nothing beyond what free-gen already says. MCQ identification is
*also* confounded (by how distinctive the option *descriptions* are — vivid humor labels vs generic
math labels), so it isn't a free lunch either.

**Verdict on the method:** the selection rule is now correct and instrumented, but using it shows
**MCQ does not give a cleaner articulability number than free-gen — if anything it's muddier**, and
the "articulation tax = T_mcq − T_free" framing fails because MCQ transmission is dominated by
option-set construction. **Free-gen transmission stays the primary measure.** If MCQ is used, report
chance-corrected identification accuracy at *matched, maximal* S, with more R/metrics (current
R=5×4-metric id estimates are too noisy for task-level claims) and description-normalized options.

### Articulability BOUNDS: free-gen (lower) vs MCQ-recognition (upper) — and CW is wired + a recognition≫recall task (2026-06-16)

Wired **creative_writing** into the harness as the 10th task (corpus `writingprompts_modeling_clean.csv.gz`,
96K balanced, 7,699-file rubric bank — was laptop-only/git-untracked, now synced to sk3; see
[[reference_metric_banks]]). Ran the bounds framing on math_se (control+), reddit_humor (control−),
creative_writing (target): Llama-8B, R=8, M=5, n_metrics=5. Both bounds on [0,1]:
**lower = free-gen `transmission_norm`** (fraction of m recovered by AUTHORING); **upper = MCQ
chance-corrected identification** `(id−1/M)/(1−1/M)`, max over graded difficulty bands (RECOGNITION ≥ recall).

| task | LOWER (free) | UPPER (raw desc) | UPPER (norm desc) |
|---|---|---|---|
| math_se | **0.223** | 0.031 | 0.406 |
| creative_writing | **0.035** | 0.438 | 0.375 |
| reddit_humor | **0.026** | 0.250 | 0.062 |

Three findings:
1. **The bracket separates the tasks by *kind*, not just degree.** math = [0.22, 0.41] (articulable —
   lower bound already well clear of 0). humor = [0.03, 0.06] (**robustly tacit — BOTH bounds ≈0**;
   can't author it, can't even recognize it once vivid wording is stripped). creative_writing =
   **[0.04, ~0.40] — a WIDE bracket**: tacit-by-authoring (free≈0) yet recognizable (id peaks cc≈+0.4
   at moderate S). CW is a genuine **recognition ≫ recall** case — the model holds CW evaluative
   knowledge (narratology criteria) it can pick out but cannot regenerate from behavior.
2. **Description normalization is NOT a clean confound-strip — it moves both ways.** Paraphrasing
   options to a uniform register DROPPED humor's id (cc +0.25→+0.06: stripped the vivid-wording cheat,
   as intended) but RAISED math's (cc ~0→+0.41: the paraphrase *sharpened* vague rubric bodies into
   crisp distinguishable one-liners, ADDING identifiability). So the paraphrase is an uncontrolled
   rewrite, not register-only. The upper bound is **phrasing-sensitive**, which re-confirms MCQ
   identification is dominated by option phrasing — **free-gen (phrasing-independent) stays the
   trustworthy measure.** (Paraphrase is generated from the metric's own description, not from items
   or held verdicts, so it cannot leak into transmission — it only perturbs identifiability.)
3. CW free-gen ≈0 means **creative writing is NOT articulable-by-generation at 8B** — same tacit
   bucket as humor on the *lower* bound, but unlike humor it has a large recognition headroom.

Caveats: R=8×5-metric → band-level id over ~40 picks, SE ~0.07; bracket *directions* are robust, exact
values noisy. Results: sk3 `tmp_vinfo/recon_{free_bounds,mcq_graded_raw,mcq_graded_norm}.json`;
analyzer `analyze_bounds.py`. Next: tighten id (more R/metrics), and the recognition≫recall CW gap is
the cleanest new phenomenon to chase.


## References (auto-verified BibTeX, 2026-06-15)

> Extracted from this document and web-verified + independently audited by an automated fact-check pass (search → fetch → resolvable id; attributed claim checked against the located paper). 18 entries. Real located works; not hand-checked. See "needs manual review" for 0 contradicted-claim and 2 unlocatable/rejected items.

```bibtex
@inproceedings{AlurEtAl2023,
  author    = {Alur, Rohan and Laine, Loren and Li, Darrick and Raghavan, Manish and Shah, Devavrat and Shung, Dennis},
  title     = {Auditing for Human Expertise},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year      = {2023},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html}
}

@book{aristotle_ethics,
  author = {Aristotle},
  title  = {Nicomachean Ethics},
  note   = {Book VI, on phronesis (practical wisdom); composed c. 350 BCE}
}

@book{daston2022rules,
  author    = {Daston, Lorraine},
  title     = {Rules: A Short History of What We Live By},
  year      = {2022},
  publisher = {Princeton University Press},
  series    = {The Lawrence Stone Lectures},
  isbn      = {9780691156989}
}

@book{dreyfus1986mind,
  author    = {Dreyfus, Hubert L. and Dreyfus, Stuart E.},
  title     = {Mind over Machine: The Power of Human Intuition and Expertise in the Era of the Computer},
  publisher = {Free Press},
  year      = {1986},
  isbn      = {9780029080603}
}

@article{gong2026llms,
  title={LLMs learn scientific taste from institutional traces across the social sciences},
  author={Gong, Ziqin and Li, Ning and Zhou, Huaikang},
  journal={arXiv preprint arXiv:2603.16659},
  year={2026}
}

@article{harada2025automated,
  title={Automated Refinement of Essay Scoring Rubrics for Language Models via Reflect-and-Revise},
  author={Harada, Keno and Yoshida, Lui and Kojima, Takeshi and Iwasawa, Yusuke and Matsuo, Yutaka},
  journal={arXiv preprint arXiv:2510.09030},
  year={2025}
}

@article{InglisAberdein2015,
  author  = {Inglis, Matthew and Aberdein, Andrew},
  title   = {Beauty Is Not Simplicity: An Analysis of Mathematicians' Proof Appraisals},
  journal = {Philosophia Mathematica},
  volume  = {23},
  number  = {1},
  pages   = {87--109},
  year    = {2015},
  doi     = {10.1093/philmat/nku014}
}

@article{JohnsonSteinerberger2019,
  author  = {Johnson, Samuel G. B. and Steinerberger, Stefan},
  title   = {Intuitions about mathematical beauty: A case study in the aesthetic experience of ideas},
  journal = {Cognition},
  volume  = {189},
  pages   = {242--259},
  year    = {2019},
  doi     = {10.1016/j.cognition.2019.04.008}
}

@article{kaplow1992rulesa,
  author  = {Kaplow, Louis},
  title   = {Rules versus Standards: An Economic Analysis},
  journal = {Duke Law Journal},
  year    = {1992},
  volume  = {42},
  number  = {3},
  pages   = {557--629},
  doi     = {10.2307/1372840}
}

@article{kennedy1976form,
  author  = {Kennedy, Duncan},
  title   = {Form and Substance in Private Law Adjudication},
  journal = {Harvard Law Review},
  volume  = {89},
  number  = {8},
  pages   = {1685--1778},
  year    = {1976},
  url     = {https://duncankennedy.net/wp-content/uploads/2024/01/form-and-substance-in-private-law-adjudication.pdf}
}

@article{Kolbel2004,
  author  = {K{\"o}lbel, Max},
  title   = {Faultless Disagreement},
  journal = {Proceedings of the Aristotelian Society},
  volume  = {104},
  number  = {1},
  pages   = {53--73},
  year    = {2004},
  doi     = {10.1111/j.0066-7373.2004.00081.x}
}

@book{kripke1982wittgenstein,
  author    = {Kripke, Saul A.},
  title     = {Wittgenstein on Rules and Private Language: An Elementary Exposition},
  year      = {1982},
  publisher = {Harvard University Press},
  isbn      = {9780674954014}
}

@article{MacFarlane2005,
  author  = {MacFarlane, John},
  title   = {Making Sense of Relative Truth},
  journal = {Proceedings of the Aristotelian Society},
  volume  = {105},
  number  = {1},
  pages   = {305--323},
  year    = {2005},
  doi     = {10.1111/j.0066-7373.2004.00116.x}
}

@book{polanyi1966tacit,
  author    = {Polanyi, Michael},
  title     = {The Tacit Dimension},
  year      = {1966},
  publisher = {University of Chicago Press},
  isbn      = {9780226672984},
  note      = {Originally published 1966 (Doubleday, Terry Lectures); ISBN refers to the Univ. of Chicago Press reprint edition with foreword by Amartya Sen}
}

@book{ryle1949concept,
  author    = {Ryle, Gilbert},
  title     = {The Concept of Mind},
  publisher = {University of Chicago Press},
  year      = {1949},
  isbn      = {9780226732961}
}

@book{schauer1991playing,
  author    = {Schauer, Frederick},
  title     = {Playing by the Rules: A Philosophical Examination of Rule-Based Decision-Making in Law and in Life},
  series    = {Clarendon Law Series},
  publisher = {Oxford University Press (Clarendon Press)},
  year      = {1991},
  isbn      = {9780198258315}
}

@article{suzgun2022harvard,
  author  = {Suzgun, Mirac and Melas-Kyriazi, Luke and Sarkar, Suproteem K. and Kominers, Scott Duke and Shieber, Stuart M.},
  title   = {The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications},
  journal = {arXiv preprint arXiv:2207.04043},
  year    = {2022},
  eprint  = {2207.04043},
  archivePrefix = {arXiv}
}

@book{wittgenstein1953pi,
  author      = {Wittgenstein, Ludwig},
  title       = {Philosophical Investigations},
  translator  = {Anscombe, G. E. M.},
  year        = {1953},
  publisher   = {Basil Blackwell},
  isbn        = {9780631146704},
  note        = {Originally published 1953 (Basil Blackwell, Oxford), trans. G. E. M. Anscombe; ISBN 9780631146704 is a later Blackwell paperback reprint of the same Anscombe translation}
}

```

### Citations needing manual review

**Could not be located / rejected by audit (2)**:

- Sa et al. 2022 — audit reject_mismatch: DOI 10.1007/s13164-022-00669-3 resolves; title and all four authors (Rentuya Sa, Lara Alcock, Matthew Inglis, Fenner Sta
- Wang 2023 — audit reject_mismatch: AUTHOR MISMATCH. DOI 10.1007/s11192-023-04881-5 resolves to a real article with the stated title, journal (Scientometric

**Partial claim-match (5)** — spot-check exact numbers/wording:

- `AlurEtAl2023`; `InglisAberdein2015`; `kennedy1976form`; `schauer1991playing`; `suzgun2022harvard`


## 2026-06-15 — Confound audit: trademark + EXA both CLEAN (multi-round controls)

User flagged the two trademark-family pools (trademark prosecution lexical 0.693, EXA
0.676) for possible confounding. Ran top-feature + class-conditional-lift + ablation +
stratified-control audit (confound_audit.py, tm_round1.py, tm_round2.py; CONFOUND_AUDIT.txt).

**EXA = clean.** drop-goods-stoplist 0.673, drop-boilerplate 0.673 (Δ0.003 from 0.676).
Signal is spread-out doctrine (win: specimen/substitute_specimen/disclaimer/design_elements;
lose: cited_mark/num_2d = §2(d)). Rare high-lift goods tokens (nut 4.5×, granola 4.8×) are
small-sample noise, not drivers.

**Trademark = clean (scary top features were a RED HERRING).** Top features LOOK like a
pro-se/drafting-register confound (registered: namely/featuring/comprising; abandoned:
etc/our/we/love/www) but those tokens are RARE and don't carry the AUC. Confound ladder:
- structured-only (draw_cd/basis/n_classes/yr) 0.601; within draw_cd=4000 0.693 → not design-code
- informality-score-alone 0.505; within has-`namely` 0.688 → not pro-se vs counseled
- within engaged (≥1 OA) 0.696 → not non-response abandonment
- first-class-ID-alone 0.593; within-single-class text 0.639(cls025)/0.722(041)/0.639(009)/0.681(035)
→ signal = **within-category, term-level registrability doctrine** (descriptiveness/distinctiveness
of the specific goods wording), surviving every control. Both pools sound as-is; no
deconfounding rounds needed. Residual soft caveat (not a leak): field-correlated business-
failure abandonment vs merits separable only with OA refusal grounds (TSDR ~22h crawl, deferred).

## 2026-06-15 — Trademark DENSE CEILING: +0.075 tacit gap (FIRST positive in ex-ante arm)

Following the confound audit (trademark clean, lexical 0.693), ran the dense ceiling on the
SHORT trademark text (mark+gs) — NOT truncation-limited unlike our long-doc ex-ante domains.
ModernBERT@256, owner-group split: **dense 0.7678 vs lexical 0.6932 = +0.075 gap** (epochs
0.744→0.768→0.766, stable). tm_dense.py / logs_tm_dense.log, GPU3.

**This is the first positive dense>lexical (tacit) gap in the entire ex-ante arm** — updates
the prior capstone ("no dense beats lexical in any ex-ante domain"), which held only because
the other domains were long-doc (truncation-masked) or mechanical (PTAB). Short-text trademark
exposes a real language-tacit layer.

**Interpretation:** the 0.075 is semantic mark↔goods descriptiveness (QUANTUM-for-tires arbitrary
vs QUICK-DRY-for-towels descriptive) — requires understanding meaning, invisible to n-grams,
exactly what the ttab_examination descriptiveness/distinctiveness rubric targets. Resolves the
"no headroom for articulated metrics" worry: trademark is a GOOD V/A/Taste domain (clean labels +
real articulable gap, not ineffable taste). Examiner lottery ruled out as residual source
(examiner-ID alone 0.546, +0.0015 over text; per-examiner register-rate p10-p90 0.40-0.60).
Residual ~0.23 (1−dense) = prior-marks landscape at filing (2(d), not in as-filed text) + post-
filing behavior. Crowded-field V-metric could inject the register context → potential A>text.

**Next:** (a) verify gap survives within-class (semantic, not cross-category artifact); (b) A-rung
(ttab_examination judge) — does articulated descriptiveness land near 0.77 (recovers tacit layer)
or ~0.70 (gap is beyond-articulation)? EXA dense untested (long briefs → likely truncation-limited).

## 2026-06-16 — A-rung on trademark + EXA: neutral-vs-advocacy natural experiment

Ran A-rung (FP8-70B vLLM, prefix-cached) after 4-stage spot-check (OpenRouter discrimination
probes + cap-150 pipeline validation: NA 0.007, 62/84 non-constant).

**Trademark prosecution (neutral filing: mark+goods): A_AUC=0.6854** (4000 cases, 84 metrics,
te=601). Ladder: V~0.60 < A 0.685 ≈ lexical 0.693 < dense 0.768. Nonlinear readout on same
metrics = 0.691 (gap to dense is metric-coverage, NOT readout capacity). On equal 4k data,
A(0.685) >> lexical-on-4k(0.607) — doctrine metrics are data-efficient; lexical needs 80k.

**EXA (advocacy appeal brief): A_AUC=0.533 = CHANCE** (1688 cases, 67 metrics ttab_dupont+ttab_exa,
te=263). Genuine, not a bug: NA 0.039, 67/67 non-constant, full briefs fed (budget 11.5k tok),
mean per-metric |AUC-0.5|=0.024 (best single dupont2_goods_relatedness 0.568). The judge reads
the applicant's advocacy framing; metrics vary but don't track outcome.

**★ Within-trademark natural experiment:** same doctrine (registrability), neutral filing A=0.685
vs advocacy brief A=0.533 — isolates the DOCUMENT-NEUTRALITY factor, holding doctrine constant
(cleaner than cross-domain; parallel to ARB-vs-BRB isolating articulability). Confirms+sharpens
the two-factor account: A>V iff neutral-doc AND articulable-doctrine. PTAB(0.50)+EXA(0.53) =
two advocacy-document A-failures. Practical: invest A-rung in trademark (neutral), not EXA.

**Feature-discovery (4 iterations, trademark):** errors = descriptiveness misses on synonym marks
(CARDIO CREATINE, FOCUS EQUINE). But (it2) mark-goods embedding cosine weak (+0.005, similarity≠
descriptiveness); (it3) descriptiveness metric SATURATED (calls ~all marks descriptive incl
registered controls); (it4) graded 0-10 descriptiveness STILL doesn't separate outcomes (AUC 0.41
on error cases) — equally-descriptive marks register vs abandon on post-filing behavior (response/
2(f)/disclaimer), not on readable descriptiveness. What DOES help: cheap STRUCTURAL/field features
(mark len/words, gs item-count/definiteness, b2b-vs-consumer) → A+feats=0.705 (~24% of gap).
Revised read: dense's +0.075 is drafting-seriousness/field cues proxying prosecution behavior, not
deep semantic doctrine. Scripts: run_alayer_tm_exa.py, iter_features{,2,3,4}.py; alayer_{trademark,exa}.npz.

## 2026-06-16 (cont) — within-class dense check OVERTURNS the field-cue hypothesis

tm_dense_byclass.py: dense beats within-class lexical in EVERY big NICE class — 009 +0.102,
025 +0.038, 035 +0.089, 041 +0.072, 042 +0.084 (mean +0.077; overall dense 0.764). So the
trademark dense>lexical gap is GENUINE MARK-LEVEL signal, NOT cross-field/drafting (my
iteration-4 "drafting/field cues" read was an artifact of the biased error subset).

**Reconciled conclusion:** the gap IS descriptiveness/distinctiveness — but the LABEL-CALIBRATED
version a zero-shot judge can't reach. Zero-shot graded descriptiveness saturates (it4 AUC 0.41)
because the LLM knows the doctrine but not WHERE the field-specific suggestive/descriptive line
sits; the dense model learns that boundary from 80k labels. So A(zero-shot doctrine)≈lexical =
explicit articulable layer; dense−A≈0.08 = calibrated application of the SAME doctrine — articulable
in principle, not by zero-shot prompting (Polanyi/Dreyfus know-how learnable from cases). Predicts:
a FEW-SHOT/calibrated descriptiveness judge should close part of A→dense; more metrics won't.
Next-experiment candidate. Supersedes the prior-entry "drafting-seriousness" interpretation.

## 2026-06-16 (cont) — calibration test REFUTES the few-shot hypothesis (gap is parametric)

calib_test.py: few-shot ICL (K=24 labeled boundary examples, group-split, predict 0-10
registrability, n=160) AUC=0.449 — BELOW chance (sign-flipped only 0.55). So the A->dense gap
is NOT closed by in-context calibration. CORRECTS the prior "needs calibration not articulation"
note. Refined ladder by label-learning: ICL-holistic 0.449 << A(rubric+trained-LR) 0.685 <
dense(fine-tuned) 0.764. The LLM's holistic merits judgment is ANTI-predictive because apparent
merits != outcome (descriptive marks register via 2(f)/disclaimer/response; strong marks abandon
for business reasons). A's 0.685 works because the LR learns from 4000 labels WHICH doctrine
metrics correlate + in which direction, correcting the LLM's naive priors. So: doctrine is
articulable, but the doctrine->outcome MAPPING is parametric (label-learned), not promptable.
The gap is "tacit = needs training on many labels," not a calibration nudge. Caveat: K=24 holistic
is one design point. This is the cleanest characterization of the legal ex-ante tacit gap so far.

## 2026-06-16 (cont) — CAVC→BVA join COMPLETE: validated record-linkage dataset (1,052 pairs)

User greenlit the CAVC→BVA assembly. Built it; key results + caveats:

**No exact key exists.** CAVC decisions reference the underlying BVA decision ONLY by date
("November 15, 2021 Board decision") + veteran name (REDACTED on anonymized BVA side) + issue.
BVA citation present in 0.0% of CAVC decisions, BVA docket in 0.1%. So this is a **record-linkage**
problem, not a key join: BVA decisions/date median 287 (p90 652). Disambiguate within the same-date
candidate set by TF-IDF cosine(CAVC fact-recital, candidate BVA text) gated by margin + a rare-token
fingerprint (years/%/$/service-branch). CAVC text used ONLY for matching, discarded; X = clean BVA text.

**Matcher** (cavc/cavc_match.py, date-grouped): 14,366 CAVC merits w/ BVA date → 1,158 accepted.
Spot-check precision HIGH (e.g. 25-1879: CAVC "Feb 18 2025 Board re a Feb 16 1984 rating decision
denying..." ↔ BVA A25014598 ORDER "the Feb 16 1984 rating decision...was not CUE" — pinned among
498 same-date candidates). Low-margin (~0.025) tail leaks ~1/3 (one granted-vs-denied mismatch) →
tightened to margin≥0.035 & fp≥0.40.

**Pool** (cavc/data/modeling_pool_cavc.jsonl.gz): 1,052 clean pairs, 424 affirmed(=1)/628 disturbed(=0),
1,023 unique veteran-dockets (group-split by BVA docket, train 721/val 176/test 155).
**LEXICAL AUC = 0.573** (group-split) — label is real but weak signal. Confounds clean (no metadata
leakage, equal length). Top features are ON-THESIS: →AFFIRMED = death/cause-of-death/etiologically/
war/COPD (verifiable causation); →DISTURBED = foot/examiner-opined/PTSD/anxiety/sleep-apnea/lay
(judgment-laden severity ratings remanded for inadequate reasons-or-bases). Almost a direct
illustration of the articulability thesis (verifiable issues upheld, tacit-judgment issues remanded).

**COVERAGE CAVEAT (the limiter).** Only ~1,158 of ~11,300 ceiling matched because the BVA corpus is
patchy: 2019-2024 AMA badly under-captured (2022_ama=1,720, 2020_ama=1,053, 2019_ama=0) vs
2025_ama=74,650. Bucket trace shows A22NNNNNN namespace has only ~1,720 entries (bucket 1 sparse,
buckets 2-3 empty) → likely a **va.gov publication limit for older AMA, NOT a fixable scraper bug**
(2025 dense, 2022 sparse — backwards from an early-stop story). Running scraper is backfilling pre-2016
legacy5 (low CAVC overlap) and will NOT revisit the .done AMA years. So dataset is ~capped near current
size unless va.gov exposes more AMA — OPEN QUESTION. Decision pending: run A/dense ladder on 1,052 now
(small, test n=155, noisy gap) vs. chase coverage first vs. treat issue-type finding as the qualitative result.

## 2026-06-16 (cont) — BVA root-cause fix + full-corpus re-scrape + DOL pool + metrics-ingest audit

GOAL set: every legal dataset in shape, maximize (ex-ante case × decision) joins, metrics ingested, ready for VAT.

**BVA scraper root-cause (the coverage cap):** old scrape_bva.py captured <10-25% of decisions —
3 bugs: (1) guessed lowercase files{seq//10000+1} but real subdirs are capital Files1..Files12 with
ARBITRARY shard numbers; (2) only fetched A-prefixed citations, but MOST decisions are bare-numbered
(2022 = 71,641 bare + 26,328 A); (3) EMPTY_BUCKETS_TO_STOP=2 early-termination. Authoritative complete
source = per-year sitemap (va.gov/vetapp{YY}/sitemap.xml lists every .txt). Verified live: sitemaps
1992-2026; true counts 2008=44,970 ... 2022=97,969 ... 2024=121,541 ... 2025=126,603 (I had <10-58% per
year). CAVC-overlap range 2008-2024 ≈ 1.4M decisions vs my 222K. Wrote scrape_bva_sitemap.py (recurses
sitemap-index, threaded polite download, gzip-member-per-batch append, resumable, matcher-compatible).
Retired old scraper (kept its data). FULL re-scrape LAUNCHED (pid running, 24 workers/36 rps, AMA-gap
years 2019-2024 first then 2008-2018; ~14h). Note: research subagent CLAIMED it wrote the scraper but
the file didn't exist — verified+rebuilt myself (don't trust unverified agent file-writes).

**CAVC:** improved date extraction (despace + bidirectional Board-anchored) lifted merits-w/-date
14,366→17,676 (97%); pairs 1,158→1,476. Will jump toward ~11K ceiling once re-scrape completes (bottleneck
is BVA coverage: only 535/2,259 needed dates populated pre-rescrape).

**DOL pool BUILT** (dol/modeling_pool.jsonl.gz): resolved x_ref→raw ALJ `content`; 6,194 binary pairs
(4,715 affirmed/1,479 disturbed; dol_brb 5,220 + dol_arb 974), group-split by case_id. LEXICAL AUC 0.740
(de-leaked: stripped hex IDs/digit-runs, AUC held = real signal). Top features substantive (rebuttable
presumption, withdrawal, remand, settlement terms, 29 CFR). Residual mild confounds (party names, KY
black-lung region, years) = follow-up de-leak.

**Metrics ingestion VERIFIED:** all 14 by-law banks load via real load_law_metrics (extracted.rubrics_metrics
w/ name): bva_veterans 61, cavc_review 45, dol_arb 34, dol_brb 54, erisa_ltd 46, flsa 51, mspb_cafc 57,
nlrb_ulp 42, ptab_aia 64, ss_disability 56, title_vii 50, trademark_examination 84, ttab_dupont 55,
ttab_exa 12 — all with what_to_look_for + applicability_note + thin flags. A-layer scorer reads
r["facts"]+r["binary_label"]. VAT-ready (GPU-gated).

**Pool inventory (canonical, per domain):** title_vii 6,410 | flsa 11,177 | ss_disability 52,560 |
erisa_ltd 2,750 | nlrb 2,585 | mspb 520 (FOIA-limited) | ttab_dupont 2,004 | ttab_exa 1,688 | cavc 1,476 |
ptab 15,345 | dol 6,194 (NEW) | trademark 79,936 (pool_v1, clean — NOT the .leaky sibling). Pair-dataset
pools = binary subset of assembled (already maxed; -1/graded excluded). MSPB coverage-limited by FOIA-only AJ text.

## 2026-06-16 (cont) — VAT harness wired (11/12 domains) + erisa gap + tier/patents precision

**Tier separation (per user):** TIER 1 = decision-derived facts narratives (facts extracted FROM the
court opinion, outcome stripped; same-doc leakage risk): title_vii 6,410, flsa 11,160, ss_disability
52,560, erisa_ltd 2,750. TIER 2 = true ex-ante (X is a real pre-decision doc, y a different body's later
outcome): trademark 79,936, ptab 15,345, dol 6,194, nlrb 2,585, ttab_dupont 2,004, ttab_exa 1,688,
cavc 1,476→, mspb 520. CORRECTION: ss_disability 52,560 is TIER 1 (district-court §405(g) facts), NOT
retrieved ex-ante docs. True ex-ante SS (claimant briefs) = DEAD (57/44,058 free, PACER paywall).

**Patents (separate arm, datasets/patents/):** prosecution-outcome = ~500K balanced each (first_draft,
final_outcome) + cpc 548K; X=abstract+claims (~6.5K chars), y=first_draft_approved/final granted.
6× trademark, richer X. BUT balanced CSVs are RANDOM-split (text+judgement only, no applicant/family
key in master patents_dataset.jsonl.gz) → applicant/family leakage risk. Clean re-split needs a
PatentsView assignee join or temporal split. Claim-level §102/§103 truecite testbed is separately
group-split-clean. FLAGGED, not auto-fixed (separate arm, needs external metadata).

**VAT harness (datasets/legal-outcome-prediction/vat_registry.py):** domain→pool→facts-field→label→
doctrine-bank map + CPU dry-run. 11/12 domains WIRED OK (facts+binary label+bank build a scoring prompt).
Pairs use x_text+y; tier-1 use facts+binary_label; trademark builds facts from mark+gs_text+class; flsa
canonical = fullpool_v3 (not relabel_v3). All 14 banks ingest (load_law_metrics, 34-84 metrics each).
GAP: **erisa_ltd has NO binary outcome label** (erisa_ltd_filtered = filtered candidates w/ `text`+
`merits_keep` flag only) → needs Tier-1 outcome extraction (LTD granted/denied) to enter VAT.

**Remaining for goal:** (1) BVA re-scrape ~63% through 2019 (→CAVC re-run, dedup-aware, when 2019-2024
land ~6h); (2) erisa binary-label extraction (GPU); (3) VAT scoring RUN across 11 wired pools (GPU,
GPU0 ~free); (4) patents re-split (flagged). MSPB stays 520 (FOIA-limited, no expansion path).

## 2026-06-16 (cont) — VAT harness PROVEN + UTF-16 scrape bug caught/fixed + CAVC dedup

**VAT harness end-to-end PROVEN (run_vat.py, FP8 Llama-70B offline batch, GPU0):** DOL smoke 300
cases × 88 metrics, NA 0.00. A-LAYER (all doctrine) CV-AUC 0.7027, V-LAYER (68 thin/checkable)
0.6898, vs lexical 0.740. Top signals real black-lung doctrine (coal_mine_employment_years_ge_15 =
statutory 15-yr presumption, filing-date gates, scope-of-review). So "go forward with VAT" demonstrated
on a NEW Tier-2 domain. run_vat reports V (thin) + A (all) separately, driven by vat_registry.

**UTF-16 scrape bug (CAUGHT before scaling):** CAVC re-run on partial new sitemap data showed 2019:
10/76,200 date-parsed. Diagnosed: 2019-era AMA .txt files are UTF-16-LE (BOM ÿþ), my scraper hard-coded
decode("latin-1") → 99.99% garbled. The scrape does AMA years 2019-2024 FIRST = exactly the garbled ones.
STOPPED scrape (exact PIDs), fixed encoding (BOM-detect utf-16/utf-8-sig/utf-8/latin-1), verified on a
live UTF-16 url (decodes clean, date parses), moved garbled 2019_sitemap aside (.utf16garbled), relaunched
(logs/sitemap_full2.log). Lesson: validate decoded text per-year before trusting a multi-hour scrape.

**CAVC dedup re-run:** added citation-dedup to BVA indexing (old legacy + new sitemap overlap). Current
1,535 pairs (from 1,476) — small gain because the big 2019 sitemap was garbled; will jump once the FIXED
re-scrape lands clean AMA years. matched_pairs label balance 663 affirmed/872 disturbed.

## 2026-06-16 (cont) — erisa built (12/12 datasets in shape) + full VAT ladder run launched

**erisa_ltd FINISHED (last dataset gap closed):** Tier-1 extraction via FP8-70B over 2,750 opinions →
erisa_extracted.jsonl (claimant_loss 1,245 / claimant_win 643 / mixed 129 / procedural 733). Built
erisa_ltd_canonical.jsonl: 1,888 binary (643 win / 1,245 loss = 34% win, matches ERISA pro-plan reality),
group-split by opinion_id. De-leaked facts (stripped insurer names unum/hartford + outcome verbs):
LEXICAL AUC 0.693 HELD after de-leak = real signal (worsened/degenerative/severe/chronic→win;
alleges/failure→loss). Label cross-check 73% vs crude disposition heuristic (fired on 464/1888) — the
one soft spot. erisa_pool.py + erisa_extract.py on sk3.

**12/12 legal datasets VAT-WIRED (vat_registry dry-run: ALL TRUE).** Final pools: title_vii 6,410, flsa
11,160, ss_disability 52,560, erisa 1,888 (Tier 1); trademark 79,936, ptab 15,345, dol 6,194, nlrb 2,585,
ttab_dupont 2,004, ttab_exa 1,688, cavc 1,476→, mspb 520 (Tier 2).

**FULL VAT LADDER RUN launched** (run_vat_all.py, GPU0, one model load, 600 balanced/domain, ~1-2h →
vat_ladder.json): per-domain A-AUC (all doctrine) + V-AUC (thin/checkable). cavc excluded (mid re-scrape,
score after). This is the "go forward with VAT metrics" capstone — the actual V/A measurement across all
legal domains. DOL proof was A=0.703/V=0.690.

GOAL STATUS: (1) every dataset in shape = 12/12 DONE. (2) maximize joins = BVA re-scrape running (CAVC
will jump from 1,535), DOL built, others maxed/FOIA-capped. (3) metrics ingested + VAT-ready = DONE
(proven + 12/12 wired + full ladder running).

## 2026-06-16 (cont) — FULL VAT LADDER (V vs A) across legal domains — core deliverable

run_vat_all.py (FP8-70B offline batch, 600 balanced/domain, 5-fold CV; A=LR over ALL doctrine metrics,
V=LR over THIN/checkable subset). vat_ladder.json. First pass 8/11 (3 failed on 4096-token cap — re-running
at 8192). Results:

| domain        | tier | A(all doctrine) | V(thin) | A−V gap |
|---------------|------|-----------------|---------|---------|
| erisa_ltd     | 1    | 0.758 | 0.548 | +0.209 |
| title_vii     | 1    | 0.637 | 0.576 | +0.061 |
| ss_disability | 1    | 0.626 | 0.529 | +0.097 |
| flsa          | 1    | 0.604 | 0.603 | +0.002 |
| dol           | 2    | 0.728 | 0.717 | +0.011 |
| trademark     | 2    | 0.641 | 0.620 | +0.022 |
| mspb_cafc     | 2    | 0.596 | 0.582 | +0.014 |
| nlrb          | 2    | 0.504 | 0.502 | +0.002 |

READINGS: (1) erisa biggest articulable-beyond-thin gap (+0.21) — much doctrine is articulable-but-not-
checkable. (2) dol/trademark = rule-heavy: V already ≈ A and both HIGH (black-lung presumptions, registrability
gates are thin+decisive). (3) nlrb at CHANCE (A=0.50) — ULP doctrine metrics carry NO signal for ALJ→Board
review; that outcome isn't doctrine-driven (real negative result). (4) flsa flat (V≈A≈0.60). Cross-domain the
A-layer (articulable doctrine) ranges 0.50→0.76 — the "articulability ceiling" is strongly domain-dependent.
Compare to lexical refs: erisa lex 0.693 < A 0.758 (doctrine reads deeper than TF-IDF); dol lex 0.740 ~ A 0.728.
Pending: ttab_dupont/ttab_exa/ptab_aia re-run (8192 ctx); cavc after re-scrape. Per-domain A-matrices cached.

## 2026-06-16 — VAT LADDER COMPLETE (11/11 legal domains; cavc pending re-scrape)

Final vat_ladder.json (FP8-70B, 600 balanced/domain, 5-fold CV; A=all doctrine metrics, V=thin/checkable):

| domain        | tier | A     | V     | A−V    |
|---------------|------|-------|-------|--------|
| erisa_ltd     | 1    | 0.758 | 0.548 | +0.209 |
| dol           | 2    | 0.728 | 0.717 | +0.011 |
| trademark     | 2    | 0.641 | 0.620 | +0.022 |
| title_vii     | 1    | 0.637 | 0.576 | +0.061 |
| ss_disability | 1    | 0.626 | 0.529 | +0.097 |
| ttab_dupont   | 2    | 0.613 | 0.594 | +0.019 |
| flsa          | 1    | 0.604 | 0.603 | +0.002 |
| mspb_cafc     | 2    | 0.596 | 0.582 | +0.014 |
| ttab_exa      | 2    | 0.584 | 0.549 | +0.035 |
| ptab_aia      | 2    | 0.523 | 0.529 | -0.006 |
| nlrb          | 2    | 0.504 | 0.502 | +0.002 |

KEY READINGS: (1) Articulable-doctrine ceiling (A) is strongly DOMAIN-DEPENDENT: 0.50 (nlrb/ptab ≈ chance) →
0.76 (erisa). (2) A−V gap = articulable-but-not-thin-checkable doctrine: largest in Tier-1 decision-derived
slices (erisa +0.21, ss +0.10, title_vii +0.06); near-zero in rule-heavy Tier-2 (dol/trademark V≈A both
HIGH — black-lung presumptions + registrability gates are thin AND decisive). (3) nlrb & ptab at chance →
those review outcomes (Board affirm, PTAB institution) are NOT captured by codified doctrine metrics =
the "tacit/strategic residual" is large there. (4) These are A-layer ceilings; dense models + lexical give
the upper rungs (e.g. dol lexical 0.740 ≈ A 0.728; erisa A 0.758 > lexical 0.693). Per-domain A score-matrices
cached. NEXT: add cavc_review once BVA AMA re-scrape completes (run_vat.py --domain cavc_review).

## 2026-06-17 — CAVC→BVA join MAXIMIZED: 1,535 → 10,117 pairs (sitemap re-scrape payoff)

The BVA sitemap re-scrape (UTF-16 fix) delivered: BVA corpus 222,830 → **1,128,951 decisions** (4.9GB),
100% date-parsed, 3,225 needed-dates populated (was 535), only 4,174 CAVC cases lack a candidate (was
15,632). CAVC matcher re-run (dedup-aware) → **10,117 accepted pairs** (was 1,535 = 6.6×), 3,984 affirmed /
6,133 disturbed. Modeling pool (tightened+scrubbed+group-split): **9,140 clean pairs**, 8,734 unique
veteran-dockets, full year coverage 2008-2025. **LEXICAL AUC 0.627** (group-split) — UP from 0.573 at 1K
scale → precision HELD at scale (noisy matches would have lowered it). Confounds on-thesis (causation/
service-connection→affirmed; rating-%/lay-evidence→disturbed). This realizes the "get the numbers up"
directive — CAVC went from a borderline 1K set to a robust ~9K Tier-2 dataset. Adding cavc to the VAT
ladder now (12th domain). Root-cause lesson banked: blind citation-sequence scraping missed ~80% of BVA;
the per-year sitemap is the complete authoritative source.

## 2026-06-17 — GOAL COMPLETE: full 12-domain VAT ladder (cavc added at 9,140-pair scale)

cavc_review scored A=0.620 / V=0.619 (n=600; top signals = reasons-or-bases adequacy metrics:
evidence_items_discussed, decision_length, caluza_element_coverage, exam_adequacy — V≈A so thin
checkable metrics already capture CAVC's doctrine; A≈lexical 0.627). FINAL vat_ladder.json (12 domains):

| domain        | tier | A     | V     | A−V    |
|---------------|------|-------|-------|--------|
| erisa_ltd     | 1    | 0.758 | 0.548 | +0.209 |
| title_vii     | 1    | 0.637 | 0.576 | +0.061 |
| ss_disability | 1    | 0.626 | 0.529 | +0.097 |
| flsa          | 1    | 0.604 | 0.603 | +0.002 |
| dol           | 2    | 0.728 | 0.717 | +0.011 |
| trademark     | 2    | 0.641 | 0.620 | +0.022 |
| cavc_review   | 2    | 0.620 | 0.619 | +0.001 |
| ttab_dupont   | 2    | 0.613 | 0.594 | +0.019 |
| mspb_cafc     | 2    | 0.596 | 0.582 | +0.014 |
| ttab_exa      | 2    | 0.584 | 0.549 | +0.035 |
| ptab_aia      | 2    | 0.523 | 0.529 | −0.006 |
| nlrb          | 2    | 0.504 | 0.502 | +0.002 |

GOAL (3 parts) COMPLETE: (1) 12/12 legal datasets in shape; (2) ex-ante×decision joins maximized
(CAVC 1.5K→10.1K via BVA 222K→1.13M sitemap re-scrape; DOL 6.2K built; pairs maxed; mspb FOIA-capped;
patents flagged needs-applicant-key); (3) all 14 metric banks ingested + 12-domain V/A ladder measured.
Harness: vat_registry.py / run_vat.py / run_vat_all.py; per-domain pools + CONFOUND_*.txt. NOTE: BVA
scraper still backfilling pre-2012 legacy years (bonus coverage); cavc could be re-scored at full 9,140
(currently 600-sample CV) if desired. Open follow-ups: patents temporal/applicant re-split; dol residual
party-name de-leak; cavc 73%→ tighten match precision if needed.


## 2026-06-19 — ERISA leakage probe: is the +0.21 A−V gap real doctrine or judge-framing leakage?

The biggest A−V gap in the 12-domain ladder (erisa_ltd, A=0.758/V=0.548) is Tier-1 (facts extracted FROM the
opinion), so the thick "decision-quality" metrics (cherry_picking, shifting_grounds) could be reading the
reversing judge's framing rather than independent doctrine. Probe (`erisa_leakage_probe.py`, FP8-70B, GPU6,
same 600 balanced cases / same 5-fold): re-state each case 2 ways — **faithful paraphrase** (control: framing
kept) and **neutralized** (strip every evaluative characterization, keep all medical findings/dates/procedure)
— re-score all 46 metrics on each, recompute A/V.

| condition | A(all) | V(thin) | gap |
|---|---|---|---|
| original | 0.756 | 0.576 | +0.180 |
| paraphrase (control) | 0.737 | 0.517 | +0.220 |
| neutralized | 0.702 | 0.558 | +0.144 |

**VERDICT: gap is PREDOMINANTLY REAL doctrine, with a measurable (~⅓) framing tax precisely where predicted.**
A degrades monotone 0.756→0.737→0.702; neutralization-beyond-rewording costs ΔA≈0.035, but A stays 0.702 —
still far above V and the LARGEST gap (+0.144) in the whole ladder. BITE confirms the manipulation was
selective: neutralization moved thick-metric scores 0.091 vs thin 0.042 (~2×; thin=dates/numbers are
framing-invariant). Per-thick-metric discrimination LOSS (orig→neut) lands exactly on the meta-assessment
metrics I flagged a priori — argument_specificity −0.060, cherry_picking −0.058, procedural_regularity −0.047,
shifting_grounds −0.041 — while the **medical-substrate doctrine survives intact**: treating_physician_support
0.613→0.609 (−0.004), governing_disability_definition_satisfaction 0.678→0.657 (strongest thick metric),
internal_consistency_of_record 0.676→0.638. So: the decision-quality-characterization doctrine is ~⅓
framing-driven; the does-the-evidence-meet-the-definition doctrine is genuine articulable-but-thick judgment.
CAVEAT (makes this a LOWER bound): the ERISA "facts" were already outcome-stripped at extraction, so limited raw
framing remained to remove (BITE only +0.012 beyond paraphrase rewording-noise); V also carries cross-condition
noise (0.58/0.52/0.56) so lean on A-levels + per-metric drops, not exact gap deltas. Cross-model Qwen-122B
neutralizer would tighten the ⅓ estimate but the qualitative conclusion is robust. No GPU zombie; clean exit.
See memory [[project_verifiability_explainability_gaps]].

## 2026-06-22 — prompt-optimality theory: anchor-free recovery + Ω-orthogonalization framework integrated

Integrated a unified critique/framework update into BOTH theory docs (`notes/2026-06-18__prompt-optimality-theory.md`
+ the clean `notes/prompt-optimality-whitepaper-latex/prompt-optimality-whitepaper.tex`) and the code. Eight standing edits, all faithful to the
agreed anchor-free recovery objective (NO holistic anchor `M`, NO label `Y`):

1. **Mechanical recovery loop made explicit** (§1): `M_train = m(X_train)` → reconstructor writes `p̂` →
   `R(p) = I_TVD(M_test ; M̂_test)`, both legs on the **held-out** split. Reads high only when `p` runs a
   *generalizable* rule, not memorized train surface features.
2. **DPI same-distribution fix** (§2.2): split `T_train` (in-sample consistency — a reliability number, NOT a
   ceiling) from held-out `T_test = I_f(M̂_test;X_test)`. `R ≤ T_test` holds within ONE distribution only; the
   old draft's "T = in-sample transmission of §1" conflation is removed. `A = T−R` is now the held-out same-`f`
   gap (≥0 by Jensen); the train→test difference is named as a *separate* generalization quantity feeding
   component (b). `vinfo.tvd_guardrail` already computes both legs from the held-out `recovered`, so the code
   was correct — the docs were the slip.
3. **Rung 1 downgraded** (§0/§3.1): `cap_f` is a **channel-capacity sanity check** (verify `R̂≤cap`; detect
   binary-readout compression), NOT a proximity-to-optimum KPI — `cap−R̂` must not be reported as
   distance-to-optimum (vacuous: a constant of the readout, not the task).
4. **Spectral-γ DROPPED** (§6.2/§6.6): Das–Kempe `λ_min` assumes a linear-regression value function, false for an
   attention-mixing LLM. Replaced by the Shannon-CMI **orthogonalization filter** + brute-force `|Ω|≤15` (bypasses
   γ) as the route to a trustworthy within-class bound.
5. **§6.5 Discovery-to-Selection rewritten** to 4 steps: mine *semantic diffs* (not winners) → **orthogonalization
   filter** (drop paraphrases whose behavior `Ω` already explains; Shannon CMI because TVD has no chain rule) →
   **canonical compiler, fixed order Format→Semantics→Negative** → certify (brute-force/double-greedy, TVD).
   Atomic unit redefined as a **behavioral partition operator** (don't split composites the executor reads as one
   signal).
6. **Good–Turing REPLACED** (§6.7c/§6.9) by the two-pronged missing-*impact* defense: (i) submodular tail-bound
   — `max_{e∉Ω}Δ(e∣Ω) ≤ min_i[greedy gain]_i` (the *certified* form; loo-min is a tighter diagnostic, not a
   guaranteed bound), conditional on tail-`γ≈1`; (ii) adversarial behavioral saturation `I(X_probe;M∣X_Ω)≈0`
   (no submodularity assumed). Halt when both fire.
7. **Permutation test formalized** (§6.8): set-abstraction valid iff `σ²_subset ≫ σ²_perm`; order is a bounded
   `√σ²_perm` residual.
8. **Pure-`R(p)` scope constraint** (§10): all rungs certify pure `R`, NOT the shipped fidelity scalarization.

**Code:** new `methods/metric_implementer/experiments/orthogonalize.py` (+ `tests/test_orthogonalize.py`, 6 pass) —
`shannon_cmi_surrogate`, `orthogonalization_filter`, `submodular_tail_bound` (returns `certified_bound` =
smallest greedy gain AND loo `tail_bound`), `adversarial_saturation`, `permutation_order_test`. Self-check ALL PASS
(filter drops planted paraphrases; noise probe saturates, M-relevant probe does not; σ²-ratio ≈ 927). In-place
relabels: `vinfo.py` (held-out `T_test` docstrings + `cap_sanity_check`), `real_gamma.py` (spectral-γ → "heuristic
diagnostic, NOT a certificate"), `harvest_gepa_omega.py` + `ladder_synthetic.py` (pipeline/cap reframing).
23 vinfo+orthogonalize tests green; whitepaper braces balanced. ALL ZERO-GPU/local. Honesty note: the user's
tail-bound used `min(loo)`; the rigorously-valid bound is the smallest greedy marginal — kept both, flagged in
docs + code.

## 2026-06-23 — GEPA loop WORKS on press_releases: GLM(mutator,subscription API)→Gemma(target)→Qwen(judge)

Built + ran the first real GEPA (Genetic Prompt Algorithm) loop for norm-extraction prompt optimization.
Artifact: `scripts/llama_norm_extraction/gepa_pr.py` on sk3 (3 composable subcommands + a `run` driver). Pipeline:
**GLM mutator (z.ai subscription API, 0 GPU)** → **Gemma-4-31b target (1 GPU, BF16)** → **Qwen-122B-A10B-FP8 judge
(1 GPU)**. Label-free distillation objective (no gold labels): maximize COVERAGE on signal-rich positives
(fraction with ≥1 faithful+valid normative signal) s.t. validity-precision ≥ 0.50 floor. Per user: press-releases
positives carry all the normative signal (wire-service negatives are signal-poor by construction) — that's fine
because the goal is ANCHOR DATA to label metrics, not a calibrated distribution; negatives serve only as a
firehose/specificity guard.

**Results (15 signal-rich positives, 30-pair eval):**
| prompt | coverage | precision | good anchors | sigs |
|---|---|---|---|---|
| baseline (v3) | 0.733 | 0.625 | 55 | 88 |
| round 1 (GLM) | 0.867 | 0.597 | 43 | 72 |
| **round 2 (GLM)** | **0.867** | **0.695** | 41 | 59 |

Round-1 mutation (GLM sharpened `role`/`inline_evidence_example`/`polarity_hint` toward "journalist is
EVALUATING/QUESTIONING the announcement") lifted coverage +18% (11→13/15) by cracking a whole source cluster the
baseline missed. Round-2 then **Pareto-improved** precision +0.10 (0.597→0.695) while holding coverage — the trimmed
signals were almost all spurious. **Coverage plateaued at 0.867 because it is JUDGE-bound, not gen-bound**: the 2
residual misses are borderline judge calls (Amazon-unionization "aggressively fought" = arguably editorial framing
the judge reads as neutral fact) + one genuinely signal-poor article (Scaramucci/SkyBridge). Judge itself validated
by spot-check: accepts real non-disclosure/omission/framing signals, rejects plain facts/neutral reports.

**Deploy:** user chose full 100K-corpus run (greenfield — no existing full extraction). Added resumable chunked
`gen_corpus` subcommand (per-2K-chunk flush + skip-done; safe-overnight). Running on GPU 1 (round-2 prompt +
`press_releases_full_v3_minimal.json` few-shot, article_only) → `deploy_round2_full.jsonl`; then Qwen-judge-filter
→ anchor pool. (~2–4h.)

**Bug fixes during the loop:** (1) Qwen3.5 judge needs `limit_mm_per_prompt={image,video,audio:0}` + greedy
`chat_template_kwargs={"enable_thinking":False}` or it emits `<think>` blocks that break JSON parsing → all-empty
verdicts; (2) GLM mutate response truncated mid-JSON at max_tokens=1400 (GLM rewrote the long evidence fields even
more verbosely) → bumped to 3200 + parser now strips ```json fences and lets `json_repair` close truncated braces;
(3) keep-criterion `coverage > best` was too strict (discarded round 2's precision win on tied coverage) → fixed to
Pareto-dominance (cov up, OR cov tied + prec up, both ≥ floor); (4) stale-file waiters — arm on PID-exit +
terminal-log marker, not file existence.

## 2026-06-23 — Patents option3 (symmetric Gemma build) spot-check + §102-tacit retraction

Spot-checked the in-flight option3 Gemma symmetric build (`option3_results_gemma.jsonl`, ~64% of 95,798 refs;
11,975 claims = 5994 pos / 5981 neg, K=8 refs each; pos = examiner gold planted among CPC fillers, neg = 8
retrieved same-CPC fillers, gold hidden in `_audit_gold_docs`). Findings:

- **Distribution symmetry SOLID** (the property we wanted): empty-spans 0.1%/0.1%, #spans 1.0/1.0 both, span
  word-len pos 35.8 / neg 33.9. Forced-passage prompt held at scale.
- **Grounding ~99%**: the 10% non-verbatim "miss" is cosmetic numeral-joining (`d4are`, `core21calculates`,
  FIGS spacing — PatentsView inlining quirk), NOT hallucination.
- **Negatives are clean by construction** (retrieved same-CPC near-miss, not rejected-app examiner cites).
  neg-disclose 15% = broad-element overlap, not contamination.
- **Within-positive localization is weak**: gold-disclose 31% vs filler 20%; gold-recall 11.6%, precision 19%.

**Disclosure rate is PARSING-dominated.** option3's `element` averages 48.5 words (long multi-clause limitation)
vs the canonical `localize_units_scale` element at 22.9 words (tight examiner-mapped). Same Gemma model → option3
31%/15%/~7pt-sep vs canonical 58%/30%/28pt-sep. A number that swings ~2× with how *we* carve the element cannot
be evidence about a property of §102 itself.

**RETRACTION — §102 is NOT tacit.** Earlier "patent §102 anticipation is largely tacit / weakly
verifiable-from-text" verdicts are retracted. User's reasoning (sound): (1) parsing-dominated measurement is
circular; (2) §102 = every element literally in ONE reference = the most text-grounded standard, the opposite of
tacit — tacitness belongs to §103 combination / inherent anticipation, and examiner §102 cites are concrete
readable disclosures, so low measured rates are pipeline failures (localize/truncate/parse), not tacitness;
(3) no evidence of "lots of negatives with very convincing prior art" — neg-disclose 15–30% is broad-element
trivialities, so the boundary looks sharp and positives are under-surfaced, not tacit. Good LLM localization
recovers ~58–82% positive disclosure = direction consistent with "§102 IS verifiable."

**Leak-probe (format-only, grouped-5fold LogReg by app_id): AUC 0.595** (target 0.50). Culprit = full-doc
detail-desc length structure (`len_mean` 0.617, `len_std` 0.603); doctype/count clean (`frac_pgpub` 0.500,
`n_refs` 0.500). The tell is in candidate-set structure (seen by the Stage-1 localizer); the model-visible span
at Stage-2 is length-symmetric. **Fix for a clean claim: length-match/cap candidate docs pos vs neg, re-probe.**

Status: build watcher armed (background, notifies on 95,798 or stop). V/A framing still UNLOCKED. On completion:
length-fix + re-probe → finalize per-claim K-ref artifact → manual pos/neg side-by-side → then any V/A.

## 2026-06-23 (cont) — GEPA on legaladvice_uk: precision-limited mirror of press_releases; baseline IS yield-optimal

Ran the corpus-parameterized GEPA (`gepa_pr.py GEPA_CORPUS=legaladvice_uk`) on a 2nd corpus in parallel with the
press_releases deploy. legaladvice_uk is the **precision-limited mirror** of press_releases: dense advisory text →
coverage near-ceiling (baseline 0.933) but precision below floor (0.457) — the extractor pulls factual-recitations-
OF-law ("you can't take out a life insurance policy except…", "part of the Consumer Contract Regulations 2013")
rather than advisory NORMS ("you have 14 days from delivery to cancel").

**Made the machinery axis-aware:** mutator GOAL branches on whether prec < FLOOR (precision-emphasis) vs ≥
(coverage-primary); keep-criterion = F1+floor (symmetric, handles both axes, picks same PR winner). Added a YIELD
objective (`GEPA_OBJECTIVE=yield`, loose 0.35 floor) to test "high volume = more anchors."

**Full tradeoff curve (15-pos eval):** baseline cov 0.933/prec 0.457/n_good 86/sigs 188 → F1-r1 0.80/0.744/67/90 →
F1-r2 0.733/0.857/48/56 (F1-winner). Yield-r1 0.867/0.527/68/129. **Key finding: baseline is already yield-optimal**
— it's the widest net (188 sigs), and EVERY GEPA mutation (any objective) reduces n_good from 86. GLM has a
persistent selectivity bias even under the wide-net goal (yield-r1 = 129 sigs < baseline 188). So GEPA's value on
this corpus is the **purity** axis (F1-r2: 0.857 prec), not yield — opposite of press_releases (coverage). **Per
user: yield objective adopted, tested, baseline confirmed as the anchor-count ceiling.** Choice is volume
(baseline, 86 anchors) vs purity (F1-r2, 48 @ 0.857) — application-dependent.

legaladvice_uk loop stopped (baseline = yield-best; F1-r2 = purity-best). press_releases 100K deploy continues
unaffected on GPU 1.

## 2026-06-23 (cont) — prompt-optimality: prose round CLOSED, held-out wiring (Q1), submodularity = optional layer

**Prose round CLOSED.** Compiler sweep (CW, saved npz): compiler-INVARIANT — `echo_prose` ceiling I=0.072
= T_prose; conjunction/weighted_sum/prose_join all I≈0.064–0.070. **T_prose is the binding cap on
I(M,M_ω), not the C(Ω) framing.** Discriminative-GEPA re-run (w_disc 0.40→0.55, warm-start) did NOT raise
T_prose — regressed 0.072→0.022 (leniency won; orthogonalization collapsed K→2). ⇒ pivot to criteria-based
parseable GEPA (dodges T_prose).

**Held-out wiring finding + Q1.** The within-class certificate (`omega_certificate`) is entirely IN-SAMPLE
(M, every mhat, R(S) on the same N items). The held-out DPI machinery (`tvd_guardrail`, `recon_channel`,
`n_reconstruct_behavioral=24`) is a SEPARATE pipeline never called from the certificate run — the gap. **Q1
WIRED** (free re-split of scored MHAT/M in `omega_certificate.run`, `--holdout-frac`/`--holdout-seed`):
pick S* on train, report held-out R/T_test/A/DPI. Prototype on Aigner–Ziegler npz (N=200, K=6): **prompt
generalizes — gen_gap +0.013±0.022, DPI R≤T_test 8/8 splits.** Significance: CI95 half-width ≈ 0.31/√N ⇒
resolve ε with N≈(0.31/ε)² items (~20–30 for R>floor; ~150–200 for near-equivalent-bound maps).
Equivalent-bounds set moderately flat (30% of subsets within ε=0.02 of OPT); singletons reach ε=0.05 ⇒
**side-channel-leakage signature**. S∈Ω vs S\*: +34% over avg subset, +60% over singleton; **full-set WORSE
than S\*** (mild PRUNE); composition saturates at ~2 units.

**Submodularity = OPTIONAL top layer (theoretical landing).** Layered: the core (DPI R≤T_test, held-out R,
exact cert |Ω|≤15 bypassing γ, articulation gap A) is **submodularity-FREE**. γ/U₂/double-greedy/tail-bound
enter only for the |Ω|>15 fallback. **Theory does NOT break without submodularity** — worst case = measured
optimum + valid DPI ceiling, no certified approx ratio (code already reports fallback as
measured-not-certified). The genuinely load-bearing axis is **validity** (real I vs confounded
side-channel), distinct from submodularity. **No heavy ignorability machinery**: v-information (Xu 2020)
gives ignorability BY CONSTRUCTION via a constrained predictor class ({criterion-prompt → E's verdict});
constrain criteria causally-semantic → I_V excludes side-channels, no causal model (sidesteps
natural-language-entropy unidentifiability). Instruments already built compose into a **channel-cleanliness
gate** (criterion stays in Ω iff `adversarial_saturation` I(X_probe;M|X_Ω)≈0 AND no PRUNE-help AND
`counterfactual_validity`). **PRUNE-help = leakage alarm** (reframed from a structural claim). **Convergence:**
clean channels remove spurious non-monotonicity → near-monotone → near-submodular EMERGES (not assumed) →
re-enables cheap γ guarantees as bonus. Never need a standalone "prove γ" step. (Theory §6.10 +
scorecard V1–V3 added to `notes/2026-06-18__prompt-optimality-theory.md`; whitepaper Remark added.)

**Plan (user-ordered):** (1) cleanliness gate [#31], (2) Q2 prompt-transfer generalization (disjoint
GEPA/test split in run_real_test) [#30], (3) multi-task GPU sweep (CW+peer-review+code-review+math)
validating Q1/Q2. Memos: `project_upper_bound_heldout_wiring`, `project_submodularity_optional_layer`.

## 2026-06-24 — GEPA anchor-harvest batch (norm-extraction → metric labels)

Four corpora moved through the sanctioned **GEPA → full-corpus deploy → Qwen judge → anchor pool** pipeline
on GPUs 1/2/3/7.

- **humor_multi GEPA: DONE — strong winner.** Axis-aware mutator traced prec 0.496→0.673→0.767→**0.866**
  at cov 0.933 (F1≈0.90) over 4 rounds. round3.json is precision-saturated (explicit VALID/INVALID
  craft-criteria + verbatim-substring + truncation-faithfulness). Deploying on full 48.5K threads.
- **press_releases: deploy DONE (100K signals) → judge GPU1** (~31.6K signal-records; coverage-limited,
  ~0.5 anchors/rec — only positives carry normative signal, by design).
- **legaladvice_uk: deploy DONE (50K) → judge GPU3** (resume; ~44.5K signal-records, ~0.8 anchors/rec).
  **flashinfer `aligned_alloc` workspace-overflow crash at engine init was GPU contention from a leftover
  crashed EngineCore, NOT a config bug** — clean GPU → no recurrence. (Kill zombie EngineCores by PID
  before relaunch; don't re-investigate vLLM/flashinfer versions for this.)
- **wp_comments GEPA: running GPU7** — ROUND0 cov 0.6 / prec 0.486 (precision-limited start). Next: deploy winner.
- Remaining priority corpora wired+smoke-tested, launch-ready: **math_se, crse, code_review, peer_review**.

Pipeline per corpus: `cmd_run` (GLM mutate via zai_anthropic 0-GPU → Gemma gen → Qwen judge; f1 or yield
keep-criterion) → `gen_corpus` (Gemma full deploy, resumable chunked) → `judge_corpus` (Qwen faithful∧valid
→ anchors.jsonl). All in `scripts/llama_norm_extraction/gepa_pr.py`; per-corpus judge framing in
`judge_sys/<corpus>.txt` (loaded via GEPA_CORPUS env).

## 2026-06-25 — press-release deconfounding audit: the 0.71 ceiling is mostly confound

Picked **press_releases** as the next corpus to push through the V/A passage (cleanest "taste-dominated"
contrast to law, per pitch). First ran a legal-vat-style **deconfounding audit** — and it overturns the
pitch. Full note: `notes/2026-06-25__press-release-audit.md`; artifacts in scratchpad `pr_audit/`.

- **Canonical training CSV is gzip-truncated** (`press_release_modeling_dataset_clean.csv.gz`: only
  41,607/128,131 rows readable, prefix 81% pos). Rebuilt the modeling table from intact id-keyed sources
  (clean jsonl ⋈ v1 metadata ⋈ topic model) → 128,131 rows, 42.0% pos. Label = covered by ≥1 tracked outlet.
- **Two dominant confounds, both near the dense 0.71:** publisher identity AUC **0.673** (LOO company
  coverage-rate), topic AUC **0.610** (pos-rate 0.05 Beauty → 0.59 Government). Minor: 38% empty clean
  extractions (skew neg), clean-body length 0.547, language 98.4% EN (non-EN skews neg).
- **TF-IDF+LR deconfounding ladder:** raw random 0.675 → publisher-grouped 0.605 → deconfounded
  (clean·EN·len) grouped natural-topic **0.584** ← honest ceiling → topic-balanced **0.546** (within-topic
  craft). Decomp: publisher memorization **+0.056**, topic-selection **+0.038**, craft residual **+0.046**
  over chance. TF-IDF leak features = wire boilerplate (`prnewswire`, `news provided`), HTML cruft (`amp`),
  firm/ticker tokens; scrubbing them moved grouped AUC only 0.546→0.545 (signal is spread across entities).
- **Implication:** the rubric(0.55)↔dense(0.71) "tacit gap" I advertised is **mostly confound, not taste** —
  rubric methods already sit near the honest ~0.58 ceiling. Press_releases is a **low-ceiling / identity-driven**
  task like `news_homepages`, not a taste showcase. For a clean taste-dominated domain, **creative_writing**
  (dense unsaturated → 0.90) is the better next pick.
- **Durable artifact:** `datasets/press-releases/press_release_deconfounded.parquet` (72,315 rows, clean·EN·len,
  stable company-hash 80/10/10 split, 0 straddle, topic kept for per-fold balancing).
- **Open:** re-run the *dense* ceiling on the publisher-grouped deconfounded split to confirm 0.71→~0.58 for a
  non-linear model (currently inferred from the linear grouped/random gap); locate intact training CSV on sk3.

## 2026-06-27 — PR test-exec: 1k scaling + stale-lock deadlock fix
- **BUG FIXED (system-wide):** every factory docker-build was deadlocked on a stale `/tmp/pr_build.lock` regular FILE on sk1/sk2/sk3 (Jun 16-20). `_docker_build_under_lock` uses `os.mkdir` (expects a dir); a stale file → FileExistsError → 4h spin → retry → forever. dvc hung 9h at 0.4% CPU. Removed stale files + patched prep_repo.py to auto-remove a stale-file sentinel. Verified dvc now building. Synced to all hosts.
- **1k goal locked:** assigned 999 repos (333/host, threshold acc≥5/rej≥2) via `scripts/assign_scaling_repos.py`. ~45K PRs to fetch+exec.
- **3-script pipeline:** `fetch_producer.py` (prep_repo A-D fetch-only, running on sk3) → `host_factory_loop.py` (per-host factory --process + run_local exec) → merge + re-parse + MH-by-repo correlation.
- **Host capacity:** sk3=11T, sk2=5.3T (workhorses); sk1=40G/100% disk-poor (datasets/ is shared data, not purgeable) → sk1 for shipped-image exec only.
- **Status:** sk3=dvc building + mx-chain-go exec + fetch_producer; sk2=etcd_rej exec; sk1=deferred (disk).

## 2026-06-29 — sk1 ONLINE as 3rd lane (corrects the "sk1 disk-poor" call) + reject coverage verified
- **CORRECTION:** sk1 is NOT disk-poor. Only `/lfs` (md0p1) is tight (~86G free); docker root `/var/lib/docker` + `/tmp` are on `/dev/nvme4n1p2` (879G, **536G free**), docker active, repo present, 40 pre-existing verdicts. sk1 runs full prep+build+exec fine. Launched as 3rd lane (PID 3723857, 343 repos).
- **Recurring corpus bug:** BOTH sk2 and sk1 were missing `code_review_modeling_dataset.csv.gz` (1.42GB; prep_repo phase A reads it) → every repo failed "No such file" instantly. Fix = rsync laptop→host (hosts can't reach each other). Lesson: verify corpus exists on a host before launching its loop.
- **All 3 loops healthy & producing** (sk1 repo1 ocs-ci, sk2 repo2 decred/dcrdex, sk3 repo7 iotex-core); ~78 repos / ~31K verdict rows at sk1-launch. dvc + presto(Java) now complete cleanly (lock+java_gradle fixes holding).
- **Reject coverage confirmed sufficient for "all rejected PRs":** manifests already carry a ~20% reject cell (qiskit 293acc/71rej) via prep_repo phase-C reject-verify; laptop consolidated reject-scrape ALIVE deep-filling the high-signal repos (containerd 809, gateway-api 160 rejects). No accept-confound at manifest level.
- **3-host death-alert monitor** b38tagy96 (ssh-fail tolerant, replaces 2-host b2w9ff0fk). Next milestone: merge+re-parse+MH-by-repo correlation when the 1k lands (task #227).
- **FIXED systemic UnicodeDecodeError** (was dropping whole repos): `prep_repo._gh` + `._run` used `subprocess(..., text=True)` → strict-utf-8 decode crashed on non-UTF-8 diffs (cp1252 smart-quote byte 0x92; minecolonies + geotools dropped). Patched both to raw-bytes + `errors="replace"` (mirrors scrape_gomod_batch), synced to all 3 hosts. Takes effect per-loop's next repo (fresh factory.py subprocess) — no restart. Dropped repos auto-retry on post-pass re-invoke.
- **Yield diagnostic (~31K rows, 38 repos):** ~17% signal (regression/P2F 3648, fix/F2P 598, new_failing/new_passing 1076), ~29% clean negatives, ~54% env/infra. Top waste = `missing_diff` (~19%) and it's **ACCEPT-dominated** (~2156 acc vs ~548 rej on sk3) → recoverable fetch-gaps (not force-pushed-away), batch-backfillable after first pass. Re-parse corrections likely mostly unneeded: new runners emit rich corrected transitions. Extrapolated 1k → ~140K rows / ~24K signal (ample).

## 2026-06-30 — ctree (metrics_tree_infilling): Codex audit fixes + power measured; tree is a stump
- **Codex audit applied (all 8 bugs verified + fixed):** reliability discount was backwards (`raw/rel`→`raw*rel`); XNOR composites dropped (best_combination now credits inverted polarity, +test); composite rule fit on N/A-as-0 rows (now applicable-only); composite persistence (to_metric was 0.0 placeholder → run_infill returns real ScoreMatrices); GEPA tiered-HEAD misread + missing task-preset + `accepted` reporting in gepa_viable; find_poolable offline early-return; diag_metric_power shared spec. **3-way discover/guard/test split** (final AUC on untouched holdout — Codex headline fix) + **direct guard-AUC acceptance gate** (mode gap_closure/auc/either/both) + loosened viability + proposer graceful degradation + tree-dump in ctree_power.
- **Two prior conclusions corrected:** (a) "GEPA rejects all mutants" was a HEAD-read bug — CW ledger shows **4/10 accepted**; the GEPA→power translation was never measured (seed-vs-HEAD export wired, run deferred). (b) CW is NOT pair-leakage-inflated — duplicated ids are a concat artifact (two halves, label-rate 0.47/0.15), not preference pairs.
- **★ Tree is a stump — config pathology, then real.** Split gate needs `n_perm > m/alpha` (≥999 for m=26 z-covariates); ctree_power had `n_perm=199` → min adj_p 0.13 > 0.05 → **splitting mathematically impossible** → all depth/coverage/placement readouts were fakes. Fixed to 999. Even at 999, CW (n=500, 26 rubrics, glm-5.2) is STILL a stump + `gap_nodes=0` (root guard AUC 0.608 > 0.55) → no infilling → honest **test AUC 0.642 (0.633 no-NA)**. So CW has no MOB-detectable region structure at n=280/Bonferroni (genuine-vs-underpower TBD; root-instability diagnostic now baked into _dump_tree). The n_perm=199 "infilling" (test 0.660→0.641) was a deviance-noise gap + guard-overfit.
- **Backend:** OpenRouter credits exhausted (402); sk3 GPUs 8/8 busy; glm-4.7 PaaS 429. Switched ctree judge+proposer to **glm-5.2 via z.ai anthropic** (subscription-free, concurrency≤2 to avoid rate-limit). Cross-backend: Gemma 0.657 ≈ glm-5.2 0.656 baseline. glm-5.2 ±0.02 AUC non-determinism across materializations.
- **Decision point (paused, not auto-launching):** accept ~0.64 CW ceiling / probe underpower (bonferroni=False or larger n) / try a corpus with region structure / run the GEPA seed-vs-HEAD power comparison (#46, needs GLM).

## 2026-06-30 — norm→metric matching pipeline COMPLETE (26 corpora) + faithfulness audit — PATH MAP

**Pipeline:** GEPA+Gemma extraction (23 corpora, faithful) → Gemma-4 batch labels (26) → per-task Llama-8B cross-encoder (26, parallelized) → base-BGE top-50 → CE top-10 cascade → `matches_ce` per corpus. Qwen LLM-rerank stage **DROPPED** (no lift + Qwen distrusted for eval). **690,998 norms mapped.** Recall@10 (full gold set, base-BGE→+CE): math 0.29→0.55, code_review 0.21→0.44, humor 0.21→0.39, press_releases 0.16→0.39 (CE = workhorse, +82–141%); **creative_writing 0.19→0.23 (+21%, plateau)** — NOT a matching failure: spot-check (workflow, Claude judges) found CE top-10 is 67% sensible, only 7% true misses, 77% of "misses" are gold problems (wrong/incomplete); catalog has R2 over-split near-duplicate metric leaves (a224≡a341 "sentence clarity/diction"); data-insensitive (2× labels → 0.230→0.232). **Domain-bounded:** works on objective, plateaus on taste.

**sk3 root = `/lfs/skampere3/0/alexspan`** (HOME pin for AFS-token safety).

**Per-corpus matching artifacts — `data/bge_pertask/<task>/`** (26 task dirs):
- `signals_<task>.jsonl` — the norms (`{i, s}`); 2,878 for creative_writing, up to 218,496 (litbench)
- `matches_ce_<task>.jsonl` — ★ **FINAL CE-reranked top-10 per norm** (the deliverable; capped 20k for giants)
- `catalog.txt` — R2 metric leaves (`a{id}: name`); 200–394 per task
- `matches_top10_<task>.jsonl` — base-BGE top-10 (covers ALL signals, incl. beyond the 20k CE cap)
- `bge_train_<task>.jsonl` — Gemma-4-labeled triplets (anchor/positive/negative)
- `cross_encoder_llama8b/` — trained CE (26/26)
- `matches_<task>.json` — GOLD (signal→3 aspects); **5 tasks only**: code_review, creative_writing, humor, math, press_releases

**GEPA+Gemma extraction — `data/<corpus>/gepa/anchors_best_full.jsonl`** (23 corpora; full self-contained signal+passage, the faithfulness-audit source). Variants: `humor/standup_multi/` = humor_multi, `creative_writing/wp_comments/` = wp_comments (+ CW base). Canonicalized 7 round-named → `anchors_best_full.jsonl` this session.

**Scripts:**
- `scripts/llama_norm_extraction/gepa_pr.py` — GEPA+Gemma runner (`gen_corpus`/`judge_corpus`/`run` subcommands; ★ GEPA_CORPUS env trap; GLM-5.2 judge via z.ai anthropic)
- `scripts/llama_norm_extraction/glm_audit.py` — GLM-5.2 faithfulness audit (full-source, 2-link: signal→passage + passage→source)
- `scripts/llama_norm_extraction/label_gemma_batch.py` — **offline Gemma-4 vLLM batch labeler** (`--gemma-for-gold` forces Gemma path for gold tasks; reuses label_pairs.py)
- `scripts/llama_norm_extraction/label_pairs.py` — labeler lib (server-client vllm + GLM modes)
- `data/bge_pertask/train_cross_encoder.py <task>` — per-task CE trainer (Llama-3.1-8B-Instruct)
- `data/bge_pertask/train_ce_parallel.py` — autonomous GPU-parallel/stacked CE-training orchestrator
- `data/bge_pertask/match_cascade.py <task> [maxsig]` — base-BGE top-50 → CE top-10 + base-BGE-vs-CE recall vs gold
- `data/bge_pertask/cascade_sweep.py` — autonomous per-task cascade orchestrator (all 26, dynamic util)

**Audit outputs:**
- `data/_glm_faithfulness_audit/` — GLM-5.2 audit (`verdicts.jsonl` 138 samples, `summary.json`): 0 fabricated passages across 23 corpora
- **laptop** `/tmp/codex_norm_audit/codex_audit_full/verdicts/` — Codex 1,150-triple audit: `signal_in_passage` varied/genuine, `passage_in_source` defaulted 100% GROUNDED (discard that axis, use GLM's)

**R2/R3 metric hierarchy:**
- `norm-research/outputs/hierarchy/<task>_general_r3_expanded.json` — R2→R3 map (`merged_groups[].source_r2_cluster_names`); roll `matches_ce` up to R3 (70 nodes) / grandparent (42) via this, anytime
- Formal level defs: `notes/2026-05-14__metric-taxonomy-and-two-axis-setup.md` (Leaf → Cluster/R1-child → R1 parent → R1-refined → **R2 merged_group [= metric M_i, what matcher uses]** → R3 → grandparent)

**Models (`shared_hf_cache`):** Qwen3.5-122B-A10B-FP8 (judge; needs `VLLM_USE_FLASHINFER_MOE_FP8=0`), **gemma-4-31b-it** (extraction + labeling; gemma4 env, vLLM 0.23, dynamic gpu_mem_util), bge-large-en-v1.5 (Stage-1 retriever fallback), Llama-3.1-8B-Instruct (CE base). **GLM-5.2** via z.ai anthropic subscription API (keys `/lfs/skampere3/0/alexspan/.z-ai-api-key*.txt`, rotate on 429; monthly quota — be sparing).

**Recall / logs:** `match_cascade.py` prints base-BGE-only vs CE recall@1/3/5/10 vs gold (gold tasks) to per-task logs `/tmp/cascade_logs/<task>.log`, `/tmp/ce_train_logs/<task>.log`.

**Open (later):** re-cluster R2 (looser τ, was 0.92) to merge near-dup metric families — clean fix for CW + litbench dup-name top-5s.

## 2026-07-01 — VAT notebook + top-V re-score

Built `notebooks/2026-07-01__patents-laws-VA-decomposition.ipynb` (+ `_executed`): V/A ladder
(table + barplot), top-V-vs-top-A features as **HTML cards** (blue V / orange A, strength bars,
protective badges), and numerical-V bonus histograms for patents. Self-contained via
`notebooks/data/{legal_va.json, patents_va_features.csv}`.

Two follow-ups from user review:
1. **Top-V cards were empty** for title_vii/flsa/ss_disability — because the saved `top_metrics`
   held only the top-15 by |AUC−0.5|, which are all thick(A). The thin metrics WERE scored, their
   AUCs just weren't persisted. Re-scoring via `/tmp/run_vat_fullfeat.py` (patched run_vat.py that
   saves `all_metrics` = every metric's AUC + `thin` flag) → `vat_fullfeat_<domain>.json`, 5 domains
   (title_vii, flsa, ss_disability, dol, cavc_review), GPU2, sample 600. PID 695697, watcher bk4isompo.
2. **`overtime_violation_hours_over_40` tagged A not V** — CORRECT: it's the doctrinal-conclusion
   metric ("did an OT *violation* occur", needs exemption/lawfulness judgment). The countable-fact
   sibling `weekly_hours_exceed_40` IS a separate thin(V) metric. Naming made the thick one look thin.

FLSA doctrine bank = 51 metrics, 28 flagged thin. Thin flag lives ON the bank metric JSON
(`thin`/`checkable`/`verifiable`), separate from THIN_METRIC_REGISTRY.json.

### 2026-07-01 (cont.) — top-V re-score COMPLETE + raw matrices cached

All 5 domains re-scored with full persistence. `datasets/legal-outcome-prediction/vat_score_cache/`
now holds `<domain>.npz` (raw n_cases×n_metrics score matrix X + y + metric_names + thin_flags) +
`<domain>_manifest.json` (every metric's AUC + thin flag) for title_vii, flsa, ss_disability, dol,
cavc_review. NO future V/A cut ever needs a GPU re-score again. Notebook `legal_va.json` rebuilt from
manifests; V/A cards now populated. `run_vat_fullfeat.py` is the canonical persisting scorer.

REAL top-V vs top-A (thin-flag from doctrine bank):
- title_vii: V weak (top tenure_years 0.434 protective, all <0.47); A carries it (legit_reason 0.386, documentation 0.566)
- flsa: V≈A both strong — V filing_within_limitations_window 0.579, effective_hourly_rate_below_minimum 0.567,
  overtime_pleading_specificity 0.565; A overtime_violation 0.589. THIN checkable facts nearly as predictive as doctrine.
- ss_disability: V near-chance (onset≤DLI 0.472, all <0.48); A duty_to_develop_record 0.598 — pure articulable/judgment domain.
- dol: V very strong — coal_mine_employment_years_ge_10 0.612, filing-date gates 0.60; A scope_of_review 0.623. V≈A (thin=decisive).
- cavc_review: both protective (thoroughness→affirmance); V evidence_items_discussed_count 0.40.

Pattern confirms VAT theory: A>V gap large where top-V is near-chance (ss_disability, title_vii = judgment domains);
A≈V where thin metrics carry real signal (flsa, dol, cavc = rule/fact-heavy). overtime_violation is A (doctrinal
conclusion); weekly_hours_exceed_40 / overtime_pleading_specificity are the V siblings.

## 2026-07-01 — ctree z-design correction + metric-count certificate theory doc

**Known-moderator positive control OVERTURNS half the CW stump verdict.** `litbench-to-train.csv.gz`
is a concat of two halves with label rates 0.44/0.18 — a real moderator the tree had never been shown.
- Bank-wide z (shipped default: 26 rubric levels + NA indicators, m_z=48): `source_half` ranks 7th,
  raw p=0.003, **adj_p=0.144 — invisible purely from the Bonferroni multiplicity tax** (perm floor at
  B=999 is already adj_p=0.048; degenerate NA-indicator stats eat the top ranks). Stump.
- Curated z = {source_half, text_cluster} (m_z=2, metrics stay in X): **root splits on source_half
  (adj_p 0.014), then depth-2 on text_cluster** inside the low-rate half. Real MOB structure on real CW.
- The bank does NOT absorb the axis: label gap A−B +0.26 raw → +0.21 after residualizing on all 26
  metric levels — genuine unexplained region structure.
- Lesson: metrics belong in X; z must be a small curated set of ITEM-level axes (source/genre/topic/
  length). Offering the whole bank as z is the partykit anti-pattern (~50× multiplicity tax).
- What survives: no rubric-level moderation (relationships homogeneous in rubric-covariate space);
  honest bank ceiling ~0.64 stands unless per-leaf models beat global on holdout
  (`diag_cw_treeholdout.py` running — fresh 500-item judge materialization, glm-5.2).
- Scripts: `diag_cw_knownmod.py`, `diag_cw_curatedz.py`, `diag_cw_treeholdout.py`.

**Theory doc written: `notes/2026-07-01__metric-count-certificates.md`.** Bank-level, WITH labels
(complement to prompt-optimality's per-metric no-label theory). Headlines: (1) with Y the bank
objective I(Y;M_S) is monotone → Minoux/U₂ stopping certificate applies raw; (2) assumption-free
ceiling upgrades from vacuous cap_f to task-intrinsic I(Y;X) via dense-stack wrap (dominance-gated;
CW FAILS the gate today — dense still climbing → A* right-censored); (3) count certificates:
N_lower = V_bits/max_j T_j (assumption-free DPI), N_upper = |S_g| + (U−V)/δ (wrap + quotient +
per-metric optimality); (4) residual trichotomy — MOB detects moderation-shaped residual only,
gap-nodes detect region-shaped, dense-residual probe detects uniform; stump ≠ saturation; (5)
mega-metric degeneracy → counts well-posed only over the articulable class (R≈T + atomic quotient).

**Holdout verdict (diag_cw_treeholdout.py, fresh 500 items, glm-5.2):** the curated-z tree
GENERALIZES. Per-leaf label rates transfer (0.44→0.50, 0.22→0.16, 0.14→0.14, 0.20→0.19).
- global bank-GLM holdout AUC **0.588** (26 metrics + NA indicators; fit on discover)
- tree-routed per-leaf GLMs **0.706** (+0.118)
- controls: axes-only (source_half+text_cluster one-hot, no bank) **0.690**;
  global+axes additive **0.679**; ⇒ **moderation-specific gain +0.027** beyond main effects.
Reading: (1) most of the +0.118 is the AXES' main effects — and source_half is partly an
identity/mixture confound (two concatenated datasets, different labeling processes), NOT an
articulable metric — analogous to publisher-id in press releases; (2) the honest new-signal
reads are text_cluster's contribution and the +0.027 moderation term; (3) the bank alone
generalizing at 0.588 (vs 0.64 on the earlier same-sample split) says the 26-rubric bank is
weaker out-of-sample than the within-sample read suggested. Next: run infilling with
curated_z_only=True and the WRONG/RIGHT contrast inside the B-half leaves — that's where the
localized deficit is; deconfound source_half (treat as nuisance, not predictor) before
claiming articulated-power gains.

**Global (tree-free) infilling module added:** `methods/metrics_tree_infilling/global_infill.py`
— boosting-style loop: global bank GLM → global WRONG/RIGHT contrast → propose corpus-wide
metric → accept iff guard-AUC gain ≥ min_auc_gain; per-metric MetricLedger tracks
(1) data-to-develop (n_proposal_examples + 25/50/100% data_curve + min_train_frac),
(2) applicability on discover+guard, (3) reconstruction R (rederive rubric from (text,verdict),
re-execute, balanced agreement + AUC; `reconstructor_fn` = GEPA plug-point). Offline oracle
tests: planted "zephyr" metric accepted (+0.20 AUC), ledger complete, reconstruction
round-trips; dud proposals stop via patience. 20/20 tests pass.
Also: `methods/metrics_tree_infilling/AGENT_PLAYBOOK.md` — phased protocol (data audit →
judge audit → verification gates G1-G4 → discovery → honest read-out) for applying
ctree+global infilling to new domains, with per-domain hazard table.

## 2026-07-02 (overnight) — theory implemented + baseline arms + generation runs

Built tonight (all offline-tested, 27/27 module tests pass):
1. **Bits readout** — `_bank_eval` returns (AUC, V_bits = guard log-loss reduction vs base rate,
   bits/item); MetricLedger + trajectories carry both; `min_bits_gain` config gate (bits is the
   certificate currency; AUC gate retained).
2. **`certificates.py`** — MCC theory as code: `dense_stack_wrap` (dominance-gated, else
   right-censored), `flux_wrap` (process-relative), `count_certificates` (N_lower=ceil(V/log2K)
   free; T_max leg REFUSED under shared-call scoring; N_upper=|S_g|+(U−V)/δ; Minoux tail),
   `report_from_ledger` (prints honesty notes: right-censoring, single-arm anti-conservatism).
3. **`generators.py`** — three proposal arms through the SAME gate: residual (ours),
   unconditional (autorubric-style, no data), label_contrast (raw-label examples, isolates the
   value of residual targeting). `proposal_fn` hook in run_global_infill; ledger tags arm.
4. **Medoid banks** — coverage-selected 40-rubric banks replacing head-of-file limit=40:
   `datasets/creative-writing/medoid-bank/bank.json` (73,702 pool),
   `datasets/peer-review/medoid-bank/bank.json` (75,649 pool);
   builders `scripts/tools/build_cw_medoid_bank.py`, `build_medoid_bank.py`.
5. **Arm-comparison runs LAUNCHED** — `scripts/tools/run_arm_comparison.py` via
   `overnight_arm_runs.sh` (driver pid 78304): CW (medoid bank) then peer-review (medoid bank),
   n=400+split, 3 arms × 4 rounds, gates min_auc_gain=0.02/min_bits_gain=0.01, glm-5.2 conc 2.
   Logs: outputs/ctree/arm_comparison/{cw,pr}.log; summary.json + per-arm certificate.json.
   Smoke test passed (gate correctly rejected a no-gain proposal; certificate right-censored).

**First arm-comparison results (guard-gated) + gate recalibration.** Both tasks, all 3 arms,
0/12 proposals accepted. Ledger autopsy: proposals were viable (applicability 0.36-1.0) and
non-redundant (R² 0.02-0.51), but gains were tiny (dAUC -0.03..+0.008; dbits -0.019..+0.010).
Best single proposal: residual arm's "Manual Markdown Formatting" on peer-review (+0.0102
bits — above the 0.01 bits gate but killed by the 0.02 AUC gate).
DIAGNOSIS: gate was below noise — Hanley-McNeil SE of guard AUC at n_guard=84 is ~0.06-0.07,
so a 0.02 AUC gate on one split is a coin flip; nothing real at this effect size could pass
reliably. FIX: added `acceptance_eval="cv"` (paired 5-fold CV over pooled discover+guard;
per-fold pairing cancels split noise) + gate 0.01 AUC / 0.005 bits; final read still on
untouched test. CV reruns launched (driver 97748, outputs .../arm_comparison/{cw,pr}-cv).
Also noted: CW medoid bank viability only 14/40 (vs head-of-file 26/40) — the coverage bank
includes poetry/publisher/virtue rubrics inapplicable to WritingPrompts stories; coverage of
the rubric POOL ≠ applicability to the item DISTRIBUTION (playbook Phase 1 refinement).
Substantive interim read: with a calibrated instrument, per-metric marginal gains on both
tasks are ≤0.01 bits — consistent with the "many small-v species" flux-tail picture (MCC §5);
no arm found a big missing criterion, including the unconditional baseline.

**CV-gated rerun results (12 proposals CW, 8 PR): FIRST ACCEPTED METRIC.**
- CW: 0/6 kept across all arms (best: unconditional "Economy of Prose" +0.0056 bits, under the
  0.01 AUC gate). Consistent verdict at a calibrated gate: no single articulable criterion
  worth ≥0.005 bits is being found for CW by ANY arm — the flux tail is genuinely thin/flat.
- PR: 1/8 kept — label_contrast arm, **"Explicit Analytical or Mathematical Contribution"**
  (YES if text claims proof/theorem/convergence bound/formal analysis): CV dAUC +0.018,
  dbits +0.0065, applicability 0.98, reconstruction R = 0.952 agreement (near-perfectly
  articulable — a reconstructor rederived it from verdicts alone). Face-valid: theory papers
  fare differently at ICLR than empirical-only ones.
- Caveats logged: its data_curve is unstable (negative at frac 0.5/1.0 — the single-guard-split
  readout inside the curve is the noisy leg; curve should move to CV too);
  test-split validation launched (validate_kept_metric.py) for the honest final read.
- Arm scoreboard so far (accepted-bits-per-proposal): label_contrast 1/8, residual 0/8,
  unconditional 0/8 — no evidence residual targeting beats naive label contrast at this scale;
  the one hit came from the naive baseline. Small-n; do not over-read.

**Test-split verdict on the one kept metric: DOES NOT SURVIVE.** "Explicit Analytical or
Mathematical Contribution" (PR, label_contrast arm) on the untouched test (n=180):
bank 0.621 AUC / 0.0278 bits -> bank+metric 0.606 / 0.0154 (delta -0.016 AUC, -0.012 bits).
Delta is within ±1 SE (test AUC SE ≈ 0.042 at n=180), so the honest read is "no detectable
test gain," not "harmful" — but the CV +0.018 did NOT replicate. Winner's curse: 20 gated
proposals, the max cleared the gate, test regressed to ~0. FINAL SCOREBOARD for the whole
comparison: 20 proposals (3 arms × 2 tasks), 0 metrics survive the full 3-stage protocol.
Substantive conclusion (stronger than any single run): at n≈420/metric-bank≈27, NO arm can
find a single new articulable criterion with detectable held-out label-signal on either task
— per-criterion signal is < the ~0.01-bit detection floor of this design. To detect the thin
tail, need: bigger n (test SE scales 1/sqrt n), aggregate acceptance (accept SETS of small
metrics jointly), or per-metric T_j certificates instead of bank-marginal gains.
Winner's-curse control to add: CV-select -> fresh-seed CV confirm BEFORE test (cheap), or
Bonferroni the gate by #proposals per round.

**2026-07-02 PM — MCC theory doc hardened (user ask: generalizable / concrete / PO tie-ins).**
`notes/2026-07-01__metric-count-certificates.md` updated: **§2a generator abstraction** — any
metric-generating algorithm is an arm 𝒢:(D_discover,S,aux)→candidates under contract G-1..G-4
(split hygiene / materializability / post-hoc uniform membership / ledger); certificate VALIDITY
is generator-independent, COMPLETENESS is generator-relative (each arm = capture-recapture list;
|G|≥2 mandatory; certified artifact = UNION ledger). **§4.4** gate-validity conditions (δ > 
instrument noise floor; winner's-curse deflation) promoted from run-lessons to theory.
**§4.5** eight-step certification chain (object→statistic→assumptions→artifact); bank cert =
conjunction over PO per-metric leaves. **§10** explicit PO↔MCC crosswalk table. **§11** empirical
status (20/0 as worked example). Scorecard +C9/C10.
Code: execution review found per-arm certificates ALWAYS fired the single-arm honesty note (no
union artifact existed) → `report_from_ledgers` added (floor = max over per-arm banks, not sum;
notes for union/floor-stage); wired into run_arm_comparison (auto `certificate_union.json`);
regenerated for all 4 finished runs (CW 0.0937 bits floor, PR 0.0521, both right-censored).
28/28 tests. Playbook Phase-3 updated to point at the union artifact.

**2026-07-03 — press_releases: label-threshold k flips floor→showcase; PROGRESS report written.**
Completed the press-release V/A/dense battery + infilling + GEPA. HEADLINE: the original `judgement`
(≥1 of 17 outlets) was 88% single-outlet noise (28,735 of 32,789 covered PRs = exactly 1 outlet), and
every method capped at ~0.55–0.58 grouped (V=A=dense≈relational; no gap; MCC certificate stump /
right-censored). Thresholding to **k≥3 outlets** (consensus/broad coverage, 1,478 pos on the
deconfounded 72k) lifts the dense ceiling 0.584→**0.705** (within-topic too → not confound) and
produces a clean articulability ladder: **V(cheap)=0.628 < A(40 rubrics,70B)=0.648 < dense(bge-m3)=0.705**
(within-topic 0.627/0.645/0.705). So press_releases FLIPS from floor to **showcase** at k≥3; the
k=1 "no signal" null was label artifact. Top A-rubrics = PR-craft (boilerplate 0.585, CTA 0.557,
ESG 0.546, lede/5Ws 0.544, research-as-newsworthy 0.534). Stage-2 GEPA POC works end-to-end:
Gemma-4-32B judge (served, gemma4 env) + GLM-5.2 proposer (z.ai `zai_anthropic`) via
`make_roles_mixed`, fidelity_scalar (recon-R) objective; 6 viable×2 rounds, 1/6 accepted ("Lede uses
concrete details" +0.11 fid). **Decision: model at k≥3.** Caveats: A-layer NA rate 65% (imputed);
GEPA is POC-scale (head-of-file, 2 rounds, quota). Consolidated reference + next steps (re-run
infilling at k≥3; scale GEPA; MCC cert at k≥3) written to **`datasets/press-releases/PROGRESS.md`**;
full chronology in `notes/2026-06-25__press-release-audit.md` §5a–§5n.

**2026-07-04 — Metric-seam: humor fleet complete, 4-task CAM frontier.** 31/31 blind improvers
+ held-out gates: 4/31 certified (representation ethics +.76, platform standards +.66,
translatability −.18→+.61, topical anchoring −.10→+.61); craft core (timing, storytelling,
SSTH/GTVH) stalls or regresses. CAM humor .120→.351 = lowest of 4 (PR .697 > CW .466 > math
.377 > humor .351); humor's articulable mass is the compliance/framing shell, not the comedy.
Full detail notes/2026-07-01__metric-seam-pilot-results.md §R7.2b; money figure now 4 tasks.

## 2026-07-05 (overnight autonomous) — CUF Tier-1 census launched; Tier-2 built; 5-task silver table
- **Silver 5-task table LIVE** (notebooks/2026-07-04__mi-vs-silver-crosstask.ipynb re-executed):
  code_review is the STRONGEST signal yet (CE +0.304/partial +0.178; GOLD +0.467/partial **+0.315**)
  but with a coverage caveat (51% of silver mass on unscored metrics — 394-catalog vs 133-R2-group gap);
  math = null (+0.04/−0.04, gold +0.08). Pattern so far: code ++, humor +, CW gold-only, math 0, PR 0.
  CW-R2 (368) still running → last row pending.
- **CUF**: redundancy taxonomy resolved (4 causes; merge ONLY on fingerprint identity; g24 GEPA host =
  substitutability, max within-host ρ=.52, duplication ruled out). δ_min materiality gate implemented
  (70B placebo case). Tier-1 metric-bank census RUNNING (GPU4: ~1,100 description hosts across
  humor/math/code-review/press-releases/creative-writing + pilot repairs). Tier-2 BUILT+TESTED (15/15):
  solo/LOO company bracket + within-host species merge (identity from SOLO fingerprints — full
  substitutability with distinct fingerprints is impossible, so solo-identity separates all 4 cases);
  validation chain armed behind the bank (g24/g29 @8B --company-profile).
- Everything detached on sk3; local watcher fires on completion. Theory: notes/2026-07-04__unit-
  certification-theory.md; roadmap: notes/2026-07-04__methods-vs-gepa-crosstask-roadmap.md.

**2026-07-05 — Framing: four functions of judgment-language + paper macro-structure.** New theory
note `notes/2026-07-05__four-functions-of-language.md`: explanation / prediction / formation /
regulation, disentangled by direction-of-fit (Anscombe/Searle) + constitutive-vs-regulative; three
couplings that merge them (Nisbett-Wilson confabulation, Bourdieu/Hacking interconversion,
Holmes-vs-Hart collapse in law); KEY law-vs-taste inversion (law's Y constituted by category-4
language → predictive failure = indeterminacy, not tacitness). Each function has a distinct
instrument in our stack: R / V / decompression grid / executor seat. Paper macro-outline discussed
(I metrics+codability, II applying to text, III other gaps, IV residual across corpora×y-types);
gap analysis captured in memory (project_paper_macro_structure.md): noise-ceiling decomposition of
the residual, comparative-interpretation section, reconciliation-ledger waterfall as money figure,
Δ_synergy + E-staircase + positive-control experiments, y-variable stated-vs-revealed contrast.

**2026-07-05 — Transport test (interpreter swap) + R5 draft.** Llama-70B re-extracted all
56,750 hybrid field prompts; frozen gates re-run. Both retrieval-theory predictions confirmed
(pooled n=120): certificate loss tracks borrowed-meaning weight (ρ=.59) and median transport
ratio .30 (~70% of field signal is shared culture). 3/12 PR certified gates fail under swap
(worst: humble-spokesperson-tone, fully extractor-bound); 8/120 improve. Certificates now
stamped with field-extractor family + transport_ratio. Paper section drafted:
notes/2026-07-05__seam-paper-section-draft.md (C1–C7, all slots filled).

**2026-07-05 — Wave-2 isomorphism expansion (all remaining domains) + calibration + notebook.**
Grids/certs for code-review, peer-review, legal, grant (sweep→cert→grid chains, 2 GPUs, same
GLM-4.7 families; two pre-sweep framing-bug fixes: peer=PAPER, code-review=PULL_REQUEST).
Screen calibration: 3 mid-z pairs judged (612 cands) = 0 SAME, RELATED-rate z-monotone for 3
judges independently; blinded anchor set caught a degenerate resumed-judge pass (new standing
protocol: anchor-test every annotation batch). 8-domain articulation curves: definition-peak
6/8; math −.055 AND legal −.058 invert to name-peak (lexicalization gradient replicated).
9-domain band+H_M verdicts: genuine DEEP exclusively institutional/technical (34: PR 10, cr 7,
legal 5...), genuine COD mostly expressive (CW 2, humor 7, grant 2). New transports honest-null
(legal×math 6/6 but p=1.0, marginals degenerate — rigor family is one-type; grant×peer = next
powered test). Results notebook: notebooks/2026-07-05__crosstask-isomorphism-results.ipynb
(6 figures, executes from artifacts). In flight: grant-grid repair, Qwen reader panel (CW+math),
gemma-2 ladder downloads. Note: notes/2026-07-05__wave2-isomorphism-expansion.md.

**2026-07-05 — Seam position, retrieval-proof battery, codability priors (pre-registered).**
New theory/design note: notes/2026-07-05__seam-position-retrieval-and-codability-priors.md.
(1) Metric program = A∘T∘R with each stage coded or LLM-prompted; whole current fleet is LCC
(LLM at the read stage). "Middle-section LLM-prompted" (CLC) = borrowed judgment through a
coded aperture — F⊥X|ν(X), the level-matching theorem turned inward; three loci of borrowed
meaning: perception (R) / conceptualization (T) / valuation (A). SEAM-POS pilot designed for
PR's 12 certified criteria. (2) Retrieval-thesis proof battery E1–E8 vs H_spec/H_idio/H_leak
(key-deprivation, stipulation-override, 3B/8B/70B staircase, base-vs-instruct, aperture, Qwen
third family [running], selected-vs-enculturated [needs sign-off], articulation). Three-scout
lit sweep in note §2.4: adopt task-recognition/task-learning vocabulary (Pan et al. 2023);
E1/E2 have word-level precedents (WinoDict; Stroop lexical-override 2606.07555; MAGNIFICo) —
construct-level in frozen certified programs is open; PromptBridge = nearest transport
neighbor (treats drift as deficit-to-fix; we measure the un-bridged shared/bound split).
(3) CODA codability-priors probe pre-registered (F1–F8 + zero-shot guess, blinded anchors,
LOTO-only) over ~140 criteria with own-judge floors as outcomes; probe running. Ops: Qwen
extraction mid-run (v2+CW done); legal field extraction queued behind it on GPU 7.

**2026-07-05 — Tacit scaling + cross-family enculturation + a-priori prediction (3 new directions, #21-23).**
Code methods/codability/{name_sufficiency,grid_auc_report}.py; note
notes/2026-07-05__tacit-scaling-enculturation-apriori.md. INSTRUMENT: report.json bal_acc
thresholds scores at absolute 0.5 → conflates reader calibration with tacitness (Qwen2.5-3B on
math: exact-0.5 on all 21 metrics, healthy AUC .649); all analyses moved to threshold-free AUC
vs executor M_i. (1) SCALING: name-sufficient = name-AUC≥.55 AND def−name≤ε, S* with up-ladder
persistence. Taste/craft/mech dissociate: TASTE names come online 1B→3B (deficient .63→.33),
CRAFT flat ~.61 through 8B, MECHANICAL never (.72 at self) — codified≠lexicalized. Gradient
replicates on transmission scale (peer +.044 … math −.023, legal −.051). 70B ordinal prereg
FROZEN (sha 62e4b3f0…, 92 persist + 136 ranked; evaluate post-Jul-10). (2) ENCULTURATION DiD
(math, byte-identical messages): stranger-minus-kin deficit positive at all tiers, sig at 1B
(+.018, p<1e-4, n=21); taxonomy 17/21 universal-lexicalized + 2 A-only — flagship "Elegance
and beauty of proofs" (TASTE: kin name beats def −.061; stranger needs def +.038) = the
user-predicted "B never taught it" cell. CW-Qwen panel scoring now (expressive pole test);
gemma-2 third family next. (3) A-PRIORI LODO: NULL on AUC scale (tags+class ρ=−.07; tags alone
ANTI-predict −.19 p=.005 — type→tacitness is domain-contextual); the bal_acc-scale ρ=.35 was
calibration structure. Scoped: within-domain eval, split-half reliability ceiling, LM zero-shot
baseline, wave-3 held-outs. ALSO: grant grid repaired+landed → census (n=16) → grant×peer
transport 5/5 label-agree p_perm=.083 (FIRST powered rigor test, 2 first MECHANICAL
transports; pooled 41/46 over 4 pairs) — wave-2 fully closed (#18).
Addendum (same day, CW-Qwen landed): 1B-tier DiD REPLICATES in CW (+.017, p=.0001; math
+.018) — two pole domains, same tier, same direction. Taxonomy splits the poles: math formal
lexicon family-INVARIANT (2 A-only/0 B-only = 11% family-specific) vs CW expressive lexicon
family-VARIANT and BIDIRECTIONAL (5 A-only: pacing-rhythm/flash-compression/opening-hooks…
+ 6 B-only: macro-plot/POV/genre… = 27%) — families carry partially disjoint articulable
inventories: dialects of craft culture. CW 8B-tier flip = self-recovery tier, artifact-prone.
Notebook §8b re-executed with both domains (0 errors). #22 closed; gemma-2 triangulation of
the 11 CW family-specific cells is the #12 continuation.

**2026-07-05 addendum (metric-seam thread) — day-4 harvest, all three landed.** Legal fleet
9/20 certified, CAM .372→.621 = 2nd of 5 fleet tasks (money figure now 5-task; two §R7.2c
survey-zeros RECOVERED by the improver round — survey floors can't distinguish extraction
shortfall from absent evidence). Qwen3.5-122B third family replicates both transport
predictions (pooled median ratio .230 vs Llama .299; the larger extractor loses less on
every task; PR nearly lossless at .026) and confirms E6: criterion-level boundness,
Spearman(ratio_l, ratio_q)=.295 pooled, a87 humble-tone bound under BOTH swaps — after a
thinking-leak incident forced a full re-extraction (Qwen llm.chat default = thinking ON;
signature: fields ≈ blank, median raw len 183 vs 8). CODA codability probe mostly NULL
(LOTO pooled −.18 via between-task offset; zero-shot guess +.18 best prior; within-PR +.45)
→ codability = phrasing × evidence-availability × judge-reliability; criterion text reveals
only factor one — run the pipeline. Notebook §12 added, 50 cells re-executed clean. Queued:
SEAM-POS pilot (#16), E1-KEY/E2-STIP field-level builds (#17). Notes: results note §R7.3 /
§TRANSPORT-3FAM / §CODA; theory+designs in 2026-07-05__seam-position-retrieval-and-codability-priors.md.

## 2026-07-05 (pm) — MCC endpoints push: confirm stage, method arms, U_flux, planted calibration

Closing the four metric-discovery debts identified in the endpoint audit (compare methods /
upper bounds / per-task certificate / rigor):

1. **Confirm stage** (`global_infill.py`): fresh-seed repeated paired-CV + Nadeau-Bengio-corrected
   one-sided t-test + Bonferroni over the PLANNED proposal count (m fixed pre-gate). Kills the
   winner's-curse path that let the 07-02 CV survivor die on test. Ledger carries confirm_* fields;
   certificates flag unconfirmed keeps.
2. **Method comparison made real**: AutoMetrics-Iterative (failure pairs + iteration memory +
   self-critique) and Metric-Tree (partition-conditioned discriminative gap-fill) ported as
   generator ARMS through the same gate — conditioning strategies with the executor held fixed
   (their own backends would confound generator with executor). 5 arms total.
3. **U_flux implemented** (`flux.py`): union ledger → embedding species quotient (tau .92, audit
   .85) → value spectrum → D1-D3 (Good-Toulmin ET-truncated, McDiarmid with T1 B-cap, anytime
   delta) via metric_implementer/experiments/value_certificate.py. report_from_ledgers takes the
   tighter valid wrap; process-relative scope note always attached. --patience 0 (never early-stop)
   so every draw feeds the capture-recapture read.
4. **Planted-bank calibration** (`run_planted_calibration.py`, MCC §7): real CW texts, synthetic y
   from 4 code features, exact V*, one feature withheld as planted-positive, judge-fidelity anchors
   built in. **The anchors caught three real bugs on the first runs**: (i) Llama-70B marked binary
   rubrics applicable=false when the answer was NO (censoring informative negatives as N/A) — fixed
   by a shared _JUDGE_PROMPT_HEADER pinning applicable-vs-fails semantics; (ii) planted truth was
   computed on full text while the judge sees a 2800-char view — truth moved to the judge's view;
   (iii) the killer: three_way_split RESETS indices, so F[df_d.index] truth lookups were silently
   row-misaligned — fidelity ~0.5 regardless of judge quality; truth now travels as _truth_*
   columns through the split. Post-fix fidelity: dialogue .87 / first-person .87 / question .75 —
   the residual gap is genuine executor noise (13 criteria per JSON call), i.e. the attenuation the
   anchors exist to measure. Probe verdict (n=100, 1 round smoke): V recovered 0.033 vs V* 0.098
   (attenuation ~3x, consistent with anchor fidelity), 0 false acceptances.

Runs (sk3, GPU 3, 70B FP8 offline-batch judge, glm-5.2 proposer): fidelity probe → overnight = CW
planted calibration (V* bank = .088 bits; withheld carries .074 ≈ 15x the .005 gate) + CW 5-arm x
30-round scaled flux run (n=600, executor-label llama-3.3-70b-fp8). PR scaled run queued. 49/49
tests local. Open: dense-evidence wiring per corpus (code-review CC/CF already plateaued with bank
>= dense → immediate full-certificate candidates), bank-eval x dense-train contamination check.

---
2026-07-05/06 (metric-seam battery, day 5): FULL E-BATTERY RUN + CODIF. sk3 GPU-1 queue 23
jobs rc=0 (~3.5h): E1/E2 Gemma 58.5k + SEAM-POS 27k + E3 llama-3B/8B 5 tasks + E4 8B
base/instr few-shot 22k. E1 domain-GRADED: PR key-like (name 1.00 vs nonce .58), CW spec-like
(.98≈.98), math INVERTED (.75 vs 1.00 — names underdetermine). E2 flips by extractor: Gemma
complies .68-.96 vs GLM-4.7 snapback .46 on same fields — override is extractor property
(GLM-5.2 like-for-like running). E5: digest keeps .59 median; CCL LLM-agg NEVER beats ridge
(0/12) — borrowed judgment localizes to READ. E3 medians monotone 3B->8B->70B all 4 tasks
(per-criterion noisy .23-.63). E4@8B: base ≈ 0-half of instr; 70B pair pending
(Llama-3.1-70B base download crawling through HF handshake timeouts, supervisor loop on).
CODIF (R11): 143 programs annotated C1-C8 (12 Sonnet batches, anchors .91/1.0 modal agr);
legal codifies with ZERO form-measure; exemplar-match is humor-only (.42); C2 signifier
programs transport 2.8x better (.26 vs .71) = code-side lexicalization echo; thick-predicate
census: CW/humor overflow=aesthetic-stance, math overflow=epistemic. METHOD: shared outdir
unblinded anchors (8/12 copied/harmonized, self-reported; excluded) + NaN-poisoned sorted()
medians (caught: "median">max). E7 SEL respec'd label-clean (distill-the-field); awaiting
sign-off. Notes: results note §BATTERY-FULL/§CODIF; seam note §5 CODIF scheme + §6 E7 spec.

## 2026-07-06 — snap-back / frame-level commitment / substitution economics
Results notebook: `notebooks/2026-07-06__snapback-frame-commitment-substitution.ipynb` (9 cells, 2 figs).
E2 name-commitment (nonce-controlled stipulation): humor +.032 p<.001 probe-clustered ×3 strong readers ×2
families; CW/math ≈0 — commitment is FRAME-level (domain package), not name-level (thickness/safety/practice/
crisp all null; user's safety hypothesis + our thickness hypothesis both disconfirmed). LOAD-BEARING per user
sign-off. Money line: **the big-model advantage is mostly not purchasable with articulation — which is precisely
the frame-level tacit content E2 measures** (gemma-2 pair: math 78% full parity at 4.5× smaller vs humor 32%).
Scale floor = cliff not slope: 3B-bound ~1%; 1B-bound 11% pooled, 31–38% institutional (codified≠light).
Spec-HURTS 10–15% math/legal/news. In flight: 6-domain E2 (prereg: vernacular-vs-institutional), stip_swap
semantic generalization, T2a/T4/T5b/T6a/T8c battery, gemma-4 scoring, deep-research prior-art sweep.

## 2026-07-10 — cross-task GLM step-down: falling limb = divergence-toward-truth (math planted proof)

First cross-task test of the humor frontier step-down (glm-4.7→5.2 = −.18 crowd-agreement), against each
task's frozen LOCAL_MID crowd (14 mid-scale executors, hard-binarized majority; `outputs/osl_multi/
adjudicate_crosstask.py`, results `crosstask_glm_stepdown.json`):

| task | glm-4.7 | glm-5.2 | Δ | planted truth-acc 4.7→5.2 | crowd truth-acc |
|---|---|---|---|---|---|
| creative_writing (36m) | .762 | .788 | **+.027** | .913→.913 (agreement, flat) | high |
| math (18m) | .825 | .765 | **−.060** | **.743→.775 (RISES)** | **.678** |
| peer_review | — | — | pending | 4.7 battery completing (12/19 done) | — |

**Money finding (math):** glm-5.2's planted crowd-agreement fell −.093 while its planted TRUTH-accuracy
rose +.032 — the crowd is only .678-competent at mechanicals on math (quote .57), so the frontier
disagrees where the crowd errs. Direct within-task evidence that the falling limb is
divergence-toward-truth, not degradation. Sharpest: PLANTED-quote truth-acc .343 (glm-4.7, a real
execution failure) → .690 (glm-5.2). Pattern across tasks tracks crowd competence: humor crowd ~.90 →
−.18 step; math crowd .68 → step; CW crowd .91 & both rungs match it → no step. The bistable threshold is
task-relative (where the crowd's competence sits), not a universal capability cliff.

Instrument notes: N&C V3 battery validated — 11/11 panels, planted AUC llama70b .966/qwen25-72b .964,
planted ladders MONOTONE in both families (llama .52/.77/.95/.97; qwen .82/.89/.93/.96). gemma2-27b is
form-fragile on N&C (medrel .494, 65 fragile metrics, planted .656) — charge ε_form, don't gate; it stays
in the crowd with a flag. Fixed: ncv3_checks planted lookup (@v3k suffix mismatch → all-NaN planted AUC);
zxa/mbarglm are different artifacts (60 z×a arm-entries vs 19-metric battery) — don't conflate.

Ops: PR silver extraction 364/546 (heavy-review region, ~4.5× text/shard; ETA ~evening), lock-coordinated
waiters live (prdef v3 8b→70b same-GPU + cascade v2 + scale_extract priority; GPU0 permanently off-limits
— co-resident job oscillates 65↔128G). v4.2_training done (138,262 rows). Qwen3 ladder closed (#138):
qwen3-max landed; frontier points complete {hermes405b, qwen3-max, kimi-k25, dsv3}.

## 2026-07-11 (eve) — Tacit line audit; unit-count grid; 70B prereg pass finally running

Audit of the tacit-scaling line (note 2026-07-11__tacit-line-audit-and-unit-count-grid.md): the
frozen 70B prereg (62e4b3f0) was overdue with NO eval code path anywhere; a 70B cw/humor pass had
existed since Jul-2 (predates the freeze → those 2 domains contaminated, eval scope = 7 clean
domains); gemma-4-31b humor/math grids sat unharvested since Jul-6 (harvested tonight; math
gemma-4 looks SUSPECT-weak, name .630 < gemma-2-2b). Built eval_prereg_70b.py + launched the
70B-FP8 score pass over the 7 clean domains (sk3 GPU4 chain, one engine/process).

NEW instrument: unit-count grid (unit_count_grid.py) — rungs u0=name, uk=name+first-k CUF
leaf-units of the verbal dossier, fk=length-matched filler control; same probes/refs/AUC as rung
grids. Interim (Llama ladder): humor 3B→1B costs ~+1.3 units/metric (rising with the bar; 18/57
metrics unreachable at .65 for 1B at ANY unit count — enculturation ceiling), math ~0 with
content-minus-filler ≈ 0 (the name IS the unit; extra articulation sometimes hurts small readers).
Math 70B rung pass landed: name-peak inversion deepens at 70B (name .707 / def .642). Δk analysis:
unit_deficit_report.py → notebooks/data/two_faces_20260702/unit_deficit_report.json.

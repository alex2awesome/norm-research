# Battery W1a v0 readout — exclusion, negation, composition, holistic (humor slice)

Date: 2026-07-23. Status: EXPLORATORY (v0 references; the 72B target pass is still queued).
Provenance: grids `outputs/tacit_channels/battery_w1/{qwen25_7b_base, qwen25_7b_real_n8192c,
qwen25_14b_base}/grid_humor_w1variants_*.npz` (871/363/871 rows × 400 items); both acceptance
gates PASSED (runner tf reproduces the frozen readout, 7B and 14B); tally =
`battery/tally_w1a.py` → `outputs/tacit_channels/battery_w1/tally_w1a_v0.json`. Adapter rows
analyzed on item-half-2. Declarations: battery-prereg W1a addendum.

## Headline table (medians; leak_self = ρ(exclusion, own tf), +1 = policy unmoved by the
## suppress instruction, −1 = full instructed inversion; not_gap = tf ρ − ρ(negated, −target))

| config | tf ρ (A / B2 / other) | leak_self (A / B2 / other) | not_gap (A / B2 / other) |
|---|---|---|---|
| 7B base | .58 / .73 / .56 | +.35 / +.38 / +.38 | .91 / 1.22 / .89 |
| 7B + real adapter | .69 / — / .62 | **+.93 / — / +.93** | **1.41 / — / 1.30** |
| 14B base | .63 / .75 / .62 | **−.18 / −.26 / −.26** | .58 / .33 / .53 |

## Finding 1 — instructed-inversion gradient (the strong result)

Under "judge AGAINST the criterion" (exclusion) and "the criterion is the ABSENCE of"
(negation), the three configs form one monotone gradient of counter-instruction control:

**14B base (can invert: leak −.2; ρ(neg,−t) up to +.42 on its best-understood B2 cells)
→ 7B base (partial: leak +.35; under NOT still tracks the POSITIVE policy +.33)
→ 7B+adapter (rigid: leak +.93; under NOT tracks the positive policy at +.72).**

The weight-installed configuration is nearly UNMOVED by counter-instruction — behavioral
automatism in the process-dissociation sense. BUT the selectivity test FAILS: the adapter
model is equally rigid on its non-trained B1 cells (+.93 = A's +.93), so this is NOT
selective protection of installed constructs — it is **GLOBAL instruction-rigidity of the
adapted model**. Two live readings: (a) narrow-format LoRA SFT damages instruction-following
generally (format collapse); (b) the coherence-installed general judgment factor (cf. the
v1c B4 result) is itself involuntary. Discriminating test (→ W1c): a trivial-inversion
g-control (invert on simple non-installed criteria) — if the adapter model inverts fine on
trivials, (b) survives; if not, (a).

Knowing-using corollary (2607.08393 frame): explicit logical USE of a policy (NOT) is
capacity-graded (14B ≫ 7B) and is made WORSE by weight-installation — installed ≠ usable.

## Finding 2 — composition v0: "symbols compose, practice blends" NOT supported

Retention (composed ρ ÷ member-mean tf ρ) at matched member levels: adapter A×A .645/.693
≈ .93; base non-A .501/.561 ≈ .89; base A×A ≈ .89; 14B A×A .627/.646 ≈ .97. Weight-side
pairs compose AS WELL AS articulation-side pairs (P-W1a-3's frozen direction is not
supported at v0). Caveats: crude min-z-blend reference (v1 = target's own composed vectors),
n=8 A×A pairs, coarse member matching.

## Finding 3 — holistic instrument DEFECTIVE (no unnamed-share numbers)

The all-things-considered YES/NO holistic row is floor-collapsed (mean p_yes .048 @7B,
.017 @14B; ≤4% items ≥ .5) → OOS R² meaningless (negative). Classic judge-score-distribution
collapse. The unnamed-residual instrument needs a graded or comparative elicitation before
any Act-3 claim. Do not quote any unnamed-share from this pass.

## Prediction status (v0; final verdicts await v1 references)

- **P-W1a-1** (exclusion leak higher for weight-installed): not evaluable AS WORDED — the
  W1a pass scored name-arm exclusion only, no articulation-arm exclusion rows (W1c scope).
  The computable adapter-vs-base contrast (+.93 vs +.35) matches the prediction's direction,
  with the global-rigidity confound noted above.
- **P-W1a-2** (NOT-gap larger for adapter-installed): direction SUPPORTED (1.41 vs .91 on
  A-cells), same confound caveat (gap equally large on non-trained B1).
- **P-W1a-3** (articulation composes better): NOT supported at v0 (see Finding 2).
- **P-W1a-4** (holistic residual shrinks with rung): NO VERDICT — instrument defective.

## Consequences queued

1. W1c scope grows: articulation-arm exclusion rows + trivial-inversion instruction
   g-control (the discriminating test) + elicitation passes + exemplar arm.
2. Holistic prompt redesign (graded 0-10 or pairwise-comparative) before unnamed-residual.
3. Route-signature bookkeeping: "instruction-rigidity" joins the tacit-profile composite
   candidates (involuntariness axis) — pending the W1c discrimination.

## W1b readout (added 2026-07-23, later): CoT-delta, verbalized confidence, P-B5

Artifacts: reason_first grids + confidence matrices (parse rates 1.000/.991/1.000 —
P-W1b-2 instrument gate PASSED all loads); tally = `battery/tally_w1b.py` →
`outputs/tacit_channels/battery_w1/tally_w1b_v0.json`. Canonical form; adapter rows half-2.

| config | CoT-delta med (fail / succ / other) | conf-acc verbal | guess-quartile agree | mean verbal conf |
|---|---|---|---|---|
| 7B base | −.134 / −.100 / −.063 | +.066 | .80 | **85.3** |
| 7B + real | −.035 / — / — | +.029 | .82 | **8.0** |
| 14B base | −.112 / −.095 / −.121 | +.047 | .81 | 29.2 |

**P-B5 FAILS (frozen body prediction, bar ≥ +.40):** proxy corr +.032 (7B) / −.284
(7B+real) / **−.729 (14B)**. Per the frozen stop rule: the |log-odds| v0 proxy is RETIRED
and W0 STAT-1 rows are FLAGGED/quarantined from confirmatory profiles — including the W0
metacognitive-block convergence (STAT-1×GEN-1 +.81), which must be re-evaluated with
VERBALIZED STAT-1 before P-B2 is scored. Nuance recorded: both conf-acc vectors are
near-zero at cell level, so the cross-cell correlation partly correlates noise — but the
bar was frozen and the consequence fires; the 14B −.729 is systematic, not noise.

**Dienes zero-correlation signature MET with real verbalized confidence, all configs:**
conf-acc ≈ 0 while bottom-confidence-quartile agreement ≈ .80 — high performance exactly
where the model claims to be guessing. This is the STAT-1 v1 result proper.

**P-W1b-1 (Beilock fluency-mismatch) FAILS as worded:** forced reason-first hurts at BOTH
rungs (7B fail −.134; 14B fail −.112, other −.121; 14B is NOT neutral) — explain-then-score
interference is rung-GENERAL here, not a below-floor phenomenon.

**Adapter dissociation (headline candidate):** the weight-installed config is (a) far LESS
disrupted by interposed reasoning (CoT-delta −.035 vs −.134 base on the same cells) and
(b) verbalizes CRASHED confidence (mean 8/100 vs 85/100 base) while judging BETTER
(tf ρ .69 vs .58). Weight-installation → interference-robust, low-self-trust competence:
two more involuntariness markers for the route-signature composite. Alt reading flagged:
0-100 elicitation format damage from narrow SFT (parse rate .991 argues the format works;
distribution check pending).

### Correction after distribution check (same day; check-score-distributions rule)

Verbalized-confidence value distributions: 7B base = 94% literal "85" (n_unique=5 —
near-CONSTANT boilerplate); 7B+real = {0: 77%, 100: 22%} (n_unique=4 — BINARY polarization,
not graded low confidence); 14B = real scale use (n_unique=12, std 28). Three revisions:

1. **Dienes zero-correlation claim DOWNGRADED to 14B-only.** At both 7B configs the verbal
   instrument is scale-degenerate, so conf-acc ≈ 0 there is instrument artifact, not
   metacognitive evidence. The clean STAT-1 v1 result: at 14B (valid instrument),
   conf-acc +.047 with guess-quartile agreement .81 — the signature holds where the
   instrument works.
2. **Adapter story re-read:** not "crashed confidence" but POLARIZED confidence — the
   weight-installed config verbalizes binary certainty {0,100} with no graded scale. Still a
   route-signature marker (graded metacognition lost under narrow SFT; cf. teacher-prob
   under-dispersion from M17), different phenomenon than low self-trust. The
   interference-robustness finding (CoT-delta −.035) is unaffected.
3. **New instrument gate (binding on future confidence loads, declared now):** parse rate
   alone is insufficient — add SCALE-USE validity: within-load n_unique ≥ 8 AND median
   within-cell std ≥ 5 on the 0-100 scale, else the load's verbal-confidence rows are void.
   P-W1b-2 as frozen (parse ≥ .80) PASSED but failed to catch degeneracy; this gate is its
   corrected successor (applies prospectively; today's 7B rows are void by it, 14B valid).

P-B5's failure stands regardless (bar frozen; at the one valid-instrument config it is
−.729, systematic anti-correlation, decisively below +.40).

### Audit corrections (2026-07-23, user-requested audit; full audit =
### notes/2026-07-23__w1-code-results-audit.md)

1. **RETRACTION of the "+.93 near-total rigidity" framing.** Cross-cell null shows the
   adapter's exclusion vector correlates +.55 with OTHER cells' tf vectors (generic
   installed-factor structure). Corrected construct-specific leak (same-cell minus
   cross-cell): 7B base +.29, adapter +.39, 14B −.19. Gradient survives; magnitudes modest.
2. **Exclusion instrument defect:** the frozen template's final line asks the STRAIGHT
   question after the inversion instruction — leak measures instruction-CONFLICT
   resolution (upper bound on rigidity). All configs shared the contradiction; comparative
   ordering retained. W1c: variant-consistent final question + cross-cell null becomes a
   standard companion statistic.
3. **Gradient re-grounded on the clean legs:** negation instrument (internally consistent)
   and base-rate semantics (14B flips mean p_yes .09→.69 under inversion; 7B/adapter do
   not move) independently confirm the ordering.
4. **guess-quartile chance = .664** (not .5): Dienes-signature claims rescoped to
   "modestly above chance."
5. Computations verified clean by independent recomputation (scipy path reproduces every
   checked number exactly); P-B5's −.729 has a real mechanism (logodds-conf-acc tracks
   competence +.52, verbal anti-tracks −.34, anti-corr persists within strata) — the two
   confidence operationalizations dissociate; retirement of the v0 proxy stands.

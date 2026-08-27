# Layer-3 closure pilot — two retrospective analyses: the SWAP decomposition and a missing-mass prototype

Date: 2026-08-06. Status: **retrospective, exploratory, CPU-only. No new judging, no GPU,
nothing killed.** Operates entirely on artifacts already on disk from the completed pilot
(`notes/2026-08-05__layer3_round4_peer_verdict.md`).

Terminology, spelled out on first mention per the standing rule:
**V** = the 17 programmatic surface features; **A** = the articulated-criterion bank;
**VA** = V and A concatenated; **VA_nl** = the gradient-boosted (HistGradientBoosting)
aggregation of that matrix, averaged over seeds {0,1,2}; **T** = the dense readout
(Llama-3.1-8B LoRA on raw text); **Δ** = T − VA_nl, the unarticulated residual;
**ε** = .005, the pilot's per-round saturation threshold; **OOF** = out-of-fold;
**AUC** = area under the ROC curve; **ρ** = Spearman rank correlation;
**GEPA** = the prompt-iteration pass required before confirmatory phrasing, deliberately
not applied in the pilot; **Good-Turing missing mass** = the estimated probability that
the next draw names a species not yet seen; **Chao1** = a lower-bound richness estimator
built from singleton and doubleton counts.

Artifacts:
- `methods/taste_decomposition/closure/recompute_round_preds.py` → `round_preds_all.npz`
  (per-round per-row VA_nl vectors; the pilot saved AUCs but not vectors)
- `methods/taste_decomposition/closure/swap_analysis.py` → `swap_analysis.json`
- `methods/taste_decomposition/closure/missing_mass.py` → `missing_mass.json`

**Reproduction gate passed.** The recomputed bank states reproduce
`round4_results.json` to 1e-9 on every reported AUC, and the pairwise machinery in
`swap_analysis.py` independently reproduces the pilot's published `group_boot_ci`
intervals exactly (MONITOR round 1 [−.0132, +.0145], round 2 [+.0023, +.0170],
round 3 [−.0091, +.0089], round 4 [−.0077, +.0059]). Everything below is computed on
the same numbers the pilot reported.

---

## PART 1 — The swap decomposition

### 1.1 The question and the algebra that answers it

Across rounds 1–3 the bank's rank agreement with the dense model rose (ρ .488 → .541 on
MONITOR) while its label AUC moved +.010 over four rounds. Did the bank **swap** — gain
pairs the dense model gets right while losing pairs it had right that the dense model
gets wrong?

AUC on a binary-label population is exactly the fraction of concordant
(positive, negative) pairs, so the question has an exact answer. Partition the
label-discordant pairs by **dense correctness**:

| cell | meaning | weight (honest rows) |
|---|---|---|
| **D+** | dense ranks the accepted paper above the rejected one | w₊ = .776 |
| **D−** | dense has it backwards | w₋ = .222 |
| **D0** | dense ties them | w₀ = .002 |

Let **C₊** = P(bank concordant with truth \| D+) and **C₋** = P(bank concordant \| D−).
Then, exactly:

```
AUC_bank  = w₊·C₊ + w₋·C₋ + w₀·C₀
agreement = w₊·C₊ + w₋·(1 − C₋) + w₀/2       (bank/dense order agreement, discordant pairs)
```

Agreeing with dense on D+ *is* being right (= C₊); agreeing with dense on D− *is* being
wrong (= 1 − C₋). So AUC and agreement are two different linear functionals of the same
pair (C₊, C₋), differing only in the **sign** on the D− term. A step with ΔC₊ > 0 and
ΔC₋ < 0 of matched magnitude raises agreement and leaves AUC flat. That is the swap
signature, and it is directly measurable.

**Population.** The honest one: the 1,244 dense-held-out rows (`dense_split` ∈
{eval, test}) with same-rows dense predictions. Bank predictions are out-of-sample
everywhere (grouped-OOF inside FIT+MINE, refit-and-predict on MONITOR) — the identical
`va_full` construction behind the pilot's `Delta_honest_level_heldout1244`. MONITOR
figures are reported alongside for continuity with the published ρ series and are flagged
dense-CONTAMINATED (943 of 1,192 MONITOR rows sat in the dense model's train split).
Uncertainty: group-level (`ntitle`) paired bootstrap, 2,000 draws.

### 1.2 The levels

Honest population, n = 1,244, dense AUC = .7769:

| | r0 | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| **C₊** (concordance on dense-right pairs) | .7597 | .7778 | .7811 | .7829 | .7830 |
| **C₋** (concordance on dense-wrong pairs) | .4227 | .3807 | .3862 | .3928 | .3946 |
| AUC (= w₊C₊ + w₋C₋ + w₀C₀) | .6844 | .6891 | .6929 | .6958 | .6962 |
| agreement with dense, label-discordant pairs | .7187 | .7421 | .7434 | .7434 | .7430 |
| agreement with dense, **same-label** pairs | .6726 | .6970 | .6975 | .6980 | .6974 |
| ρ(VA_nl, dense), honest rows | .5433 | .6024 | .6038 | .6044 | .6016 |
| ρ(VA_nl, dense), MONITOR (published) | .4884 | .5225 | .5314 | .5412 | .5371 |

**First observation, before any decomposition:** on the honest rows the alignment climb is
**entirely round 1** (ρ +.059, then +.001, +.001, −.003). The published MONITOR series
keeps climbing through round 3 because MONITOR's dense predictions are partly in-sample,
so "agreement with dense" there partly means "agreement with memorised labels" and
inflates as the bank's own accuracy improves. The honest rows are authoritative.

### 1.3 The per-step decomposition

Honest population (primary):

| step | ΔC₊ | ΔC₋ | ΔAUC [95% CI] | Δagreement | error−insight inheritance [95% CI] P(>0) |
|---|---|---|---|---|---|
| **r0→r1** | **+.0181** | **−.0420** | +.0047 [−.0100, +.0190] | **+.0234** | **+.0239 [−.0085, +.0573] .94** |
| r1→r2 | +.0033 | **+.0056** | +.0038 [−.0040, +.0110] | +.0013 | −.0089 [−.0272, +.0098] .18 |
| r2→r3 | +.0018 | **+.0066** | +.0029 [−.0058, +.0116] | −.0000 | −.0084 [−.0292, +.0124] .22 |
| r3→r4 | +.0001 | **+.0018** | +.0005 [−.0066, +.0079] | −.0004 | −.0019 [−.0209, +.0149] .42 |
| **r0→r4** | **+.0233** | **−.0281** | +.0118 [−.0052, +.0300] | **+.0243** | +.0048 [−.0376, +.0457] .60 |

MONITOR (dense-contaminated; shown because the published ρ series lives here, and because
the ΔAUC column reproduces the pilot's stopping-rule statistic exactly):

| step | ΔC₊ | ΔC₋ | ΔAUC [95% CI] | Δagreement | error−insight inheritance P(>0) |
|---|---|---|---|---|---|
| **r0→r1** | +.0059 | **−.0334** | **+.0003** [−.0132, +.0145] | **+.0098** | **+.0275 .93** |
| r1→r2 | +.0097 | **+.0083** | **+.0095** [+.0023, +.0170] | +.0071 | **−.0180 .03** |
| r2→r3 | +.0017 | −.0096 | +.0001 [−.0091, +.0089] | +.0028 | +.0079 .71 |
| r3→r4 | −.0007 | −.0016 | −.0008 [−.0077, +.0059] | −.0004 | +.0023 .59 |

"error−insight inheritance" = (−ΔC₋) − ΔC₊, i.e. how much faster the step took on dense's
**errors** than dense's **insights**, per pair. Positive = swap.

### 1.4 The flow table

Pairs the bank flipped between r0 and r4, cross-tabbed by dense correctness (honest rows;
569 accepted × 675 rejected = 384,075 label-discordant pairs):

| | D+ (298,070 pairs) | D− (85,356 pairs) | D0 (649) |
|---|---|---|---|
| right → right | 209,240 | 26,189 | 279 |
| right → **wrong** | **17,194** | **9,892** | 79 |
| **wrong** → right | 24,134 | 7,494 | 70 |
| wrong → wrong | 47,502 | 41,781 | 221 |
| loss rate among start-right | 7.6% | **27.4%** | 22.1% |
| gain rate among start-wrong | 33.7% | 15.2% | 24.1% |
| net flips per pair | **+.0233** | **−.0281** | −.0139 |

Totals: **31,698 pairs gained, 27,165 lost, net +4,533** — an 86% payback rate on a churn
of 15.3% of all pairs, for a net AUC move of +.0118. Losses are enriched **1.64×** in D−
relative to D−'s share of pairs (36.4% of losses land in a cell holding 22.2% of pairs);
gains are enriched **0.98×** in D+ — i.e. at the base rate. **The asymmetry is entirely on
the loss side**: the bank's per-pair loss rate is 3.6× higher on pairs the dense model gets
wrong (27.4%) than on pairs it gets right (7.6%).

### 1.5 Verdict

**It is swapping, but the swap is one round rather than the whole curve, and it is a
partial cancellation rather than a full one.**

1. **Round 1 is a textbook swap.** Concordance rose +.018 on the 78% of pairs the dense
   model ranks correctly and fell −.042 on the 22% it ranks wrongly. Agreement with dense
   rose +.023 while label AUC moved +.005 (CI includes zero; on MONITOR +.0003, the
   pilot's own null). Error-inheritance exceeded insight-inheritance by +.024 per pair,
   P(>0) = .94 honest / .93 MONITOR. Round 1's 14 criteria taught the bank the dense
   model's mistakes at roughly 2.3× (honest) to 5.7× (MONITOR) the per-pair rate at which
   they taught it the dense model's insights.
2. **Rounds 2–4 are not swaps — they are simply small.** On the honest rows all three
   moved **both** cells the right way; C₋ recovered +.014 of round 1's −.042. Round 2, the
   one round that cleared ε, is *significantly anti-swap* on MONITOR (error−insight
   = −.018, P(>0) = .03). It was genuine articulation, not alignment.
3. **Over the pilot as a whole, the swap costs about a third of the gain.** D+ contributed
   w₊·ΔC₊ = +.0181 of AUC; D− subtracted w₋·ΔC₋ = −.0062. The D− loss cancels **34%** of
   the D+ gain, and 86% of gained pairs were paid back by lost ones. The r0→r4
   error−insight statistic is +.005 with CI [−.038, +.046], P(>0) = .60 — the pilot-level
   swap is real in direction but not individually significant, precisely because rounds
   2–4 partly undid round 1.
4. **It is not "pure redundancy" either.** The third hypothesis — that ρ rose only on
   pairs where AUC cannot see it — fails: same-label-pair agreement rose +.0248, almost
   exactly matching the label-discordant rise of +.0243. The alignment gain is spread
   evenly across pair types, not hidden in the AUC-invisible ones.

**Programme-level reading.** The pilot's headline that "articulation and prediction came
apart" is right, but the mechanism is narrower than it looked. A null round is not a round
in which nothing was learned; round 1's null is a round in which what was learned was
**dense-model-specific error**. That is a distinguishable failure mode from "the remainder
is tacit," and it is measurable at a cost of zero extra judge calls. It also sharpens the
pilot's own caution — "some of Δ is dense-model idiosyncrasy that no articulation *should*
capture" — into a number: mining transferred idiosyncrasy worth −.0062 AUC while
transferring insight worth +.0181.

**Design implication.** The stopping rule reads ΔAUC only, so it cannot tell round 1
(swap) from round 4 (nothing). Adding the per-step (ΔC₊, ΔC₋) pair to the round readout
costs nothing — it is a re-slice of predictions already computed — and separates
"the proposer found nothing" from "the proposer found dense's mistakes."

---

## PART 2 — Good-Turing / missing-mass prototype

Motivated by the capture-recapture discipline from the prompt-scaling line: **mining moves
the bound's LEVEL; the audit has to give its WIDTH.** The 2-consecutive-sub-ε rule says
the last two draws were small. It does not say how much articulable signal was left
unmined.

### 2.1 (a) How much recoverable AUC did the 4-round stop leave on the table?

Honest per-round marginal gains: **+.00465, +.00380, +.00289, +.00046** (levels
.6844 → .6891 → .6929 → .6958 → .6962).

| model | fit | remaining Σ_{r≥5} g_r | predicted next-round gain g₅ |
|---|---|---|---|
| geometric, g_r = a·λ^(r−1) | a = .00497, **λ̂ = .668** | **+.0030** | **+.0010** |
| saturating exponential on levels, V_r = V_∞ − c·λ^r | V_∞ = .6996, λ̂ = .662 | **+.0033** | +.0010 |
| power law, g_r = a·r^(−b) | **at-bound (b → 1.05)** | .100 — **NOT IDENTIFIED, do not quote** | +.0010 |

Group-bootstrap (2,000 draws, honest rows, prediction vectors held fixed):

| quantity | median | IQR | 95% CI | P(> ε = .005) |
|---|---|---|---|---|
| remaining, geometric | .0019 | [.0001, .0293] | [.0000, .2235] | .41 |
| remaining, geometric, **non-degenerate fits only** | .0009 | [.0001, .0051] | [.0000, .0533] | .25 |
| remaining, saturating exponential | .0040 | [.0014, .0430] | [−.0017, .2404] | .46 |
| **next-round gain g₅, geometric** | **.0006** | [.0001, .0019] | **[.0000, .0048]** | **.02** |
| next-round gain g₅, saturating | .0008 | [.0001, .0022] | [.0000, .0051] | .03 |
| decay rate λ̂ | .652 | [.346, .980] | [.010, .980] | — |

**Headline: the 4-round stop plausibly left ≈ +.003 AUC of articulable signal unmined
(≈ 4% of the +.081 residual, ≈ 25% of the +.0118 the four rounds actually recovered),
with a readout-only 95% interval of [.000, .053] and an unrestricted interval of
[.000, .224] that is an artefact, not evidence.**

Two honesty flags that matter more than the point estimate:

- **A 4-point gain series does not identify a decay rate.** The bootstrap λ̂ runs to
  *both* bounds (34.9% of geometric fits, 35.2% of saturating fits, 56.8% of power-law
  fits terminate at a bound). The fat upper tail of every remaining-mass CI comes from
  λ → 1 replicates, not from evidence of a hidden reservoir. The infinite-tail sum is
  the wrong statistic to bootstrap.
- **The well-conditioned statistic is the predicted next-round gain**, and it corroborates
  the stop cleanly: g₅ = +.0010, 95% CI [.0000, .0048], with only a **2%** bootstrap
  probability of clearing ε = .005. A fifth round was very unlikely to fire the rule.
- **The CI is readout noise only.** It holds the prediction vectors fixed and resamples
  rows. It does *not* include proposer variance — a different proposer, or the GEPA pass
  the pilot deliberately skipped, would move the gains by more than this interval. Treat
  it as a lower bound on the true width. Closing that gap is exactly what (c) is for.

**λ sensitivity** (the honest way to report a non-identified tail), anchored on the fitted
amplitude:

| λ | .4 | .5 | .6 | **.67 (fitted)** | .7 | .8 | .9 |
|---|---|---|---|---|---|---|---|
| remaining AUC mass | .0002 | .0006 | .0016 | **.0030** | .0040 | .0102 | .0326 |

Even at λ = .9 — a decay far slower than anything the four observed rounds support — the
remaining mass is +.033, well under half the +.081 residual. **Under no defensible decay
rate does continued criterion-proposal close the residual.** That is the strongest form of
the pilot's conclusion, and it is stronger than the stopping rule alone delivers.

### 2.2 The mechanism: value without increment

Independently of the stack curve, from the per-round mechanism JSONs:

| round | new block alone-AUC (MONITOR) | best new criterion, alone-AUC | stack gain (MONITOR) | conversion of alone-excess into stack gain |
|---|---|---|---|---|
| 1 | .618 | .621 | +.0003 | 0.2% |
| 2 | .650 | .609 | +.0095 | 6.3% |
| 3 | .615 | .585 | +.0001 | 0.1% |
| 4 | .553 | .550 | −.0008 | −1.5% |

Every round's new block carries **real standalone signal** and converts almost none of it
into a stack increment. Fitting the per-round best-criterion alone-AUC excess
(|AUC − .5| = .121, .109, .085, .049) gives λ = .78 per round and predicts roughly **six
more rounds** before the best of 15 proposals is individually unreadable (excess < .01).

**So the pilot's saturation is redundancy saturation, not value exhaustion.** The proposer
had not run out of readable criteria; it had run out of criteria the bank does not already
span. That is a materially different claim from "no further nameable criteria exist," and
it is the claim the data supports.

### 2.3 (b) Redundancy / recapture census

Embeddings: `BAAI/bge-large-en-v1.5`, CLS pooling, L2-normalised, run locally on CPU.
Two text variants (name only; name + definition with scoring boilerplate stripped).

**Threshold calibration first.** bge cosine is compressed — the 154-bank's own internal
median is .58–.63 — so an absolute threshold from another corpus does not transfer. The
pilot's **planted probes** supply a principled floor: they were authored to be lexically
similar to a real criterion and conceptually distinct, and the blind auditor caught both.

| planted pair | cos (name) | cos (name + definition) |
|---|---|---|
| "Bare mention of the word *ablation*" vs "Necessity established — an ablation shows the components are actually required" | .698 | **.739** |
| "Mathematical symbols embedded in the method's own name" vs "Cost accounting includes the method's own overhead" | .619 | .615 |

Any threshold at or below .74 would call genuinely distinct concepts duplicates. **The
defensible band is τ ≥ .78.**

**Recapture among the 56 mined criteria** (1,540 pairs):

| τ | name-only: dup pairs (cross-round) | name+definition: dup pairs (cross-round) | species (name+def) | f1 / f2 | Good-Turing missing mass |
|---|---|---|---|---|---|
| .70 *(below probe floor)* | 38 (27) | 180 (145) | 9 | 8 / 0 | .143 |
| .75 *(below probe floor)* | 6 (4) | 28 (23) | 31 | 29 / 1 | .518 |
| **.78** | **0 (0)** | **7 (6)** | **50** | 46 / 2 | **.821** |
| **.80** | **0 (0)** | **3 (3)** | **53** | 50 / 3 | **.893** |
| .85 | 0 (0) | 0 (0) | 56 | 56 / 0 | 1.000 |
| .90 | 0 (0) | 0 (0) | 56 | 56 / 0 | 1.000 |

**Near-duplicate rate in the defensible band: 0.2–0.5% of mined pairs (3–7 of 1,540),
6 of 7 of them crossing rounds; concept-level recapture 5–11% (50–53 species from 56
proposals).** The genuine recaptures are real and legible — the top pairs are
"Ambition of the claim together with the specificity of the evidence" (r1) vs
"Claim scope matches method scope" (r3) at .813, and "Cost stated TOGETHER WITH gain"
(r2) vs "Efficiency gain stated TOGETHER WITH verified non-degradation" (r3) at .812.

**Mined-vs-original-154-bank overlap is ≈ 0** (0 of 56 with a bank match at τ ≥ .80,
max cosine .793). **This is NOT evidence of proposer novelty.** The 154-bank is a general
scientific-reporting rubric bank (CONSORT / PRISMA / STROBE / ARRIVE / TIDieR / CHEERS
items) while the mined criteria are machine-learning-abstract specific. The near-zero
overlap is largely domain mismatch.

**Chao1 on the mined set is degenerate and must not be quoted.** With f2 ∈ {0, 2, 3} the
estimator returns nothing (τ ≥ .85) or wildly unstable values (579 at τ = .78; 470 at
τ = .80), and Good-Turing missing mass returns .82–1.00 — "almost the entire concept space
is unseen." **That is a design artefact, not a finding.** The pilot's rounds were
sequential, each proposer saw the current bank, and each was instructed not to duplicate
it. Recapture was suppressed by construction, so the estimator has nothing to work with.
This is precisely the gap (c) closes.

### 2.4 The one place the pilot *does* support capture-recapture — and what it says

The original A bank was not authored as 154 distinct concepts. It was produced by
clustering aspects harvested from many independent source rubrics, and the merge left
exact repeats in place. Those repeats are genuine recaptures by independent draws.

| quantity | value |
|---|---|
| delivered criteria | 154 |
| **distinct names (= distinct descriptions)** | **95** |
| repeat-group size distribution | {1: 59, 2: 22, 3: 8, 4: 3, 5: 3} |
| duplicate-name column pairs | 94 |
| of those, **bit-identical score columns** | **94 (100%)** |
| A columns surviving the degeneracy screen | 79 |
| **distinct concepts among surviving columns** | **54** |
| Good-Turing missing mass (f1/n = 59/154) | **.383** |
| Chao1 (bias-corrected) | 169.4 (classic form 174.1) |

**Three findings, all new:**

1. **The "154-criterion A bank" is 95 distinct concepts, and the model sees 54.** All 94
   duplicate-name column pairs carry bit-identical scores — they are copies, not
   independent re-scorings, so they contribute zero information. The round-0 feature count
   of 96 (79 A + 17 V) is really **54 A concepts + 17 V features = 71**.
2. **This reframes the closure curve's x-axis.** Four rounds added 56 distinct mined
   concepts to an effective base of 54 — mining roughly **doubled** the bank's distinct
   concept count and bought +.0118 AUC. Stated as "154 criteria plus 56 more" the pilot
   looks like a 36% expansion; stated in distinct concepts it is a 104% expansion for
   1.2 AUC points. The second framing is the honest one and it makes the residual look
   *more* robust, not less.
3. **Good-Turing on the bank's own construction says it captured ~56% of its reachable
   concept space** (95 seen of ~169 estimated), with a 38% chance the next independent
   draw names something new. The bank builder's sampling process was nowhere near
   exhausted either — which is consistent with (2.2): the constraint is redundancy against
   what is already spanned, not scarcity of nameable criteria.

**Hygiene action for the freeze:** deduplicate the incoming bank before round 0 and report
the closure curve against **distinct concepts**, not delivered criteria. The pilot's
existing "corpus hygiene: dedup pass before use" recommendation covered the *abstracts*;
it needs to cover the *criteria* too.

---

## PART 3 — (c) Prospective missing-mass estimator for confirmatory runs

**One page. Slots into the freeze as an ADDITIONAL stopping diagnostic, not a replacement
for the 2-consecutive-sub-ε rule.**

**The defect it fixes.** The pilot's rounds were sequential: each proposer was shown the
current bank and told not to duplicate it. That is good bank hygiene and fatal to
richness estimation — it drove observed recapture to ~0 and made Good-Turing return
"missing mass = 1.00" (§2.3). Independence is the missing ingredient, and it is cheap.

**Design, per round r.**

1. **P independent proposers, P ≥ 5.** Each runs in a *sealed* context: the same
   disagreement slice, the same instructions, the same k, and **no sight of the current
   bank and no sight of each other**. Total proposals N_r = P × k. Bank-conditioning moves
   from the proposal step to the admission step, where it belongs.
2. **Species assignment.** Deduplicate the N_r proposals by embedding cosine at threshold
   τ, single-linkage. τ is **calibrated per round on the planted probes** — it must sit
   strictly above the maximum cosine of a planted lexical-lookalike pair (that floor was
   .739 in the pilot; τ = .78–.80 was defensible). Pairs within ±.02 of τ go to the same
   blind auditor that already does A/B routing — roughly 10 extra judgments per round.
   The probes now do double duty: auditor test and threshold calibration.
3. **Frequencies.** n_s = proposals in species s; S_obs = species; f1, f2 = singleton and
   doubleton counts.
4. **Estimators, reported per round alongside Δ_r:**
   - **Good-Turing missing mass** M̂_r = f1 / N_r — the probability the next independent
     proposal names a concept not yet seen this round.
   - **Chao1 richness** Ŝ_r = S_obs + f1(f1−1) / (2(f2+1)). Use the bias-corrected form
     only; the classic f1²/(2f2) form is unusable at small f2, as §2.3 demonstrates.
   - **Remaining-AUC bound** R̂_r = [M̂_r / (1 − M̂_r)] × Δ_r × λ̂_r, where Δ_r is the round's
     realised stack gain and λ̂_r is the geometric decay fitted to the marginal-gain series
     to date. The odds factor scales the realised gain by the unseen mass; λ̂_r discounts
     it because unseen species are by construction the rare ones and rare is low-value in
     this instrument (§2.2 measured that decay at λ ≈ .78 per round on criterion value).
     Report λ̂'s at-bound fraction alongside — if it exceeds ~1/3, say the tail is not
     identified rather than quoting an interval (§2.1).
   - **Two bootstraps, both required.** (i) group-level (`ntitle`) over the readout, as
     now; (ii) **proposer-level**: resample the P proposers with replacement and recompute
     f1, S_obs and Δ_r. The second is the width the pilot could not produce and is the
     whole point of the design.
5. **Stopping diagnostic STOP-M.** Fires when R̂_r < ε **and** M̂_r < M_crit (suggest
   M_crit = .25). It is a **second, independent gate that must also be satisfied**, never a
   substitute: the 2-consecutive-sub-ε rule stays primary and stays unchanged. If the Δ
   rule fires while STOP-M does not, the run still stops, but the report must carry
   "saturation declared with estimated remaining mass R̂ = x [CI]" — which is exactly the
   width the pilot could not state.

**Cost — the reason this is affordable.** Independence is bought at *proposal* cost, not
*scoring* cost. Only deduplicated species **representatives** are scored, which is what
the bank needs anyway, so the judge-call ledger grows by the number of **new species**, not
by P. Proposal calls are cheap; the pilot's 151,650-prompt-per-round scoring bill is
untouched. P = 5 proposers at k = 25 gives 125 proposals per round; if they collapse to
~40 species, scoring cost rises ~60% over the pilot's 25 while producing a real f1/f2
spectrum for the first time.

**Two supporting requirements, both free.**

- **Run the census on the incoming bank at round 0** (§2.4). The pilot's 154-criterion bank
  turned out to be 95 distinct concepts and 54 effective columns. The closure curve's
  x-axis must be distinct concepts.
- **Add the (ΔC₊, ΔC₋) pair to every round's readout** (§1.5). It is a re-slice of
  predictions already computed, costs nothing, and separates "the proposer found nothing"
  from "the proposer found the dense model's mistakes" — a distinction the ΔAUC stopping
  statistic is structurally blind to.

---

## Caveats that travel with every number here

1. **Retrospective and exploratory.** No prereg covers these two analyses; they are
   descriptive re-slices of a completed exploratory pilot. Nothing here changes the pilot's
   saturation verdict or its Δ_plateau = +.081.
2. **Pre-GEPA**, like the pilot. Better-phrased criteria could move both the swap
   decomposition and the remaining-mass estimate.
3. **All missing-mass CIs are readout-only.** Proposer variance is not in them and is
   larger. (c) exists to fix this.
4. **The mined-set Chao1 and Good-Turing figures are degenerate by design** and are
   reported to exercise the machinery, never as richness estimates.
5. **The mined-vs-bank near-zero overlap is confounded by domain mismatch** (general
   scientific-reporting rubrics vs machine-learning abstracts).
6. **MONITOR figures remain dense-contaminated.** Where the honest rows and MONITOR
   disagree — notably on the sign of ΔC₋ in round 3 — the honest rows govern.

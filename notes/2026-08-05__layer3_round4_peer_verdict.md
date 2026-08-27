# Layer-3 articulation closure — ROUND 4 + FINAL PILOT SUMMARY (peer-review VERDICT)

Date: 2026-08-05 (round 4 completed 2026-08-06 00:04 PT).
Status: **exploratory pilot, PRE-GEPA. SATURATION DECLARED — pilot closed at round 4 of a
5-round cap.**
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` (Amendments 1 and 2).
Rounds 1-3: `notes/2026-08-05__layer3_round{1,2,3}_peer_verdict.md`.
Code + artifacts: `methods/taste_decomposition/closure/`.

**Headline: round 4 gain = −.0008 (95% CI [−.0077, +.0059], P(>0)=.41), sub-ε. With
round 3 (+.0001) that is two consecutive sub-ε rounds → SATURATION DECLARED.** The
closure curve plateaus at **Δ = +.081** on the honest same-rows population.

Terminology per the spell-out rule: **V** = 17 programmatic surface features;
**A** = the articulated-criterion bank; **VA** = both concatenated; **lin**/**nl** =
frozen linear / gradient-boosted aggregation of the same matrix; **T** = dense readout
(Llama-3.1-8B LoRA on raw text); **Δ_beyond** = T − VA_nl; **ε** = .005; **OOF** =
out-of-fold; **AUC** = area under the ROC curve; **GEPA** = the prompt-iteration pass
required for confirmatory phrasing, deliberately not applied in the pilot.

---

## 1. Round-4 procedure and two instrument flags

Slice mined against the round-3 bank (152 features after round 4); rows read in rounds
1-3 excluded. Mining-slice rank gap **.589 → .470 → .386 → .344**.

Track A k=15 with the composite count **asserted in code before scoring** (9 composite
/ 6 simple) — the round-3 deviation is now structurally preventable. Track B k=10 fresh
channels including **two planted probes**.

| | r1 | r2 | r3 | r4 |
|---|---|---|---|---|
| misrouting rate | 4.0% | 4.0% | 0.0% | **4.0%** |
| disputes → arbiter | 1 | 1 | 0 | **1** |
| final routing (pre-gate) | 14 A / 11 B | 14 A / 11 B | 15 A / 10 B | **14 A / 11 B** |

**Both planted probes were caught.** "Bare mention of the word *ablation*" and
"multiplicative speedup notation" were routed incidental while their substantive pairs
(ablation-establishes-necessity; cost-accounting-includes-own-overhead) went to the
bank. The audit separates tallying from judging.

**The dispute repeated my own authoring failure mode.** "Poses a specific question and
states its answer plainly" carried the clause *"score the question-and-answer
structure, not the importance of the question"*. The auditor flagged exactly that
clause; the arbiter upheld it, noting that as written a trivial question answered
plainly scores 10 while an abstract asserting a sharp finding declaratively scores low
— penalising the strongest form of the underlying virtue is the signature of a
document-shape feature. The arbiter added the asymmetry that matters: **admitting a
shape feature to the A bank inflates the articulated aggregation and shrinks the
residual, manufacturing a false "we articulated it away."** Twice now (round-2
belief-change, round-4 question-form) I wrote an instruction telling the judge to score
form rather than merit, and twice the blind audit caught it. That is evidence the audit
is load-bearing, not ceremonial.

### FLAG 1 — three criteria collapsed, one of them A-routed

| id | route | criterion | modal frac | distinct |
|---|---|---|---|---|
| P10 | B | bare mention of "ablation" (**planted probe**) | .98 | 3 |
| P24 | B | math symbols in the method's own name | .98 | 5 |
| **P23** | **A** | **reports variability across seeds, splits, or runs** | **.99** | 7 |

All three gated out. **P23 is the first A-routed collapse of the pilot**: ML abstracts
essentially never mention seed/split variability (mean .05), so the criterion is a real
corpus fact rather than a decoding failure — but it means round 4 contributes **13 A**
and **9 B** criteria, not 14/11. One planted probe collapsing also means that probe
tested the auditor but could not test the discount.

### FLAG 2 — the positive/negative anchor separation INVERTED this round

| anchor class | r1 | r2 | r3 | **r4** |
|---|---|---|---|---|
| positive | 3.11 | 2.84 | 3.30 | **1.76** |
| negative | 2.77 | 2.74 | 2.87 | **2.07** |
| scrambled | 1.06 | 0.94 | 0.12 | **0.38** |
| coherent vs scrambled AUC | .972 | 1.000 | 1.000 | **.997 PASS** |
| **positive vs negative AUC** | .719 | .608 | .736 | **.361 INVERTED** |

The scrambled check passes emphatically, so the instrument is still reading text. But
known-accepted anchors scored *lower* than known-rejected ones on the round-4 criteria
as a set. Two readings, and I cannot separate them with K=12: (a) noise — at 12 vs 12
the standard error on this AUC is ≈.11, so .361 sits ~1.3 SE below .5 and is not
individually significant; (b) signal — round 4's Track A is dominated by
self-criticism criteria (volunteers the unflattering number, tests where it breaks,
checks a trivial baseline, reports seed variability), and if accepted papers volunteer
*less* self-criticism than rejected ones the inversion is real and substantive. Several
round-4 criteria do sit below .5 alone (P13 .4961, P17 .4992, P11 .4981, P05 .4994).
**Recorded as an open question, not resolved.** It does not affect the saturation
verdict, which rests on VA_nl gains, but it is the single most interesting anomaly the
pilot produced and it warrants a dedicated K≥50 anchor battery.

---

## 2. THE CLOSURE CURVE — final

All five bank states recomputed in one pass, identical estimator, identical split.

| | round 0 | round 1 | round 2 | round 3 | round 4 |
|---|---|---|---|---|---|
| features after screen | 96 | 110 | 124 | 139 | 152 |
| **VA_nl, MONITOR (n=1,192)** | **.6633** | **.6635** | **.6730** | **.6731** | **.6723** |
| VA_nl seed spread | .0030 | .0118 | .0060 | .0032 | .0043 |
| VA_lin, MONITOR | .6597 | .6677 | .6667 | .6683 | .6645 |
| Δ_interact = VA_nl − VA_lin | +.0035 | −.0042 | +.0063 | +.0048 | +.0078 |
| VA_nl, honest level (n=1,244) | .6844 | .6891 | .6929 | .6958 | .6962 |
| **Δ, honest level (T = .7769)** | **+.0925** | **+.0878** | **+.0840** | **+.0811** | **+.0807** |
| Δ, same-rows (n=249, T = .7838) | +.1363 | +.1485 | +.1309 | +.1292 | +.1433 |

### Round-over-round gains (group-level paired bootstrap, 2,000×)

| round | VA_nl gain on MONITOR | 95% CI | P(>0) | sub-ε (.005)? |
|---|---|---|---|---|
| r0 → r1 | +.0003 | [−.0132, +.0145] | .54 | yes |
| r1 → r2 | +.0095 | [+.0023, +.0170] | .996 | no |
| r2 → r3 | +.0001 | [−.0091, +.0089] | .50 | yes |
| **r3 → r4** | **−.0008** | **[−.0077, +.0059]** | **.41** | **yes** |

**Flags [yes, no, yes, yes] → trailing run = 2 → SATURATION DECLARED.**

Per-round Δ decrement on the honest level: −.0047, −.0038, −.0029, **−.0004**. The
curve is visibly flattening, not merely stopping on a threshold.

---

## 3. Saturation is corroborated by four independent diagnostics

The stopping rule fires on one statistic; these were not part of it and all agree.

| diagnostic | r1 | r2 | r3 | r4 |
|---|---|---|---|---|
| best new criterion, alone-AUC | .621 | .609 | .585 | **.550** |
| best new criterion, ρ vs dense | .362 | .337 | .268 | **.123** |
| new block alone, nonlinear AUC | .618 | .650 | .615 | **.553** |
| new block vs dense, ρ | .371 | .434 | .326 | **.135** |

And the rank-agreement climb that ran through rounds 1-3 has itself stopped:

| ρ(VA_nl, dense) | r0 | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| MONITOR | .488 | .522 | .531 | **.541** | **.537** |
| same-rows | .360 | .397 | .414 | **.426** | **.423** |

The proposer has run out of articulable structure that the bank does not already carry.

---

## 4. Track B — final: discount null at every set size

| spurious-alone AUC | r1 (11) | r2 (22) | r3 (31) | **r4 (40)** |
|---|---|---|---|---|
| linear, MONITOR | .6040 | .6286 | .7122 | **.7133** |
| HistGB, MONITOR | .5957 | .6303 | .7062 | **.7077** |

The nuisance set saturated too: 31 → 40 channels moved spurious-alone by +.001.

Strongest channels overall:

| id | nuisance | alone AUC |
|---|---|---|
| r3:P15 | **a public repository URL appears in the abstract** | **.5934** |
| r1:P02 | abstract length and verbosity | .5672 |
| r2:P11 | sentence complexity independent of length | .5631 |
| r4:P07 | **passive-voice density** | **.4386** |
| r2:P14 | non-idiomatic English / grammatical slips | .4384 |
| r1:P20 | dominant research topic | .5486 |
| r2:P22 | first-person narrative density | .5310 |
| r1:P24 | anonymised-repo / on-acceptance boilerplate | .4778 |

| discounted readout | n | pooled Δ | **Δ_adj** |
|---|---|---|---|
| dense-held-out (honest), q10 joint B | 1,244 | +.0807 | **+.1098** |
| MONITOR ∩ held-out, q5 joint B | 249 | +.1433 | +.1774 |

**Bottom line: spurious discounting never shrank the residual, at 11, 22, 31 or 40
nuisances.** Stratifying costs VA more than T (VA −.094, T −.065 at r4), so Δ_adj rises.
**Do not quote +.1098 as an effect size**: at spurious-alone .713 the joint-B score is
92% as predictive as T itself (.777), so conditioning on its deciles approaches
conditioning on the label, and both AUCs are attenuated. The defensible claim is the
negative one: **the residual is not explained by the mined shortcut channels.**

---

## 5. FINAL PILOT SUMMARY

### (a) The curve

| round | criteria added (post-gate) | bank size | VA_nl MONITOR | Δ honest level |
|---|---|---|---|---|
| 0 | — (154 A + 17 V) | 96 | .6633 | **+.0925** |
| 1 | +14 A | 110 | .6635 | +.0878 |
| 2 | +14 A | 124 | .6730 | +.0840 |
| 3 | +15 A | 139 | .6731 | +.0811 |
| 4 | +13 A | 152 | .6723 | **+.0807** |

56 new criteria across four rounds, 606,600 judge calls, total Δ movement **−.0118**.

### (b) The plateau value — the pilot's taste-bound estimate

**Δ_plateau = +.081 AUC** (T = .777 vs VA_nl = .696), on the 1,244 dense-held-out rows
of the peer-verdict A/V population.

Caveats that travel with that number, all load-bearing:

1. **PRE-GEPA.** Criteria are fidelity-phrased but not GEPA-iterated. The A-bank
   standard requires GEPA for confirmatory work; better phrasing could close more.
2. **Protocol-specific level, not comparable to Layer 1's Δ_beyond = +.065.** The
   closure split fits on 80% and reads on a fresh 20%, and T is restricted to
   dense-held-out rows; both shift the level. Only round-over-round changes and this
   honest level are quotable, never the +.14 same-rows figure or the +.19 contaminated
   MONITOR figure.
3. **MONITOR contamination.** 943 of 1,192 MONITOR rows sat in the dense model's TRAIN
   split, so "T on MONITOR" (.857) is meaningless. Every Δ here uses either the 249-row
   MONITOR ∩ held-out set or the better-powered 1,244-row dense-held-out set.
4. **The honest level is mildly mining-contaminated and therefore conservative.** Its
   FIT+MINE portion uses OOF predictions from a bank whose criteria were authored by
   reading 240 rows of that same split, which inflates VA and understates Δ.
5. **Exploratory, single cell, no TEST split.** The pilot has no third split; the
   confirmatory design adds one, quoted once.
6. **No downward spurious adjustment is applied**, because the discount was null.

### (c) Track-B bottom line

Forty declared nuisances reach **.713 alone** — 92% of the way from chance to the dense
model — yet stratifying on them leaves Δ unmoved or larger at every set size. Two
concrete instrument findings fell out:

- **The artifact-availability channel is bipolar**: a live public repository URL is the
  single most predictive nuisance (.593), while anonymised-repo / release-on-acceptance
  boilerplate runs the other way (.478). The existing 154-criterion bank scores
  availability statements without distinguishing them, so part of the bank is wired to
  a channel whose sign flips on a detail it cannot see.
- **Surface fluency is strongly anti-predictive**: non-idiomatic English (.438) and
  passive-voice density (.439) are the two strongest negative channels. Whether fluency
  is a nuisance or a merit is a real decision, not a default.

### (d) The program-level finding: articulation and prediction came apart

Across rounds 1-3 the articulated stack's *ranking* moved steadily toward the dense
model's (ρ .488 → .541) while its *label AUC* moved +.010 and Δ fell only −.011. Mining
kept hitting its stated label-blind target and kept not converting that into predictive
parity. By round 4 both had stopped together.

The natural reading is that what the dense model has beyond the bank is not a list of
further nameable criteria. Four rounds of a well-resourced proposer, reading the exact
rows where the two disagree most, with composites allowed and explicitly encouraged,
recovered .010 AUC. Either the remainder is genuinely tacit for this instrument, or it
is reachable only by a different *kind* of articulation than criterion-proposal — and
the pilot cannot distinguish those two. That distinction is the obvious next study.

A caution against over-reading: the mining target is the dense score, and the dense
model is itself an imperfect .777-AUC instrument. Some of Δ is dense-model idiosyncrasy
that no articulation *should* capture. Δ remains an upper bound on taste, tightened
from +.0925 to +.0807, not a point estimate of it.

### (e) Recommended freeze parameters for confirmatory cells

| parameter | recommendation | basis |
|---|---|---|
| **ε** | keep .005 | worked; round 2 (+.0095) cleared it decisively while nulls sat at ±.0008 |
| **stopping rule** | keep **2 consecutive** sub-ε; do NOT relax to 1 | round 1's null would have been declared a bound that round 2 then moved |
| **B (cap)** | 5 is right; saturation arrived at 4 | curve flattened by r3-r4 on every diagnostic |
| **k_A / k_B** | keep 15 / 10 | audit separated the pools at 96-100% throughout |
| **proposal shape** | **do not mandate a composite quota**; record the count and assert it in code | the shape hypothesis died in round 3 (14/15 composite → +.0001) |
| **MONITOR** | define **inside dense-held-out rows** | pilot's 943-row contamination forced a 249-row honest set |
| **mining slice** | keep FIT+MINE ∩ dense-held-out | worked; \|M\|=995 supported four non-overlapping 60-row reads |
| **audit** | fresh Sonnet auditor per round + Opus arbiter on disputes + **≥1 planted probe pair per round** | 4.0/4.0/0.0/4.0% misrouting; probes caught every time |
| **collapse gate** | programmatic, pre-readout | one nuisance and one A criterion collapsed silently before the gate was enforced |
| **sign-check** | report-only for the pilot; **promote to a re-audit trigger** for confirmatory | 5 sign-contradicting criteria across the pilot, none re-examined |
| **anchors** | raise to **K ≥ 50 per class**; gate on scrambled, report pos/neg | K=12 cannot resolve the round-4 inversion |
| **discount estimator** | replace decile stratification with **matched sampling** once spurious-alone > ~.65 | at .713 the stratification approaches conditioning on the label |
| **GEPA** | required before any confirmatory Δ is quoted | pilot is pre-GEPA by design |
| **corpus hygiene** | dedup pass before use | near-duplicate abstract pair found in the round-3 slice |

---

## 6. Judge-call ledger

| round | prompts |
|---|---|
| 1 | 151,650 |
| 2 | 151,650 |
| 3 | 151,650 |
| 4 | 151,650 |
| **cumulative** | **606,600** |

Plus one non-judge GPU job (the 6,030-row same-rows dense rescore, round 1). All
criteria were scored each round; the collapse gate drops from the readout only.

GPU discipline across the pilot: GPU1 (v3 code-review training, PID 1809024) verified
alive and excluded at every launch; nothing was ever killed except my own two processes
once, wrapper shell first; every launch used the race-free retry loop with utilisation
sized from actual free memory.

---

## 7. Artifact locations

`methods/taste_decomposition/closure/` — per round r ∈ {1,2,3,4}:
`stage1_round{r}.py`, `round{r}_disagreement_slice.json`, `round{r}_track_{a,b}.json`,
`round{r}_proposals_blinded.json`, `round{r}_routing_{audit,final}.json`,
`score_round{r}_gemma.py`, `run_round{r}_when_free.sh`, `round{r}_scores.npz`,
`round{r}_score_report.json`, `stage4_round{r}.py`, `round{r}_results.json`,
`stage4_mechanism_r{r}.py`, `round{r}_mechanism.json`.

Shared: `closure_lib.py` (frozen Layer-1 protocol refit for fit-within-FIT+MINE),
`build_splits.py` + `peer_verdict_splits.json`, `rescore_dense_same_rows.py` +
`peer_verdict_dense_preds.csv` (freeze #2 same-rows T), `criterion_names_tagged.json`.

**`round4_results.json` carries the full five-state curve and is the canonical result.**

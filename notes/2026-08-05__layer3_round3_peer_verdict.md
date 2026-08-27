# Layer-3 articulation closure — ROUND 3, peer-review VERDICT (dual-track pilot)

Date: 2026-08-05. Status: **exploratory pilot, PRE-GEPA**.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` (Amendment 1 + Amendment 2).
Rounds 1-2: `notes/2026-08-05__layer3_round1_peer_verdict.md`,
`notes/2026-08-05__layer3_round2_peer_verdict.md`.
Code + artifacts: `methods/taste_decomposition/closure/`.

**Headline: round 3 is sub-ε (+.0001) but SATURATION IS STILL NOT DECLARED**, because
the rule requires two *consecutive* sub-ε rounds and round 2 broke the run. The
sub-ε flags now read **r1 yes, r2 no, r3 yes** → trailing run = 1. Round 4 decides.

Terminology per the spell-out rule: **V** = 17 programmatic surface features;
**A** = the articulated-criterion bank; **VA** = both concatenated; **lin**/**nl** =
frozen linear / gradient-boosted aggregation of the same matrix; **T** = dense
readout (Llama-3.1-8B LoRA on raw text); **Δ_beyond** = T − VA_nl; **Δ_r** = Δ_beyond
after r rounds; **ε** = .005 saturation threshold; **OOF** = out-of-fold;
**AUC** = area under the ROC curve; **GEPA** = the prompt-iteration pass required for
confirmatory phrasing (not applied — pilot is pre-GEPA).

---

## 1. PROTOCOL DEVIATION — report first

Amendment 2 requires proposal **shape held constant** from round 3 on, so that
consecutive-round comparisons are apples-to-apples. Round 2 was **10/15 composite**.
Round 3 was authored **14/15 composite**. The round-3 proposal file's own
`n_composite` field said 10 and was simply wrong; the error surfaced post hoc when
the composite-vs-simple sub-block split in the mechanism analysis returned n = 1 for
the non-composite arm.

I did not re-author: round-3 scores already existed, so re-authoring would violate
freeze-before-eval. Recorded in `round3_track_a.json` under `protocol_deviation`.

**This deviation is informative rather than merely regrettable.** Round 3 had *more*
composites than round 2 and produced **+.0001** against round 2's **+.0095**. So the
round-2 reading — "the interaction-shaped steer is what moved the curve" — does not
survive round 3. Composite shape alone does not drive closure gain.

---

## 2. What ran

Split, population, dense predictions unchanged from rounds 1-2, so the
round-over-round MONITOR gains stay comparable. Round 3 mined against the **round-2
bank** (V + 154 A + 14 round-1 + 14 round-2 criteria = 124 features; VA_nl OOF inside
FIT+MINE .7016). Rows read in rounds 1 *and* 2 were excluded (120 rows).

The mining-slice disagreement keeps narrowing — median |dense − VA| rank gap
**.589 → .470 → .386** across rounds. |M| = 995 throughout.

Corpus observation: two slice rows (R3-DL39/40) are near-duplicate abstracts of the
same paper with slightly different text, so the population contains at least one
near-duplicate pair. Not a blocker; worth a dedup pass before confirmatory use.

---

## 3. Proposals and the blind routing audit

Track A k = 15 (interaction-shaped, see the deviation above), Track B k = 10, all ten
on channels untouched by rounds 1-2. Two Track-B proposals were deliberate **paired
probes**: "density of percentage figures" (the shallow counterpart of round 1's
quantified-headline criterion) and "count-signalling of experimental scale" (the
shallow counterpart of round 1's evaluation-breadth criterion).

| | round 1 | round 2 | round 3 |
|---|---|---|---|
| proposals | 25 (15 A / 10 B) | 25 (15 A / 10 B) | 25 (15 A / 10 B) |
| **misrouting rate** | 4.0% | 4.0% | **0.0%** |
| disputes → arbiter | 1 | 1 | **0** |
| final routing | 14 A / 11 B | 14 A / 11 B | **15 A / 10 B** |

The fresh auditor agreed with the proposing track on all 25. Both paired probes were
routed to the nuisance set while their substantive originals had gone to the bank —
the audit distinguishes "count the percentages" from "is the headline anchored to a
named baseline", which is the discrimination the probes were built to test.

**The belief-change salvage rewrite worked.** Round 2's version was exiled because its
instruction stipulated that both contribution genres were legitimate, making it a
merit-neutral genre classifier. Reworded to score an epistemic act — name a belief the
field holds, then give evidence against it — it was routed into the bank unanimously.
That is the second arbiter diagnosis in a row to yield a working rewrite (round 1's
resource-magnitude → scale-free reusable-artifact was the first).

Label-blindness: 0 outcome-referring terms in any of the 25 name/instruction fields;
the slice carries no `judgement` column.

---

## 4. Scoring and instrument checks

Identical instrument to rounds 1-2 (Gemma-4-31B, offline-batch vLLM, spawn,
temperature 0, single-token 0-10, 5,000-char truncation). 151,650 prompts, GPU0
exclusive, 19 min. GPU1 was excluded throughout — the v3 code-review training
(PID 1809024) was verified alive at launch.

### Anchors (blinded, K = 12 per class)

| class | round 1 | round 2 | round 3 |
|---|---|---|---|
| positive | 3.11 | 2.84 | 3.30 |
| negative | 2.77 | 2.74 | 2.87 |
| scrambled | 1.06 | 0.94 | **0.12** |
| **coherent vs scrambled AUC** | .972 | 1.000 | **1.000 PASS** |
| positive vs negative AUC | .719 | .608 | **.736** |

Best anchor separation of the pilot on both readouts.

### Collapse check — **1 of 25 COLLAPSED, and gated out**

**P19 "Content-warning or ethics-disclaimer boilerplate"**: mean .02, modal fraction
1.00, 3 distinct values — 99.6% of abstracts score 0. This is a true corpus fact
(content warnings are vanishingly rare in ML abstracts), not a guided-decoding
failure. The prereg gates use on the collapse check, so it was **dropped from the
readout**; `stage4_round3.py` now enforces this gate programmatically and records the
dropped ids. It was a **B-routed nuisance**, so **Δ₃ is unaffected**; round 3
contributes 15 A and 9 B criteria to the banks.

Its prompts still count in the judge ledger — the judge ran before the gate applied.

Overall NA .006. Two A criteria are near-ceiling and worth watching rather than
gating: **"internal consistency"** (mean 9.34, sd 0.64) and **"claim scope matches
method scope"** (mean 9.00, sd 0.75). Both pass the collapse test (7 and 9 distinct
values, modal < .6) but have little headroom, which caps their discriminative power.

---

## 5. The closure curve

All four rounds recomputed in one pass, identical estimator, identical split
(`stage4_round3.py`).

| | round 0 | round 1 | round 2 | round 3 |
|---|---|---|---|---|
| features after screen | 96 | 110 | 124 | 139 |
| **VA_nl, MONITOR (n=1,192)** | **.6633** | **.6635** | **.6730** | **.6731** |
| VA_nl seed spread | .0030 | .0118 | .0060 | .0032 |
| VA_lin, MONITOR | .6597 | .6677 | .6667 | .6683 |
| Δ_interact = VA_nl − VA_lin | +.0035 | −.0042 | +.0063 | +.0048 |
| VA_nl, honest level (n=1,244) | .6844 | .6891 | .6929 | .6958 |
| **Δ, honest level (T = .7769)** | **+.0925** | **+.0878** | **+.0840** | **+.0811** |
| Δ, same-rows (n=249, T = .7838) | +.1363 | +.1485 | +.1309 | +.1292 |

### Round-over-round gains (group-level paired bootstrap, 2,000×)

| round | VA_nl gain on MONITOR | 95% CI | P(>0) | sub-ε (.005)? |
|---|---|---|---|---|
| r0 → r1 | +.0003 | [−.0132, +.0145] | .54 | **yes** |
| r1 → r2 | +.0095 | [+.0023, +.0170] | .996 | no |
| **r2 → r3** | **+.0001** | **[−.0091, +.0089]** | **.50** | **yes** |

**sub-ε flags: [yes, no, yes] → trailing consecutive run = 1. SATURATION NOT
DECLARED.** Under the two-consecutive rule the run was broken by round 2 and has
restarted; round 4 (of a hard cap of 5) decides.

The honest Δ level continues its monotone decline — **.0925 → .0878 → .0840 → .0811**
— a cumulative **−.0114** over three rounds and 43 added criteria, i.e. about
**−.0038 per round**. Δ levels remain protocol-specific and are not comparable to the
Layer-1 Δ_beyond of +.065 (Amendment 1); only these changes and the honest level are
quotable.

---

## 6. Mechanism — why round 3 added nothing

**The round-3 block is not inert but is fully redundant.** Alone on MONITOR it
reaches linear .6187 / nonlinear .6147 — comparable to round 1's block (.620) and
round 2's (.617 linear). Fifteen criteria that each carry signal added +.0001 once
the bank already contained 124 features.

**And it produced no interaction structure.** Round 2's block showed a large nonlinear
premium (linear .6166 → nonlinear .6504, +.034). Round 3's block shows **none**
(.6187 → .6147, −.004), despite being *more* composite-heavy. So "composite" as a
proposal instruction does not reliably produce interaction-shaped features: round 2's
composites happened to span complementary directions, round 3's did not.

**Mining still hits its label-blind target, monotonically:**

| ρ(VA_nl, dense) | round 0 | round 1 | round 2 | round 3 |
|---|---|---|---|---|
| MONITOR (n=1,192) | .488 | .522 | .531 | **.541** |
| same-rows (n=249) | .360 | .397 | .414 | **.426** |

Every round moves the articulated stack's ranking closer to the dense model's, and
the label AUC has stopped following. That gap — rank agreement still climbing while
Δ barely moves — is the pilot's central empirical pattern.

Per-criterion (all composite except P23):

| id | criterion | alone AUC | ρ vs dense |
|---|---|---|---|
| P12 | claim scope matches method scope | .5728 | **.268** |
| P01 | assumption load vs strength of conclusion | **.5849** | .263 |
| P21 | currency of comparison × margin | .5372 | .140 |
| P03 | internal consistency (opening vs closing) | .5522 | .135 |
| P07 | problem hard AND tractable | .5374 | .117 |
| P18 | reinterpretation × machinery unlocked | .5208 | .080 |
| P22 | structural property identified × exploited | .5281 | .079 |
| P05 | attribution of the gain | .5076 | .075 |
| P17 | overturns a stated prior belief (salvage rewrite) | .5277 | .057 |
| P13 | prior failure shown × targeted fix | .5158 | .052 |
| P25 | provable guarantee × practicality | .5216 | .050 |
| P04 | limitation-remedy correspondence | .5109 | .030 |
| P09 | efficiency gain × verified non-degradation | .4849 | .018 |
| P16 | negative space | .5036 | .009 |
| P23 | imported tool × justified analogy | .4950 | −.002 |

No criterion this round matches round 1's top (.621/.362) or round 2's (.609/.337).
The best available new criteria are getting weaker as the bank grows — the expected
signature of approaching saturation.

**Sign-contradicting criteria (report-only per Amendment 2):** P09 (efficiency gain
with verified non-degradation, .4849) and P23 (imported tool with justified analogy,
.4950) sit on the wrong side of .5 relative to their quality rationales. Both stay in
the bank. This makes three across the pilot, with round 2's "restraint" (.4583) the
strongest.

---

## 7. Track B — 31 declared nuisances, and the discount finally bites

| spurious-alone AUC | r1 (11 feat) | r2 (22 feat) | **r3 (31 feat)** |
|---|---|---|---|
| linear, MONITOR | .6040 | .6286 | **.7122** |
| HistGB, MONITOR | .5957 | .6303 | **.7062** |

Top channels:

| id | nuisance | alone AUC |
|---|---|---|
| **r3:P15** | **a public repository URL appears in the abstract** | **.5934** |
| r1:P02 | abstract length and verbosity | .5672 |
| r2:P11 | sentence complexity independent of length | .5631 |
| r2:P14 | non-idiomatic English / grammatical slips | .4384 |
| r1:P20 | dominant research topic | .5486 |
| r2:P22 | first-person narrative density | .5310 |
| r3:P06 | mathematical notation density | .5285 |
| r1:P14 | raw LaTeX residue | .5290 |
| r1:P16 | engineering/data resource magnitude | .5268 |
| r1:P24 | anonymised-repo / on-acceptance boilerplate | .4778 |

**The strongest single nuisance of the whole pilot is a public repository URL
(.5934)** — and it is the exact complement of round 1's anonymised-repo /
release-on-acceptance boilerplate, which is *anti*-predictive at .4778. The
artifact-availability channel is sharply **bipolar**: a live public link is among the
most predictive surface features in the corpus, while a placeholder or a promise runs
the other way. Since the existing 154-criterion bank scores availability statements
without distinguishing the two, part of the bank is wired to a channel whose sign
flips on a detail it cannot see. That is a concrete, actionable defect.

### Discounted readouts

| set | n | strata | pooled T | pooled VA | pooled Δ | T_adj | VA_adj | **Δ_adj** |
|---|---|---|---|---|---|---|---|---|
| dense-held-out (honest) | 1,244 | q10 joint B | .7769 | .6958 | +.0811 | .7110 | .5965 | **+.1146** |
| MONITOR ∩ held-out | 249 | q5 joint B | .7838 | .6546 | +.1292 | .7330 | .5777 | **+.1552** |

For the first time the discount moves Δ materially — and it moves it **up**, by
+.034. Stratifying on the 31-nuisance joint model costs VA (−.099) far more than it
costs T (−.066): the articulated stack's predictive power leans on nuisance channels
more heavily than the dense model's does.

**Read this cautiously.** At spurious-alone .712 the joint-B score is now nearly as
predictive as T itself (.777), so conditioning on its deciles is close to
conditioning on the label. Both T and VA fall steeply, within-stratum label variance
shrinks, and the stratified AUC becomes attenuated and noisy. The defensible claim is
the negative one that has held all three rounds: **spurious discounting does not
shrink the residual.** The apparent +.034 increase should not be quoted as an effect
size until a matched-sampling check replaces decile stratification at this strength.

---

## 8. Judge-call ledger

| round | prompts |
|---|---|
| round 1 | 151,650 |
| round 2 | 151,650 |
| round 3 | 151,650 (150,750 population + 900 anchors) |
| **cumulative** | **454,950** |

All 25 round-3 criteria were scored; the collapse gate removed P19 from the readout
afterwards, so its prompts still count.

---

## 9. Reading

1. **Round 3 is sub-ε, but saturation still requires round 4.** The rule is working as
   designed: it refused to stop at round 1's null, and it refuses to stop now on a
   run of one.
2. **The composite explanation for round 2 is dead.** Round 3 was more composite-heavy
   and produced 1/100th the gain, with no nonlinear premium in its block. Whatever
   moved round 2 was the *content* of those particular relations, not their syntactic
   shape. This is a genuine correction to the round-2 note's reading, and it only
   surfaced because the deviation was counted rather than assumed.
3. **New criteria are getting weaker.** Best-per-round dense alignment ρ: .362 → .337
   → .268. Best alone-AUC: .621 → .609 → .585. The proposer is running out of
   articulable structure that the bank does not already carry.
4. **Rank agreement keeps rising while Δ stalls** (ρ .488 → .541 across rounds, Δ
   −.0114 total). Articulation keeps capturing more of *what the dense model does*
   without capturing more of *what predicts the outcome*. That dissociation is the
   most interesting thing the pilot has produced and deserves its own analysis.
5. **Δ has fallen .0925 → .0811 in three rounds** (−.0038/round). At that rate the
   B = 5 cap will end the pilot with roughly +.07 of residual intact.
6. **Spurious discounting still does not shrink Δ** — now across 31 nuisances.

### For round 4 (the decider)
- Hold shape constant **and verify the count programmatically before scoring** — the
  deviation this round was preventable by one assertion.
- Consider mining against the *dense-vs-VA rank residual* directly rather than more
  criteria of the same kind, given point 4.
- Add a dedup pass (near-duplicate abstracts found in the slice).

### Freeze-checklist additions from round 3
- **Assert proposal-shape composition in code before scoring**, not in prose.
- **Collapse gating must be programmatic** (now implemented) — a nuisance collapsed
  silently this round and would have entered the discount set unnoticed.
- **Decile stratification breaks down once the nuisance model approaches T**; specify
  matched sampling for high-AUC nuisance sets.
- **Split the bank's availability-statement criteria** into live-link vs
  placeholder/promise — one channel, two opposite signs.
- Blind-audit misrouting across three independent auditors: 4.0%, 4.0%, 0.0%.

---

## 10. Artifact locations

All under `methods/taste_decomposition/closure/`:

| file | contents |
|---|---|
| `stage1_round3.py`, `round3_oof_fitmine.json` | round-2-bank OOF inside FIT+MINE |
| `round3_disagreement_slice.json` | the 60 new rows read (label-blind) |
| `round3_track_a.json` | proposals + composite flags + **`protocol_deviation`** |
| `round3_track_b.json` | 10 fresh nuisance channels incl. the two paired probes |
| `round3_proposals_blinded.json` | exactly what the auditor saw |
| `round3_routing_audit.json`, `round3_routing_final.json` | verdicts, 0 disputes, final routing |
| `score_round3_gemma.py`, `run_round3_when_free.sh` | scoring + race-free launcher |
| `round3_scores.npz`, `round3_score_report.json` | scores, anchors, collapse flags |
| `stage4_round3.py`, `round3_results.json` | 4-round curve, collapse gate, Track-B discount |
| `stage4_mechanism_r3.py`, `round3_mechanism.json` | block-alone, composite split, dense alignment |

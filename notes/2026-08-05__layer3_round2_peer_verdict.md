# Layer-3 articulation closure — ROUND 2, peer-review VERDICT (dual-track pilot)

Date: 2026-08-05. Status: **exploratory pilot, PRE-GEPA**.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` (incl. the 2026-08-05 amendment).
Round 1: `notes/2026-08-05__layer3_round1_peer_verdict.md`.
Code + artifacts: `methods/taste_decomposition/closure/`.

**Headline: SATURATION IS NOT DECLARED.** Round 2's MONITOR VA_nl gain is
**+.0095, 95% CI [+.0023, +.0170], P(gain>0) = .996** — above ε = .005 and
significantly positive. The saturation rule needs two *consecutive* sub-ε rounds;
round 1 was sub-ε (+.0003), round 2 is not, so the counter resets to 0 and the
pilot continues to round 3 (hard cap B = 5).

Terminology unpacked per the spell-out rule: **V** = 17 programmatic surface
features; **A** = the articulated-criterion bank; **VA** = both concatenated;
**lin** / **nl** = the frozen linear / gradient-boosted aggregation of the same
matrix; **T** = dense-standard readout (Llama-3.1-8B LoRA on raw text);
**Δ_beyond** = T − VA_nl; **Δ_r** = Δ_beyond after r mining rounds; **OOF** =
out-of-fold; **AUC** = area under the ROC curve; **ε** = the saturation threshold;
**GEPA** = the prompt-iteration pass required for confirmatory phrasing (not applied
here — the pilot is pre-GEPA by design).

---

## 1. What ran

Split, population and dense predictions are **unchanged from round 1**, so the
round-over-round MONITOR VA_nl gain — the only quantity the stopping rule is defined
on — stays apples-to-apples. (The amendment's "MONITOR inside dense-held-out" rule
governs *confirmatory* cells; for the pilot the honest level is reported on the 1,244
dense-held-out rows instead.) Population CSV verified byte-identical local↔remote
(sha256 `1a5042f697bd4ac2`), so round-2 score rows align with rounds 0/1 by
construction.

Round 2 mined against the **round-1 bank** (V + 154 A + the 14 round-1 criteria;
VA_nl OOF inside FIT+MINE = .7068). The median dense-vs-VA rank gap in the mining
slice narrowed **.589 → .470**, and the 60 rows read in round 1 were excluded so
round 2 read new text. |M| = 995, unchanged.

---

## 2. Proposals, and the allowed interaction-shaped steer

Per the amendment, round 2's Track-A instruction was explicitly steered toward
**composite / interaction-shaped** criteria — recorded as an allowed steer, not a
silent change — because round 1's single composite (ambition × evidence-specificity)
was its strongest new feature by ~2×. 10 of the 15 Track-A proposals are composite
("X TOGETHER WITH Y", ratios, orderings).

Track B ran a fresh sweep over channels round 1 left untested: inline author-year
citations, first-person density, sentence-level syntactic complexity *independent of
total length*, typographic emphasis, rhetorical questions, non-idiomatic English,
acronym density, priority/"first" claims, the privacy-security-adversarial subfield,
and hedging density.

Files: `round2_track_a.json`, `round2_track_b.json`, `round2_proposals_blinded.json`.

### Blind routing audit (fresh auditor) + frontier arbiter

| quantity | round 1 | round 2 |
|---|---|---|
| proposals | 25 (15 A / 10 B) | 25 (15 A / 10 B) |
| **misrouting rate** | **4.0%** | **4.0%** |
| final routing | 14 A / 11 B | **14 A / 11 B** |

All 10 Track-B proposals were classified incidental again, and 14/15 Track-A
quality-relevant — the two authoring mindsets remain separable by an independent
judge at 96%.

The one dispute: **"Belief-change — the output is a changed understanding rather than
a delivered artifact"** (proposed A). The auditor called it incidental with *high*
confidence on a textual point: my own instruction said "Both are legitimate
contributions; score only which one the abstract is offering", which makes a judge
obeying it return a **genre label**, not a merit. The Opus arbiter **upheld the
auditor**: a value-neutral genre axis is not relevantly different from the
topic/subfield markers both parties already exiled — it partitions items on an axis
explicitly stipulated to be merit-neutral, draws its lift from corpus composition,
and would flip sign under a different venue mix. Re-routed to the nuisance set. Its
salvage rewrite is queued for round 3 as a *fresh* Track-A proposal rather than
swapped in mid-round (freeze-before-eval).

That the criterion then scored **alone-AUC .5162** as a nuisance — real but small
signal, on the value-neutral axis — is consistent with the arbiter's reasoning.

Label-blindness audit: 0 outcome-referring terms across all 25 name/instruction
fields; the disagreement slice carries no `judgement` column.

---

## 3. Scoring and instrument checks

`score_round2_gemma.py`, Gemma-4-31B, offline-batch vLLM, spawn, temperature 0,
single-token 0-10 readout, 5,000-char truncation — identical instrument to round 1.
151,650 prompts (150,750 population + 900 anchors).

### Anchors (blinded, K = 12 per class)

| class | round 1 | round 2 |
|---|---|---|
| positive | 3.11 | 2.84 |
| negative | 2.77 | 2.74 |
| scrambled | 1.06 | 0.94 |
| **coherent vs scrambled AUC** | **.972 PASS** | **1.000 PASS** |
| positive vs negative AUC | .719 | .608 |

The round-2 criteria separate coherent from scrambled text perfectly. Positive-vs-
negative is lower than round 1, which is expected and not a fault: round-2 criteria
are deliberately about *reasoning shape* (coupling, subsumption, portability) rather
than about visible merit markers, so they discriminate accept/reject less directly
even while adding label signal to the stack.

### Collapse check — **0 of 25 collapsed**, overall NA .031

Two criteria carry high but *meaningful* NA rather than degeneracy:
**Reusable artifact** NA .676 (most abstracts leave no artifact — the instruction
says score what is left behind, so NA is the correct answer) and **Simplicity
relative to difficulty** NA .084. Sparse-but-not-collapsed: cost-with-gain (.92
modal), rhetorical questions (.97), named phenomenon (.92), priority claims (.89).
All have 7-11 distinct values.

---

## 4. The closure curve

Fit inside FIT+MINE, read on MONITOR; VA_nl = mean over seeds {0,1,2}. All three
rounds recomputed in **one pass with the identical estimator** (`stage4_round2.py`)
so the curve is internally consistent.

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| features after screen | 96 | 110 | 124 |
| **VA_nl, MONITOR (n=1,192)** | **.6633** | **.6635** | **.6730** |
| VA_nl seed spread | .0030 | .0118 | .0060 |
| VA_lin, MONITOR | .6597 | .6677 | .6667 |
| Δ_interact = VA_nl − VA_lin | +.0035 | −.0042 | **+.0063** |
| VA_nl, same-rows (n=249) | .6476 | .6353 | .6530 |
| **Δ, same-rows (T = .7838)** | +.1363 | +.1485 | **+.1309** |
| VA_nl, honest level (n=1,244) | .6844 | .6891 | .6929 |
| **Δ, honest level (T = .7769)** | **+.0925** | **+.0878** | **+.0840** |

### Round-over-round gains (group-level paired bootstrap, 2,000×)

| round | VA_nl gain on MONITOR | 95% CI | P(>0) | sub-ε (.005)? |
|---|---|---|---|---|
| r0 → r1 | +.0003 | [−.0132, +.0145] | .54 | **yes** |
| r0 → r2 (round 2) | **+.0095** | **[+.0023, +.0170]** | **.996** | **no** |

**Consecutive sub-ε rounds: 1. SATURATION NOT DECLARED.**

The honest Δ level falls monotonically — **.0925 → .0878 → .0840**, a cumulative
−.0085 over two rounds. The same-rows (n=249) column is much noisier (seed spread
alone is .02-.03 there) and moves non-monotonically; it should not be read as a
trend.

Per the amendment, these Δ *levels* are protocol-specific and are **not** comparable
to the Layer-1 Δ_beyond of +.065; only the round-over-round changes and the honest
level are quotable.

---

## 5. Why round 2 moved when round 1 did not — the mechanism

This is the informative part, and it vindicates the steer.

**The gain is interaction-shaped.** Round 2 is the only round where VA_lin *fell*
(−.0010) while VA_nl *rose* (+.0095). Δ_interact goes +.0035 → −.0042 → **+.0063**.
The composite criteria contribute signal a linear aggregation structurally cannot
use — exactly what "interaction-shaped" predicts, and a clean internal validation
that the steer changed the *kind* of feature being mined, not just the count.

**The round-2 A block alone** (14 criteria, no bank): linear .6166, **nonlinear
.6504** — a +.034 nonlinear premium, versus round 1's A block which was flat
(.6198 linear / .6179 nonlinear).

**Composite vs non-composite sub-blocks** (the steer test):

| sub-block | k | linear | nonlinear |
|---|---|---|---|
| composite | 10 | .6139 | .6107 |
| non-composite | 4 | .5627 | .6165 |
| all 14 together | 14 | .6166 | **.6504** |

Read carefully: the composite block carries much more *linear* signal (.614 vs .563),
but neither sub-block alone reaches what the two reach together (.650). The honest
reading is **complementarity**, not "composites are simply better" — the steer's
payoff comes from composites and simple criteria spanning different directions. With
only 4 non-composite criteria this is a descriptive observation, not a test.

**Mining kept hitting its label-blind target.** Spearman ρ between VA_nl and the dense
score rises monotonically across rounds:

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| ρ(VA_nl, dense), MONITOR | .488 | .522 | **.531** |
| ρ(VA_nl, dense), same-rows | .360 | .397 | **.414** |

Round-2 A block alone vs dense: ρ = .434.

**Per-criterion** (MONITOR; C = composite):

| id | C | criterion | alone AUC | ρ vs dense |
|---|---|---|---|---|
| P10 | C | generality of statement **together with** concreteness of instantiation | **.6093** | **.337** |
| P17 | – | reusable artifact (scale-free salvage rewrite) | .5614 | .229 |
| P08 | C | precise delta — prior work characterised checkably | .5540 | .170 |
| P21 | C | insight portability | .5436 | .134 |
| P19 | C | simplicity relative to problem difficulty | .5151 | .085 |
| P09 | C | derivation coupling | .5351 | .077 |
| P18 | – | mechanistic explanation of an observed fact | .5128 | .056 |
| P16 | – | subsumption | .5194 | .054 |
| P06 | C | theory–experiment coupling | .5251 | .045 |
| P04 | C | limiting result with scope conditions | .5194 | .038 |
| P20 | – | names a phenomenon | .5051 | .032 |
| P03 | C | cost together with gain | .5044 | .027 |
| P01 | C | discriminative evaluation | .5136 | .004 |
| P13 | C | **restraint** — language calibrated below results | **.4583** | **−.169** |

Two results worth carrying forward:

1. **The top criterion is again a composite** (P10, .6093/.337), closely mirroring
   round 1's P05 (.621/.362). Two independent rounds now put a "general statement ×
   concrete instantiation"-shaped composite at the top.
2. **Restraint is inverted**: understated language predicts *lower* dense score
   (alone AUC .458, ρ = −.169). The dense model rewards confident phrasing. The blind
   auditor routed this to the bank as quality-relevant, and by the letter of the
   routing protocol it stays there — but it is behaving like a register channel with
   the sign the promotional-language nuisance would predict. **Flagged for the freeze
   checklist**: a criterion whose sign contradicts its quality rationale should
   trigger re-audit, and the current protocol has no such trigger.
3. **The arbiter's salvage rewrite worked.** Round-1 A05 (resource *magnitude*) was
   exiled as an institutional-circumstances signal at alone-AUC .527. Its scale-free
   rewrite (artifact *reusability*) was audited into the bank and scores .5614 / ρ
   .229 — second-strongest in the round. Rewriting a nuisance into a scale-free
   quality criterion recovered real signal.

---

## 6. Track B — 22 declared nuisances

| spurious-alone AUC | round 1 (11 features) | round 2 (22 features) |
|---|---|---|
| linear, MONITOR | .6040 | **.6286** |
| HistGB, MONITOR | .5957 | **.6303** |

Strongest channels (deviation from .5 in either direction):

| id | nuisance | alone AUC |
|---|---|---|
| r1:P02 | abstract length and verbosity | .5672 |
| **r2:P11** | **sentence complexity, independent of total length** | **.5631** |
| **r2:P14** | **non-idiomatic English / grammatical slips** | **.4384** |
| r1:P20 | dominant research topic | .5486 |
| r2:P22 | first-person narrative density | .5310 |
| r1:P14 | raw LaTeX residue | .5290 |
| r1:P16 | engineering/data resource magnitude | .5268 |
| r2:P23 | typographic emphasis | .5242 |
| r1:P24 | anonymised-repo boilerplate | .4778 |
| r2:P05 | hedging density | .4692 |

Round 2's fresh sweep found **two channels stronger than anything round 1 found
except raw length**. Sentence-level complexity (.5631) nearly matches total length
(.5672) despite being explicitly instructed to be judged independently of it —
suggesting the "length" channel is at least partly a *syntactic-density* channel.
And non-idiomatic English is the second-strongest channel overall at .4384, i.e.
**strongly anti-predictive**: surface fluency tracks the label. This was the round's
deliberate hard audit case — fluency is arguably quality-adjacent — and the blind
auditor routed it to the nuisance set. Given its strength, whether writing fluency is
a nuisance or a merit is now a substantive question for the freeze, not a formality.

### Discounted readouts (threshold-free, n-weighted within-stratum AUCs)

| set | n | strata | pooled T | pooled VA | pooled Δ | T_adj | VA_adj | **Δ_adj** |
|---|---|---|---|---|---|---|---|---|
| dense-held-out (honest) | 1,244 | q10 on joint B score | .7769 | .6929 | **+.0840** | .7456 | .6635 | **+.0821** |
| MONITOR ∩ held-out | 249 | q5 on joint B score | .7838 | .6530 | +.1309 | .7722 | .6423 | +.1298 |

Stratifying on the joint 22-feature spurious model moves Δ by **−.0019** on the
honest set. Doubling the nuisance set and raising spurious-alone AUC from .604 to
.630 still does not touch the residual: T and VA fall almost exactly together
(T −.031, VA −.029). **The residual continues to survive spurious discounting.**

---

## 7. Judge-call ledger

| round | prompts |
|---|---|
| round 1 (population + anchors) | 151,650 |
| round 2 (150,750 + 900) | 151,650 |
| **cumulative** | **303,300** |

Plus one non-judge GPU job (the 6,030-row same-rows dense rescore, round 1).

---

## 8. Reading, and what happens next

1. **The closure is moving again, and saturation is not declared.** +.0095
   [+.0023, +.0170] is a real gain, four times round 1's and comfortably above ε.
   The round-1 flattening was **premature**: it reflected the *shape* of the criteria
   proposed, not exhaustion of the articulable space. Had the pilot stopped at
   round 1's null, it would have declared a taste bound that round 2 then moved.
   That is the most important methodological finding of the pilot so far, and it is
   an argument for the prereg's two-consecutive-round rule over a one-round rule.
2. **The articulable residual has interaction structure.** Round 2 is the first round
   with a positive Δ_interact (+.0063) and a nonlinear premium in the new block
   (+.034). What the bank was missing is not more single properties but *relations
   between* properties — which is also why Layer 1 found Δ_interact ≈ 0 over the
   original bank: those 154 criteria had no interaction structure to find.
3. **But the honest Δ is barely moving**: .0925 → .0878 → .0840, −.0085 over two
   rounds and 28 criteria. Extrapolating naively, closing +.084 at this rate would
   take ~20 more rounds — far beyond the B = 5 cap. The defensible statement remains
   that the residual is large relative to what active articulation removes per round.
4. **Spurious discounting still does nothing** (Δ_adj −.0019 with 22 nuisances at
   .630 alone-AUC). Both candidate deflations of Δ have now failed twice.

### For round 3
- Continue (counter = 1 of 2 needed; cap 5).
- Keep the composite steer; add the belief-change salvage rewrite as a fresh Track-A
  proposal.
- Re-audit **restraint** (sign contradicts its rationale) and adjudicate **writing
  fluency** as nuisance-vs-merit; both belong on the freeze checklist.
- Track B: probe whether "length" is really syntactic density, now that the two
  separate cleanly.

### Carry into the freeze checklist
- **A one-round null is not saturation** — the two-consecutive rule earned its keep.
- **Proposal shape is a hidden design parameter.** Round 1 and round 2 differed
  almost entirely in criterion *shape*, and that produced a 30× difference in gain.
  Confirmatory rounds must fix the composite/simple mix in advance or the closure
  curve is not comparable across cells.
- **Add a sign-check trigger**: any A-routed criterion whose alone-AUC lands on the
  opposite side of .5 from its rationale should be re-audited before it is quoted.
- Blind-audit misrouting is stable at 4.0% across two independent auditors.

---

## 9. Artifact locations

All under `methods/taste_decomposition/closure/`:

| file | contents |
|---|---|
| `stage1_round2.py`, `round2_oof_fitmine.{json,npz}` | round-1-bank OOF inside FIT+MINE |
| `round2_disagreement_slice.json` | the 60 new rows read (label-blind) |
| `round2_track_a.json`, `round2_track_b.json` | proposals with rationales + composite flags |
| `round2_proposals_blinded.json` | exactly what the auditor saw |
| `round2_routing_audit.json`, `round2_routing_final.json` | verdicts, arbiter ruling, final routing |
| `run_round2_when_free.sh`, `round2_launcher.log` (sk3) | race-free launcher, util auto-sized to free memory |
| `score_round2_gemma.py`, `round2_scores.npz`, `round2_score_report.json` | scoring + anchors + collapse check |
| `stage4_round2.py`, `round2_results.json` | full curve r0/r1/r2 + Track-B discounting |
| `stage4_mechanism_r2.py`, `round2_mechanism.json` | A-block-alone, composite vs simple, dense alignment |
| `criterion_names_tagged.json` | round-tagged id→name map (r1/r2 blind ids both run P01-P25) |

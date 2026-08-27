# Layer-3 articulation closure — ROUND 1, peer-review VERDICT (dual-track pilot)

Date: 2026-08-05. Status: **exploratory pilot, PRE-GEPA** (the prereg declares the peer
verdict cell exploratory; confirmatory cells wait on the freeze checklist).
Prereg: `notes/2026-08-05__layer3-closure-prereg.md`. Parent design:
`notes/2026-08-05__taste-decomposition-design.md` §3.
Code + artifacts: `methods/taste_decomposition/closure/`.

Terminology unpacked on first use, per the spell-out rule.
**V** = the 17 cheap programmatic surface features; **A** = the articulated-criterion
block (the 154-criterion Gemma-judged bank); **VA** = V and A concatenated;
**lin** = the frozen linear aggregation (StandardScaler + LogisticRegression C=1);
**nl** = the frozen nonlinear aggregation (HistGradientBoostingClassifier, the Layer-1
grid); **T** = the dense-standard readout (Llama-3.1-8B LoRA on raw abstract text);
**Δ_beyond** = T − VA_nl, the only quantity eligible to be called taste;
**Δ_r** = Δ_beyond after r rounds of Track-A mining (the closure curve);
**OOF** = out-of-fold; **AUC** = area under the ROC curve;
**FIT+MINE / MONITOR** = the 80/20 stable-hash split; **M** = the mining slice;
**GEPA** = the prompt-iteration procedure the A-bank standard requires for
confirmatory phrasing (not applied here — round 1 is pre-GEPA by design).

---

## 0. What ran

| stage | where | outcome |
|---|---|---|
| 1. splits + same-rows dense rescore | local CPU + sk3 GPU0 | done, \|M\| = 995 |
| 2. dual proposals (15 A + 10 B) + blind routing audit + frontier arbiter | local | done, misrouting 1/25 |
| 3. Gemma-4-31B scoring, 25 criteria × 6,030 rows + anchors | sk3 GPU0, offline batch vLLM | done, 16 min, 151,650 prompts |
| 4. round-1 readout + Track-B discounting + mechanism diagnostics | local CPU | done |

GPU discipline: every sk3 job ran on **GPU0 only**, launched after `nvidia-smi` showed
it at 0 MiB. GPU1 (v3 code training) was never touched. Nothing was killed. No source
data was modified.

---

## 1. Splits (prereg: stable sha256 on the group key, threshold .80)

Population = the 6,030-row peer-verdict A/V evaluation population (`vat_3y`), the
identical row set and row order as `layer1_stack.py::load_cell("verdict")`.
Group key = `ntitle`. Split map: `closure/peer_verdict_splits.json`.

| set | rows | groups | positive rate |
|---|---|---|---|
| population | 6,030 | 5,999 | .4982 |
| FIT+MINE | 4,838 | 4,813 | .4950 |
| MONITOR | 1,192 | 1,186 | .5109 |
| dense-held-out ∩ population | 1,244 | — | — |
| **mining slice M = FIT+MINE ∩ dense-held-out** | **995** | — | — |
| MONITOR ∩ dense-held-out (honest same-rows set) | 249 | — | — |
| MONITOR ∩ dense-TRAIN (in-sample for T) | 943 | — | — |

### 1b. Same-rows T rescore (design freeze change #2) — a deliverable in its own right

No per-row eval predictions had been saved by the dense chain, so the frozen
`best_model` was rescored over all 6,030 population rows
(`closure/rescore_dense_same_rows.py`, sk3 GPU0, ~4 min).
Per-row probabilities: `closure/peer_verdict_dense_preds.csv`.

| T readout | n | AUC |
|---|---|---|
| registry (dense model's own eval split) | 2,808 | **.753** |
| same-rows, all population — **in-sample-contaminated** | 6,030 | .853 |
| same-rows, dense-held-out rows | 1,244 | **.777** |
| same-rows, MONITOR ∩ dense-held-out | 249 | **.784** |
| same-rows, all MONITOR — **contaminated** | 1,192 | .857 |
| dense eval / dense test, population rows only | 603 / 641 | .793 / .762 |

**Landmine recorded.** 943 of the 1,192 MONITOR rows sat in the dense model's TRAIN
split, so "T on MONITOR" is .857 only because the dense model has memorised most of
those rows. Every table below reports the honest same-rows figure (249 rows) as
primary and prints the contaminated one only for transparency. This is a structural
consequence of MONITOR being a fresh 80/20 hash over a population that the dense
model split independently — worth fixing in the confirmatory design by intersecting
the two split systems up front.

---

## 2. Mining slice and the disagreement read

VA_nl was refit **inside FIT+MINE only** (grouped OOF, frozen Layer-1 spec, seeds
0/1/2; the pre-existing `peer_verdict_va_nl_oof_seed0.npy` was fit on the full
population and would have straddled MONITOR, so it was not reused).
Column screening and imputation medians are learned on FIT+MINE only.
Internal check: VA_lin OOF .6969, VA_nl OOF .6917 within FIT+MINE — consistent with
the Layer-1 pilot's population-level .6896/.6876.

The disagreement slice = top-|dense percentile − VA_nl percentile| rows within M,
30 per direction, 60 read (`closure/round1_disagreement_slice.json`; the file carries
text, dense probability and VA_nl prediction and **no `judgement` column** — the
proposer context never loaded y).

What the two directions look like, descriptively:

- **dense-high / VA-low**: work with a stated insight and a mechanism for it, a
  conceptual recast of the problem, a numeric headline against a named baseline,
  wide and enumerated evaluation, and often a surprising or self-undercutting result.
- **dense-low / VA-high**: competent, smoothly written, generic — a proposal followed
  by "extensive experiments demonstrate superior performance", with the bank's
  availability-statement criteria firing on anonymised-repository links and
  release-on-acceptance promises.

That second pattern is the Layer-1 SHAP finding made visible: the top interaction pair
in the Layer-1 screen was `v_kw_code` × the availability-statement rubric, and here it
shows up as the mechanism by which VA_nl *over*-scores a whole class of abstracts.

---

## 3. Proposals and the blind routing audit

Track A (15) and Track B (10) were authored in separate passes, Track B with an
explicit "predictive but not quality" instruction.
Files: `closure/round1_track_a.json`, `closure/round1_track_b.json`.
Blinded pool (stable sha256 order, provenance and rationales stripped):
`closure/round1_proposals_blinded.json`.

The audit ran per prereg step 4: an independent Sonnet judge saw **only** the blinded
file, classified each criterion quality-relevant vs incidental, and the single
disagreement with the proposing track went to a frontier (Opus) arbiter.
`self_audit = false`. Result: `closure/round1_routing_final.json`.

| quantity | value |
|---|---|
| proposals | 25 (15 A, 10 B) |
| **misrouting rate** | **1 / 25 = 4.0%** |
| final routing | 14 A (join the bank), 11 B (declared nuisances) |

The one misroute: **A05 "Magnitude of the engineering or data resource the work
required"** was proposed to Track A and classified *incidental*. The arbiter upheld
the auditor — it is collinear with the already-declared industrial-compute nuisance on
its own exemplars (a 340M-image corpus, 7B/13B/33B models), and it scores expended
input, i.e. the authors' circumstances, rather than a property of the contribution.
It was re-routed to B and is scored as a nuisance. The arbiter's salvage rewrite
(score artifact **reusability** — novelty, documentation, fitness for reuse —
explicitly scale-invariant) is queued for round 2 and was **not** scored this round.

All 10 Track-B proposals were classified incidental, and 14 of 15 Track-A proposals
quality-relevant: the two mindsets produced pools an independent judge can tell apart
almost perfectly. That is a positive control on the dual design, and it means the
Track-A gain reported below is not an artifact of spurious features smuggled into the
bank.

---

## 4. Scoring (Stage 3) — instrument checks

`closure/score_round1_gemma.py`, Gemma-4-31B, offline-batch vLLM (`llm.chat`, never an
HTTP server), spawn start method, `CUDA_VISIBLE_DEVICES=0`, util .85, max_model_len
4096, prefix caching on, temperature 0, single-token readout, abstracts truncated to
5,000 chars — the `vat_3y/score_va_gemma_3y.py` recipe. One deliberate change: the
round-1 criteria are authored on a **0-10** scale rather than the bank's 0/0.5/1/NA,
which gives more resolution and makes collapse easier to detect.
Both splits scored; MONITOR rows were scored but never read by a proposer.

- 151,650 prompts (6,030 rows × 25 criteria = 150,750, plus 36 anchor rows × 25 = 900)
- wall clock 16 min 12 s on one B200

### Blinded anchor battery (pos / neg / scrambled, K = 12 per class)

| anchor class | mean item score | sd |
|---|---|---|
| positive | 3.11 | 0.66 |
| negative | 2.77 | 0.96 |
| scrambled | 1.06 | 0.59 |

- coherent vs scrambled AUC **.972** → **PASS** (the instrument is reading the text)
- positive vs negative AUC .719 — in line with what the dense model gets on these rows
  (.777-.784), i.e. the ceiling on this label is genuinely low, not an instrument fault

### Guided-collapse check — **0 of 25 criteria collapsed**

Overall NA rate .014. Every criterion has ≥5 distinct values and a modal fraction
below .96. Per-criterion distributions (`closure/round1_score_report.json`):

| id | route | criterion | NA | mean | std | modal frac | distinct |
|---|---|---|---|---|---|---|---|
| P01 | B | Applied-domain framing outside core machine learning | .000 | 1.24 | 2.68 | .77 | 11 |
| P02 | B | Abstract length and verbosity | .000 | 5.38 | 1.21 | .34 | 9 |
| P03 | A | Explanatory mechanism — says WHY the approach should work | .000 | 4.36 | 2.13 | .28 | 11 |
| P04 | A | States the costs, limitations, or failure conditions of its own method | .003 | 0.34 | 1.34 | .92 | 10 |
| P05 | A | Ambition of the claim together with the specificity of the evidence | .000 | 6.25 | 1.43 | .37 | 9 |
| P06 | A | Theoretical claims tied to the proposed method with stated scope | .034 | 2.57 | 3.37 | .56 | 11 |
| P07 | A | Reframing — recasts the problem in a different formalism | .000 | 4.06 | 2.74 | .23 | 9 |
| P08 | B | Branded method name or acronym | .000 | 5.50 | 4.78 | .48 | 11 |
| P09 | A | Motivation explains why the problem is hard | .000 | 4.38 | 2.82 | .24 | 11 |
| P10 | A | Breadth and diversity of the empirical evaluation enumerated | .000 | 2.86 | 1.77 | .34 | 11 |
| P11 | A | Technical specificity — objects and operations named precisely | .000 | 4.55 | 1.70 | .30 | 10 |
| P12 | A | Contributions are separable components each with a stated role | .172 | 6.51 | 2.15 | .39 | 10 |
| P13 | B | Enumerated list formatting inside the abstract | .000 | 0.97 | 2.60 | .85 | 8 |
| P14 | B | Raw LaTeX or markup residue left in the abstract | .000 | 1.72 | 3.60 | .78 | 8 |
| P15 | A | Diagnoses a specific failure mode of prior work with evidence | .000 | 3.05 | 2.51 | .26 | 9 |
| P16 | B | Magnitude of the engineering or data resource required | .004 | 2.50 | 1.41 | .45 | 11 |
| P17 | A | Reports a surprising, negative, or self-undercutting finding | .000 | 1.13 | 2.54 | .81 | 11 |
| P18 | B | Density of promotional superlatives and hype vocabulary | .000 | 2.85 | 1.33 | .35 | 10 |
| P19 | A | Generality demonstrated beyond the primary setting | .000 | 3.02 | 2.79 | .34 | 11 |
| P20 | B | Topic sits in a currently dominant research area | .000 | 5.71 | 3.29 | .24 | 10 |
| P21 | A | Quantified headline result against a named baseline | .000 | 1.39 | 3.04 | .78 | 11 |
| P22 | B | Famous benchmark and dataset name-dropping | .000 | 1.11 | 2.37 | .76 | 11 |
| P23 | A | Human or expert judgment used as evaluation evidence | .142 | 0.51 | 2.06 | .94 | 8 |
| P24 | B | Anonymized-repository or release-on-acceptance boilerplate | .000 | 0.44 | 2.01 | .95 | 5 |
| P25 | B | Signals of industrial-scale compute or frontier models | .000 | 0.37 | 1.30 | .90 | 11 |

Four criteria are **sparse but not collapsed** (P04, P23, P24, P25 at modal fraction
.90-.95): they name genuinely rare properties — stated costs, human evaluation,
anonymised-repo boilerplate, frontier-compute tells — and each still has 5-11 distinct
values. Sparse zero-inflated features are the reason the Track-B stratification below
degenerates for some features, and that is flagged there rather than hidden.

---

## 5. Round-1 readout (Stage 4) — the closure number

Protocol for both rounds, identical: fit everything (column screen, imputation medians,
scaler, GBM grid selection by inner grouped CV) inside FIT+MINE, then predict MONITOR.
VA_nl = mean over seeds {0,1,2} with the spread reported (FREEZE CHANGE 1).
Round 0 = V + the 154-criterion bank (96 features after the degeneracy screen).
Round 1 = round 0 + the 14 A-routed criteria (110 features).
Results: `closure/round1_results.json`.

| readout | round 0 | round 1 | change |
|---|---|---|---|
| features after screen | 96 | 110 | +14 |
| **VA_lin, MONITOR all (n=1,192)** | .6597 | .6677 | **+.0080** |
| **VA_nl, MONITOR all (n=1,192)** | **.6633** | **.6635** | **+.0003** |
| VA_nl seed spread, MONITOR all | .0030 | .0118 | — |
| VA_lin, same-rows (n=249) | .6389 | .6431 | +.0042 |
| **VA_nl, same-rows (n=249)** | .6476 | .6353 | −.0122 |
| VA_nl seed spread, same-rows | .0239 | .0323 | — |
| **T, same-rows (n=249)** | .7838 | .7838 | — |
| **Δ_0 / Δ_1 (same-rows, honest)** | **+.1363** | **+.1485** | +.0122 |
| Δ_beyond, MONITOR all (contaminated T) | +.1933 | +.1931 | −.0003 |
| *(context)* VA_lin OOF inside FIT+MINE | .6969 | .7052 | +.0083 |
| *(context)* VA_nl OOF inside FIT+MINE | .6917 | .7068 | +.0151 |

Group-level paired bootstrap (2,000×, resampling `ntitle` groups per FREEZE CHANGE 3)
on the VA_nl gain:

| set | gain | 95% CI | P(gain > 0) |
|---|---|---|---|
| MONITOR all (n=1,192) | +.0003 | [−.0132, +.0145] | .54 |
| same-rows (n=249) | −.0122 | [−.0419, +.0151] | .17 |

**Round-1 verdict against the stopping rule.** The prereg's saturation test is on
MONITOR VA_nl with ε = .005. The gain is **+.0003**, an order of magnitude below ε,
and its CI excludes anything above +.015. **Round 1 is the first of the two
consecutive sub-ε rounds that would declare Track-A saturation** (hard cap B = 5).

**Claim discipline — the Δ levels here are NOT the Layer-1 Δ_beyond.** Layer 1 quoted
Δ_beyond = +.065 from pooled OOF over all 6,030 rows (VA_nl .6876) against the
registry T = .753. This pilot's split protocol moves both terms: VA is fit on 80% and
read on a fresh 20% holdout (.663, i.e. −.025 from the pooled OOF number), and T on
the honest same-rows subset is .784 (+.031 on the registry number). Both moves inflate
Δ, which is why Δ_0 reads +.136 rather than +.065. **Only the round-over-round change
inside this protocol is interpretable**; the +.136 must never be quoted as a taste
bound alongside the Layer-1 +.065. For the best-powered honest level, see the
dense-held-out set in §6 (Δ = +.088 over 1,244 rows).

---

## 6. Track B — spurious-alone signal and the discounted readouts

11 declared nuisances, never added to the bank.

| spurious-alone AUC (B features only) | value |
|---|---|
| linear, MONITOR | **.6040** |
| HistGB, MONITOR | .5957 |
| linear, OOF inside FIT+MINE | .5894 |
| HistGB, OOF inside FIT+MINE | .5863 |

Per-feature alone-AUC on MONITOR (deviation from .5 in either direction):

| id | B feature | alone AUC |
|---|---|---|
| P02 | abstract length and verbosity | .567 |
| P20 | dominant research topic | .549 |
| P14 | raw LaTeX residue | .529 |
| P16 | engineering/data resource magnitude | .527 |
| P22 | famous-benchmark name-dropping | .524 |
| P13 | enumerated list formatting | .521 |
| P25 | industrial-compute signals | .517 |
| P08 | branded method name | .509 |
| P18 | superlative density | .492 |
| P01 | applied-domain framing | .483 |
| P24 | anonymised-repo / on-acceptance boilerplate | **.478** |

Eleven features nobody claims are quality reach **.604** together — about 40% of the
distance from chance to the dense model on these rows. Two run *against* the label:
anonymised-repository boilerplate (.478) and applied-domain framing (.483). The
boilerplate result matters because it is precisely the channel the existing bank
rewards through its availability-statement criteria — i.e. part of the 154-bank is
wired to a nuisance with the wrong sign.

### Discounted readouts (decile/quintile-stratified, n-weighted; threshold-free only)

| set | n | strata | pooled T | pooled VA | pooled Δ | T_adj | VA_adj | **Δ_adj** |
|---|---|---|---|---|---|---|---|---|
| **dense-held-out (best powered, honest T)** | 1,244 | q10 on joint B score | .7769 | .6891 | **+.0878** | .7656 | .6761 | **+.0895** |
| MONITOR ∩ dense-held-out (same-rows) | 249 | q5 on joint B score | .7838 | .6353 | +.1485 | .7824 | .6272 | +.1552 |
| MONITOR all (contaminated T) | 1,192 | q10 on joint B score | .8566 | .6635 | +.1931 | .8506 | .6521 | +.1986 |

Per-feature stratification moves Δ by at most ±.006 in any cell, and the sign of the
move is as often positive as negative. On the honest 1,244-row set the largest single
discount is length (Δ_adj +.0891) and the largest inflation is the re-routed resource
criterion (+.0914); the joint B model gives +.0895 against a pooled +.0878.

**Reading: the mined spurious channels do not explain the residual.** They carry real
standalone signal (.604), but that signal is evidently shared with what the bank
already measures rather than being the private content of Δ — stratifying it away
leaves Δ untouched, and if anything slightly larger.

**Honest limitation on this table.** Several B features are strongly zero-inflated, so
after the min-20-per-stratum guard the "decile" stratification degenerates: P24, P25
and P13 collapse to a single stratum in the 249-row cell (Δ_adj = pooled Δ exactly,
which is why those rows read identically). Only length (P02, 5 strata), topic (P20,
4-6 strata) and superlatives (P18, 3-4 strata) get a genuinely graded stratification.
The joint-B-score stratification is the one to quote; the per-feature rows for sparse
features are near-vacuous and should not be read as evidence of "no length effect".

---

## 7. Mechanism — what the mined criteria actually did

`closure/round1_mechanism.json`. This is the part that makes the null informative.

**The 14 new criteria are not inert.** On their own, with no access to the bank:

| A-block alone | AUC |
|---|---|
| linear, MONITOR all | **.6198** |
| HistGB, MONITOR all | .6179 |
| linear, same-rows | .6201 |
| OOF inside FIT+MINE (mining-contaminated) | .6410 |

Fourteen criteria authored in one pass reach .620 — essentially the whole 17-feature V
block's published .613, and only .04 short of the entire 96-feature bank's .663.

**And the mining hit its stated target.** The prereg's mining target is the dense
score, never y. Rank agreement (Spearman ρ) between VA_nl and the dense score:

| set | round 0 | round 1 | change |
|---|---|---|---|
| MONITOR all (n=1,192) | .488 | **.522** | +.034 |
| MONITOR ∩ dense-held-out (n=249) | .360 | **.397** | +.037 |
| A block alone vs dense, same-rows | — | .381 | — |

So round 1 did what it set out to do — it moved the articulated stack's *ranking*
measurably toward the dense model's ranking — **and label AUC did not move at all.**
The articulated content the dense model was using turns out to be redundant with the
bank *for predicting the outcome*, even though it is not redundant for predicting the
dense model.

Per-criterion, ranked by alignment with the dense score:

| id | A criterion | alone AUC vs y | ρ vs dense |
|---|---|---|---|
| P05 | ambition **together with** evidence specificity (composite) | **.621** | **.362** |
| P11 | technical specificity | .565 | .172 |
| P19 | generality demonstrated not asserted | .530 | .127 |
| P12 | separable contributions with stated roles | .526 | .119 |
| P10 | breadth of evaluation enumerated | .535 | .099 |
| P06 | theory tied to method with stated scope | .541 | .086 |
| P21 | quantified headline vs named baseline | .521 | .082 |
| P07 | reframing | .519 | .069 |
| P09 | motivation explains why it is hard | .521 | .064 |
| P03 | explanatory mechanism | .531 | .063 |
| P15 | diagnoses a failure mode with evidence | .534 | .049 |
| P23 | human/expert judgment as evidence | .504 | .040 |
| P17 | surprising or self-undercutting finding | .517 | .028 |
| P04 | states its own costs/limits | .508 | −.013 |

The single strongest new feature is the **composite** — "how big is the claim,
*together with* how specific the evidence for it is" — at alone-AUC .621 and ρ = .362
with the dense score, roughly double the next criterion on both. That is the one
result here that points somewhere: the prereg allowed composite criteria, and the
composite is what carried the round. A round-2 proposer should be pushed hard toward
interaction-shaped criteria rather than more single-property ones.

---

## 8. Judge-call ledger

| item | prompts |
|---|---|
| round-1 population scoring (6,030 × 25) | 150,750 |
| round-1 blinded anchors (36 × 25) | 900 |
| **cumulative, pilot to date** | **151,650** |

Plus one non-judge GPU job: the same-rows dense rescore (6,030 forward passes).

---

## 9. Reading, and what it changes

1. **The closure is not moving.** One well-resourced round of active articulation, 14
   criteria that survive a blind audit and reach .620 on their own, moved MONITOR
   VA_nl by **+.0003** [−.013, +.014]. Δ_beyond did not shrink. Round 1 is the first
   of the two consecutive sub-ε rounds that would declare saturation.
2. **The null is informative, not empty**, because the same criteria demonstrably
   moved VA_nl's ranking toward the dense model (ρ .488 → .522) while moving label AUC
   not at all. The articulable part of what the dense model perceives was already
   priced into the bank. What is left over is not "criteria we failed to think of" of
   the kind a proposer reading disagreements can reach.
3. **The spurious story does not rescue the residual either.** Declared nuisances hit
   .604 alone but stratifying on them leaves Δ unchanged (+.0878 → +.0895 on the
   honest 1,244-row set). Both candidate deflations of Δ — nonlinear interaction
   (Layer 1: Δ_interact = −.002, null) and mined shortcuts (here: Δ_adj ≈ Δ) — have now
   failed to explain the peer-verdict residual.
4. **Composites are where the signal was.** P05 doubled every other new criterion on
   both readouts. This is the concrete lever for round 2.
5. **One real instrument finding for the existing bank**: anonymised-repository /
   release-on-acceptance boilerplate is *negatively* associated with the label (.478),
   and it is what the bank's availability-statement criteria fire on. That is a wrong-
   signed nuisance living inside the A bank, consistent with the Layer-1 SHAP screen
   naming `v_kw_code` × availability-statement as the top pair. Worth an explicit
   audit before any confirmatory quote.

### Carry into the prereg freeze

- **ε = .005 and B = 5 look right**, but the pilot shows the binding constraint is
  *power*, not the threshold: the MONITOR VA_nl CI is ±.014 wide at n = 1,192 and
  ±.03 at n = 249. A confirmatory cell needs a MONITOR large enough to resolve .005,
  or the stopping rule fires on noise.
- **Split systems must be intersected up front.** The 943 contaminated MONITOR rows
  forced a 249-row honest same-rows set. Confirmatory design: define MONITOR *within*
  the dense model's held-out rows.
- **k_A = 15 / k_B = 10 worked**; the blind audit separated the pools at 96%, so the
  dual design is doing its job and the audit is not a formality.
- **Push round 2 toward composite/interaction criteria** and add the arbiter's salvage
  rewrite (artifact reusability, scale-invariant) as a Track-A candidate.
- **Scale note**: 0-10 with a single-token readout gave 0/25 collapse and NA .014 —
  better behaved than the bank's 0/0.5/1/NA (.652 NA on this corpus). Consider it for
  confirmatory rounds, with the caveat that mixed-scale banks are fine for the GBM but
  must never be pooled into one figure.

---

## 10. Artifact locations

All under `methods/taste_decomposition/closure/`:

| file | contents |
|---|---|
| `build_splits.py`, `peer_verdict_splits.json` | stable-hash split map + counts |
| `peer_verdict_population.csv` | the 6,030 rows (text, y, ntitle, splits) |
| `rescore_dense_same_rows.py`, `peer_verdict_dense_preds.csv`, `.report.json` | same-rows T (freeze #2) |
| `closure_lib.py` | frozen Layer-1 protocol refit for fit-within-FIT+MINE |
| `stage1_disagreement.py`, `peer_verdict_oof_fitmine.npz/.json` | VA_nl OOF inside FIT+MINE |
| `round1_disagreement_slice.json` | the 60 rows read (label-blind) |
| `round1_track_a.json`, `round1_track_b.json` | proposals with rationales and provenance |
| `round1_proposals_blinded.json` | exactly what the auditor saw |
| `round1_routing_audit.json`, `round1_routing_final.json` | audit verdicts, arbiter ruling, final routing |
| `score_round1_gemma.py`, `round1_scores.npz`, `round1_score_report.json` | Gemma scoring + anchors + collapse check |
| `stage4_readout.py`, `round1_results.json` | round-0/round-1 readout + Track-B discounting |
| `stage4_mechanism.py`, `round1_mechanism.json` | A-block-alone, dense alignment, per-criterion |
| `ref/rubrics_154.jsonl` | the existing bank, pulled for duplicate-checking |

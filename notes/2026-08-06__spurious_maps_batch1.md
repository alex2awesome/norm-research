# Spurious maps, batch 1 — map-focused dual-track Layer-3 rounds on six cells

Date: 2026-08-06/07. Protocol: the FROZEN preregistration
`notes/2026-08-05__layer3-closure-prereg.md` (FREEZE DECLARATION 2026-08-06 +
FREEZE ADDENDUM + **FREEZE ADDENDUM 2**, the Track-B upstream-factor mode), run in
the **map-focused** variant the user asked for: two rounds per cell rather than to
saturation, both tracks every round, and the **spurious map** — not the closure
curve — as the headline deliverable. Track A is reported as secondary.

Cells: **peer curation, peer revealed, humor caption crowd, humor caption finalist,
N&C outcome** (the five ready-but-small-or-negative-residual cells), plus **N&C
agree**, added mid-run by the coordinator with its own brief
(`methods/taste_decomposition/closure/nc_agree_brief.md`).

Code + artifacts: `methods/taste_decomposition/closure/maps_batch1/`.

---

## 0. Terms, spelled out (standing rule)

**V** = the block of deterministic surface features (16–27 per cell);
**A** = the articulated-criterion bank, Gemma-4-31B-judged rubric scores;
**VA_nl** = the HistGradientBoosting aggregation of the V+A matrix, mean over seeds
{0,1,2} (FREEZE CHANGE 1); **VA_lin** = its logistic counterpart;
**T** = the dense readout (Llama-3.1-8B LoRA reward model on raw text), always the
**same-rows** rescore restricted to that model's own eval/test rows;
**Δ_beyond** = T − VA_nl, the unarticulated residual; **Δ_r** = Δ_beyond after r
rounds of mining; **ε** = .005, the frozen per-round saturation threshold;
**AUC** = area under the ROC curve;
**FIT+MINE / MONITOR** = the closure splits (below);
**M** = the mining slice, the dense-held-out rows inside FIT+MINE;
**HONEST population** = every dense-held-out row (M ∪ MONITOR), the population on
which T and VA are both out-of-sample;
**Track A** = the quality-criterion miner (k_A = 15 scored criteria/round);
**Track B** = the suspected-spurious-channel miner (k_B = 10 scored channels/round);
**alone-AUC** = the AUC of a single judged channel's raw 0–10 score, no fitting;
**joint B model** = HistGB on the B channels only;
**stacked increment** = AUC(logistic stack of joint-B + X) − AUC(joint-B), the
stratification-free version of a discount;
**mixed channel** = a Track-B channel whose conjectured upstream parent plausibly
causes real quality as well (FREEZE ADDENDUM 2), reported in both the discounted and
the undiscounted readout as a sensitivity band;
**Good-Turing missing mass M̂** = f1/N over proposal *species*, the estimated
probability that the next independent proposal names a species not yet seen;
**LOPO** = leave-one-proposer-out jackknife; **P** = fleet size;
**GEPA** = the prompt-iteration pass required before any final quoted phrasing —
**not applied here**, so everything below carries a pre-GEPA flag.

---

## 1. What was run, and the four operational decisions that are not in the frozen text

Per cell, per round: disagreement slice → sealed fleet proposes on both tracks →
species clustering + Good-Turing → top-k selection → blind routing audit with planted
probes → frontier arbiter on disputes → corpus-wide Gemma-4-31B scoring → readout.

Four decisions had to be made that the freeze does not pin down. All four are
recorded here rather than silently taken.

**(a) MONITOR is defined inside the dense-held-out rows, and the hash threshold
inside that set is .50, not .80.** The pilot amendment says "MONITOR must be defined
INSIDE the dense-held-out rows"; the freeze restates it as "MONITOR ⊂ dense-held-out".
Every one of these cells is ~80% dense-train, so applying .80 *inside* the held-out
rows lands MONITOR at 92–310 rows (peer revealed: 92 rows, 54 positives). The
held-out groups are therefore split 50/50: MONITOR = held-out groups with
sha256(group)/2^256 ≥ .50, mining slice M = the other half, FIT+MINE = M plus every
dense-train group. Precondition verified in code: on all six cells the dense split is
**group-pure** (no ntitle / contest / docket straddles two dense splits), so FIT+MINE
and MONITOR share zero groups. Consequence worth stating plainly: **T on MONITOR is
honest for all six cells** — the pilot's 943-contaminated-row problem cannot recur —
and every readout is additionally reported on the HONEST population (2–4× larger),
which is the pilot's own primary discount population.

**(b) The fleet proposes, a species rule selects, and only k_A + k_B get scored.**
The freeze fixes both k_A = 15 / k_B = 10 *and* a sealed fleet of P proposers; a
P = 4 fleet emits 60 A + 40 B proposals per round, and scoring all of them corpus-wide
would multiply the frozen judge budget by four. Each round the fleet's proposals are
clustered into species (single-linkage on bge-large cosine at τ = .79), Good-Turing
runs on the **full** species pool, and the scored set is the top-k species by
cross-proposer support (n distinct proposers, then n members, then a stable sha256),
one representative per species chosen by smallest sha256(pid) so no family is
favoured. Consensus-first selection is the right bias for a map: a channel several
sealed proposers name independently is the one whose alone-AUC most deserves a
corpus-wide measurement. Singletons enter once multi-proposer species run out.
The τ = .79 threshold is legitimate here because every proposal comes from the same
fleet reading the same slice — one register. The cross-register prohibition in
`notes/2026-08-06__missing-mass-robustification.md` §2.3 does not apply and is not
being violated.

**(c) The fleet is P = 4 over 2 families (Claude ×2, gpt-5.6-luna ×2), uniformly.**
GLM-5.2 was attempted first and returned HTTP 429 `1308` — *"Usage limit reached for
5 hour"* — on both Lite keys, i.e. the account's rolling 5-hour cap, not the
per-request 1302 rate limit the missing-mass battery hit. Rather than have some cells
at P = 6 and others at P = 4 (which would make their Good-Turing numbers
non-comparable), the GLM leg was dropped for the whole batch and recorded. This is
the freeze's own permitted degradation ("degrade gracefully to P ≥ 4 / ≥ 2 families
under GLM rate limits, recorded") and the robustification note's recommendation #4
("treat GLM as a bonus family and build the guaranteed P = 4 from Claude + Codex").
All 48 core slots (6 cells × 2 tracks × 4 proposers) returned complete, parseable,
distinctly-named sets — 0 parse failures, 0 retries.

**(d) Planted probes are corpus-general, and on one cell that is a defect.** The four
planted probes (two obviously quality-relevant, two obviously incidental) are shared
across cells. On **cap_crowd** the auditor scored 2/4, failing both quality-relevant
probes ("explicit statement of the limitation or failure regime", "reasoning given for
why the approach should work") — which are, read honestly, *inapplicable* to a
one-line cartoon caption. The cap_finalist auditor passed the same two 4/4. This is a
probe-design limitation on very short text, not evidence that the cap_crowd audit is
unreliable; its misrouting rate was 0/25. **Carry-forward: planted probes must be
corpus-matched.** All other cells: 4/4.

### Fleet and audit bookkeeping (all six cells, round 1)

| cell | proposals | slots | A/B after routing | mixed B | misrouting | probes | disputes → arbiter |
|---|---:|---|---|---:|---:|---:|---|
| peer_curation | 100 | 8/8 | 14 / 11 | 6 | 1/25 | 4/4 | 1 → B |
| peer_revealed | 100 | 8/8 | 14 / 11 | 9 | 1/25 | 4/4 | 1 → B |
| cap_crowd | 100 | 8/8 | 15 / 10 | 8 | 0/25 | 2/4 † | 0 |
| cap_finalist | 100 | 8/8 | 15 / 10 | 6 | 1/25 | 4/4 | 1 → B |
| nc_outcome | 100 | 8/8 | 14 / 11 | 9 | 1/25 | 4/4 | 1 → B |
| nc_agree | 100 | 8/8 | 14 / 11 | 9 | 3/25 | 4/4 | 3 (1 → A, 2 → B) |

† see (d). All seven arbitrated disputes were resolved by a frontier (Opus) arbiter
with provenance visible by design; **six of seven went to B**, and five of those were
also flagged `mixed`. The disputed cases are systematically "quality-ish surface"
channels — original-vs-template prose, acronym packaging density, membership counts
and statutory mandates, whether a number carries the joke — exactly the boundary the
prereg's parked open question (b) named as a per-cell substantive decision.

---

## 2. Round-0 baseline (the number every round-1 result is read against)

Same rows, both models out-of-sample: T from the same-rows dense rescore restricted to
dense eval/test; VA_nl grouped-OOF inside FIT+MINE and held-out on MONITOR.

| cell | n | HONEST n | MONITOR n (groups) | T (HONEST) | VA_nl (HONEST) | **Δ_beyond (HONEST)** | 95% CI | T (MONITOR) | VA_nl (MONITOR) | Δ (MONITOR) |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| peer_curation | 7,941 | 1,571 | 775 | 0.594 | 0.582 | **+0.011** | [-0.028, +0.052] p=0.72 | 0.597 | 0.560 | +0.036 |
| peer_revealed | 2,387 | 478 | 244 | 0.884 | 0.724 | **+0.160** | [+0.115, +0.205] p=1.00 | 0.863 | 0.696 | +0.167 |
| cap_crowd | 10,893 | 2,190 | 1,072 | 0.555 | 0.624 | **-0.068** | [-0.095, -0.041] p=0.00 | 0.547 | 0.646 | -0.099 |
| cap_finalist | 5,218 | 1,055 | 526 | 0.612 | 0.669 | **-0.057** | [-0.107, -0.008] p=0.01 | 0.650 | 0.692 | -0.042 |
| nc_outcome | 7,084 | 1,417 | 694 | 0.624 | 0.632 | **-0.008** | [-0.051, +0.033] p=0.36 | 0.616 | 0.631 | -0.015 |
| nc_agree | 5,046 | 1,009 | 487 | 0.603 | 0.626 | **-0.022** | [-0.074, +0.024] p=0.17 | 0.601 | 0.617 | -0.016 |

---


## 3. Cross-cell synthesis — three regimes, and what the maps buy

The **stacked increment** is the load-bearing readout of this batch, because it is
stratification-free and does not degrade as the nuisance set grows: fit a grouped-OOF
logistic stack of (joint nuisance model + X) inside the HONEST population and ask how
much X adds over *everything the miner named*.

| cell | round | AUC(named nuisance model) | AUC(dense) | AUC(bank) | dense increment over nuisance | p | bank increment over nuisance | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| peer_curation | r1 | 0.545 | 0.594 | 0.570 | **+0.0452** | 0.99 | **+0.0141** | 0.80 |
| peer_curation | r2 | 0.561 | 0.594 | 0.577 | **+0.0316** | 0.98 | **+0.0118** | 0.81 |
| peer_revealed | r1 | 0.752 | 0.884 | 0.751 | **+0.1331** | 1.00 | **+0.0200** | 0.97 |
| peer_revealed | r2 | 0.768 | 0.884 | 0.763 | **+0.1188** | 1.00 | **+0.0318** | 1.00 |
| cap_crowd | r1 | 0.545 | 0.555 | 0.630 | **+0.0105** | 0.83 | **+0.0845** | 1.00 |
| cap_crowd | r2 | 0.580 | 0.555 | 0.634 | **+0.0052** | 0.79 | **+0.0564** | 1.00 |
| cap_finalist | r1 | 0.646 | 0.612 | 0.679 | **+0.0038** | 0.71 | **+0.0303** | 0.99 |
| cap_finalist | r2 | 0.598 | 0.612 | 0.664 | **+0.0273** | 0.97 | **+0.0650** | 1.00 |
| nc_outcome | r1 | 0.637 | 0.624 | 0.632 | **+0.0152** | 0.95 | **+0.0071** | 0.83 |
| nc_outcome | r2 | 0.621 | 0.624 | 0.625 | **+0.0263** | 1.00 | **+0.0248** | 0.99 |
| nc_agree | r1 | 0.596 | 0.603 | 0.624 | **+0.0193** | 0.91 | **+0.0248** | 0.93 |
| nc_agree | r2 | 0.607 | 0.603 | 0.632 | **+0.0203** | 0.94 | **+0.0269** | 0.98 |

Read down the last four columns and the six cells fall into **three regimes**:

**(i) DENSE-CARRIES — peer curation, peer revealed.** The dense model adds a large,
significant increment over every named nuisance channel (+.032 to +.133, p ≥ .98 in
all four rounds) while the bank adds little (+.012 to +.032). On **peer revealed** the
numbers are stark: the eleven-to-thirteen judged nuisance channels *alone* reach
**.752 / .768**, statistically indistinguishable from the entire 85-feature articulated
bank (**.751 / .763**), while the dense model sits at **.884**. The residual survives
discounting and in fact *grows*: Δ_adj = **+.197** (r1) / **+.169** (r2) against a
pooled +.133 / +.121, because the bank leans on the nuisance channels harder than the
dense model does.

**(ii) BANK-CARRIES — humor caption crowd, humor caption finalist.** The mirror image.
The bank adds +.030 to +.085 over the nuisance model (p ≥ .99 in all four rounds); the
dense model adds +.004 to +.027, and on three of the four rounds its increment is not
distinguishable from zero (p = .71–.97). Discounting does **not** close the bank-over-
dense gap — on cap_crowd it is −.075 pooled and −.079/−.085 after discount — so the
articulation win on these cells is *not* a nuisance artifact. What the discount does
expose is the other side: on **cap_finalist r1**, stratifying on the joint nuisance
score drives **T to .501 — chance** — while the bank holds .615. The negative-Δ control
cell is negative not because articulation is unusually strong there but because the
dense model's above-chance performance is, on that round, essentially all length- and
topicality-borne.

**(iii) NUISANCE-SATURATED — N&C outcome, N&C agree.** The joint nuisance model reaches
**.596–.637**, i.e. *the same level as both instruments* (dense .603–.624, bank
.624–.632). Both increments are small (+.007 to +.027). Almost everything either
instrument measures on these two cells is already spanned by channels the miner named
in one round — and those channels are overwhelmingly upstream fingerprints, not surface
proxies: professional legal drafting, legal-citation apparatus, document volume,
scanned-attachment/OCR signatures, organizational standing, sender authority,
submission apparatus. This is the textual mechanism behind Layer 2's docket-identity
finding (`notes/2026-08-06__layer2_robustness.md` §1: docket-identity-alone AUC .856
/.862, within-docket VA_nl .558/.493). The map does not merely restate the leak — it
names what in the *text* carries it.

### 3.1 The Addendum-2 upstream mode is where the signal is

The strongest channel in every cell except cap_crowd has a conjectured **upstream
parent**, not "surface-only":

| cell | strongest channel | alone AUC (HONEST) | conjectured upstream parent |
|---|---|---:|---|
| peer_revealed | Trend-Aligned Vocabulary (r2) / Trend-hype keyword density (r1) | **.711** / .677 | submission timing, field fashion |
| peer_revealed | Canonical Ecosystem Naming | .641 | author network / institutional ecosystem |
| peer_revealed | Evaluation breadth | .640 | institutional resources and team size |
| nc_outcome | Technical identifier density | .609 | technical domain and venue style |
| nc_outcome | Legal/regulatory citation apparatus density | .598 | professional/legal drafting assistance |
| nc_outcome | Professional legal drafting | .589 | professional editing or legal assistance |
| nc_agree | Concrete organizational standing disclosed | .594 | (proposed surface-only; parent is org identity) |
| nc_agree | Sender authority cues | .580 | author seniority and professional standing |
| cap_finalist | Caption brevity / word count | .585 | surface-only |
| peer_curation | Memorable title framing (r2) | .561 | surface-only |
| cap_crowd | Editorial elaboration | .546 | professional editing or assistance |

Two structural facts follow. First, **the upstream-reasoning mode is not decorative**:
the six highest alone-AUCs in the batch (.711, .677, .641, .640, .609, .598) are all
Mode-2 fingerprints of who produced the item, with what resources, and when. Pure
surface pattern-hunting (Mode 1) tops out around .58 and on cap_crowd never leaves the
noise (max .546). Second, the **interpretation note in Addendum 2 is confirmed in the
direction it predicts**: these are *traced* unseen factors — they leave a textual
fingerprint, they are measurable, and they are exactly the ones that can bias Δ. On
peer revealed and the two N&C cells they bias it enough to be the whole story of the
bank's performance.

### 3.2 The MIXED band matters exactly where the upstream parents are strong

| cell | round | pooled Δ (HONEST) | Δ_adj ALL B | Δ_adj STRICT | band width | T_adj ALL | VA_adj ALL |
|---|---|---:|---:|---:|---:|---:|---:|
| peer_curation | r1 | +0.0233 | +0.0288 | +0.0295 | 0.0007 | 0.591 | 0.562 |
| peer_curation | r2 | +0.0163 | +0.0212 | +0.0130 | 0.0082 | 0.583 | 0.562 |
| peer_revealed | r1 | +0.1331 | +0.1974 | +0.1394 | 0.0580 | 0.836 | 0.638 |
| peer_revealed | r2 | +0.1210 | +0.1686 | +0.1336 | 0.0350 | 0.842 | 0.673 |
| cap_crowd | r1 | -0.0748 | -0.0853 | -0.0744 | 0.0109 | 0.541 | 0.626 |
| cap_crowd | r2 | -0.0783 | -0.0786 | -0.0765 | 0.0021 | 0.543 | 0.621 |
| cap_finalist | r1 | -0.0668 | -0.1139 | -0.0815 | 0.0324 | 0.501 | 0.615 |
| cap_finalist | r2 | -0.0518 | -0.0573 | -0.0558 | 0.0015 | 0.582 | 0.640 |
| nc_outcome | r1 | -0.0084 | +0.0169 | -0.0011 | 0.0179 | 0.578 | 0.561 |
| nc_outcome | r2 | -0.0017 | +0.0044 | +0.0045 | 0.0001 | 0.583 | 0.578 |
| nc_agree | r1 | -0.0205 | -0.0217 | -0.0168 | 0.0049 | 0.576 | 0.597 |
| nc_agree | r2 | -0.0284 | -0.0231 | -0.0340 | 0.0109 | 0.579 | 0.602 |

The band is negligible on cells whose channels are surface-only (peer curation r1
.0007, cap_crowd r2 .0021, nc_outcome r2 .0001) and widest exactly where the upstream
parents are strong and plausibly quality-causing: **peer revealed .058 / .035**
(institutional resources, evaluation breadth) and **cap_finalist r1 .032** (cultural
literacy). Quote peer revealed's discounted residual as a band, **Δ_adj ∈ [+.134,
+.197]**, never as a point.

### 3.3 Track A (secondary): the miner is near-exhausted on these cells

| cell | r1 gain HONEST | r1 gain MONITOR | r2 gain HONEST | r2 gain MONITOR | cleared ε=.005? | Δ_beyond HONEST after r2 |
|---|---:|---:|---:|---:|---|---:|
| peer_curation | -0.0119 | -0.0111 | +0.0071 | +0.0048 | r1 no / r2 yes | +0.0163 |
| peer_revealed | +0.0271 | +0.0066 | +0.0121 | +0.0142 | r1 yes / r2 yes | +0.1210 |
| cap_crowd | +0.0067 | +0.0063 | +0.0035 | -0.0023 | r1 yes / r2 no | -0.0783 |
| cap_finalist | +0.0097 | +0.0155 | -0.0151 | +0.0085 | r1 yes / r2 no | -0.0518 |
| nc_outcome | -0.0001 | +0.0048 | -0.0067 | -0.0113 | r1 no / r2 no | -0.0017 |
| nc_agree | -0.0019 | -0.0110 | +0.0079 | -0.0003 | r1 no / r2 yes | -0.0284 |

Six of twelve rounds cleared ε = .005 on the HONEST population; two rounds were
outright **negative** (peer curation r1 −.012, cap_finalist r2 −.015 — the added
criteria were net noise). **N&C outcome fired the frozen saturation rule** (two
consecutive sub-ε rounds, −.0001 then −.0067) and is the one cell in this batch where
the closure curve would legitimately have stopped on its own. Nothing here approaches
the pilot's peer-verdict curve; on map-focused cells the quality miner is close to
exhausted after one round, which is why the map and not the curve is the deliverable.

A second, structural observation about Track A: as the bank absorbs the easy criteria,
the miner's "quality" proposals **drift into style**. Round-2 misrouting rose on three
cells (cap_crowd 0→24%, nc_agree 12→16%, peer revealed 4→12%), and **21 of the 23
arbitrated disputes across the whole batch were routed to B**, most of them flagged
mixed. The arbiter's language is consistent — "presentation-style axis", "scores
discourse position", "genre and packaging", "both poles described neutrally with no
better end declared". This is a measurable form of the redundancy saturation the
missing-mass note described, visible in the *routing* rather than the AUC.

### 3.4 B-side missing mass: the spurious space is LESS exhausted than the quality space

| cell | round | A: S_obs | A: M̂ | A: LOPO | A: recapture | B: S_obs | B: M̂ | B: LOPO | B: recapture |
|---|---|---:|---:|---|---:|---:|---:|---|---:|
| peer_curation | r1 | 54 | 0.817 | [0.76, 0.96] | 0.09 | 28 | **0.550** | [0.57, 0.67] | 0.21 |
| peer_curation | r2 | 46 | 0.633 | [0.53, 0.91] | 0.17 | 30 | **0.650** | [0.60, 0.73] | 0.13 |
| peer_revealed | r1 | 50 | 0.750 | [0.76, 0.82] | 0.10 | 27 | **0.500** | [0.50, 0.70] | 0.26 |
| peer_revealed | r2 | 50 | 0.700 | [0.71, 0.78] | 0.12 | 31 | **0.675** | [0.63, 0.73] | 0.13 |
| cap_crowd | r1 | 42 | 0.417 | [0.22, 0.78] | 0.33 | 34 | **0.750** | [0.73, 0.87] | 0.12 |
| cap_crowd | r2 | 39 | 0.500 | [0.44, 0.67] | 0.23 | 27 | **0.500** | [0.47, 0.60] | 0.22 |
| cap_finalist | r1 | 34 | 0.400 | [0.38, 0.53] | 0.26 | 30 | **0.600** | [0.60, 0.70] | 0.20 |
| cap_finalist | r2 | 35 | 0.383 | [0.44, 0.49] | 0.31 | 29 | **0.575** | [0.53, 0.67] | 0.17 |
| nc_outcome | r1 | 29 | 0.383 | [0.36, 0.51] | 0.21 | 35 | **0.775** | [0.77, 0.87] | 0.11 |
| nc_outcome | r2 | 36 | 0.500 | [0.38, 0.69] | 0.17 | 33 | **0.650** | [0.73, 0.80] | 0.21 |
| nc_agree | r1 | 29 | 0.350 | [0.22, 0.56] | 0.28 | 27 | **0.450** | [0.43, 0.57] | 0.30 |
| nc_agree | r2 | 30 | 0.450 | [0.31, 0.58] | 0.10 | 34 | **0.750** | [0.70, 0.87] | 0.12 |

Mean Good-Turing missing mass: **Track B .619 vs Track A .524**, with B > A on **8 of
12 rounds**. Track-B cross-proposer recapture is correspondingly lower (.11–.30 vs
.09–.33). The reading is not that the B miner is worse: it is that at P = 4 the
*space of nameable nuisance channels is larger and less consensual* than the space of
nameable quality criteria. Practically this means the maps below are **lower bounds on
the nuisance channel set** — the honest statement for every cell is "these are channels
this fleet named", with an estimated 45–78% chance that a fifth independent proposer
names a species not in the table. Per the FREEZE ADDENDUM's own wording, the
value-weighted version of that bound carries the assumption that unfound channels
resemble found ones in influence, which the batch cannot test; the species-mass number
is the defensible one and Chao1 is not quoted.

### 3.5 Judging quality gates

| cell | round | anchors coherent-vs-scrambled (gate ≥.70) | anchors pos-vs-neg | collapsed criteria | NA rate | routing misroute | probes |
|---|---|---:|---:|---:|---:|---:|---:|
| peer_curation | r1 | 0.933 PASS | 0.589 | 0 | 0.006 | 0.04 | 4/4 |
| peer_curation | r2 | 0.979 PASS | 0.617 | 0 | 0.024 | 0.08 | 4/4 |
| peer_revealed | r1 | 0.970 PASS | 0.642 | 0 | 0.013 | 0.04 | 4/4 |
| peer_revealed | r2 | 0.972 PASS | 0.641 | 0 | 0.011 | 0.12 | 4/4 |
| cap_crowd | r1 | 1.000 PASS | 0.622 | 2 | 0.006 | 0.00 | 2/4 |
| cap_crowd | r2 | 0.981 PASS | 0.584 | 0 | 0.003 | 0.24 | 4/4 |
| cap_finalist | r1 | 1.000 PASS | 0.398 | 0 | 0.002 | 0.04 | 4/4 |
| cap_finalist | r2 | 0.984 PASS | 0.368 | 1 | 0.002 | 0.00 | 4/4 |
| nc_outcome | r1 | 0.945 PASS | 0.718 | 0 | 0.002 | 0.04 | 4/4 |
| nc_outcome | r2 | 0.995 PASS | 0.667 | 0 | 0.002 | 0.04 | 4/4 |
| nc_agree | r1 | 0.983 PASS | 0.590 | 0 | 0.005 | 0.12 | 4/4 |
| nc_agree | r2 | 0.873 PASS | 0.606 | 0 | 0.002 | 0.16 | 4/4 |

Every batch passed the scrambled-text gate (.873–1.000 vs the ≥.70 threshold), NA rates
are .002–.024, and 3 of 300 scored criteria collapsed and were dropped by the FIT+MINE
degeneracy screen. Two flags to carry: **cap_finalist's pos-vs-neg anchor AUC is below
.5 in both rounds (.398 / .368)** — expected for a criterion set that is deliberately
nuisance-heavy on a cell whose positives are a 13%-rate finalist class, but it means
the pos-vs-neg anchor is uninformative there and only the scrambled gate is doing work;
and cap_crowd r1's 2/4 planted-probe result, discussed in §1(d).

---

## 4. Where a map changes an existing conclusion

1. **peer revealed's "topic floor" is now named and sized.** The registry and the VAT
   plan carried it as a qualitative caveat ("citation percentile is substantially
   topic-predictable"). It is now: **eleven-to-thirteen judged channels — trend-aligned
   vocabulary, canonical ecosystem naming, evaluation breadth, artifact-release
   signalling, publication polish — reach .752/.768 alone, matching the whole
   articulated bank (.751/.763).** The bank on this cell is, to within noise, a
   topic-fashion-and-resources detector. The dense model is not: it adds +.12 to +.13
   over the same channels, and discounting *raises* the residual to +.134…+.197. So the
   caveat cuts against the BANK, not against Δ_beyond — the opposite of how a topic-floor
   caveat is usually read.
2. **cap_finalist's negative Δ is a dense-model weakness, not an articulation triumph.**
   Stratifying on the joint nuisance score puts **T at .501 (chance)** in r1 while the
   bank holds .615. The pre-registered negative-Δ control still behaves as designed, but
   its mechanism is now visible and should be stated that way.
3. **N&C's docket-identity leak has a textual mechanism.** Layer 2 showed
   docket-identity-alone AUC .856/.862 with within-docket VA_nl at .558/.493. The maps
   name the carrier: professional legal drafting, citation apparatus, document volume,
   scanned/OCR submission signatures, organizational standing and membership counts,
   sender authority. On both cells that channel set alone matches both instruments.
   Any future N&C claim about "comment quality" has to clear this set first.
4. **peer curation's Layer-2 date failure is named.** Layer 2 found publication year the
   only nuisance dimension failing peer curation's .02 survival tolerance (.561 → .536).
   The map supplies the mechanism: **trend-term concentration (.544) and
   trend-currentness markers (.505)**, both tagged to submission timing / field fashion.
5. **cap_crowd is the negative case, and it is informative.** Its map is the flattest in
   the batch (max alone-AUC .546, joint .545/.580, almost all channels surface-only).
   Crowd humour preference has no strong named nuisance channel — which is why the
   bank's .63 advantage there survives every discount unchanged.

---

## 5. Caveats that travel with every number

1. **Pre-GEPA.** No criterion phrasing was GEPA-iterated. Alone-AUCs are therefore
   lower bounds on what each *concept* could achieve with tuned phrasing.
2. **P = 4, two families.** GLM was capped out (§1c). Missing mass is measured at P = 4;
   a third family could only add species.
3. **The maps are lower bounds on the channel set** (§3.4): B-side M̂ .45–.78.
4. **Two rounds, not to saturation.** This was the instruction; only N&C outcome
   independently satisfied the frozen stopping rule.
5. **MONITOR is small on peer revealed (244 rows) and cap_finalist (526 rows, 66
   positives).** MONITOR-column numbers on those two cells are noisy; the HONEST
   population governs every claim above.
6. **Δ_adj is a stratified readout, not a causal estimate.** It answers "does the gap
   survive holding the named channels fixed", not "what would the gap be in a world
   without the upstream factor".
7. **nc_agree's T must be quoted with both legs** (registry: eval .566 / test .639;
   pooled honest .603). Its Δ numbers here use the pooled honest .603 and are reported
   as such; the divergence is a property of the cell, not of this batch.
8. **Split adaptation (§1a).** MONITOR is 50% of the held-out groups, not 20%. Closure
   levels are protocol-specific and not comparable to Layer-1 Δ_beyond; only
   round-over-round changes and the same-rows honest levels are quoted.

---

## 6. Artifacts

- Driver + all per-cell code: `methods/taste_decomposition/closure/maps_batch1/`
  (`cells.py` loaders incl. the new nc_agree entry, `build_splits.py`,
  `stage1_slice.py`, `harness_maps.py` sealed dual-track prompts, `run_fleet.py`
  codex/GLM legs, `species.py` clustering + Good-Turing + selection, `audit.py`
  blind audit + routing, `arbiter.py`, `score_gemma_maps.py` (sk3),
  `closure_core.py` frozen fitting spec, `readout.py`, `report.py`).
- Per-cell per-round results: `maps_batch1/<cell>_r<r>_results.json`
  (spurious map, discount tables, stacked increment, Track A, missing mass).
- Round-0 baselines: `maps_batch1/round0_context_all.json` + `<cell>_r0_context.json`.
- Splits: `maps_batch1/<cell>_splits.json` (+ `splits_summary.json`).
- Fleet proposals with full provenance: `maps_batch1/<cell>_r<r>_proposals_fleet.json`;
  species tables `<cell>_r<r>_species.json`; sealed prompts and raw proposer outputs in
  the session scratchpad `maps_batch1/<cell>_r<r>/`.
- Audit + arbitration: `<cell>_r<r>_audit_prompt.txt`, `_audit_key.json`,
  `_audit_verdicts.json`, `_arbiter.json`, `_routing_final.json`.
- Gemma score matrices + collapse/anchor reports: `<cell>_r<r>_scores.npz`,
  `<cell>_r<r>_score_report.json` (mirrors on sk3 under the same path).
- **New same-rows dense T for the two peer cells** (this batch's GPU job):
  `methods/taste_decomposition/closure/samerows_preds/peer_{curation,revealed}_dense_preds_slim.csv`
  and `methods/taste_decomposition/results/samerows_T_peer_cells.json`
  (peer curation held-out T .5936 on n=1,571; peer revealed .8842 on n=478).
- Cross-cell tables: `maps_batch1/summary_tables.md`; per-cell rendered sections:
  `maps_batch1/sections.md` (generated by `report.py`).
- GPU ledger entries under `agent=claude-maps-batch1` in
  `/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt`.

---

# Per-cell sections

<!-- generated by maps_batch1/report.py -->

## peer_curation — peer curation (ICLR oral/spotlight selection)

Population n=7941; HONEST (dense-held-out) n=1571; MONITOR n=775 (775 groups). Round-0: T 0.594 / VA_nl 0.582 / **Δ_beyond +0.011** on HONEST; T 0.597 / VA_nl 0.560 / Δ +0.036 on MONITOR.

### Spurious map (headline)

| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |
|---|---:|---:|---|:--:|
| Memorable title framing | 0.561 | 0.539 | surface-only |  |
| Trend-term concentration | 0.544 | 0.543 | submission timing and community fashion | YES |
| Abstract length / verbosity | 0.543 | 0.517 | surface-only | YES |
| Concrete, vivid endpoint achievement stated in plain language | 0.534 | 0.515 | surface-only | YES |
| Branded method acronym or typographic coinage | 0.529 | 0.537 | institutional PR / lab branding practice and professional writing support | YES |
| Importance-of-problem preamble before any technical content | 0.522 | 0.539 | surface-only |  |
| Artifact-release advertising: code URLs, project pages, videos | 0.514 | 0.502 | institutional release capacity and lab open-source policy (engineering support, legal clearance, hosting) | YES |
| Template-structure conformity | 0.509 | 0.512 | venue and community writing conventions | YES |
| Trend-currentness markers | 0.505 | 0.508 | submission timing / field fashion | YES |
| Formalism density | 0.499 | 0.519 | surface-only | YES |

### Discount

| readout | spurious-alone AUC (joint) | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL B channels, HONEST q10 | 0.561 | 0.583 | 0.562 | +0.021 | 0.594 | 0.577 | +0.016 |
| ALL B channels, MONITOR q5 | 0.561 | 0.581 | 0.530 | +0.050 | 0.597 | 0.554 | +0.043 |
| STRICT (mixed dropped), HONEST q10 | 0.528 | 0.589 | 0.576 | +0.013 | 0.594 | 0.577 | +0.016 |
| STRICT (mixed dropped), MONITOR q5 | 0.528 | 0.600 | 0.555 | +0.045 | 0.597 | 0.554 | +0.043 |

### Stacked increment

| population | AUC(B) | AUC(dense) | AUC(bank) | AUC(B+dense) | dense increment | AUC(B+bank) | bank increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| HONEST (n=1571) | 0.561 | 0.594 | 0.577 | 0.593 | +0.032 | 0.573 | +0.012 |
| MONITOR (n=775) | 0.575 | 0.597 | 0.554 | 0.598 | +0.023 | 0.553 | -0.023 |

### Track A (secondary)

| round | bank features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |
|---|---:|---:|---:|---:|---:|---|---:|
| r1 | 123 | 0.549 | 0.570 | -0.011 | -0.012 | [-0.031, +0.007] p=0.10 | +0.023 |
| r2 | 138 | 0.554 | 0.577 | +0.005 | +0.007 | [-0.011, +0.025] p=0.78 | +0.016 |

### Missing mass (both tracks)

| round | track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO jackknife | cross-proposer recapture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r1 | A | 60 | 4 | 2 | 54 | 49 | 4 | 0.817 | [0.756, 0.956] | 0.09 |
| r1 | B | 40 | 4 | 2 | 28 | 22 | 3 | 0.550 | [0.567, 0.667] | 0.21 |
| r2 | A | 60 | 4 | 2 | 46 | 38 | 6 | 0.633 | [0.533, 0.911] | 0.17 |
| r2 | B | 40 | 4 | 2 | 30 | 26 | 1 | 0.650 | [0.600, 0.733] | 0.13 |

### Round bookkeeping

- r1: routing A=14 / B=11 (mixed 6), misrouting 0.04, planted probes 4/4; anchors pos-vs-neg AUC 0.589, coherent-vs-scrambled 0.933 (PASS), collapsed criteria 0, NA rate 0.006
- r2: routing A=15 / B=10 (mixed 8), misrouting 0.08, planted probes 4/4; anchors pos-vs-neg AUC 0.617, coherent-vs-scrambled 0.979 (PASS), collapsed criteria 0, NA rate 0.024

## peer_revealed — peer revealed (citation percentile)

Population n=2387; HONEST (dense-held-out) n=478; MONITOR n=244 (244 groups). Round-0: T 0.884 / VA_nl 0.724 / **Δ_beyond +0.160** on HONEST; T 0.863 / VA_nl 0.696 / Δ +0.167 on MONITOR.

### Spurious map (headline)

| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |
|---|---:|---:|---|:--:|
| Trend-Aligned Vocabulary | 0.711 | 0.734 | submission timing |  |
| Canonical Ecosystem Naming | 0.641 | 0.638 | author network and institutional exposure | YES |
| Superlative Claim Rhetoric | 0.591 | 0.579 | surface-only |  |
| Artifact-release signaling | 0.591 | 0.574 | institutional resources and artifact infrastructure | YES |
| Density of specific quantified result figures | 0.584 | 0.591 | surface-only | YES |
| Concrete numeric result density | 0.583 | 0.605 | surface-only | YES |
| Explicit code/artifact release mention | 0.580 | 0.567 | surface-only | YES |
| Sheer expository volume of the abstract body | 0.571 | 0.566 | surface-only | YES |
| Publication-Polish Fluency | 0.553 | 0.581 | professional editing | YES |
| Density of formal mathematical/symbolic notation | 0.474 | 0.509 | surface-only | YES |
| Extent of adversarial/security-threat subject framing | 0.521 | 0.508 | surface-only |  |
| Explicit first-to-achieve priority claims | 0.494 | 0.496 | author seniority / competitive positioning | YES |
| Rhetorical framing as adjudicating a field debate | 0.504 | 0.491 | surface-only | YES |

### Discount

| readout | spurious-alone AUC (joint) | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL B channels, HONEST q10 | 0.768 | 0.842 | 0.673 | +0.169 | 0.884 | 0.763 | +0.121 |
| ALL B channels, MONITOR q5 | 0.768 | 0.825 | 0.603 | +0.222 | 0.863 | 0.717 | +0.146 |
| STRICT (mixed dropped), HONEST q10 | 0.724 | 0.856 | 0.722 | +0.134 | 0.884 | 0.763 | +0.121 |
| STRICT (mixed dropped), MONITOR q5 | 0.724 | 0.836 | 0.643 | +0.193 | 0.863 | 0.717 | +0.146 |

### Stacked increment

| population | AUC(B) | AUC(dense) | AUC(bank) | AUC(B+dense) | dense increment | AUC(B+bank) | bank increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| HONEST (n=478) | 0.768 | 0.884 | 0.763 | 0.887 | +0.119 | 0.800 | +0.032 |
| MONITOR (n=244) | 0.765 | 0.863 | 0.717 | 0.862 | +0.097 | 0.765 | -0.000 |

### Track A (secondary)

| round | bank features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |
|---|---:|---:|---:|---:|---:|---|---:|
| r1 | 97 | 0.703 | 0.751 | +0.007 | +0.027 | [+0.008, +0.048] p=1.00 | +0.133 |
| r2 | 109 | 0.717 | 0.763 | +0.014 | +0.012 | [-0.001, +0.025] p=0.97 | +0.121 |

### Missing mass (both tracks)

| round | track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO jackknife | cross-proposer recapture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r1 | A | 60 | 4 | 2 | 50 | 45 | 3 | 0.750 | [0.756, 0.822] | 0.10 |
| r1 | B | 40 | 4 | 2 | 27 | 20 | 5 | 0.500 | [0.500, 0.700] | 0.26 |
| r2 | A | 60 | 4 | 2 | 50 | 42 | 7 | 0.700 | [0.711, 0.778] | 0.12 |
| r2 | B | 40 | 4 | 2 | 31 | 27 | 1 | 0.675 | [0.633, 0.733] | 0.13 |

### Round bookkeeping

- r1: routing A=14 / B=11 (mixed 9), misrouting 0.04, planted probes 4/4; anchors pos-vs-neg AUC 0.642, coherent-vs-scrambled 0.970 (PASS), collapsed criteria 0, NA rate 0.013
- r2: routing A=12 / B=13 (mixed 10), misrouting 0.12, planted probes 4/4; anchors pos-vs-neg AUC 0.641, coherent-vs-scrambled 0.972 (PASS), collapsed criteria 0, NA rate 0.011

## cap_crowd — humor caption crowd-C (median split, votes>=100)

Population n=10893; HONEST (dense-held-out) n=2190; MONITOR n=1072 (32 groups). Round-0: T 0.555 / VA_nl 0.624 / **Δ_beyond -0.068** on HONEST; T 0.547 / VA_nl 0.646 / Δ -0.099 on MONITOR.

### Spurious map (headline)

| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |
|---|---:|---:|---|:--:|
| Enters mid-conversation and implies an unheard prior turn | 0.533 | 0.529 | surface-only | YES |
| Morbid/dark content (death, injury, loss) as the comic engine | 0.530 | 0.524 | surface-only |  |
| Specific persona markers (named addressee or verbal tic) | 0.470 | 0.470 | surface-only | YES |
| Caption length / verbosity (word count, clause count) | 0.530 | 0.544 | surface-only | YES |
| Joke is inert as standalone text; needs the drawing to complete | 0.479 | 0.474 | surface-only | YES |
| Density of identifiable wordplay/puns | 0.514 | 0.520 | surface-only | YES |
| Explicit lexical naming of the depicted object or setting | 0.511 | 0.517 | submitter's familiarity with the contest's conventions (novice one-off entrant vs. repeat veteran entrant) | YES |
| Numeric logistical specificity | 0.511 | 0.501 | surface-only | YES |
| Institutional register | 0.510 | 0.511 | surface-only |  |
| Fragmentary minimalism | 0.491 | 0.492 | surface-only | YES |
| Kinship framing and first-name familiar address | 0.491 | 0.481 | submitter's life stage and household composition (age cohort, parenthood, marital status) |  |
| Discourse-marker hedge opener (veteran stylistic tic) | 0.509 | 0.520 | author's contest experience / seniority in the submitter community | YES |
| Pun spelled out vs. left compressed for the reader to complete | 0.506 | 0.511 | surface-only | YES |
| Named partisan political figures and outlets | 0.502 | 0.500 | the submitter's political community and the assumed politics of the venue's readership |  |
| Time-sensitive framing | 0.501 | 0.513 | submission timing | YES |
| Question-format framing | 0.499 | 0.493 | surface-only |  |

### Discount

| readout | spurious-alone AUC (joint) | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL B channels, HONEST q10 | 0.580 | 0.543 | 0.621 | -0.079 | 0.555 | 0.634 | -0.078 |
| ALL B channels, MONITOR q5 | 0.580 | 0.531 | 0.632 | -0.101 | 0.547 | 0.650 | -0.103 |
| STRICT (mixed dropped), HONEST q10 | 0.540 | 0.556 | 0.633 | -0.076 | 0.555 | 0.634 | -0.078 |
| STRICT (mixed dropped), MONITOR q5 | 0.540 | 0.546 | 0.647 | -0.101 | 0.547 | 0.650 | -0.103 |

### Stacked increment

| population | AUC(B) | AUC(dense) | AUC(bank) | AUC(B+dense) | dense increment | AUC(B+bank) | bank increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| HONEST (n=2190) | 0.580 | 0.555 | 0.634 | 0.585 | +0.005 | 0.637 | +0.056 |
| MONITOR (n=1072) | 0.590 | 0.547 | 0.650 | 0.591 | +0.001 | 0.650 | +0.060 |

### Track A (secondary)

| round | bank features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |
|---|---:|---:|---:|---:|---:|---|---:|
| r1 | 377 | 0.652 | 0.630 | +0.006 | +0.007 | [-0.001, +0.015] p=0.96 | -0.075 |
| r2 | 386 | 0.650 | 0.634 | -0.002 | +0.003 | [-0.002, +0.009] p=0.88 | -0.078 |

### Missing mass (both tracks)

| round | track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO jackknife | cross-proposer recapture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r1 | A | 60 | 4 | 2 | 42 | 25 | 16 | 0.417 | [0.222, 0.778] | 0.33 |
| r1 | B | 40 | 4 | 2 | 34 | 30 | 2 | 0.750 | [0.733, 0.867] | 0.12 |
| r2 | A | 60 | 4 | 2 | 39 | 30 | 4 | 0.500 | [0.444, 0.667] | 0.23 |
| r2 | B | 40 | 4 | 2 | 27 | 20 | 3 | 0.500 | [0.467, 0.600] | 0.22 |

### Round bookkeeping

- r1: routing A=15 / B=10 (mixed 8), misrouting 0.00, planted probes 2/4; anchors pos-vs-neg AUC 0.622, coherent-vs-scrambled 1.000 (PASS), collapsed criteria 2, NA rate 0.006
- r2: routing A=9 / B=16 (mixed 11), misrouting 0.24, planted probes 4/4; anchors pos-vs-neg AUC 0.584, coherent-vs-scrambled 0.981 (PASS), collapsed criteria 0, NA rate 0.003

## cap_finalist — humor caption finalist-B (finalist vs hard negative)

Population n=5218; HONEST (dense-held-out) n=1055; MONITOR n=526 (23 groups). Round-0: T 0.612 / VA_nl 0.669 / **Δ_beyond -0.057** on HONEST; T 0.650 / VA_nl 0.692 / Δ -0.042 on MONITOR.

### Spurious map (headline)

| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |
|---|---:|---:|---|:--:|
| Cultural specificity | 0.405 | 0.421 | author cultural network or audience familiarity | YES |
| Identity-role markers | 0.457 | 0.485 | author demographic or social position | YES |
| Off-colour, bodily and dark-subject content | 0.464 | 0.472 | surface-only |  |
| Quantification density | 0.481 | 0.464 | surface-only | YES |
| Phonetic wordplay compressed into a single word | 0.485 | 0.483 | surface-only |  |
| Pandemic or geopolitical event-pinning vocabulary | 0.486 | 0.474 | submission timing pinned to a specific real-world event or era (pandemic, geopolitical conflict) rather than a timeless reading of the cartoon | YES |
| Stock dialogue scaffolding | 0.489 | 0.504 | professional editing or contest coaching | YES |
| Genre-pastiche of a non-cartoon written register | 0.489 | 0.479 | surface-only | YES |
| Staccato repetition or fragmenting punctuation | 0.493 | 0.462 | surface-only |  |
| Copyediting normalization | 0.504 | 0.507 | professional editing | YES |

### Discount

| readout | spurious-alone AUC (joint) | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL B channels, HONEST q10 | 0.598 | 0.582 | 0.640 | -0.057 | 0.612 | 0.664 | -0.052 |
| ALL B channels, MONITOR q5 | 0.598 | 0.606 | 0.708 | -0.102 | 0.650 | 0.716 | -0.066 |
| STRICT (mixed dropped), HONEST q10 | 0.538 | 0.609 | 0.665 | -0.056 | 0.612 | 0.664 | -0.052 |
| STRICT (mixed dropped), MONITOR q5 | 0.538 | 0.639 | 0.710 | -0.071 | 0.650 | 0.716 | -0.066 |

### Stacked increment

| population | AUC(B) | AUC(dense) | AUC(bank) | AUC(B+dense) | dense increment | AUC(B+bank) | bank increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| HONEST (n=1055) | 0.598 | 0.612 | 0.664 | 0.625 | +0.027 | 0.663 | +0.065 |
| MONITOR (n=526) | 0.626 | 0.650 | 0.716 | 0.669 | +0.042 | 0.714 | +0.087 |

### Track A (secondary)

| round | bank features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |
|---|---:|---:|---:|---:|---:|---|---:|
| r1 | 360 | 0.707 | 0.679 | +0.016 | +0.010 | [-0.010, +0.031] p=0.83 | -0.067 |
| r2 | 375 | 0.716 | 0.664 | +0.008 | -0.015 | [-0.033, +0.005] p=0.07 | -0.052 |

### Missing mass (both tracks)

| round | track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO jackknife | cross-proposer recapture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r1 | A | 60 | 4 | 2 | 34 | 24 | 7 | 0.400 | [0.378, 0.533] | 0.26 |
| r1 | B | 40 | 4 | 2 | 30 | 24 | 3 | 0.600 | [0.600, 0.700] | 0.20 |
| r2 | A | 60 | 4 | 2 | 35 | 23 | 7 | 0.383 | [0.444, 0.489] | 0.31 |
| r2 | B | 40 | 4 | 2 | 29 | 23 | 3 | 0.575 | [0.533, 0.667] | 0.17 |

### Round bookkeeping

- r1: routing A=15 / B=10 (mixed 6), misrouting 0.04, planted probes 4/4; anchors pos-vs-neg AUC 0.398, coherent-vs-scrambled 1.000 (PASS), collapsed criteria 0, NA rate 0.002
- r2: routing A=15 / B=10 (mixed 7), misrouting 0.00, planted probes 4/4; anchors pos-vs-neg AUC 0.368, coherent-vs-scrambled 0.984 (PASS), collapsed criteria 1, NA rate 0.002

## nc_outcome — N&C outcome (agency response outcome)

Population n=7084; HONEST (dense-held-out) n=1417; MONITOR n=694 (332 groups). Round-0: T 0.624 / VA_nl 0.632 / **Δ_beyond -0.008** on HONEST; T 0.616 / VA_nl 0.631 / Δ -0.015 on MONITOR.

### Spurious map (headline)

| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |
|---|---:|---:|---|:--:|
| Submission artifact markers | 0.580 | 0.557 | submission workflow and document-ingestion process |  |
| Personal and local grounding | 0.565 | 0.581 | author proximity and lived experience | YES |
| Institutional status cues | 0.564 | 0.553 | author seniority and institutional resources | YES |
| Overall length and verbosity of the submission | 0.556 | 0.556 | surface-only | YES |
| Timing and urgency cues | 0.550 | 0.553 | submission timing and available preparation time | YES |
| Written for this docket versus repurposed campaign or marketing copy | 0.544 | 0.534 | surface-only | YES |
| Boilerplate language | 0.536 | 0.526 | campaign or communications coordination |  |
| Signatory's seniority title paired with a direct personal contact line | 0.529 | 0.530 | author seniority | YES |
| OCR and scan-degradation noise | 0.522 | 0.503 | document-production and submission medium (fax or paper scan versus a native electronic file) |  |
| Recitation of the submitter organization's size or market share | 0.483 | 0.477 | institutional resources/reputation | YES |
| Coalition breadth | 0.510 | 0.488 | social and organizational network access | YES |

### Discount

| readout | spurious-alone AUC (joint) | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL B channels, HONEST q10 | 0.621 | 0.583 | 0.578 | +0.004 | 0.624 | 0.625 | -0.002 |
| ALL B channels, MONITOR q5 | 0.621 | 0.580 | 0.592 | -0.012 | 0.616 | 0.625 | -0.009 |
| STRICT (mixed dropped), HONEST q10 | 0.596 | 0.598 | 0.594 | +0.004 | 0.624 | 0.625 | -0.002 |
| STRICT (mixed dropped), MONITOR q5 | 0.596 | 0.590 | 0.596 | -0.006 | 0.616 | 0.625 | -0.009 |

### Stacked increment

| population | AUC(B) | AUC(dense) | AUC(bank) | AUC(B+dense) | dense increment | AUC(B+bank) | bank increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| HONEST (n=1417) | 0.621 | 0.624 | 0.625 | 0.647 | +0.026 | 0.645 | +0.025 |
| MONITOR (n=694) | 0.617 | 0.616 | 0.625 | 0.628 | +0.011 | 0.629 | +0.012 |

### Track A (secondary)

| round | bank features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |
|---|---:|---:|---:|---:|---:|---|---:|
| r1 | 209 | 0.636 | 0.632 | +0.005 | -0.000 | [-0.010, +0.009] p=0.49 | -0.008 |
| r2 | 223 | 0.625 | 0.625 | -0.011 | -0.007 | [-0.017, +0.003] p=0.10 | -0.002 |

### Missing mass (both tracks)

| round | track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO jackknife | cross-proposer recapture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r1 | A | 60 | 4 | 2 | 29 | 23 | 5 | 0.383 | [0.356, 0.511] | 0.21 |
| r1 | B | 40 | 4 | 2 | 35 | 31 | 3 | 0.775 | [0.767, 0.867] | 0.11 |
| r2 | A | 60 | 4 | 2 | 36 | 30 | 2 | 0.500 | [0.378, 0.689] | 0.17 |
| r2 | B | 40 | 4 | 2 | 33 | 26 | 7 | 0.650 | [0.733, 0.800] | 0.21 |

### Round bookkeeping

- r1: routing A=14 / B=11 (mixed 9), misrouting 0.04, planted probes 4/4; anchors pos-vs-neg AUC 0.718, coherent-vs-scrambled 0.945 (PASS), collapsed criteria 0, NA rate 0.002
- r2: routing A=14 / B=11 (mixed 8), misrouting 0.04, planted probes 4/4; anchors pos-vs-neg AUC 0.667, coherent-vs-scrambled 0.995 (PASS), collapsed criteria 0, NA rate 0.002

## nc_agree — N&C agree (agency agrees vs disagrees)

Population n=5046; HONEST (dense-held-out) n=1009; MONITOR n=487 (230 groups). Round-0: T 0.603 / VA_nl 0.626 / **Δ_beyond -0.022** on HONEST; T 0.601 / VA_nl 0.617 / Δ -0.016 on MONITOR.

### Spurious map (headline)

| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |
|---|---:|---:|---|:--:|
| Contested consequential question versus ministerial housekeeping | 0.383 | 0.396 | surface-only | YES |
| Technical identifier density | 0.605 | 0.624 | surface-only | YES |
| Formal address to decision-maker as submission | 0.565 | 0.579 | surface-only | YES |
| Legal citation density | 0.551 | 0.541 | professional editing | YES |
| Submission-artifact markers | 0.546 | 0.543 | submission pipeline |  |
| Representational-mandate and constituency-size self-description | 0.545 | 0.527 | institutional resources, membership scale and network of the submitter | YES |
| Density of formal legal/regulatory citations | 0.536 | 0.522 | professional legal editing/assistance | YES |
| Explicit structural organization of distinct points | 0.531 | 0.524 | surface-only | YES |
| OCR and transcription artifacts | 0.525 | 0.528 | document conversion or archival processing |  |
| Individually composed rather than templated or duplicated | 0.522 | 0.531 | surface-only | YES |
| Procedural posture about deadlines, extensions and prior filings | 0.517 | 0.543 | submission timing and process dynamics on the receiving side (comment-period length, docket renumbering, calendar conflicts) |  |
| Deadline and urgency language | 0.509 | 0.523 | submission timing |  |
| Document bulk and verbosity | 0.504 | 0.547 | surface-only | YES |
| Orthographic noncompliance and unedited-typing residue | 0.502 | 0.523 | surface-only |  |

### Discount

| readout | spurious-alone AUC (joint) | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL B channels, HONEST q10 | 0.607 | 0.579 | 0.602 | -0.023 | 0.603 | 0.632 | -0.028 |
| ALL B channels, MONITOR q5 | 0.607 | 0.588 | 0.580 | +0.008 | 0.601 | 0.606 | -0.005 |
| STRICT (mixed dropped), HONEST q10 | 0.544 | 0.589 | 0.623 | -0.034 | 0.603 | 0.632 | -0.028 |
| STRICT (mixed dropped), MONITOR q5 | 0.544 | 0.592 | 0.596 | -0.003 | 0.601 | 0.606 | -0.005 |

### Stacked increment

| population | AUC(B) | AUC(dense) | AUC(bank) | AUC(B+dense) | dense increment | AUC(B+bank) | bank increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| HONEST (n=1009) | 0.607 | 0.603 | 0.632 | 0.627 | +0.020 | 0.634 | +0.027 |
| MONITOR (n=487) | 0.595 | 0.601 | 0.606 | 0.603 | +0.008 | 0.593 | -0.002 |

### Track A (secondary)

| round | bank features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |
|---|---:|---:|---:|---:|---:|---|---:|
| r1 | 206 | 0.606 | 0.624 | -0.011 | -0.002 | [-0.020, +0.017] p=0.43 | -0.021 |
| r2 | 217 | 0.606 | 0.632 | -0.000 | +0.008 | [-0.005, +0.021] p=0.89 | -0.028 |

### Missing mass (both tracks)

| round | track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO jackknife | cross-proposer recapture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r1 | A | 60 | 4 | 2 | 29 | 21 | 5 | 0.350 | [0.222, 0.556] | 0.28 |
| r1 | B | 40 | 4 | 2 | 27 | 18 | 5 | 0.450 | [0.433, 0.567] | 0.30 |
| r2 | A | 60 | 4 | 2 | 30 | 27 | 1 | 0.450 | [0.311, 0.578] | 0.10 |
| r2 | B | 40 | 4 | 2 | 34 | 30 | 2 | 0.750 | [0.700, 0.867] | 0.12 |

### Round bookkeeping

- r1: routing A=14 / B=11 (mixed 9), misrouting 0.12, planted probes 4/4; anchors pos-vs-neg AUC 0.590, coherent-vs-scrambled 0.983 (PASS), collapsed criteria 0, NA rate 0.005
- r2: routing A=11 / B=14 (mixed 9), misrouting 0.16, planted probes 4/4; anchors pos-vs-neg AUC 0.606, coherent-vs-scrambled 0.873 (PASS), collapsed criteria 0, NA rate 0.002


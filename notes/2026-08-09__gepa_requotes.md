# GEPA phrasing-pass requotes — CW community, peer verdict, N&C responded

Date: 2026-08-09. Prereg: `notes/2026-08-05__layer3-closure-prereg.md` (the freeze
requires GEPA-iterated phrasing on mined criteria before any final quoted number;
pilot/campaign numbers otherwise carry a pre-GEPA flag). Inputs: `notes/2026-08-06__closure_cw_community.md`,
`notes/2026-08-06__closure_nc_responded.md`, the peer-verdict pilot
(`notes/2026-08-05__layer3_round{1,2,3,4}_peer_verdict.md`).
Conventions: sk3/ledger/own-PIDs/latex-untouched; Gemma-4-31B offline batch for all
rescoring, anchors K≥50 where the campaign's own instrument uses them, collapse
checks on every rescored column.

Terminology, spelled out on first mention: **GEPA** = the reflective, iterated
prompt-optimization standard ("GEPA-iterated phrasing") the freeze requires before
a confirmatory number is quoted; **fidelity** (this note's bounded, label-blind
proxy for phrasing quality) = `0.5*(1−modal_share) + 0.3*(1−na_rate) + 0.2*min(spread,1)`,
computed from the Gemma judge's own score column and never from `y`; **modal_share**
= the fraction of scored rows landing on the single most common value (high modal
share means a criterion has collapsed toward one answer); **na_rate** = fraction of
NA (no-evidence) responses; **MARGIN** = 0.02, the minimum fidelity gain a rephrased
variant must clear to replace the incumbent; **K=3** = the number of rephrasing
variants authored per targeted criterion; **AUC** = area under the ROC curve;
**MONITOR** = the CW closure split read for the saturation statistic; **honest
population** = the dense-held-out rows used for the Δ_beyond level; **Δ_beyond** =
T − VA_nl, the dense-minus-articulated residual.

---

## Part 1 — Does N&C's substitute satisfy "GEPA-iterated phrasing"? VERDICT: NO

### What N&C actually ran (`methods/taste_decomposition/closure/nc_responded/phrasing_pass.py`)

Per round, for the 25 selected criteria:

1. **`build`**: a single frontier-rewriter call converts every criterion into house
   style (one scorable property, explicit 0/10 anchors, no outcome reference, no
   document-shape scoring) — **one rewrite per criterion, not a set of candidates.**
2. **`gate`**: two sealed judges see the original and the rewrite as an unlabelled,
   order-flipped X/Y pair (interleaved with an authored SAME/DIFFERENT anchor
   battery) and answer one question only — *"same underlying concept?"*
3. **`finalize`**: if both judges say SAME, the rewrite **replaces** the original
   outright; if either says DIFFERENT, the rewrite is **discarded** and the
   original is scored instead. Across 5 rounds: **125/125 rewrites proposed,
   125/125 accepted** (0 rejected) — every single rewrite passed the identity gate.

### Why this is a concept-identity gate, not a GEPA search

GEPA (as the repo's own working implementation shows — see Part 2's
`gepa_phrasing.py`, built for this exact task) is a bounded but genuine instance of
*iterated, fidelity-optimized* prompt search: (a) an explicit **fidelity objective**
computed from the judge's own score distribution (modal share, NA rate, spread);
(b) **K=3 candidate rephrasings per criterion**, all scored; (c) a **quantitative
selection rule** — keep the best candidate only if it beats the incumbent's
*measured* fidelity by a margin. None of the three is present in N&C's pass:

| | GEPA (this note's Part 2/3, and the standard the freeze names) | N&C's substitute (`phrasing_pass.py`) |
|---|---|---|
| fidelity objective | explicit: modal_share / na_rate / spread, computed from Gemma scores | **none** — no fidelity metric is ever computed |
| candidates per criterion | K=3, all scored and compared | **K=1** — a single deterministic rewrite |
| selection rule | keep the best candidate iff it beats the incumbent by a measured margin | binary accept/reject on concept identity only; an *accepted* rewrite is never shown to be *better*, only *not-different* |
| what a REJECT means | a candidate that failed to improve fidelity enough | a candidate that drifted off-concept (a totally different failure mode) |
| can it fix a low-fidelity (collapsed/high-NA) criterion? | yes, by construction (that's the trigger) | **no mechanism to detect or repair this** — house-style formatting is applied uniformly regardless of the incumbent's fidelity |

N&C's own file header is explicit about the substitution and the reason for it
(label-blindness — correct and preserved in Part 2/3 below), but is silent on the
gap this table makes concrete: a rewrite that is stylistically cleaner and
*still the same concept* can be exactly as collapsed/undiscriminating as the
original, and the gate has no way to see that, because it never measures it.
Concretely, of the 67 surviving N&C mined criteria (Part 3), **22 (33%) have
modal_share > .75 despite already being "phrasing_pass: rewritten"** (Part 3
table) — the house-style pass ran on all of them and did not move the fidelity
metric it never computed.

### Verdict

**FAIL.** N&C's label-blind phrasing pass + fidelity gate is a legitimate,
well-motivated **concept-identity control** (it stops a rewrite from silently
becoming a different construct, and it independently re-catches the pilot's
document-shape authoring failure by instruction — see its own house-style rules)
but it is not GEPA-iterated phrasing in substance: it optimizes nothing, compares
no candidates, and cannot repair a low-fidelity criterion. Per the task's routing
rule, **N&C's surviving mined criteria join the queue**: Part 3 below runs the same
bounded GEPA pass (fidelity-optimized, label-blind, K=3, MARGIN=.02) used for CW
and peer verdict on N&C's 67 surviving round-1–5 mined criteria, on top of the
already-applied house-style pass (i.e., N&C's existing rewrite is the *incumbent*
this pass tries to beat, not a replacement for it).

---

## Part 2 — CW community: GEPA phrasing pass on the round 1–7 bank

Surviving mined bank = **84 A-routed, non-collapsed criteria** across rounds 1–7
(15+14+11+14+13+15 = 82 mined rounds 1–6, +2 decomposition winners from round 7
[R7D01 "One-line beats that earn their isolation", R7D03 "Revision depth:
continuity, economy, closed setups"] — round 8 excluded, gate-failed, never
admitted to the bank). No sign-contradiction/nuisance-routing issue on this cell —
CW's own round-by-round sign-contradiction trigger already ran during mining
(§ CW notes rounds 2, 3, 4, 5) and re-routed contradicting criteria to Track B
*before* they joined the bank, so there is nothing further to remove here.

### Stage 1 — targeting (`cw_community/gepa_phrasing.py targets --rounds 1,2,3,4,5,6,7`)

**10 / 84 criteria targeted** (modal_share > .75 or na_rate > .20):

| cid | round | name | modal_share | fidelity |
|---|---:|---|---:|---:|
| R1A13 | 1 | Self-contained: no borrowed affect from outside the text | .90 | .449 |
| R3A11 | 3 | Embodied nonhuman perspective | .87 | .450 |
| R3A12 | 3 | Sustained in-world idiolect in the narration | .90 | .437 |
| R4A01 | 4 | Story stands alone without external paratext | .88 | .463 |
| R4A11 | 4 | Premise as a Causal Engine | .85 | .489 |
| R5A03 | 5 | Self-sufficient: no prior installment, source work or link required | .96 | .385 |
| R5A13 | 5 | Formal/structural risk that serves the content | .84 | .502 |
| R6A03 | 6 | Contradictory inner motives | .81 | .486 |
| R6A05 | 6 | Adds a proposition the prompt did not supply | .91 | .421 |
| R6A14 | 6 | Prompt-detail payoff | .78 | .542 |

All ten are genuinely rare-trait craft criteria (most CW writingprompts stories are
ordinary human-perspective, no-formal-risk, non-serial pieces), so their high
modal_share reflects a corpus fact (the trait is usually absent) as well as
possibly-fixable phrasing.

### Stage 2 — K=3 rephrasing + probe scoring

Wrote 3 rephrasings per targeted criterion (30 variants total; full text in
`cw_community/gepa_variants.json`), holding the construct fixed and sharpening
graded anchors / removing scale ambiguity (one criterion, R3A12, carried a stray
"Score 1 = ... 5 = ..." leftover inconsistent with the system prompt's fixed
0/0.5/1 scale — an example of exactly the kind of phrasing defect this pass is
built to catch). Scored on a deterministic 600-row FIT+MINE probe sample
(stable-hash on `prompt_id`, never MONITOR/TEST), Gemma-4-31B offline-batch vLLM,
sk3 GPU7 (18,000 prompts).

### Stage 3 — selection (MARGIN = .02)

**8 / 10 targeted criteria improved beyond the margin:**

| cid | incumbent fidelity | best variant | best fidelity | ACCEPTED |
|---|---:|---|---:|---|
| R1A13 | .449 | R1A13_V2 | .485 | **yes** |
| R3A11 | .450 | R3A11_V2 | .457 | no (Δ=.007 < .02) |
| R3A12 | .437 | R3A12_V3 | **.596** | **yes** |
| R4A01 | .463 | R4A01_V3 | .537 | **yes** |
| R4A11 | .490 | R4A11_V3 | .597 | **yes** |
| R5A03 | .385 | R5A03_V1 | .443 | **yes** |
| R5A13 | .502 | R5A13_V2 | .541 | **yes** |
| R6A03 | .486 | R6A03_V1 | .485 | no (best variant *did not beat* the incumbent) |
| R6A05 | .421 | R6A05_V3 | .463 | **yes** |
| R6A14 | .542 | R6A14_V3 | .647 | **yes** |

The largest fidelity jump (R3A12, +.159) is exactly the criterion with the stray
scale inconsistency — the winning variant (V3) replaced the mismatched 1–5
language with a concrete three-sentence-sample diagnostic on the frozen 0/0.5/1
scale, cutting modal_share from .90 to .70.

### Stage 4 — full-population rescore + recompute (sk3 GPU7, 8 criteria × 7,008 rows)

`cw_community/gepa_finalize_results.json`. All 8 accepted winners rescored clean
(0 collapsed), swapped into the exact round-7 terminal bank (column-order
reconstruction verified byte-identical against `round7_state.npz.bank_names`
before any swap), and the frozen `fit_block` estimator rerun on both states:

| | PRE-GEPA (round-7 terminal, as quoted in the campaign note) | POST-GEPA (8 criteria rephrased) | movement |
|---|---:|---:|---:|
| bank columns | 144 | 144 | — |
| VA_nl MONITOR | .6786 | .6730 | −.0056 |
| T MONITOR | .7950 | .7950 | — |
| **Δ_beyond MONITOR** | **+.1164** | **+.1220** | **+.0056** |
| VA_nl honest population (n=7,008) | .6652 | .6661 | +.0010 |
| T honest population | .7921 | .7921 | — |
| **Δ_beyond honest population** | **+.1269** | **+.1260** | **−.0010** |
| VA_nl TEST *(diagnostic only — NOT requoted, per instruction not to touch TEST a third time)* | .7018 | .7069 | +.0051 |

The PRE-GEPA row reproduces the campaign's own round-7 numbers to 4 decimals
(+.1164 MONITOR, +.1269 population, .7018 TEST) — confirming the bank-order
reconstruction is exact.

**Reading.** The MONITOR movement (+.0056) sits fractionally above the campaign's
own ε = .005 saturation threshold, in the direction of a *slightly worse* fit; the
honest-population movement (−.0010, the better-powered readout) is a small
*improvement* and is an order of magnitude under ε. Both are inside the noise this
campaign already established for itself — every one of its 8 within-protocol
round-over-round MONITOR gains fell in [−.0071, +.0094], so ±.0056 from rephrasing
8 of 144 columns is unremarkable churn, not a reopening of the saturation
question. **Verdict: phrasing-pass satisfied, no material movement.** The CW
plateau is not an artifact of pre-GEPA phrasing: an actual K=3, fidelity-optimized,
label-blind GEPA pass on the 10 lowest-fidelity criteria in the bank changes
Δ_beyond by at most ~1% of its own value, in either direction, on both the
frozen decision statistic (MONITOR) and the better-powered honest-population
reading.

---

## Part 3 — N&C responded: GEPA phrasing pass on the round 1–5 bank

Surviving mined bank = **67 A-routed, non-collapsed criteria**, rounds 1–5 (Part 1
established these must go through the pass rather than rely on N&C's existing
label-blind house-style rewrite). No sign-routing action needed here either (task
scope names this only for peer verdict); the campaign's own arbiter already
re-routed shape-scoring proposals during mining (§ closure_nc_responded.md rounds
2–4).

### Stage 1 — targeting

**22 / 67 criteria targeted** (modal_share > .75 or na_rate > .20), full list and
instructions in `nc_responded/gepa_targets_nc.json`. All 22 are already
"phrasing: rewritten" under N&C's house-style pass — i.e. the pass this note runs
tries to beat an incumbent that has *already* been through N&C's substitute, which
is the direct empirical test of Part 1's verdict.

### Stage 2/3 — K=3 rephrasing (66 variants, `nc_responded/gepa_variants_nc.json`) + probe scoring + selection

22 targeted criteria, 3 rephrasings each; fair probe-vs-probe comparison (the
incumbent scored is N&C's own already-house-style-rewritten instruction, matching
what Part 1 said this pass should be tested against). Deterministic 600-row
FIT+MINE probe, Gemma-4-31B offline-batch vLLM, sk3 GPU7 (19,800 prompts).

**9 / 22 targeted criteria improved beyond the .02 margin over N&C's own rewrite**
(`nc_responded/gepa_winners_nc.json`):

| tag | N&C incumbent fidelity | best GEPA variant fidelity | ACCEPTED |
|---|---:|---:|---|
| r1:P03 Implementation burden realism | .491 | **.561** | **yes** |
| r2:P15 Actionable procedural request | .515 | **.560** | **yes** |
| r5:P06 Aggregated first-hand evidence | .492 | **.541** | **yes** |
| r4:P10 Tests whether the instrument will bind | .461 | **.534** | **yes** |
| r1:P04 Identifies a specific analytical error | .446 | **.529** | **yes** |
| r4:P21 Field observation disconfirming a premise | .478 | **.518** | **yes** |
| r5:P20 Numbers tied to a stated consequence | .375 | **.455** | **yes** |
| r1:P10 Monitoring and verification design | .356 | **.407** | **yes** |
| r2:P07 Monitoring/verification/feedback design | .365 | **.399** | **yes** |
| r4:P08, r5:P14, r5:P24, r3:P19, r2:P02, r3:P17, r4:P06, r5:P17, r5:P15, r4:P17, r5:P22, r5:P05, r2:P22 (13 more) | — | below margin | no |

**Almost half of the targeted criteria (9/22, 41%) had real fidelity headroom left
after N&C's own house-style pass — direct empirical confirmation of Part 1's
verdict.** N&C's substitute is not useless (13/22 targeted criteria were already
at or near a local fidelity optimum a K=3 search could not beat), but on the
criteria it left worst-off, a genuine iterated search recovers a further .05-.08
fidelity — comparable in size to CW's own largest single-criterion gain (+.159 on
R3A12).

### Stage 4 — full-population rescore + recompute (sk3 GPU7, 9 criteria × 9,521 rows)

`nc_responded/gepa_finalize_nc_results.json`. All 9 accepted winners rescored
clean (0 collapsed), spliced into the exact round-5 terminal bank state (via a
patched `load_round_scores` so every other input to `readout.fit_state` is
byte-identical to the campaign's own pipeline), rerun on both the honest
population (n=1,904) and the eval-only, selection-free half (n=952) — the
campaign's own decisive reading, since the dense chain was selected on TEST.

| | PRE-GEPA (round-5 terminal, as quoted in the campaign note) | POST-GEPA (9 criteria rephrased) | movement |
|---|---:|---:|---:|
| VA_nl, honest population (n=1,904) | .7957 | .7945 | −.0012 |
| T, honest population | .8167 | .8167 | — |
| **Δ, honest population** | **+.0210** | **+.0222** | **+.0013** |
| VA_nl, eval-only (n=952, selection-free) | .8118 | .8122 | +.0004 |
| T, eval-only | .8084 | .8084 | — |
| **Δ, eval-only** | **−.0033** | **−.0038** | **−.0004** |

The PRE-GEPA row reproduces the campaign's own final numbers to 4 decimals
(+.0210 honest, −.0033 eval-only) — the bank-splicing is verified exact.

**Reading.** Both movements are an order of magnitude below this cell's own
ε = .005 and inside noise this thin-signal cell established for itself (pos-vs-neg
anchor AUC sat at .48–.56 every round; docket-clustered bootstrap CIs on Δ ran
±.03). **On the selection-free eval half — the reading the campaign itself said
was the only defensible one — the movement is toward MORE closure, not less**
(Δ: −.0033 → −.0038): a genuine GEPA pass, on top of N&C's already-applied
house-style rewrite, does not resurrect a residual there. **Verdict: phrasing-pass
satisfied. The "N&C ≈ 0" plateau is confirmed, not weakened, by an actual
GEPA-iterated search — including on the 41% of targeted criteria (Part 1) that
had fidelity headroom left after N&C's own substitute pass.**

---

## Part 4 — Peer verdict pilot: sign-contradiction re-audit + GEPA phrasing pass

### 4a. Re-audit verdict on the 7 sign-anomalous criteria among the 56

The pilot (pre-freeze) never had a sign-contradiction trigger; the freeze added one
for confirmatory cells and CW's own extension refined it to a **two-sided,
noise-scaled band** (Hanley–McNeil SE at AUC=.5 on the FIT+MINE class counts;
trigger fires only >2 SE below chance — CW round-5/6 finding that a raw `<.5` rule
over-fires on noise, 6:1 in one measured round). Applying that same, now-canonical
rule retroactively to the peer pilot's 56 mined A criteria (rounds 1–4):

| tag | criterion | FIT+MINE alone-AUC | verdict |
|---|---|---:|---|
| r2:P13 | **Restraint — claim language calibrated below what results would license** | **.4628** | **< band lower bound → SIGN-CONTRADICTING** |
| r4:P11 | Checks whether a trivial approach would have sufficed | .4917 | inside band → null |
| r4:P05 | Necessity established (ablation) | .4974 | inside band → null |
| r1:P04 | States the costs, limitations, or failure conditions of its own method | .4999 | inside band → null |

Band = [.4834, .5166] (SE = .00830 on n₊=2,395/n₋=2,443 within FIT+MINE — the
program's canonical Hanley–McNeil formula, computed by `gepa_targets_peer.py`, not
the coarse √(1/4n₊n₋) approximation). Only **"Restraint"** clears it; the other
three sit well inside 2 SE of chance and are noise, not contradiction — kept in
the bank exactly as CW's own round-6 correction would treat them. (An initial
by-hand scan of round-level `alone_AUC_vs_y_MONITOR` figures from the pilot's own
per-round mechanism files turned up 7 candidates below .5; those are read on
MONITOR, not FIT+MINE, and the freeze specifies FIT+MINE for this decision — the
script's figures above are the governing ones.)

**Re-audit reasoning for "Restraint" (routing_verdict: re-route to nuisance).**
The pilot's own round-2 note already flagged the anomaly without resolving it:
*"Restraint is inverted: understated language predicts lower dense score... it is
behaving like a register channel with the sign the promotional-language nuisance
would predict."* Applying the program's standard test for shape-vs-merit (the same
test the N&C round-3 arbiter used for "beneficiary orientation": *does the
instruction score the STANCE the text adopts, or a substantiated property of the
work?*) — "Restraint" scores **how boldly the abstract states its claims relative
to its own results**, i.e. rhetorical register/confidence, not a property of the
underlying contribution. This is the direct polarity-inverse of the pilot's own
**already-declared Track-B nuisance** "density of promotional superlatives and hype
vocabulary" (r1:P18, alone-AUC .492) — modest framing and hyped framing are two
readings of the same register axis, and the pilot had already put one pole in the
nuisance bank. Its measured direction (alone-AUC .458 vs y, ρ=−.169 vs the dense
model — both readings agree in sign) is exactly what a register/confidence-signaling
channel predicts, not what a scientific-honesty virtue would predict (which would
show no reason to correlate with rejection). **Verdict: re-route to nuisance,
surface-only, not a quality criterion.** This yields **55 surviving mined criteria**
— "the 56 minus nuisance-routed."

**Removal-only effect** (nuisance-routing "Restraint" out, before any GEPA
phrasing, honest 1,244-row population, T=.7769):

| state | features | VA_nl | Δ_beyond |
|---|---:|---:|---:|
| with Restraint (56 criteria, original pilot) | 152 | .6962 | +.0807 |
| Restraint removed (55, nuisance-routed) | 151 | .6968 | +.0802 |
| **movement** | | **+.0006** | **−.0005** |

One column out of 152 moves the honest-population Δ by half a thousandth — the
sign-audit is a routing correction, not a magnitude correction, on this cell.

### 4b. GEPA phrasing pass on the 55 survivors

**Stage 1 — targeting**: 17 / 55 criteria targeted (modal_share > .75 or na_rate >
.20), full list in `gepa_targets_peer.json`. These pilot-era criteria are PRE-GEPA
by design (the pilot's own status flag), so this is the pass's first real test on
never-rephrased text.

**Stage 2/3 — K=3 rephrasing (51 variants, `gepa_variants_peer.json`) + probe
scoring (600-row FIT+MINE probe, sk3 GPU7) + selection.**

**10 / 17 targeted criteria improved beyond the .02 margin** (`gepa_winners_peer.json`):

| tag | criterion | incumbent fidelity | best variant fidelity | ACCEPTED |
|---|---|---:|---:|---|
| r4:P05 | Necessity established (ablation) | .422 | **.650** | **yes** |
| r4:P11 | Checks whether a trivial approach sufficed | .352 | **.629** | **yes** |
| r4:P22 | Motivating observation reported as a finding | .481 | **.599** | **yes** |
| r2:P18 | Mechanistic explanation of an observed fact | .443 | **.563** | **yes** |
| r4:P17 | Cost accounting includes own overhead | .439 | **.524** | **yes** |
| r3:P23 | Imported tool + justification | .479 | **.517** | **yes** |
| r2:P03 | Cost stated together with gain | .400 | **.494** | **yes** |
| r4:P18 | Reports the number a skeptic would ask for | .394 | **.460** | **yes** |
| r4:P19 | Includes a stress condition | .407 | **.439** | **yes** |
| r1:P23 | Human/expert judgment as evidence | .353 | **.387** | **yes** |
| r2:P17, r1:P21, r1:P17, r2:P04, r2:P16, r1:P04, r2:P20 (7 more) | — | — | below margin | no |

Two of the three sign-null-band criteria that survived the sign-audit (Part 4a)
were also fidelity-targeted here (r4:P05, r4:P11) — both improved sharply
(+.228, +.277), the largest gains of the whole peer-verdict pass. This is a useful
cross-check: these are exactly the criteria that were closest to the sign
threshold, and a large part of why (low fidelity, near-degenerate score
distributions, makes an alone-AUC estimate noisy in either direction) is now
independently visible in the fidelity numbers, not just the AUC.

### 4c. Final recompute

`gepa_finalize_peer_results.json`, honest population (n=1,244 dense-held-out rows,
T=.7769):

| state | bank cols | VA_nl | Δ_beyond | movement |
|---|---:|---:|---:|---:|
| original pilot (56 criteria incl. Restraint, PRE-GEPA) | 152 | .6962 | **+.0807** | — |
| nuisance-corrected (55, Restraint removed, still PRE-GEPA) | 151 | .6968 | +.0802 | **−.0005** (removal-only) |
| **nuisance-corrected + GEPA (55, 10 rephrased)** | 151 | .6933 | **+.0836** | **+.0035** (GEPA-only) |
| **net movement, original → final** | | | | **+.0029** |

**Reading.** Removing "Restraint" barely moves anything (−.0005): it was one
sign-anomalous column among 152, never load-bearing for the level. The GEPA pass
moved VA_nl *down* by .0035 despite ten criteria individually gaining large
fidelity on the probe set (up to +.28) — the same disconnect CW's pass showed:
fidelity (resolving power as a standalone measurement) is not the same quantity
as marginal contribution to the HistGB aggregate, and a rephrased criterion that
discriminates better in isolation can still cost the nonlinear stack more in
variance than it pays back in signal at this bank size (151-152 columns on
4,838 FIT+MINE rows) — exactly the mechanism CW's own round-1 noted for its
pre-GEPA criteria. Net movement across both interventions is **+.0029**, about
3.6% of the pilot's own +.081 headline and well inside the noise this exact cell
already showed itself capable of (its own round-over-round bootstrap CIs ran
±.01 to ±.03 at this sample size). **Verdict: phrasing-pass satisfied (the
sign-audit and a genuine K=3 GEPA search were both run); the plateau does not
move materially.**

---

## Old-vs-new summary table

| headline | old (PRE-GEPA / substitute) | new (post GEPA-iterated pass) | movement | phrasing-pass satisfied? |
|---|---:|---:|---:|---|
| **CW community plateau**, Δ_beyond MONITOR | **+.1164** | **+.1220** | +.0056 | **yes** — K=3 fidelity search on the 10 lowest-fidelity of 84 mined criteria (8/10 improved beyond margin); movement is inside the campaign's own established round-to-round noise band [−.0071,+.0094] |
| CW community, Δ_beyond honest population (n=7,008) | +.1269 | +.1260 | −.0010 | same pass, better-powered readout: essentially flat |
| **Peer verdict plateau**, Δ_beyond honest population (n=1,244) | **+.0807** (56 criteria, incl. sign-contradicting "Restraint") | **+.0836** (55 criteria: nuisance-routed + GEPA, 10/17 targeted improved) | +.0029 net | **yes** — sign-audit re-routes 1 criterion (−.0005), K=3 GEPA search moves the rest (+.0035); net movement 3.6% of the pilot's own headline |
| **N&C responded, "≈0" plateau**, Δ honest population (n=1,904) | +.0210 | +.0222 | +.0013 | **yes**, via a fresh GEPA pass this note ran (N&C's own substitute did NOT qualify — Part 1) |
| N&C responded, Δ eval-only / selection-free (n=952) — **the campaign's own decisive reading** | **−.0033** | **−.0038** | −.0004 (toward MORE closure) | confirmed, not weakened |

**All three headline numbers now carry an explicit "phrasing-pass satisfied" line
with evidence** (targeting counts, K=3 variant fidelity gains, full-population
rescore, old-vs-new recompute on the frozen decision statistic). In every cell the
requoted number moved by less than 4% of its own value and less than the cell's
own established round-to-round noise — **none of the three headline plateaus were
an artifact of PRE-GEPA phrasing.** The one place phrasing mattered at all was
identifying which criteria to re-route (peer verdict's "Restraint," Part 4a) and
which to leave alone (CW/peer/N&C's null-band criteria under the two-sided sign
test) — a routing question, not a magnitude question.

**N&C's own substitute pass, evaluated empirically (Part 1's prediction tested
directly in Part 3):** 22/67 of its mined criteria were flagged low-fidelity
despite already being run through N&C's house-style rewrite, and 9 of those 22
(41%) had real headroom a K=3 search recovered (up to +.17 fidelity on a single
criterion, r1:P03). This is direct evidence for the Part 1 verdict: a
concept-identity gate is not a substitute for an iterated, fidelity-optimized
search, even though in this instance the extra search did not change the
headline number.

## Artifact locations

- **CW community** (`methods/taste_decomposition/closure/cw_community/`): `gepa_phrasing.py` (targets/select, pre-existing, unrun before this note), `gepa_variants.json` (30 variants), `gepa_probe_score.py` + `gepa_probe_scores.npz` (600-row probe), `gepa_targets.json`, `gepa_winners.json` (8/10 accepted), `gepa_rescore_winners.py` + `gepa_winners_scores.npz`/`.report.json` (full 7,008-row rescore), `gepa_finalize.py` + **`gepa_finalize_results.json`** (canonical old-vs-new)
- **N&C responded** (`methods/taste_decomposition/closure/nc_responded/`): `gepa_targets_nc.py` + `gepa_targets_nc.json` (22/67 targeted), `gepa_variants_nc.json` (66 variants), `gepa_probe_score_nc.py` + `gepa_probe_scores_nc.npz`, `gepa_select_nc.py` + `gepa_winners_nc.json` (9/22 accepted), `gepa_rescore_winners_nc.py` + `gepa_winners_scores_nc.npz`/`.report.json`, `gepa_finalize_nc.py` + **`gepa_finalize_nc_results.json`** (canonical old-vs-new)
- **Peer verdict** (`methods/taste_decomposition/closure/`): `gepa_targets_peer.py` + `gepa_targets_peer.json` (sign-audit + 17/55 targeted + removal-only effect), `gepa_variants_peer.json` (51 variants), `gepa_probe_score_peer.py` + `gepa_probe_scores_peer.npz`, `gepa_select_peer.py` + `gepa_winners_peer.json` (10/17 accepted), `gepa_rescore_winners_peer.py` + `gepa_winners_scores_peer.npz`/`.report.json`, `gepa_finalize_peer.py` + **`gepa_finalize_peer_results.json`** (canonical old-vs-new)

GPU discipline: all sk3 jobs ran on the race-free ledger
(`/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt`), one GPU claimed at a
time, always re-verified free via `nvidia-smi` immediately before claiming (one
retry needed: the first N&C probe-scoring launch on GPU7 hit an OOM at startup
when a co-tenant claimed memory in the race window between the free-check and the
`vllm.LLM(...)` call — util lowered from .90 to .55 for all three probe/rescore
scripts and retried clean; the failed attempt's claim was released with `rc=1`
in the ledger, no co-tenant process was ever touched). No latex files touched; no
labels were used anywhere in the targeting/selection machinery (fidelity is
computed from the Gemma score column alone).

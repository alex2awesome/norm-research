# W1 code + results audit (user-requested: "lots of failures — audit this")

Date: 2026-07-23. Scope: `battery/passes.py`, `run_variant_pass.py`,
`run_reason_first_pass.py`, `tally_w1a.py`, `tally_w1b.py`, and every W1a/W1b headline
number. Method: full-prompt reconstruction, independent recomputation on a fresh code path
(scipy, raw npz), null-model checks, degeneracy checks, hand verification.

## Verdict in one line

The COMPUTATIONS are clean (every checked number reproduces exactly on an independent
path); the scoring path is validated by the ρ≥.999 acceptance gates; the "failures" split
into (i) three NEW instrument defects found by this audit — now corrected or gated,
(ii) honest frozen-prediction failures that are real and informative, and (iii) findings
that survive with reduced magnitude. One headline framing is RETRACTED (below).

## A. Computation audit — CLEAN

- Independent scipy recomputation from raw npz (no shared code with the tallies):
  14B CoT-delta median −.1135 (tally −.112/−.121 by strata) ✓; 14B P-B5 −.7291 (tally
  −.729, n=90) ✓.
- not_gap arithmetic hand-verified on an adapter cell: tf +.624, ρ(negated, −target)
  −.656 → gap +1.280 ✓ (gaps >1 are legitimate: they mean the model under NOT still
  tracks the POSITIVE policy).
- Masks (adapter=half-2), form selection (canonical for W1b), and adverse-over-forms
  verified in place. tally_w1a has one dead no-op loop (cosmetic only).

## B. Instrument defects found by THIS audit (beyond the two already caught)

1. **Exclusion prompt is internally contradictory.** The inversion instruction rides the
   {rubric} slot, but the frozen template then ends with the STRAIGHT question ("Does the
   text satisfy the criterion? Answer ... YES or NO."). leak_self therefore measures
   instruction-CONFLICT resolution, not clean Jacoby suppression; it is an upper bound on
   rigidity. All configs faced the same contradiction, so the comparative gradient retains
   meaning; the absolute levels do not. W1c fix: variant-consistent final question.
2. **leak_self needs a cross-cell null.** ρ(exclusion_c, tf_OTHER-cell): 7B base +.12,
   **adapter +.55**, 14B −.14. Most of the adapter's headline +.93 is its generic
   cross-cell vector correlation (installed shared factor + under-dispersion), not
   construct-specific leak. **Corrected statistic (same-cell minus cross-cell): 7B +.29,
   adapter +.39, 14B −.19.** The "near-total rigidity +.93" framing is RETRACTED; the
   gradient survives with modest magnitudes.
3. **guess-quartile chance is .664, not .5** (item_agreement of random ranks; verified
   empirically .6644). "Agreement .80 on lowest-confidence items" = modestly above chance,
   not dramatic. All Dienes-signature phrasing downgraded accordingly.

(Previously caught, still standing: holistic floor-collapse; verbalized-confidence scale
degeneracy at both 7B configs — 94% constant "85" / binary {0,100}.)

## C. Prediction failures — REAL, not bugs

- **P-B5 (−.729 @14B) is systematic and now has a mechanism:** across cells,
  logodds-conf-acc tracks competence (ρ +.52 with tf_rho) while verbal-conf-acc
  ANTI-tracks it (−.34), and their anti-correlation persists within competence strata
  (−.84 high-tf half, −.59 low-tf half) — not a third-variable artifact. The two
  "confidence" operationalizations measure opposed things. This is the
  multi-operationalization lesson working as designed: the constructs dissociate; the v0
  proxy retirement stands.
- P-W1b-1 (interference rung-general) and P-W1a-3 (composition parity) reproduce and
  stand as honest failures of the frozen directions.

## D. What survives the audit

The instructed-inversion gradient survives on THREE independent legs, none sharing the
defective statistic: (1) differenced leak +.39/+.29/−.19; (2) the CLEAN negation
instrument (criterion redefinition is internally consistent): adapter under NOT tracks the
positive policy at +.66 while 14B genuinely inverts on B2 cells; (3) base-rate semantics:
under the inversion instruction 14B flips its mean p_yes .09 → .69 (exactly consistent
with executing "YES = fails"), while 7B (.17 → .09) and the adapter (.16 → .11) do not
move base rates. Also surviving: adapter interference-robustness (CoT-delta −.035 vs
−.134 base, recomputed), the composition-parity null, rubric-sensitivity (adapter
cross-cell tf ρ .51 — elevated vs base .44 but NOT rubric-blind; the adapter still
differentiates constructs).

## E. Why so many failures? (the meta-answer)

Three sources, none of them pipeline rot: (1) ~half the operationalizations are being run
for the first time anywhere, and first-run instruments have first-run defects — the gates
(acceptance, distribution checks, nulls, prereg bars) are catching them ON TIME, before
anything was quoted confirmatory; (2) the frozen directional predictions were deliberately
falsifiable bets from the literature, and several literatures are simply wrong about this
regime (Beilock boundary condition, symbols-compose); (3) the one repeated true failure
mode — degenerate score distributions — has now produced a standing gate three times
(judge collapse, holistic floor, confidence scale), and the scale-use gate generalizes it.

## Corrections executed

- Readout note: leak table re-stated with differenced statistics; RETRACTION of the +.93
  framing; chance-level context added; Dienes claims re-scoped.
- Prereg addenda: exclusion-instrument defect + corrected statistic declared; W1c fix
  specified (variant-consistent final question + cross-cell null as standard companion).

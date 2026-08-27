# CW decompression grid v1 — first Face-2 curves (46 metrics × 6 type rungs × 3 readers)

*Run overnight 2026-07-02 (report 06:33). Messages written once by local 70B from {name + R3
rubric}; readers Llama-3.2-1B / 3.2-3B / 3.1-8B, 300 probes, verbal rungs orbit-averaged over 3
forms, exemplars (k=2, ≤400-char excerpts) held out of evaluation. Reference = corrected orbit
target m̄ (aligned_8b_orbit_v2). Artifacts: `outputs/r3_cw/grid_cw_v1/{messages.json,
grid_*.npz, report.json}`. bal_acc = balanced accuracy vs reference; span_r2 = CV-R² of the rung
judge on the metric's criteria/species basis (in-span vs out-of-span); self_agree = agreement
with the same reader's dossier judgment (within-reader curve, no cross-executor conflation).*

## Median curves (46 metrics)

*8B column = self-consistency reference (scored vs its own orbit target — see CORRECTION below,
NOT a cross-reader data point). Clean gap column = 3B−1B (neither is the reference executor).*

| rung | 8B (self-ref) | 3B bal/span | 1B bal/span | 3B−1B gap (clean) |
|---|---|---|---|---|
| name | .793 / .92 | .634 / .44 | .565 / .31 | +.060 |
| definition | .849 / .91 | .688 / .48 | .558 / .30 | **+.133** |
| explanation | .781 / .88 | **.692** / .50 | .551 / .31 | **+.139** |
| full_rubric | .927 / .91 | .682 / .45 | .558 / .31 | +.121 |
| exemplars | .643 / .46 | .530 / .21 | .541 / .14 | −.006 |
| dossier | .658 / .62 | .636 / .39 | .521 / .18 | +.106 |

## ⚠ CORRECTION (2026-07-02 audit) — the 8B-reader numbers are self-referential

The reference target is the **8B executor's** orbit reading (`aligned_8b_orbit_v2`), and 8B is one
of the readers. So every 8B-reader cell is scored against a target 8B itself generated: 8B
full_rubric = .927 is near-ceiling BY CONSTRUCTION (self-consistency), not a decompression finding,
and any reader gap computed as **8B − 3B is contaminated** (the earlier "+.245 on full_rubric"
headline is RETRACTED). The clean cross-reader comparison uses readers NEITHER of which is the
reference executor. Fix forward: feature **70B − 3B** (evening, once 70B reader scored; reference
stays 8B so both are non-reference); treat the 8B-reader row only as the self-consistency reference;
the WITHIN-8B curve (name .79 → rubric .93) is still legible as "how much the 8B reading depends on
message richness," just not a cross-reader gap. Also apply an H_M ≥ 0.15-bit floor (1 CW metric has a
near-constant target = vacuous). Messages verified legible (coherent English, length-matched
def/expl); signatures NOT collapsed (rowstd .11–.21, 0% degenerate) — those are clean.

## The findings (corrected)

1. **The compression→capability trade — clean version (3B vs 1B, neither is the reference).** The
   reader gap is SMALLEST at the bare name (+.060) and largest at definition/explanation
   (+.133/+.139), ~zero at exemplars (−.006). Reading: both weak readers just retrieve what little
   the name indexes (small gap); the stronger of the two can USE verbal unpacking while the weaker
   cannot, so the gap opens exactly at the index→articulated-content transition; neither weak reader
   can decompress ostension at k=2 (no gap). Same "richer telling separates readers" thesis, located
   at definition/explanation rather than the (contaminated) full_rubric. NOTE: 3B and 1B are both
   weak, so this gap is a compressed lower bound — the 70B−3B gap (evening) is the real dynamic range.
2. **The name is a strong index for the enculturated reader.** Bare name = .793 at 8B (span_r2
   .92 — almost fully an assembly of census species: the name RETRIEVES known content) vs .634 at
   3B. Pointer-vs-payload confirmed.
3. **Ostension FAILS at this budget — showing lost to telling everywhere.** Exemplars are the
   WORST rung for every reader (8B .643) and drag the dossier down with them (.658 < full_rubric
   .927); exemplar span_r2 collapses (.46) with LOW value = the exemplar-prompted judge wanders
   out of the verbal span into noise, not into tacit gold. 0/46 metrics at 8B (5/46 at 1B) have
   exemplars beating all verbal rungs. CAVEATS before quoting: k=2 only, ≤400-char excerpts of
   long stories (brutal truncation of the ostensive channel), one prompt format. The k-curve and
   longer excerpts are the v2 test; ALSO redesign dossier as def+expl+rubric WITHOUT exemplars
   (v1 dossier is contaminated by the failing exemplar block, which breaks rel_to_dossier
   normalization — use full_rubric as the rich reference for v1 numbers).
4. **1B is below the task floor, not degenerate** (all rungs ≈ .52–.57; readout NOT collapsed —
   0/644 rows with near-zero variance, mean P(YES) .61). CW judging needs ≥3B to be a reader at
   all; use 3B as the weak anchor, keep 1B as the floor exhibit.
5. **Form-fragility staircase is ~flat across 23× parameters** (target seat, raw→calibrated):
   3B 13.0→6.5%, 8B 13.0→6.2%, 70B (30/46 partial) 11.6→5.3%. Mild decline, far from vanishing ⇒
   paraphrase fragility is substantially a property of the rubric LANGUAGE, not reader weakness —
   the on-thesis branch. Calibration removes ~52% at every scale (the sign-uniform strictness
   component); the surviving ~5–6% item×form entanglement persists at 70B.

## Face-1 verdicts alongside (same morning)

- source llama8b_glm, now **64 metrics** (10-fill + 16-fill): 57 FORM-DOMINATED / 7 UNDERSAMPLED
  (single-form targets; new metrics α≈0.97–0.99, D≈600–654 species — hugely productive spaces).
- aligned_8b_orbit_v2 (orbit + gate): 39 FORM-DOMINATED / 5 UNDERSAMPLED / **2 CODIFIABLE** — the
  first CW CODIFIABLE verdicts. Orbit fixes the TARGET seat; the gate (instrument seat) still
  fires at the 10% bar, which the calibrated ~6% would clear ⇒ the §6 gate redesign (ε_form band,
  NEEDS SIGN-OFF) is now the single biggest verdict lever (~39 metrics).
- aligned_3b_orbit_v2 (NO forminv → gate silently absent): 33 UNDERSAMPLED / 8 CODIFIABLE —
  **empirical confirmation of the gate-vanishing trap; do NOT quote until the 3B forminv pass runs.**

Humor Face-1 sweep: 60 R3 clusters, running (19/60 at 09:54, ~10.5 min/metric, done ~17:30);
Day-0 cert auto-fires at ≥40 via waiter. 70B v2 rescore at 30/46.

Related: [[project_cw_day0_certificate_read]], [[decompression-rungs-are-types]],
notes/2026-07-01__form-effects-control-plan.md (FIRST RESULTS + staircase),
notes/2026-07-02__r3cw-data-catalog.md.

# v14 GEPA (both channels): search works, pre-registered transfer gate blocks acceptance

> **Update 01:00**: the behavioral-channel tune (MI-based fitness, unconstrained arm) finished
> with the SAME outcome — `no behavioral/MCQ decoder template passed held-out transfer` — after
> executing the full pipeline (2,960 inductions + 64,908 rule/probe cells in
> tuning_cells.sqlite). Both channels share `stratified_reference_states`, so the held-out split
> is non-canonical-only in both; the structural critique below applies to both. GEPA is
> mechanically healthy end-to-end; the gate decision (below) is the only blocker.

**2026-07-17 overnight.** Status of "run the GEPA decoder" for the v14 mcq channel, after fixing
three plumbing crashes (`n_examples` panel-size hard-code, `option_codebook` schema nesting,
missing sk2 model-path overrides) and rewiring the dev pool onto the **prior-recalibrated
codebooks** (`cr3-v14.1-recal/`, which had been computed on 07-15 but never consumed by any
design — see project memory).

## What ran
Full GEPA round on sk2 GPUs 6,7: seed + 7 GLM-5.2-proposed mutations, all validated and scored
on the 8-metric dev pool (recal menus, 6-item panels, 4-option menus). Evidence:
`outputs/fast/development/tuning_cells.sqlite`, kind `v14_mcq_reference_batch` (8 distinct
template SHAs × 192 cells).

## Result

| template | held-out mean fitness | canonical-state mean |
|---|---|---|
| best mutation | **−0.032** | −0.041 |
| ... 6 others ... | −0.041 … −0.089 | |
| seed | −0.128 | −0.155 |

- **The search improves the objective**: one GEPA round moved held-out fitness +0.10 (−0.128 →
  −0.032) — GLM-5.2 mutations are usable and the fitness signal is live.
- **No template passed the gate** `heldout_prompt_transfer_ok = (mean ≥ 0)` →
  `RuntimeError: no behavioral/MCQ decoder template passed held-out transfer` after round 1
  (admissible set empty ⇒ `transfer_failure`).
- Recalibration effect (old menus → recal menus, seed template): blind option probability
  0.286 → **0.215** (flat = .25), pooled fitness −0.12 → −0.07. Direction right, not sufficient.

## Why the gate is (probably) structurally unpassable as frozen
`stratified_reference_states` (v14_decoder_tuning.py:110) sends **canonical + low/mid/high into
`search_states`** and reserves the remaining relabelings for `heldout_prompt_states` — so the
held-out split contains **only non-canonical states**: panels labeled by *other candidates'*
realized labelings. The row fitness is target-option lift over strongest control; when the demos
show a labeling that is not the target's, a faithful reconstructor *should not* raise the target
probability — near-zero-to-negative lift is expected behavior there, minus shuffled-control noise
⇒ mean < 0 in expectation. Requiring mean ≥ 0 on exactly these states makes admissibility fight
faithfulness. (Positive mass does exist — 32% of held-out cells — via population correlation
between candidates and the target, so a gate pass is not literally impossible, but one full GEPA
round of headroom (+0.10) still left the best template at −0.03.)

## Decision needed (user sign-off — changes the tuning estimand)
Options, not enacted:
1. Gate on canonical-state fitness (or search-split fitness) instead of relabeling-control mean.
2. Gate on improvement-over-seed for held-out transfer (monotonicity, not sign).
3. Keep gate; accept "no admissible tuned template" as the pre-registered negative.
Note the iteration-2 precedent: GEPA raised fidelity but *dropped* A-AUC (predictive residue) —
sign-forcing gates exist for a reason; option 3 is defensible.

## Also relevant
- The mcq raise destroys the GEPA trace (raise precedes trace write) — numbers above were
  recovered from tuning_cells.sqlite. If gates stay, consider writing the trace before raising.
- The behavioral-channel tune (MI-based fitness, different gate) launched separately;
  see logs/gepa_tune_behavioral_recal.log on sk2.
- FAST lane MCQ state tables are still on OLD (uncalibrated) menus — separate open decision.

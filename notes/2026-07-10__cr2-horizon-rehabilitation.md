# CR-2: rehabilitated capture–recapture horizon estimate

Date: 2026-07-10

## Why

The 2026-07-10 M_ω audit retracted `OPT_Ω + ε` as an upper bound: the head is an achieved value, ε is an
uncalibrated process-horizon extrapolation, and the tail-XOR planted control showed ε undercounting
hidden synergy (true 0.096 bits, ε 0.031). The audit's constructive conclusion was "the fixed-target DPI
cap is the only certified all-prompt object."

**Project-lead decision (2026-07-10):** the paper's target claim is CONCEPT-LEVEL — *how much better could
this metric possibly be, over many instantiations* — not recovery of a rubric-as-stated. Self-consistency
(the DPI cap `T(M_ω)`) is accepted as ONE bound, but it is instantiation-specific and does not answer the
concept question. Capture–recapture is the right tool for the concept question and is to be rehabilitated
to trusted-instrument grade rather than abandoned. Zero-capture-probability species (criteria the mining
process cannot emit) are an accepted, stated limitation.

## What CR-2 is

`methods/metric_implementer/experiments/cr_horizon.py`. Two estimators, both `certified=False`:

- `cr2_certificate(sigs, M, tags, lam=...)` — POOL-HORIZON estimate for a fixed target: what the mining
  process would deliver at its horizon. Capped by `H(M)`; `T(M_ω)` still binds any fixed instantiation.
- `concept_horizon(t_values)` — INSTANTIATION-HORIZON estimate: EVT endpoint over `T` across many
  re-instantiations. The direct answer to the concept question; needs ≥ ~40 distinct instantiations.

Trust credential: **out-of-sample coverage on a planted-world battery with analytic truth**
`I(M;R) = H(M) − H_b(flip)`. Not a theorem — a calibrated instrument, reported as such.

## Design, and the failure each element fixes

All flaws below were caught by planted controls or a 4-skeptic adversarial-verification workflow
(statistics / adversarial-worlds / code / semantics), never on real data.

| element | fixes |
|---|---|
| probe-split freeze (head/species/strata on half A, values on half B, both directions, adverse end; frozen-seed random split, not index parity) | selection optimism (~0.15 bits, measured on CW#24); B-leakage into the partition; systematic probe-order bias |
| permutation-null significance gates on every single value and pair excess; passing values used RAW (no mean subtraction — GT's alternating series doesn't commute with shifts) | phantom bits: positively-clipped CMI noise summed over hundreds of statistics manufactured up to ~21 bits (pure-noise null test) |
| conditional (within-y-strata) pair permutation null | the null is "both related to M, no interaction", not "unrelated feature" |
| greedy pair-CHAIN with multiplicity-adjusted acceptance gate | double-counting of overlapping pairs; winner's-curse over ~240 chain re-tests |
| bulk stratum gated at `z=Φ⁻¹(1−0.5/n_rest)`, requires ≥2 significant draws before ×n_rest extrapolation | one lucky pair amplified by the pair count into fake bits |
| **unseen-pair growth REFUSED** (`pair_unseen ≡ 0`) | the quadratic Chao factor exploded on noise and extrapolates into the hidden-partner blind spot by construction |
| (κ, λ) two-parameter calibration fit on TRAIN split, coverage reported on disjoint TEST split; tightness + vacuity shipped | in-sample coverage optimism; a vacuous always-`H` estimate masquerading as a good bound; κ specifically corrects the achieved reading's finite-sample attenuation (CMI on a probe half under-reads `I(M;U)` — the v2 battery failed at every λ because λ alone cannot fill an attenuation gap when the flux is honestly zero) |
| calibration target = **analytic POOL truth** `I(M; noisy units)` (exact by enumeration), NOT the concept ceiling `I(M;R)` | the pool-horizon cannot exceed what noisy units carry (unit-noise attenuation ≈ 0.22 bits on linear3); calibrating against the unreachable concept ceiling forces either failure or vacuity — `I(M;R)` belongs to `concept_horizon` |
| capture continuum (all species draw from the same natural law; no boosting) | v1's boosted design put every seen-valuable species at high multiplicity, leaving the singleton mass all-junk — Good–Turing's entire signal (valuable species captured exactly once) had been designed out of the worlds |

**Declared, measured blind spots** (reported, never averaged into coverage): unreachable species
(accepted limitation), parity depth ≥ 3, pair synergy with a never-captured partner.

**Scope is a generator-side property, never an outcome.** The pool-horizon's declared scope is
"criteria reachable within ~10× the observed stream." In the battery this is checked from the world's
generating parameters — every true species' capture probability ≥ 1/(10·n_draws) — computable at
generation time, before any estimation; worlds violating it are auto-classified to the `beyond-horizon`
blind stratum regardless of the stratum label they were drawn under. (The v3 battery's only in-scope
failures were geom-capture worlds that had hidden *most or all* true components at capture probabilities
of ~3×10⁻⁵ — unreachability wearing a truncation costume. Reclassifying by ground-truth reachability is
honest; reclassifying by estimator failure would not be.) On real data the same scope is carried as
language: the estimate covers pool value reachable at ~10× more mining; beyond that is the accepted
limitation.

## Results

- **Pure-noise null:** raw_gap collapses from ~21 bits (v2) to ~0.09; estimate well below the `H` cap.
- **Calibration (battery v5, 135 worlds incl. 30 skewed-base-rate/low-H strata; per-H-regime fits,
  h_split=0.85; credential = out-of-sample TEST coverage):**
  - low-H bin (the real-data regime): (κ,λ)=(1.0,1.0) — coverage .86, tightness +0.09, **vacuity 0.00**
  - high-H bin: (κ,λ)=(1.2,1.0) — coverage .82, tightness +0.25, vacuity .18
  - v4's single-regime test coverage of 1.00 was real but regime-narrow; v5's ~.82–.86 with near-zero
    low-H vacuity is the honest transferable credential.
- **Blind-spot strata:** parity3 and beyond-horizon fail as declared (~.25–.33 in v4) — proof coverage
  is earned, not vacuous.
- **Real-data pilot v2 (2026-07-11, 4 checkpoints, per-bin frozen calibration):** only audit flag = the
  *known-optimistic* old OPT on CW#24 (0.585 > horizon 0.441); its de-biased frozen re-measurement
  (0.444/0.431, the audit's own experiment on this metric) sits at the horizon. All genuine prompts
  inside. **Estimates still reach the H cap — now for a real reason** (low-H bin has κ=1.0, no
  inflation): the actual mining streams are massively UNSATURATED — 128–395 species / 600 draws,
  singleton-heavy, Chao1 unseen ≈ 231–2,100, gated singleton flux G1 = 0.11–2.68 bits. The pools are
  barely scratched; the pool-horizon honestly cannot exclude full-H articulability. Consequences:
  (i) achieved-vs-ceiling gaps on these metrics are NOT evidence of inarticulability (pool unexhausted —
  consistent with the plateau-is-upstream diagnosis); (ii) the informative readout is the component
  decomposition: linear flux dominates, pair-synergy chain ≈ 0 (little configural value among SEEN
  criteria on these four metrics).

Artifacts: `notebooks/data/cr2_calibration_v5.json` (v4 kept), `notebooks/data/cr2_realdata_pilot_v1.json`.
Notebook: `2026-07-03__prompt-optimality-gepa-vs-ceiling.ipynb` §8. Tests:
`tests/test_cr_horizon.py` (pure-noise null is the load-bearing test).

## Standing story — REVISED 2026-07-11 (external review accepted; see theory note §12.6b + cr_audit.py)

The earlier chain `R_frozen ≤ true pool optimum ≤ CR-2 horizon ≤ H(M)` is RETIRED as typography: a
calibrated estimator with out-of-sample test coverage 30/36 = .83 (exact 95% CI ≈ .67–.94) cannot sit
inside an inequality chain. CR-2's honest conclusion form is:

> *Under the declared species rule, this mining process shows no saturation — the experiment cannot
> rule out substantial additional value. It does NOT establish that full-H recovery is attainable,
> nor that its horizon number upper-bounds the pool optimum.*

CR-2 is retained as a **descriptive saturation diagnostic** (G1 one-doubling flux + component
decomposition). Certified statements now come from **CR-3** (`experiments/cr_audit.py`): a
discovery/audit split of the capture stream (freezing the species map + head BEFORE the audit draws),
exact per-family Clopper–Pearson bounds on missing behavioral mass (both directions: U0 upper, L0
lower — on today's streams the LOWER bound is the informative one: pilot M0 ∈ [.13, .77] across the
4 metrics, i.e. unsaturation is now certified), horizon/support conversions under a DECLARED p_min
floor with a sensitivity curve, an empirical-Bernstein UCB on missing value flux with the mark cap
predeclared at H(M), and the assumption-free improvement bound `H(M)·min(1, m·U0)`. No
recovery-optimum conversion is claimed without a certified submodularity ratio; the assumption-free
fallback is the DPI residual.

Phrasing correction (review flagged an apparent conflict): **2-unit XOR is in scope** via the pair
chain; **parity depth ≥ 3 remains the declared blind spot**. There was never a depth-3 claim; earlier
wording ("the tail-XOR case is now in scope") should be read as the pairwise case only.

Calibration honesty: v5 test coverage is 30/36 overall (.86 low-H, .82 high-H); the battery is easier
than the real regime (≈35 latent species / 120 draws vs 128–395 species / 600 draws) and its test
split informed development across v3→v5, so it no longer functions as a lockbox. Any future
calibrated estimator needs a fresh, frozen lockbox battery (realistic regimes, rare high-value
species, proposer bias, drift, over-merging, degree-3/4 synergy).

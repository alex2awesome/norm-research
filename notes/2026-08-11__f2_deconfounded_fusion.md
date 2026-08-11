# F2 — DECONFOUNDED FUSED LEDGER (all terminal cells)

Frozen spec: `notes/2026-08-09__full_sweep_queue.md` §F2, run under the §13 certified
stack (`notes/2026-08-05__taste-decomposition-design.md` §13.0–§13.4). This note is the
run record; registry / strict-list logging is the coordinator's.

## Design (as frozen)

Per cell, on the **same E rows and the same frozen Layer-1 stack as the master ledger**
(`direction1_mirror.fit_arm`: GroupKFold(5) on the cell's canonical grouping unit,
HistGB seeds {0,1,2} mean, nested grid per fold, cell's own bank family):

| arm | matrix |
|---|---|
| (a) `VA_enr` | bank_enriched (terminal bank incl. promoted Track-A criteria) |
| (b) `NUIS` | nuisance only (Track-B spurious channels + declared STRUCT) |
| (c) `VA_enr+NUIS` | bank_enriched ⊕ nuisance |
| (d) `VAT_dec_trained` | bank_enriched ⊕ nuisance ⊕ **T** |
| (e) `VAT_dec_untrained` | bank_enriched ⊕ nuisance ⊕ **T₀** |

**PRIMARY** = the stacked increment **(d)−(c)**, group paired bootstrap, 2,000 draws.
**SECONDARY** = (e)−(c).

`fit_arm` is called four times on **shared folds** (`L1.outer_folds` is a pure function
of `(n, groups)`), so arm (c) appears in two calls and is **asserted bit-identical** —
that is what makes the paired bootstraps legitimate.

`NEVER quote (d)−(c) against the old Δ_beyond without naming both designs`: this arm
conditions on the enriched bank *plus* named nuisance and refits on E; the old one did
neither.

---

## E-VALUE ANALOG — FROZEN DEFINITION (written before any cell's value was computed)

Causal-inference idiom: VanderWeele & Ding E-values / Cinelli & Hazlett robustness
values, adapted to an AUC-stacking frame. The statement being formalised:

> for the PRIMARY stacked increment (d)−(c) to be fully absorbed, a single unfound
> nuisance channel would need alone-AUC ≥ **X**; the strongest channel actually found by
> the sealed fleets is **Y**; the strict Track-B missing mass bounds the unfound space at
> **M̂**, whose odds-form remaining-AUC bound is **Z**.

### X — the required strength of one unfound channel

For a candidate channel `U`, let `NUIS⁺ = NUIS ⊕ U` and

    Δ(U) = AUC( stack[bank ⊕ NUIS⁺ ⊕ T] ) − AUC( stack[bank ⊕ NUIS⁺] )

    X = inf { alone-AUC(U) : Δ(U) ≤ 0 }   over the adversarial family below.

**Adversarial family (frozen).** Absorbing power depends on a channel's ORIENTATION, not
only its strength. The worst case — hence the conservative choice — is a channel
maximally aligned with whatever T contributes beyond (c). `U` is therefore built as a
degraded copy of the T column:

    u = rank(T)/n ∈ (0,1);  z ~ Uniform(0,1), frozen seed 2026
    U(w) = (1 − w)·u + w·z,  w solved by bisection so |AUC(y, U(w)) − s| ≤ .002

so that `alone-AUC(U) = s` by construction, measured with the same `roc_auc_score` used
everywhere else. **X is therefore a LOWER BOUND** on what a real unfound channel would
need: a channel of equal strength but arbitrary orientation absorbs strictly less.
Reporting the lower bound is the skeptic-favouring choice.

**Sweep design (frozen before running).**
- `s` grid `{.55, .60, .65, .70, .75, .80}`, truncated at `AUC(T)` on that cell. A
  channel is never asked to be stronger than T itself; if Δ has not crossed zero by
  `s = AUC(T)`, X is reported as `> AUC(T)` = **not absorbable by any single channel
  weaker than T**.
- Locate the bracketing interval where Δ crosses zero, then **2 bisection refinements**.
  Budget ≤ 8 stack fits per cell. Resolution ±.0125.
- **Cost control, frozen:** the sweep uses `gbm_seeds=(0,)` (one seed), not {0,1,2} — the
  crossing is a threshold, not a quoted AUC. The seed-0 unswept Δ is reported next to
  the 3-seed PRIMARY so the seed offset is visible.
- **Monotonicity is assumed in `s` and CHECKED**: grid values must be non-increasing in
  `s` within tolerance .01; a violation is recorded and the cell flagged `NON_MONOTONE`.

### Y — the strongest channel actually found

`Y = max` over the cell's found nuisance channels of alone-AUC on E (already emitted as
`top_nuisance_channels[0]`).

### Robustness ratio

AUC is bounded with origin .5, so the ratio is taken on **excess over chance**:

    RR = (X − .5) / (Y − .5)

`RR > 1` ⇒ the unfound channel would have to be *more informative than anything the
sealed fleets actually found*. The raw `X/Y` is reported too, but `RR` is the quantity
with the interpretation.

### Z — the missing-mass–coupled bound

Let `M̂` be the cell's terminal-round **strict Track-B** Good–Turing missing mass
(`closure/<cell>/round{terminal}_missing_mass.json → tracks.b.M_hat`, or that campaign's
equivalent field — the field actually read is recorded per cell), and `S_obs` the number
of observed B species. The effective number of unfound B species is

    Ŝ_unf = S_obs · M̂ / (1 − M̂)

Assume unfound channels are **exchangeable in strength** with found ones. (Discovery is
strength-biased in reality — strong channels get proposed first — so this assumption is
*generous to the skeptic*.) Model found-channel strength in **odds form**,
`o_i = AUC_i / (1 − AUC_i)` with `AUC_i` folded to ≥ .5, fit an exponential upper tail to
`log o_i` above the median, and set

    Z = expected maximum of Ŝ_unf further draws
      = the (1 − 1/Ŝ_unf) quantile of the fitted law, converted back to AUC

`Z` is the strongest single unfound channel the B-side missing mass can still be hiding.

### Verdict rule (frozen)

| condition | verdict |
|---|---|
| Δ ≤ 0 | **n/a** — nothing to absorb; the E-value analog is undefined |
| X > Z | **ROBUST** — no single channel the missing mass can still hide is strong enough |
| X ≤ Z | **ABSORBABLE-IN-PRINCIPLE** — flagged; the increment is quoted with that caveat |

### Limitations, carried verbatim wherever X/Z are quoted

- **X is a lower bound** (worst-case orientation). Real channels of the same alone-AUC
  absorb less, so the true requirement is higher than X.
- **Z inherits every assumption of Good–Turing on a sealed-fleet species count**, plus
  the exchangeable-strength assumption. It is an order-of-magnitude guide, not a
  confidence bound.
- **Feng et al. 2019 still applies**: this bounds a SINGLE unfound channel. It does not
  bound interactions among unfound channels, nor a coordinated set of several, nor
  channels outside the proposable space entirely.
- Cells where only **τ-era** B-mass is available (no strict-merge species count) are
  flagged `TAU_ERA_MASS`; their Z is provisional pending the strict certificate backfill
  (`certA_strict.json`).

### Schema

The block is added to each `results/f2_deconf_<cell>.json` as a **schema-versioned
addendum** — `evalue_analog` with `schema: "evalue_analog/v1"` — and **no existing arm is
recomputed**.

---

## Results

<!-- RESULTS -->

## Artifacts

| what | where |
|---|---|
| engine | `methods/taste_decomposition/fusion/f2_deconf.py` |
| cell adapters | `methods/taste_decomposition/fusion/f2_cells.py` |
| E-value addendum | `methods/taste_decomposition/fusion/f2_evalue.py` |
| per-cell results | `methods/taste_decomposition/results/f2_deconf_<cell>.json` |
| T / T₀ columns, E rows | `methods/taste_decomposition/fusion/t0_{rows,scores}/` |

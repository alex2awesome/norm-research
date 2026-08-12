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

### Governing increment per cell

The **companion** governs wherever the enriched-bank gap exceeds the .02 trigger; otherwise the E-refit primary does. All are INCREMENTAL information, never LEVEL residual.

| cell | governing increment | which footing | P | verdict |
|---|---|---|---:|---|
| `peer_verdict` | +.0588 [+.0339,+.0832] 1.00 | E-refit (companion n/a — E is the whole population) | 1.00 | **significant positive** |
| `peer_curation` | +.0008 [−.0211,+.0241] 0.54 | matched-strength companion | 0.54 | null / not significant |
| `peer_revealed` | +.0969 [+.0610,+.1339] 1.00 | matched-strength companion | 1.00 | **significant positive** |
| `nc_responded` | +.0200 [+.0025,+.0377] 0.99 | E-refit primary (gap below .02 trigger) | 0.99 | **significant positive** |
| `cw_community` | +.0988 [+.0880,+.1098] 1.00 | E-refit (companion n/a — E is the whole population) | 1.00 | **significant positive** |
| `hashtagwars_verdict` | −.0230 [−.0630,+.0289] 0.21 | matched-strength companion | 0.21 | null / not significant |
| `cap_finalist` | −.0224 [−.0454,−.0005] 0.02 | matched-strength companion | 0.02 | **SIGNIFICANTLY NEGATIVE** |
| `jokes_community` | +.0115 [+.0070,+.0156] 1.00 | matched-strength companion | 1.00 | **significant positive** |
| `mathse_accepted_verdict` | +.0073 [−.0024,+.0169] 0.93 | matched-strength companion | 0.93 | null / not significant |
| `mathse_vote_score` | +.0085 [−.0040,+.0213] 0.92 | E-refit primary (gap below .02 trigger) | 0.92 | null / not significant |
| `press_verdict` | +.0622 [+.0326,+.1052] 1.00 | matched-strength companion | 1.00 | **significant positive** |

### Arms and increments

| field | cell | n_E | (a) bank_enr | (b) NUIS | (c) enr+NUIS | (d) +T | (e) +T₀ | PRIMARY (d)−(c) [CI] P | WY band | SECONDARY (e)−(c) P | §11 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Peer review | `peer_verdict` | 1244 | .6684 | .6900 | .6828 | .7413 | .6941 | +.0588 [+.0339,+.0832] 1.00 | [+.0556,+.0695] | +.0175 1.00 | PASS |
| Peer review | `peer_curation` | 1571 | .5291 | .5545 | .5408 | .5481 | .5438 | −.0006 [−.0304,+.0278] 0.48 | [−.0006,−.0006] | +.0051 0.70 | PASS |
| Peer review | `peer_revealed` | 478 | .7202 | .7511 | .7837 | .8736 | .7843 | +.0927 [+.0565,+.1288] 1.00 | [+.0927,+.1306] | −.0006 0.47 | PASS |
| Regulatory (N&C) | `nc_responded` | 1904 | .7882 | .7195 | .8042 | .8325 | .8006 | +.0200 [+.0025,+.0377] 0.99 | [+.0200,+.0434] | −.0058 0.12 | PASS |
| Creative writing | `cw_community` | 7008 | .6652 | .6651 | .6931 | .7896 | .6919 | +.0988 [+.0880,+.1098] 1.00 | [+.0988,+.1132] | +.0005 0.65 | PASS |
| Humor | `hashtagwars_verdict` | 924 | .5357 | .6267 | .5863 | .6127 | .5879 | +.0297 [−.0067,+.0634] 0.94 | [+.0162,+.0326] | −.0039 0.32 | PASS |
| Humor | `cap_finalist` | 1055 | .6014 | .6298 | .6131 | .6165 | .6178 | +.0002 [−.0249,+.0241] 0.52 | [−.0091,+.0185] | +.0140 0.86 | PASS |
| Humor | `jokes_community` | 3163 | .7214 | .7282 | .7462 | .7596 | .7448 | +.0151 [+.0086,+.0206] 1.00 | [+.0151,+.0241] | −.0002 0.43 | PASS |
| Math | `mathse_accepted_verdict` | 2600 | .5801 | .6853 | .6918 | .7007 | .6931 | +.0086 [−.0015,+.0192] 0.95 | [+.0086,+.0241] | +.0029 0.84 | PASS |
| Math | `mathse_vote_score` | 2326 | .6238 | .6691 | .6814 | .6881 | .6775 | +.0085 [−.0040,+.0213] 0.92 | [+.0085,+.0162] | −.0017 0.28 | PASS |
| Journalism/press | `press_verdict` | 605 | .6860 | .5809 | .6686 | .7517 | .6806 | +.0899 [+.0638,+.1470] 1.00 | [+.0850,+.0936] | +.0072 0.85 | PASS |

### Matched-strength companion (D1b-style two-stage)

| cell | bank E-refit (a) | bank full-strength on E | gap | >.02 | (c*) | (d*) | COMPANION (d*)−(c*) [CI] P | E-refit primary (contrast) | matched X | matched RR | matched verdict |
|---|---:|---:|---:|:-:|---:|---:|---|---:|---:|---:|---|
| `peer_verdict` | .6684 | — | — | — | — | — | _n/a — E is the whole population; companion identical to primary_ | +.0588 | — | — | — |
| `peer_curation` | .5291 | .5974 | +.0684 | **Y** | .5688 | .5758 | +.0008 [−.0211,+.0241] 0.54 | −.0006 | .5447 | 0.73 | ABSORBABLE-IN-PRINCIPLE |
| `peer_revealed` | .7202 | .7724 | +.0523 | **Y** | .7800 | .8691 | +.0969 [+.0610,+.1339] 1.00 | +.0927 | > AUC(T) = 0.8842 | 1.78 | ROBUST |
| `nc_responded` | .7882 | .7995 | +.0114 | n | .8059 | .8330 | +.0274 [+.0108,+.0435] 1.00 | +.0200 | > AUC(T) = 0.8167 | 3.41 | ROBUST |
| `cw_community` | .6652 | — | — | — | — | — | _n/a — E is the whole population; companion identical to primary_ | +.0988 | — | — | — |
| `hashtagwars_verdict` | .5357 | .6751 | +.1393 | **Y** | .6279 | .6240 | −.0230 [−.0630,+.0289] 0.21 | +.0297 | — | — | n/a |
| `cap_finalist` | .6014 | .6813 | +.0799 | **Y** | .6452 | .6217 | −.0224 [−.0454,−.0005] 0.02 | +.0002 | — | — | n/a |
| `jokes_community` | .7214 | .7542 | +.0328 | **Y** | .7550 | .7691 | +.0115 [+.0070,+.0156] 1.00 | +.0151 | > AUC(T) = 0.7469 | 1.55 | ROBUST |
| `mathse_accepted_verdict` | .5801 | .6206 | +.0405 | **Y** | .6969 | .7017 | +.0073 [−.0024,+.0169] 0.93 | +.0086 | > AUC(T) = 0.6439 | 1.00 | ABSORBABLE-IN-PRINCIPLE |
| `mathse_vote_score` | .6238 | .6415 | +.0177 | n | .6698 | .6859 | +.0178 [+.0060,+.0296] 1.00 | +.0085 | .6456 | 1.41 | ROBUST |
| `press_verdict` | .6860 | .7546 | +.0686 | **Y** | .6900 | .7538 | +.0622 [+.0326,+.1052] 1.00 | +.0899 | > AUC(T) = 0.7744 | 2.66 | ROBUST |

Where `>.02` is **Y**, the E-refit primary is a matched-footing readout and is **not** comparable to any full-strength bank comparison (including a closure campaign's same-rows verdict); the COMPANION is the quotable number there.

### E-value analog

| cell | Δ (d)−(c) | X | Y | RR=(X−.5)/(Y−.5) | X/Y | M̂ (strict?) | Z | verdict |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `peer_verdict` | +.0588 | > AUC(T) = 0.7769 | .5972 | 2.85 | 1.30 | — | — | Z_UNAVAILABLE |
| `peer_curation` | −.0006 | — | .5614 | — | — | 0.67 (τ-era) | — | n/a |
| `peer_revealed` | +.0927 | > AUC(T) = 0.8842 | .7162 | 1.78 | 1.23 | 0.43 (τ-era) | .7388 | ROBUST |
| `nc_responded` | +.0200 | > AUC(T) = 0.8167 | .5929 | 3.41 | 1.38 | 0.72 (τ-era) | .6050 | ROBUST |
| `cw_community` | +.0988 | > AUC(T) = 0.7921 | .5816 | 3.58 | 1.36 | 0.15 (τ-era) | .5744 | ROBUST |
| `hashtagwars_verdict` | +.0297 | > AUC(T) = 0.7315 | .6496 | 1.55 | 1.13 | 0.65 (τ-era) | .7800 | ABSORBABLE-IN-PRINCIPLE |
| `cap_finalist` | +.0002 | .5935 | .5987 | 0.95 | 0.99 | 0.38 (strict) | .6213 | ABSORBABLE-IN-PRINCIPLE |
| `jokes_community` | +.0151 | > AUC(T) = 0.7469 | .6595 | 1.55 | 1.13 | 0.30 (strict) | .6676 | ROBUST |
| `mathse_accepted_verdict` | +.0086 | .5929 | .6442 | 0.64 | 0.92 | 0.26 (τ-era) | .6704 | ABSORBABLE-IN-PRINCIPLE |
| `mathse_vote_score` | +.0085 | .5926 | .6029 | 0.90 | 0.98 | 0.35 (τ-era) | .6318 | ABSORBABLE-IN-PRINCIPLE |
| `press_verdict` | +.0899 | > AUC(T) = 0.7744 | .6031 | 2.66 | 1.28 | 0.45 (τ-era) | .6138 | ROBUST |

`X` reported as `> AUC(T)` means the sweep never crossed zero even at a channel as strong as T itself: **not absorbable by any single channel weaker than T**.

**§13 flags:** `cw_community` TAU_ERA_MASS -- no strict-merge marker found in this cell's ; `cw_community` spurious-alone 0.6651 > .65; `hashtagwars_verdict` TAU_ERA_MASS -- no strict-merge marker found in this cell's ; `jokes_community` spurious-alone 0.7282 > .65; `mathse_accepted_verdict` TAU_ERA_MASS -- no strict-merge marker found in this cell's ; `mathse_accepted_verdict` spurious-alone 0.6853 > .65; `mathse_vote_score` TAU_ERA_MASS -- no strict-merge marker found in this cell's ; `mathse_vote_score` spurious-alone 0.6691 > .65; `nc_responded` TAU_ERA_MASS -- no strict-merge marker found in this cell's ; `nc_responded` spurious-alone 0.7195 > .65; `peer_revealed` NON_MONOTONE sweep; `peer_revealed` TAU_ERA_MASS -- no strict-merge marker found in this cell's ; `peer_revealed` spurious-alone 0.7511 > .65; `peer_verdict` spurious-alone 0.6900 > .65; `press_verdict` TAU_ERA_MASS -- no strict-merge marker found in this cell's 

## Resumption hooks (battery closed 2026-08-11; reopen only for these)

The battery is complete for all 11 terminal cells. Three things reopen it, and each
has a one-command path — nothing needs re-deriving.

**1. `peer_verdict` Z backfill.** The cell is marked `Z_UNAVAILABLE`: its pilot campaign
predates the species/Good-Turing instrument, so no Track-B mass exists on disk (absent,
not τ-era). A retroactive sealed Track-B fleet round is ordered (proposals + strict
two-judge species only, no bank changes). When its species file lands in
`closure/` with `tracks.B.good_turing` and `b_merge.strict`, run:

    python3 methods/taste_decomposition/fusion/f2_evalue.py --cell peer_verdict

`f2_evalue.find_mass` resolves species files automatically and prefers the strict
figure; `f2_rulings.py` is idempotent and re-stamps the rulings blocks. X and RR
(> AUC(T) = .7769, RR 2.85) are already computed and will not change — only Z and the
verdict fill in. NOTE the resolver looks in `MASS_DIR["peer_verdict"] = "."` (the
closure ROOT, where that campaign's round files live), not a per-cell subdir.

**2. The strict-mass backfill for the other nine τ-era cells.** Same command per cell,
same idempotence. Expect verdict flips toward ROBUST: at equal robustness ratio the two
strict-mass cells already certify differently from the τ-era ones (`jokes` RR 1.55,
strict M̂ .30, Z .667 → ROBUST vs `hashtagwars` RR 1.55, τ-era M̂ .65, Z .780 →
ABSORBABLE). Only Z and `verdict` move; X, Y and RR are mass-independent.

**3. New terminal cells** (SO votes, AoPS, cap_crowd, homepage). Per cell:

    # a) adapter: add to GENERIC in f2_cells.py (dir, tag, round pattern, rounds,
    #    loader_arg, struct_npz if the cell has declared observed ordinals)
    python3 methods/taste_decomposition/fusion/f2_cells.py --cell <cell>   # verify n_E
    # b) requires t0_rows/<cell>.npz + t0_scores/<cell>.jsonl.gz to exist first
    #    (t0_build_rows.py then t0_score_vllm.py — the T0 arm is a STANDING COLUMN)
    python3 methods/taste_decomposition/fusion/f2_deconf.py  --box <box> --cell <cell>
    python3 methods/taste_decomposition/fusion/f2_evalue.py               --cell <cell>
    python3 methods/taste_decomposition/fusion/f2_matched.py              --cell <cell>
    python3 methods/taste_decomposition/fusion/f2_evalue.py  --matched    --cell <cell>
    python3 methods/taste_decomposition/fusion/f2_rulings.py                          # all cells
    python3 methods/taste_decomposition/fusion/f2_summarise.py --write

### Binding operating rules this battery established

- **ONE CELL PER PROCESS.** Every closure dir ships modules with identical names
  (`cells.py`, `oof_alignment_gate.py`, `closure_core.py`); two adapters in one process
  cross-contaminate `sys.modules`. It failed loudly here only because `mathse_vote` has
  an alignment gate — a cell without one would mis-join silently.
- **Never join an adapter to E through an id dict.** `peer_verdict` repeats 5 ntitles
  among its 1,244 E rows. The identity fast path plus the `y_equal_elementwise`
  assertion is what caught this in all three scripts; the assertion stays mandatory.
- **A declared STRUCT block that is missing must FAIL, not warn.** A warn-only guard
  produced a fully quotable-looking `mathse_accepted` increment against the wrong
  conditioning block. Diagnostic that caught it: NUIS alone read .5615 where the
  campaign's own position model reads .6600 on the same rows.
- **Which box.** Run each cell on the box that reproduced its master-ledger row in the
  T₀ arm (mirror cells → mac, scale-up-wave-C / mirror-2 → sk3 `envs/ai_usage`).
  GroupKFold fold membership is sklearn-version AND architecture dependent.

## Artifacts

| what | where |
|---|---|
| engine | `methods/taste_decomposition/fusion/f2_deconf.py` |
| cell adapters | `methods/taste_decomposition/fusion/f2_cells.py` |
| E-value addendum | `methods/taste_decomposition/fusion/f2_evalue.py` |
| per-cell results | `methods/taste_decomposition/results/f2_deconf_<cell>.json` |
| T / T₀ columns, E rows | `methods/taste_decomposition/fusion/t0_{rows,scores}/` |

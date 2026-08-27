# code_competitions (AtCoder "same approach as editorial") — UNPARK

Date: 2026-08-10. Trigger: user — *"retrieve the scores we have … go forward with new
features. Don't redo work."* Cell was parked with a lost matrix; this note records what
survived, what was built, and what is still running.
Code + artifacts: `methods/taste_decomposition/code_competitions/`.

## 0. Terminology

**AtCoder** = a competitive-programming contest site. **Editorial** = the official
solution write-up for a problem. **The label y** = 1 if a submitted solution uses the
*same approach* as the editorial, 0 if it solves the problem a different way. **A** = the
articulated-criterion bank (139 LLM-judged code-review criteria). **V** = deterministic
code/regex features — no language model anywhere, which is what makes them V.
**T** = a dense model reading the raw code. **OOF** = out-of-fold.
**canonical_pid** = the problem identifier, and the grouping key for every split.

## 1. What survived

`methods/taste_decomposition/results/code_competitions_layer1.json` survives, and both of
its source tables are on disk:

| artifact | path (sk3) | shape |
|---|---|---|
| A bank scores | `outputs/v2_analysis/comp_fourplatform_cells/ac_bank_scores.parquet` | 1,000 × 283 (139 aspects × score+applied) |
| labels, groups, **candidate_code** | `outputs/v2_analysis/dense_ceiling/cell_ac_l1.parquet` | 2,495 × 6 |

**Population reproduced exactly and asserted in code**: the inner join on `pair_id` gives
**n = 999, 634 canonical_pid groups, 850 positive / 149 negative (pos rate .8509)**.
Languages cpp 565 / python 434; median code length 652 characters.

> **The number to carry everywhere is 149 — the absolute minority count.** Every readout
> below is bounded by it (mathlib lesson: never quote a rate without the count).

Two flags recorded in the surviving file are binding here: **no V layer had ever been
built** for this cell (the candidate-only exec-pass-rate layer was ~chance and unused),
and **its T = .69 was population-mismatched**, computed on the full 2,495-row L1 set.

## 2. Protocol correction — stratification is not optional at 149 minority rows

The A bank was **kept exactly as scored** and never recomputed. But the fold scheme had to
be settled first, and it turned out to matter more than anything else in this note:

| fold scheme | negatives per test fold | A_lin |
|---|---|---:|
| plain `GroupKFold(5)` | **26 – 37** | .6683 |
| `StratifiedGroupKFold(5, shuffle=True)`, fold seeds 0–4 | **29 – 30** | .6834 – .6919 |
| *recorded ledger* | — | *.6907* |

So the unstratified reading was a **~.02 artifact of fold composition**, not a number, and
the stratified protocol reproduces the recorded A_lin within its fold-seed spread. Every
block below is read under `StratifiedGroupKFold(5, shuffle=True)` by `canonical_pid`,
linear averaged over fold seeds 0–4, HistGB on the frozen Layer-1 grid over seeds 0–2, and
the fold-seed spread is reported rather than hidden.

## 3. The new V layer, and the readouts

27 deterministic code/regex features on `candidate_code`, language-aware: size (chars,
lines, non-blank, mean/max line length, blank ratio), comments, max nesting, control-flow
counts (for/while/if/else/return), function count, library-idiom hits, include/import and
`#define` counts, identifier count/uniqueness/reuse, digit density, int64 use, fast-IO
idioms, tabs, operator count, and a language indicator.

| block | features kept | **linear** (spread over 5 fold seeds) | **nonlinear** (spread over 3 seeds) |
|---|---:|---:|---:|
| **V** (new) | 27/27 | **.7258** (.0203) | **.7289** (.0059) |
| A (bank, kept as scored) | 52/139 | .6875 (.0086) | .7130 (.0142) |
| **V+A** | 79/166 | .7157 (.0183) | **.7383** (.0306) |

**The newly built V layer beats the 139-criterion LLM bank**, on both aggregations
(+.038 linear, +.016 nonlinear), and V+A adds only **+.009** over V alone. On this cell the
deterministic surface layer is the stronger instrument and the articulated bank is largely
redundant to it.

### What V actually is: an anti-predictive code-size channel

| feature | alone AUC |
|---|---:|
| `v_n_chars` | **.262** |
| `v_n_ident` | .263 |
| `v_n_uniq_ident` | .266 |
| `v_n_lines` | .270 |
| `v_n_nonblank` | .276 |
| `v_n_operators` | .281 |
| `v_n_if` | .282 |

Every strong V feature is a **size** feature and every one is **anti**-predictive: shorter,
simpler code is more likely to be the editorial's approach (`v_n_chars` reads .262, i.e.
.738 reversed). That is substantively sensible — the editorial approach is the intended
compact solution, and long code is the ad-hoc alternative — and it is a genuine V signal
rather than a leak. Language is **inert on its own** (.497) but removing it costs V_lin
.018, so it conditions the size features rather than predicting directly.

### Reproduction against the recorded ledger

| | recorded | live (stratified) | verdict |
|---|---:|---:|---|
| A_lin | .6907 | **.6875** | reproduces, inside fold-seed spread |
| A_nl (mean of 3) | .6696 | **.7130** | **does not reproduce** |

A_lin reproduces. **A_nl does not**, under either fold scheme (.6929 unstratified / .7130
stratified, both *above* the recorded .6696). The surviving file already carried
`GATE_FAILED_PROCEED_WITH_FLAG` for its own reproduction attempt, so the recorded
nonlinear number should be treated as **superseded, not as a target**.

## 4. Honest same-population dense T — running

The retired T = .69 was computed on a different, larger population and is not used. The
replacement reuses the very script that produced it —
`scripts/dense_ceiling/run_dense_ceiling.py` (ModernBERT-base, StratifiedGroupKFold(5,
shuffle=True) by `canonical_pid`, 3 seeds, bf16, 3 epochs, lr 2e-5) — importing its
`CodeDataset` and `mb_predict` rather than rewriting them, and changing exactly two things:

1. **class weighting** — the shared runner uses plain `BCEWithLogitsLoss`, which at pos
   rate .851 is dominated by the majority class; `pos_weight = n_neg / n_pos` is now
   computed on the **train fold only** (the `--class_weight_auto` convention);
2. **pooled OOF predictions saved** — the shared runner averages per-fold AUCs, but A/V/VA
   here are *pooled* grouped-OOF AUCs, so a same-rows comparison needs the OOF vector.

**Minority discipline**: absolute negative counts are asserted and logged per fold, and a
fold carrying fewer than 15 negatives aborts the run rather than emitting a number.

**Status: COMPLETE.** The chain fired exactly as designed while the laptop was cut off from
sk3 by the jump-host outage: the scorer released GPU 7 at 02:29:57Z, the chain re-checked
the card (0 MiB), re-claimed, ran, and released `rc=0` at 02:35:34Z. Fold test-negatives
came out **29–31** (floor was 15), so no fold was starved.

| | seed 0 | seed 1 | seed 2 | **mean** | spread | **seed ensemble** |
|---|---:|---:|---:|---:|---:|---:|
| **T, pooled grouped-OOF** | .7060 | .7080 | .6762 | **.6967** | .0318 | **.7241** |

## 5. THE SAME-ROWS LADDER — and the cell has NO positive taste residual

All four blocks on the identical 999 rows, identical fold seeds (so every vector is
fold-aligned; asserted by matching `pair_id` order between the dense OOF and the ladder).

| block | per-seed mean | seed ensemble |
|---|---:|---:|
| **V** (new deterministic layer) | **.7289** | **.7429** |
| A (bank, kept as scored) | .7130 | .7282 |
| **V+A** | **.7383** | **.7535** |
| **T (dense, same population)** | **.6967** | **.7241** |

| quantity | value |
|---|---:|
| Δ_total = T − V | **−.0322** |
| Δ_beyond = T − V+A (means) | **−.0416** |
| Δ_beyond (ensemble vs ensemble) | **−.0294** |

**The dense model lands below the articulated stack, and below the deterministic V layer
on its own.** The sign is not a seed artifact: every one of the three dense seeds
(.7060 / .7080 / .6762) sits below the V+A mean of .7383. So code_competitions joins the
dense-below-bank group (cap_crowd, Style Invitational, press) rather than the
positive-residual group.

Worth stating precisely, because it is the whole point of the rebuild: the **retired,
population-mismatched T was .69 and the honest same-population T is .6967 / .7241** — the
*level* barely moved. What changed is that the number is now legitimate and the comparison
is same-rows, and on that footing the cell's verdict flips from "unknown, flagged" to
"no residual".

### Design §11 — fused vs bank: a WASH, no audit trigger

| arm | value |
|---|---:|
| bank V+A (ensemble) | .7535 |
| dense (ensemble) | .7241 |
| **fused = grouped-OOF stack of [V+A, dense]** | **.7537** |
| fused − bank | **+.00019** |
| fused − dense | +.0296 |
| audit trigger (fused ≤ bank)? | **no** — but by .0002 |

Compared like-for-like (a stack built on ensembles, read against the ensemble bank),
**fusion adds .0002 over the bank — a wash.** It clears the §11 rule numerically and fails
it substantively, and it reproduces the registry's cross-cutting statement exactly:
*fusion reaches max(parents), nowhere reliably exceeds it* on dense-below-bank cells. This
is the opposite of the code_v3 PR-merge cell, where fusion beat both parents by +.058 over
the bank — and the contrast between the two code cells is the useful finding.

### Bounds on this verdict

**149 negatives.** The dense seed spread is .0318, comparable to the −.03/−.04 deltas
themselves, so the *magnitude* of the shortfall is soft even though its *sign* is
consistent across seeds. And the V layer's advantage rests on a code-size channel (§3):
if a future arm neutralises length, the V and V+A numbers should be expected to fall.

## 5. Artifacts

| artifact | path |
|---|---|
| V features (999 × 27) | `code_competitions/ac_v_features.parquet` |
| V / A / V+A readouts + per-feature alone-AUCs | `code_competitions/ac_v_and_readout.json` |
| V builder + readout | `code_competitions/build_v_and_readout.py` |
| 999-row population subset | `code_competitions/cell_ac_l1_999.parquet` |
| same-population dense T (running) | `code_competitions/ac_dense.py`, `ac_dense_T.json`, `ac_dense_oof.npz` |
| lane-C chain launcher | `code_competitions/chain_ac_dense.sh` |

# Layer-3 articulation closure — homepage curation (story-grouped, rubrics_v2)

**TERMINAL AT ROUND 0 — noise-floor, not saturation.**

Cell: journalism homepage curation (spatial placement), STORY-GROUPED design with the
rebuilt 29-criterion `rubrics_v2` bank. Layer-1 ledger:
`methods/taste_decomposition/results/homepage_curation_storygrouped_ledger.json`.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + ADDENDA.
Campaign dir: `methods/taste_decomposition/closure/homepage_curation/`.
Agent: `claude-v9-journalism-tweets` (journalism discovery lane, cell 2 of 3).

## Bottom line

The frozen gate stops a campaign at round 0 when the residual is ≤ .02. This cell's
closure-split residual is **Δ₀ = +.0035**, so it stops. But the gate is the *weaker* of
the two reasons to stop, and the campaign note's substance is the stronger one:

> **The residual is not resolvable at this cell's resolution.** T − VA_nl on MONITOR is
> **+.0093 with a 95% CI of [−.0429, +.0642]** — a band roughly **eleven times** the
> point estimate, spanning zero, P(>0) = .65. And the **ε-resolvability power check
> FAILS**: a comparison whose true change is *zero* has a paired SD of **.00882**,
> which is **1.8× ε = .005**. Rounds run on this cell could not have produced an
> interpretable sub-ε reading, whatever they found.

The honest verdict is therefore the press_verdict one, in the same words: **residual
closed at this cell's resolution; the stopping rule was NOT fired** (no rounds ran, so
nothing saturated). This cell should never be quoted as "articulation saturates on
homepage curation" — only as "no residual is measurable here".

**The ε-resolvability check earned its place.** It is a pre-round-0 gate precisely so
that a campaign is not spent producing uninterpretable numbers, and on this cell it
fired before any fleet, any judge call, or any GPU time was spent. Contrast BBC
most-read, run the same day with the same code, where it passed at .00252.

## Round 0

### Hard gates

**Dense join gate — PASS.** Preds carry no row_id, so the join is by order; proven
element-wise on the `(judgement, group)` sequence for all three seeds, with shuffled
counterfactuals **.5036 (eval) / .5085 (test)**.

**Splits — PASS.** Stable-hash `sha256("homepage-curation-closure-v1|" + snapshot_id)
< .20`, MONITOR taken strictly inside dense-held-out (AMENDMENT 1), snapshot-disjointness
asserted.

| | value |
|---|---|
| MONITOR | **622 rows / 72 snapshots** |
| FIT+MINE | 12,376 rows |
| mining slice M | 2,009 rows |
| pos rate MONITOR / FIT+MINE | .4711 / .5021 |

**MONITOR is the binding constraint at 622 rows** — a fifth of BBC most-read's 2,060 —
and that is what drives everything below.

**Coverage / article dedup, recorded.** The dense arm dropped 630 TRAIN rows (300
distinct normalised headlines) that recur in eval/test, because a story sits on the
homepage across successive captures and snapshot grouping alone leaks that way. The
dense arm therefore covers 12,368 of the 12,998 A/V-scored rows.

**sklearn drift, recorded.** The Layer-1 ledger was produced under **1.9.0**; this box
runs **1.8.0**, and GroupKFold assignments move across releases. This is why the curve
is anchored on this script's own round-0 fit rather than the Layer-1 number
(AMENDMENT 1), and it is asserted in the results JSON.

**Item view, recorded as a pre-existing asymmetry, not one introduced here.** V is
computed on the HEADLINE half only (`v_features.headline_of`), while the dense arm reads
the whole stored `text` (HEADLINE + CONTEXT block) and the A bank uses the rubrics_v2
item view. Unlike BBC most-read — where V, A and dense read a byte-identical string —
this cell does carry a genuine view asymmetry. Measured on the real tokenizer the
stored text runs to max 720 tokens (p99 703), so **nothing truncates** and the SO
audit's answer-displacement failure mode cannot occur here. The asymmetry is a property
of the cell's design, flagged for the record; it is not what produces the terminal,
which rests on width.

### The anchor

| quantity | MONITOR (n = 622) |
|---|---|
| VA_lin | .6813 |
| VA_nl per seed | .7277 / .7283 / .7305 |
| **VA_nl** (mean of per-seed AUCs) | **.7288** |
| T per dense seed | .7516 / .7240 / .7213 |
| **T** | **.7323** |
| **Δ₀** | **+.0035** |

Note the dense seed spread on MONITOR is **.0303** (.7213–.7516) — by itself larger
than Δ₀ by a factor of nine. That is the same fact the resolvability block states,
visible directly in the raw seed numbers.

### ε-RESOLVABILITY POWER CHECK — **FAIL**

Known-zero comparisons (two VA_nl fits differing only by GBM seed, paired difference
bootstrapped at the snapshot level):

| | boot SD | 95% width |
|---|---|---|
| worst pair | .0102 | .0388 |
| **mean paired SD** | **.00882** | ~.031–.039 |

**.00882 ≥ ε = .005 → NOT resolvable.** Cross-fitting is the freeze's remedy and would
shrink the *fit-seed* component, but it cannot shrink the part that binds here — the
MONITOR is 622 rows in 72 snapshots, and the residual CI below is dominated by that
width, not by fit noise. So the terminal stands under the remedy.

### Residual resolvability — the decisive statistic

| | value |
|---|---|
| T − VA_nl on MONITOR | **+.0093** |
| 95% CI (snapshot bootstrap, 4,000) | **[−.0429, +.0642]** |
| SD | .0276 |
| P(> 0) | .6475 |
| spans zero | **yes** |
| Layer-1 same-rows Δ_beyond (eval, n=1,313) | +.0068 |

### Swap baseline

C₊ **.8413** / C₋ **.4159** (dense concordance .7438, 96,397 pairs). Recorded for the
grid's swap column; no round-over-round movement exists on this cell.

### GATE

Δ₀ = **+.0035** vs the frozen **.02** → **STOP AT ROUND 0 (terminal)**.

## What this means for the journalism column

The column's three community/curation residuals now read:

| cell | Layer-1 same-rows Δ_beyond | closure Δ₀ | ε-resolvable? | status |
|---|---|---|---|---|
| BBC most-read | +.0864 / +.0690 | **+.0749** | **yes** (.00252) | rounds run — the column's genuine taste question |
| homepage curation | +.0068 / +.0109 | **+.0035** | **no** (.00882) | **TERMINAL round 0, noise-floor** |
| tweets (V9) | +.0348 / +.0212 | — | — | appendix cell, queued last |

So the journalism column's articulation residual is concentrated in **one** cell. The
curation column is not "fully articulable" in the strong sense — it is a cell where no
residual is measurable, which is a weaker and different claim, and the distinction is
the whole point of running the resolvability check before the curve.

## Artifacts

| what | path |
|---|---|
| round-0 driver | `methods/taste_decomposition/closure/homepage_curation/round0_homepage.py` |
| round-0 results | `.../closure/homepage_curation/round0_results.json` |
| Layer-1 ledger | `methods/taste_decomposition/results/homepage_curation_storygrouped_ledger.json` |
| A/V matrices (v2 bank, 29 criteria) | `outputs/va_gemma_banks_homepage_v2/` |
| dense arm | `datasets/news-homepages/va/dense_standard_storygrouped/` |

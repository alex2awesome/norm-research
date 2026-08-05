# OSL unsupervised-metric panel — deeper quantitative analysis (2026-08-05)

User directive: wrap the M_ω-vs-GEPA campaign (no further robustness arms) and move to
scaling-law work, starting with a deeper analysis of the OSL (observational-scaling-law-style
recovery-vs-executor-capability) panel for unsupervised metrics.

**Data**: `notebooks/data/2026-07-07-osl-multi/curves_<domain>.json` — 1,270 bank metrics
(8 domains), recovery y ± se vs capability index z across 14 executors, original fitted ceiling
L [L_lo, L_hi], regime verdict (RISING 884 / REACHES 321 / BOUNDED 65). Blinded 9-type labels:
`outputs/analyses/osl_metric_types_20260728.json` (O-id → curves-file enumeration verified
metric-by-metric: verdict and domain matched for all 1,270).

**Code/artifacts**: `outputs/analyses/osl_deep_20260805/osl_deep_analysis.py` →
`osl_deep_report.json`. All refits use y = L·σ(k(z−z0)), L ∈ (0, 1.2], 1/se-weighted —
self-consistent within this analysis; they do NOT replace the original panel fits.

## A. Quantitative regime profiles (median [IQR])

| | RISING (884) | REACHES (321) | BOUNDED (65) |
|---|---|---|---|
| original fitted L | .946 [.685, 1.2] | .761 [.681, .832] | .527 [.446, .634] |
| L CI width | .56 | .33 | .105 |
| realized max y | .760 | .798 | .611 |
| y at top executor | .707 | .751 | .542 |
| headroom L − y_max | +.15 | −.04 | −.09 |
| refit R² | .625 | .860 | .814 |
| z90 − z_frontier (refit) | −.34 [−.93, +1.26] | −.87 | −1.07 |
| % saturating beyond frontier | 41.7% | 12.5% | 4.6% |

Reading: BOUNDED and REACHES are clean, well-fit, attained-in-range regimes (negative headroom =
realized max already at/above the fitted ceiling; z90 about 1z inside the frontier). RISING is
the noisy regime (R² .625, CI width .56) — consistent with the standing trust caveat.

## B. RISING decomposes — censoring is a minority story (refit-relative)

Under the logistic refit, the 884 RISING metrics split:

| slice | share |
|---|---|
| already ≥90% of refit ceiling inside observed range | **58.3%** |
| predicted to reach 90% within +0.5z of frontier | 8.5% |
| within +1.0z (≈ 0.8–1.3 params-decades) | 15.2% (cum.) |
| deep-censored (z90 > frontier + 1z) | **26.6%** |

Median z90 − z_frontier = −.34. Family z-per-params-decade: Llama 1.21, Qwen2.5 0.77 (heuristic
ordinal map only — Koyejo guard). **Caveat**: refit-form-dependent — the refit puts RISING's
median ceiling at .816 vs the original .946, and "rising" was assigned under the original form;
the honest statement is that only ~a quarter of RISING is deeply censored under a logistic
refit, while the majority is near its (lower) logistic plateau. The original "rising mostly
records censoring by our strongest executor" caveat is thus conservative but coarse: the regime
mixes near-saturated curves with a deep-censored tail rather than being uniformly censored.

## C. Leave-top-executor-out backtest (extrapolation calibration)

| | RISING | REACHES | BOUNDED |
|---|---|---|---|
| median abs err at held-out top executor | .051 | .036 | .063 |
| % errors > .10 | 21.4% | 14.7% | 16.9% |
| ceiling shift from dropping top exec | ~0 | ~0 | ~0 |

One-step-ahead extrapolation is decently calibrated (median 4–6 pts) with a real heavy tail
(15–21% of metrics miss by >10 pts) — quote median error, never a guarantee, mirroring the
E1 benchmark-prefix backtest phrasing.

## D. Beyond-text ↔ BOUNDED survives within-domain (not a composition artifact)

Permutation test (20k) on the externality index (in-text 0 / interface .5 / beyond-text 1)
within each domain with ≥3 bounded metrics:

| domain | n bounded | beyond-text share: bounded vs rest | Δ index | perm p |
|---|---|---|---|---|
| humor | 41 | .268 vs .122 | +.12 | **.049** |
| peer review | 9 | .333 vs .120 | +.41 | **.004** |
| news homepages | 13 | .231 vs .050 | +.10 | .33 (underpowered) |
| creative writing | 2 | — | — | untestable |

The BOUNDED-loads-on-beyond-the-text association is not purely the humor-heavy composition of
the bounded set: it replicates within humor alone and within peer review alone.

## E. Why REACHES finishes early: steeper, and somewhat earlier

REACHES vs RISING refit shapes: slope k median 2.22 vs 1.44 (Mann-Whitney p ≈ 3e-22); midpoint
z0 median .67 vs .86 (p ≈ 8e-6). Both differ, but the slope gap is the dominant discriminator:
reaches-type criteria (local self-contained patterns) switch on over a narrower capability band.

## Campaign wrap-up decision (same day)

User declined the remaining GEPA robustness arms (+2 seeds @16.7k hotpot, ifbench seed
replicate, token-parity extension, AIME iso-compute probe). The M_ω-vs-GEPA line CLOSES when
the in-flight hover full-parity certification rescore (hvcert10110, prereg HB197 §3) lands —
no further arms after it. Advisor rebuttal-list items are recorded as considered-and-declined.

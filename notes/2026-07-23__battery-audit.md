# Battery audit — full manual verification (user-requested, 2026-07-23)

Scope: every battery implementation + every W0 result, with special force on convergence
integrity (bugs that create spurious probe-agreement would FABRICATE the "underlying tacit
concepts" evidence). Verdict up front: **implementations verified; results reproduce exactly;
the headline convergence survives two adversarial nulls and three partials; four fixes
applied; three interpretive flags recorded.**

## 1. Reproduction checks (battery vs independent standalone computations) — ALL PASS
| check | result |
|---|---|
| conditioned rescues per rung (P-CHAN-core) vs ladder tallies | 0/4/19/9 = EXACT match |
| differentiation PC1 (P-SCAL-3) vs standalone factor run | .514/.634/.389 EXACT |
| subspace caps (P-CEIL-1) vs standalone jsonl, 90 cells | max |diff| = 0.000000 |
| scaling-tacit class count (P-SCAL-1) vs OSL v0 | 41/90 EXACT |
| STAT-1 spot recomputes from raw grids | 5/5 exact |

## 2. Convergence integrity (the decisive audit) — HEADLINE SURVIVES
Question: is STAT-1 × GEN-1 (+.81 rung-collapsed; +.649 at the 7B rung alone) real
construct-level structure, or artifact? Both probes share a component (the per-item
agreement vector), so this was the highest-risk pair.
- **Partial correlations (attenuation test):** controlling construct-level mean agreement
  (+.650), rho-to-target (+.652), executor-vec std (+.669) — the correlation DOES NOT MOVE.
  Not an attenuation/noise-thermometer artifact.
- **Null A (marginal- and agreement-matched synthetic executors, iid noise):** mean +.199,
  p95 +.354, max +.395 over 50 sims. → **~+.20 of any STAT×GEN correlation is structurally
  induced by the shared agreement component — a real, quantified artifact floor.**
- **Null B (adds shared executor item-effects):** p95 +.422.
- **Observed +.649 decisively exceeds both null p95s.** The convergence is signal above every
  artifact channel we could model. NOTE for P-B2: its ≥+.50 bar sits ABOVE the measured
  artifact floor (+.35/.42 p95) — the prereg threshold is retroactively justified, and the
  artifact floor is now part of the interpretation standard (report correlations MINUS floor).

## 3. Code-level findings and fixes (applied; 21/21 tests green; w0_v4 regenerated)
1. **Degenerate executor vectors:** 4/90 humor cells at 7B have vec std < .02 (one exactly
   0.0) — ranks/agreement meaningless there. FIX: std<.01 guard in STAT-1 and GEN-1 (cells
   skipped, not fabricated). Post-fix PC1: humor .29 / n&c .27 / math .36 (was .30/.29/.36 —
   structure unchanged).
2. **write_profile_rows docstring** claimed append+dedup that isn't implemented — corrected
   to immutable-file-per-(domain,tag) semantics.
3. **Earlier fixes re-verified as sound:** per-domain store filenames; SCAL-2
   divergence-slope primary (dup of CHAN-core eliminated: +1.00 → −.24).
4. Audit-harness note: audit scripts must run from repo root (PACKET_ROOT is root-relative)
   — the one audit-script failure was in the harness, not the battery.

## 4. Interpretive flags (recorded, not bugs)
- **P-GEN-1 sign convention is theoretically contestable** (is steeper typicality-decay
  "more tacit"? we read it as core-boundedness). The finding it supports is stated
  sign-explicitly: metacognitively opaque constructs are core-bound constructs.
- **P-GT-3 tacitness-direction is weakly theorized** (descriptive dimension; keep in the
  profile, exclude from any single-factor "tacitness level" summary).
- **Rung-collapsing in convergence** (mean across rungs) mixes rung effects; single-rung
  values reported alongside (7B STAT×GEN = +.649 vs +.81 collapsed).
- **item_agreement under near-constant vectors** produces a structured V-pattern — this is
  the mechanism behind fix #1; guard prevents it entering statistics.

## 5. Standing conclusions after audit
- The multidimensionality result (PC1 ~.27–.36, all three domains) and the two blocks
  (STAT×GEN; CHAN×SCAL-1) are implementation-trustworthy.
- Convergence claims now carry a measured artifact floor (+.20 for shared-component pairs);
  the confirmatory analysis (and Model-A MIRT) must clear it explicitly.
- All W0 quotable numbers migrate to run_tag **w0_v4** (guarded probes). w0_v1 remains
  never-quote (dup-inflated); w0_v2/v3 superseded.

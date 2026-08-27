# CW §12.6 ceiling structure — two axes + the probe-coverage artifact (2026-07-02)

Source: `methods/metric_implementer/experiments/gepa_vs_ceiling.py` + `metric_entropy_check.py`
on `aligned_8b_orbit_v2` (native 8B, un-confounded). n=46 metrics. Also holds on 3b_v2.

## Two independent axes of the upper-bound certificate
1. **Ceiling value OPT_Ω = H(M_i) × %H.**
   - **H(M_i)** is the dominant driver (range 0.08→1.00, 12×). %H = OPT/H_M, the articulation
     efficiency, varies only 0.47→0.81. So the ceiling is set mainly by how much the metric's
     verdict varies across the 300 probes.
   - |Ω| is ~constant (raw ~610–843, distinct D_obs ~16–62, greedy head ~3–13) and does NOT
     drive OPT — e.g. Grounded-fantastical-realism has the highest D_obs (62) but mid OPT (0.46).
2. **Combiner gap g₁/OPT** (best single criterion vs the subset ceiling) is a SEPARATE axis:
   - Need-a-checklist (low g₁/OPT ~0.48–0.52): Sustained tension, Diction, Originality, Worldbuilding.
   - Single-codable (high g₁/OPT ~0.69–0.76): Macro plot, Foreshadowing, Visual storytelling.
   - Independent of the ceiling: a metric can be high-ceiling+checklist (Diction) or
     moderate-ceiling+single-codable (Macro plot).

## The GEPA-prompt-vs-ceiling question (no GEPA was run — freegen-only Ω)
- OPT_Ω = best multi-criterion CHECKLIST (combiner F₁), NOT a single prompt.
- g₁ = best single criterion (a LOWER bound on any single prompt incl. GEPA-optimized).
- Δ = OPT_Ω − g₁ = combiner gap (structural: a single prompt is one term in the additive combiner).
- Best single criterion captures only **~57%** of the ceiling (mean g₁/OPT); best-3 capture ~84%;
  greedy head ~9. So CW metrics are multi-criterion-codable, NOT single-prompt-codable.
- Caveat: g₁ uses freegen criteria (not recovery-optimized); a true GEPA prompt ≥ g₁, so the gap
  is an over-estimate of what an optimized single prompt would leave. But single < checklist
  structurally, so the gap is real.

## ⚠️ Probe-coverage artifact — the low ceilings are NOT inherent un-articulability
H(M_i) is itself explained by base-rate balance (how often the metric FIRES on the probe set):
- **Subplot/timeline: fires on 3/300 probes (base 0.01) → H_M 0.08 → OPT 0.04.**
- Line-level clarity 25/300 (0.08); Authentic sociocultural 28/300 (0.09); Setting-as-active-force
  32/300 the other way (0.89); Show–tell 41/300; Authorial voice 44/300.
- Well-exercised (base ≈ 0.5, n_min ≥ ~100): Diction 150/300, Worldbuilding-quality 149,
  Short/flash 147, POV 129, Dialogue 125, Sustained tension 133, Opening 123, Pitch 135, … ~12 metrics.

⇒ **OPT_Ω / trichotomy claims are only valid for the ~12 well-balanced metrics.** The ~11
probe-undersampled metrics (esp. Subplot) need a probe set that exercises them before any ceiling
statement. This is **probe-undersampling** (few probes fire the metric) — DISTINCT from the
certificate's **species-undersampling** (few criteria). A metric can be species-rich but
probe-starved (Subplot: D_obs 23 species, only 3 firing probes) — its ceiling is artifactual.

## Fixes / next steps
- Rebalance or stratify the probe pool per metric (or restrict ceiling claims to base ∈ [0.3,0.7],
  n_min ≥ ~80) before quoting OPT_Ω or CODIFIABLE/DEEP verdicts.
- For the GEPA-vs-ceiling number precisely: run GEPA prompt-optimization on the ~12 well-exercised
  metrics (freegen g₁ is only a lower bound).
- The two-axis structure (ceiling = H_M×%H; combiner gap = independent) is the right frame for
  reporting CW upper-bound certificates.

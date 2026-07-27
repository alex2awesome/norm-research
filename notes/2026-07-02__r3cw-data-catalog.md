# R3-CW / R3-humor data catalog — provenance, validity, and the standing gate

*Written 2026-07-02 (~2:30 AM) as part of the scale-up. Machine-checked twin:
`outputs/r3_cw/CATALOG.json`, produced by
`python -m methods.metric_implementer.experiments.catalog_check --root <root> --task <task>`.
**Policy: no certificate, grid, or scaling number consumes a dir that isn't OK (or explicitly
acknowledged) in the catalog.** The checker verifies per checkpoint: C1 name↔hierarchy identity
(the level-dispatch bug-catcher), C1b rubric-text identity (via the `target_desc` field persisted
by rescore_executor after 2026-07-02), C2 target/probe shape, C3 stale τ₀ literal, C4 form-gate
data presence, C5 orbit-field consistency, C6 rescore provenance. Manual verdicts live in
`CATALOG_OVERRIDES.json` and are carried into the catalog verbatim. 12 unit tests cover the
checker, the form decomposition, and the grid driver.*

## Directory registry (outputs/r3_cw unless noted)

| dir | what it is | validity |
|---|---|---|
| `llama8b_glm` | SOURCE: run_alpha_probe, 8B executor, 3×GLM-4.7 families, 300 probes; 54 R3 metrics after tonight's 10-fill (61/63/64 + in-flight 16-fill remainder outstanding) | **OK** — correct by construction (its loader dispatches levels properly); new metrics carry forminv WITH per-pair records |
| `aligned_8b_orbit_v2` | corrected 8B orbit retarget (46) — the Φ-averaged m̄ₒ targets | **OK** — forminv copied from source (sigs verbatim, provenance note) |
| `aligned_3b_orbit_v2` | corrected 3B M over the 3B dir's own sigs (41) | **WARN** — no forminv (3B criteria-fragility pass queued); τ₀ literal carried from v1 |
| `aligned_70b_orbit` | 70B v2 full rescore, other agent's relaunch 23:46 (post-fix code), filling | **OK so far** — lacks forminv (launch predates `--form-invariance-n`) and `target_desc` (predates C1b patch) |
| `aligned_8b_orbit`, `aligned_3b_orbit` | v1 aligned runs | **DEPRECATED** — M_i computed from the WRONG metric's rubric (level-dispatch bug); own-pair vs cross-pair spearman 0.725 vs 0.744. Sigs + copied forminv remain valid |
| `llama8b_to_llama3b` (41), `llama8b_to_llama70b_fp8` (8), `llama8b_to_qwen122b` (14) | OLD pre-orbit cross-executor rescores, discovered BY the catalog run | **SUSPECT-M** — same bug era: sigs/B_E-based numbers valid, **any M-based number (T/R staircase) from them is invalid** |
| `_smoke_formfix`, `_smoke_grid` | FakeVLLM wiring fixtures | **FIXTURE** — never data |
| `grid_cw_v1` | Face-2 decompression grid (46 metrics × 6 type rungs × readers 1B/3B/8B; 70B reader later); messages by local 70B writer | building tonight |
| `../r3_humor/llama8b_glm` | humor Face-1 sweep: 60 R3 clusters, same instrument config as CW source | building tonight |

## The two standing lessons encoded here

1. **Identity must be machine-checked, not filename-implied.** The level-dispatch bug lived for
   weeks because `r2_idx` + a filename convention *implied* identity that no artifact verified.
   Now every checkpoint carries `name` (+ `target_desc` going forward) and the catalog compares
   them against the hierarchy the filename claims.
2. **Optimizer/adaptive outputs are catalogued separately from iid samples** (gepa-tagged criteria
   already excluded from capture–recapture; grid messages carry a writer-model field) — provenance
   of *how a text was produced* is part of validity, not metadata.

Related: [[rescore-level-dispatch-bug]], notes/2026-07-01__form-effects-control-plan.md,
notes/2026-07-01__cw-unified-grid-roadmap.md §2b STATUS.

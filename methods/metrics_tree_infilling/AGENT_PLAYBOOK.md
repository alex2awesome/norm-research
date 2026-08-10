# Agent playbook — verifying and applying ctree/MOB + global infilling to a new domain

*2026-07-01. A step-by-step protocol an agent (or human) follows to bring the metric-infilling
machinery to any labeled corpus. Written after the CW post-mortem, whose failure modes each gate
below exists to catch. Theory references: `notes/2026-07-01__metric-count-certificates.md` (MCC)
and `notes/2026-06-18__prompt-optimality-theory.md` (PO).*

**The two engines** (run BOTH; they see different residual shapes — MCC §5):

| engine | entry point | detects |
|---|---|---|
| ctree/MOB tree infilling | `loop.run_infill` | (i) moderation-shaped + (ii) region-shaped residual |
| global infilling | `global_infill.run_global_infill` | (iii) corpus-uniform residual (and corpus-wide (ii)) |

Run global FIRST (cheaper, no z-design decisions); then the tree on what remains.

---

## Phase 0 — data audit (do NOT skip; every CW confound was here)

1. **Label sanity.** `y` binary, both classes ≥ 20% after any balancing. Spot-check ~20 items by
   hand: is the label what you think it measures? (news_homepages lesson: label was spatial
   layout, not clicks.)
2. **Mixture check.** Plot label rate vs original row position, source file, and any id prefix.
   CW's file was TWO concatenated datasets (rates 0.47/0.15). Any such axis becomes (a) a
   **stratification variable** — grouped splits, or per-stratum runs — never a predictor;
   (b) a **known-moderator positive control** (Phase 2).
3. **Identity confounds.** Author/publisher/venue ids that predict y are reputation channels,
   not articulable metrics (press-release lesson: 0.71 → 0.58 after deconfounding). Decide the
   deconfound BEFORE measuring any ceiling.
4. **Grouping.** If multiple rows share a source (pairs, threads, same author), set
   `group_split_by_id=True` so discover/guard/test never straddle a group.
5. **Splits.** Three-way `discover/guard/test` via `io_metrics.three_way_split`. The test split
   is READ ONCE, at the very end.

## Phase 1 — bank materialization + judge audit

6. **Load metrics** (`load_rubric_metrics` / `load_code_metrics`). Record pool size. If the pool
   is huge (CW: 73,702), select the working bank by **coverage, not head-of-file**: embed, cluster,
   take medoids (the `limit=40` CW diagnostics read the first 40 of one taxonomy — acceptable for
   algorithm diagnostics, NOT for a power claim).
7. **Judge score-distribution check** (standing memo): per metric, applicability rate and level
   std on a 60-item probe. A structured-output judge silently emitting all-min scores = valid
   JSON, zero signal. Drop or re-prompt collapsed metrics; log `viable k / total`.
8. **Baseline bank AUC** on guard (fit discover). This is the number everything must beat.

## Phase 2 — algorithm verification gates (all must PASS before any discovery run)

9. **G1 planted-break control** (`diag_mob_soundness.py` pattern): synthetic sign-flip moderator
   among m_z-many nulls at the REAL run's (n, m_z, n_perm). Must split on the planted axis and
   stump on stable data. Print the p-floor: `min adj_p = m_z/(1+n_perm)`; require `< alpha`
   (n_perm ≥ 999 for m_z ≥ 20 — the CW n_perm=199 bug made splitting impossible).
10. **G2 known-moderator control on REAL data** (`diag_cw_knownmod.py` pattern): put the Phase-0
    mixture/source axis into z; the tree MUST split on it **under the z-design you will ship**.
    If it does not, your z is over-populated (multiplicity tax — CW: p=0.003 → adj_p=0.144 at
    m_z=48) or the run is underpowered. Fix design, not alpha.
11. **G3 z-design.** `curated_z_only=True`; z = 3–6 hypothesized ITEM-level axes only
    (source/stratum, embedding text_cluster fit on discover, genre/topic tags, length). Metrics
    stay in X. Every z column divides alpha — justify each.
12. **G4 planted-metric control for the GLOBAL loop** (`test_global_infill.py` pattern, on real
    texts): inject a synthetic marker correlated with relabeled y; the loop must propose-accept
    it and reconstruction must round-trip. Verifies proposer plumbing + acceptance gate + ledger
    end-to-end on the domain's actual text distribution.

## Phase 3 — discovery runs

**Multi-arm requirement (2026-07-02).** Run at least TWO proposal arms through the same gate —
`generators.py` provides `residual`, `unconditional` (autorubric-style), and `label_contrast`;
`scripts/tools/run_arm_comparison.py` is the driver. A single arm makes the flux upper bound
anti-conservative (false saturation); the certificate report prints an honesty note when it
sees one arm. Compare arms on accepted-BITS-per-proposal. Certificates: `certificates.py`
(`report_from_ledger` -> value bracket + N_lower/N_upper + Minoux read); requires
`min_bits_gain > 0` so the ledger carries the bits currency. **The certified artifact is the
UNION ledger across arms** (MCC §2a): `report_from_ledgers` -> `certificate_union.json`
(emitted automatically by `run_arm_comparison.py`); per-arm certificates are diagnostics only.

13. **Global infilling** (`run_global_infill`): `acceptance = guard AUC gain ≥ min_auc_gain`
    (set ≥ 2× the guard split's AUC noise — measure by refitting the baseline with 5 seeds;
    CW used 0.03). patience=2, max_rounds≈6. Budget: each round ≈ 1 proposer call +
    (n_discover + n_guard) judge calls + ~120 reconstruction calls.
14. **Tree infilling** (`run_infill` with the G3 z-design): gap nodes → WRONG/RIGHT contrasts →
    per-leaf proposals; same acceptance gate on guard.
15. **Per-metric ledger** — the three tracks are MANDATORY for every proposal (global path emits
    them automatically; MCC §4 needs them for the count certificates):
    - **data-to-develop**: `n_proposal_examples` + `data_curve` (gain at 25/50/100% train) +
      `min_train_frac`;
    - **applicability**: fraction applicable on discover AND guard (a metric applying to 8% of
      items is a leaf-metric even if found globally);
    - **reconstruction R**: rederive the rubric from (text, verdict) pairs (reconstructor never
      sees the rubric), re-execute, report balanced agreement + AUC. Built-in = best-of-k;
      the GEPA plug-point is `reconstructor_fn` — wire `gepa_viable.py`'s loop to OPTIMIZE the
      rederived rubric against agreement on a dev slice, report the optimized R (this is the
      "GEPA-optimized reconstruction accuracy" number; PO §11.1 headroom analog). Set
      `GEPA_CORPUS` env var (standing trap) and be sparing with GLM quota.

## Phase 4 — the honest read-out (touch test split ONCE)

16. **Final power**: baseline-bank AUC vs (bank + kept metrics) AUC on the untouched test split,
    same split, same input (apples-to-apples memo). Also in BITS (log-loss reduction) — the MCC
    count certificates only compose in bits.
17. **Attribution decomposition** (the CW holdout lesson — do not skip): tree-routed gain must be
    decomposed vs (a) axes-only model, (b) bank+axes additive model. Only the margin over (b) is
    moderation signal; axis main effects may be confounds (source_half) or legitimate
    articulable content axes (text_cluster) — classify each axis explicitly.
18. **Certificates** (MCC §4): `N_lower = ceil(V_bits / log2 K)`; `N_upper = |S_g| + (U − V)/δ`
    with U = dense-stack wrap ONLY if the dominance gate passes (dense scaling curve plateaued
    AND bank ≤ C − margin); otherwise report the articulable ceiling as right-censored.
19. **Verdict per MCC §5 trichotomy**: which residual shapes fired, which were silent, and what
    silence means at your (n, m_z, n_perm) — a stump is "no (i)-residual surviving THIS
    Bonferroni," never "saturated."

## Known per-domain hazards

| domain | hazard | phase |
|---|---|---|
| creative-writing | two-dataset mixture; 73k near-dup rubric pool; noisy gold | 0.2, 1.6 |
| press releases | publisher-id + topic confound (0.71 → 0.58 deconfounded) | 0.3 |
| code review | dynamic subtest names → false F2P labels (Go); prefer pre-2023 PRs | 0.1 |
| math | AoPS V/A artifacts are laptop-only; Math.SE identity features not artifact-intrinsic | 0.3 |
| peer review | agency-response structural mismatch; semantic-S confound in same/diff judging | 0.1 |
| humor | New Yorker caption ratings are crowd-worker, not genuine humor judgment | 0.1 |

## Operational rules (standing)

- Judge/proposer backend: glm-5.2 via z.ai anthropic endpoint (`ANTHROPIC_BASE_URL=
  https://api.z.ai/api/anthropic`, key `~/.z-ai-api-key.txt`), concurrency ≤ 2; be sparing
  (monthly quota). sk3 vLLM for bulk if a GPU is free (max 1 GPU; stack processes).
- Cache every judge call (`cache_dir`); the CW holdout rerun was free because of it.
- Verify long runs by log line + `ps`, never by launch success. Kill by specific PID only.
- Never delete data; new artifacts get `_v2` suffixes.

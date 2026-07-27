# v13.1 Multi-MCQ value measurement: the certified recipe (handoff spec)

**Status: design-frozen candidate for the v13 prereg. 2026-07-13. Author: Claude (pipeline owner as of today). Audience: the implementing agent.**

This supersedes the single-pool "S-superset" sketch. It answers three demands: (1) get multiple-MCQ *exactly* right (estimand first, estimator second); (2) only trust V_unseen quantities that are certified — enumeration and convex/MILP duals, never extrapolation; (3) generalization over teaching examples must be *measured*, not assumed.

---

## 0. Objects

- Executor E (frozen; Llama-3.1-8B, constrained two-token readout v3). Metric M_b = frozen verdict vector on the probe pool. Design split D (|D| ≈ 120), eval split disjoint.
- Prompt behavior σ(p) ∈ {0,1}^|D| (hard verdicts on D; total by construction).
- Teaching panel T: ordered 8-subset of D. MCQ instrument: reconstructor sees the 8 texts of T labeled by σ(p)|_T + a frozen 4-option menu; per-panel value v_T(s) for s ∈ {0,1}^8 =
  clip( mean over the declared permutation design of P(gold) − max(blind, shuffled controls), 0, 1 − q0 ).
  **v_T is a finite, exactly-measurable function: 256 entries per panel, one deterministic constrained-logits query per (panel, state, permutation). No sampling noise with a local logits reconstructor.**
- FACT (load-bearing, already fail-closed-guarded in `cr3_sampled_value_certify.py`): V_T(p) = v_T(σ(p)|_T). Value factors through the panel restriction; nothing else about p enters the query.

## 1. Panel design: replicated frozen pools (answers "do we trust 12–16?")

We do NOT trust one small pool. Design:

- **G = 6 disjoint pools** S₁…S₆ ⊂ D, each of size **S = 12**, chosen by the existing deterministic library builder logic (design-split-only; stratified on executor-verdict balance and behavioral diversity; NO prompt or label input), frozen into the manifest before any prompt valuation. Coverage: 72/120 design texts.
- **Panel family 𝒫 is finite and declared**: per pool, **R_g = 12** panels (deterministic subsets of the pool's 12 texts, verdict-balanced), so |𝒫| = 72 panels total. Panels are evaluated **exhaustively over 𝒫** — the estimand is a finite mean, not a sampled one:

  **V̄(p) = (1/G) Σ_g V_g(p),  V_g(p) = (1/R_g) Σ_{T∈pool g} v_T(σ(p)|_T).**

  Consequences: (a) no panel-sampling CI is needed — V̄ is exact given the tables; (b) the generalization evidence becomes a **variance decomposition**, reported per prompt and per metric: within-pool spread (panel choice) vs across-pool spread (teaching-text choice). The across-pool component is the honest answer to "does value depend on which examples we picked?" — measured across 6 independent 12-text windows, not assumed.
- Sensitivity row required in every certificate: min over pools of V_g(best prompt) — the worst-pool value of the headline prompt.

Why S=12 per pool: the pool-joint behavior space is 2^12 = 4,096 — small enough that (i) exact caps are enumerable, (ii) capture-recapture on the pool-pattern species **saturates measurably** (2,378 mined prompts vs 4,096 possible patterns; Good-Turing is sharp there). Breadth comes from G, not from S.

## 2. The trust hierarchy for V_unseen (only levels 1–4 may enter certificates)

1. **Exact per-pool cap (enumeration).** U_g = max over x ∈ {0,1}^12 of V_g(x-as-pattern). Needs the full 256-entry table for each of the pool's panels; the 4,096-pattern outer max is free CPU (each pattern's V_g is R_g table lookups).
2. **Recombination-exact joint cap (MILP dual).** V̄_cap* = max over x ∈ {0,1}^72 of (1/G)Σ_g V_g(x|_{S_g}) — but pools are disjoint so this separates: V̄_cap* = (1/G) Σ_g U_g **exactly** (no MILP needed with disjoint pools — the separability is the point of disjointness). Keep the MILP formulation in the toolbox for any future OVERLAPPING panel family: variables y_{T,s} ∈ {0,1} one-hot per panel-state, channel constraints y_{T,s} ≤ x_i / 1−x_i per bit, objective Σ w_T v_T(s) y_{T,s}; the **solver's global dual bound is a certified upper cap at any time budget** (anytime-safe); on timeout fall back to level 3. ~5K binaries at this scale — trivial for HiGHS/CBC.
3. **Free-recombination cap**: mean_g max_s (per-panel maxes). Valid, loosest structural cap. Fallback only.
4. **Headroom cap**: 1 − q0. Always available.
5. ~~Extrapolation (Lipschitz/Hamming smoothness, GPs, learned predictors, VQ-style codes)~~ — **BANNED from certificates.** Certification requires exact factorization of value through the quotient; learned discretizations only approximate it and the approximation error is unmeasurable without the enumeration you were trying to avoid. Learned codes are welcome as *diagnostics* (e.g., variance explained by a K-code clustering of σ(p)) in a clearly-labeled non-certified appendix.

Bonus diagnostic (free from the tables): the full Hamming-1 flip-sensitivity spectrum of every v_T — publishes exactly why smoothness assumptions would have been false.

## 3. Value-added of mining: two certified gain bounds (report both, quote the min)

Let A = observed best V̄, and per pool A_g = observed best V_g.

**(a) Pool-decomposed bound (usually the tight one).** For each pool g, run per-family Clopper-Pearson first-contact novelty on the **pool-pattern species** (the 12-bit pattern σ(p)|_{S_g}) → U0_g (CP upper). Then per horizon m:

  gain_g(m) ≤ (1 − (1 − U0_g)^m) · max(0, U_g − A_g)
  gain(V̄, m) ≤ (1/G) Σ_g gain_g(m) + Δ_recomb, Δ_recomb = (1/G)Σ_g A_g − A ≥ 0 (observed recombination slack, measured not assumed).

  Derivation: max of a mean ≤ mean of maxes; within each pool a draw improves V_g only by hitting an unseen 12-bit pattern; the slack term prices the fact that today's best-per-pool prompts are different prompts.

**(b) Joint-species bound.** CP first-contact on the concatenated 72-bit pattern → U0_joint; gain(m) ≤ (1 − (1 − U0_joint)^m) · max(0, V̄_cap* − A). Conservative when the joint species is fine-grained; kept because it needs no slack term.

**(c) DKW expected-best-gain** on the observed V̄ distribution per family (already implemented) covers the seen-pattern regime. Certificates carry (a), (b), (c); the headline value-added claim is the minimum, each labeled with its premises.

Premise flags in every payload: iid-within-family exchangeability for CP; disjoint-pool separability for the cap; **no smoothness, no submodularity, no independence-across-lists assumptions anywhere.**

## 4. Menu integrity gates (responds to today's reconstructor crisis — REQUIRED before any sentinel)

The pending fidelity diagnostic tests whether menu golds (written descriptions) diverge from executor behavior. Regardless of outcome, the recipe adds two frozen gates at codebook-build time:

- **Gold-fidelity gate:** a strong judge (Sonnet-class, anchor-tested), blind to the menu, applies the gold description to the pool texts; agreement with σ_target must be ≥ 0.75 on teaching texts (else the metric is FORMAL_ONLY: its behavior is not describable by its own description — that's a finding, not a discard).
- **Best-explanation gate:** given the executor labels on a panel, the gold must be the argmax "which option explains these labels" under the *qualified* reconstructor for ≥ 60% of panels (else menu regeneration with behavior-grounded distractor/gold text, produced by a model disjoint from the reconstructor, then re-gate).
- Blind prior balance is re-calibrated per reconstructor (already the v13 rule); the qualification battery (predeclared 3-rule gate, now run 3×: Gemma ✗, GLM ✗, Llama-70B pending) picks the reconstructor.

## 5. Cost + tiering (scoped)

Per metric, logits reconstructor, perms = 8 counterbalanced, G=6, R_g=12:
| item | queries |
|---|---|
| full tables (72 panels × 256 states × 8 perms) | 147K |
| pool valuation rides the same tables (lookups) | 0 extra |
| blind + shuffled controls | ~2K |

~150K deterministic short queries/metric ≈ 45–75 min on one GPU with prefix caching. **Tier A (sentinel + headline metrics): full recipe.** **Tier B (fan-out):** G=3, R_g=8, tables only on states observed + unseen-slice of 2 pools, caps at level 1 for those pools and level 4 elsewhere ≈ 35K queries ≈ 12–20 min/metric. Tier assignment predeclared. All caps/tiers recorded per certificate; a Tier-B certificate can be upgraded to Tier-A later by evaluating the missing table entries (append-only; same frozen instrument).

## 6. Implementation mapping (for the implementing agent)

Already built on branch `cr3-sampled-v13` (`cr3_sampled_value_certify.py`): state-factorized valuation with per-state caching (fail-closed panel-order guard), exact-finite-mean aggregation is a parameter change (evaluate all declared panels instead of sampling), DKW machinery, immutable certificates, dual 95/90 tiers. In flight from a prior instruction: joint-tuple CP + unseen-state pricing — that work item is the single-pool special case of §3(b)+§2; extend, don't discard: pools = the panel library builder called G times on disjoint strata; per-pool CP = same first-contact counter keyed by 12-bit pattern; Δ_recomb and the pool decomposition are ~40 lines on top. The 256-state table builder already exists (`write_finite_state_scored_artifact` / `build_finite_state_envelope`) — reuse per panel. MILP path: optional module, HiGHS via scipy.optimize.milp or OR-tools, only needed if pools ever overlap; ship the formulation with a planted test (known optimum) but keep pools disjoint in v13.1 so level-2 = Σ exact level-1.

## 7. What is future-proofed

- Reconstructor swap: everything downstream of the tables is reconstructor-agnostic; a new reconstructor re-runs prior calibration + tables under the same frozen pools/panels (append-only, new namespace).
- Executor swap (the likely v14 move if gold-fidelity fails at 8B): pools/panels are index sets over D — re-score σ under the new executor, rebuild tables; species definitions unchanged.
- Sampled-readout reconstructors (API models): every certified quantity gains a per-query binomial CI; the cap levels 1–3 become upper *confidence* caps (Bonferroni over 256×perms per panel) — formulas unchanged, α bookkeeping already in the tier machinery. Predeclared, but discouraged: logits readout is strictly cleaner.
- Larger S or overlapping pools: MILP dual (level 2) is the designated upgrade path; never extrapolation.

## 8. Open parameters for the prereg (recommended defaults)

G=6, S=12, R_g=12, perms=8, Tier-A set = sentinel 6 + top-2 per task, gold-fidelity threshold 0.75, best-explanation threshold 0.60, horizons m ∈ {100, 300}, α=0.05 with the existing 95/90 tier split.

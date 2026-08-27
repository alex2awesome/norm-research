# Shared-Ω retarget — adversarial validity review (2026-07-02)

**Context.** The aligned CW §12.6 ladder can't feasibly full-rescore the ~775-criteria
Ω pool on 70B/Qwen (~12 days/executor; the certificate's **flux/ε-gap needs ALL ~775
scored per executor** — capture-recapture over criteria-as-species). 8B/3B already have
their own full-pool sigs (prior multi-day runs). User-approved decision (Option 1): 70B/Qwen
use `--retarget-mi-only` — **reuse 8B's full Ω (signatures) and compute only their own
orbit-averaged M_i**. So a 70B checkpoint = {8B sigs (775×300), 70B M_i (300,)}.

**Review.** 4-lens adversarial workflow (`w88lvmy73`, glm-4.7) → synthesis verdict:
**CAVEATED (salvageable, not unsound).**

## What holds
- **HEAD (OPT_Ω) is sound under PROCESS-RELATIVE framing.** `greedy_head` is partition-free
  (duplicates gain≈0 → tail), so OPT_Ω(8B-Ω, 70B-M_i) is a well-posed answer to *"how much
  checklist-articulability does the FIXED proposal process Ω recover when evaluated by
  executors of increasing capability?"*
- **Recovery lower bound is unaffected.** R(p̂_executor) targets the executor's own M_i
  (recon_channel, separate). The M_i scaling / recovery story stands.

## What is compromised (load-bearing caveats)
1. **FLUX / ε-gap for retargeted tiers ≠ executor's own.** ε(70B) = "value 8B has NOT
   surfaced, evaluated against 70B's M_i," NOT "value 70B hasn't surfaced." ⇒
   **executor-specific flux claims are INVALID; cross-tier ε comparison is FORBIDDEN.**
2. **Coverage-bias LOWER bound.** If 8B collapses signatures 70B would distinguish,
   OPT_Ω(70B) UNDERESTIMATES 70B's native ceiling. Report as a lower bound, not intrinsic.
3. **GV7 stationarity** holds for the HEAD but is ambiguous for the flux under mixed executors.
4. **Mixed-executor conditional value.** Certificate computes I(M_70B; B_8B[selected]),
   not I(M_70B; B_70B[selected]).

## Decision
**Option 1 stands.** Review explicitly does NOT recommend Option B (head-only panel — would
add a third, incomparable 120-species flux estimate). If executor-specific flux ever becomes
load-bearing, only Option D (reduce Ω to ~150, re-run ALL executors full) is sound.

## Mandatory paper disclosure (from review)
- **Provenance columns**: every 70B/Qwen cert reports `source_executor=8B, target=70B/Qwen,
  retarget_mode=True`. Scaling table distinguishes native-Ω tiers (3B, 8B) from retargeted (70B, Qwen).
- **Frame scaling as PROCESS-RELATIVE**, not executor-relative.
- Report OPT_Ω(70B/Qwen) as a **lower bound** on native ceilings.
- **No cross-tier ε comparison; no "70B discovers more hidden value" claims.**

## Cheap validations on existing/coming data (de-risk the top 2 risks)
1. **Head-overlap / sigs-source effect** (CPU, NOW on 8B/3B native certs): Jaccard of the
   greedy head — (A) 8B-native vs 3B-native; (B) 8B-sigs vs 3B-sigs holding the 8B target
   fixed (= the exact retarget operation). (B)>0.7 ⇒ sig-reuse empirically benign; <0.4 ⇒ strong confound.
2. Gain-correlation 8B vs 70B (when 70B certs land).
3. v_additive spectrum comparison (Hill ξ) across executors.
4. f1/N stability across order permutations.
5. **Recovery-bracket coherence**: R(p̂_executor) ≤ OPT_Ω must hold; a violation ⇒ retarget
   breakdown (8B sigs too coarse) → flag for full re-score.

## Files
- Full review JSON: `/private/tmp/.../tasks/w88lvmy73.output`
- Validation script: `methods/metric_implementer/experiments/validate_retarget_overlap.py`

---

## MEASURED CONFOUND (2026-07-02 ~03:30) — review's "caveated/salvageable" is OVERTURNED

Ran `validate_retarget_overlap.py` on 8b_v2/3b_v2 (41 common metrics, same criteria +
probes, own sigs). The head-Jaccard is ~0 (mean 0.013–0.036) but that is NOT decisive
with the ~775-redundant freegen pool (disjoint-but-equivalent representatives). The decisive
metric is **OPT_Ω (bits) under sig-source vs target swaps**:

| contrast | mean\|ΔOPT_Ω\| | corr | means |
|---|---|---|---|
| **(C) sigs-source** OPT(8Bsigs,8Bm) vs OPT(3Bsigs,8Bm) — the CONFOUND | **0.354** | 0.73 | 0.577 → 0.223 |
| **(D) target** OPT(8Bsigs,8Bm) vs OPT(8Bsigs,3Bm) — scaling SIGNAL | **0.341** | 0.41 | 0.577 → 0.236 |
| (E) native OPT(8Bsigs,8Bm) vs OPT(3Bsigs,3Bm) | 0.143 | 0.50 | — |

**D/C = 0.96: the signature source moves OPT_Ω as much as the executor's own M_i.** Per-metric,
*native* (matched-executor) OPT_Ω ≈ 2.5× the *cross* (mismatched) value (metric0: 0.762 vs ~0.29).

**Conclusion: criteria signatures are executor-idiosyncratic — they do NOT transfer.** The
retarget (8B sigs + 70B/Qwen M_i) gives a CROSS OPT_Ω ≈ 0.4× the executor-native value. So:
- The current ladder is **INCONSISTENT**: 8b_v2/3b_v2 are NATIVE (own sigs, OPT_Ω ~0.5); 70b_v2/
  qwen_v2 are SHARED-8B-Ω (cross, ~0.23). A Δ(E) scaling across them is a MISMATCH ARTIFACT.
- Retarget **M_i stays valid** (each executor's own verdict → recovery lower bound unaffected).
  Only the OPT_Ω upper bound is compromised for 70B/Qwen.

### Morning decision fork (for user)
- **(Q1) Consistent shared-8B-Ω ladder**: ALL tiers use 8B sigs (recompute 3B from llama8b_glm —
  cheap). Answers "how well does the FIXED 8B criteria-process articulate each executor's verdict"
  — feasible, consistent, but PEAKS AT 8B (not a capability monotone). Not "per-executor ceiling."
- **(Q2) Native per-executor ceilings**: needs 70B/Qwen's own sigs — full rescore infeasible (~6
  days); **Option B (head-only, score top ~120 criteria with 70B/Qwen, ~25–40 hr)** is the
  feasible approximation. The measured confound now JUSTIFIES Option B (review dismissed it
  before this data). Cleanest for the "bounds across all models" goal.
- **(Q3) Accept lower bound**: keep retarget, report 70B/Qwen OPT_Ω as loose lower bounds,
  emphasize M_i/recovery scaling only. Cheapest, weakest.

Overnight: 70B_v2 + Qwen_v2 retargets STILL WORTH RUNNING (valid M_i for recovery + valid Q1
shared-8B-Ω value). Do NOT run a Δ(E) scaling cert across mixed native/cross tiers.

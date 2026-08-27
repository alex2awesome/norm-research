# Overnight handoff — 2026-07-02 (creative-writing §12.6 aligned run)

## What ran (all on the corrected `_v2` dirs; v1 `aligned_*_orbit` had WRONG M_i — level-dispatch bug, fixed)
Retarget ladder collected (`--retarget-mi-only`: reuse sigs + orbit M_i):

| dir | ckpts | sigs source | M_i |
|---|---|---|---|
| aligned_8b_orbit_v2 | 46 | 8B (own) | 8B |
| aligned_3b_orbit_v2 | 41 | 3B (own) | 3B |
| aligned_70b_orbit_v2 | 54 | 8B (shared) | 70B |
| aligned_qwen_orbit_v2 | 57 | 8B (shared) | Qwen |

Common 4-executor overlap = 41 metrics (capped by 3B). 8B/3B are **native** (own sigs);
70B/Qwen are **shared-8B-Ω** (8B sigs reused). 70B & Qwen both finished all metrics then the
engine died post-completion (cosmetic; all data saved). Full rescore of 70B/Qwen was infeasible
(~775-criteria pool → ~6 days/executor).

## ⚠️ Finding 1 — shared-Ω retarget OPT_Ω is CONFOUNDED (measured)
`validate_retarget_overlap.py` on 8b_v2/3b_v2 (both native → clean test). Head-Jaccard ~0 is
NOT decisive (redundant pool). **OPT_Ω(bits) is decisive: sigs-source effect ≈ target effect
(D/C=0.96); cross-executor OPT_Ω ≈0.4× native.** Signatures are executor-idiosyncratic.
⇒ **Do NOT run a Δ(E) scaling cert across mixed native(8B/3B)+retarget(70B/Qwen) tiers.**
M_i (recovery target) is valid for ALL 4 (own verdict). Details: `2026-07-02__shared-omega-retarget-validity-review.md`.

## Finding 2 — Q1 trichotomy at 2 NATIVE tiers (valid; `q1_trichotomy_2tier_native.txt`)
| tier | CODIFIABLE | UNDERSAMPLED | FORM-DOMINATED |
|---|---|---|---|
| 3B (41) | 8 | 33 | 0 |
| 8B (46) | 2 | 5 | 39 |

Strong executor-dependence: 3B→8B shifts UNDERSAMPLED→FORM-DOMINATED (8B criteria are
form-fragile, ~12% flip; 3B form-stable but undersampled). "Codifiable" set is executor-dependent
(8@3B vs 2@8B). Scaling table shows mixed 3B↔8B OPT_Ω (no clean capability monotone).

## Morning decision fork (yours)
- **Q1 — consistent shared-8B-Ω ladder**: all 4 tiers use 8B sigs (recompute 3B from llama8b_glm,
  cheap). Answers "8B-process reach per executor"; peaks at 8B, not a capability monotone.
- **Q2 — native per-executor ceilings**: 70B/Qwen own sigs. Full rescore infeasible; **Option B
  (head-only, top ~120 criteria with 70B/Qwen, ~25-40 hr)** is the feasible approximation. The
  measured confound now JUSTIFIES Option B (review dismissed it before this data).
- **Q3 — accept lower bound**: keep retarget, report 70B/Qwen OPT_Ω loose, emphasize M_i/recovery.

## State
- GPUs 1 & 3 free (left idle — further GPU work pending your decision).
- Monitor stopped (data collection complete).
- 16fill done (llama8b_glm=57). Follow-up to full 57 for all executors deferred (3B-extension is Q1/Q2-dependent).

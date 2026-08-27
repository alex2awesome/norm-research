# math — unified math task portfolio

Unified 2026-06-10 (previously `datasets/math-aops/` and `datasets/math-stackexchange/`;
sk3 mirrors moved the same day with back-compat symlinks at the old paths).

The math arm of the V/A/T project has three sub-directions, each probing a
different *kind* of community judgment about mathematical work:

| Subdir | Source | Judgment captured | Label | Status |
|---|---|---|---|---|
| `stackexchange/` | Math.SE dump | community answer quality (clarity-heavy) | accepted ∧ score≥3 vs score≤0 (v3.3 propensity-balanced, 99,722 rows) | **VAT CLOSED** (V/A/C below) |
| `aops/` | AoPS forum | competition-math solution preference | primary: same-approach-as-editorial; secondary: thanks/nothanks | **VAT CLOSED** (V/A/C below) |
| `mathlib/` | leanprover-community/mathlib4 PRs | formal-library review norms | accept (merged) vs reject (closed-unmerged) | **VAT CLOSED** + MathlibPR replication (below) |
| `combined/` (sk3 only) | Math.SE + ProofBench + IMO-GradingBench | expert-graded proof quality | per-source | built 2026-06-02: 848 + 1,000 + 98K rows, `datasets/math/combined/combined.parquet` (no AoPS in it) |
| `scripts/` | — | — | — | `build_combined_math_dataset.py` (salvaged from agent worktree 2026-06-10) |

## ★ VAT RESULTS (all three legs closed) — see `VAT_CLOSURE.md` for the full audited writeup

Canonical numbers (June 2026, independently adversarially re-verified to 3–4 digits).
**V** = deterministic/verifiable features; **A** = LLM-judged articulable norms; **C** = dense
reward-model ceiling (Llama-3.1-8B LoRA). AUC vs chance 0.5.

| Leg | Label | base | **V** (deterministic) | **A** (LLM-judge) | **C** (dense) | Shape |
|---|---|---|---|---|---|---|
| **Math.SE** | answer quality (accepted ∧ score≥3) | 0.50 (floor 0.461) | **0.55–0.58** | **0.66** (+0.10–0.12 over V) | **0.79** (+0.14–0.16 over A, same rows) | big tacit band |
| **AoPS** | same-approach-as-editorial (PRIMARY, ungated) | 0.70 | **0.64–0.71** (lexicons) | **~0.73** (vocab ceiling) | **0.777** (+0.05 over TF-IDF) | thin |
| **Mathlib** | accept (merged) vs reject (closed) | 0.94 | **0.68** (V′, incl. tactic-idiom) | ~0.46–0.56 (redundant, ≤V) | **0.736** (author-stripped TF-IDF) | flat |

**Cross-leg thesis (one line):** as a domain's correctness norms harden (Math.SE → AoPS →
mathlib), V saturates and the recoverable-preference signal migrates from a **large tacit band**
(Math.SE, C−A ≈ +0.15 — dense reads math the judge can't articulate), to a **thin lexical one**
(AoPS, C−A ≈ +0.05 — the label is approach-fingerprint vocabulary), to **flat** (mathlib,
C−V ≈ fully mechanizable, A buys nothing). Math.SE's 0.46 < 0.55 < 0.66 < 0.79 is the cleanest
single illustration of the V/A/T decomposition.

**Per-leg notes:**
- **Math.SE** — the sharpest V-vs-A head-to-head. A's +0.12 gap over V is NOT a weak-V artifact
  (a 30K-feature TF-IDF V plateaus ~0.59); mechanical lexicon-twins of the judged norms are NULL
  (judged "honesty" ≠ hedge-count) → the A-gain is genuine judgment. C−A ≈ +0.14 on identical
  rows = a large fully-tacit/semantic band. Scripts: `stackexchange/mathse_vavc2.py`;
  A-verdicts `a_metric_verdicts_clean.jsonl`; V-features `mathse_lint_features.csv`.
- **AoPS** — report **0.73 full-pool** (ungated), NOT the gated 0.59. The 0.73 is mostly a
  *solution-completeness* gradient (serious solution vs fragment/wrong); restricting to
  verified-correct solutions drops *which-approach* predictability to ~0.59 (a caveat, not the
  headline). Secondary thanks/preference label: **elegance** is the first live A-signal
  (within-problem ρ +0.066, p≈5e-11, length- and correctness-independent); approach-match/novelty
  null (community rewards execution quality, not the route). Dir `aops/` (notes
  `approach_judge_results.md`, `thanks_deconfounded.md`); dense `runs/aops_same_approach_dense_llama8b/`.
- **Mathlib** — the flat / max-verifiability leg. V′ (code metrics + per-tactic idiom counts:
  grind/aesop/simp/simpa/refine/rwa→accept; intro/apply/have/unfold→reject) reaches 0.68;
  dense C=0.736; **A is redundant/below V** (a 122B judge over-thinks a mechanical decision).
  Canonical slice `mathlib/accept_reject_clean_deconf.parquet` (n=7,956, base 0.943; author-text
  stripped, topic-residualized). See `mathlib/README.md`.

### MathlibPR replication (2026-06-29/30) — full V/A/retrieval decomposition on an external benchmark

Ran our decomposition on the public **MathlibPR** benchmark (Xie/Liu/Zhang, UVA, arXiv:2605.07147;
15,895 build-passing PR snapshots, 12,063 PRs). Their finding: LLM models/agents can't beat chance
at merge-readiness — we explain why. Diffs reconstructed via git SHAs in `mathlib/mathlib4_repo`.

| Arm | Our data | MathlibPR |
|---|---|---|
| V″ (deterministic) | 0.68 | **0.604** |
| Retrieval (duplication) incremental | +0.009 | **+0.015 (null)** |
| Style linters (8 mechanized) incremental | −0.007 | **−0.002 (null)** |
| A-judge (qwen-122B, 7 norms) incremental | ~null | **−0.009 (null)** |

**All three arms null on their build-passing negatives** → mathlib merge-readiness isn't legible
from the static diff (confirms the flat leg on independent, larger, harder-negative data). **One
positive:** the within-PR *pairwise* task (their cleanest, where their LLMs are at chance) — V″
recovers the **revision direction** at AUROC **0.661** (final snapshots shrink + swap manual
tactics→automation; `tac_simpa` best). *"Being merge-ready" isn't diff-legible, but "becoming
merge-ready" is.* Artifacts: `mathlib/mathlibpr/` (see memory `project_mathlibpr_replication_2026_06_29`).

**V″/V′ note:** the mathlib 0.68 appears as **V′** in VAT_CLOSURE (V + tactic-idiom, topic-resid)
and as **V″** in the MathlibPR work (V + tactic counts, GroupKFold-by-PR) — same feature family,
same ~0.68 value; the primes just track which confound-control revision.

### Top metrics per leg (single-feature AUC vs the leg's label)

These are the individual best-discriminating metrics, computed on held-out data. (`A` = LLM-judged
1–5; `V` = deterministic checker.) Full per-metric definitions in each leg's `METRICS.md`.
Live numbers regenerated by `notebooks/2026-07-01__math-vat-summary.ipynb` from
`notebooks/data/math_vat_summary.json`.

**Math.SE — top A metrics** (the articulable layer; all ≫ any single V):

| id | metric | AUC |
|---|---|---|
| a07 | elegance markers (one key idea, minimal machinery, aha) | **0.656** |
| a11 | directness (actually answers what was asked) | 0.646 |
| a03 | right generality (fit to the question, not abstraction per se) | 0.643 |
| a02 | audience calibration | 0.642 |
| a06 | proof idea visible (reader learns *why*) | 0.639 |
| a08 | precision / rigor | 0.637 |

**Math.SE — top V metrics** (all near chance — deterministic checkers barely separate *preference*):

| metric | AUC |
|---|---|
| n_steps_total (derivation length) | 0.532 |
| max_jaccard_to_sibling (near-dup of a sibling answer) | 0.522 |
| frac_steps_verified (sympy step-chain) | 0.516 |

→ the A>V gap is the whole story: judged **elegance / directness / rigor** (0.64–0.66) sit far above
the best sympy/lint checker (~0.53). The 14 A-metrics are collinear (PC1=79% var) → read as **one
judged quality axis**, elegance-led.

**Mathlib — top V′ metrics** (the flat leg; tactic-idiom + size are the whole signal, A adds nothing):

| metric | AUC | direction |
|---|---|---|
| net / added / churn (size) | 0.61–0.62 | bigger → reject |
| n_comment, n_import | 0.60–0.62 | — |
| n_docstring, doc_ratio | 0.58–0.59 | documented → accept |
| tactic-idiom profile (per-tactic) | 0.61 (combined) | grind/simp/simpa/refine→accept; intro/apply/unfold→reject |

→ no A metric beats V here (A ≈ 0.46–0.56, redundant). See `mathlib/METRICS.md` for the full V/A bank.

**AoPS — V ladder** (thin/lexical; the vocabulary fingerprint already gets most of it):

| feature set | AUC |
|---|---|
| V-struct | 0.544 |
| V-answer | 0.602 |
| V combined | 0.636 |
| register | 0.653 |
| post-only vocabulary (approach fingerprint) | **0.727** |
| everything | 0.737 |

→ A adds only the *elegance* signal on the secondary thanks/preference label (within-problem ρ +0.066,
p≈5e-11); approach-match/novelty null (community rewards execution, not the route).

## Why three sub-directions

Per `project_math_elegance_research.md`, mathematical quality decomposes into
elegance / profundity / clarity / precision, and different communities weight
them differently:

- **Math.SE** upvotes are suspected to track *clarity of exposition* most.
  **CONFIRMED:** the recoverable signal is dominated by a large tacit/semantic band
  (dense C 0.79 ≫ judge A 0.66 ≫ code V 0.55).
- **AoPS** thanks, on competition problems with many alternative solutions,
  should track *elegance* better (multiple writeups of the same proof idea).
  **CONFIRMED (partial):** elegance is the first live A-signal on the thanks/preference
  label (ρ +0.066), but the primary same-approach label is thin/lexical (C−A ≈ +0.05).
- **mathlib** review is the *precision + style-norm* extreme: correctness is
  machine-checked (Lean type-checker), so everything reviewers argue about is
  by construction articulable-or-taste. This makes it the project's cleanest
  high-V anchor. **CONFIRMED:** the flat leg — V′ 0.68 ≈ dense C 0.74, A redundant;
  no tacit residue (contrast Math.SE). The static diff is nearly all mechanizable signal.

## V/A/T positioning

Math is the hypothesized **high-V, low-Taste anchor** of the cross-task
comparison (`project_verifiability_explainability_gaps.md`): correctness is
formally checkable (max V), style norms are heavily documented
(`stackexchange/online-rubrics/`, 122 Claude-parsed expert essays: Halmos,
Gowers, Hardy, Aigner–Ziegler, the Mathlib style guide, …), and the residual
taste component should be smaller than creative writing or humor.

## sk3 paths

```
/lfs/skampere3/0/alexspan/norm-research/datasets/math/
├── stackexchange/    (old symlink: datasets/math-stackexchange)
├── mathlib/          (old symlink: datasets/lean_mathlib)
└── combined/         (old symlink: datasets/combined_math)
```

AoPS raw crawl shards live separately at `/lfs/skampere3/0/alexspan/aops/raw/shards/`.

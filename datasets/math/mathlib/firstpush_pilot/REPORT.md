# First-push reconstruction + cheap V-metric pilot

2026-06-10. Script: `datasets/math/mathlib/firstpush_pilot.py`; metrics:
`datasets/math/mathlib/firstpush_pilot_metrics.csv` (local + sk3). Sample:
200 PRs (100 per class), `split=train` of `friction_dataset_v2.csv.gz`,
seed 42. All git work on sk3 against `mathlib4_repo/` (full clone,
`origin/master` @ 2026-06-10), CPU only.

## 1. Reconstruction feasibility (the infrastructure question)

| step | result |
|---|---|
| `refs/pull/N/head` fetch (batched, 50/call) | **200/200 (100%)** — no GC'd refs in sample |
| first_commit_oid reachable after fetch | **199/200 (99.5%)** → `state_used = first_commit` |
| head_fallback needed | 0 |
| diff failures | 1 (PR 32673: first commit is an **orphan branch**, no merge-base with master) |
| PRs with 0 added `.lean` lines | 30/199 (15%; 19 y=1 / 11 y=0 — docs/CI/deletion-only PRs) |
| wall time | ~7 min total (4 fetch batches + 200 merge-base/diff/metric passes) |

The first-push reconstruction pipeline is fully validated: extrapolating,
~99% of the 16,884 PRs can be reconstructed at the first-commit state with
no API calls beyond `git fetch`.

## 2. Per-metric discrimination vs y (y=1 = frictionless merge)

Rates per 100 added lines unless noted; n=199 usable rows.

| feature | r | p | AUC |
|---|---|---|---|
| v02 frac lines >100 chars | -0.130 | 0.066 | 0.463 |
| v04 `λ`/`fun =>` lambdas | -0.027 | 0.702 | 0.461 |
| v05 `erw` uses | -0.079 | 0.269 | 0.495 |
| v06 non-terminal `simp` | -0.052 | 0.468 | 0.479 |
| v07 unsqueezed `simp` | +0.049 | 0.489 | 0.458 |
| v10 frac decls missing docstring | -0.122 | 0.087 | 0.440 |
| v13 focusing dots `·` | -0.041 | 0.569 | 0.471 |
| v15 mean proof-block length | -0.029 | 0.685 | 0.449 |
| v15 max proof-block length | +0.004 | 0.956 | 0.456 |
| v15 tactic-mode ratio | -0.013 | 0.854 | 0.497 |
| v16 new declarations | -0.010 | 0.888 | 0.448 |
| v17 `@[deprecated]` w/o date | -0.071 | 0.316 | 0.495 |
| log(1 + added lines) | -0.123 | 0.083 | 0.427 |

**Combined logistic regression, 5-fold stratified CV: AUC 0.518 ± 0.074**
(folds 0.443–0.657).

Reading: no single metric is significant at 0.05; the three closest
(long lines p=.066, missing docstrings p=.087, first-push size p=.083) all
point the expected direction (more violations / bigger first push → more
review friction) but are tiny. Combined model is chance.

Prevalence check (frac of PRs with any hit): missing-docstring 59%,
unsqueezed simp 28%, lambda 27%, focus dots 27%, non-terminal simp 16%,
long lines 9%, `erw` 2%, deprecated-no-date 1%. So the nulls for
v05/v17 are partly floor effects (almost nobody violates them even at
first push); v10/v07/v04 have real variance and *still* don't discriminate.

## 3. Interpretation: this null CONFIRMS the V-floor hypothesis

README §5 ("Where V discrimination actually lives") and §6 predict exactly
this: mathlib has spent a decade thinning its thick rules into linters and
CI, so by the time a PR is *visible to reviewers* the cheap style dimension
is already near-compliant — authors run the linters locally, and the few
residual violations get fixed mechanically without generating review
threads. Residual review friction is therefore about **content** (API
duplication, generality, naming semantics, placement — the A metrics), not
text-rule style. A strong signal here would have *contradicted* the
project's thin-rule story for mathlib; AUC 0.518 supports it.

## 4. Caveats (honest list)

- **n=200**: powered only to detect single-feature AUC ≳ 0.60-0.62. We can
  rule out "cheap style carries the label," not "cheap style contributes
  0.52-0.55 of marginal signal."
- **first_commit_oid is post-force-push**: GraphQL `commits(first:1)`
  reflects the final branch state. For force-pushed PRs the "first push" we
  reconstruct may already be partially review-polished → attenuation toward
  null. True first-push recovery needs timeline `HeadRefForcePushedEvent`
  before-SHAs (README §4.3).
- **Text-rule approximations** (documented in the script docstring):
  non-terminal `simp` judged by next-added-line column within a diff run;
  proof blocks truncate at hunk boundaries; docstrings present only as diff
  context are invisible (overcounts "missing"); `@[...]` attributes are
  stripped so `@[simp]` is not counted as a tactic. Validated by hand
  against PR 6803's full diff (all counts matched).
- **Added-lines-only view**: pure refactor/deletion PRs and the 15%
  zero-`.lean`-line PRs contribute empty feature vectors.
- The balanced dataset already matches size×prefix×year×assoc×topic cells,
  so any style signal that rides on those strata is removed by construction
  — that is intended (it is the residual label), but it means these AUCs
  are within-cell residual discrimination, the hard version of the task.

## 5. Go / no-go for scaling to all 16,884

**GO on the reconstruction, NO-GO on expecting these metrics to predict.**

1. **Scale the first-push reconstruction**: validated at 100% fetch /
   99.5% first-commit; it is required infrastructure regardless of this
   null — for the §3b revision pairs (first-push vs merged deltas, the
   A-track gold) and for v18 (running mathlib's real `lint-style` on
   reconstructed trees). Extrapolated cost: ~340 fetch batches + 16,884
   diffs, a few hours single-threaded on sk3, no GPU, no API quota.
2. **Do scale the cheap text metrics alongside it** — they are nearly free
   once the diff is in hand — but as the *documented V leg* of the V/A/T
   decomposition, not as a feature bank: at n≈16,884 the CI on the combined
   AUC will be ±~0.01, which is what README §6's "smallest C−B gap" claim
   needs as its V baseline.
3. **Redirect discriminative effort to A metrics** (a01 duplicate-API
   retrieval first) and to first-push→merged *deltas*, where §5 says the
   discrimination should actually live.

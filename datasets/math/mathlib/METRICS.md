# mathlib metric inventory (v1, 2026-06-10)

Selected from `datasets/math/stackexchange/online-rubrics/claude-parsed/lean_mathlib_style_guide.md`
(the parsed Mathlib style guide) plus the review-norm categories observed in PR threads.
Each metric: V (code-checkable — tool listed) or A (LLM-judge). V metrics are computed on
the **first-push state** (reconstructed from `refs/pull/N/head` + force-push events via
`mathlib4_repo/`); the merged state is review-polished and saturates them by construction.

Label context (see README §3a + label-mechanics findings): y = review friction among
merged PRs, operationalized as zero-review-threads (62/38 split). Stratify/bin-match on:
size quartile, title-convention prefix (feat 0.546 zero-thread vs chore 0.761 / fix
0.775), author association, year.

## V metrics (mechanical, first-push state)

| id | metric | checker |
|---|---|---|
| v01 | builds / type-checks | `lake build` on touched files (expensive; sample) |
| v02 | line length ≤ 100 | text scan |
| v03 | two-space indent / operator-at-EOL / `by` placement | text rules |
| v04 | `fun ↦` not `λ`/`=>` | grep |
| v05 | no `erw` | grep (style guide: signals missing API) |
| v06 | no non-terminal `simp` | tactic-block parse |
| v07 | unsqueezed terminal `simp` (no `simp only` squeezing w/o perf reason) | tactic parse |
| v08 | naming grammar: `A_of_B_of_C` tokens consistent with statement shape | name tokenizer vs statement AST (Lean parser or regex approx) |
| v09 | case conventions: snake_case props / UpperCamelCase types / lowerCamelCase terms | decl parse |
| v10 | docstring present on new `def`s | decl parse (mathlib linter exists) |
| v11 | hypotheses left of colon | statement parse |
| v12 | calc block alignment | text rules |
| v13 | focusing dots `·` for subgoals | tactic parse |
| v14 | import minimality (no unused imports) | `mk_all` / lake script |
| v15 | proof length (lines) & term-vs-tactic mode | trivial |
| v16 | n declarations touched / new lemmas added | diff parse |
| v17 | deprecation hygiene (`@[deprecated]` with date) | grep |
| v18 | first-push lint-clean rate (run `scripts/lint-style` wholesale) | mathlib's own linter |

## A metrics (LLM-judge, statement + proof ± retrieved library context)

| id | metric | judge input |
|---|---|---|
| a01 | duplicates existing API ("this is `Foo.bar`") | decl + top-k embedding-retrieved mathlib decls |
| a02 | stated at right typeclass generality | decl + ambient instances used in proof |
| a03 | missing-API smell (workarounds where a simp lemma should exist) | proof body |
| a04 | name reads as idiomatic English of the statement | name + statement |
| a05 | right file/module placement | decl + file context |
| a06 | proof idiom quality (golfing appropriate, no detours) | proof body |
| a07 | statement usefulness as API (would downstream users call this?) | decl + docstring |
| a08 | docstring quality (says what, why, not how) | docstring + decl |

## Notes

- v18 subsumes v02-v07/v10 wherever mathlib's linter covers them — run the linter first,
  use the granular rules only where we need per-rule attribution.
- a01 needs the embedding index over mathlib decl docstrings (build once from
  `mathlib4_repo/`); it is the most common substantive review-comment class and the
  highest-value A metric.
- The A-side ground truth corpus = review-thread comments from the running fetch
  (label 3b in README): norm-extract them and check which a-metrics actually appear
  in real reviews (silver alignment, per `methods/metric_implementer/` scorecard item 5).

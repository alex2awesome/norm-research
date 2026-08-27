---
source_url: https://leanprover-community.github.io/contribute/style.html
title: Mathlib Library Style Guide (Lean Theorem Prover)
source_type: professional_org
fetched: 2026-05-09
---

# Mathlib Library Style Guidelines — Lean Theorem Prover Community

Style guide for the largest formal mathematics library in the world. The conventions encode the community's values for what counts as a "good formalization": readable, maintainable, idiomatic.

## Variable conventions

- `u`, `v`, `w` — universes
- `α`, `β`, `γ` — generic types
- `a`, `b`, `c` — propositions (or arbitrary terms)
- `x`, `y`, `z` — elements of generic types
- `h`, `h₁`, `h₂` — assumptions / hypotheses
- `p`, `q`, `r` — predicates and relations
- `s`, `t` — lists and sets
- `m`, `n`, `k` — natural numbers
- `i`, `j`, `k` — integers
- Uppercase (`G`, `R`, `K`, `𝕜`, `E`) — types with mathematical content (groups, rings, fields, vector spaces).

## Layout

- **Line length** ≤ 100 characters.
- **Indent** 2 spaces.
- **Spaces** around `:`, `:=`, infix operators.
- Operators at end of line, not start.
- Multi-line theorem statement indented 4 spaces; proof indented 2.
- `by` at end of preceding line, never alone.

## Naming

Theorem named `A_of_B_of_C` proves a conclusion of form `A` from hypotheses of form `B`, `C`. Read as English: `add_le` for additive ≤, `mul_pos` for products positive, etc.

- Props/proofs: `snake_case`.
- Types/structures: `UpperCamelCase`.
- Other terms: `lowerCamelCase`.
- Class names that are nouns begin with `Is` (`IsNormal`); adjectival classes can omit it.

## Hypothesis placement

Prefer arguments to the left of the colon:
```
example (n : ℝ) (h : 1 < n) : 0 < n := ...
```
Not:
```
example (n : ℝ) : 1 < n → 0 < n := ...
```

## Tactic style

- Subgoals introduced with focusing dot `·` (not indented).
- One tactic per line; short sequences may use `;`.
- Don't squeeze a terminal `simp` unless for performance — unsqueezed calls survive lemma renames.
- Avoid `erw` and extra `rfl` — they signal missing API.
- Use `case` for named subgoals.

## `have`/`show` proof style

- Short have: `have h1 : P := proof` on one line.
- Long: place proof on next line indented 2.
- With tactic: `have h : P := by tac`.

## Calc blocks

```
calc a = b := h1
  _ ≤ c := h2
  _ < d := h3
```
Align relations across lines. Underscores left-justified.

## Anonymous functions

- Use `fun ... ↦` (the `\mapsto` arrow), not `λ`, not `=>`.
- Use `·` for very simple functions: `(· ^ 2)`.

## Normal forms

For statements that can be written multiple equivalent ways, mathlib picks one and standardizes. Examples:
- In types with bottom: `x ≠ ⊥` in hypotheses, `⊥ < x` in conclusions.
- In types with top: `x ≠ ⊤` in hypotheses, `x < ⊤` in conclusions.

## Transparency

- Default `semireducible`.
- Use type synonyms over `irreducible` for API boundaries.
- Use `irreducible_def` only with documented justification.

## Module docstrings

Every file starts with a docstring containing:
- Title.
- Contents summary.
- Main definitions and theorems.
- Proof techniques used.
- Notation introduced.
- Literature references.

## Performance

- All PRs benchmarked.
- Negative regressions must be explained or fixed.

## Deprecation

- `@[deprecated]` attribute on removed public names.
- Aliases retained for 6 months after deprecation.

## Why the style guide is a "good math" rubric

Implicitly defines a quality formalization as one that is:
1. **Readable** by someone other than the author.
2. **Idiomatically named** so library users can guess the name.
3. **Performant** so it doesn't slow library compilation.
4. **Maintainable** under future refactors.
5. **Consistent** with neighboring mathlib conventions.
6. **Documented** at module and declaration level.
7. **API-shaped** so others can build on it.

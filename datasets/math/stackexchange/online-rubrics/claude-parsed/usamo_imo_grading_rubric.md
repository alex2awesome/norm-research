---
source_url: https://web.evanchen.cc/upload/usamo-2003-rubric.pdf
title: USAMO / IMO Grading Rubric (Olympiad Solution Scoring)
source_type: rubric
fetched: 2026-05-09
---

# USAMO / IMO Olympiad Grading Rubric

The standard 0–7 scoring scale used at USAMO, IMO, and most national olympiads. The most influential rubric for grading single mathematical proofs in the world.

## The 0–7 scale

| Score | Meaning |
|---|---|
| 7 | Complete, correct solution. Minor presentational issues only. |
| 6 | Essentially complete; small gap easily filled by the grader. |
| 5 | Major progress; correct main idea but one substantive step missing or hand-waved. |
| 4 | Substantial progress; correct overall framework but key step incomplete. |
| 3 | Several useful observations but no path to a complete proof. |
| 2 | A few correct ideas but missing the central insight. |
| 1 | One non-trivial observation. |
| 0 | Nothing of value, or contains a fatal incorrect claim. |

## The "two-track" grading principle

Each solution is approached from one of two directions:

- **From 7 going down** — assume a complete solution; deduct for errors. Used for solutions that look essentially correct.
- **From 0 going up** — assume nothing; add for ideas. Used for solutions that lack a key insight.

Crucially, **most middle-range scores (3-5) are not awarded by combining "partial credit" linearly**. A solution either has the key idea (track 7→) or doesn't (track 0→). On hard problems (3, 6) the middle scores are extremely rare.

## The "Gap of Death" (Putnam analogue)

Putnam uses 0, 1, 2, 8, 9, 10 — the middle scores 3–7 are essentially never awarded. Same philosophy: an olympiad problem either has been solved (>=8) or has not (<=2). USAMO's middle range is wider but follows the same logic.

## Specific scoring guidance (Evan Chen's compiled USAMO 2003 rubric and others)

### Deductions from a 7
- $-1$ for an algebra/arithmetic slip that doesn't affect the argument.
- $-1$ for missing one easy case.
- $-1$ for unproved "WLOG" that has a non-trivial reduction.
- $-2$ for skipping a step that genuinely requires work.
- $0$ if the solution is correct but written illegibly.

### Additions from a 0
- $+1$ for a correct restatement showing understanding of the problem.
- $+1$ for a non-trivial necessary condition derived.
- $+1$ for a worked-out small case providing insight.
- $+2$ for the central lemma but missing the conclusion.

## What graders look for

1. **Logical completeness** — every step justified.
2. **No gaps** — no "obvious" / "clearly" / "WLOG" hiding real work.
3. **Correct quantifier handling** — every variable scoped.
4. **Case exhaustiveness** — disjoint and complete.
5. **The right key insight** — the problem's intended trick or an alternative.
6. **Clarity** — graders should not have to reconstruct.
7. **Verification** — for "find all" problems, both showing the solutions work and showing no others.

## What graders dock

- Skipping the verification step in "find all" problems.
- Asserting WLOG without checking the symmetry holds.
- "It is easy to see" where it isn't.
- Inductive arguments without an explicit base case.
- Continuity / convergence assertions without proof in analysis problems.
- Using a known theorem above the level of the contest without restatement.

## Influence

- Used essentially without modification at every national olympiad: USAMO, BMO, Iran, China, Russia.
- Adopted by IMO grading panel since the 1980s.
- Underlies AoPS partial-credit estimates.
- Cited by every "how to write an olympiad solution" guide.

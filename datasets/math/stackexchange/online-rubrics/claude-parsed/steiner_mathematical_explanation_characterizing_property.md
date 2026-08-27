---
source: Mark Steiner, "Mathematical Explanation" (Philosophical Studies 34, 1978, 135–151) and "Mathematical Knowledge" (Cornell, 1975)
url: http://web.mit.edu/bskow/www/810-S12/steiner_mathematical-explanation.pdf
type: classical_academic
year: 1975-1978
domain: explanatory_proof_account
collected: 2026-05-09
---

# Steiner: Mathematical Explanation via Characterizing Properties

Steiner's framework is the foundational philosophical account of when a proof *explains* (vs. merely verifies). It yields concrete operational tests.

## The Core Criterion: Characterizing Property

A proof is **explanatory** when it:

> "Makes critical use of some characterizing property" — a property unique to the entity in question within its domain.

The proof works *because of* what makes the object what it is, not because of accidental features.

## The Paradigm Example: Gauss's Schoolboy Proof

Gauss reportedly summed $1 + 2 + \cdots + n$ as a schoolboy by:
- Writing the sum forward and backward
- Pairing terms: $(1+n) + (2+(n-1)) + \cdots = n(n+1)/2$

**Why this is explanatory** (Steiner): the proof exploits the **symmetry** of the sum — symmetry is the characterizing property. Any sum with this symmetry would yield to the same proof, with the result varying as the symmetry changes.

**Contrast**: a proof by induction verifies the formula but does not exploit symmetry — it doesn't tell you *why* the formula has this form. Inductive proofs of formulae are often non-explanatory in Steiner's sense.

## The Deformation Test

Steiner's most operational criterion:

> Vary the characterizing property in the proof. The proof should yield a corresponding theorem under the deformation.

If you can change the characterizing property and the proof still works (giving a different theorem about the new property), the original proof was **explanatory**. If the proof breaks under any deformation, it was **accidental** — it works for this case only.

## How to Apply the Deformation Test

Given a proof $P$ of theorem $T$:

1. **Identify the property** $P$ uses critically (the one without which $P$ fails).
2. **Substitute a related property** (e.g., a different symmetry, a different invariant).
3. **Check whether the proof goes through** with corresponding modifications.
4. **If yes**, the proof is explanatory: it has identified what's responsible for $T$.
5. **If no**, the proof verified $T$ accidentally — it doesn't explain.

## Example: Why π Is Irrational

A proof of π's irrationality that exploits **transcendence properties of trigonometric functions** explains: it tells us irrationality follows from a deeper property. A brute-force proof that produces a contradiction without invoking transcendence verifies but doesn't explain.

## Implied Rubric for Explanatoriness

For each proof:

- **Identify the proof's load-bearing property** (Steiner's "characterizing property").
- **Vary that property** — does the proof generalize?
- **If yes**, score high on explanatoriness.
- **If no**, score low — even if the proof is correct, beautiful, and short.

## Steiner's Three Conditions (consolidated)

A proof is explanatory iff:

1. **It depends on a characterizing property** of the entity.
2. **Variations in that property** yield variations in the conclusion.
3. **The proof exhibits these variations** as a natural family.

A proof is non-explanatory if it:

- Uses generic machinery (induction, term-by-term verification, brute-force calculation).
- Cannot be deformed into proofs of related theorems.
- Yields the result without revealing what about the entity caused it.

## Critiques and Refinements

Resnik & Kushner (1987) and others argued that:

- Steiner's distinction may be **context-dependent**: what is "characterizing" depends on what was previously known.
- There are **counterexamples** to Steiner's strict criterion.
- A weaker, **multi-criterion** account (e.g., Lange's bottom-up framework) may be needed.

But the **deformation test** survives as a robust operational tool for evaluating explanatoriness, even where Steiner's stricter criterion fails.

## What Counts as a "Characterizing Property"

Steiner intends:

- **Mathematically intrinsic**: the property is part of the object's essential nature within its domain.
- **Distinguishing**: it picks the object out from others in the domain.
- **Generalizable**: it admits natural variations.

Examples:
- For the symmetry-based sum proof: symmetry is intrinsic to arithmetic-progression sums.
- For the prime-number theorem: distribution properties of primes are intrinsic.
- For the Brouwer fixed-point theorem: topology of the disk is intrinsic.

## Implications for Rubric Design

A rubric for "good proof" should:

- **Score explanatoriness independently** from rigor and beauty.
- Use the **deformation test** as a concrete check.
- **Reward** proofs that suggest variants and generalizations.
- **Distinguish** structural from brute-force arguments.
- **Recognize** that a single theorem's "best" proof depends on what one wants to learn (verification vs. explanation).

## The Deeper Significance

Steiner's framework formalizes the working mathematician's intuition that some proofs are "just verifications" while others "really show why." This intuition is central to mathematical taste; without it, all valid proofs would be equivalent. Steiner gives it a testable structure.

## Connection to Other Frameworks

- **Lange**: bottom-up account of explanation (multiple criteria, no single characterizing property).
- **Kitcher**: unification — orthogonal to Steiner's characterizing-property account.
- **Detlefsen-Arana (purity)**: pure proofs tend to be explanatory because they use only intrinsic properties.
- **Avigad (functions of proof)**: explanation is one of the multiple functions a proof can serve.

Sources:
- [Steiner, Mathematical Explanation (MIT mirror PDF)](http://web.mit.edu/bskow/www/810-S12/steiner_mathematical-explanation.pdf)
- [Steiner, Mathematical Knowledge (Cornell, 1975)](https://philpapers.org/rec/STEMK)
- [Steiner, Mathematical Explanation (PhilPapers)](https://philpapers.org/rec/STEME)
- [Mathematical Knowledge, Objects and Applications (memorial volume)](https://link.springer.com/book/10.1007/978-3-031-21655-8)
- [Functional Explanation in Mathematics (Synthese)](https://link.springer.com/article/10.1007/s11229-019-02234-5)

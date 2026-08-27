---
source_url: https://math.berkeley.edu/~hutching/teach/proofs.pdf
title: Introduction to Mathematical Arguments (Michael Hutchings, Berkeley)
source_type: course_handout
fetched: 2026-05-09
---

# Introduction to Mathematical Arguments — Michael Hutchings (UC Berkeley handout)

A widely-used Berkeley handout codifying the basic vocabulary and grammar of mathematical proof.

## Definition of Proof
"A mathematical proof is an argument which convinces other people that something is true. Math isn't a court of law, so a 'preponderance of the evidence' or 'beyond any reasonable doubt' isn't good enough. In principle we try to prove things beyond any doubt at all."

Caveats:
- People make mistakes; total rigor can be impractical for large projects.
- Foundational subtleties exist (Gödel's theorem, etc.).

## Two Roles of a Good Proof
1. **Convince** other mathematicians the statement is true.
2. **Help them understand** *why* it is true.

## Logical Vocabulary (precise meanings, sometimes differing from English)
- **not** p — true iff p is false.
- **and** — true iff both true.
- **or** — inclusive: true iff either or both true. ("In English, sometimes 'p or q' means... but not both. However, this is *never* the case in mathematics.")
- **if...then** (p ⇒ q) — false only when p is true and q is false; **vacuously true** when p is false.
- **if and only if** (⇔) — equivalence; true iff both have the same truth value.
- **for every** (∀), **there exists** (∃) — order matters: ∀x∃y x<y is true; ∃y∀x x<y is false.
- **definitions use "if" to mean "iff"** (a known abuse).

## Logic in a Nutshell (Table 1 — proof techniques)

| Statement | Ways to Prove | Ways to Use | How to Negate |
|---|---|---|---|
| p | Prove directly; or assume ¬p, contradict | p is true; if false, contradict | not p |
| p and q | Prove p; prove q | both true | (not p) or (not q) |
| p or q | Assume ¬p, deduce q; or vice versa; or prove p; or prove q | If p⇒r and q⇒r, then r | (not p) and (not q) |
| p ⇒ q | Assume p, deduce q; or assume ¬q, deduce ¬p (contrapositive) | If p, then q; if ¬q, then ¬p | p and (not q) |
| p ⇔ q | Prove p⇒q then q⇒p; or prove p∧q; or prove ¬p∧¬q | interchangeable | (p ∧ ¬q) or (¬p ∧ q) |
| ∃x∈S P(x) | Find an x in S with P(x) | "Let x ∈ S with P(x)" | ∀x∈S not P(x) |
| ∀x∈S P(x) | "Let x be any element of S"; prove P(x) | If x ∈ S, P(x) | ∃x∈S not P(x) |

## Implication for "Good Math Writing"
Hutchings's table is a working *machine* for proof writing: identify the statement type, look up the prove/use/negate column. A proof is "good" partly because each step matches an entry in this table — making each step justifiable.

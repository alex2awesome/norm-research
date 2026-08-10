---
source_url: https://users.metu.edu.tr/serge/courses/111-2011/textbook-math111.pdf
title: How to Prove It — A Structured Approach (Daniel J. Velleman)
source_type: textbook
fetched: 2026-05-09
---

# How to Prove It: A Structured Approach — Daniel J. Velleman (1994, 3rd ed. 2019)

Cambridge Univ. Press. Standard introduction to proofs in many U.S. universities. Velleman's distinctive contribution: a system of *templates* indexed by the logical form of the statement to be proved. Each template tells you exactly how to start, what to assume, and what to conclude.

## Velleman's proof templates

### To prove $\forall x. P(x)$ (universal)
- **Template:** "Let $x$ be arbitrary. [Argue $P(x)$.] Since $x$ was arbitrary, $\forall x. P(x)$."
- Choose the variable name to avoid clash.
- Don't make any assumption about $x$ beyond its type.

### To prove $\exists x. P(x)$ (existential)
- **Template:** "Let $x = $ [the witness you have in mind]. We show $P(x)$. [Verify.]"
- Often you find the witness by working backward.

### To prove $P \Rightarrow Q$ (implication)
- **Template:** "Suppose $P$. [Argue $Q$.] Since $Q$ followed from $P$, the implication holds."

### To prove $P \Leftrightarrow Q$ (biconditional)
- **Template:** Two separate proofs. "First, suppose $P$ … so $Q$. Conversely, suppose $Q$ … so $P$."

### To prove $P \wedge Q$ (conjunction)
- **Template:** "We prove $P$ and $Q$ separately. (a) [Prove $P$.] (b) [Prove $Q$.]"

### To prove $P \vee Q$ (disjunction)
- **Template (case split):** "Either $R$ or not-$R$. Case 1: $R$ … so $P$. Case 2: not-$R$ … so $Q$. In either case, $P \vee Q$."
- Or: assume not-$P$, derive $Q$.

### To prove $\neg P$ (negation)
- **Template:** "Suppose, for contradiction, $P$. [Derive an absurdity.] Therefore $\neg P$."

### To use $\forall x. P(x)$ (universal as hypothesis)
- "Apply the hypothesis to the specific $x = c$. Then $P(c)$ holds."

### To use $\exists x. P(x)$ (existential as hypothesis)
- "Choose $x_0$ such that $P(x_0)$. [Treat $x_0$ as a known new constant.]"

## Velleman's emphasis on structure

The proof should *visibly mirror* its logical form:
- Indentation tracks logical depth.
- Each "Suppose" / "Let arbitrary" opens a scope.
- Every scope must close with the conclusion that depends on it.

## Velleman's checklist

1. Does the proof start with the right setup for the statement's logical form?
2. Are quantifier scopes explicit?
3. Are assumptions discharged at the end of their scope?
4. Are case splits exhaustive and disjoint?
5. Are existential witnesses produced before being used?
6. Does the structure of the proof match the structure of the statement?

## Influence

- Required text in dozens of U.S. transition-to-proof courses.
- Models the logical-template approach now widespread in proof assistants (Lean's `intro`, `obtain`, `rcases` mirror Velleman's templates).
- Companion to Hammack's *Book of Proof* and Sundstrom's *Mathematical Reasoning: Writing and Proof*.

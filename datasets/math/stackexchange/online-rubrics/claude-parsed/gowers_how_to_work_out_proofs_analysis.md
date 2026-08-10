---
source_url: https://gowers.wordpress.com/2014/02/03/how-to-work-out-proofs-in-analysis-i/
title: How to Work Out Proofs in Analysis I (Gowers's Weblog)
source_type: mathematician_blog
fetched: 2026-05-09
---

# How to Work Out Proofs in Analysis I — Tim Gowers (2014)

A long blog post articulating a small set of procedural "moves" that compose almost every undergraduate analysis proof.

## Core Philosophy
Analysis I proofs are "easy" in the technical sense that they follow predictable patterns rather than requiring creative leaps. The right goal: understanding over memorisation. "You just keep doing the obvious thing except that from time to time the next step isn't obvious."

## The Seven Moves

1. **The "Let" Move.** To prove "for all x, P(x)", write "Let x be arbitrary" and shift the goal to P(x).
2. **The "Naming" Move.** When told something exists, name it. (E.g. given a convergent subsequence, name its limit a.)
3. **Expansion.** When high-level arguments don't work, unpack definitions into their formal, quantifier-laden form (replace "convergent" with the ε–N statement).
4. **Substitution into Hypotheses.** Given ∀u P(u), substitute any object x to obtain P(x).
5. **Modus Ponens.** Given ∀u P(u)⟹Q(u) and P(x), conclude Q(x).
6. **Substitution into Targets.** For ∃u P(u), pick a promising candidate x and redirect to proving P(x).
7. **Triangle Inequality (and similar closure moves).** Combine inequalities strategically — e.g. ε/2 splits to bridge intermediate steps.

## Three Hallmarks of a Manageable Proof
1. Steps follow logically from expanded definitions without introducing novel mathematical objects.
2. The reasoning remains "non-silly" — recognisably sensible.
3. Most moves are procedural; only isolated steps demand genuine insight.

## Reader's Comparative Advantage
Humans beat naive algorithms by:
- Understanding English-language phrasing of statements.
- Recognising obvious simplifications.
- Judging reasonableness of intermediate goals.

## Implication for "Good Proof"
A good proof in this register exposes its move-sequence so the reader can recognise each step as one of the standard moves, rather than appearing as inspired magic.

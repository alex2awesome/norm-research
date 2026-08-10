---
source: Leslie Lamport, "How to Write a 21st Century Proof" (J. Fixed Point Theory Appl. 11, 2012, 43–63)
url: https://lamport.azurewebsites.net/pubs/proof.pdf
type: modern_academic_cs
year: 2012
domain: structured_proof
collected: 2026-05-09
---

# Lamport: How to Write a 21st Century Proof — The Complete Framework

NOTE: An existing file `lamport_how_to_write_a_proof.md` covers the 1995 paper. This file is the 2012 *21st Century* paper, which is a substantial expansion with new TLA+ proof language elements.

## Core Thesis

> "Writing non-trivial prose proofs now seems as archaic to me as outdated mathematical notation."

Lamport argues that **prose proofs systematically hide errors**. The remedy is **hierarchical structure with formal markers**, derived from computer-science practice but applicable to all mathematics.

## The Hierarchical Numbering System

A proof at level $n$ is decomposed into sub-proofs at level $n+1$, with explicit numbering: ⟨1⟩1, ⟨1⟩2, ..., where step ⟨k⟩j is justified by sub-steps ⟨k+1⟩1, ⟨k+1⟩2, etc.

- Each step has a **fixed reference** so later steps can cite it precisely.
- Sub-proofs cannot reference results from sibling sub-proofs at deeper levels (variable-scoping discipline).
- The structure is **mechanically checkable** for scope violations.

## Key Statements (TLA+ proof primitives)

1. **ASSUME / PROVE** — explicitly declares the hypotheses in scope and the goal at each step. Eliminates the prose ambiguity "now we want to show..."
2. **NEW** — introduces a new variable with stated type/conditions. Eliminates implicit variable introduction.
3. **SUFFICES** — explicitly states that proving statement $A$ would prove the current goal. Documents reductions.
4. **PICK** — chooses an element from a set with stated properties. Forces the witness to be named.
5. **CASE** — structures case-by-case reasoning explicitly.
6. **QED** — marks proof completion at each level.

## Errors That Structured Proofs Catch (and Prose Doesn't)

Lamport identifies common error classes:

1. **Forgotten cases** — a case enumeration that misses one. Prose proofs often miss these because the WLOG/symmetry claim is informal.
2. **Quantifier-scope confusion** — variables silently introduced or used outside their scope.
3. **Hidden hypotheses** — a step that depends on an assumption not currently in scope.
4. **Circular reasoning** — using the conclusion (or a consequence of it) to prove a sub-step.
5. **Vacuous reductions** — proving the wrong thing because the SUFFICES step was wrong.
6. **Overlooked details in routine steps** — assuming "by induction" without checking the induction goes through.

The **curious-child principle**: imagine a curious child sitting next to you. After every assertion, they ask "Why?" If you cannot answer at the next level of detail, the proof is incomplete.

## The Anti-Confirmation-Bias Principle

> Maintain suspicion of beliefs to avoid confirmation bias.

Lamport observes that mathematicians, like all humans, are inclined to overlook gaps that support their preferred conclusion. Structured proofs externalize the verification, removing the writer's bias.

## Implied Rubric for Proof Quality (21st-century version)

A proof scores higher when it:

- Uses **explicit hierarchical numbering** (or its equivalent).
- States **ASSUME/PROVE** at each step explicitly.
- **Names variables** as they are introduced (NEW).
- **Documents SUFFICES** when reducing one goal to another.
- **Names the witness** when picking an element.
- **Enumerates cases** with explicit CASE markers.
- Allows the reader to **check each step mechanically** without needing to reconstruct context.
- Has **machine-checkable structure** in principle, even if not formally verified.

A proof scores lower when it:

- Hides the proof structure in flowing prose.
- Uses "WLOG" or "clearly" to skip steps without justification.
- Introduces variables implicitly.
- Fails to mark which hypotheses are in scope at each step.
- Cannot survive the curious-child test.

## When Prose Proofs Are Still Acceptable

Lamport allows that for **short, conceptually deep proofs** the structured form may be excessive. The case for structured proofs is strongest for:

- **Long proofs** with many cases.
- **Algorithm proofs** with many invariants to track.
- **Proofs with subtle quantifier order** (especially in logic, set theory).
- **Proofs intended for verification** by skeptical reviewers.

For a one-page "magic" proof, the structure may obscure rather than clarify.

## The Computer-Verification Connection

Structured proofs are **a stepping-stone to formal verification** in systems like Lean, Coq, Isabelle. A proof written in Lamport's style can often be translated to a formal proof script with relatively little effort, while a prose proof requires major reconstruction.

This makes Lamport's framework not just a quality criterion but a **bridge to a higher tier of rigor**.

Sources:
- [Lamport, How to Write a 21st Century Proof (PDF)](https://lamport.azurewebsites.net/pubs/proof.pdf)
- [How to Write a 21st Century Proof (Springer)](https://link.springer.com/article/10.1007/s11784-012-0071-6)
- [the morning paper summary](https://blog.acolyer.org/2015/01/12/how-to-write-a-21st-century-proof/)
- [Why We Need Structured Proofs in Mathematics (CICM)](https://cicm-conference.org/2020/NFM/paper_4_Ayala_Silva.pdf)

---
source_url: https://lamport.azurewebsites.net/pubs/lamport-how-to-write.pdf
title: How to Write a Proof (Leslie Lamport)
source_type: academic_paper
fetched: 2026-05-09
---

# How to Write a Proof — Leslie Lamport (1995, revised 2012)

Original 1995 paper and the 2012 follow-up "How to Write a 21st Century Proof". Proposes **structured proofs**: hierarchical, numbered, machine-checkable-friendly format. Argues that ordinary mathematical proof prose hides errors; structure exposes them.

## Lamport's central claim

> "A method of writing proofs is proposed that makes it much harder to prove things that are not true."

In rewriting published, peer-reviewed proofs in his structured form, Lamport repeatedly found errors that the prose form had concealed.

## The structured-proof format

Each proof is a tree. Each node is a numbered step labeled with the assertion it proves. Each step either:
- Is a terminal "obvious" step (small enough to verify at a glance), or
- Has a sub-proof: a list of numbered children whose conjunction implies the parent.

Notation example:
```
Theorem T.
Proof.
  ⟨1⟩1. Assertion 1
  ⟨1⟩2. Assertion 2
    ⟨2⟩1. Sub-step
    ⟨2⟩2. Sub-step
    ⟨2⟩3. Q.E.D. (proves ⟨1⟩2)
  ⟨1⟩3. Q.E.D. (proves T)
```

## Why ordinary proof prose fails

1. The prose hides what is being proved at each point — the reader does not know which lemma the next sentence belongs to.
2. The flow of "since … and … so" obscures the dependency graph.
3. "Obvious" claims accumulate; one false "obvious" step ruins the whole.
4. Proofs by cases lose the case structure.
5. Quantifier scope is ambiguous.

## Lamport's checklist for a "good" proof

1. Every step is numbered.
2. Every step states what it proves.
3. The "obvious" leaves are short enough that an attentive reader cannot disagree.
4. The hypotheses currently in force at each step are explicit (assumptions are stated, not assumed from context).
5. Quantifiers and case splits are explicit.
6. The Q.E.D. at each level is named: it tells you what assertion is now established.
7. The proof can be re-checked top-down or bottom-up independently.

## When NOT to use structured proofs

Lamport concedes: very short, very famous proofs (e.g., infinitude of primes) need no structure. The format is for **non-trivial** proofs where errors are likely.

## Influence

Structured proofs are the model for modern proof assistants (Isar in Isabelle, Lean's `have`/`show` blocks). Lamport later created the TLA+ proof system around this format.

## 21st-century version (2012) additions

- Hyperlinked references in electronic proofs.
- Collapsible sub-proofs in the reader's interface.
- Integration with mechanical checkers; the prose proof and the formal proof should be the *same* artifact.
- A "bug rate" measurement: in a formal verification effort, structured proofs caught significantly more errors per page than prose.

---
source: Jeremy Avigad, "Mathematical Method and Proof" (Synthese 153(1), 2006, 105–159) and related papers on informal proofs and rigor
url: https://www.andrew.cmu.edu/user/avigad/Papers/method.pdf
type: modern_academic
year: 2006
domain: methodology_of_mathematics
collected: 2026-05-09
---

# Avigad: Mathematical Method and Proof — Beyond Verification

In this paper (and related work on informal proofs and rigor), Avigad argues that the standard formalist account of proof — proof as warrant for truth — fails to explain mathematical practice. He proposes a multi-criterion account.

## The Core Critique

> The traditional view that the primary role of a mathematical proof is to warrant the truth of the resulting theorem **fails to explain why new proofs of theorems are often deemed important**.

If verification were the sole purpose, a second proof of an established theorem would be redundant. But mathematicians celebrate new proofs. This means proofs do something *beyond* verification.

## Multiple Functions of a Proof (Avigad's Expansion of Rota)

A proof can be evaluated for whether it:

1. **Warrants the truth** of the conclusion (the formalist criterion).
2. **Explains why** the conclusion is true (Steiner-style explanation).
3. **Identifies essential vs. accidental hypotheses**.
4. **Reveals connections** to other parts of mathematics.
5. **Provides a method** that generalizes to new problems.
6. **Establishes the optimality** of the result (sharpness).
7. **Justifies the choice of definitions** used.
8. **Yields tools** (lemmas, techniques) reusable elsewhere.
9. **Gives a "feel" for the area** that helps the reader navigate it.

## Why Multiple Proofs of the Same Theorem Are Not Redundant

Each new proof can excel on a different one of the above dimensions:

- The first analytic proof of PNT (Hadamard, de la Vallée Poussin) gave understanding via complex analysis.
- The Selberg-Erdős elementary proof gave **purity** (Detlefsen-Arana sense).
- Newman's contour-integration proof gave **simplicity**.
- Modern proofs via Tauberian theorems gave **methodological generality**.

A rubric that scored only "verifies the theorem" would treat these as equivalent — losing the point.

## The Informal-Formal Gap

Avigad emphasizes a tension:

- **Standard view**: rigor = derivability in a formal axiomatic system.
- **Practice**: research proofs are not derivable in the standard view; they are informal arguments.
- **Resolution attempts**:
  - The "in principle" view: informal proofs are abbreviations of formal ones (Avigad finds this empirically dubious).
  - The "different standard" view: informal rigor is its own thing, not reducible to formal derivability (Avigad sympathetic).
  - The "graduated rigor" view: rigor is a continuum, not binary.

## Diagrams Are Rigorous

A specific Avigad contribution: the use of diagrams in (e.g., Euclidean) proofs is **not soft and fuzzy**. It is governed by **discernible logic** that can be formalized (Avigad et al. provide a formal system for Euclid's *Elements*).

This means a rubric should not penalize diagrams as "informal." A well-used diagram is a rigorous step in many subfields.

## Implied Rubric for Mathematical Method

For each proof, ask:

1. **Does it verify the theorem?** (necessary but not sufficient)
2. **Does it explain why?** (Steiner test)
3. **Does it isolate essential hypotheses?** (could you weaken any?)
4. **Does it reveal connections?** (to what other theorems / areas?)
5. **Does it provide a method?** (generalizable to other problems?)
6. **Does it establish sharpness?** (could the conclusion be strengthened?)
7. **Does it justify definitions?** (why these concepts and not others?)
8. **Does it yield reusable tools?** (lemmas worth their own statement?)
9. **Does it give a feel** for the area?

A great proof scores high on **multiple** dimensions, not just (1).

## On Reliability

Avigad's "Reliability of mathematical inference" further develops the idea that mathematical reliability is **socially distributed and graduated**, not a property of formal systems alone. A complete rubric must:

- Acknowledge that **community trust** is a legitimate input.
- Recognize that **formal verification is one tool**, not the gold standard.
- Treat **multiple independent proofs** as additional evidence, not redundancy.
- Distinguish **first-principles certainty** from **practical reliability**.

## Implications for Computer-Verified Proofs

Avigad has been a central figure in formal verification (Lean, mathlib). His view:

- Formal verification provides a **new dimension** of certainty.
- It does **not replace** the other dimensions of proof quality.
- A formally verified proof can still be **inelegant, unexplanatory, hard to read**, etc.
- The quality landscape is **enlarged** by formalization, not collapsed.

## Connection to Other Frameworks

- **Rota**: Avigad expands Rota's "proof has multiple functions" into a more systematic framework.
- **Lakatos**: dialectical proof is one of Avigad's "feel for the area" sources.
- **Hamami & Morris**: plan reconstruction is one of Avigad's "give a method."
- **Detlefsen-Arana**: purity is one of Avigad's "isolate essential hypotheses."

Sources:
- [Avigad, Mathematical Method and Proof (PDF)](https://www.andrew.cmu.edu/user/avigad/Papers/method.pdf)
- [Avigad, Mathematical Method and Proof (Synthese)](https://link.springer.com/article/10.1007/s11229-005-4064-5)
- [Avigad, Reliability of mathematical inference (Semantic Scholar)](https://www.semanticscholar.org/paper/Reliability-of-mathematical-inference-Avigad/5ab251f679fc42d4270639db22a7bf5ba113f3c8)
- [Avigad, Formally Verified Mathematics (CACM)](https://www.andrew.cmu.edu/user/avigad/Papers/cacm.pdf)
- [Hamami, Mathematical Rigor and Proof (PDF)](https://www.yacinhamami.com/wp-content/uploads/2019/12/Hamami-2019-Mathematical-Rigor-and-Proof.pdf)

---
source: George Pólya, "Mathematics and Plausible Reasoning" (Princeton, 1954, two volumes)
url: https://press.princeton.edu/books/paperback/9780691025094/mathematics-and-plausible-reasoning-volume-1
type: classical_academic
year: 1954
domain: heuristic_reasoning_in_mathematics
collected: 2026-05-09
---

# Pólya: Mathematics and Plausible Reasoning — The Discovery Rubric

NOTE: An existing file `polya_how_to_solve_it.md` covers Polya's earlier and more famous book. This file extracts the additional rubric content from his more philosophically ambitious *Mathematics and Plausible Reasoning* (which extends How to Solve It with explicit non-deductive reasoning).

## Core Thesis

> "Certainly, let us learn proving, but also let us learn guessing."

Mathematical work involves **two distinct logics**:

1. **Demonstrative reasoning** — deductive proof, certain.
2. **Plausible reasoning** — inductive, analogical, abductive guessing — *uncertain but indispensable*.

The bulk of mathematical work is plausible reasoning; proof is the consolidation phase. A piece of mathematical work can be evaluated on both axes.

## Volume I: Induction and Analogy (the heuristic patterns)

### Induction in Mathematics

Mathematical induction (formal proof technique) is distinct from **inductive reasoning** (gathering evidence for a conjecture). Polya's example:
- Observe: 4 = 2+2, 6 = 3+3, 8 = 3+5, 10 = 3+7, 12 = 5+7, ...
- Conjecture: every even number > 4 is a sum of two odd primes (Goldbach).
- This is *inductive evidence*, not proof.

A piece of mathematics is well-grounded inductively when:

- **A non-trivial number of cases** have been checked.
- **Cases have been chosen variably** (not just consecutive small ones).
- **No exceptions** have been found despite search.
- **The conjecture has survived attempts at refinement**.

### Analogy in Mathematics

Analogy: *A is to B as C is to D*. Polya gives many mathematical examples:
- Triangle in plane :: tetrahedron in space.
- Series of integers :: series of polynomials.
- Diophantine equations :: equations over function fields.

A mathematical work that **explicitly notes its analogies** to known cases is more comprehensible and more generative than one that hides them.

### Generalization and Specialization

- **Generalization**: pose a problem in wider terms; the wider problem may be easier.
- **Specialization**: try the simplest case; what works there may extend.

Both are **plausible-reasoning moves**, not proofs. Their use is part of high-quality mathematical work.

## Plausible Reasoning Patterns (Polya's catalogue)

Polya enumerates patterns of plausible inference:

1. **Verification of consequence**: A implies B; B is true; this *increases* plausibility of A (but does not prove it).
2. **Verification of incompatible consequence**: A implies B; B is false; this *refutes* A.
3. **Verification of one of many possible consequences**: many consequences C₁, C₂, ... follow from A; verifying one increases A's plausibility.
4. **Successful prediction**: A predicts a previously-unknown phenomenon, which is then observed; strong plausibility increase.
5. **Plausibility of a conjecture by analogy**: a similar conjecture in an analogous setting was true; this transfers some plausibility.
6. **Reduction to a previously settled case**: if A reduces to B and B is known, A is plausible.

A research program is judged for its plausible-reasoning hygiene:
- Are conjectures supported by **multiple independent confirmations**?
- Have **possible refutations** been actively sought?
- Are **analogies** to known cases noted explicitly?

## Volume II: Patterns of Plausible Inference (formal-ish)

Polya attempts a quasi-quantitative theory of plausibility (precursor to Bayesian confirmation theory). Key principles:

- The plausibility of a conjecture **increases more** when verified consequences are **less likely a priori**.
- The plausibility increases more when **multiple independent** consequences are verified.
- A conjecture is **suspect** if its only support is restricted to a special case.

## Implied Rubric for Mathematical Work (Beyond Proof)

A piece of mathematical work scores high when it:

- **Acknowledges its inductive evidence** for any conjecture made.
- **Explicitly notes analogies** to known cases.
- **Uses generalization and specialization** as conscious moves, not accidents.
- **Sought refutations actively**, not only confirmations.
- **Distinguishes plausible reasoning from proof** — does not present analogy-based suggestions as if proven.
- **Attempts to refute** its own conjectures before publishing.

A piece of mathematical work scores low when it:

- **Confuses inductive evidence with proof**.
- **Hides its analogies** behind formal language.
- **Treats specialization as embarrassing** ("we present the general case for completeness").
- **Asserts conjectures** without indicating the basis of confidence.
- **Has no negative evidence sought** (only checked confirming cases).

## Connection to AI / Machine-Generated Math

Polya's plausible-reasoning framework has direct implications:

- An AI that generates mathematical claims should provide **plausibility evidence** (confirming cases, analogies, reductions).
- An AI that only outputs claimed proofs without supporting plausibility is suspect — real mathematicians always have plausibility evidence first.
- The **distinction between conjecture and theorem** must be respected by automated tools.

## Connection to Other Frameworks

- **Lakatos**: dialectical proof-and-refutation is one specific plausible-reasoning pattern.
- **Cellucci**: heuristic philosophy generalizes Polya into a full account of mathematical method.
- **Hadamard**: psychology of invention provides empirical evidence for the role of plausible reasoning.
- **Schoenfeld**: heuristics catalog operationalizes Polya in pedagogy.

Sources:
- [Pólya, Mathematics and Plausible Reasoning Vol 1 (Princeton)](https://press.princeton.edu/books/paperback/9780691025094/mathematics-and-plausible-reasoning-volume-1)
- [Mathematics and Plausible Reasoning (Wikipedia)](https://en.wikipedia.org/wiki/Mathematics_and_Plausible_Reasoning)
- [Pólya, MPR Vol 1 on JSTOR](https://www.jstor.org/stable/j.ctv14164db)

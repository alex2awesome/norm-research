---
source: Michael Detlefsen & Andrew Arana, "Purity of Methods" (Philosophers' Imprint 11(2), 2011)
url: https://quod.lib.umich.edu/p/phimp/3521354.0011.002/1
type: modern_academic
year: 2011
domain: epistemology_of_proof
collected: 2026-05-09
---

# Detlefsen & Arana: Purity of Methods

A historically informed analysis of the long-standing mathematical preference for **pure** proofs — proofs that use only resources intrinsic to the problem.

## The Core Notion: Topical Purity

A proof of theorem T is **topically pure** if it uses only concepts that are needed to clarify the **content of T itself** — i.e., the concepts already implicitly invoked by the statement.

**Hilbert's formulation**: a pure proof uses only "means that are suggested by the content of the theorem."

## The Paradigm Example

The Selberg-Erdős elementary proof of the Prime Number Theorem (1949):

- **Statement**: $\pi(n) \sim n / \ln n$ — pure number-theoretic content.
- **Standard proof**: uses complex analysis (the Riemann zeta function on the line $\Re s = 1$). Powerful, but "impure" — it imports machinery from analysis.
- **Selberg-Erdős proof**: uses only elementary methods. Topically pure: every concept is one already implicit in the statement.

The Selberg-Erdős proof was celebrated specifically because it was *pure*, not because it was easier or shorter.

## Two Notions of Purity

1. **Topical purity** (Detlefsen-Arana's primary focus): proof uses only concepts from the theorem's *content*.
2. **Geographical purity**: proof stays within a recognized subfield (e.g., a number-theoretic proof of a number-theoretic theorem).

Topical purity is more demanding: a proof can be geographically pure (stays within number theory) but topically impure (drags in number-theoretic concepts not implicit in the statement).

## Epistemological Significance

Pure proofs offer **epistemic stability**:

- They reveal what the conclusion really *depends on*.
- They show the result is not an artifact of imported machinery.
- They support cross-foundational robustness: the theorem survives even if the imported framework is rejected.

Impure proofs are not bad — they are often easier or stronger — but they leave open the question: does the theorem really require this machinery, or could it be derived more economically?

## Implied Rubric for Purity Evaluation

For a given proof:

- **Identify the concepts in the theorem statement.** What is the topic?
- **List the concepts the proof uses.** Which ones are not in the statement?
- **For each imported concept**, ask: is it required, or could a pure proof exist?
- **If imported concepts seem necessary**, what does this tell us about the theorem? (Possibly that it is "really" about something larger.)
- **If a pure proof exists**, prefer it for foundational reasons, even if longer.

## When Impurity Is Justified

The Detlefsen-Arana framework does not condemn impure proofs:

- Impurity is justified when **no pure proof is known** (often the case).
- Impurity is justified when the impure proof reveals **deeper connections** (e.g., the analytic proof of PNT shows links to Riemann zeros).
- Impurity is justified when **purity would obscure** rather than illuminate (e.g., a contrived elementary proof avoiding intuitive analytic tools).

## Connection to Explanation

Pure proofs and explanatory proofs partially overlap. A proof using only the theorem's intrinsic concepts is more likely to **explain why the theorem is true** — because the explanation uses only what the theorem is about. But the two notions are distinct:

- A pure proof can fail to explain (e.g., a long elementary calculation).
- An impure proof can explain (the analytic PNT proof explains *why* via the connection to zeros).

## Anti-Patterns Detected by the Purity Lens

- **Methodological hyper-extension**: dragging in heavy machinery to prove a simple statement.
- **Concealed dependence on choice**: using AC where ZF would suffice (a purity issue at the foundational level).
- **Cross-domain showing-off**: proving a combinatorial fact via algebraic geometry when an elementary argument exists.

Sources:
- [Detlefsen & Arana, Purity of Methods (Philosophers' Imprint)](https://quod.lib.umich.edu/p/phimp/3521354.0011.002/1)
- [Detlefsen, Purity as an Ideal of Proof (PhilPapers)](https://philpapers.org/rec/DETPOM-2)
- [Arana et al., Purity and Explanation: Essentially Linked? (Springer)](https://link.springer.com/chapter/10.1007/978-3-031-21655-8_3)
- [Arana, Elements of Purity (Cambridge Elements)](https://www.cambridge.org/core/elements/abs/elements-of-purity/493DEBE942EB8A4494BADB505483BF02)

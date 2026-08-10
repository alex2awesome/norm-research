---
source_url: https://faculty.weatherhead.case.edu/dxs8/htradp.html
title: How to Read and Do Proofs (Daniel Solow)
source_type: textbook
fetched: 2026-05-09
---

# How to Read and Do Proofs: An Introduction to Mathematical Thought Processes — Daniel Solow (5th ed., Wiley)

Long-running undergraduate text. Solow's distinctive contribution: a *taxonomy* of proof techniques, indexed by surface keyword in the statement to be proved. Reading "for every" triggers the Choose Method; reading "there exists" triggers the Construction Method. Proof becomes a partly mechanical pattern-recognition skill.

## Solow's proof techniques

### 1. Forward-Backward Method
- Forward: from hypotheses, derive consequences.
- Backward: from the goal, identify what would suffice.
- Meet in the middle.
- Default method for most statements.

### 2. Construction Method
- Triggered by "there exists $x$ such that $P(x)$".
- Find an explicit $x$; verify $P(x)$.

### 3. Choose Method
- Triggered by "for every $x$, $P(x)$".
- Let $x$ be an arbitrarily chosen object of the relevant type.
- Prove $P(x)$ without using any property of $x$ beyond its type.

### 4. Specialization
- Triggered when applying a "for every" hypothesis to a specific $x$.
- Substitute and use the consequent.

### 5. Contradiction Method
- Triggered when no direct path is visible.
- Assume $\neg \text{conclusion}$, derive a falsehood.

### 6. Contrapositive Method
- Triggered for $P \Rightarrow Q$ when $\neg Q \Rightarrow \neg P$ is easier.

### 7. Uniqueness Method
- Triggered by "there exists a unique $x$".
- Existence proof + assume two such, show they are equal.

### 8. Induction
- Triggered by "for all $n \in \mathbb{N}$, $P(n)$".
- Base case + inductive step.
- Strong induction when needed: assume $P(k)$ for all $k \le n$.

### 9. Either-Or Method
- Triggered by hypothesis $P \vee Q$.
- Two cases.

### 10. Max/Min Method
- Triggered by claims about extreme elements.
- Show every element ≤ the proposed max; show the proposed max is achieved.

## Solow's algorithm for "doing a proof"

1. **Identify keywords** in the statement to be proved.
2. **Match to the technique** indicated by those keywords.
3. **Apply the template** of that technique.
4. **Recursively** apply the algorithm to the resulting sub-goals.

## Solow on "reading a proof"

A proof is a *sequence of techniques applied*. Reading well = identifying which technique was used at each step. Solow trains the student to label each paragraph: "Construction step", "Forward step", "Specialization", etc.

## Implicit rubric

A "good proof" in Solow's framework:
1. Uses techniques whose triggers are present in the statement.
2. Applies each technique by its standard template.
3. Labels (or makes obvious) which technique is in use at each step.
4. Doesn't invoke a technique without its trigger.
5. Sub-goals are themselves provable by the same algorithm.

## Influence

- Standard text since 1982 (now in 5th ed.).
- Translated into multiple languages.
- Models the keyword-pattern approach used in many automated theorem provers' tactic libraries.

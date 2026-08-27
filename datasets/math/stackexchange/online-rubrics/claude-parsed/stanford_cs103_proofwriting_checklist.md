---
source_url: https://web.stanford.edu/class/cs103/proofwriting_checklist
title: CS103 Proofwriting Checklist (Stanford)
source_type: university
fetched: 2026-05-09
---

# CS103 Proofwriting Checklist — Stanford CS103: Mathematical Foundations of Computing

The eight-item checklist used by Stanford's introduction-to-proofs course. One of the most-downloaded undergraduate proof-writing rubrics.

## The eight principles

### 1. Clearly articulate assumptions and "want-to-shows"
Every proof should make explicit:
- What is being assumed at the start.
- What is being proved (the goal / want-to-show).
- The proof technique chosen (direct, contrapositive, contradiction, induction).
At each major step, restate the current want-to-show.

### 2. Make each sentence load-bearing
> "Every statement in a proof should do one of the following things: set up a goal, introduce a new variable, or combine preceding results into something new."
Cut sentences that don't advance the argument or aren't referenced again.

### 3. Scope and properly introduce variables
Every variable must be classified as:
- **Universally instantiated** ("Let $n$ be an arbitrary even integer").
- **Existentially instantiated** ("There exists $k$ such that $n = 2k$").
- **Explicitly chosen value** ("Let $n = 6$").
Use "let" / "consider" / "choose" — not "for every" — when introducing.

### 4. Make specific claims about specific variables
Statements must reference the specific variables in scope; avoid floating abstract claims. Manipulate concrete expressions, not high-level summaries.

### 5. Don't repeat definitions; use them
Apply each definition; don't restate it. Bad: "An even number is one of the form $2k$. Therefore, since $n$ is even...". Good: "Since $n$ is even, write $n = 2k$ for some integer $k$."

### 6. Write in complete sentences and paragraphs
The "mugga mugga" test: replace every formula with the words "mugga mugga" and the prose must still be grammatical English. No symbols-as-verbs. Proofs are paragraphs, not bullet lists.

### 7. Avoid quantifiers and propositional connectives in the prose
Do not use $\forall$, $\exists$, $\Rightarrow$, $\wedge$, $\vee$, $\neg$ as words inside the proof. Write "for all", "there exists", "implies", etc.

### 8. Avoid the "contradiction sandwich"
If the body of an alleged contradiction proof is in fact a direct proof, present it directly. Don't wrap it in "Suppose for contradiction... [direct proof]... but this contradicts our assumption."

## Common mistakes the checklist catches

- Variables used before being introduced.
- Reuse of a variable name with a different meaning.
- Implicit quantifier scope.
- Conflating $A = B$ as both equation and definition without saying which.
- Missing case analysis.
- Skipping a "back-substitute" at the end of an induction.
- Using "clearly" / "obviously" to skip a step that the grader will not believe.

---
source_url: https://web.stanford.edu/class/archive/cs/cs103/cs103.1242/guide_to_proofs
title: Stanford CS103 Guide to Proofs (full structure rules)
source_type: course_handout
fetched: 2026-05-09
---

# Stanford CS103 Guide to Proofs — proof-pattern-by-statement-type

A widely-used CS-flavored proof-writing rubric organised by statement type.

## Direct Proofs of Implications

Four-step structure:
1. "Clearly state to your reader that you are assuming the antecedent is true."
2. State you will prove the consequent.
3. Provide reasoning connecting assumption to conclusion.
4. "Indicate to the reader that you ended up where you said you were going."

## Universally-Quantified Statements

Pattern:
1. "Instruct the reader to pick an arbitrary object" — use "Pick", "Consider", "Let", "Fix", or "Choose."
2. Prove the object necessarily has the required property.
3. Provide the logical reasoning.
4. Confirm the stated goal was achieved.

## Existentially-Quantified Statements

Key difference: tell the reader *which* object you're picking.

Structure:
1. Specify the exact object satisfying the requirement.
2. Demonstrate it possesses all necessary properties.

## Biconditionals

"Write out two separate proofs, one for each of the two implications."

## Proof by Contrapositive

1. Announce "I will prove the contrapositive."
2. Write out the contrapositive explicitly.
3. Prove the contrapositive using any standard technique.

## Proof by Contradiction

1. "Assume that X is false" and "explicitly write out the negation."
2. Derive an impossible conclusion.
3. "State that you've reached a contradiction."
4. Conclude the original assumption was incorrect.

## Implication for "Good Math Writing"
The CS103 rubric is **explicitly metalinguistic**: every proof signposts what kind of proof it is and where in the structure each step sits. A "good" proof in this register is one in which every move is announced.

---
source_url: https://www-cs-faculty.stanford.edu/~knuth/papers/cs1193.pdf
title: Mathematical Writing (Knuth, Larrabee, Roberts)
source_type: academic_paper
fetched: 2026-05-09
---

# Mathematical Writing — Donald E. Knuth, Tracy Larrabee, Paul M. Roberts (1989)

Stanford CS report STAN-CS-88-1193 (Jan 1988); published as MAA Notes vol. 14 (1989). Records the 31 lectures of Knuth's Stanford CS 209 "Mathematical Writing" course in 1987, with guest lectures by Herb Wilf, Jeff Ullman, Leslie Lamport, Nils Nilsson, Mary-Claire van Leunen, Rosalie Stemer, and Paul Halmos.

## The 27 "Knuth Rules" (Lecture 1, distilled from his ten years writing TAOCP)

1. **Symbols in different formulas must be separated by words.** "$x_n$, $y_n$" not "$x_n y_n$".
2. **Don't start a sentence with a symbol.** Recast or add a word.
3. **Don't use the same notation for two different things.** And don't use two different notations for the same thing.
4. **Don't compose a sequence of formulas with no intervening words.** Display formulas need English glue.
5. **Display unwieldy formulas; do not run them in.** A formula that takes more than a third of a line is a candidate for display.
6. **Punctuate displayed formulas as if they were ordinary words.** Comma, period, semicolon belong to the surrounding sentence.
7. **Don't omit "the".** "Equation 3", not "Equation 3"; prefer "the equation $x=y$".
8. **Don't use "any" when you mean "every" or "some".** Ambiguous quantifier.
9. **Avoid the use of the same word in two different senses in the same paragraph.**
10. **Avoid using even one word that will be unfamiliar to most readers.**
11. **Vary the sentence structure.** Long, short, long, short.
12. **Don't use jargon that adds no information.** "The fact that" usually means "since" or "because".
13. **Use parallel construction in parallel constructions.**
14. **Re-read what you wrote, often.** Out loud.
15. **Avoid lists of three with implicit "and" / "or".** State the connective.
16. **Resist the urge to use elaborate words.** Use short familiar Anglo-Saxon words when possible.
17. **The word "we" usually means "the author and the reader".** Don't use "I"; don't be afraid of "we".
18. **Don't number an equation unless it is referenced.**
19. **Don't use the word "infinity" as if it were a number.**
20. **Use the symbols < > ≤ ≥ correctly and don't mix them in one chain.**
21. **Italicize variables; do not italicize numerals or function names like $\sin$, $\log$.**
22. **Use real punctuation marks in math mode** (Knuth uses TeX so this is precise).
23. **Display equations should be readable when the surrounding text is masked.**
24. **Avoid undefined acronyms.**
25. **State the theorem before proving it.**
26. **Be careful with the words "if", "iff", "implies", "only if".**
27. **Avoid implicit subjects.** "Adding x to both sides yields..." — adding *what* to *whose* sides?

## Mary-Claire van Leunen's contributed rules

- Footnotes are usually evidence of bad writing.
- Citations belong inline, not floating.
- Don't write "the author" when you mean "I".

## Halmos's contributed lecture (reprised)

See the Halmos rules; the Stanford lectures distilled them with examples.

## "Before and after" exercises

The course used systematic editing examples — taking a passage from a published paper and rewriting it. Lessons:
- Cut adjectives and adverbs first.
- Move definitions to the point of first use.
- Replace nested negatives with positives.
- Convert nominalizations back to verbs ("a proof of the convergence" → "we prove the convergence").

## Paragraph and section structure

- Each paragraph has one theme; the first sentence states it.
- Sections follow the order: motivation → definitions → results → proofs → examples → applications.
- Lemmas should be motivated by the theorem they prepare.

## The role of footnotes, references, indices

Knuth (Lecture 17): A reference that the reader cannot find should not be cited. A footnote that the reader needs to read should be in the text. A footnote that the reader does not need to read should be cut.

## Closing principle

> "The reader is your friend, not your adversary, not your judge."

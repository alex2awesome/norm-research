---
source_url: https://math.mit.edu/~poonen/papers/writing.pdf
title: Practical Suggestions for Mathematical Writing (Bjorn Poonen)
source_type: math_writing_essay
fetched: 2026-05-09
---

# Practical Suggestions for Mathematical Writing — Bjorn Poonen (MIT, 2026 update)

A 65-item color-coded guide ("red is bad and green is good") in five sections.

## 1. Important things

1. **Justify each step.** If a claim does not follow immediately from the previous sentence alone, explain what it does follow from ("Combining the previous two sentences shows that..." / "By Lemma 8.3, ...").
2. **In multi-claim sentences, attach reasons to each claim.** Place reasons before/after the chain or use `\begin{align*}` with right-hand justifications.
3. **End-of-sentence understanding.** A reader at the period should know why each claim is true. If proof comes later, signal it ("..., as we now explain").
4. **Eliminate ambiguity.** It's not enough that your words *can* be interpreted correctly; they must not admit any other interpretation.
5. **Break long arguments into lemmas**, even if used only once. Minimise what the reader must keep in mind.
6. **Flag relevance.** If only one of several propositions is needed later, say so.
7. **Quantifiers must be unambiguous.** Write "for all x ∈ R" or "for some x ∈ R", not bare "for x ∈ R".
8. **Indicate where each hypothesis is used** in proofs.
9. **Citations specify location.** Theorem number or page, unless citing the entire work.
10. **Cite published versions** over preprints when available.
11. **arXiv preprints**: include version number or precise date.
12. **"Forthcoming work"** only if a publicly available preprint exists.

## 2. Title, abstract, and introduction

13. **Matryoshka rule.** Title, abstract, intro, and body each describe the entire article, decreasing in abbreviation.
14. **Title length.** Long enough to convey content; specific enough to distinguish.
15. **Drop "A note on", "Remarks on"** from titles.
16. **Abstract states main results** if they fit; omit precision-only definitions.
17. **Abstract is self-contained**: no citations, no body references.
18. **Get to new theorems fast** in the introduction; postponing notation to a later section is OK.
19. **Math papers usually do not have a conclusions section.**

## 3. Other things

20. **Theorem statements short.** Definitions precede theorems.
21. **Sentences short.** Combine only to clarify logic.
22. **Chain equalities** in correct order; transitivity is universal.
23. **Easier parts first** when a proof breaks into parts.
24. **Isomorphism claims:** specify the map explicitly.
25. **Induction:** start at n=0 if easier than n=1.
26. **"Clear" is suspicious.** "Usually when people write that something is 'clear', it is because they could not figure out a good explanation of why it is true."
27. **Define before use.** Avoid the ", where..." construction; introduce variables before they appear.
28. **Sentence subjects can't span formulas.** Rewrite "the discriminant ∆" to keep the formula functional.
29. **Skip filler intros.** "We now prove the following proposition" adds nothing.
30. **No abbreviations** like WLOG, iff, s.t. (blackboard only).
31. **Write quantifiers in words.** ∃ → "there exists", ∀ → "for all".
32. **Don't start a sentence with a symbol.** "H denotes the Sylow p-subgroup" is wrong as a full sentence.
33. **No contractions** ("don't") in formal writing.
34. **Don't use proof-by-contradiction** when a direct proof suffices.
35. **Refer to theorems by number**, not "the previous theorem."

## 4. LaTeX issues

36. Single numbering for all theorems/lemmas.
37. Use `\DeclareMathOperator` for operator-style names.
38. `\hfill` before `\begin{enumerate}` to prevent first-item misalignment.
39. `f \colon X \to Y` not `f : X \to Y` (spacing).
40. Try `\usepackage{fullpage}`, `\usepackage{microtype}`, `\usepackage{colonequals}`.

## 5. Nitpicks

43. **"so that" implies purpose; "such that" imposes condition.** Replaceable by "in order that" / "with the result that"? Then "so that".
44. **"so" for plain implication.** "A, so B".
45. **Connect sentences with comma + conjunction**, not bare semicolon, unless a colon explains.
46. **"Only" goes next to the word it modifies.**
47. **"Given an element g of G"**, not "Given g an element of G."
48. **After "Let," the variable being defined.** "Let Z be the center of G", not "Let the center of G be Z."
49. **Capitalise "Theorem" when referring by number.**
50. **Display formulas keep prior punctuation.**
51. **"Assume that G is a finite group"**, not "Assume G is a finite group" (unless a noun-phrase like "Assume Hypothesis A").
52. **"the 1980s"** not "the 1980's".
53. **i.e./e.g. need a comma** in American English.
54. **Minimise parentheses.** `log x` over `log(x)`; `sin 2x` is fine; `sin(x+y)` requires parens.
55. **Skip multiplication symbols** when juxtaposition is unambiguous.
56. **"We remark that" is filler** unless the alternative starts with a symbol.
57. **No "This concludes the proof"** if you have a QED symbol.
58. **Fractions in exponents/subscripts use slash**, not stacked.
59. **Sequences are tuples**: `(a_i)_{i≥0}` not `{a_i}_{i≥0}`.
60. **"Principal" = main; "principle" = rule.**
61. **Spell-check pitfalls:** *separable* not seperable; *archimedean* not archimedian; *homogeneous* not homogenous.
62. **No hyphen in "non-"** prefixes (nonempty, nonzero...).
63. **Numerals for most numbers**, except single-digit counting numbers.
64. **"since" preferred over "as"** to avoid ambiguity.
65. **"by" preferred over "per"** when "by" is meant.

## Implication for "Good Math Writing"
A "good" math paper passes all 65 of Poonen's checks. The list itself functions as a near-objective rubric.

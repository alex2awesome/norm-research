---
source_url: https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Mathematics
title: Wikipedia Manual of Style — Mathematics
source_type: professional_org
fetched: 2026-05-09
---

# Wikipedia Manual of Style/Mathematics

The largest collaboratively-maintained guide for writing mathematics for a general audience. Covers article structure, prose style, notation, typography, and proof inclusion.

## Article structure

- **Lead section** must be accessible to a general reader. State alternate names in bold. Provide informal motivation, history, and applications.
- **Body** introduces formal definitions in dedicated "Definition" sections, then theorems, then examples, then proofs.
- **Generalizations**, "See also", "References" sections at the end.
- A picture before mathematics whenever possible.

## Writing style

1. **Don't begin a sentence with a symbol.**
2. **Avoid textbook-style rhetorical devices** ("Note that", "Obviously", "It is easily seen that", leading questions).
3. **Minimize the authorial "we".**
4. **Avoid blackboard abbreviations**: no "wrt", "wlog", "iff" in prose.
5. **Replace symbols with words in prose**: write "for all" rather than $\forall$, "in" rather than $\in$.
6. **Avoid the ambiguous "any"**; use "every" or "some".
7. **Don't use "if and only if" in definitions**; rephrase as "is defined to be".
8. **Define every variable before use.**
9. **Explain formulas with words.**

## Conventions on the mathematics itself

- **Rings** are assumed associative and unital; non-unital rings are "rngs". Exception: operator algebras.
- **Compact** spaces are not assumed Hausdorff (state Hausdorff explicitly when needed).
- **Natural numbers**: be explicit whether 0 is included.
- **Subsets**: $\subseteq$ for subset, $\subsetneq$ for proper subset.
- **Matrix transpose**: superscript non-italic T.

## Typesetting

- **Displayed formulae**: prefer LaTeX (`<math display=block>`).
- **Inline formulae**: no consensus; HTML or LaTeX, but never mix in one expression.
- **Italicize variables**, never numerals or function names like sin, log.
- **Capital Greek**: not italicized.
- **Function names**: use upright font (sin, cos, log).
- **Sets of numbers**: bold $\mathbb{R}$ or LaTeX `\mathbb{R}` — be consistent within an article.
- **Multiplication in formulae**: juxtaposition; in explanations for general readers, use × .
- **Minus sign**: U+2212 −, not hyphen-minus.
- **Subscripts/superscripts**: HTML `<sub>`/`<sup>`, not Unicode characters.
- **Roman numerals**: ASCII letters only.

## Formula explanation

- Explain via prose, not bullet lists, when possible.
- "where $b$ is the … vector, $a$ is the … coefficient, and $r$ is the … vector".
- Punctuation after a formula must be the natural English punctuation (period, comma).
- For LaTeX in inline use, place punctuation inside the `</math>` tag to avoid line-wrap orphans.

## Proofs in articles

- **Include** proofs when they illuminate concepts.
- **Exclude** proofs that only establish correctness without insight.
- **Set apart** proofs in collapsible boxes or separate sections so the article remains readable to those who don't want to read them.

## Algorithms

- Pseudocode preferred over a specific language.
- Use syntax highlighting.
- One implementation; not many.

## Citations

- Cite historical papers for important theorems.
- Link to free online sources where possible.
- Verifiability is mandatory.

## Specific notation pitfalls flagged

- $\equiv$ vs. $=$ in definitions — use $=$ unless modular.
- $\setminus$ for set difference, never `\` alone.
- $\circ$ for composition (LaTeX `\circ`), not Unicode ring operator.
- `\cdot` for inner product (or use $\langle\cdot,\cdot\rangle$), never juxtaposition.

## Influence

This is the de facto common-language style guide for mathematical exposition online; Wikipedia's mathematics articles are among the most-read mathematical text on the web, so this guide propagates broadly.

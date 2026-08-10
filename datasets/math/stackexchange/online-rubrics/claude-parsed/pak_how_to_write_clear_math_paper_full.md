---
source_url: https://www.math.ucla.edu/~pak/papers/how-to-write1.pdf
title: How to Write a Clear Math Paper — 21st Century Tips (Igor Pak, full PDF)
source_type: math_writing_essay
fetched: 2026-05-09
---

# How to Write a Clear Math Paper: Some 21st Century Tips — Igor Pak

A long-form essay (15+ pages) — distinct from his blog summary — articulating concrete, opinionated rules for math paper writing.

## 1. Be Clear (the Golden Rule)

1.1 Clarity is "absolutely paramount." Means clarity in idea, not just phrasing.
1.2 Being clear is *hard* and slow. "It gets easier after the first 300 papers." (Noga Alon, quoted.)
1.3 Why be clear? "Being clear is not about you. You must think of the reader and how they will read your paper."
- A grad student with poor English will give up at page 3.
- A postdoc skimming 20 papers will silently drop yours.
- For juniors: "Clear writing will make people take you seriously."
- For all: "Clear writing will give you a competitive advantage" — same result, two papers; the clear one wins.
1.4 Journals don't fix clarity (copy editors only catch grammar).
1.5 **For the sake of clarity, ignore all rules** (echoing Wikipedia's IAR principle).

## 2. Where to Start
2.1 Start with other literature: Halmos, Higham, Knuth, Krantz, Berndt, Goldreich, S.P. Jones; Tao's blog.
2.2 Read a non-fiction writing guide (Zinsser is recommended). Adapted Zinsser advice:
- Keep paragraphs short.
- Two-three sentence newspaper paragraphs are fine; air around prose helps.
- But don't go to the opposite extreme of midget verbless paragraphs.
2.3 Aim for *clear* rather than *perfect* writing. Modern readers skim, search arXiv, read only intros.

## 3. Macro Tips

### 3.1 Structure (Matryoshka principle)
Title -> abstract -> introduction -> main part -> final remarks -> references. Brief summary first, longer summary next, full facts only after the reader is hooked.

### 3.2 Title
- Don't be too long, short, vague, or generic ("On some problems in group theory").
- First approximation of the paper's contents.
- Tricks: name your objects ("Munro permutations") so the title can reference them.
- Self-quote (MathOverflow): emphasise *content*. "Short proof that all tennis balls are white" > "Tennis ball coloration." For surveys: signal "A survey on..." in the title.

### 3.3 Abstract
- Easy section: think of a short MathSciNet-summary.
- Dry facts. State key results first; mention generalisations briefly.
- No precise statements; no details; no other-work connections unless necessary.
- Length rule of thumb: 0.3–0.5 lines per page.

### 3.4 Table of Contents
- Skip unless paper >60 pages (Adobe Reader does it).

### 3.5 Introduction
- Hardest section. "Probably the only part of your paper that will be read by all but a few most devoted readers."
- Have a senior coauthor or colleague write/comment on it.
- Draft early; rewrite after the rest is written; let it stew for a week with trusted colleagues.
- *What to include:* problem set-up, statements of main results, first theorem on page 1 or 2.
- *What not to include:* technical definitions, examples, big figures (use "(see §3.4)" links instead).
- Ignore Rota's "give everybody his due" — explain only directly-relevant history.
- Last paragraph: outline the paper's structure.

### 3.6 Foreword
- For long papers (>3 pages of intro), prepend a non-technical "Foreword" subsection.
- May contain literary, big-picture, philosophical content — the only place to do so.
- "If it's beautiful or sufficiently memorable, it might be quoted in other papers."
- Even short papers benefit from one literary opening paragraph; "your only place to shine."

### 3.7 Final Remarks
- Function: expanded endnotes/footnotes section.
- Untitled subsections (`\subsection{}`), one paragraph to one page each.
- Order by decreasing importance: history first, then where-do-you-go, then speculative conjectures, then others' conjectures.
- Use as a placeholder dump while writing; refer to it as "(for more on this, see §6.1)."

### 3.8 Acknowledgements
- "Give lavish acknowledgements" (Rota) — but make choices.
- Order of *increasing* importance.
- Thank everyone discussed the work, by name in alphabetical order.
- Single out specific contributors with explanations.
- Email permission only when using private information.
- End with institutions and granting agencies.

## 4. References

### 4.1 Why important?
"If we are uncited, ignored, all hope is lost." Citations are the social capital system.

### 4.2 Twelve citation styles (decreasing reliability)
1. "Roth proved Murakami's conjecture in [Roth]." Clear.
2. "Roth proved Murakami's conjecture [Roth]." Possibly different paper, same author, definitive.
3. "Roth proved Murakami's conjecture, see [Roth]." [Roth] = paper, follow-up, or survey.
4. "Roth proved Murakami's conjecture [Roth], see also [Woolf]." Woolf added something important.
5. "Roth proved Murakami's conjecture in [Roth] (see also [Woolf])." Woolf has a complete proof, fixing minor errors.
6. "Roth proved Murakami's conjecture (see [Woolf])." Woolf is the definitive monograph.
7. "Roth proved Murakami's conjecture, see e.g. [Faulkner, Fitzgerald, Frost]." Important enough to be in textbooks.
8. "Roth proved Murakami's conjecture (see e.g. [F,F,F])." Classical / well-known.
9. "Roth proved Murakami's conjecture.^7 See [Mailer]." Author hasn't read [Mailer].
10. "...^7 Love letter from H. Fielding to J. Austen, dated December 16, 1975." Letter likely exists.
11. "...^7 Personal communication." Roth claimed in private; may or may not be correct.
12. "Roth claims to have proved Murakami's conjecture in [Roth]." Known gap or error.

### 4.3 How to cite a list of papers
Don't write "see [2-19] for relevant work." Disservice to readers and authors. Walk through individually, most-important first; describe each contribution.

### 4.4 Where to cite
- Most relevant papers in the Introduction.
- Rest in Final Remarks.
- Nothing in the main part *except* a precise reference for a borrowed lemma.

### 4.5 Forming references
- Use section/subsection numbering (e.g. "[A, §3.1]") since arXiv pagination shifts.

### 4.6 Style of references
- Do NOT use BibTeX unless advanced; MathSciNet citations are bloated.
- Make each reference concise but findable.
- Use alphanumeric style for long papers (e.g. "[SY09]"); "[Con17+]" for unpublished.
- "For papers with 5 or more authors, use [A+13] in place of [ABCDEF13]."
- Don't emulate Knuth's perfectionism.
- For unpublished papers, include arXiv number or link.

## 5. Micro Tips

### 5.2 Don't be pedantic
Skip pedantic notation distinctions (against Serre's Q⊂R example). "Just draw a picture and get on with your math."

### 5.3 Downshift your style
- Audience includes non-native English speakers, short attention span, no patience.
- Be repetitive; don't vary. Use "ten therefores"; avoid "henceforth."
- Outside intro and final remarks, use only present indefinite (occasionally past indefinite).
- Short sentences. Commas for sentence structure clarity, not pauses. No semicolons (implies vague logical connection).
- No long dashes without spaces.
- Use Standard American spelling; let Google break ties.
- If you define a term ("nice graph"), don't use variations ("niceness of graphs").

### 5.4 LaTeX tips
- Create a *ton* of macros (`\al` for `\alpha`, etc.). Use the same macros across all papers.
- Avoid letters that look alike (Ξ, ι, ϖ, ι, ȷ, ϰ); avoid κ vs k confusion.
- Use ∅ in place of ∅, ℓ in place of l, ε in place of ϵ.
- Mnemonic letters: f,g,φ,ϕ,ζ,ξ,η,ρ for functions; a–e for constants; x,y,z for variables; i,j,k,ℓ,m,n for integers; p,q,r,s,t,u,v for anything.
- Place macro placeholders for unsure notation; play with fonts at the end.
- Avoid Gothic fonts (𝔊 vs 𝔖).

### 5.5 Do NOT trust LaTeX
- LaTeX spacing/typesetting can introduce ambiguity (Φ(2a+c)Φ(c-2a)... example).
- Insert manual `\,` to disambiguate.

### 5.6 Figures
[Continued in remaining pages.]

## 6. What to Do When Done
- "You are never really done." Keep updating arXiv version.
- Advertise and popularise.
- Rewrite and republish if needed.

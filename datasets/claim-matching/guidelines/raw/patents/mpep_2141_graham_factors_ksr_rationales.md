# MPEP 2141 — Examination Guidelines for Determining Obviousness Under 35 U.S.C. 103
SOURCE_URL: https://www.uspto.gov/web/offices/pac/mpep/s2141.html
DOMAIN: patents

## Overview

MPEP 2141 sets out the framework examiners must use to decide whether a claimed invention would have been obvious under 35 U.S.C. 103, combining the three-factor factual inquiry from *Graham v. John Deere Co.* with the flexible, non-rigid approach mandated by the Supreme Court in *KSR International Co. v. Teleflex Inc.*, 550 U.S. 398 (2007). This is the doctrinal anchor for any claim-vs-prior-art obviousness comparison: it tells you *what facts must be found* before a legal conclusion of obviousness can be reached.

## The Graham Factual Inquiries

Obviousness is ultimately a legal conclusion, but it must rest on findings for three (sometimes four) underlying factual questions:

1. **Scope and content of the prior art** — what was actually known/disclosed in the field at the relevant time (see MPEP 2141.01).
2. **Differences between the prior art and the claims at issue** — the delta between what is claimed and what the closest reference(s) show (see MPEP 2141.02).
3. **The level of ordinary skill in the pertinent art** — the capabilities imputed to the hypothetical PHOSITA (person having ordinary skill in the art) (see MPEP 2141.03).
4. Secondary considerations (commercial success, long-felt need, failure of others, unexpected results, etc.) — objective evidence bearing on obviousness/nonobviousness, considered when present.

These are not boilerplate steps; the examiner (or an evaluator reconstructing the analysis) must actually articulate findings for each before concluding the claim would or would not have been obvious.

## KSR's Correction to the Rigid TSM Test

Before *KSR*, the Federal Circuit's teaching-suggestion-motivation (TSM) test required an explicit textual "reason" in the prior art to combine references. *KSR* held this too rigid and identified four specific errors in that rigid application:

- Limiting the obviousness analysis only to the problem the patentee was trying to solve (ignoring other problems the prior art elements were known to address).
- Assuming a PHOSITA would only combine prior art elements aimed at solving the *same* problem the inventor solved.
- Rejecting "obvious to try" reasoning outright as insufficient for obviousness.
- Overemphasizing avoidance of hindsight to the point of using overly rigid preventative rules.

*KSR* reaffirmed the core, older principle that a "combination of familiar elements according to known methods is likely to be obvious when it does no more than yield predictable results." It also confirmed that a PHOSITA is a person of ordinary *creativity*, not an automaton, capable of fitting together the teachings of multiple references "like pieces of a puzzle."

## The Seven Rationales (Full List, Detailed in MPEP 2143)

Following *KSR*, Office personnel may rely on any of the following exemplary rationales to support an obviousness conclusion, provided the underlying factual findings are made explicit:

- **(A)** Combining prior art elements according to known methods to yield predictable results.
- **(B)** Simple substitution of one known element for another to obtain predictable results.
- **(C)** Use of a known technique to improve similar devices (methods, or products) in the same way.
- **(D)** Applying a known technique to a known device (method, or product) ready for improvement to yield predictable results.
- **(E)** "Obvious to try" — choosing from a finite number of identified, predictable solutions, with a reasonable expectation of success.
- **(F)** Known work in one field of endeavor prompting variations for use in the same or a different field based on design incentives or other market forces, if the variations are predictable to a PHOSITA.
- **(G)** Some teaching, suggestion, or motivation (TSM) in the prior art that would have led a PHOSITA to modify the reference or combine references to arrive at the claimed invention.

This list is explicitly non-exhaustive; other rationales are permissible provided the examiner articulates a rational underpinning tying factual findings to the legal conclusion.

## MPEP 2141.01 — Scope and Content of the Prior Art (Analogous Art Requirement)

A reference must qualify as "analogous art" before it can be used in an obviousness rejection. The test has **two independent prongs** — satisfying *either* is sufficient:

1. **Same field of endeavor** — the reference comes from the inventor's general field/industry, regardless of the specific problem addressed.
2. **Reasonably pertinent to the particular problem** — even if from a different field, the reference addresses the problem the inventor was facing, such that it would have "logically commended itself to an inventor's attention" when addressing that problem.

An inventor is not presumed to be aware of all prior art in unrelated fields; examiners must articulate specific reasons why someone confronting the identified problem would have consulted the reference. Note that anticipation (35 U.S.C. 102) does *not* require an analogous-art analysis — analogous art is an obviousness-specific gate. *KSR* did not change the analogous-art doctrine itself but endorsed reading it broadly, observing that "when a work is available in one field of endeavor, design incentives and other market forces can prompt variations of it, either in the same field or a different one," and such variations are obvious if predictable to a PHOSITA.

## MPEP 2141.02 — Differences Between the Prior Art and the Claimed Invention (Identifying the Delta)

This is the section most directly relevant to computing the "difference" between a claim and its closest reference:

- **Claim as a whole.** The comparison must consider the claimed invention "as a whole" — not a distilled "gist" or "thrust" of it. Focusing on one structural change while ignoring the claim's functional character improperly narrows the inquiry.
- **Discovery of a problem's source can be patentable**, even if the fix, once the cause is known, is itself obvious — *unless* the prior art already discloses the identical solution to a comparable problem, in which case discovering the cause does not, by itself, overcome obviousness. Applicants relying on "problem discovery" as their inventive contribution must substantiate that with declarations or clear disclosure in the specification.
- **Inherent properties** are part of the claim "as a whole" and may be used in the comparison, but obviousness cannot rest on a property that was *unknown* at the relevant time, even if later shown to be inherent.
- **Teaching away.** The prior art must be read in its entirety, including portions that discourage the claimed approach. A reference that criticizes, discredits, or otherwise discourages the solution actually claimed weighs against obviousness. However, a reference's *mere disclosure of multiple alternatives* does not, by itself, constitute a teaching away from any one of them — teaching away requires actual discouragement or criticism of the specific option claimed.

## MPEP 2141.03 — Level of Ordinary Skill in the Art

Level of skill is typically assessed using the **GPAC factors**:

1. Type of problems encountered in the art.
2. Prior art solutions to those problems.
3. Rapidity with which innovations are made.
4. Sophistication of the technology.
5. Educational level of active workers in the field.

Not every factor need be explicitly present in every case — one factor may predominate. Key clarifications:

- A PHOSITA has **ordinary creativity**, not automaton-like literalism, and can combine/adapt multiple references' teachings.
- A specification's silence on how to achieve a step may itself indicate that the step was within ordinary skill (subject to enablement).
- The hypothetical PHOSITA is not defined by a specific credential (e.g., a particular degree or years of experience) but by an understanding of the scientific/engineering principles germane to the field.
- Materials that do not themselves qualify as prior art (e.g., evidence of near-simultaneous independent invention by others) can still be used as evidence of what the ordinary skill level actually was.

## Requirements for a Proper Prima Facie Rejection

Putting the pieces together, a proper obviousness rejection must:

- Make explicit findings on the scope/content of the prior art and the differences from the claims.
- State (explicitly or implicitly) the level of ordinary skill assumed.
- Explain, with articulated reasoning (not conclusory statements), why the claimed invention *as a whole* would have been obvious to a PHOSITA at the relevant time.
- Consider all claim limitations together, not a subset.

This guidance applies to both AIA applications (relevant time = "before the effective filing date") and pre-AIA applications (relevant time = "at the time of invention").

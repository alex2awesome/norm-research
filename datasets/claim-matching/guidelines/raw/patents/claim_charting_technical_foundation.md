# Claim Charting: The Technical Foundation of Patent Litigation (PerspireIP)

SOURCE_URL: https://www.perspireip.com/blog/claim-charting-guide/
DOMAIN: patents

## Overview

Claim charting is the backbone of modern patent litigation. The process systematically maps each element of a patent claim against specific features of an accused product or prior-art reference, creating a visual, element-by-element correspondence — or lack thereof — that stakeholders can evaluate.

## Types of Claim Charts

**Infringement Charts:** demonstrate that an accused product satisfies every claim element, either literally or under the doctrine of equivalents. Prepared by both patent owners and defendants assessing exposure.

**Invalidity Charts:** map claim elements against prior art, showing the invention was disclosed before the patent's priority date. Anticipation charts require a single reference disclosing all elements; obviousness charts spread elements across multiple references with a motivation to combine.

**Claim Construction Charts:** map disputed terms against competing arguments from specification, prosecution history, and evidence.

## Claim Parsing: The Critical First Step

Before mapping begins, patent claims must be carefully parsed into individual limitations. The process requires identifying:

- Preamble language (whether limiting or descriptive)
- Transitional phrases
- Claim body elements
- Means-plus-function language governed by 35 U.S.C. § 112(f)
- Jepson format distinctions

"Claim parsing must resolve these questions explicitly, because the number and identity of claim elements determine the analytical framework for the entire chart."

## Building Infringement Claim Charts

Each element mapping requires:

1. **Specific product feature identification** satisfying the element
2. **Concrete evidence citation** (photographs, reverse-engineering reports, source code, specifications, circuit diagrams, network analysis, screenshots)
3. **Plain-language explanation** of why the feature satisfies the element

"This requirement for explicit reasoning distinguishes litigation-quality claim charts from the superficial charts sometimes submitted in licensing demand letters."

## Building Invalidity Claim Charts

**Anticipation Standards:** prior art must disclose every element "as understood by a person having ordinary skill in the art at the time of the invention." Inherency can supply missing elements but requires careful analysis and expert support — an inherency argument is inherently weaker than an explicit-disclosure match and should be flagged as such in the chart.

**Obviousness Standards:** analysis must address:
- Technical disclosures in cited references
- Motivation for combining references
- Reasonable expectation of success
- Secondary considerations of non-obviousness

Prior-art references must be properly authenticated as publicly available or commercially available before the priority date — a foundational/date-qualification step that precedes the substantive element mapping.

## Key Implementation Facts

- Courts require element-by-element infringement contentions in virtually all U.S. district court patent cases.
- PTAB denies institution in roughly 20–30% of IPR petitions, often citing inadequate claim chart analysis.
- Complex software patent charts can exceed 100 pages.
- Expert declarations add meaningfully to persuasive impact (est. 30–50%).
- Deficient infringement contentions risk being struck by the court.

## PerspireIP's Seven-Step Process

1. Identify the strongest claims based on coverage and validity.
2. Parse each claim into individual limitations with consistent labels.
3. Collect technical documentation and prior art.
4. Map each element to specific features or disclosures.
5. Write detailed correspondence explanations.
6. Coordinate with technical experts for supporting declarations.
7. Conduct attorney review for legal accuracy and cross-element consistency.

## Frequently Asked Questions

**Preparation:** the best charts combine technical analysts' domain expertise with patent attorneys' litigation experience. Technical analysts with deep domain expertise perform the initial element mapping and evidence gathering, while patent attorneys with litigation experience refine the legal analysis, ensure consistency with claim construction positions, and draft the explanatory text. This "division of labor leverages the respective strengths of technical and legal professionals."

**Scope:** claim selection depends on the proceeding stage — IPR petitions can challenge all claims but should prioritize independent claims; district court contentions typically chart the broadest coverage; defendants should prioritize independent claims for invalidity (since invalidating an independent claim generally also invalidates its dependents, while a dependent claim can survive even if the independent claim it depends from is found not infringed).

**Timeline:** moderate-complexity patents require 2–4 weeks; highly complex technologies need 6–8 weeks; expedited versions take 1–2 weeks.

**Errors:** consequences vary by context — district court deficiencies result in striking; PTAB denies institution; licensing negotiations collapse credibility.

**Trial Use:** charts themselves aren't typically admitted as evidence but function as attorney argument. Underlying technical evidence gets admitted; demonstrative charts help juries understand the analysis visually.

## Relevance to Judging Mapping Quality

Read together, the "seven-step process" and the anticipation/obviousness standards give a compact quality rubric for a chart entry:

- A **strong** mapping cites a specific, dated, authenticated piece of evidence, states the corresponding feature/disclosure explicitly, and explains — in plain language, not just claim-language repetition — why the cited evidence satisfies the element.
- A **weak/stretch** mapping either (a) relies on an unauthenticated or undated reference, (b) invokes inherency without expert support, (c) skips the plain-language explanation and only restates claim terms, or (d) is missing an expert declaration/technical citation where the element is non-obvious from the reference's face.
- The **single missing/weakest limitation** in a chart is typically the one where the reference's disclosure is most indirect (inherency-based) or where no reasonable expectation of success / motivation-to-combine rationale has been separately documented for an obviousness combination.

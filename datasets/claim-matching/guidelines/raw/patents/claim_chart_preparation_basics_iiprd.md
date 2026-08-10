# Why Are Claim Charts Required and Basics of How They Are Prepared (IIPRD)

SOURCE_URL: https://www.iiprd.com/why-are-claim-charts-required-and-basics-of-how-they-are-prepared/
DOMAIN: patents

## Introduction

Claim charts function as visual tools in patent law, particularly during enforcement proceedings. They present complex technical information clearly and help courts understand the scope of infringement (or invalidity) by examining each claim detail. "A strong claim chart is supported by expert comments from an industry expert testifying [to] the infringement of claimed features by the infringing product" (or, for invalidity, testifying that a reference discloses the claimed features).

## What is a Claim Chart?

A claim chart uses tabular format to display patented features alongside relevant excerpts from product descriptions (or prior-art disclosures). The goal is to demonstrate how an allegedly infringing product infringes each claimed feature, or how a reference anticipates/renders obvious each claimed feature. Effective preparation requires both technical knowledge and understanding of marketing-related terminology (since accused products and even prior-art datasheets often use marketing language rather than claim-congruent engineering language).

## Step 1: Understanding Patented Technology and Identifying Infringing Products / Candidate References

### Step 1.1: Understanding the Technology

Begin by thoroughly examining the patented technology and interpreting claim scope using detailed descriptions, prosecution history, and extrinsic sources. Focus on "independent claims with the broadest scope of protection" to maximize coverage while remaining consistent with the patent's detailed description — i.e., construe under something like a broadest-reasonable-interpretation approach, but staying tethered to the specification and to what a person skilled in the art would understand, not an unbounded reading.

### Step 1.2: Identifying Application Areas

Review the patent's IPC/CPC classification codes and citation lists to determine relevant industries. This identifies product/reference categories where the patented technology might appear (e.g., mobile phones or smartwatches for inertial-measurement-unit patents).

### Step 1.3: Identifying Target Companies / Candidate Prior Art

Select companies (or references) based on: no existing licensing agreement with the patent owner; significant presence and revenue in the preferred litigation jurisdiction; active sales of products launched after the patent's filing date; products/references containing the specific patented technology feature (for invalidity work, substitute "publication/product predating the priority date" for "launched after the patent's filing date").

## Step 2: Claim Charting Methodology — Evidence-Gathering Techniques

This source's distinctive contribution is a taxonomy of *evidence types* used to fill the mapping column, organized by how directly they disclose the claimed feature:

### 2.1: Direct References

Gather official product documents, datasheets, marketing materials, videos, and related publications from the target company/publisher, verifying source authenticity and publication dates. (For invalidity purposes, date verification is not optional — it is what makes the reference qualify as prior art at all.)

### 2.2: Indirect References

When direct claim-element disclosure isn't found, evaluate whether backend logic implies use of the claimed feature. For instance, if a product advertises secure transactions without explicitly mentioning a specific encryption algorithm, backend analysis might reveal the claimed method is necessarily used to achieve the advertised functionality. This is a weaker, inference-based mapping and should be flagged as such — it typically needs corroborating technical analysis (source code, teardown) rather than resting on the inference alone.

### 2.3: Standard Mapping (SEP claim charts)

For Standard Essential Patents, employ a two-step test: (1) locate documents revealing technical-standard usage/requirements, then (2) map patent claims to demonstrate the patent's essentiality to that standard, and separately show the accused product also complies with the standard. Essentiality mapping (claim-to-standard) and compliance mapping (standard-to-product) are treated as two distinct steps, each independently supportable.

### 2.4: Source Code Review

For computer-implemented inventions claiming algorithms or underlying logic, examine source code to uncover embedded algorithms — tracing "data flow and operations, understanding the decision-making process, and ascertaining how different components interact." This is treated as the highest-confidence evidence type for software claims, since it shows the actual implementation rather than a marketing description of it.

### 2.5: Teardown Reports

For mechanical or apparatus claims, disassemble products to reveal underlying components and connections, enabling comprehensive comparison with patent claims (e.g., analyzing a foldable smartphone's hinge mechanism against a claimed hinge structure).

### 2.6: Reverse Engineering (semiconductor claims)

For semiconductor inventions, reverse engineering deconstructs chips to analyze layered structures, logic gates, and data paths, determining whether the device's structure matches claimed specifications.

### 2.7: Doctrine of Equivalents

When claim elements aren't literally disclosed, infringement may still exist under the doctrine of equivalents if "the accused product or process contain[s] elements identical or equivalent to each claimed element" — requiring analysis of whether the substitute element matches the claimed element's function, way, and result.

## Practical Takeaway for Judging Mapping Strength

This source implies an informal evidence hierarchy for how confidently a mapping can be asserted:

1. **Strongest:** source code / teardown / reverse-engineering evidence directly showing the claimed structure or algorithm.
2. **Strong:** direct, dated, authenticated product documentation or publication explicitly describing the feature.
3. **Weaker:** indirect/backend-logic inference (2.2) or standard-compliance inference (2.3) — these require an explicit secondary showing (that the standard mandates the feature, or that the advertised function could only be achieved via the claimed method) rather than being asserted on their own.
4. **Fallback:** Doctrine of Equivalents, invoked only when literal/explicit disclosure of a specific element is unavailable, and requiring its own function/way/result showing.

A chart entry that relies on evidence lower in this hierarchy without the accompanying corroboration is the practical signature of a "stretch" mapping rather than a genuine match.

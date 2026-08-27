# Patent Litigation Part Six: An Introduction to Patent Claims, "Limitations," Infringement, and Invalidity (DisputeSoft)

SOURCE_URL: https://www.disputesoft.com/patent-litigation-part-six-an-introduction-to-patent-claims-limitations-infringement-and-invalidity/
DOMAIN: patents

## Core Definition of Patents and Claims

A patent functions primarily as "a right to sue for infringement" rather than a self-enforcing property right. The operative portions of any patent are its claims — structured statements that define what the invention encompasses. Claims contain "limitations," discrete elements (for devices) or steps (for methods) selected to distinguish the invention from prior art while clearly defining infringement.

Patent claims serve dual functions: they establish the boundaries for what constitutes infringement and provide the framework for testing whether the claim itself remains valid. Claims act as "devices for testing patent infringement and invalidity."

## Claim Limitations Explained

A limitation represents a single, scope-defining component within a broader claim structure. When analyzing infringement, each limitation must be individually identified within an accused product or service. Critically, a limitation need not use identical terminology to what appears in the accused product — different naming conventions do not prevent infringement if the functional elements match. (The same principle governs whether prior art discloses a limitation for invalidity purposes: what matters is the underlying element, not the label.)

## Claim Construction

Before comparing any claim to an accused product or prior-art reference, claim construction must occur. This interpretive process determines what specific terminology within the claim actually means. Claim construction draws from:

- The claim language itself
- The patent specification
- Patent drawings
- The prosecution history (communications with the Patent Office)
- Relevant technical dictionaries and industry standards

This construction phase typically culminates in a "Markman hearing," where a judge (not a jury) determines the proper meaning of disputed claim terms.

## The Infringement Analysis: Limitation-by-Limitation Comparison

Establishing literal infringement requires demonstrating that "each and every limitation" of an asserted claim appears in the accused product or method. This proceeds through systematic comparison:

**Step 1: Identify Every Limitation.** Divide the claim into individual limitations. While semicolons often mark boundaries, this represents only a starting point — complex claim language may require subdivision into multiple limitations for meaningful analysis.

**Step 2: Map Each Limitation to the Accused Product (or Reference).** For each limitation, identify a corresponding feature, component, or step within the accused instrumentality (or prior-art reference). This mapping must be specific, identifying not merely that something exists, but *where and how* it exists.

**Step 3: Explain the Correspondence.** Simply juxtaposing claim language with product/reference features proves insufficient. The analysis must explain the relationship: how does the feature satisfy the claim limitation's requirements? What functional or structural equivalence exists?

## Claim Charts: The Standard Litigation Tool

Claim charts organize this limitation-by-limitation analysis into a structured comparison format. Most federal district courts with specialized patent rules (Local Patent Rules) require claim charts as mandatory submissions early in litigation.

### Structure of a Claim Chart

**Left Column:** patent claim text with individual limitations listed sequentially, typically with designated reference numbers (e.g., [1a], [1b]). May also include the proper claim construction for each limitation, constraints that other limitations impose, and technical requirements the limitation must satisfy.

**Right Column:** for each corresponding limitation — an explicit assertion that the limitation is (or is not) present; factual support for this assertion; an explanation connecting the facts to the limitation's requirements; specific locations where the limitation is embodied.

### Critical Requirements for Claim Charts

Charts must include "specifically where and how each limitation of each asserted claim is found within each Accused Instrumentality." This imposes four distinct requirements:

1. **Comprehensiveness:** every asserted patent claim must be charted; every limitation within each claim must be addressed; every accused product must be included (representative products may substitute for closely related variants if representativeness can be justified).
2. **Specificity of Location:** the chart must identify specific locations — part names, component designations, source code filenames, or functional locations.
3. **Specificity of Mechanism:** the chart must explain *how* the accused product's feature constitutes an instance of the claim limitation.
4. **Explicit Contention:** rather than vague references like "see, for example," the chart should contain explicit assertions: "The accused product includes [specific component] which embodies the [claim limitation]."

### Common Deficiency: "Aping" the Claim Language

Merely repeating claim language without explanation constitutes inadequate charting. For example, asserting "Product X includes a framis because it has a framis" adds nothing without explaining what constitutes a "framis" under the proper claim construction and how Product X's component satisfies that definition. (This is the clearest single criterion in the source for spotting a "stretch"/weak mapping — the assertion restates the claim term instead of independently establishing it.)

## Doctrine of Equivalents (DoE)

Beyond literal infringement, a limitation may be found present through the Doctrine of Equivalents: a feature that differs from the claim limitation may still satisfy it if it performs substantially the same function, in substantially the same way, to achieve substantially the same result. When asserting DoE, claim charts should explicitly identify which limitations are asserted under literal infringement, which under DoE, and the functional-equivalence reasoning supporting DoE assertions.

## Patent Invalidity Analysis

Patent claims can be invalidated through several mechanisms, each tested against specific limitations.

### Anticipation

A single piece of prior art that discloses every limitation of an asserted claim, exactly as claimed, renders the claim invalid. The analysis mirrors infringement charting: identify each limitation of the claim and demonstrate that a single prior-art reference contains each limitation.

### Obviousness

The invention would have been obvious to a person skilled in the field at the time of invention, based on the combination of multiple prior-art references or knowledge in the field. This analysis requires: identifying the motivation to combine separate prior-art references; establishing that combining them would have been obvious; demonstrating that the resulting combination renders each claim limitation obvious.

### Lack of Enablement

The patent specification fails to provide sufficient detail enabling a person skilled in the field to make and use the full scope of the claimed invention without undue experimentation. Each limitation must be addressed in the specification with adequate guidance.

### On-Sale and Public-Use Bars

If the inventor publicly used or sold the invention more than one year before filing, the patent becomes invalid — this applies to the inventor's own activities, not merely third-party prior art.

## Pre-Filing Investigation Requirements

Before filing an infringement lawsuit, the patent owner must conduct a reasonable investigation into the accused product, based on publicly available information: attempting to acquire and inspect the accused product; reverse engineering (examining externals, disassembling components, analyzing software through decompilation or dynamic analysis); diligently searching for technical documentation, manuals, specifications, and other publicly accessible materials; documenting the investigation to demonstrate "exhaustion" of public sources. This establishes the foundation for "plausible" infringement contentions required under pleading standards.

## Local Patent Rule (LPR) Requirements

**Infringement Contentions** (submitted early, often before discovery): a chart for each asserted claim and each accused product; specific identification of where each limitation appears in the product; explanation of how each limitation is satisfied; identification of claim construction relied upon; for means-plus-function limitations, identification of the corresponding structure in the accused product; for DoE assertions, explanation of functional equivalence.

**Invalidity Contentions** (submitted by defendant): a chart for each asserted claim and each prior-art reference; specific identification of where each limitation appears in the prior art; explanation of how each limitation is disclosed; for obviousness, identification of the motivation to combine multiple references.

**Amendment Restrictions:** once contentions are submitted, amendments require "good cause." Courts examine whether new information was genuinely unavailable despite diligent search, whether the party could have obtained it earlier, and whether the amendment represents a significant shift in theory rather than mere refinement. Diligence is the central inquiry.

## Standards for Infringement and Invalidity

The standard of proof for infringement (and validity challenges) at trial is a preponderance of the evidence — "more likely than not." At the pleading and initial contention stage, only a "plausible" factual allegation is required.

## Key Principles for Effective Analysis

1. **Claim construction precedes comparison.**
2. **Every limitation must be addressed** — a single unaddressed limitation defeats an infringement or invalidity theory.
3. **Specificity matters** — vague references or generic assertions fail to satisfy contention requirements.
4. **Explanation is essential** — juxtaposing evidence with claim language, without connecting explanation, is insufficient.
5. **Consider claim scope interplay** — construing one limitation may affect others.
6. **Document the investigation** — thorough documentation protects against dismissal and supports discovery requests.
7. **Use consistent terminology** — establish and consistently apply designators (e.g., [1a], [1b]) linking specific claim language to specific product/reference features.

## Practical Application Example

For a claim reciting "a web server for transmitting data to client devices":

1. **Establish claim construction:** define "web server" through specification, industry standards, and technical understanding (not merely accept vendor labels).
2. **Identify location in accused product:** specify which component constitutes the web server (e.g., "SERVER.EXE file located in D:\Applications\").
3. **Provide supporting facts:** reference documentation showing the component's function, e.g., "Product manual Section 3.2 describes how the server transmits data; source code reveals HTTP protocol implementation."
4. **Explain the connection:** "The identified component implements the HTTP protocol and responds to client requests by transmitting data in standard web formats, thereby satisfying the construction of 'web server.'"
5. **Distinguish from mere assertion:** avoid stating only "Product X includes a web server" without the explanatory foundation.

This framework — combining claim construction, limitation-by-limitation analysis, structured claim charting, and documented investigation — forms the foundation of modern patent litigation in federal courts, and applies equally (with prior art substituted for the accused product) to invalidity/prior-art claim-chart construction.

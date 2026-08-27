# Patent Claim Charts: Part III - Drafting Essentials (Software Litigation Consulting)

SOURCE_URL: https://www.softwarelitigationconsulting.com/claim-charts-book/claim-charts-book-part-iii/
DOMAIN: patents

## Core Purpose and Non-Holistic Analysis

Claim charts compare patent claim limitations against accused products or prior-art references through disaggregated, element-by-element comparison rather than holistic assessment. "The Name of the Game Is the Claim" — infringement and invalidity turn on specific claim language, not patent abstracts or specifications.

Key principle: limitations are individual components that narrow claim scope. Finding infringement (or anticipation) requires establishing that "each and every limitation of the claim or its equivalent is found" in the accused product or single prior-art reference.

## Chart Structure Basics

**Standard Format:**
- Two-column table (landscape orientation recommended)
- Left-Hand Column (LHC): patent claim text split into limitations
- Right-Hand Column (RHC): evidence from the accused product or prior-art reference
- One row per limitation or sub-limitation
- Each page header should include patent number and claim designation

**Multi-Patent/Multi-Product(Reference) Considerations:**
- Separate charts for each patent (except complex multi-patent cases using "limitations charts")
- Separate charts for each accused product or each prior-art reference (avoid "Frankencharts" mixing elements from different products/references)

## Parsing Claims into Limitations (the LHC)

### Mechanical Division by Punctuation

Patent claims are single sentences typically divided by semicolons marking natural limitation boundaries. Process:

1. Split the claim at semicolon "seams"
2. Split at the initial colon (separating the preamble)
3. Move a final "and" to the previous line
4. Assign designations like [1a], [1b], etc.

**Example Structure:**
```
Claim preamble (often non-limiting)
[1a] First limitation
[1b] Second limitation
[1c] Third limitation
```

### Method Claims

Method claims use gerunds (-ing words) as step indicators: "collecting a sample...", "analysing the emission...", "diagnosing the presence...". Built-in step designators (a), (b), (c) should be retained or cross-referenced to the claim number.

### Handling Lengthy/Complex Limitations

Some limitations are "unwieldy" and benefit from subdivision despite lacking semicolons:

- Create sub-limitation designations like [1c.1], [1c.2] to show subsidiary requirements without claiming them as separate limitations
- Use subheadings in the RHC to break down complex features
- Never assume a single long limitation should occupy one row without acknowledging its internal structure

**Warning signs of excessive length**: limitations spanning multiple clauses with embedded "wherein"/"comprising" language; requirements for specific capabilities, configurations, or calculations; nested conditions or sequential requirements.

### Word-Based Division Points

When semicolons are absent:
- **"Wherein"** often begins a sub-limitation or separate requirement
- **"Comprising:"** introduces component lists within a main limitation
- **"And"** typically divides complete limitations but may indicate subparts
- **"Or"** indicates a choice between alternatives (see below)
- **Commas** sometimes delineate limitations but unreliably (also used for multi-part descriptors) — caution: commas often modify a single element rather than separating limitations

## Deceptively Simple Limitations

### Invented/Non-Art Terms

Limitations using invented nomenclature (the patentee acting as its own lexicographer) require special attention. Examples: "vector prediction candidate," "universal hash value," "facade server."

**Test**: place the term in quotation marks; if hits are only from this patent/patentee, it likely requires construction. Cannot assume identical nomenclature will appear in the accused product or reference — claim construction is necessary to identify what the term actually encompasses.

### Means-Plus-Function Limitations

Governed by 35 U.S.C. § 112(f). Functional language like "means for supplying" must match the specific structure disclosed in the specification, not any structure performing that function.

**Process:**
1. Identify "means for..." language
2. Locate the corresponding structure in the patent specification
3. Add the structure as bullet points in the LHC showing specific disclosed means
4. Comparison must be to the disclosed means, not a generic function

Example (lubricant apparatus): "means for rotating the drum" = Motor 40 (from specification); "means for supplying predetermined volume" = Dosing device 20; "means for controlling distribution" = Distribution holes 34.

## Limitations Requiring Absence (Negative Limitations)

Limitations with negating words require showing absence in the accused product/reference: "without cutting or scattering," "in the absence of," "substantially free of," "no seam."

**Key point**: absence must be shown within the confines of what matches other non-negative limitations. Cannot cherry-pick absence from unrelated product features. Courts may allow de minimis presence (e.g., "without significant interference" construed as "accurate enough for the purposes").

### "Consisting Of" Claims

Closed claiming with "consisting of" requires absence of unspecified elements/steps (opposite of "comprising"). Rare in practice; difficulty of proving absence increases with product/reference complexity.

## Non-Limiting Clauses and Preconditions

- **"Capable of" / "configured to"** are limitations but do not require actual performance: "capable of controllably switching" requires the capability, not that switching actually occur.
- **Distinction from performance**: "receiving quantized data" requires actual receipt; "tank mounted on hull" requires mounting (by anyone, not necessarily the infringer); "programmed microprocessor" requires programming (by anyone); "programmable microprocessor" requires only capability.
- **Loose requirement language**: "associated with," "based on" are permissive; "coupled with," "located on"/"disposed on"/"formed on" are more concrete/tangible requirements.
- **"Whereby" clauses** are generally non-limiting (contrast "wherein," which introduces sub-limitations) — patent law focuses on structure/function, not purpose or desired outcome.
- **Preambles** are typically non-limiting unless they "breathe life and meaning into the claim" (*In re Wertheim*). If a preamble proves limiting, the chart must address it explicitly.

## Limitations Involving Choices

- **"Or" alternatives**: not every alternative must be present — only at least one (e.g., cosmetic "in the form of a serum, a lotion, an emulsion, a cream...or a nail varnish").
- **"At least one of" / "selected from"**: similarly indicates choice, not every listed member need be present.
- **Device-based selection vs. descriptive choice**: a *descriptive* choice ("in the form of X, Y, or Z") requires only one form to be present; an *invention-based* choice (e.g., "controller either adjusts the operation of the pump, or issues a warning...based on a comparison") requires the device to actually perform the selection, and the entire group of options must be available even if the controller selects among them at runtime.

## Filling the Left-Hand Column (LHC)

The LHC serves as a template guiding RHC evidence discovery. Useful additions beyond bare claim text:

- **Means-plus-function bullets**: showing specific disclosed structures for "means for..." limitations
- **Claim construction**: court-ordered constructions as bullets to guide RHC analysis (e.g., "automatically expiring" construed as "records becoming obsolete because of some condition, event, or period of time") — critical: the construction must then be actively used in the RHC, not merely displayed
- **Specification examples and dependent-claim examples**: as temporary scaffolding to guide RHC development (non-limiting examples only; independent claims are not limited to dependent-claim specifics, and "further comprising" dependent claims cannot be used as examples of the independent claim's requirements)

## Filling the Right-Hand Column (RHC) — the core "mapping" craft

Each RHC entry should contain three components:

1. **Assertion** — a statement that the limitation is (or is not) present
2. **Facts** — specific locations and quotations from the accused product or reference
3. **Explanation** — a "because" statement showing why the facts match the limitation

### Deficiency Types to Avoid

| Problem | Description |
|---|---|
| **Bare facts** | Facts without assertion or explanation ("Source code at line 123") |
| **Conclusory** | Language that merely mimics the claim without substantive support |
| **Data dump** | Excessive material without pinpointing to the specific limitation |
| **Merely exemplary** | Bulleted lists with unclear "and/or" logic ("See, e.g.," formulations) |

### Facts in the RHC

Facts must pinpoint where the limitation appears using: name/designation of a part or component; line numbers or coordinate references; filename and function/method names (software); column:line citations (patent references); part numbers or technical manual sections.

**Levels of evidence**: direct (source code, blueprints, schematics, photographs) vs. indirect (technical manuals, deposition testimony, functional specifications, emails, marketing materials, screenshots).

**Standards and third-party material**: avoid assuming full product compliance with a cited standard; don't infer operation merely from standard compatibility.

### Assertions in the RHC — templates

**Infringement**: "[Accused product/entity] embodies this limitation in [specific element/location/code], in the following [structure/process/algorithm/feature/function]."

**Invalidity**: "Prior-art reference [X] at [citation] discloses this limitation at [location]." Alternative: "...discloses, teaches, or suggests..."

**Non-infringement/non-anticipation**: "Accused product X does not embody, practice, or otherwise contain an element matching this limitation, because [specific reason]." Or: "Product/reference contains Y, but Y does not match the limitation because [specific reason]."

### Explanations in the RHC — the "because" formulation

The word "because" practically forces non-conclusory analysis and prevents mimicking claim language.

**Template**: "[Accused product/reference] at/in [location/nomenclature] embodies/practices/discloses [limitation nomenclature] because [explanation of why the product feature/reference disclosure = claim limitation]."

**Adequate example**: "D's product X123 in LocalServer.c and AppServer.c modules embodies the claimed facade server, because a facade server is a web server hosting local applications, and these modules use web services such as HTML and local CGI to support local applications (not network-based)."

**Inadequate example**: "D's product includes a facade server" (merely mimics the claim — this is the archetypal weak/stretch mapping).

### Structuring Comparisons — attributes to compare

Break each limitation into its constituent attributes for comparison:
- **Function** — purpose or role of the component
- **Way** — implementation or mechanism
- **Result** — output or effect
- **Input** — what feeds into the component
- **Structure** — physical/logical makeup

For compound limitations, decompose further. Example: "The program creates an interface between the facade server and web-browser for exchanging data associated with the application" breaks into: (1) facade server presence, (2) web browser presence, (3) interface between them, (4) interface purpose (exchanging data), (5) data's association with the application. Each sub-piece needs its own factual support — a mapping that only nails 3 of 5 sub-pieces is a partial/weak match on that limitation.

**Means-plus-function structure**: comparison must align accused/reference structures with the disclosed means, not generic functional equivalents — e.g., evidence must show Motor 40 (the disclosed means) present and used, not merely any rotating mechanism.

## Multi-Reference Obviousness Charts

For obviousness, separate a limitation into subheadings by reference:

- **Limitation [1a] met by:**
  - **Reference 1 at [cite]:** discloses first part
  - **Reference 2 at [cite]:** discloses second part
  - **Motivation to combine:** [reasoning]

This structure makes explicit which reference supplies which part of the limitation, and isolates the "missing element" that the primary reference alone does not disclose — the gap that the secondary reference(s) and the combination rationale must fill.

## Common Deficiencies and Corrections

- **Nomenclature mismatch**: patent uses "facade server," product/reference uses "local app server" — not fatal by itself. Comparison is on *characteristics, not names*: identify the relevant feature/structure, identify the claim requirement, explain how the feature satisfies the requirement despite different nomenclature, using the "because" formulation.
- **Missing sub-limitations**: common oversight — addressing only obvious main components while missing qualifying descriptors (e.g., overlooking that "interface for exchanging data associated with the application" requires the data specifically be *associated with the application*).
- **Affirmed negative limitations**: for "without X" limitations, show that X is genuinely absent in the relevant accused feature/reference disclosure (not just absent from the product/reference generally), allowing for de minimis presence per claim construction.

## All-Elements Rule Application

Infringement/invalidity require **every** limitation present (possibly via equivalents or combinations). Therefore: include even "trivial" limitations ("a CPU," "a network"); chart every limitation separately; show interconnections among limitations (not just a parts list); for "consisting of" claims, affirmatively show absence of unspecified elements.

## Summary Checklist for an Adequate RHC (mapping quality standard)

- Does the RHC contain an **assertion** of whether the limitation is met?
- Does the RHC contain **facts** with specific locations in the accused product/reference?
- Does the RHC contain an **explanation** using "because" to connect facts to the assertion?
- Is the explanation structured around the limitation's constituent attributes?
- Is the comparison limited to **only** this limitation (avoid data dumps)?
- Are facts **pinpointed** to specific lines/sections (not general references)?
- Are conclusions **non-conclusory** (not mimicking claim language)?
- For choices/selections, is it clear whether **all** or **any one** is required?
- For negative limitations, is absence shown **within the relevant context**?
- For dependent claims, are they used only as **non-limiting examples**?

This checklist is effectively the practitioner standard for distinguishing a strong (assertion + pinpointed fact + causal "because" explanation, correctly scoped to the construed limitation) chart entry from a weak/stretch one (bare facts, conclusory mimicry, data dumps, or inferences from naming/output alone).

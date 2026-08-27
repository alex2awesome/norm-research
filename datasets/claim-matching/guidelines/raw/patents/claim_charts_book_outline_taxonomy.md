# Claim Charts for Patent Litigation: Comprehensive Outline (Software Litigation Consulting)

SOURCE_URL: https://www.softwarelitigationconsulting.com/claim-charts-book/claim-charts-book-outline/
DOMAIN: patents

## Core Definition & Purpose

A claim chart "breaks patent asserted claims into elements or steps, and compares each element or step with an assertedly corresponding element or step found in an accused product (instrumentality) or piece of asserted prior art."

The fundamental structure maps:
- **Left column**: Patent claim language disaggregated into limitations
- **Right column**: Facts showing where/how each limitation appears in external reality (accused product or prior-art reference)

## Essential Requirements ("Each x3")

Charts must identify "specifically and in detail where each element of each asserted claim is found within each accused instrumentality" (the same standard applies, mutatis mutandis, to invalidity charts against prior art). This means:

1. Each claim limitation requires explicit treatment
2. Each asserted claim requires separate analysis
3. Each accused product/reference requires mapping

No elements may be "NOPed" (dropped via "not operating present"), per the "all elements rule" — a single unaddressed or unmatched limitation defeats the theory (whether infringement or invalidity/anticipation).

## Location Specificity Standards

"Specifically where" mandates factual equivalents to legal "pinpoint" citations. Reviewing courts/panels reject:
- Mere assertions ("it's in there")
- Inferred presence without evidence
- Vague references ("See document X")

Acceptable pinpointing examples: source code line numbers, filenames with pathnames, product version numbers, functional component names with brief descriptions, and — for prior art — column:line citations, figure numbers, and specific passages of a reference document.

## Claim Disaggregation Approach

**Starting point**: Semicolons typically delimit major limitations, but only as a "first approximation."

**Identifying all limitations requires examining**:
- Explicit clauses beginning with "wherein"
- Preambles (often limiting despite conventional wisdom that they are not)
- Preconditions and "whereby" clauses
- Implicit sub-limitations emerging from claim structure
- Negative limitations ("without," "substantially free of")
- Functional language under 35 U.S.C. § 112(f) (means-plus-function)

**Structural analysis**: Identify the attributes/facets along which parts can be compared (e.g., function/way/result, used originally for the Doctrine of Equivalents but useful more generally for judging whether a mapping is a genuine match).

## Right Column Content Requirements

Facts presented in the mapping column must:
1. **Assert explicitly** what connection exists between right-column facts and left-column limitations
2. **Explain "why"** the facts support the assertion (not proof, but a reasoned basis)
3. **Interweave** fact language with construed limitation language (not mere side-by-side juxtaposition)
4. **Isolate presence** of each limitation within the accused instrumentality/reference — avoid "dumping" pages of undifferentiated material
5. **Avoid assumptions** about operation from external appearance or output alone (the "inventor's fallacy")

## Evidence Requirements by Litigation Stage

**Pre-Discovery (Preliminary Infringement Contentions / initial invalidity contentions)**:
- Must reflect "reverse engineering or its equivalent"
- Requires exhaustion of publicly accessible information
- May invoke "information and belief" only after diligent investigation
- Should identify what information remains unavailable despite diligence

**Post-Discovery (final Infringement/Invalidity Contentions)**:
- Enhanced specificity and granularity
- Source code citations with pathname, filename, line numbers or function names
- Detailed explanation of why code structure (or reference disclosure) satisfies claim limitations
- Claim construction explicitly applied in the fact comparison

## Problematic Chart Patterns to Avoid (criteria for judging a genuine match vs. a stretch)

These failure taxonomies double as diagnostic criteria for whether a chart entry is a *strong* match or a *weak/stretch* entry:

- **"Mimicking" charts**: Merely repeat claim language; fail to describe the accused instrumentality (or reference) in its own terms and nomenclature. (Weak — no independent factual content.)
- **"Laundry list" charts**: Present disconnected parts/elements without showing how they interoperate as required by the claim structure. (Weak — elements present individually but not shown combined as claimed.)
- **"Frankencharts"**: Assemble one limitation from one product mode/embodiment and another limitation from an incompatible mode — creating a non-functional combination that was never actually practiced or disclosed as a unified whole. (Fatal flaw, especially for anticipation, where a single reference must disclose the elements as an integrated, operable whole, not stitched from separate teachings.)
- **"Dumping" charts**: Juxtapose claim language with massive undifferentiated source-code or document excerpts lacking pinpoint citations. (Weak — burden-shifting, not analysis.)
- **"Information and belief" overreliance**: Assert facts as "I&B" without explaining what reasonable investigation was conducted or why it proved insufficient. (Weak unless diligence is documented.)
- **"Inventor's fallacy"**: Infer internal operation/structure from product output or naming conventions without examination of the actual implementation. (Weak — a stretch, since the surface label does not establish the underlying structure/function required by the limitation.)

A **strong** chart entry, by contrast, cites a specific location, quotes/paraphrases the actual disclosure, and explains — using the construed claim term — why that specific disclosure satisfies (not just resembles) the limitation.

## Claim Construction Integration

Claim charts must employ the construed claim language, not merely unconstrued statutory text. This means:

- Identify the features/attributes the construction emphasizes
- Compare each construed-limitation attribute against the corresponding accused-instrumentality/reference attribute
- Use the construction to structure the comparison, not as an afterthought
- For pre-Markman charts, adopt some reasonable provisional construction (not mere conclusory statements)
- Post-Markman, adjust charts to reflect the court's construction while identifying any limitation's presence under multiple construction alternatives (useful when the construction itself is still contested)

## Doctrine of Equivalents Charting

When asserting DoE, charts must explain, per limitation:

**Function**: The role/purpose the limitation performs in the system
**Way**: The implementation/means/mechanism employed
**Result**: The output/effect achieved

Beyond this "F/W/R" framework, charts may reference additional equivalence factors: known interchangeability, evidence of copying versus design-around, industry practice.

Critical: If literal infringement exists, DoE would also apply by definition; F/W/R analysis therefore helps validate whether literal infringement itself exists, and — by extension — whether an invalidity mapping is a true anticipation match versus merely an equivalent-but-not-identical disclosure (relevant because inherency/anticipation has stricter identity requirements than DoE).

## Means-Plus-Function Limitations (§ 112(f))

Charts must identify "structure(s), act(s), or material(s) in the Accused Instrumentality [or prior-art reference] that performs the claimed function." This differs from general practice because:

- Infringement/anticipation requires matching not just the function but the *specific structure/act/material disclosed in the patent specification* (or its equivalent)
- Merely performing the claimed function is insufficient — this is a classic "stretch" mapping error
- Charts must show how the accused/reference structure matches or is equivalent to the specification disclosure
- Functional claim language marked by "means for" requires this analysis

## Multiple Claims & Products/References Strategy

**Dependent claims**: The chart must separately address additional limitations not present in the base (independent) claim. If the independent claim is invalid, the dependent claims charted against it are also invalid (for anticipation/obviousness purposes) — but each dependent limitation still needs its own row and its own mapped disclosure.

**Multiple independent claims**: Each requires separate treatment even with identical limitations, since different claim types (apparatus/method/media) may implicate different infringers/invalidity theories.

**Representative instrumentalities/references**: Permitted when charting similar product variants, but require a separate showing explaining why the charted product is representative of uncharted products. Each "class" of products should have at least one charted example.

**Avoid**: "Copy & paste" solutions that create repetitive charts. Instead, use cross-references with explicit explanatory language connecting limitations across claims.

## Pre-Filing Investigation Standards

Preliminary contentions must demonstrate "reverse engineering or its equivalent," including:

- **Acquiring the product/reference**: Obtaining and operating the to-be-accused instrumentality (or obtaining the prior-art document/product) through legitimate means
- **Static analysis**: Examining structure, documentation, specifications, drawings, schematics
- **Dynamic analysis**: Operating the product with instrumentation to observe functional behavior (packet sniffers, runtime monitoring)
- **Public sources**: Mining publicly available technical documentation, standards-compliance statements, marketed features

**Exceptions to reverse-engineering requirement**: information genuinely unavailable (hidden manufacturing processes, DRM-protected software), technical inaccessibility, disproportionate expense, or information demonstrably available from indirect sources.

**Non-exceptions**: mere desire to avoid effort, or waiting for discovery to obtain information that should be publicly accessible.

## Amending Claim Charts

**Good cause** requires showing:
- Newly discovered non-public information despite earlier diligent investigation
- Recent product launches post-filing
- Information from a consultant/expert previously unavailable
- Materials produced in discovery not earlier publicly available
- For prior art specifically: material genuinely newly discovered (a less flexible standard than for infringement contentions, since prior art should have been publicly accessible all along)

**Diligence required**: chart amendments based on discovery materials assume the party exhausted mining those materials within discovery deadlines; delayed amendments suggest inadequate diligence.

**Prejudice analysis**: Does the amendment require the responding party to conduct new investigation or expert analysis? Does it significantly alter case theory?

## Common Failure Patterns (summary checklist)

Charts fail sufficiency standards when they:
- Present "conclusory" assertions without factual support or explanation
- Omit entire limitations (particularly trivial-seeming elements like generic processors)
- Miss entire accused products/references while claiming others are "representative"
- Fail to pinpoint location within multi-page source-code or reference excerpts
- Compare facts that do not correspond to the claim type (e.g., asserting method infringement/anticipation without identifying who/what performs each step)
- Use outdated or forward-looking documents without clarifying timing relevance (critical for prior-art date qualification)
- Rely on marketing materials instead of technical examination
- Apply an unreasonable claim construction implied through the limitation mappings

## Responding to the Other Side's Charts

Defensive/rebuttal options include:
- **Deficiency charts**: Demonstrating missing elements, products, or theories through organized tables
- **Expert declarations**: Highlighting factual inaccuracies without requiring chart revision
- **Rebuttal charts**: Presenting contrary facts with the same element-by-element structure
- **Deposition use**: Mining charts for party admissions and implied claim constructions
- **Offensive mining**: Extracting implied constructions or omitted theories for use in one's own arguments

## Key Underlying Policies

Mandatory early claim charts serve to:
1. **Avoid "shifting sands"**: Lock in case theories pre-discovery/pre-Markman to prevent radical post-ruling repositioning
2. **Prevent "Whack-a-Mole"**: Eliminate the pattern of multiple uncommitted theories forcing excessive opposing-party investigation
3. **Replace interrogatories**: Eliminate the need for contention interrogatories traditionally deferred post-discovery
4. **Narrow discovery**: Early theory specification focuses discovery requests on genuinely disputed issues
5. **Set a plausibility threshold**: Establish a reasonable prospect of proof ("something more than labels and conclusions") without achieving the ultimate proof standard

Charts are "not intended to prove a party's case, nor as an arena for substantive disputes" — yet inadequate charts can result in summary judgment for failure of proof, or striking/dismissal of the underlying contention.

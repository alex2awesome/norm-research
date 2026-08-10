# MPEP 2143 — Examples of Basic Requirements of a Prima Facie Case of Obviousness
SOURCE_URL: https://www.uspto.gov/web/offices/pac/mpep/s2143.html
DOMAIN: patents

## Overview

MPEP 2143 operationalizes the *KSR* rationales into concrete, examiner-usable tests. For each of the seven rationales, the Office must make a specific set of factual findings (in addition to the general Graham findings on scope of prior art, differences, and level of skill) before concluding a claim is obvious. The unifying requirement across all seven: **the analysis must be explicit — conclusory statements ("it would have been obvious to combine A and B") are insufficient.** Office personnel must articulate *why* a person having ordinary skill in the art (PHOSITA) would have made the combination/substitution/modification, and *why the result would have been predictable*.

This section is the core doctrinal tool for claim-matching against combined prior art: it defines what counts as a legally sufficient mapping of claim limitations onto one or more references.

---

## Rationale A — Combining Prior Art Elements According to Known Methods to Yield Predictable Results

**Required findings:**
1. All claimed elements are found in the prior art (not necessarily in a single reference).
2. A PHOSITA could have combined the elements using known methods.
3. Each element, in combination, retains its separate known function — the results are predictable (not synergistic/unexpected).
4. Any additional Graham findings needed.

**Illustrative contrasts:**
- *Anderson's-Black Rock v. Pavement Salvage Co.*: mounting a known radiant-heat burner onto a paving machine chassis alongside known spreading/shaping equipment was obvious — the burner performed its known function independent of the rest, producing only predictable convenience, not a new function.
- *United States v. Adams*: combining known magnesium and cuprous-chloride battery electrodes with water activation was **not** obvious, because prior art actively taught that such combinations were impractical (i.e., the art taught away) — a case where combining "known" elements still failed because the result was contrary to what the art predicted.
- *Crocs, Inc. v. ITC*: combining a known foam base with known elastic straps was **not** obvious — prior art counseled *against* foam straps (they stretch/deform), so the resulting comfortable, low-friction fit was an unpredictable, superior result, not a mere aggregation of known functions.
- Key takeaway for claim-matching: merely finding every claim element somewhere in the prior art is *not* enough to complete a Rationale A rejection. There must also be a finding that the combined result would have been predictable — if the art teaches away, or if the combination produces an unexpected/synergistic effect, Rationale A fails.

## Rationale B — Simple Substitution of One Known Element for Another to Obtain Predictable Results

**Required findings:**
1. A prior art device differs from the claim only by substitution of a component.
2. The substituted component, and its function, was known.
3. A PHOSITA could have performed the substitution with predictable results.

**Examples:** *In re Fout* (substituting evaporative distillation for aqueous extraction in decaffeination — obvious because both were known caffeine-from-oil separation methods; an express suggestion to substitute is not required); *Agrizap v. Woodstream* (substituting a resistive electrical switch for a mechanical pressure switch in pest-control devices — a "textbook case" of predictable substitution, reinforced because both switch types were known to solve the identical dirt/dampness malfunction problem, even though the resistive switch was known from a *different* device — hand-held stunners/cattle prods — illustrating that analogous art need not come from the same specific product category).

Counter-examples where substitution rationale **failed**: *Eisai v. Dr. Reddy's* — swapping one substituent for a structurally similar one was not obvious because the prior art gave reasons to *expect worse* properties from the swap (destroying an advantageous property), so there was no reasonable expectation of a predictable, acceptable result.

## Rationale C — Use of a Known Technique to Improve Similar Devices (Methods, or Products) in the Same Way

**Required findings:** identification of a "base" device the claim improves, a "comparable" (not identical) device improved by the *same* technique in the prior art, and a finding that a PHOSITA could apply that known improvement technique to the base device with predictable results.

Example: *In re Nilssen* — using a known cutoff-switch technique (known for protecting one type of inverter circuit) to protect a different but comparable inverter circuit was obvious, because the technique's function (disabling on overload) transferred predictably.

## Rationale D — Applying a Known Technique to a Known Device (Method, or Product) Ready for Improvement to Yield Predictable Results

**Required findings:** a known base device/method, a known technique applicable to it, and a finding that a PHOSITA would have recognized applying the known technique would yield predictable, improved results.

Example: *Dann v. Johnston* — adding category codes (a known indexing technique) to standard bank check-processing computer systems (a device "ready" to support finer-grained reporting) was obvious; the "gap" between the automated bank record-keeping system and the claimed categorized-reporting system was not large enough to be nonobvious. Also *In re Urbanski* — shortening a known enzymatic reaction time within a range taught by a second reference was obvious because both references recognized reaction time as a "result-effective variable" and neither taught away from shortening it.

## Rationale E — "Obvious to Try": Choosing From a Finite Number of Identified, Predictable Solutions, With a Reasonable Expectation of Success

**Required findings:**
1. A recognized problem or need existed in the art (potentially design need or market pressure).
2. A **finite** number of identified, predictable potential solutions existed.
3. A PHOSITA could have pursued those known options with a reasonable expectation of success.

This is the most litigated rationale and matters heavily for chemical/biotech claim-matching, but it applies across arts. *KSR* itself framed it: "When there is a design need or market pressure to solve a problem and there are a finite number of identified, predictable solutions, a person of ordinary skill has good reason to pursue the known options within his or her technical grasp... [T]he fact that a combination was obvious to try might show that it was obvious under § 103."

- **Where it succeeds:** *Pfizer v. Apotex* (amlodipine besylate) — only 53 pharmaceutically acceptable anions existed to try for a known "stickiness" manufacturing problem, a small enough finite set with reasonable expectation of success. *In re Kubin* — a limited, routine set of cloning/sequencing techniques applied to an already-identified target protein.
- **Where it fails:** *In re O'Farrell* identified two failure patterns still controlling post-*KSR*: (1) "obvious to try" is improperly invoked when it means varying *all* parameters or trying numerous possibilities with no direction on which are likely to succeed; (2) it is improperly invoked when it means merely exploring a new/promising *general approach* with only general guidance toward the particular claimed form. *Takeda v. Alphapharm* (pioglitazone) — obviousness failed because prior art disclosed hundreds of millions of candidate compounds generally, gave no reason to select the specific "lead compound" starting point, and that lead compound had a known disadvantageous property directing skilled artisans *away* from it. *Ortho-McNeil v. Mylan* (topiramate) — no finite, easily-traversed option set existed connecting the anti-diabetic synthesis pathway to an anti-convulsant use; the discovery was serendipitous, not obvious to try.
- **Takeaway for claim-matching:** the delta between a claim and prior art is only "obvious to try" if the solution space was small, identified, and predictable — a broad or open-ended solution space, or a disadvantageous/discouraged starting point, defeats the rationale even when *some* prior art disclosure exists.

## Rationale F — Known Work in One Field of Endeavor Prompting Variations for Use in the Same or a Different Field Based on Design Incentives or Market Forces, if Predictable

This rationale supports combining across fields when market/design pressure (not necessarily an explicit textual suggestion) would have prompted the variation, and the resulting variation is predictable to a PHOSITA. It overlaps with the analogous-art "reasonably pertinent" prong (MPEP 2141.01) and is frequently invoked alongside Rationale B/C — e.g., *Leapfrog Enterprises v. Fisher-Price* (applying modern electronics to prior-art mechanical learning devices was prompted by predictable market/design pressure to modernize).

## Rationale G — Some Teaching, Suggestion, or Motivation (TSM) in the Prior Art

This preserves the pre-*KSR* TSM approach as **one valid rationale among seven**, not the exclusive test. An examiner may still point to an explicit or implicit suggestion in the prior art itself to combine or modify references. The suggestion must be specific enough to actually point a PHOSITA toward the claimed combination with a reasonable expectation of success — general encouragement to explore a research area is not equivalent to a specific direction.

## Cross-Cutting Lessons for Claim-vs-Prior-Art Delta Analysis

1. **All elements present ≠ automatically obvious.** Under every rationale, presence of all claim elements somewhere in the prior art is necessary but not sufficient; predictability of the combined/substituted/modified result is an independent, separately-required finding.
2. **Teaching away defeats every rationale.** If the closest reference(s) criticize, discourage, or warn against the specific modification needed to bridge the claim-to-art delta, no rationale can support obviousness on that record (*Adams*, *Crocs*, *DePuy Spine*).
3. **"Ready for improvement" and "recognized problem" require evidence**, not assumption — a device or process is only "ready for improvement" (Rationale D) or facing a "recognized problem" (Rationale E) if the record shows a PHOSITA would have perceived it that way at the relevant time.
4. **Structural similarity alone (chemical/biotech cases) is not obviousness** — there must additionally be a reason to select the particular "lead compound" and a reasonable expectation that modifying it would preserve or improve (not destroy) the relevant properties (*Eisai*, *Altana*, *Procter & Gamble v. Teva*).
5. **The list of rationales is non-exhaustive**; other rationales are permitted as long as the examiner ties factual findings to the ultimate legal conclusion with a "rational underpinning."

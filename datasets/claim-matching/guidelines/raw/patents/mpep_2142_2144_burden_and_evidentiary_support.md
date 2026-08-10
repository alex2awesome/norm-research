# MPEP 2142 & 2144 — Legal Concept of Prima Facie Obviousness and Supporting a Rejection Under 35 U.S.C. 103
SOURCE_URL: https://www.uspto.gov/web/offices/pac/mpep/s2142.html
DOMAIN: patents

(See also MPEP 2144, https://www.uspto.gov/web/offices/pac/mpep/s2144.html, for the evidentiary-support rules combined into this file.)

## MPEP 2142 — The Burden-Shifting Framework

Obviousness examination is structured as an allocation of evidentiary burdens between examiner and applicant, not a single up-front substantive judgment:

- **Examiner's initial burden.** The examiner must first set forth a *prima facie* case, supported by evidence/reasoning, showing why the claims would have been obvious in light of the prior art (*ACCO Brands Corp. v. Fellowes, Inc.*). If the examiner does not establish this, the applicant has **no obligation** to submit rebuttal evidence.
- **Burden shift.** Once a prima facie case is established, the burden shifts to the applicant to come forward with evidence or argument sufficient to rebut it.
- **Final determination.** The examiner then decides patentability by weighing the prima facie evidence against the rebuttal evidence, under a preponderance standard, considering the **totality of the record** — not the prima facie case in isolation.

**What the examiner must specifically show** when rejecting a claim under § 103:
- The relevant prior art teachings, with specific citations (column/page/line) to the reference(s).
- The differences between the claim and the applied reference(s).
- Any modifications to the reference(s) that would be necessary to arrive at the claimed subject matter.
- An explanation of *why* a PHOSITA would have found the invention, modified in that way, obvious.

Per *Ex parte Clapp*, the references must "expressly or impliedly suggest the claimed invention" or the examiner must otherwise "present a convincing line of reasoning" — a bare citation of references without explanation does not satisfy the prima facie burden.

## MPEP 2144 — Supporting a Rejection: Sources of Rationale

The rationale for combining or modifying references **need not be expressly stated in the references themselves**. It may instead come from:
- Implicit teachings in the prior art;
- Knowledge generally available to a PHOSITA (common knowledge);
- Established scientific/technical principles; or
- Legal precedent, when the facts of a controlling case are sufficiently analogous.

The strongest form of rationale is a showing — express or implied — that the combination would produce "some advantage or expected beneficial result." A PHOSITA need not have been targeting the *same* advantage the applicant discovered, and need not have been trying to solve the *identical* problem the specification describes; a different motivating purpose is permissible as long as it would still have led to the claimed combination.

### 2144.01 — Implicit Disclosure
A reference is read to include what a PHOSITA would reasonably infer from it, not only its literal words — e.g., disclosure of a process at one operating point implies feasibility at nearby points within a range the reference itself acknowledges.

### 2144.02 — Reliance on Scientific Theory
Examiners may invoke "logic and sound scientific principle" as part of the rationale, but if a scientific theory is relied upon, the examiner must support that the theory exists and means what the examiner says it means (i.e., theory-based reasoning still needs evidentiary grounding, not just assertion).

### 2144.03 — Common Knowledge and Official Notice
Official notice of a fact **without documentary evidentiary support** is permitted only where the fact is "capable of such instant and unquestionable demonstration as to defy dispute" — i.e., notorious, beyond genuine dispute, and used to fill a narrow gap (appropriate mainly in first actions, not final rejections). Esoteric or technical assertions must cite a recognized reference work. If an applicant properly and specifically traverses an official-notice assertion (a bare request for evidence does not count as traversal), the examiner must supply documentary evidence in the next action or drop the point. **It is never proper to rely solely on unsupported common knowledge as the principal evidence for a rejection** — a decision-maker cannot substitute personal understanding/experience for concrete record evidence.

### 2144.04 — Legal Precedent as Rationale
Precedent may be invoked, but the examiner must explain how the cited case's facts map onto the application at hand; if the applicant shows the specific claimed feature is critical (unlike in the precedent), reliance on the precedent alone is improper. MPEP 2144.04 catalogs several modifications *routinely* (not per-se, but commonly) found obvious absent a showing of criticality or unexpected results:
- Purely aesthetic/ornamental changes with no mechanical function;
- Omitting an element whose function is no longer needed (though omitting an element while *keeping* its function is not obvious);
- Automating a previously manual activity to the same result;
- Changes in size/proportion, changes in shape (ordinary design choice), or reversal of process-step sequence (absent new/unexpected results);
- Making something portable, integral, separable, adjustable, or continuous;
- Reversing movement direction, duplicating parts, or rearranging part positions;
- Purifying a known product — pure forms *can* be patentable, but purity alone is not automatically nonobvious; it depends on whether the pure form shares the same utility as the known impure material and whether the art suggests the particular purified form or a route to it.

### 2144.05 — Overlapping and Similar Ranges
When a claimed numeric range overlaps, lies within, or closely approaches a prior-art range, this establishes a **prima facie case of obviousness**, even where the ranges are disclosed across multiple references rather than one. Rebuttal requires either (a) **criticality** — evidence of unexpected results specifically tied to the claimed range versus the prior art range, (b) **teaching away** from the claimed range, (c) showing the parameter was a **non-result-effective variable** not recognized by the art as affecting the relevant property, or (d) showing the prior art range is so broad it does not actually invite optimization toward the claimed sub-range. "Routine optimization" rejections still require the examiner to articulate specific fact-findings for why optimization would have been routine and why success was reasonably expected — it is not automatic just because ranges overlap.

### 2144.06 — Art-Recognized Equivalence
Substituting or combining known equivalents — where the *prior art itself* (not merely the applicant's own disclosure) recognizes them as interchangeable for the same purpose — supports a prima facie obviousness case.

### 2144.07 — Art-Recognized Suitability
Selecting a known material because the prior art discloses characteristics making it suitable for the claimed use supports obviousness.

### 2144.08 — Species Within a Disclosed Genus
A claimed species falling within a broader prior-art genus is **not automatically obvious**; the examiner must still separately find, per Graham: the genus's structure/properties/predictability; the specific similarities/differences between the closest disclosed species and the claimed species; the level of ordinary skill; and — critically — a **reason to select** the claimed species specifically (genus size alone cannot support the rejection; but express teachings of a "preferred"/"typical"/"optimum" species, similar properties/uses, and predictability in the art can).

### 2144.09 — Close Structural Similarity
Chemical compounds with very close structural similarity and similar utility support a prima facie obviousness finding on the expectation that structurally similar compounds share similar properties — but this requires only a *reasonable* expectation, not absolute predictability, and evidence of genuine unpredictability in the art rebuts it.

## Overarching Rule

No per-se rules exist — every one of the "routine" categories above is a starting heuristic, not an automatic disposition, and requires case-specific articulation. Scientific theories, common knowledge, and technical assertions require documentary or record support; once a prima facie case is properly built this way, the burden shifts to the applicant to rebut it with evidence or argument.

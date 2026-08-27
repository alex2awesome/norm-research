# How Examiners Map Claim Limitations to Combined Prior Art References — Practitioner Guidance
SOURCE_URL: https://patentlawyer.io/mpep-2143-obviousness-rejections/
DOMAIN: patents

(This file also draws on MPEP 2143.03 "All Claim Limitations Must Be Considered," https://www.bitlaw.com/source/mpep/2143-03.html, and the practitioner explainer at https://www.patenttrademarkblog.com/prior-art-office-action/ on analyzing prior art cited in office actions.)

## Why This Matters for Claim-Matching

MPEP 2143's seven rationales (see companion file `mpep_2143_exemplary_rationales_obviousness.md`) are the formal legal tests, but in practice an obviousness rejection is built and attacked through a specific, repeatable mechanical process: map every claim limitation to a specific disclosure in a specific reference, articulate why a PHOSITA would combine those references, and confirm the result was predictable. This file summarizes how that mapping is actually done and where it typically breaks down.

## Step 1: Determine Whether the Rejection Is Anticipation or Obviousness

- **35 U.S.C. 102 (anticipation):** a single prior art reference must show **every** limitation of the rejected claim, arranged as claimed. If the Office Action treats a reference as "anticipating," it is claiming the whole claim is disclosed in one document.
- **35 U.S.C. 103 (obviousness):** used when no single reference discloses everything. The examiner combines multiple references, with **different limitations mapped to different references** — e.g., for a claim reciting elements A, B, and C, Reference 1 might be cited for showing A and B, and Reference 2 cited only for showing C.

**Practical consequence:** if a claim is rejected as obvious over References 1+2, an argument that "Reference 2 doesn't teach A or B" is non-responsive, because the examiner never relied on Reference 2 for A or B. Effective rebuttal must attack the specific element each reference was actually cited for, or attack the motivation to combine.

## Step 2: All Claim Limitations Must Be Given Patentable Weight (MPEP 2143.03)

The bedrock rule, from *In re Wilson* and reaffirmed in cases like *In re Gulack*: "All words in a claim must be considered in judging the patentability of that claim against the prior art." An examiner cannot silently drop a limitation to make the mapping work.

- Claim scope is defined by whichever terms actually limit it, construed broadly but reasonably. Courts have reversed examiners (e.g., *Axonics, Inc. v. Medtronic, Inc.*) for improperly narrowing prior art searches to match an unstated assumption (there, restricting prior art to sacral-nerve anatomy when the claim language contained no such restriction) — the opposite error of ignoring a limitation, but the same underlying principle that claim language, not assumption, controls scope.
- Language that makes a feature optional does not limit scope; where a claim recites a choice among alternatives, prior art showing **any one** of the alternatives satisfies that limitation.
- Categories that commonly raise limiting-effect disputes: preambles, "adapted to"/"whereby" clauses, contingent limitations, printed matter, and purely functional language. Each requires a case-specific determination of whether it actually narrows the claim or is merely descriptive/intended-use language.
- Indefinite claim language cannot simply be ignored; where multiple reasonable interpretations exist, best practice is to reject under both 112(b) (indefiniteness) and 103 (using the broadest reasonable, patentability-defeating interpretation) rather than picking one interpretation silently.
- Limitations that lack support in the specification are **still** given weight in the prior-art comparison (they just may separately raise a written description issue) — a limitation is not disregarded merely because it wasn't well-supported when written.

## Step 3: Evaluate Whether the Examiner's Claim Construction Is Correct

A frequent failure mode is that examiners adopt an overly broad reading of a claim term and then locate *that* broad reading somewhere in a reference. The correct response is not to argue in the abstract that the art is "different," but to:
1. Precisely define what the claim term means (via remarks, or by tightening claim language if needed).
2. Show that the reference in fact discloses something different — call it Y, not X.
3. Explain concretely why Y cannot reasonably be read as X under the correct construction.

## Step 4: Verify Every Element Is Actually Mapped — and Nothing Extra Is Smuggled In

- Confirm each claim limitation has a specific citation (column/page/line) in some reference in the combination. A rejection lacking a citation for even one limitation is deficient (see MPEP 2142 — the examiner must set forth specific prior-art teachings tied to specific claim differences).
- Watch for the reverse error: applicants sometimes argue a distinguishing feature that sounds compelling but was **never actually recited in the claim**. Even a genuinely distinguishing characteristic is legally irrelevant if the claim language doesn't capture it — the fix is to amend the claim to recite the feature, not to argue around it.
- Claims or disclosures of *other granted patents* (as opposed to what they describe as prior art) are not relevant to patentability — a common misconception is to treat a competitor's claim scope as informative of what is or isn't patentable; the patentability question concerns what the **reference discloses**, not what any other patent's claims cover, and infringement is a wholly separate question from patentability.

## Step 5: Confirm the Motivation-to-Combine Explanation, Not Just the Element Mapping

Even a complete, correct element-by-element mapping is not itself a complete rejection. The examiner must still separately articulate, per MPEP 2143/2144:
- Which of the seven (or another) rationale is being invoked;
- Why a PHOSITA would have been motivated to combine these specific references (not just that each contains a piece of the puzzle); and
- Why the combined/substituted/modified result would have been predictable.

A rejection that stops at "Reference 1 shows A and B, Reference 2 shows C, therefore obvious" — without the motivation-to-combine and predictability explanation — is a common but legally deficient shortcut. Requesting the examiner identify the specific MPEP 2143 rationale being relied on is a legitimate and often clarifying procedural step.

## Step 6: Look for Hindsight Reconstruction Red Flags

Because examiners read the specification before searching for prior art, hindsight risk is structural. Typical indicators of improper hindsight:
- Combining references from disparate, unrelated fields without explaining why a PHOSITA in the claimed field would have consulted the other field (i.e., failing the analogous-art test in MPEP 2141.01).
- Using the *applicant's own specification* as the source of the motivation to combine, rather than an independent reason grounded in the prior art or common knowledge/market forces.
- Selecting narrow, cherry-picked passages from a broad reference that happen to map precisely onto claim elements, while ignoring that the reference's actual focus/teaching runs counter to the claimed combination (a lighter-weight version of "teaching away").

## Step 7: Assess Secondary Considerations and Response Strategy

Even a technically sound element mapping and motivation showing can be overcome by objective evidence of nonobviousness: commercial success, long-felt unmet need, failure of others, unexpected results, copying, licensing, and expert skepticism — each requiring a **nexus** to the specifically claimed delta (see companion file on MPEP 2145). Practically, the strongest rebuttals combine multiple secondary-consideration categories into one reinforcing narrative, supported by declarations under 37 CFR 1.132 with comparative data against the closest prior art, rather than attorney argument alone.

## Summary Checklist for Reconstructing/Auditing an Obviousness Rejection

1. Is this an anticipation (single reference, all elements) or obviousness (combination) rejection?
2. Is every claim limitation mapped to a specific citation in a specific reference — with none silently dropped or ignored?
3. Is the claim construction underlying the mapping correct (not an overbroad reading manufactured to match the art)?
4. Does the rejection articulate *which* rationale (A–G or other) is invoked, a *specific* motivation to combine, and a *predictability* finding — not just an element inventory?
5. Are there teaching-away, hindsight, or non-analogous-art problems in how the references were selected or combined?
6. Does any rebuttal evidence (secondary considerations) have a genuine nexus to the claimed delta, sufficient to outweigh the prima facie showing on the whole record?

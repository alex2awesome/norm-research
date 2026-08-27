# MPEP § 2131 — Anticipation — Application of 35 U.S.C. 102
SOURCE_URL: https://www.uspto.gov/web/offices/pac/mpep/s2131.html
DOMAIN: patents

## Core doctrine: the "each and every element" test

A claim is unpatentable under 35 U.S.C. 102 as anticipated only when a single prior art reference discloses every element required by the claim, either expressly or inherently. This is the foundational element-matching rule for anticipation:

> "A claim is anticipated only if each and every element as set forth in the claim is found, either expressly or inherently described, in a single prior art reference." — *Verdegaal Bros. v. Union Oil Co.*, 814 F.2d 628, 631, 2 USPQ2d 1051, 1053 (Fed. Cir. 1987).

This is a strict, mechanical test at the level of the individual claim limitation: partial disclosure, or disclosure that is "close" but omits one element, is not anticipation — it may only support an obviousness rejection under § 103. There is no room for a "gist" or overall-similarity comparison; the comparison must proceed limitation-by-limitation.

### Genus/alternative claims

Where a claim reads generically or covers alternative structures/compositions, anticipation is established if *any* single structure or composition falling within the scope of the claim is found in the prior art:

> *Brown v. 3M*, 265 F.3d 1349, 1351, 60 USPQ2d 1375, 1376 (Fed. Cir. 2001) — a generic/Markush-type limitation is anticipated by prior art disclosure of one instance within its scope.

## Identity of invention — the matching standard

Anticipation requires strict identity, not mere overlap or similarity of function:

> "[T]he identical invention must be shown in as complete detail as is contained in the … claim." — *Richardson v. Suzuki Motor Co.*, 868 F.2d 1226, 1236, 9 USPQ2d 1913, 1920 (Fed. Cir. 1989).

This "identical invention" language is the operative matching criterion examiners and practitioners use to separate anticipation (identity) from obviousness (difference bridged by motivation/reasoning). If the reference and the claim differ by even one limitation, § 102 does not apply regardless of how minor or predictable the difference is.

## "Arranged as in the claim" — the structural/relational requirement

Element identity alone is not sufficient: the elements disclosed in the reference must also be arranged, connected, or combined the same way the claim recites. A reference that disclose all the same components used in a different configuration does not anticipate.

Two clarifications from the case law on how strict this "arrangement" matching must be:

1. **Not an *ipsissimis verbis* test.** The reference need not use the same words as the claim. Matching is a substantive comparison of structure/function, not a lexical string match:

   > *In re Bond*, 910 F.2d 831, 833, 15 USPQ2d 1566, 1567 (Fed. Cir. 1990) — anticipation does not require the prior art to use the identical terminology as the claim; what matters is whether the same subject matter, arranged the same way, is disclosed.

2. **A reference can anticipate an arrangement it never spells out verbatim, if a skilled reader would "at once envisage" that specific arrangement from the reference's teaching** (see the "at once envisage" doctrine developed further under § 2131.02 for genus/species and combination claims):

   > *Kennametal, Inc. v. Ingersoll Cutting Tool Co.*, 780 F.3d 1376, 1381, 114 USPQ2d 1250 (Fed. Cir. 2015).

   But this "at once envisage" gloss does not rescue a reference that is missing a limitation outright — it only resolves cases where the reference discloses a limited, closed set of possible combinations and the claimed combination is one of the small number a skilled artisan would immediately recognize as taught:

   > *Nidec Motor Corp. v. Zhongshan Broad Ocean Motor Co.*, 851 F.3d 1270 (Fed. Cir. 2017) (clarifying that *Kennametal* does not permit anticipation where a limitation is simply absent from the reference).

**Practical matching rule:** every claim limitation must (a) be found in the single reference, and (b) be found standing in the same structural/functional relationship to the other disclosed elements that the claim recites. Finding all the "parts" scattered across a reference, without the claimed connectivity/arrangement, is not anticipation.

## § 2131.01 — Multiple-reference 35 U.S.C. 102 rejections

The general rule is that anticipation rejections rely on a *single* reference. However, MPEP 2131.01 recognizes three narrow situations in which an examiner may cite additional references without converting the rejection into an obviousness (§103) rejection, because the extra material does not supply a missing claim limitation — it only supports interpretation of the primary reference:

**(A) To prove the primary reference's disclosure is enabling.** When a claimed composition or machine is identically disclosed by the primary reference, additional references may be used to show that the primary reference's disclosure was enabled (i.e., that a skilled artisan could have made/used it). No motivation to combine is required, because the second reference is not filling an element gap. *In re Samour*, 571 F.2d 559, 197 USPQ 1 (CCPA 1978); *In re Donohue*, 766 F.2d 531, 226 USPQ 619 (Fed. Cir. 1985).

**(B) To explain the meaning of terms used in the primary reference.** Extrinsic evidence (including other references, dictionaries, or expert testimony) may be used to construe — but not to expand — the meaning of terms/phrases actually used in the anticipatory reference. *In re Baxter Travenol Labs.*, 952 F.2d 388, 21 USPQ2d 1281 (Fed. Cir. 1991); see also *Actelion Pharmaceuticals Ltd v. Mylan Pharmaceuticals Inc.*, 85 F.4th 1167 (Fed. Cir. 2023) (extrinsic evidence proper where intrinsic evidence does not resolve the meaning of a measurement term).

**(C) To show that a characteristic not explicitly disclosed is nonetheless inherent.** Where the primary reference is silent as to an inherent characteristic, that evidentiary gap may be filled with extrinsic evidence demonstrating the characteristic is necessarily present. *Continental Can Co. USA v. Monsanto Co.*, 948 F.2d 1264, 1268, 20 USPQ2d 1746, 1749 (Fed. Cir. 1991). The evidence must make clear that "the missing descriptive matter is necessarily present in the thing described in the reference, and that it would be so recognized by persons of ordinary skill." Failure of skilled artisans to have actually, contemporaneously recognized the inherent property at the time is not a bar to a finding of inherent anticipation. *Atlas Powder Co. v. IRECO, Inc.*, 190 F.3d 1342, 1348-49, 51 USPQ2d 1943, 1947 (Fed. Cir. 1999).

## § 2131.03 — Anticipation of ranges

**A single example within a claimed range anticipates the whole range.** If a claim covers a genus of numerical values/compositions, disclosure of one specific point falling within that range anticipates the claim, even though the claim as a whole covers a broader range:

> *Titanium Metals Corp. v. Banner*, 778 F.2d 775, 227 USPQ 773 (Fed. Cir. 1985); *UCB, Inc. v. Actavis Labs. UT, Inc.*, 65 F.4th 679 (Fed. Cir. 2023) ("If the prior art discloses a point within the claimed range, the prior art anticipates the claim.").

**Overlapping/touching ranges require case-by-case specificity analysis.** Where the prior art discloses a *range* that touches or overlaps the claimed range, but does not give a specific example falling within the claimed sub-range, anticipation depends on whether the disclosure is specific enough to constitute "anticipation" of the narrower claimed range. Contrast:

- *ClearValue Inc. v. Pearl River Polymers Inc.*, 668 F.3d 1340 (Fed. Cir. 2012): claimed alkalinity below 50 ppm was anticipated by a reference disclosing the process works for systems with alkalinity "150 ppm and less," because there was no evidence of criticality or difference in the claimed sub-range.
- *Atofina v. Great Lakes Chem. Corp.*, 441 F.3d 991 (Fed. Cir. 2006): a reference's disclosed range of 100–500 °C did **not** anticipate a claimed sub-range of 330–450 °C, because "the disclosure of a range is no more a disclosure of the end points of the range than it is each of the intermediate points," and the patentee showed the claimed range was critical (skilled artisans would expect the process to behave differently outside it).

When it is unclear whether the disclosed range meets the "sufficient specificity" bar, examiners may issue a combined 102/103 rejection giving reasons for both grounds. *Ex parte Lee*, 31 USPQ2d 1105 (Bd. Pat. App. & Inter. 1993).

**A value/range that is merely close, but does not overlap, never anticipates.** Anticipation requires the reference to disclose "exactly what is claimed"; any gap between reference and claim — however small — must be addressed under § 103, not § 102. *Titanium Metals Corp. v. Banner*, 778 F.2d 775 (Fed. Cir. 1985).

## § 2131.04 — Secondary considerations are irrelevant to anticipation

Evidence of secondary considerations (unexpected results, commercial success, long-felt need, etc.) has no bearing on a § 102 anticipation rejection and cannot rebut it, because anticipation is a purely factual identity-of-disclosure question, not a question of obviousness-weighing:

> *In re Wiggins*, 488 F.2d 538, 543, 179 USPQ 421, 425 (CCPA 1973).

## § 2131.05 — Nonanalogous or disparaging prior art is still anticipatory

**"Analogous art" is not a requirement for anticipation.** Arguments that the reference is "nonanalogous art" or "teaches away" from the claimed invention are not germane to a § 102 rejection — these arguments matter only for obviousness (§ 103):

> *In re Self*, 671 F.2d 1344, 1350–51, 213 USPQ 1, 7 (CCPA 1982); *Twin Disc, Inc. v. United States*, 231 USPQ 417 (Ct. Cl. 1986).

A reference may come from an entirely different field of endeavor or address an entirely different technical problem than the claimed invention, and it is nonetheless anticipatory so long as it explicitly or inherently discloses every claim limitation:

> *State Contracting & Eng'g Corp. v. Condotte America, Inc.*, 346 F.3d 1057, 1068, 68 USPQ2d 1481, 1489 (Fed. Cir. 2003).

**Disparagement/teaching-away in the reference does not defeat anticipation.** A reference that discloses the invention and then criticizes or disparages it is still anticipatory as to that disclosure — the "teaches away" doctrine has no application to a § 102 analysis:

> *Celeritas Technologies Ltd. v. Rockwell International Corp.*, 150 F.3d 1354, 1361, 47 USPQ2d 1516, 1522−23 (Fed. Cir. 1998) ("The fact that a modem with a single carrier data signal is shown to be less than optimal … does not vitiate the fact that it is disclosed.").

## Summary of actionable matching criteria for a claim-matching metric

1. **Single-reference constraint**: all claim limitations must be located in one reference (subject to the narrow § 2131.01 exceptions for enablement, term-meaning, and inherency support — none of which supply a *missing* limitation).
2. **Completeness**: every limitation, not most limitations, must be disclosed; one missing limitation defeats anticipation entirely (push to obviousness instead).
3. **Identity, not similarity**: the standard is "identical invention... in as complete detail as is contained in the claim," not family resemblance or functional equivalence.
4. **Arrangement/connectivity must match**: scattered disclosure of the same components without the claimed structural relationship is insufficient; but exact wording is not required (no *ipsissimis verbis* test) — substance controls.
5. **"At once envisage" carve-out**: a reference can anticipate an un-spelled-out combination only when the universe of possible combinations it discloses is small/closed enough that a skilled reader would immediately recognize the claimed combination as one of them; it cannot cure an outright missing element.
6. **Ranges**: a single in-range data point anticipates; overlapping ranges need specificity analysis (criticality evidence can save the claim); near-miss, non-overlapping values never anticipate.
7. **Irrelevant rebuttals**: secondary considerations, non-analogous-art status, and disparagement of the disclosed subject matter do not defeat an otherwise-valid anticipation finding.

# MPEP § 2131.02 — Genus-Species Situations (Anticipation)
SOURCE_URL: https://www.bitlaw.com/source/mpep/2131-02.html
DOMAIN: patents

This subsection of MPEP § 2131 addresses one of the hardest element-matching questions for anticipation: when does a prior art reference's *generic* disclosure (a genus) count as disclosure of a narrower, claimed *species* — and conversely, when does prior art disclosure of one species anticipate a broader claimed genus? Both directions matter for claim-matching because they define when "coverage" of a limitation by a reference is real disclosure versus mere overlap of scope.

## I. A disclosed species anticipates a claimed genus

If the claim is generic (i.e., covers a broad genus) and the prior art discloses even a single species falling within that genus, the generic claim is anticipated:

> "A generic claim cannot be allowed to an applicant if the prior art discloses a species falling within the claimed genus." — *In re Slayter*, 276 F.2d 408, 411, 125 USPQ 345, 347 (CCPA 1960).

This holds even where the reference discloses many other species in addition to the anticipating one — comprehensiveness of the reference's own disclosure does not dilute the anticipatory force of the one overlapping species.

## II. A clearly named species anticipates regardless of how many other species are listed

Where a reference expressly names/identifies the specific claimed species — even within a very long list of alternatives — that express naming is itself sufficient disclosure. The size of the surrounding list is irrelevant to whether the named species was "taught":

> *Ex parte A*, 17 USPQ2d 1716 (Bd. Pat. App. & Inter. 1990): "the comprehensiveness of the listing did not negate the fact that the compound claimed was specifically taught." The Board analogized this to a dictionary — a reference containing thousands of entries still constitutes a distinct, individual disclosure of each entry.

**Matching rule:** if the specific claimed species/value appears by name/express identification anywhere in the reference, list-length is not a defense; the express recitation itself is the disclosure.

## III. Generic (non-enumerated) disclosure anticipates a species only when the species can be "at once envisaged"

This is the harder case: the reference does not name the specific species, but describes a genus or a set of variable substituents/parameters from which the species could be assembled. Whether this generic disclosure anticipates the specific claimed species is a factual question that depends on the breadth and structure of the disclosure — not a bright-line rule:

> "Whether a generic disclosure necessarily anticipates everything within the genus … depends on the factual aspects of the specific disclosure and the particular products at issue." — *Sanofi-Synthelabo v. Apotex, Inc.*, 550 F.3d 1075, 1082, 89 USPQ2d 1370 (Fed. Cir. 2008).

The controlling standard is whether a person of ordinary skill in the art, reading the reference, would "at once envisage" the claimed species/arrangement, even though the reference never spells it out explicitly:

> *Kennametal, Inc. v. Ingersoll Cutting Tool Co.*, 780 F.3d 1376, 1381, 114 USPQ2d 1250 (Fed. Cir. 2015).

Important limiting gloss: *Kennametal*'s "at once envisage" language does not mean any generic disclosure with a limitation missing can be cured by showing a skilled artisan would guess/expect it. It applies specifically to situations where the reference discloses a **limited number of possible combinations**, such that the claimed combination is simply one member of a small, closed set the reference already teaches:

> *Nidec Motor Corp. v. Zhongshan Broad Ocean Motor Co.*, 851 F.3d 1270, 1274−75 (Fed. Cir. 2017).

### Applying "at once envisage" to chemical/compound claims

Where the claimed compound is not itself named, but must be assembled by selecting and combining substituents from lists given in the reference, anticipation requires that the classes of substituents offered be "sufficiently limited or well delineated" that a skilled artisan could, from the reference alone, "draw the structural formula or write the name" of each compound covered:

> *Ex parte A*, 17 USPQ2d 1716.

**Worked example — *In re Petering*, 301 F.2d 676, 133 USPQ 275 (CCPA 1962):** A generic formula in the reference nominally covered a very large number (potentially thousands) of compounds. However, because the reference specified *preferred* substituents at each variable position, the actual combinatorial space collapsed to roughly 20 compounds sharing one large, unchanging structural core. The court held this was anticipatory: the limited number of substituent choices, combined with the fixed core structure, meant each of the ~20 permutations was described "as fully as if [the reference's author] had drawn each structural formula." This is the paradigm case for when a genus disclosure counts as disclosure "in substance" of an unnamed species.

### When the genus is too broad/open-ended, there is no anticipation

Conversely, where the reference's generic language covers an unbounded or very large number of undifferentiated possibilities, with no narrowing (no preferred substituents, no closed list), the species is not "at once envisaged" and there is no anticipation:

- *In re Meyer* — a reference disclosing "alkaline chlorine or bromine solution" was too broad a genus to anticipate a claim specifically directed to "alkali metal hypochlorite"; the genus term did not point a skilled reader to the specific claimed species.
- *Akzo N.V. v. Int'l Trade Comm'n*, 808 F.2d 1471 (Fed. Cir. 1986) — a claim requiring a 98% sulfuric acid solution was not anticipated by a reference disclosing merely "sulfuric acid solution" without specificity as to concentration.

## Matching criteria this subsection contributes to a claim-matching rubric

1. **Species-in-genus-claim direction**: any single disclosed species within a claimed genus anticipates the genus claim outright — no need for the reference to disclose the genus itself.
2. **Genus-in-species-claim direction**: a genus disclosure anticipates a specific claimed species only if:
   - the species is expressly named in the reference (list length is irrelevant), OR
   - the species can be "at once envisaged" — i.e., the reference's combinatorial space is small/closed/well-delineated enough (e.g., named preferred substituents, fixed core structure) that a skilled reader would immediately recognize the specific claimed instance as one of the disclosed possibilities.
3. **Failure conditions**: open-ended, unbounded, or merely qualitative genus language (e.g., a broad chemical class name, a vague range with no preferred/narrowing values) does not anticipate a specific claimed species/value, even if the species technically falls within the genus's literal scope. Overlap of *scope* is not the same as disclosure "in substance."
4. This is directly relevant to a claim-matching metric's need to distinguish **identity of disclosure** from **mere set-membership/scope-overlap** — the latter is not sufficient for anticipation unless the "at once envisage" or express-naming conditions are met.

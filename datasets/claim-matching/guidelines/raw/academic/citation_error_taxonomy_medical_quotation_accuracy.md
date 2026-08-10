# Quotation accuracy in medical journal articles — a systematic review and meta-analysis

SOURCE_URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4627914/
DOMAIN: academic

## What this source is

A systematic review and meta-analysis (medical/biomedical literature) that pools decades of studies auditing whether citations in published articles actually support the claims attached to them. Because so many independent research groups have run this exact audit (pick a sample of citing sentences, pull the cited source, judge whether it substantiates the sentence), this review is one of the most authoritative places to find a stable, cross-study taxonomy of citation/quotation error, together with the actual verification and reliability methodology used.

## Core distinction: citation accuracy vs. quotation accuracy

The review is explicit that "citation accuracy" audits (in the classic sense) and "quotation accuracy" audits are different tasks:

- **Citation accuracy** = correctness of the *bibliographic* information (author names, journal, volume, page numbers, year) — i.e., does the reference list entry correctly describe the source, independent of what it's used to claim.
- **Quotation accuracy** — the object of interest for claim-matching — is defined as: "the correctness of the content of a literature reference, that is, whether the reference supports or is in accordance with the statement by citing authors." This is a semantic/content check, not a formatting check: does the *substance* of the cited source actually back up the sentence it's attached to.

This is the key upstream distinction for any claim-matching metric: a citation can be bibliographically perfect (correct authors/pages/DOI) while still being a total quotation error (the source says something different from, or unrelated to, the claim), and vice versa.

## Severity taxonomy: major vs. minor errors

Across the pooled studies, quotation errors are consistently graded into two severity tiers, with slightly different wordings by author but a stable underlying structure:

**Major errors** — "not at all in accordance with the claim of the authors."
- De Lacey et al.'s canonical formulation (one of the earliest and most frequently reused definitions in this literature): a major error is one "seriously misrepresenting or bearing no resemblance to the original source."
- Luo et al. (2013) operationalize this more concretely: references that "contradicted, failed to substantiate, or were irrelevant to the author's assertion." This decomposition — contradiction / failure-to-substantiate / irrelevance — is itself a useful 3-way split within "major error" and maps closely onto NLI-style SUPPORTS/REFUTES/NEUTRAL-type labels used in later NLP work.

**Minor errors** — "inconsistencies and factual errors not severe enough to contradict a statement by citing authors."
- De Lacey's original wording: the error "misled or could mislead, but the errors were not sufficiently serious to destroy or fundamentally to alter the meaning." I.e., the gist of the claim survives, but some detail (a number, a qualifier, a scope restriction) is wrong or fabricated.

**Indirect/secondary references** — citations to a review article or secondary summary rather than to the original/primary source for a fact. Some studies fold this into "minor error" (the claim may be technically supported by the chain of citations, but the immediate cited work is not where the evidence originates); others track it as a separate, non-error category. For a claim-matching metric this is worth keeping as its own flag, since "does the cited paper itself contain the evidence, or does it just cite someone else who does" is a distinct and checkable property.

## How "support" was actually verified

The review is candid that most underlying studies did not use a rigid formal checklist; verification was done by trained raters reading the cited source and judging semantic correspondence to the citing sentence, with some notable calibration choices:
- Some studies counted a citation as "correct" even when it was "not logically accurate," as long as the reference was serving as a supporting *example* consistent with the general point being made — i.e., a looser "consistent with the gist" standard rather than strict entailment.
- Errors were counted three different ways across the literature, which matters if you are trying to compute a comparable "error rate" statistic:
  1. **Reference-based, restricted** — count at most one error per reference, regardless of how many quotation problems that single reference has.
  2. **Reference-based, unrestricted** — count every distinct error found against a given reference.
  3. **Quotation-based** — use each in-text citation instance (not each unique reference) as the denominator, so a reference cited five times contributes five denominators.

This denominator choice alone can swing reported "error rate" by a large factor between studies and is a critical methodological detail for anyone trying to compare or reproduce citation-accuracy numbers.

## Inter-rater reliability

- Pooled/average kappa across studies that reported it: **κ ≈ 0.76**, which clears the conventional Landis & Koch threshold for "substantial" agreement.
- But reliability is not uniform across error severity: Reddy et al. (2008) found much lower agreement specifically for **minor errors (κ = 0.26)**, i.e., near-chance-to-fair agreement. The review attributes this to "subjective factors" — minor/major is an easy top-level split, but graders disagree substantially on what counts as a "misleading but not fundamentally altering" minor error versus a non-error. This is a direct warning for claim-matching metrics: agreement on a coarse SUPPORT / NOT-SUPPORT binary will look much better than agreement on finer-grained partial-support judgments, and reported IAA numbers should be read at the granularity they were actually computed for.

## Practical takeaways for a claim-matching metric bank

1. Use a **content/semantic support check** separate from a bibliographic/formatting check — they measure different things and conflating them will understate errors.
2. A workable minimal taxonomy, replicated across dozens of independent studies: **no error (supported)** → **minor error (partially/misleadingly supported, gist survives)** → **major error**, with major further splittable into **contradicts**, **fails to substantiate (irrelevant/unrelated)**, and **misrepresents/bears no resemblance**.
3. Track **indirect/secondary sourcing** (citing a review instead of the primary evidence) as its own flag rather than silently merging it into "minor error."
4. Report which **denominator convention** (per-unique-reference restricted, per-unique-reference unrestricted, or per-citation-instance) any error-rate statistic uses — these are not interchangeable.
5. Expect and report reliability **separately by severity tier** — coarse binary agreement is not evidence that fine-grained partial-support judgments are also reliable.

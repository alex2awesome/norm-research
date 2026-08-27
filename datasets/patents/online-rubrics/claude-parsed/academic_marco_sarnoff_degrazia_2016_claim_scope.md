---
source_url: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2844964
title: Marco, Sarnoff & deGrazia (2016) - Patent Claims and Patent Scope
source_type: academic_journal
publication: USPTO Economic Working Paper 2016-04 (later Research Policy 2019)
authors: Alan C. Marco, Joshua D. Sarnoff, Charles deGrazia
year: 2016 (working paper) / 2019 (Research Policy)
era: modern empirical
fetched: 2026-05-09
---

# Operationalizing Patent Scope: Claim Length and Count

This USPTO Office of Chief Economist working paper introduces the now-standard quantitative measures of patent scope. The paper provides the most rigorous empirical operationalization of "claim breadth" available, validated against multiple downstream patent-quality outcomes.

## The Two Validated Scope Measures

**Independent Claim Length (Word Count)**: The number of words in the first independent claim. Shorter claims are systematically broader (fewer limitations restrict scope). Longer claims are narrower.

**Independent Claim Count**: The number of independent claims in the patent. Higher counts often signal scope-laddering strategy (multiple breadth tiers asserted in parallel).

Both measures are computed by USPTO's Python claim-parsing algorithm against the Patent Claims Research Dataset.

## Validation Against Quality-Relevant Outcomes

The authors validate these measures against:

- **Allowance probability**: Broader claims (shorter) are more likely to be rejected; narrower claims more likely allowed without amendment
- **Forward citations**: Broader-scope patents receive more forward citations (signaling pioneering / blocking value)
- **Litigation incidence**: Patents with broader independent claims are more likely to be litigated
- **Claim amendment patterns**: Word-count typically grows during prosecution as examiners force narrowing amendments

## Quality Implications of Scope Measures

**Anomalously Short Claims at Issuance**: An independent claim that issued without lengthening through prosecution—and is significantly shorter than peers in the same art unit—is a quality red flag. Either the claim is genuinely groundbreaking (rare) or examination was inadequate (common, given Frakes & Wasserman findings).

**Claim-Lengthening as Quality Signal**: A patent whose independent claim grew substantially during prosecution shows examiner engagement and applicant concession. The final claim is more likely to survive validity challenge than the as-filed version.

**Claim-Count Strategies**: Multiple independent claims at varying lengths (a "claim ladder") provides litigation flexibility. Quality-conscious drafting includes:
- One broad independent claim (litigation reach)
- Mid-tier independent claims (validity fallback)
- Narrow independent claims (commercial-product-specific)

## Operative Quality Markers from Marco-Sarnoff-deGrazia

For any issued patent, scope-related quality markers include:

**At-Issuance Markers**:
- Final independent claim length is at or above the art-unit median (suggests substantive narrowing)
- Number of independent claims appropriate to the technology (typically 1-3 for chemistry, 1-5 for software)
- Claim word count grew between filing and issuance

**Red Flags**:
- Final claim length significantly below art-unit median
- Single short independent claim with minimal amendment history
- Claim count inconsistent with strategic ladder (e.g., 20+ independent claims, signaling fee-bypass)
- All independent claims clustered at extreme breadth without intermediate fallback

## Industry Calibration

The paper documents systematic differences in claim length across art units:

- Pharmaceutical compound claims: typically short (structure suffices)
- Mechanical claims: medium length
- Software/method claims: longer (must specify steps)
- Biotech genus claims: variable; structural support requirements drive length

Quality assessment must account for art-unit baselines, not raw word counts.

## Methodological Importance

The Marco-Sarnoff-deGrazia measures are now the standard for empirical patent scholarship. They are used in subsequent studies of:

- Examiner heterogeneity (do strict examiners produce longer claims?)
- Litigation outcomes (do broader claims face higher invalidation rates?)
- Continuation strategies (do continuations strategically draft shorter claims?)
- Cross-national comparisons (USPTO vs. EPO claim breadth)

## Operative Test for Scope Quality

For any patent, evaluate:

1. What is the first independent claim word count?
2. How does it compare to the art-unit median?
3. How did the word count change from as-filed to as-issued?
4. How many independent claims are present, and at what breadth distribution?
5. Are dependent claims meaningfully narrower (real scope ladder vs. trivial dependents)?

A patent with a final independent claim at art-unit median or longer, evidence of meaningful narrowing during prosecution, and a coherent claim ladder is a high-scope-quality patent. The opposite pattern—short, unamended, with anomalous claim counts—signals either pioneering merit or examination failure.

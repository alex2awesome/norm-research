---
source_url: https://www.oecd.org/content/dam/oecd/en/publications/reports/2013/06/measuring-patent-quality_g17a22f5/5k4522wkw1r8-en.pdf
title: Squicciarini, Dernis & Criscuolo (2013) - OECD Measuring Patent Quality
source_type: academic_research_report
publication: OECD Science, Technology and Industry Working Paper 2013/03
authors: Mariagrazia Squicciarini, Hélène Dernis, Chiara Criscuolo
year: 2013
era: modern empirical
fetched: 2026-05-09
---

# OECD Patent Quality Indicator Framework

The OECD framework is the most comprehensive multi-indicator patent quality system in current use. It is the standard reference for cross-national patent quality comparisons.

## The Indicator Suite

### Technological Value Indicators

**Forward Citations (5-year window)**: Number of times the patent is cited by subsequent patents within 5 years of grant. Normalized by technology field and year.

**Originality Index** (Trajtenberg, Henderson & Jaffe 1997):
- Captures breadth of technology fields the patent draws on
- Computed as 1 - Σ(share of backward citations in each IPC class)²
- Higher = patent draws on more diverse fields = more original

**Generality Index**:
- Captures breadth of technology fields the patent influences
- Computed as 1 - Σ(share of forward citations in each IPC class)²
- Higher = patent influences more diverse fields = more general

**Radicalness Index**:
- Time-invariant count of IPC classes that the patent's backward citations are in but that the patent itself is not classified in
- Higher = patent draws from foreign technological domains = more radical

**Patent Scope (IPC classes)**: Number of distinct IPC sub-classes assigned to the patent. Broader IPC coverage suggests broader scope.

### Economic Value Indicators

**Family Size**:
- Number of patent offices where the family is filed
- Triadic Patent Family (US + EP + JP) = highest signal
- Larger family = greater applicant belief in commercial value

**Patent Renewals**:
- Number of years maintenance fees are paid before lapse
- Patents renewed to full term (US: 20 years from filing) signal high value
- Early abandonment signals low commercial significance

**Grant Lag**:
- Time from application to grant
- Long grant lag may signal disputed patentability OR strategic delay
- Short grant lag may signal weak examination OR clearly patentable invention

### Examination-Based Indicators

**Number of Inventors**: Larger inventor teams correlate with more substantial inventions (R&D investment).

**Number of Backward Citations**: More citations may signal deeper technological lineage; few citations may signal pioneering OR weak prior-art search.

**Claim Count and Independent Claim Count**: Following Marco-Sarnoff-deGrazia, both signal scope strategy.

## Quality Implications

### Composite Quality Score

OECD constructs an aggregate quality score weighting the indicators. High-quality patents score high on:

- Multiple indicator dimensions (not just citations)
- Both technological value AND economic value indicators
- Cohort-normalized comparisons (within technology field and year)

### Use Cases for OECD Indicators

The framework supports:

**Cross-National Comparison**: USPTO vs. EPO vs. JPO patent quality
**Industry Comparison**: Pharmaceutical vs. software quality differences
**Firm-Level Comparison**: High-quality vs. low-quality patentees
**Policy Evaluation**: Did patent reform increase or decrease quality?

### Originality and Generality as Quality Markers

A patent that scores high on originality (draws on diverse fields) AND high on generality (influences diverse fields) is a high-quality cross-cutting invention. Examples:

- Foundational software algorithms
- General-purpose technologies (e.g., CRISPR)
- Cross-industry chemical processes

A patent low on both is typically a narrow incremental improvement.

## Operative Quality Markers from OECD Framework

**Strong Quality Composite**:
- Forward citations top-quartile (within cohort)
- Triadic family
- Originality and Generality both top-quartile
- Renewed to full term
- Multiple inventor team

**Weak Quality Composite**:
- Forward citations bottom-quartile
- US-only family
- Originality and Generality both bottom-quartile
- Abandoned at first maintenance fee
- Single inventor

## Limitations Acknowledged

OECD documentation explicitly notes:

- Indicators are correlated; no single indicator is sufficient
- Measurement requires sufficient post-grant time (5+ years)
- Industry differences are large; cross-industry comparison requires normalization
- Strategic patenting (defensive, blocking) can confound value-quality interpretation

## Synthesis

The OECD framework operationalizes patent quality as a multi-dimensional latent construct measured by a battery of indicators. For evaluating any individual patent, the framework supports:

1. Cohort-normalized scoring on each dimension
2. Composite score combining multiple dimensions
3. Quality classification (high / medium / low) within technology field
4. Cross-jurisdictional comparison (US, EP, JP, KR, CN)

A patent in the top quartile of the composite OECD index, conditional on technology field and grant year, is a high-quality patent with confidence. The framework is the de facto standard for academic and policy use.

## Operative Test

For any issued patent (5+ years post-grant):

1. Compute forward citations vs. cohort: top quartile?
2. Compute originality and generality indices: both top quartile?
3. Determine family size: triadic?
4. Determine renewal status: paid through current period?
5. Aggregate into composite score; classify as high/medium/low quality

A patent passing all five tests is among the highest-quality grants in its cohort.

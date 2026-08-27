---
source_url: https://www.ipeg.com/wp-content/uploads/2015/02/Lanjouw-Schankerman-PATENT-QUALITY-AND-RESEARCH-PRODUCTIVITY-measuring-innovation-with-multiple-indicators.pdf
title: Lanjouw & Schankerman (2004) - Patent Quality with Multiple Indicators
source_type: academic_journal
publication: Economic Journal, 114(495), 441-465
authors: Jean O. Lanjouw, Mark A. Schankerman
year: 2004
era: modern empirical
fetched: 2026-05-09
---

# Multi-Indicator Patent Quality Index

Lanjouw & Schankerman develop the canonical multi-indicator approach to patent quality measurement. Their key insight: combining multiple noisy indicators dramatically reduces measurement error in latent quality.

## Methodology and Findings

The authors model patent quality as a latent variable jointly determining four observable indicators:

1. **Number of patent claims** (scope proxy)
2. **Forward citations** (importance proxy)
3. **Backward citations** (technological lineage)
4. **Family size** (international filings; commercial significance proxy)

Estimating on ~8,000 US patents (1960-91) across four technology areas, they find:

**Variance Reduction**: The variance in latent quality, conditional on all four indicators, is **only one-third** of the unconditional variance. Multi-indicator measurement substantially reduces noise.

**Forward Citations Dominant**: Among the four indicators, forward citations carry the strongest signal weight. But each indicator adds non-redundant information.

**Quality and Productivity**: Firm research productivity (patents per R&D dollar) is **inversely** related to patent quality—firms produce many low-quality patents or few high-quality patents, suggesting a productivity-quality tradeoff.

**Quality and Market Value**: Patent quality is positively associated with firm stock market value, validating the latent-quality construct against external market measures.

## Quality Indicators in Detail

### Claim Count

More claims signal:
- Broader scope strategy (multiple breadth tiers)
- Higher prosecution investment
- More careful claim drafting

Higher claim counts correlate (weakly) with quality, but the correlation is much stronger for "independent claim count" than total claims.

### Forward Citations

The single most powerful indicator. Conditioned on technology and grant year:
- Top quartile = high quality
- Bottom quartile = low quality
- Zero forward citations after 5+ years = strong negative signal

### Backward Citations

Indicators of technological context:
- Many backward citations may signal cumulative innovation (improvement patent)
- Few backward citations may signal pioneering invention OR weak prior-art search
- Mixed quality signal; must be combined with other indicators

### Family Size (International Patent Family)

The number of jurisdictions where the patent family is filed:
- Larger families signal applicant belief in commercial value
- Triadic families (US + EP + JP) are particularly strong quality signals
- Single-jurisdiction filings signal limited commercial expectation

## Quality Implications

### The Multi-Indicator Quality Score

Lanjouw-Schankerman demonstrate that no single indicator is sufficient. A composite measure—weighted combination of claims, forward citations, backward citations, and family size—reduces noise by ~67%.

For evaluating any patent, calculate:

1. Z-score of independent claim count vs. cohort
2. Z-score of forward citations vs. cohort (5-year window)
3. Z-score of backward citations vs. cohort
4. Family size (1 = US only; 2-3 = small family; 4+ = global family; triadic = high signal)

Composite top-quartile patents are reliably higher quality than composite bottom-quartile patents.

### Productivity-Quality Tradeoff

The inverse productivity-quality relationship implies:

- High-volume patenters (e.g., large tech firms with thousands of patents per year) systematically produce lower-quality patents
- Selective patentees (e.g., universities, individual inventors with few patents) produce higher-quality patents on average
- Patent quality cannot be inferred from patent count without normalization

## Operative Quality Markers Derived

**Strong Quality Signals**:
- Patent in top quartile on 3+ of the 4 indicators
- Triadic family (US + EP + JP)
- Claim count above cohort median, forward citations above cohort 75th percentile
- Backward citations diverse across art units (originality)

**Weak Quality Signals**:
- Patent in bottom quartile on 3+ of the 4 indicators
- US-only family (no international filing)
- Few claims, no forward citations after 5+ years
- All backward citations within single art unit

## Methodological Importance

The Lanjouw-Schankerman approach is now standard in:

- USPTO Office of Chief Economist studies
- OECD Patent Quality Reports (Squicciarini, Dernis, Criscuolo)
- WIPO Statistical Yearbook patent quality measures
- Cross-national innovation studies

Subsequent work (Higham, de Rassenfosse, Jaffe 2021) shows that even multi-indicator measures have limitations and recommends considering pre-grant indicators (PHOSITA characterization, prosecution effort) alongside post-grant indicators.

## Operative Test

For any issued patent with sufficient time to accumulate downstream signals:

1. Compute Z-scores of claim count, forward citations, backward citations, and family size against same-cohort patents
2. Construct composite (weighted average; forward citations get 2x weight)
3. Patents in top quartile of composite are high quality with confidence
4. Patents in bottom quartile are low quality with confidence
5. Middle quartiles require additional evidence (litigation history, examiner identity, prosecution depth)

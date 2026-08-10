---
source_url: https://onlinelibrary.wiley.com/doi/10.1111/1756-2171.12307
title: Kuhn, Younge & Marco (2020) - Patent Citations Reexamined
source_type: academic_journal
publication: RAND Journal of Economics, 51(1), 109-132
authors: Jeffrey M. Kuhn, Kenneth A. Younge, Alan C. Marco
year: 2020
era: modern empirical
fetched: 2026-05-09
---

# Patent Citations Reexamined: Quality Measurement Critique

This paper challenges the foundational use of forward citations as a quality measure. Kuhn, Younge & Marco show that citation patterns have changed dramatically, undermining naive citation-quality interpretations.

## Headline Empirical Findings

**Citation Concentration**: A small minority of patent applications generate the majority of patent citations. Citation distribution has become substantially more skewed over time.

**Falling Technological Similarity**: Mean technological similarity between citing and cited patents has fallen considerably. Many citations are formal/strategic rather than substantive technological references.

**Methodology**: The authors develop a vector-space patent-similarity model using new USPTO data to identify which citations are technologically informative.

**Strategic Citation**: A non-trivial fraction of citations are strategic (defensive, blocking) rather than reflective of technological lineage.

## Quality Implications

### Citation-Quality Reinterpretation

Kuhn, Younge & Marco show that raw citation counts have become less reliable as quality measures over time:

- 1980s-90s: Citations primarily technological, well-correlated with quality
- 2000s-present: Citations include substantial strategic/formal content
- Naive citation-quality use overstates quality of citation-rich patents that are mostly cited for non-technological reasons

### Improved Citation Quality Measures

The authors propose:

**Informative Citation Subsetting**: Only count citations where citing and cited patents have above-threshold technological similarity. Strategic citations (low similarity) are filtered out.

**Citation Weighting**: Weight each citation by technological similarity. Substantive citations get more weight than formal references.

**Vector-Space Similarity**: Use NLP-derived patent-text similarity to validate citation informativeness.

These methods substantially improve the predictive power of citation-based quality measures.

## Quality Markers from Refined Citations

For evaluating any patent's citation-based quality:

**Strong Quality Markers**:
- Forward citations from technologically similar patents (substantive citations)
- Citations from across diverse art units (high originality)
- Citations from major industry firms (commercial validation)
- Citations grow over time, accelerating in years 5-10 post-grant

**Weak Quality Markers**:
- Forward citations from technologically dissimilar patents (formal/strategic)
- Citations concentrated in single art unit (no cross-field impact)
- Citations only from minor entities or own firm
- Citations stagnate or decline after first 3 years post-grant

## Implications for Established Quality Frameworks

The Kuhn-Younge-Marco findings require revision of:

**Hall-Jaffe-Trajtenberg (2005)**: Their 3% per-citation market value premium may be inflated by inclusion of strategic citations. Technology-similar citations only would yield a higher per-citation premium with greater predictive power.

**Lanjouw-Schankerman (2004)**: Their citation-dominant quality index may overweight strategic citations. Refined indicators should weight by similarity.

**OECD Quality Framework (2013)**: Citation indicators should be similarity-weighted for accurate cross-cohort comparison.

**Trajtenberg (1990)**: His citation-welfare correlation (~0.75) was estimated on 1968-86 data when strategic citations were less prevalent. Modern patents require similarity-adjusted citation measures.

## Operative Quality Markers

**For Citation-Based Quality Assessment**:

1. Compute raw forward citation count
2. Compute technologically-similar citation count (using NLP similarity)
3. Compare: ratio of similar/total citations indicates citation informativeness
4. Compare patent's similar-citation count to cohort distribution
5. Top quartile of similar citations = high quality

**Strategic Citation Markers**:
- Patent is in a cross-licensing portfolio (citations may be strategic)
- Patent is part of a standard-essential portfolio (citations may be defensive)
- Owner is a known prolific patentee (citations may be self-strategic)

## Synthesis

Kuhn, Younge & Marco provide a critical quality-measurement update:

**Old paradigm**: More citations = higher quality
**New paradigm**: More technologically-similar citations = higher quality

For evaluating any patent (5+ years post-grant):

1. Compute forward citation count
2. Estimate technological similarity to citing patents (using NLP or proxy)
3. Compute similarity-weighted citation score
4. Compare to cohort distribution
5. Patents in top quartile of similarity-weighted citations are high quality

The naive citation count remains useful as a screening signal but should not be the primary quality measure for high-stakes assessments.

## Methodological Importance

The paper has been highly influential in shaping subsequent patent-quality research:

- Higham, de Rassenfosse & Jaffe (2021) cite Kuhn-Younge-Marco as motivation for multi-indicator caution
- USPTO Office of Chief Economist research now incorporates similarity-weighted measures
- Commercial patent analytics (PatentSight, Innography) have begun adopting similarity-adjusted citation scores
- Academic patent-quality research increasingly uses NLP-derived similarity measures

## Operative Test

For any issued patent:

- Raw citation count: useful but limited
- Similarity-weighted citation count: more reliable quality measure
- Citation source diversity (across firms, art units): originality signal
- Citation acceleration over time: continued relevance signal
- Citation in litigation context: commercial significance signal

A patent with high similarity-weighted citations, diverse citers, accelerating pattern, and litigation citations is reliably high quality. Naive citation-rich patents lacking these refinements may be inflated by strategic citations.

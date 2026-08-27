---
source_url: https://www.aeaweb.org/articles?id=10.1257%2Faeri.20190499
title: Kelly, Papanikolaou, Seru & Taddy (2021) - Measuring Technological Innovation Over the Long Run
source_type: academic_journal
publication: American Economic Review: Insights, 3(3), 303-320
authors: Bryan T. Kelly, Dimitris Papanikolaou, Amit Seru, Matt Taddy
year: 2021
era: modern empirical (NLP-driven)
fetched: 2026-05-09
---

# Textual Patent Quality: Breakthrough Identification

This paper develops the most influential text-based patent quality measure: identifying breakthrough patents through textual similarity to past and future patents.

## Core Methodology

The "KPST" measure identifies high-impact patents based on two textual properties:

**Backward Distinctiveness**: The patent should be textually different from prior patents (signaling novelty)

**Forward Influence**: Subsequent patents should be textually similar to the patent (signaling influence on later innovation)

A patent that is both backward-distinctive and forward-influential is a "breakthrough" — a textually novel innovation that subsequent inventors built upon.

## Quality-Relevant Properties

The KPST measure has several quality-relevant features:

**Pure Text-Based**: No reliance on citations (which are strategic, examiner-dependent, time-lagged)
**Long Time Series**: Constructed from 1840 to present, enabling cross-century comparison
**TF-IDF Weighted**: Important terms (rare across documents) get more weight
**Sectoral Decomposition**: Aggregate and sector-specific innovation indices

## Quality Implications

### Breakthrough vs. Incremental Patents

A breakthrough patent (high KPST score) is qualitatively different from an incremental improvement:

**Breakthrough Markers**:
- High textual distance to prior patents (uses novel vocabulary or combinations)
- Subsequent patents adopt similar vocabulary (influence)
- Sustained influence over years (vs. temporary citation spikes)

**Incremental Markers**:
- Textually similar to many prior patents (uses established vocabulary)
- Subsequent patents do not adopt new vocabulary from this patent
- Citation-rich but not text-influential (strategic citations)

### KPST vs. Citation-Based Quality

The KPST measure complements citation-based quality assessment:

- Citations measure community reference patterns (subject to strategic citation)
- KPST measures actual technological influence through vocabulary adoption
- Both signals combined are stronger than either alone
- Discrepancies (high citations but low KPST, or vice versa) flag interesting patents

## Operative Quality Markers from KPST

**Strong Quality Markers**:
- Top-quartile KPST score in the patent's cohort (technology + grant year)
- High backward distinctiveness combined with high forward influence
- KPST score validated by independent quality measures (citations, family, renewals)

**Weak Quality Markers**:
- Bottom-quartile KPST score
- Backward similar (uses standard vocabulary) and forward unimaginative (no new vocabulary adopted)
- Citation-rich but textually not influential (strategic patenting)

## Validation Against External Quality Measures

KPST validates against:

- Stock-market value of inventor firms
- Patent renewal patterns
- Subsequent technology adoption
- Industry productivity growth

Patents in the top KPST decile correlate strongly with downstream economic impact, supporting the measure's quality-relevance.

## Implications for Patent Drafting

If quality is interpreted via KPST principles, drafting recommendations include:

**Linguistic Innovation**:
- Use technically novel vocabulary where the invention warrants
- Avoid copy-pasting from prior patents (reduces backward distinctiveness)
- Develop precise technical terminology that subsequent inventors might adopt

**Structural Innovation**:
- Disclose detailed technical implementation (gives subsequent inventors something to build on)
- Articulate the innovation's relationship to prior approaches (highlights distinctiveness)
- Provide working examples that subsequent inventors can extend

These drafting choices position the patent for KPST-measurable breakthrough quality.

## Operative Test

For any issued patent (5+ years post-grant):

1. Compute textual similarity to prior patents in same field (should be low for breakthrough)
2. Compute textual similarity from subsequent patents to this patent (should be high for breakthrough)
3. Combine into KPST-style breakthrough score
4. Compare to cohort distribution
5. Top decile = breakthrough; lower deciles = incremental or non-influential

The KPST framework provides a quality measure orthogonal to citations and renewals, capturing actual technological influence through vocabulary diffusion.

## Synthesis

The Kelly-Papanikolaou-Seru-Taddy framework supports a text-based quality assessment principle:

**True breakthrough patents are textually distinctive backward and influential forward**.

This complements:
- Citation-based quality (community reference patterns)
- Renewal-based quality (revealed-preference value)
- Family-based quality (international commercial belief)

A patent ranking high on KPST, citations, renewals, and family is reliably a top-quality grant. Discrepancies among these measures flag patents requiring deeper investigation.

## Use Cases

The KPST measure is now used in:

- Cross-century innovation studies
- Sectoral productivity analysis
- Firm-level innovation valuation
- Patent quality benchmarking
- AI-augmented patent search and analysis

The measure is one of the leading examples of NLP-based patent quality assessment, paired with the Suzgun et al. HUPD dataset and PatentBERT-style models.

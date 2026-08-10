---
source_url: https://arxiv.org/abs/2207.04043
title: Suzgun et al. (2022) - The Harvard USPTO Patent Dataset
source_type: academic_journal
publication: NeurIPS 2023 Datasets and Benchmarks Track
authors: Mirac Suzgun et al.
year: 2022/2023
era: modern (NLP infrastructure)
fetched: 2026-05-09
---

# HUPD: NLP Infrastructure for Patent Quality Research

The Harvard USPTO Patent Dataset (HUPD) provides the foundational NLP corpus for empirical study of patent application quality. Critically, it includes inventor-submitted versions, enabling pre-grant quality analysis.

## Dataset Characteristics

**Scale**: 4.5+ million patent applications filed 2004-2018
**Distinctive Feature**: Includes as-filed (inventor-submitted) versions, not just granted versions
**Structured Metadata**: Each application includes:
- Application/publication numbers
- Title, abstract, claims, background, summary, full description
- Filing/publication dates, decision status
- Inventor, examiner, attorney information
- Primary and secondary classification codes (CPC/IPC)

**Format**: One JSON file per application, distributed via HuggingFace Datasets

## Quality-Relevant Tasks Enabled

### Patent Acceptance Prediction

The dataset enables training models that predict, from as-filed text alone, whether an application will be granted, abandoned, or pending. This task is quality-relevant because:

- Pre-grant quality signals can be extracted from text
- Models reveal which textual features correlate with grant
- Quality prediction at filing enables proactive drafting improvement

### Subject Area Classification

Multi-class classification of patent technology areas. Quality applications:

- Are correctly classified by NLP models (signal of clear technical communication)
- Use vocabulary and structure consistent with their technology field
- Match expected examiner art-unit assignment

### Language Modeling

Patent-specific language models trained on HUPD enable:

- Quality assessment via perplexity (anomalous-perplexity patents may have drafting issues)
- Style analysis (does the patent follow industry-standard drafting conventions?)
- Generation/evaluation of patent text

### Summarization

Quality patent abstracts:

- Accurately summarize the claims
- Convey the technical contribution
- Are extractable by NLP summarization models

## Quality Implications

### Pre-Grant Quality Markers Identifiable Through HUPD

Using HUPD-trained models, several pre-grant quality markers can be extracted:

**Drafting Quality Markers**:
- Clear technical language (not legal boilerplate-only)
- Detailed technical disclosure
- Specific embodiment descriptions
- Consistent terminology across sections

**Patentability Quality Markers**:
- Claims appropriately calibrated to disclosure
- Background distinguishing from prior art
- Detailed description sufficient to enable POSITA
- Specification anticipates examiner objections

**Strategic Quality Markers**:
- Continuation potential (open-ended disclosure supporting later claims)
- International family planning (PCT-friendly drafting)
- Litigation readiness (clear claim construction support)

### Examination-Quality Markers Identifiable

HUPD enables study of examination quality through:

- Examiner identity → examination patterns
- Office action characteristics
- Time to grant
- Continuation/RCE patterns
- Eventual outcome (granted/abandoned)

## Operative Quality Test for As-Filed Applications

Using HUPD-style models, evaluate any application:

1. **Predicted Grant Probability**: Run as-filed text through grant-prediction model. Top quartile = high pre-grant quality signal.

2. **Classification Confidence**: Does the application get correctly classified by subject-area model? High confidence = clear technical communication.

3. **Perplexity Analysis**: Does the text fall within expected perplexity distribution for the technology area? Anomalous perplexity may signal drafting issues.

4. **Claim-Disclosure Alignment**: Do summarization models extract claim content from the description? Misalignment suggests support issues.

5. **Section Coherence**: Are background, summary, claims, and detailed description coherent in topic and terminology?

A pre-grant application passing all five tests is high quality and grant-likely. Failures indicate drafting issues correctable before filing.

## Implications for Patent Quality Research

HUPD enables empirical research that was previously impossible:

**Pre-Grant Quality Studies**: Previous research focused on grant outcomes; HUPD enables analysis of as-filed versus as-issued differences (capturing examination effects).

**Drafting Best-Practice Identification**: Comparing successful and unsuccessful applications by topic, drafter, examiner enables data-driven drafting recommendations.

**Examiner Effects**: HUPD's examiner metadata enables fine-grained study of examiner heterogeneity (extending Cockburn-Kortum-Stern, Lemley-Sampat, Frakes-Wasserman).

**Cross-National Comparison**: HUPD enables comparison with EPO, JPO, CNIPA datasets for jurisdictional quality analysis.

## Synthesis

The HUPD framework supports the operative principle: **patent quality is increasingly measurable via NLP at the as-filed stage, before any examiner intervention**.

For evaluating any application:

1. Apply HUPD-trained grant prediction model: what is predicted probability of allowance?
2. Apply subject-area classifier: is the application clearly in its claimed technology field?
3. Apply summarization model: do claims align with description?
4. Apply perplexity model: is the text within normal drafting range for the field?
5. Compare examiner identity to examiner-quality statistics

This NLP-augmented quality framework integrates with classical empirical patent quality measurement.

## Use Cases and Extensions

HUPD has been used in:

- LLM-based patent drafting tools
- Pre-filing quality assessment systems
- Examiner-applicant negotiation prediction
- Patent valuation models
- Patent landscape analysis

The dataset is the de facto standard for NLP-based patent quality research, paired with PatentBERT (Lee & Hsiang 2020) and PatentSBERTa for embeddings.

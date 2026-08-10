---
source_url: https://arxiv.org/html/2408.14622v1
title: LLM Story Evaluation - Comprehensive Survey of Quality Criteria
source_type: academic_paper
fetched: 2026-05-09
---

# LLM Story Evaluation — Survey of Quality Criteria

A comprehensive survey of how researchers (and LLM benchmarks) evaluate story quality.

## 14 Quality Dimensions Used in Story Evaluation Research

1. **Nuanced Characters** — complex, multi-dimensional character development
2. **Emotionally Engaging** — evokes genuine emotional response
3. **Compelling Plot** — engaging narrative structure and pacing
4. **Coherent** — logical consistency and clarity throughout
5. **Causally Sound** — events follow causally
6. **Character Intentionality** — characters act from believable motivations
7. **Dramatic Conflict** — meaningful tensions
8. **Originality / Creativity** — fresh ideas
9. **Relevance** — adheres to prompt
10. **Fluency** — sentence-level grammatical quality
11. **Surprise** — unexpected but earned developments
12. **Empathy** — reader feels with characters
13. **Engagement** — reader wants to continue
14. **Complexity** — depth of theme/character/plot

## Major Frameworks Catalogued

### TTCW (Chakrabarty et al. 2024)
14 binary tests across 4 Torrance dimensions (Fluency, Flexibility, Originality, Elaboration). Already saved.

### HANNA (Chhun et al. 2022)
Human-Annotated NArratives benchmark; 6-dim ratings: Relevance, Coherence, Empathy, Surprise, Engagement, Complexity.

### EQ-Bench Longform Creative Writing
Multi-criteria human ratings of long-form creative writing with LLM-as-judge correlations.

### StoryER (EMNLP 2022)
Ranking, rating, and reasoning automatic evaluation.

### OpenMEVA (Guan et al. 2021)
Meta-evaluation benchmark for open-ended generation.

### UNION (Guan & Huang 2020)
Unreferenced metric trained on synthetic negatives.

### MAUVE (Pillutla et al. 2021)
Distributional similarity between human and machine text.

### GPTScore / G-Eval / G-Bench
LLM-as-judge approaches with detailed rubrics.

### SS-Bench (Social Stories)
Social-story specific evaluation.

## GPT-4 vs GPT-3 (Empirical Findings)
- GPT-4 outperforms GPT-3 in 62% of human-eval comparisons
- Better at language alignment, vocabulary, situation handling
- Still weak at: sustained character intentionality, dramatic conflict, planning across long contexts

## Key Research Finding
"There is no clear consensus on how to evaluate stories, with even human experts using explicitly delimited rating criteria giving substantially divergent ratings to the same stories."

This is precisely the problem motivating the LitBench / norm-discovery research direction.

## Methodological Lessons
1. Detailed rubrics with criterion definitions improve evaluation reliability
2. Fine-grained per-aspect evaluation > overall holistic rating
3. Human evaluation remains required for final judgments of quality
4. Automatic metrics often don't align with human judgment
5. LLM-as-judge converges with human experts when given the same rubric

---
source_url: https://aclanthology.org/2022.emnlp-main.114.pdf
title: NLP Automatic Story Evaluation - Survey of Metrics
source_type: academic_paper
fetched: 2026-05-09
---

# Automatic Story Evaluation in NLP — Surveyed Metrics

## Multi-Dimensional Evaluation Categories
- **Character Consistency** — characters behave/sound the same across the story
- **Plot Progression** — events build causally toward an outcome
- **Emotional & Psychological Realism** — believable affect
- **Continuity & Consistency in Story Elements** — facts/objects don't contradict
- **Coherence** — semantic and discourse-level connection across sentences
- **Engagement / Interestingness** — does the story hold attention?
- **Fluency** — sentence-level grammatical quality
- **Diversity** — lexical/structural variety across the story
- **Relevance** — adherence to prompt or premise

## Common Automatic Metrics
- **BERTScore** — embedding similarity to reference
- **BERT Next Sentence Prediction** — local coherence
- **Entity-Grid Coherence** — referential continuity of entities
- **Perplexity** — fluency proxy via language model probability
- **BLEU / ROUGE / METEOR** — n-gram overlap (poor for stories, but often reported)
- **MAUVE** — distributional similarity between human and machine text
- **NCI / NCI-2.0** — narrative coherence indices
- **EASM** — emotional arc similarity / consistency
- **Discriminator-based metrics** — train a BERT/RoBERTa classifier to distinguish good vs. bad stories

## Frameworks
- **StoryER** (EMNLP 2022) — automatic story evaluation via Ranking, Rating, and Reasoning
- **HANNA** (Chhun et al.) — Human-Annotated NArratives benchmark; 6-dim human ratings (Relevance, Coherence, Empathy, Surprise, Engagement, Complexity)
- **TTCW** (Chakrabarty et al. 2024) — 14 binary expert tests across 4 Torrance dimensions
- **OpenMEVA** (Guan et al. 2021) — meta-evaluation benchmark for open-ended generation
- **UNION** (Guan & Huang 2020) — unreferenced metric trained on synthetic negatives

## Key Limitation
"Many automatic metrics do not align well with human preferences." Human evaluation remains required for qualities like interestingness or overall narrative quality.

## Most-Used Human-Evaluation Dimensions
1. Relevance to prompt
2. Coherence (logical flow)
3. Fluency (sentence grammar/style)
4. Interestingness / Engagement
5. Empathy / Emotional resonance
6. Surprise / Originality
7. Complexity (theme/character/plot)

---
source_url: https://arxiv.org/abs/2209.06293
title: Hessel et al. — Caption Contest Annotation Guidelines
source_type: academic_paper
fetched: 2026-05-09
---

# Hessel et al. — Caption Contest Annotation Guidelines

## Annotation Guidance for Joke Explanation
> "In a few sentences, explain the joke as if to a friend who doesn't 'get it' yet."

A corpus of 651 human-created joke explanations was formed (avg ~60 words).

## Annotations Per Cartoon
For each cartoon image, annotators provide:
- **Image description** — literal scene, locations, entities.
- **Uncanny / unusual elements** — what is incongruous in the image.
- **Caption explanation** — why the winning caption is funny, connecting unusual elements to the punch.

## Best-Performing Features for Identifying Funny Captions
From prior work analyzed in the paper:
- **Perplexity** (under a language model) — surprisingness predicts humor.
- **Match to image setting and uncanniness description** — caption must engage what's weird.
- **Readability** — short, simple, clear language.
- **Proper nouns** — specific named entities help.
- **Overlap with WordNet's "person" and "relative" synsets** — human-centeredness.
- **Lexical centrality among submissions** — but unique enough.
- **Sentiment** — appropriate emotional tone.

## Why Captions Are Hard for AI
- Indirect, playful image-caption relationships.
- Reference real-world entities and norms.
- Theory-of-mind reasoning about characters.
- Cultural context required.
- "Several attempts to solicit explanations from crowdworkers were not satisfactory."
- "Prompting experiments with GPT-3 were similarly unsuccessful."

## Three Tasks
1. **Matching**: caption-to-cartoon.
2. **Quality Ranking**: finalist > low-quality.
3. **Explanation**: generate joke explanation.

## Rubric (used in annotation)
1. Does the caption engage the image's unusual element?
2. Does it provide a coherent explanation/frame for the absurd?
3. Does it ascribe a believable interior to a character?
4. Is it readable, with appropriate sentiment?
5. Could one explain in 60 words why it's funny?

---
source_url: https://aclanthology.org/H05-1067/
title: Mihalcea & Strapparava — Making Computers Laugh (humor recognition features)
source_type: academic_paper
fetched: 2026-05-09
---

# Mihalcea & Strapparava — Computational Humor Recognition

Foundational work (2005-2006) showing automatic classifiers can distinguish humorous from non-humorous text.

## Key Findings
- Computational approaches successfully applied to humor recognition.
- Significant improvements over baselines.
- Phonetic features at least as important as content.

## Feature Categories

### Phonetic / Stylistic Features
- **Alliteration chains**: repeated initial sounds.
- **Rhyme chains**: end-sound repetition.
- **Antonymy**: presence of contrasting words.
- These are surface, content-independent markers of "joke-like" text.

### Content-Based Features
- Bag-of-words / n-grams trained on jokes vs. non-jokes.
- Vocabulary characteristic of humor.

### Human-Centeredness
- Jokes tend to be about people more than things.
- Pronouns, relational words, body parts overrepresented.

### Negative Polarity
- Slight bias toward negative words/sentiment.
- Sex, violence, body humor, embarrassment dominate joke vocabularies.

### Sentiment / Polarity
- Often punchlines flip polarity from setup.

## Methods
- Naïve Bayes
- Support Vector Machines
- Trained on one-liner corpus vs. proverbs/news.

## Implications for "What Makes a Joke Funny"
- **Sound matters**: alliteration, rhyme.
- **Topic matters**: humans, sex, body, taboo.
- **Polarity contrast** between setup and punch.
- **Antonymy** (script opposition surface marker).

## Rubric
1. Does the joke use phonetic devices (alliteration, rhyme)?
2. Are humans / human relationships at the center?
3. Are taboo or charged topics involved?
4. Is there polarity contrast between setup and punch?
5. Are antonymous words present?

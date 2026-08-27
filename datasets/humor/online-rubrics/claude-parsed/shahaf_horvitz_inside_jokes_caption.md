---
source_url: https://erichorvitz.com/phumor.pdf
title: Inside Jokes — Identifying Humorous Cartoon Captions (Shahaf, Horvitz, West)
source_type: academic_paper
fetched: 2026-05-09
---

# Inside Jokes: Identifying Humorous Cartoon Captions

By Dafna Shahaf, Eric Horvitz, Robert West (KDD 2015).

## Approach
- Studied the New Yorker Caption Contest with crowdsourced humor judgments.
- Built a classifier to identify funnier captions automatically.
- Goal: reduce the load on the cartoon contest's judges.

## Linguistic Features that Predict Funnier Captions

### Lexical / surface features
- **Caption length**: short captions tend to be funnier.
- **Word frequency**: less common words can be funnier (or more pretentious).
- **Concreteness**: more concrete words often funnier.

### Perplexity-based features
- **Negative log-likelihood under a language model**: surprisingness predicts humor.
- Funnier captions often have unexpected words at the end.

### Semantic distance
- **Distance from the scene**: captions semantically distant from what the cartoon literally depicts (less descriptive) tend to be funnier.
- The caption should not just describe what's already obvious in the cartoon.

### Sentiment
- Positive sentiment captions can be funnier in some contexts.
- Negative sentiment can work for darker cartoons.

### Personification / voice
- Captions in first person (a character speaking) often funnier.
- Specific, named, characterized speakers > vague.

### Incongruity
- Mixing terms from different domains (incongruity-based features).
- Domain-mixing predicts humor.

## Rubric
1. Is the caption short and tight?
2. Does it avoid simply describing the scene?
3. Are domains mixed (incongruity)?
4. Is the language surprising at the punch position?
5. Does a clearly characterized speaker voice the line?

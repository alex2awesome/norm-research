---
title: "Engelthaler & Hills: Humor Norms for 4,997 English Words"
source_type: "psycholinguistics"
source: "Tomas Engelthaler & Thomas T. Hills, 'Humor Norms for 4,997 English Words,' Behavior Research Methods 50: 1116-1124 (2018)"
url: "https://link.springer.com/article/10.3758/s13428-017-0930-6"
era: "21st_century"
tradition: "computational_psycholinguistics"
---

# Engelthaler & Hills: Word-Level Humor Ratings

Engelthaler & Hills (2018) provide the largest published normative dataset for word-level funniness, rating 4,997 English words on a 1-5 scale of how humorous they are in isolation. This enables word-level computational and psycholinguistic studies of humor.

## The dataset

- 4,997 English words from a common normed pool (overlapping with arousal, valence, dominance, concreteness, age-of-acquisition, reaction-time databases).
- 821 participants on Amazon Mechanical Turk, each rating 211 words.
- Scale: 1 (humorless) to 5 (humorous).
- Includes demographic breakdowns by gender, age, education.

## The funniest and least-funny words

- **Funniest:** "booty" (4.32), "tit" (4.22), "booby" (4.19), "hooter" (4.17), "nitwit" (4.10), "twit" (4.05), "waddle" (4.00), "tinkle" (3.98), "bebop" (3.97), "egghead" (3.94).
- **Least funny:** "torture" (1.26), "gunshot" (1.31), "nightmare" (1.33), "abortion" (1.34), "rape" (1.36), "incest" (1.42).

## What predicts word funniness

The strongest empirical correlate with humor rating is **inverse word frequency** (r = -0.42 with British National Corpus frequencies):

- Less common words tend to be funnier.
- Familiar words tend to be unfunny.
- This is a robust, large effect.

## Weak predictors

- Valence (positive/negative emotional tone): only weak correlation.
- Arousal: weak.
- Concreteness: weak.

So funniness is largely *not* predictable from semantic content; it has its own dimension.

## Why low frequency works

Several explanations are plausible:

- **Surprise** — rare words violate expected vocabulary, produce mild surprise.
- **Distinctiveness** — rare words stand out, attract attention, create separate processing.
- **Phonetic markedness** — rare words tend to have unusual phonological properties (clusters, sounds) that themselves cue play.
- **Cultural loadings** — many low-frequency words are slangy, archaic, technical, or playful in register.

## The phonetic/iconic correlate

Follow-up work (Westbury and others) has linked word funniness to specific phonological features:
- Voiceless plosives (/k/, /p/, /t/) → funnier.
- Sounds with bilabial articulation (/b/, /p/, /m/) → funnier.
- Reduplication (booby, hooter, bebop, tinkle) → funnier.
- Long vowel + plosive endings → funnier.

## The "funny words" practical principle

When choosing words for a comic line:

- All else equal, pick the less common synonym.
- Pick words with humor-correlated phonemes (k-sounds especially).
- Pick concrete, picturable, low-register words over abstract or formal ones.
- Pick words with reduplication or playful sound structure.

## What this dataset enables

- **Word-level humor scoring** for any text via lookup.
- **Computational humor models** can use these norms as features.
- **Cross-cultural / cross-dialect comparisons** (e.g., subsequent work compared American, British, Singapore English ratings).

## What this excludes

- Context effects: the same word in different contexts can be more or less funny.
- Compositional effects: combinations of words can be funnier than the sum of parts.
- Joke-level funniness: word-level ratings don't predict joke-level success.

## Diagnostic questions

- Does the joke use lexically-funny words where there is choice?
- Does the joke avoid lexically un-funny words (especially high-frequency abstract nouns) at climactic positions?
- Is the comic word a low-frequency, phonetically-marked, reduplicated, or k-containing word?
- Does substituting a synonym change the comic effect — and which substitution makes it funnier?

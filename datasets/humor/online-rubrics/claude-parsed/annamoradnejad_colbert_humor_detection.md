---
title: "Annamoradnejad & Zoghi (ColBERT): Parallel BERT for Computational Humor Detection"
source_type: "computational_nlp"
source: "Issa Annamoradnejad & Gohar Zoghi, 'ColBERT: Using BERT Sentence Embedding in Parallel Neural Networks for Computational Humor,' Expert Systems with Applications 249: 123685 (2024); arXiv:2004.12765 (2020)"
url: "https://arxiv.org/abs/2004.12765"
era: "21st_century"
tradition: "computational_humor_recognition"
---

# ColBERT: BERT-Based Architecture for Humor Detection

Annamoradnejad & Zoghi (2020/2024) propose ColBERT, an architecture for detecting humor in short texts. The design is explicitly grounded in script-opposition theory of humor: separate sentences are embedded independently and then compared in a network that detects incongruity.

## The architecture

1. **Sentence segmentation** — input text is split into its constituent sentences.
2. **BERT embedding per sentence** — each sentence is independently encoded with BERT, producing a fixed embedding.
3. **Parallel hidden layers** — each sentence embedding feeds its own line of hidden layers (one parallel "channel" per sentence).
4. **Concatenation** — the parallel channels' outputs are concatenated.
5. **Classification head** — a final layer produces the humor label or score.

## The theoretical motivation

The architecture mirrors the **script opposition** principle of SSTH/GTVH:
- Each sentence carries a separate script.
- Humor arises from the *relation between* sentence-level scripts (especially incongruity).
- Embedding sentences separately and then combining preserves the script-level information that pooling the whole text would destroy.

## The dataset (200K humor pairs)

The paper releases a dataset of 200,000 short formal texts:
- 100,000 humorous (curated jokes).
- 100,000 non-humorous (matched length and style from news headlines and other formal sources).
- Available for benchmark testing.

## Performance

ColBERT outperforms baselines (single-channel BERT, traditional ML) on humor detection tasks. The architecture's design choice (parallel channels for sentence-level scripts) is the source of the gain.

## What this implies about humor

The success of ColBERT confirms several humor-theoretic predictions:

1. **Scripts are real cognitive structures** that operate at sentence (or near-sentence) granularity.
2. **Humor depends on between-script relations** (especially incongruity), not just within-text features.
3. **Architectures that preserve script-level structure outperform** architectures that merge text into a single representation.
4. **A two-script structure is detectable from text** — the model learns to recognize the asymmetry between setup-script and punchline-script.

## What ColBERT detects (operationally)

Empirically, ColBERT learns to detect:

- **Sentence-level incongruity** — the second/last sentence shifts the interpretive frame.
- **Lexical surprise patterns** — particularly low-frequency or phonetically marked words at punchline position.
- **Structural patterns** of joke text (length distributions, sentence count, ending position of comic load).
- **Domain-specific markers** (specific topics frequently associated with jokes).

## Limitations the model exposes

The architecture works on:
- Short jokes with clear sentence-segmented structure.
- Setup-punchline format.

It struggles with:
- Long-form humor (where Attardo's strands/jab lines distribute humor across many sentences).
- Subtle/ironic/pragmatic humor where surface features are unrevealing.
- Humor requiring world knowledge or cultural background not in the training data.

## Implications for "what makes a joke funny"

A joke is detectable as humorous to the extent that:

1. **Multiple sentences are present** with internally coherent meaning.
2. **The punchline sentence's embedding is meaningfully different** from the setup's (script asymmetry).
3. **Lexical and structural features at the punchline** match learned humor patterns (length, position, vocabulary distinctiveness).

## Practical principle

For computational humor production / evaluation:

- Generate or evaluate text with sentence boundaries clear.
- Maintain the setup-punchline asymmetry between sentences.
- Place comic-significant words at the punchline position, not in the setup.
- Use sentence-separable scripts that contrast cleanly.

## Diagnostic questions

- Are sentences within the joke individually coherent under their own script?
- Do the sentence embeddings reflect a meaningful asymmetry between setup and punch?
- Is the comic load concentrated in the final sentence (or last clause)?
- Could a parallel-architecture model detect this as humor based on between-sentence relations?

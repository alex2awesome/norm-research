---
source_url: https://schwegmanlundberg.com/blog/ai-patent-drafting-best-practices
title: Schwegman Lundberg Woessner - AI Patent Drafting Best Practices
source_type: law_firm
patent_office: USPTO
fetched: 2026-05-09
---

# Schwegman Lundberg Woessner: AI Patent Drafting Best Practices

## Frame the Invention as a Technical Improvement

Post-Alice, AI patents need a clear technical-improvement narrative. Examples that survive 101 scrutiny: a new model architecture that reduces inference latency; a new training procedure that uses less memory; a new data augmentation strategy that improves robustness; a new neural network module specialized for a domain.

## Avoid Pure Math Claims

A claim that recites only mathematical operations (matrix multiplications, activation functions) without anchoring to a technical use case is a textbook abstract idea. Anchor every AI claim to (a) what the model does in operational terms, (b) what hardware/system it runs on, and (c) what technical improvement it provides.

## Disclose the Model Architecture

For an AI patent, the spec should disclose the model architecture in enough detail that a POSITA could reproduce it. Not full source code; but layer counts, layer types (conv, attention, MLP), connection patterns, hyperparameter ranges, training data characteristics.

## Recite Specific Architectural Choices in Claims

A claim that recites "a transformer with 12 layers and 768-dimensional embeddings, where attention is computed using a sparse pattern that..." is more eligible than "a neural network that predicts X." Specific architecture buys eligibility and distinguishes prior art.

## Training-Phase vs. Inference-Phase Claims

Different infringers exist at training time (the company that trained the model) vs. inference time (the company that deploys it). File parallel claims for both.

## Claim the Trained Model as Data

A claim to "a non-transitory computer-readable medium storing model parameters trained according to method X" reaches model distribution as well as model use.

## Disclose the Training Data Characteristics

Not the full dataset, but its characteristics: size, modality, label distribution, augmentation strategy. Training data choices are often inventive and supportable as claim limitations.

## Address the "Black Box" Problem

A claim that recites "a model trained to determine X" without describing the training procedure may face 112 enablement challenges - especially if the result is not reproducible. Disclose the training objective, loss function, and key hyperparameters.

## Federated and Distributed Training

For inventions involving federated learning or distributed training, draft method claims with single-actor architecture: have the orchestrator perform all the steps that combine data, even if the underlying gradients come from multiple clients.

## Foreign Filing Considerations

EPO is generally more favorable to AI patents than USPTO post-Alice; CNIPA has its own AI-specific guidance. Tailor the spec to support both technical improvement framing (US/EPO) and the EPO-specific "further technical effect" framework.

## Open-Source Model Risk

If the foundational model architecture is open-source (e.g., a Llama variant), claims must focus on what the inventor added: the fine-tuning data, the prompt strategy, the post-processing, the system architecture. Claims that read on the unmodified foundation model are likely anticipated.

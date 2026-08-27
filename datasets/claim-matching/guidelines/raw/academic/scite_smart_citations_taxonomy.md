# scite: A smart citation index that displays the context of citations and classifies their intent using deep learning

SOURCE_URL: https://direct.mit.edu/qss/article/2/3/882/102990/scite-A-smart-citation-index-that-displays-the
DOMAIN: academic

## What this source is

The peer-reviewed paper (Nicholson, Mordaunt, Lopez, Uppala, Rosati, Rodrigues, Grabitz & Rife, *Quantitative Science Studies*, 2021) behind scite.ai's "Smart Citations" product — the most widely deployed commercial/production system that automatically classifies, at web scale, whether a citing sentence supports, contradicts, or merely mentions a cited work. It is the clearest example of a deployed, three-way claim-to-citation classification taxonomy operating over hundreds of millions of real citation statements, and it comes with published information about the classifier's real-world label distribution and known reliability limitations — useful both as a taxonomy reference and as a cautionary data point about base-rate imbalance and misclassification risk in this task.

## The three-way taxonomy

Every citation statement (the sentence in a citing paper that references another work) is classified into exactly one of:
- **Supporting** — the citing sentence's claim is backed up by the cited work; the two are in agreement.
- **Contrasting** (also referred to as "disputing" / "contradicting" in related product documentation) — the citing sentence's claim conflicts with, disputes, or contradicts findings in the cited work.
- **Mentioning** — the citation exists (the cited work is referenced) but the surrounding text does not take a clear evidentiary stance either way; it's a neutral/background reference rather than an explicit claim of support or contradiction.

The system is built on a SciBERT-based deep-learning classifier trained to make this three-way call directly from the citation-context sentence(s), rather than requiring a human to read the full cited paper — i.e., it operates as a lightweight context-only classifier, not a full-document entailment system. At the scale described in the paper, the underlying corpus spans over 25 million full-text scientific articles, yielding a database of more than 880 million classified citation statements — this is orders of magnitude larger than any manually-annotated citation-accuracy corpus, illustrating the tradeoff between manual, high-fidelity annotation (as in the medical quotation-accuracy literature and the biomedical NLP annotation scheme) and automated, web-scale-but-noisier classification.

## Real-world label distribution (base-rate skew)

Across the classified corpus, the average distribution of citation statement classifications is approximately:
- **92.6% mentioning**
- **6.5% supporting**
- **0.8% contrasting**

This is an important base-rate fact for anyone designing a claim-matching evaluation or metric around a supporting/contrasting/mentioning-style taxonomy: the overwhelming majority of real citations in the wild are neutral background mentions, not explicit evidentiary claims. A metric or classifier evaluated only on raw accuracy against this distribution could score >92% by always predicting "mentioning," so any evaluation of a support/contradiction classifier needs a metric robust to this skew (e.g., per-class recall/F1, or balanced accuracy) rather than raw accuracy — directly consistent with the general principle that threshold-free, per-class readouts are needed whenever label prevalence is this imbalanced.

## Known classifier reliability limitations

Independent assessment of the classifier's outputs found meaningful misclassification concentrated in the "mentioning" bucket in particular: of 96 citations that scite's classifier labeled as "mentioning," reviewers judged that 40 were more appropriately classified as "supporting" and 17 were more appropriately classified as "contrasting." In other words, over half of a sampled "mentioning" bucket was arguably mislabeled — the model appears biased toward the (correctly) most common label, and struggles most at correctly promoting a citation out of the neutral "mentioning" bucket into an explicit stance category. This is a concrete, citable data point that automated context-only (sentence-level, no full-document read) classifiers systematically under-detect explicit support/contradiction and over-predict the neutral default class — a failure mode any claim-matching metric relying on short citation-context windows (rather than full source verification) should anticipate and audit for.

## Practical takeaways for a claim-matching metric bank

1. **Three coarse stance categories — supporting / contrasting / mentioning —** is a proven, deployed taxonomy distinct from (and coarser than) the support/partial-support/no-support "does this citation prove the claim is true" taxonomies used in academic quotation-accuracy audits. scite's taxonomy is about the citing author's *stated intent/relationship* toward the cited work, not strictly about whether the cited work's content actually substantiates the specific claim text — a subtly different target that's worth distinguishing when building a claim-matching metric bank (stance classification vs. actual evidentiary verification).
2. Expect **severe class imbalance** (~93% neutral/mentioning) in any citation corpus sampled from real papers; design and report metrics accordingly (per-class precision/recall, not raw accuracy).
3. Sentence/context-window-only classifiers (no full source-document verification) have a documented tendency to **under-classify into the neutral "mentioning" bucket** rather than correctly detecting explicit support or contradiction — treat "mentioning" predictions from such a classifier as lower-confidence and worth secondary review, not as a settled negative.
4. At production scale, this task has been operationalized on **880M+ citation statements** using automated classification alone — a benchmark for how far this problem has already been pushed in industry, useful context when scoping the ambition of a research-grade claim-matching metric.

# Assessing citation integrity in biomedical publications: corpus annotation and NLP models

SOURCE_URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11231046/
DOMAIN: academic

## What this source is

A biomedical NLP paper that builds a labeled corpus for automatically detecting whether a citing sentence's claim is accurately supported by its cited reference article, then trains/evaluates models (including a claim-verification architecture and GPT-4) to predict the label. This is the most directly transferable source for a claim-matching metric bank because it operationalizes an eight-category annotation scheme with explicit natural-language definitions, reports real inter-annotator agreement numbers per category, and gives a concrete mapping down to a 3-way SUPPORT / REFUTE / NOT-ENOUGH-INFO scheme for modeling.

## The eight-label citation-accuracy taxonomy

The scheme is organized hierarchically: one "accurate" label, three "major error" labels, and four "minor error" labels.

**Accurate**
- **ACCURATE**: "The citation context is consistent with an evidence segment in the reference article." — i.e., there exists a sentence/segment in the cited paper that entails the citing sentence's claim.

**Major errors** (most severe — treated with priority in labeling, i.e., if a citation instance qualifies for a major-error label, that takes precedence over any minor-error label):
- **CONTRADICT**: "The citation context contradicts a statement made in the reference article." — the cited source actively says the opposite of what's claimed.
- **NOT_SUBSTANTIATE**: "The citation is relevant to the content of the reference article but the cited reference fails to substantiate all statements made in the citing paper." — topically on-target, but the specific claim isn't actually backed by anything in the source (a common real-world failure mode: right paper, wrong/absent evidence).
- **IRRELEVANT**: "There is no information in the reference article relevant to the citation." — wrong source entirely; nothing in the cited work bears on the claim.

**Minor errors:**
- **OVERSIMPLIFY**: "The findings of the reference article are oversimplified or overgeneralized." — the cited evidence is real but the citing sentence stretches it beyond its actual scope/conditions (a very common quotation-error subtype — e.g., turning a conditional/qualified finding into an unconditional one).
- **MISQUOTE**: "The numbers or percentages are misquoted." — a specific quantitative distortion category, kept separate from qualitative oversimplification.
- **INDIRECT**: "The evidence segment includes a citation to other articles, indicating that the reference article is not the original source." — the cited paper is itself just citing someone else for the evidence (secondary/chained sourcing), mirroring the "indirect reference" category found in the medical quotation-accuracy literature.
- **ETIQUETTE**: "The citation style is ambiguous and it is unclear what is being cited from the reference article." — a citation-practice/clarity problem (e.g., citing an entire paper for a specific number) rather than a truth-value problem per se.

## Corpus statistics

- 3,063 annotated citation instances (3,420 citation-context sentences; 3,791 evidence sentences), drawn from 100 reference articles.
- Label distribution: **60.82% accurate; 18.02% major errors; 21.16% minor errors** — i.e., roughly 2 in 5 citation instances in this biomedical sample have *some* form of quotation-accuracy problem, consistent with error-rate magnitudes reported elsewhere in the medical quotation-accuracy literature.

## Collapsed label mapping (for modeling as a 3-way claim-verification task)

The paper explicitly collapses its 8 fine-grained labels into a 3-way scheme compatible with standard scientific-claim-verification setups (e.g., SciFact-style SUPPORTS/REFUTES/NOINFO):
- **Support** = ACCURATE + INDIRECT (i.e., indirect/secondary sourcing is still counted as "supported," just flagged for provenance, not truth)
- **Refute** = CONTRADICT + NOT_SUBSTANTIATE + OVERSIMPLIFY + MISQUOTE + ETIQUETTE (note: this lumps genuine contradiction together with overgeneralization, quantitative misquotation, and mere citation-style ambiguity into one bucket — a modeling simplification worth flagging if reusing this mapping, since these are substantively different failure types)
- **Not Enough Information** = IRRELEVANT

## Annotation process (useful as a protocol template)

Five annotators (graduate/undergraduate life-science students) worked in three phases designed to bootstrap agreement before scaling to solo annotation:
1. **Phase 1**: all annotators independently label the same 10 articles, then fully reconcile disagreements as a group (calibration phase).
2. **Phase 2**: pairwise annotation with reconciliation, scaled to 20 articles.
3. **Phase 3**: individual annotation of the remaining 70 articles, with peer review and a final consistency pass.

## Inter-annotator agreement (Cohen's κ)

- **Citation-context identification** (finding the sentence(s) that constitute the citing claim): κ = 0.96 — near-perfect; this is a comparatively easy, mostly syntactic task.
- **Evidence-sentence retrieval** (finding which sentence(s) in the cited paper are the relevant evidence): κ = 0.20 (phase 1) → 0.37 (phase 2) — low, improving with calibration but still only "fair" at best. This is the genuinely hard sub-task: humans disagree substantially about which sentence(s) in a long source paper count as "the" evidence.
- **Accuracy-label assignment** (choosing among the 8 categories): κ = 0.18–0.31 — likewise low/fair, and the paper notes this is comparable to agreement ranges (0.16–0.52) reported for other cross-document linking tasks in the literature.

This is an important calibration point for a claim-matching metric bank: even trained domain-expert annotators achieve only fair agreement on fine-grained accuracy labeling and on locating supporting evidence within a long source — the "obviously easy" part of the pipeline is only finding *which sentence* claims something, not judging *whether the source backs it up*.

## Model results (as an upper-bound reference for automated claim-matching)

- Fine-tuned PubMedBERT for citation-context sentence classification: F1 = 0.94 (the easy sub-task, consistent with the high human κ above).
- Evidence-sentence retrieval (BM25 + MonoT5 reranker): recall@20 = 0.54; MRR = 0.31 — far from solved.
- Best end-to-end accuracy classifier (MultiVerS/Longformer, a claim-verification architecture, top-20 retrieved sentences as input): Micro-F1 = 0.59, Macro-F1 = 0.52; per-class F1 was notably weak for the error classes (NOT_ACCURATE F1 = 0.43, IRRELEVANT F1 = 0.42) — i.e., models are much better at confirming "yes this is accurately cited" than at correctly flagging *why* a bad citation is bad.
- GPT-4 in-context learning: Micro-F1 = 0.65, Macro-F1 = 0.45 — better at identifying accurate citations, but still poor at discriminating error subtypes.
- **Oracle** condition (given the correct/gold evidence sentences plus citation context, i.e., retrieval solved): Micro-F1 = 0.75, Macro-F1 = 0.78 — showing that most of the current performance ceiling is bottlenecked by *evidence retrieval*, not by the final support/no-support judgment itself.

## Practical takeaways for a claim-matching metric bank

1. An 8-way taxonomy (accurate / contradict / not-substantiate / irrelevant / oversimplify / misquote / indirect / etiquette) is a validated, empirically-grounded fine-grained scheme; it collapses cleanly to 3-way SUPPORT/REFUTE/NEI if a coarser metric is needed, but the collapse loses real distinctions (contradiction ≠ overgeneralization ≠ mere ambiguous style).
2. **NOT_SUBSTANTIATE is its own category, distinct from IRRELEVANT** — "the right paper but no actual evidence for this specific claim" is a materially different and probably more common failure mode than "wrong paper entirely," and a claim-matching metric should distinguish them.
3. **OVERSIMPLIFY/overgeneralization** (stripping qualifiers/conditions from a finding) and **MISQUOTE** (numeric distortion) are worth separate categories rather than one generic "misrepresentation" bucket, since they may require different detection strategies (numeric consistency checking vs. scope/qualifier checking).
4. Expect **evidence-sentence retrieval within the source, not final labeling, to be the bottleneck** for both human annotators and models — a claim-matching pipeline should invest disproportionately in surfacing candidate evidence spans, since given gold evidence the labeling task itself is comparatively tractable (oracle Macro-F1 0.78 vs. realistic-retrieval 0.52).

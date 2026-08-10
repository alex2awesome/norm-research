---
source_type: modern_academic_empirical
source_name: Tim Loughran, Bill McDonald — "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks" — Journal of Finance 66(1), 2011
url: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x
also_see: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
domain: empirical financial textual analysis; sentiment dictionaries
era: 2011
collected_for: pickup-worthiness rubric for press releases
---

# Loughran & McDonald (2011) — Financial-domain word lists for press release tone

Loughran & McDonald show that general-purpose sentiment dictionaries (Harvard IV-4) systematically *misclassify* financial language: in 10-Ks, ~75% of words flagged "negative" by Harvard are not negative in financial context. They build domain-specific lists (negative, positive, uncertainty, litigious, modal-strong, modal-weak) that have become the standard for financial-PR text analysis.

## Loughran-McDonald rubric criteria for earnings/financial press releases
1. **Domain-correct tone audit.** Use the Loughran-McDonald financial dictionary, not general-purpose sentiment lexicons, when auditing release tone — analysts and algorithms increasingly do.
2. **Negative-word minimization (where genuine).** Words on the L-M negative list ("loss", "decline", "litigation", "impairment") will be picked up by algorithmic readers and amplify negative coverage. Use them only where required by the underlying fact.
3. **Litigious-word screen.** L-M's *litigious* list flags legal exposure language. Releases that overuse litigious vocabulary trigger lawyer-side scrutiny and analyst downgrades. Use legal language precisely and only where required by disclosure obligations.
4. **Uncertainty-word calibration.** L-M's *uncertainty* list ("approximately", "could", "may") is associated with higher return volatility around announcements. Use uncertainty markers where actually required; do not pad.
5. **Modal-strong vs. modal-weak.** "Will", "must", "shall" (modal-strong) signal commitment; "may", "might", "could" (modal-weak) signal hedging. Releases over-relying on modal-weak phrases are downgraded by algorithms as evasive.
6. **Algorithm-readability awareness.** Modern earnings-release pickup includes algorithmic news consumers (high-frequency-trading text-analytics pipelines). Releases written without awareness of algorithmic tone-scoring underperform their financial peers.
7. **Verb-tense disambiguation.** Past-tense factual vs. forward-looking should be lexically distinguishable; ambiguity invites algorithmic mis-tagging.
8. **Sentence-length control.** Algorithmic parsers favor short sentences; press releases with long, embedded clauses are mis-parsed and tone-mis-scored.
9. **Internal-consistency check.** Tone in headline, lead, and body should converge. Algorithms compute weighted-tone scores across sections; inconsistency triggers anomaly flags that move prices.

## Pickup-worthiness implication
For financial press releases in the post-2010 algorithmic-trading era, Loughran-McDonald-aware writing is a precondition for accurate market reception. A release that fails the domain-correct tone audit will be mis-scored by algorithms and may move the stock in unintended directions; a release that passes the audit gets correctly-priced market reception and is picked up by more sophisticated outlets that themselves use L-M tone analytics.

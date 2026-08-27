# Bluebook Introductory Citation Signals: A Graded Taxonomy of Degree-of-Support

SOURCE_URL: https://tarlton.law.utexas.edu/bluebook-legal-citation/intro-signals
DOMAIN: academic

## What this source is

A law-library research guide explaining the Bluebook (the standard US legal-citation manual) system of "introductory signals" — the words placed immediately before a legal citation (e.g., *See*, *Cf.*, *But see*) that explicitly encode, in a controlled vocabulary, exactly how strongly and in what direction the cited authority relates to the proposition it follows. This is directly relevant to claim-matching as a real, long-standing, human-designed **graded taxonomy of citation-support strength** that predates and parallels the SUPPORTS/PARTIALLY-SUPPORTS/CONTRADICTS schemes independently arrived at in medical and NLP citation-accuracy research — except here the gradation is a *required, explicit, standardized annotation* that the citing author themselves must select for every citation, rather than something inferred after the fact by an auditor.

## The signal taxonomy, graded by degree and direction of support

**Direct support (no inferential gap)**
- **[No signal]** — used when the authority "directly states the proposition," provides a direct quotation, or is itself the source being identified in the text. This is the strongest possible support signal: no signal at all is required precisely because the source-to-claim mapping is unambiguous and direct.
- **E.g.** ("exempli gratia" / "for example") — the cited authority states the proposition, and citing other authorities that also state it would be unhelpful/redundant. Direct support, explicitly flagged as merely one of several available equally-direct examples.
- **Accord** — used (typically after a quotation or explicit statement from one jurisdiction/authority) to cite additional authorities that also directly state the same proposition — direct, corroborating support from multiple sources.

**Indirect support (an inferential step required)**
- **See** — "the cited authority clearly supports a proposition but there is an inferential step between the proposition as stated and the cited authority." This is the single most common signal and is the explicit legal-citation term for what the medical/NLP literature calls "partial support" or requires "interpretation" to connect — the source doesn't say the claim verbatim, but a reader can clearly derive the claim from it.
- **See also** — cites additional/supplementary authority supporting a proposition *already* supported by a preceding *See* or [no signal] citation; requires an explanatory parenthetical stating the relevance of the extra material. Functions as corroborating-but-secondary support.

**Analogous support (support by comparison, not directly on point)**
- **Cf.** ("confer" / "compare") — introduces a source that supports a *different but sufficiently analogous* proposition, such that it indirectly supports the stated proposition by analogy. This is a distinct and useful category for claim-matching: a citation can be a legitimate, honest citation while still not being *directly* on-point — it supports a structurally similar claim, and the burden is on the reader/author to draw the analogy explicitly (a required parenthetical typically explains the analogy).
- **Compare ... with ...** — a comparison between two or more sources is offered as illustrating or supporting the proposition (e.g., contrasting two cases whose difference in outcome illustrates the point); also requires an explanatory parenthetical.

**Contradiction (direct and indirect)**
- **Contra** — the cited authority "directly states a contradictory proposition" — the direct-contradiction analog of [no signal], i.e., maximally strong disagreement with no inferential gap.
- **But see** — the cited authority "clearly supports a proposition contradictory to the textual assertion," but (mirroring *See*) with an inferential step required to get from the source to the contradiction — the indirect-contradiction analog of *See*.
- **But cf.** — an authority that is "analogously contradictory" — supports, by analogy, a proposition that conflicts with the stated one. The contradiction-side analog of *Cf.*

**Background only (no support/contradiction claim at all)**
- **See generally** — cited for background/context reading; explicitly does *not* claim to directly support or refute the specific proposition, only to provide useful related context. Requires an explanatory parenthetical. This is the legal-citation equivalent of scite.ai's "mentioning" category — a citation present in the text but making no explicit support/contradiction claim.

## Why this taxonomy is useful for a claim-matching metric bank

1. It is a **complete, symmetric lattice**: for every support-strength tier (direct / inferential / analogical) there is a mirrored contradiction-strength tier (direct / inferential / analogical), plus a separate neutral/background-only tier. This symmetric structure (support ↔ contradiction, each graded by directness) is more systematic than most ad hoc citation-error taxonomies and is a good template for a claim-matching label schema that wants to capture both *strength* and *polarity* of source-claim relationship in one scheme.
2. It distinguishes **"directly states" vs. "requires an inferential step" vs. "supports only by analogy"** as three genuinely different strengths of support — a finer-grained version of the "fully substantiated / partially substantiated" split used elsewhere, and arguably more precise about *why* something is partial (is it an inference gap, or is it merely analogous rather than on-point?).
3. It formally requires an **explanatory parenthetical** for every signal beyond the strongest tier (*See also*, *Cf.*, *Compare...with*, *But cf.*, *See generally*) — i.e., the citing author must, in the citation itself, articulate *why* the source counts as supporting/contradicting/relevant. This "show your work" requirement is a useful design principle for any claim-matching pipeline: a good claim-citation-support judgment should come with a stated reason, not just a categorical label.
4. Because this is a citing-author-selected signal (not a third-party auditor's post-hoc judgment), it is a rare source of **ground truth about intended citation relationship at time of writing** — useful as a natural label source if ever mining legal texts for claim-citation pairs, since the signal itself already encodes the author's own claimed support/contradiction/background distinction.

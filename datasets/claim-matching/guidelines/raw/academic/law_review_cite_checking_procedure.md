# Law Review Cite-Checking: Verifying That a Source Actually Supports the Proposition

SOURCE_URL: https://guides.law.stanford.edu/c.php?g=1203858&p=8805203
DOMAIN: academic

## What this source is

A law-school library guide describing "cite-checking" — the formal, mandatory quality-control process that student editors at academic law journals perform on every footnote of every article before publication. Unlike most citation-accuracy research (which audits *published* articles after the fact), cite-checking is a **pre-publication verification procedure**: a person other than the author is assigned to pull every single cited source and confirm, footnote by footnote, that it actually supports the sentence it's attached to. This makes it one of the only widely-institutionalized, mandatory, human-in-the-loop claim-citation verification workflows in academic publishing, and a useful source of actionable procedural steps (not just post-hoc error taxonomy). (A parallel guide, the University of San Diego law journal cite-checking guide, describes essentially the same workflow and is referenced below for corroboration: https://lawlibguides.sandiego.edu/c.php?g=1453723&p=10806215.)

## Definition

Cite-checking is described as "a method of verifying an author's statements are adequately supported by other sources and the citation conforms to the appropriate style guide." Two distinct jobs are bundled into one role: (1) **substantive verification** — does the source actually support the claim — and (2) **formatting verification** — is the citation itself correctly styled. The guide is explicit that cite checkers must "verify the quoted or paraphrased text. The passage cited must actually appear in the source and support the author's point" — i.e., verification is not satisfied merely by confirming the source exists and is correctly formatted; the checker must locate the specific passage and confirm it says what the author claims it says.

## The four-part quality standard ("good" citation)

The guide frames the goal of cite-checking around four named criteria a citation must satisfy:
- **Accurate** — "the citation content and format leads the reader to the correct source" (a reader who follows the citation ends up at the actual cited material).
- **Valid** — "the source actually supports the author's claim/statement" (the core claim-matching criterion).
- **Non-plagiarized** — every proposition in the text receives proper attribution (no uncredited borrowing).
- **Useful** — the citation "provides value to the reader," i.e., is not a superfluous or decorative citation.

Of these four, **"valid"** is precisely the claim-matching property of interest — explicitly separated out from mere bibliographic accuracy, plagiarism-avoidance, and reader-usefulness. This four-way decomposition is a clean, reusable checklist frame: any citation-quality metric could report these as four separate axes rather than one composite score, since a citation can fail on any one independently of the others (e.g., a citation can be "accurate" — correctly leads to the right source — while being "invalid" — the source doesn't actually say what's claimed).

## The four-question checklist actually applied to each footnote

The guide reduces the process to four sequential questions asked of every citation:
1. **"Is a citation needed?"** — a threshold/necessity check before verification even begins (is this actually a claim requiring source support, e.g., a factual/empirical assertion rather than the author's own original argument).
2. **"Does the cited authority support the statement made in the text?"** — the core claim-matching verification question.
3. **"Are the introductory signals and parentheticals appropriate according to Bluebook Rule 1?"** — i.e., given that the source does or doesn't directly/indirectly/analogically support the claim (see the companion note on Bluebook signals), has the author selected the *correct signal* (no signal / *See* / *Cf.* / *But see* / etc.) to represent the true strength of that support? This step effectively re-uses the graded support-signal taxonomy as a checkable output of the verification process, not just an input.
4. **"Does the citation include all necessary content (including pincites) and is it formatted properly?"** — the pure bibliographic/formatting check, explicitly last and separate from substantive validity.

## Step-by-step verification procedure

1. **Locate and pull the source** from an online database or in print — the guide specifies checkers must use "the final, publisher's version with enough information to create relevant pincites" (i.e., not a preprint or draft version that might differ from what's actually cited, and not a version lacking page/section granularity needed to point to the exact supporting passage).
2. **Verify the source is accurate and supports the propositions made in the text** — read the actual passage, not just confirm the document exists.
3. Cross-reference: **"Do the claims presented in the article match the claims made in or about the cited source?"** — phrased as a bidirectional match check (the citing text's characterization of the source, and the source's actual content, must correspond), plus confirm the version cited is the most current/authoritative available.
4. **Pincite to the exact location** (page number, statute section, etc.) within the source that supports the specific proposition — a citation to an entire 40-page article without a pincite to the actually-relevant page is treated as inadequate, since it doesn't let a reader (or a downstream auditor) efficiently verify the specific supporting passage.
5. If a source is ambiguous or hard to locate, escalate (consult a reference librarian, dictionary of legal abbreviations, alternative search) rather than approving on partial verification.

## Practical takeaways for a claim-matching metric bank

1. Treat **"does the source exist / is correctly formatted"** and **"does the source actually support the claim"** as two independent axes to score (mirrors the "citation accuracy vs. quotation accuracy" split found independently in the medical literature) — a claim-matching metric should not conflate bibliographic correctness with evidentiary validity.
2. A citation without a **pincite/specific location pointer** to the supporting passage should be treated as harder-to-verify / lower-confidence by default — "cites the whole source" is weaker evidence of true support than "cites the specific page/section," and a claim-matching pipeline should reward or require localization of the supporting span, not just document-level linkage (this directly parallels the "evidence-sentence retrieval" bottleneck found in the biomedical NLP annotation-scheme literature).
3. The **four-criteria decomposition (accurate / valid / non-plagiarized / useful)** is a reusable rubric structure: score claim-citation pairs on "valid" (does the source support the claim) as its own separate axis from formatting/attribution/usefulness.
4. Verification requires checking that the **cited version is the actual final/published version** the passage appears in — an easily-overlooked confound (a claim can be "supported" by a preprint that changed before final publication, which downstream automated matching against only the final version would incorrectly flag as unsupported, or vice versa).
5. Institutionalized cite-checking (independent-checker-verifies-every-footnote-before-publication) is itself a validated real-world design pattern for a claim-matching *process*, not just a *metric* — worth citing as precedent if the research also touches on process/workflow recommendations rather than purely automated scoring.

# Peer review and publishing at eLife, and the eLife Assessment vocabulary
SOURCE_URL: https://elifesciences.org/about/peer-review
DOMAIN: academic

eLife's model (publish first as a "Reviewed Preprint," then attach peer review) forces reviewers to produce two artifacts that are unusually explicit and structured for claim-evidence assessment: an **eLife Assessment** (a short, standardized-vocabulary rating of significance and evidence strength) and a **Public Review** (prose justifying that rating). Both are published alongside the article, making eLife's claim-matching rubric one of the few peer-review instruments that is itself a public, machine-readable artifact.

## What reviewers are asked to determine

Public Reviews must "describe the strengths and weaknesses of the article, and indicate whether the claims and conclusions are justified by the data." This is the operative claim-matching question eLife reviewers answer directly and in public: for each major claim in the paper, is it justified by the data presented, partially justified, or not justified?

"Where the reviewers disagree with the claims in an article, this is made explicit within the eLife Assessment and the Public Reviews" — disagreement about whether a claim is supported is not resolved into a single up/down verdict; it is preserved and surfaced to readers.

## The eLife Assessment: a standardized two-axis rubric

The eLife Assessment is a short (two-to-three sentence) statement that rates the manuscript on two independent axes, each drawn from a fixed, defined vocabulary (source: https://elifesciences.org/about/elife-assessments):

### Axis 1 — Significance of the findings (importance/novelty axis)
- **Landmark**: findings with profound implications that are expected to have widespread influence.
- **Fundamental**: findings that substantially advance our understanding of major research questions.
- **Important**: findings that have theoretical or practical implications beyond a single subfield.
- **Valuable**: findings that have theoretical or practical implications for a subfield.
- **Useful**: findings that have focused importance and scope.

### Axis 2 — Strength of evidence (claim-support axis — the direct claim-matching rubric)
- **Exceptional**: exemplary use of existing approaches that establish new standards for a field.
- **Compelling**: evidence that features methods, data, and analyses more rigorous than the current state-of-the-art.
- **Convincing**: appropriate and validated methodology in line with current state-of-the-art.
- **Solid**: methods, data and analyses broadly support the claims with only minor weaknesses.
- **Incomplete**: main claims are only partially supported.
- **Inadequate**: methods, data and analyses do not support the primary claims.

This second axis is, in effect, an explicit ordinal scale for the exact judgment this claim-matching project is trying to formalize: how well do the presented methods/data/analyses support the paper's primary claims, ranging from "does not support" (inadequate) through "partially supported" (incomplete) to "broadly supported with minor weaknesses" (solid) up to methodology that itself sets a new standard (exceptional). Note that the significance axis is explicitly independent of the evidence axis — a paper can report a highly significant (landmark) claim that is only "incomplete"ly supported, and eLife's format is designed to let reviewers say exactly that rather than collapsing both judgments into one accept/reject decision.

## Historical operating principle

Historically, "if conclusions are not adequately supported by the existing data, the submission should be rejected" under eLife's prior (pre-Reviewed-Preprint) model; referees were directed to limit revision requests to changes that bear directly on major conclusions, and eLife would only require new experiments where the data were essential to support the major conclusions — i.e., a request for more evidence had to be tied to a specific claim it would newly support, not a general call for more rigor.

## Practical implication for a claim-matching metric

eLife's two-axis rubric offers a template that separates "is this an important claim" from "is this claim adequately evidenced" — precisely the distinction a claim-matching system needs to avoid conflating impressiveness of a stated finding with the strength of the evidence actually offered for it. The six-point "strength of evidence" ladder (inadequate → incomplete → solid → convincing → compelling → exceptional) is a ready-made ordinal target for scoring how well a given piece of evidence/citation supports the claim it is attached to.

# The Principles of the Truth-O-Meter: PolitiFact's methodology for independent fact-checking
SOURCE_URL: https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
DOMAIN: journalism

PolitiFact's methodology statement is the most explicit, publicly documented rubric among major outlets for converting an evidentiary assessment of a claim into a discrete rating. It separates the process into three stages — what to check, how to check it, and how to rate it — each with its own criteria.

## Stage 1: What counts as a checkable claim (selection criteria)

Not every statement is eligible for fact-checking. PolitiFact applies several filters before beginning verification work:

- **Verifiability.** The threshold question: **"Is the statement rooted in a fact that is verifiable?"** Statements of pure opinion, or predictions about the future that cannot yet be checked, are explicitly excluded. The methodology also builds in **"license for hyperbole"** for political rhetoric — obvious figures of speech are not treated as factual claims to be checked literally.
- **Significance.** The claim should be one that would, if wrong, actually mislead people — trivial misstatements or slips of the tongue are deprioritized in favor of claims that are substantively "misleading or sound wrong."
- **Reach.** Priority goes to statements "likely to be passed on and repeated by others" — i.e., claims with viral or rhetorical potential, since the harm of an unchecked false claim scales with how far it spreads.
- **Public interest test.** The operational heuristic used by editors: **"Would a typical person hear or read the statement and wonder: Is that true?"** If the claim wouldn't provoke that reaction, it's not a priority.
- **Newsworthiness / balance.** Claims tied to current news are prioritized, and the outlet explicitly tries to balance coverage across parties, while also checking whichever party currently holds power more frequently (on the theory that claims from those in power carry more real-world consequence).

## Stage 2: How claims are researched (verification process)

Once a claim is selected, the research process includes:
- **Contact the speaker.** Reporters attempt to reach the original speaker (or their office/campaign) to ask for the basis of the claim and any supporting evidence — this mirrors the "right of reply" step also found in Full Fact's process and gives the claimant a chance to substantiate before independent research proceeds.
- **Check prior work.** Reporters review whether the same or a similar claim has already been fact-checked (by PolitiFact or elsewhere), to build on established findings rather than duplicate effort or contradict a previous verdict without cause.
- **Independent search.** Thorough searches across the open web, academic literature, and specialized databases.
- **Expert consultation.** Reporters consult multiple subject-matter experts, deliberately not relying on a single expert's framing.
- **Primary-source preference.** The methodology explicitly states a preference for **"primary sources and original documentation"** over second-hand accounts or another outlet's paraphrase of the underlying document.
- **Independence from the claimant's own framing.** Even when a campaign or speaker provides supporting material, PolitiFact does its own independent verification rather than accepting the claimant's characterization of that material at face value.

## Stage 3: The Truth-O-Meter ratings (the rating rubric)

The six-point scale, each definition given verbatim, in decreasing order of accuracy:

| Rating | Definition |
|---|---|
| **TRUE** | "The statement is accurate and there's nothing significant missing." |
| **MOSTLY TRUE** | "The statement is accurate but needs clarification or additional information." |
| **HALF TRUE** | "The statement is partially accurate but leaves out important details or takes things out of context." |
| **MOSTLY FALSE** | "The statement contains an element of truth but ignores critical facts that would give a different impression." |
| **FALSE** | "The statement is not accurate." |
| **PANTS ON FIRE** | "The statement is not accurate and makes a ridiculous claim." |

Note the structure of this scale for claim-matching purposes: the top three bands (True/Mostly True/Half True) are all "accurate" but are differentiated purely by *completeness and context* — i.e., a claim can be literally true and still receive a lower rating if it omits material context that would change a reader's impression ("takes things out of context" is an explicit criterion at Half True). This means claim-evidence matching under this rubric is not binary (supported/unsupported) but graded on **completeness of support relative to the full available evidence**, and separately, the bottom two bands (False/Pants on Fire) are differentiated purely by the *rhetorical extremity* of the false claim, not by any additional factual criterion — "Pants on Fire" adds no new evidentiary test beyond "False," only a judgment that the claim was especially ridiculous.

## Burden of proof and adjudication rules

- **"The burden of proof is on the speaker"** — the default posture is skeptical; a claim is not assumed supportable unless evidence for it can be found or supplied.
- Claims are rated **based on the information known at the time the statement was made**, not with the benefit of hindsight — a claim that was reasonable given contemporaneous evidence is not penalized for later being overtaken by new facts.
- **Editorial process:** the reporter who researches a claim proposes a rating to an assigning editor. That proposed rating and the underlying research are then reviewed by additional editors who probe specific questions: is the statement literally true or false; are there alternative reasonable interpretations of what the speaker meant; what evidence did the speaker have available at the time; and how does this claim compare to precedent (how similar claims were rated in the past, for consistency). The methodology specifies that **three editors vote on the final rating, with two votes deciding** — an explicit multi-rater agreement mechanism rather than a single fact-checker's unilateral call.

## Relevance to claim-matching

PolitiFact's rubric supplies three transferable design elements for an automated or human claim-matching task: (1) a **pre-filter** for whether a statement is even a "claim" worth checking (verifiable, non-opinion, non-hyperbole, significant); (2) a **graded, not binary, support scale**, where partial/contextual support is a distinct category from full support or full contradiction, and where "true but missing context" is explicitly a lower-support category than "true and complete"; and (3) a **multi-rater adjudication step** (independent reviewers voting, majority deciding) as the mechanism for converting an evidentiary writeup into a final label, rather than trusting a single assessor's judgment.

# GRADE: An emerging consensus on rating quality of evidence and strength of recommendations
SOURCE_URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC2335261/
DOMAIN: academic

This is the seminal GRADE (Grading of Recommendations Assessment, Development and Evaluation) paper (BMJ 2008), which established the framework now used across Cochrane, WHO, NICE, UpToDate, and most clinical guideline bodies for judging how well a body of evidence supports a stated claim about an effect, and how that certainty should translate into the strength of a recommendation. The core actionable move of GRADE is to separate two judgments that are often conflated: (1) how certain are we that the estimated effect is close to the true effect (certainty/quality of evidence), and (2) given that certainty, how strongly should we act on the claim (strength of recommendation). A claim can be well-supported by high-certainty evidence yet still warrant only a weak recommendation (e.g., if benefits and harms are closely balanced), and conversely a claim can carry a strong recommendation even from lower-certainty evidence if the benefit/harm balance is decisive.

## The four certainty (quality) categories

GRADE assigns one of four discrete labels to a body of evidence for a specific outcome — not to an individual study, and not to a review as a whole. Each label carries an explicit epistemic meaning about what future research could still do to the estimate:

- **High quality**: "Further research is very unlikely to change our confidence in the estimate of effect."
- **Moderate quality**: "Further research is likely to have an important impact on our confidence in the estimate of effect and may change the estimate."
- **Low quality**: "Further research is very likely to have an important impact on our confidence in the estimate of effect and is likely to change the estimate."
- **Very low quality**: "Any estimate of effect is very uncertain."

This is the operational test for whether a claim is "strongly," "moderately," or "weakly" supported: ask what a well-conducted additional study would most likely do to the current estimate. If it would probably leave the estimate essentially unchanged, the claim is strongly supported (high certainty). If a new study could plausibly overturn the current point estimate or its direction, the claim is weakly supported (low/very low certainty) regardless of how many studies currently exist or how confidently the original authors state their conclusion.

## Starting point and the five downgrading domains

Randomized controlled trial (RCT) evidence starts at **high** quality; observational (non-randomized) evidence starts at **low** quality, reflecting the greater a priori risk of confounding and selection bias in the latter. From that starting point, the certainty rating is adjusted down (never below "very low") based on five explicit domains. A claim's match to its evidence should be checked against each domain in turn:

1. **Study limitations (risk of bias)** — Are the individual studies contributing to the claim well-designed and well-executed (adequate randomization, allocation concealment, blinding, complete follow-up, and outcome reporting free of selective reporting)? Serious flaws in the studies underlying a claim downgrade certainty even if the claim's chain of reasoning is otherwise correct.
2. **Inconsistency of results** — Do the studies point in materially different directions or magnitudes without an identifiable explanation (e.g., different populations, doses, follow-up length)? Unexplained heterogeneity across the studies cited for a claim is direct evidence that the claim is not well pinned down by the body of evidence.
3. **Indirectness of evidence** — Is the evidence about the actual population, intervention, comparator, and outcome the claim is being made about, or does it require an inferential bridge (e.g., surrogate outcome standing in for a patient-important outcome, or a different population than the one the claim targets)? Any such gap between "what was studied" and "what is claimed" downgrades certainty.
4. **Imprecision** — Is the confidence interval around the pooled/study estimate narrow enough, and the sample/event count large enough, to rule out both a null effect and an effect of the opposite or much larger/smaller magnitude than claimed? Wide intervals mean the claim's specific magnitude (or even its direction) is not actually pinned down by the cited evidence.
5. **Publication/reporting bias** — Is there reason to think unfavorable or null studies were not published or not reported, biasing the visible evidence base toward the claim being tested? This is a threat to the *very existence* of the evidence a claim relies on, not just its internal quality.

Each of these is a genuine downgrade axis: a claim can fail evidence-matching not because the studies are individually bad, but because they disagree, don't directly address what's claimed, are too small/imprecise to support the claimed magnitude, or represent only part of the true evidence base.

## Upgrading criteria (for observational evidence)

Symmetrically, GRADE allows evidence that starts low (observational designs) to be upgraded when the pattern of the data itself provides evidence against alternative (confounding) explanations for the claim:

- **Large magnitude of effect**: a very large or very consistent effect size is harder to explain away via plausible confounding alone.
- **Dose-response gradient**: if the outcome varies systematically with the dose/intensity of exposure, this is evidence for a causal (not merely associative) reading of the claim.
- **Plausible confounding would work against the observed effect**: if all identifiable biases would be expected to reduce or reverse the observed association, yet the association still holds, this is evidence the claim is more secure than the raw study design would suggest.

These upgrade criteria are the actionable answer to "when can weaker-design evidence still strongly support a claim?" — they require an explicit, checkable argument (not just intuition) about why confounding cannot plausibly account for the result.

## From certainty of evidence to strength of recommendation

GRADE explicitly separates the "is the claim well-supported by the evidence" judgment from the "should we act strongly on this claim" judgment. Recommendations are graded as **strong** or **weak** (also called conditional/discretionary), based on four inputs (Table 1 of the paper):

1. **Quality of evidence** — the certainty rating above (high-quality RCTs vs. case series).
2. **Balance of benefits vs. harms/burdens/costs** — e.g., minimal toxicity vs. increased bleeding risk.
3. **Values and preferences** — how much the outcome matters to the people affected (which can vary, e.g., by age or life stage).
4. **Resource use/cost-effectiveness** of the intervention.

A **strong recommendation** is made when the desirable effects of an intervention clearly outweigh the undesirable effects, or clearly do not — i.e., the evidence-to-decision calculus is decisive even allowing for plausible variation in values or additional evidence. A **weak recommendation** is made when the trade-offs are closely balanced or the evidence is of low certainty, such that reasonable people could differ on whether to act, or such that new evidence could plausibly change the decision.

## Actionable takeaways for claim-evidence matching

1. Rate certainty **per outcome/claim**, not per study or per review — a review can make some well-supported claims and some poorly-supported claims simultaneously.
2. Start from study design (RCT=high, observational=low) and then walk through the five downgrade domains explicitly, documenting which ones apply and by how much (one level = "serious," two levels = "very serious").
3. Ask the counterfactual test for certainty: would a new, well-conducted study likely change this estimate? If yes, the claim is not strongly supported no matter how it is worded in the text.
4. Distinguish "the studies are high quality" from "the studies actually address the claim being made" (that's indirectness) and from "the studies agree with each other" (that's inconsistency) — a claim can fail evidence-matching on any one of these axes independently.
5. Recognize that strength of recommendation is not identical to certainty of evidence; a well-supported (high-certainty) claim about a marginal benefit may still only justify a weak recommendation, while a lower-certainty claim about a decisive benefit/harm trade-off can justify a strong one.

# AMSTAR 2: A critical appraisal tool for systematic reviews that include randomised or non-randomised studies of healthcare interventions, or both
SOURCE_URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5833365/
DOMAIN: academic

AMSTAR 2 (Shea et al., BMJ 2017) is a 16-item instrument for judging whether a *systematic review itself* (not the primary studies inside it) was conducted rigorously enough that its stated conclusions can be trusted — i.e., whether the review's claim actually follows from a sound evidence-gathering and synthesis process. It is widely used by guideline panels, umbrella reviewers, and peer reviewers to decide how much weight to give a review's conclusions. Its central actionable contribution is separating methodological flaws into **critical** (fatal to trustworthiness) and **non-critical** (mildly degrading) domains, then providing an explicit algorithm for combining them into a single confidence verdict.

## The 16 checklist items

1. Did the research questions and inclusion criteria for the review include the components of PICO (Population, Intervention, Comparator, Outcome)?
2. Did the report contain an explicit statement that review methods were established prior to conduct, and does it justify any deviations from the protocol?
3. Did review authors explain their selection of study designs for inclusion?
4. Did review authors use a comprehensive literature search strategy?
5. Did review authors perform study selection in duplicate?
6. Did review authors perform data extraction in duplicate?
7. Did review authors provide a list of excluded studies and justify the exclusions?
8. Did review authors describe the included studies in adequate detail?
9. Did review authors use a satisfactory technique for assessing the risk of bias (RoB) in individual studies that were included in the review?
10. Did review authors report on the sources of funding for the studies included in the review?
11. If meta-analysis was performed, did the review authors use appropriate methods for statistical combination of results?
12. If meta-analysis was performed, did the review authors assess the potential impact of RoB in individual studies on the results of the meta-analysis or other evidence synthesis?
13. Did the review authors account for RoB in individual studies when interpreting/discussing the results of the review?
14. Did the review authors provide a satisfactory explanation for, and discussion of, any heterogeneity observed in the results of the review?
15. If quantitative synthesis was performed, did the review authors carry out an adequate investigation of publication bias (small study bias) and discuss its likely impact on the results of the review?
16. Did the review authors report any potential sources of conflict of interest, including any funding they received for conducting the review?

## The seven "critical" domains

Per Box 1 of the paper, these items are judged to fundamentally threaten the validity of a review's conclusions if failed:

- **Item 2** — protocol registered/established before the review began
- **Item 4** — adequacy of the literature search
- **Item 7** — justification for excluding individual studies
- **Item 9** — adequacy of risk-of-bias assessment in included studies
- **Item 11** — appropriateness of meta-analytical methods
- **Item 13** — whether risk of bias was accounted for when interpreting results
- **Item 15** — assessment of publication bias and its impact

The remaining nine items are "non-critical" — their absence degrades quality but does not, on its own, invalidate the review's conclusions. AMSTAR 2 authors explicitly note reviewers should feel free to prespecify their own critical-item set appropriate to the review topic if the default seven do not fit.

## Response categories

Items are rated **"Yes," "Partial Yes,"** or **"No."** The authors deliberately removed "not applicable" and "cannot answer" options to force a judgment; where information is simply unavailable in the report, the rater should assign "No" rather than giving the benefit of the doubt.

## Overall confidence rating algorithm (Box 2)

The 16 item ratings are combined — critical and non-critical together — into one of four overall confidence verdicts about whether the review's conclusions can be trusted as an accurate and comprehensive summary of the included studies:

- **High**: no or one non-critical weakness — "the review provides an accurate and comprehensive summary of the results of the available studies that address the question of interest."
- **Moderate**: more than one non-critical weakness but no critical flaws — "the review may provide an accurate summary of the results of the available studies."
- **Low**: one critical flaw, with or without non-critical weaknesses — "the review may not provide an accurate and comprehensive summary of the available studies."
- **Critically low**: more than one critical flaw, with or without non-critical weaknesses — "the review should not be relied on to provide an accurate and comprehensive summary of the available studies."

Multiple non-critical weaknesses can also justify a discretionary downgrade from moderate to low even absent a critical flaw. Crucially, **no numeric summary score should ever be calculated** from the 16 items — the rating is a qualitative, algorithmic combination, not an average.

## What counts as adequate for the critical items

- **Item 2 (protocol)**: a written protocol (ideally independently registered/verifiable) with any deviations from it explicitly justified in the review report.
- **Item 4 (search)**: search across multiple relevant databases, with particular attention to whether non-randomized/grey literature sources were also searched where relevant.
- **Item 7 (exclusions)**: a full accounting of excluded studies with transparent, study-by-study or category-level reasoning.
- **Item 9 (risk of bias)**: assessment must cover the bias domains specified in validated instruments (e.g., Cochrane RoB tools) — baseline confounding, selection bias, measurement bias, selective outcome reporting — with appropriate, separate tools used for randomized vs. non-randomized designs.
- **Item 11 (meta-analysis methods)**: an explicit, protocol-level statement of the decision rules for whether/how to pool studies, including judgment of clinical and methodological compatibility before pooling.
- **Item 13 (bias in interpretation)**: explicit reference to the risk-of-bias findings when drawing the review's conclusions and making recommendations — i.e., the claim's confidence language must actually reflect the quality of the underlying studies, not just their number or direction.
- **Item 15 (publication bias)**: use of statistical tests or graphical displays (e.g., funnel plot) with discussion of likely impact on the conclusion; the paper notes a practical constraint that funnel-plot asymmetry tests are unreliable with fewer than ~10 studies.

## Implementation notes relevant to claim-evidence matching

- Completion time is roughly 15–32 minutes per review (about double the original AMSTAR).
- Inter-rater reliability is moderate to substantial for most items.
- AMSTAR 2 is **not** designed for diagnostic-accuracy reviews, individual-patient-data meta-analyses, network meta-analyses, scoping reviews, or realist reviews — its critical-item logic assumes a conventional intervention-effectiveness review structure.
- The tool's deepest actionable insight for claim-matching is that a review's stated conclusion can fail to be well-supported for *process* reasons entirely separate from whether individual included studies were high quality: an incomplete search (item 4), unaccounted risk of bias (items 9/13), inappropriate pooling (item 11), or undetected publication bias (item 15) can each independently sever the link between "what the cited studies show" and "what the review claims they show."

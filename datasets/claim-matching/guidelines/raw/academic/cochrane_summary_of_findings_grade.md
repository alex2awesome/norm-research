# Cochrane Handbook Chapter 14: Completing 'Summary of findings' tables and grading the certainty of the evidence
SOURCE_URL: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-14
DOMAIN: academic

This chapter of the Cochrane Handbook for Systematic Reviews of Interventions is the operational manual (authored by Schünemann, Higgins, Vist, Glasziou, Akl, Skoetz and Guyatt for the Cochrane GRADEing Methods Group) for how Cochrane review authors are required to decide whether their synthesized evidence actually supports a stated effect estimate/conclusion, and how confidently. It operationalizes the GRADE approach specifically for meta-analytic (pooled) evidence and mandates it via Cochrane's MECIR conduct standards (C74–C75).

## Definition of certainty of evidence

Cochrane defines certainty of evidence as **"the extent to which one can be confident that an estimate of effect or association is close to the quantity of specific interest."** This is the anchor concept for claim-evidence matching: a "supported" claim is one where the reviewers can be confident the reported number (or direction) is close to the true quantity, not merely one where a number was computed from the included studies.

## Starting point by study design

- **Randomized trials** begin at **HIGH** certainty.
- **Non-randomized studies of interventions (NRSI)** begin at **LOW** certainty by default, reflecting inherent confounding and selection-bias risk — although if using the ROBINS-I risk-of-bias tool, all studies may start at "high" and NRSI are still expected to typically fall two levels once bias is accounted for.

## The four certainty levels and their meaning for claim strength

| Level | Symbol | Meaning |
|---|---|---|
| High | ⊕⊕⊕⊕ | Few/no concerns across domains; we are very confident the true effect lies close to the estimate |
| Moderate | ⊕⊕⊕◯ | Downgraded one level; we are moderately confident, true effect likely close but could be substantially different |
| Low | ⊕⊕◯◯ | Downgraded two levels; confidence in the estimate is limited |
| Very low | ⊕◯◯◯ | Downgraded three levels (the floor); very little confidence in the effect estimate |

## The five domains for downgrading (the actionable checklist)

### 1. Risk of bias
Classified as "no serious limitations" (no downgrade), "serious limitations" (downgrade one level), or "very serious limitations" (downgrade two levels). Specific concerns for randomized trials include: failure to generate a proper random sequence, lack of allocation concealment, inadequate blinding (especially for subjective outcomes), large loss to follow-up (attrition above roughly 50% warrants consideration of a two-level downgrade), and selective outcome reporting. When translating from a structured risk-of-bias tool: "low risk of bias" across studies → no downgrade; "some concerns" that are unlikely to lower confidence → no downgrade; "some concerns" that are likely to lower confidence → downgrade one level; "high risk of bias" with one crucial limitation → downgrade one level; "high risk of bias" substantially weakening confidence → downgrade two levels.

### 2. Inconsistency (heterogeneity)
Downgrade when there is unexplained heterogeneity that affects interpretation of the estimate — i.e., wide variation in effect estimates across studies without a plausible explanation. Assessment relies on the I² statistic (described qualitatively as "not important," "moderate," "substantial," or "considerable"), the Chi² test and tau, confidence-interval overlap across studies, and variation in point estimates. Do **not** downgrade when heterogeneity is explained by identifiable subgroups (consider separate Summary of Findings tables for clearly different populations instead of downgrading), or when only a single study contributes to the outcome (rate inconsistency as "none," not "not applicable").

### 3. Indirectness
Two distinct kinds:
- **Indirect comparison**: no head-to-head trials of A vs. B exist, only A-vs-placebo and B-vs-placebo trials, requiring an indirect comparison.
- **PICO mismatch**: the evidence addresses a narrower or different version of the review's question — a different population (e.g., only diabetic patients when the claim is meant to generalize), a different intervention delivery (specialist vs. general practice), a suboptimal comparator (not reflecting current standard of care), or surrogate/disease-oriented outcomes standing in for the patient-important outcome actually being claimed.

Assess each PICO element separately, then classify as "no indirectness" (no downgrade), "serious indirectness" (downgrade one level), or "very serious indirectness" (downgrade two levels). The decision rule is to make explicit, documented judgments about whether the population/intervention/comparator/outcome differences materially change whether the studied evidence still supports the target claim.

### 4. Imprecision
Uses an **Optimal Information Size (OIS)** threshold logic: if the 95% CI excludes a risk ratio of 1.0 *and* the total number of events/participants exceeds the OIS, precision is judged adequate. Downgrade when: the sample size/event count is below the OIS; the CI includes both appreciable benefit and appreciable harm; or the CI is wide enough to cross clinically important thresholds. Cochrane offers rough (approximate, not rigid) guides — RR thresholds of <0.75 or >1.25 are commonly used to define "appreciable" effects, and roughly 400 events as a traditional minimum rule of thumb. A single study can still be evaluated for imprecision via the OIS criterion. Explicitly flagged as a misapplication: downgrading for imprecision merely because there are "few studies" rather than because the CI/event-count is actually inadequate.

### 5. Publication bias
Downgrade when there is an asymmetric funnel plot suggesting missing (usually negative or null) studies, selective non-reporting of outcomes across the included studies as a set (distinct from within-study selective reporting, which belongs under risk of bias), an unusually large number of small studies not contributing data to an outcome, or a "prototypical" red flag pattern such as multiple small, industry-funded studies all reporting favorable results. Serious evidence of publication bias downgrades one level; strong/serious evidence downgrades two.

## Upgrading domains (non-randomized evidence only)

Low/very-low-certainty NRSI evidence can be upgraded under specific, checkable conditions:

- **Large effects**: RR > 2 or < 0.5 supports a one-level upgrade if no plausible confounders explain the association and estimates are consistent and precise; RR > 5 or < 0.2 supports consideration of a larger upgrade. Worked example: bicycle helmets reducing head-injury risk (OR 0.31, 95% CI 0.26–0.37) was upgraded to moderate certainty despite an observational design.
- **Dose-response gradient**: a clear monotonic relationship between exposure level and outcome (e.g., bleeding risk rising with INR; cardiovascular risk of rofecoxib rising with dose, RR 1.33 at <25 mg/day vs. 2.19 at >25 mg/day) supports a one-level upgrade because it is harder to explain via confounding alone.
- **Opposing plausible confounding**: if all plausible biases would be expected to work *against* the observed effect (e.g., sicker patients receiving the intervention yet still showing better outcomes), the true effect is probably at least as large as observed — supporting an upgrade, potentially even to high certainty if confounding would need to be implausibly large to explain the result away.

## Practical downgrading rules

Each domain typically downgrades certainty by one level, with a floor of "very low" (max three-level fall from "high" for RCTs, one domain can occasionally cause a two-level fall alone if "very serious"). Cochrane's MECIR conduct standards require: use all five GRADE considerations for every reported outcome; explicitly justify every downgrade/upgrade decision in the review's text and Summary of Findings footnotes; and have two independent reviewers assess and reach consensus on certainty.

## Worked example (compression stockings for DVT prevention)

- **Symptomless DVT, low-risk population**: RR 0.10 (95% CI 0.04–0.26), 2637 participants/9 studies → **High certainty** (no downgrades: randomized, consistent, precise).
- **Superficial vein thrombosis**: RR 0.45 (95% CI 0.18–1.13), 1804 participants/8 studies → **Moderate certainty** (downgraded one level for imprecision — CI crosses no-difference and does not rule out a small increase in risk).
- **Oedema**: mean difference −4.7 (95% CI −4.9 to −4.5), 1246 participants/6 studies → **Low certainty** (downgraded two levels: outcome measurement was unvalidated [indirectness] and unblinded [risk of bias]).

This example is the template for claim-evidence matching in practice: the *same body of studies* can support different claims (different outcomes) at very different certainty levels, and the downgrade rationale must be traceable to specific, named domain concerns rather than a global impression.

## Summary of Findings table requirements

Every Cochrane Summary of Findings table must report, per outcome (up to 7, pre-specified as critical/important, included whether or not data are actually available): population/setting; comparison; assumed comparator risk; absolute effect with intervention; relative effect (RR/OR/HR with 95% CI); participant/study counts; the GRADE certainty rating; a comments field; and explanatory footnotes justifying every certainty judgment made. This structure is itself an actionable template for auditing whether a stated claim is properly evidenced: each cell must be traceable to a specific number of studies/participants and a specific, justified certainty rating.

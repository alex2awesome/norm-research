# EXP-EXAMPLES-BOUNDED-1 — prereg: do examples help BOUNDED (tacit-camp) metrics?

Status: PREREGISTERED 2026-08-17, user-approved; runs AFTER EXP-EXAMPLES-BANK-1 (sha
c42a8f54db2e6f54) completes, sharing its machinery. Hypothesis under test (user's): examples
help tacit metrics, which concentrate in the bounded regime. Prior evidence is conflicting:
flips panel shows a bounded-winners tail (4/10 ≥ +.05) but with two confounds this design
must expose — definition-headroom (delta inflates where definitions fail) and
reference shrinkage (tie-exclusion removes the contested core for near-κ≈0 metrics).

## Sample (regime-stratified; fixed before any new selection call)
- ALL fitted BOUNDED metrics (65).
- Matched controls: for each bounded metric, 1 REACHES + 1 RISING metric from the SAME task,
  same 6-category where available (else same task, category-free), seeded random (seed 0),
  without replacement. ≈195 metrics total.
- Metrics already run in EXP-EXAMPLES-BANK-1 REUSE those results verbatim (declared; no re-run).

## Procedure
Identical to EXP-EXAMPLES-BANK-1 (2-voter mbar2 bank reference with ties→−1; selection at
llama8b only, theta=.01, train-A/train-B/holdout stable-hash splits, exemplars masked, null
control every 3rd metric; holdout evaluated at llama70b and qwen25-72b separately).

## Preregistered readouts (all reported per metric AND pooled by regime)
1. delta = functional − definition (per judge), by regime, metric-level bootstrap CIs.
2. ABSOLUTE arm levels (name / definition / functional) by regime — separates "examples teach"
   from "definitions fail" (headroom confound): the hypothesis-relevant pattern is functional
   HIGH where definition is low, not merely delta > 0.
3. Reference-quality covariates: decided-item fraction (1 − tie rate), 2-voter agreement rate,
   and n_flips_from_crowd for the selected sets; regression of delta on regime dummies WITH
   these covariates — the bounded effect must survive them to count.
4. Exploratory (labeled): delta vs the metric's fitted ceiling L.

## Decision rule
"Examples help bounded metrics" is SUPPORTED iff pooled bounded delta > +.05 with metric-level
bootstrap 95% CI > 0 at BOTH evaluation judges, AND the bounded coefficient survives the
covariate regression (CI > 0 with decided-fraction and flip-count included), AND null controls
pass L1. UNSUPPORTED if the pooled bounded CI spans 0 at either judge or the effect vanishes
under covariates. Mixed otherwise, reported descriptively. No optional stopping.

## Not yet run
No calls made as of freeze. Queued behind EXP-EXAMPLES-BANK-1 on sk3.

## Amendment inheritance (2026-08-17, before any run)
This experiment inherits EXP-EXAMPLES-BANK-1 AMENDMENT 1 (sha 49dd908dacce36f9) in full:
primary evaluation key = LOFO family-balanced 11-panel consensus; 2-voter bank key demoted to
sensitivity; per-item arm labelings saved; silver cross-check on the four sound-silver tasks;
selection-vs-evaluation key asymmetry disclosed as conservative. The decision rule's "at BOTH
judges" clauses are evaluated under the PRIMARY key.

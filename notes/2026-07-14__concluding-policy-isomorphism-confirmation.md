# Concluding policy-isomorphism confirmation

## Scientific aim

Test whether explicit, construct-specific articulation lets a smaller same-version model
reconstruct a larger model's own name-invoked policy, without using dataset outcomes, human
labels, community targets, compiler behavior, or any other external ground truth.  The target is
the Llama-3.1-70B name-only score vector and the executor is Llama-3.1-8B.

This batch is a prior-selected **existence confirmation**, not a prevalence study.  Prevalence is
reserved for the performance-unselected sample selected after the breadth calibration report is
available.

## Frozen construct batch

1. `N_humor_23`: Laugh density and economy.
2. `N_humor_11`: Parody/pastiche craft: fidelity, tone, and commentary.
3. `N_press-releases_35`: Specific, quantified, and checkable claims with meaningful context.

The first two constructs share the source-group-disjoint humor item panel, so the three tests are
not treated as three independent domains.  The exact legacy `gi11` construct is humor, not the
similarly named creative-writing construct.  Llama 3.1 has no 3B checkpoint, so a 3B-to-70B rung
is excluded from the same-family confirmatory family and may only be added later as an explicitly
cross-version exploratory analysis.

## Frozen test

Each construct has two content arms:

- `source_definition`
- `source_full_rubric`

Each has an exact-word-count inert control and wrong-construct control.  Together these form one
six-member Bonferroni union family per construct (two content-arm tests plus four content-control
contrasts).  The name-only executor is the adverse-rank baseline.  The functional floor is frozen
at Spearman rho 0.70; effect estimates and intervals remain primary, and the floor is a declared
convention rather than a theoretically derived constant.

Calibration uses the open 400-item `tacit_breadth_search` partition.  The sealed 400-item
`tacit_breadth_validation` partition cannot be read until the exact production-only calibration
report produces a hash-bound release artifact.  Results must be reported construct by construct;
`X of 3` is a selection-biased existence count and is never a prevalence denominator.

## Implementation and verification

The existing H49/breadth implementation is parameterized rather than forked.  The integrated
compiler, scorer, sharder, analyzer, authorization barrier, and teacher-forced readout are reused.
The only new launcher is a thin batch scheduler restricted to one physical GPU among 5, 6, and 7;
physical GPUs 0--4 are forbidden for this batch, and the account-wide four-GPU cap is checked.

Local verification on 2026-07-14:

- 421/421 codability tests passed.
- Frozen scoring, analysis, and compilation implementation hashes validated.
- Full fake calibration traversed scoring, immutable sharding, and three-cell analysis.
- A lockbox attempt without the authenticated calibration release failed before reading or
  writing any lockbox item artifact.

## Authoritative files

- Construct panel: `methods/codability/experiments/concluding_policy_construct_panel_v1.json`
- Target/readout manifest: `methods/codability/experiments/concluding_policy_target_manifest_v1.json`
- Arm bank: `notebooks/data/two_faces_20260702/concluding_policy_arm_bank_v1.json`
- Frozen selection: `methods/codability/experiments/concluding_policy_selection_v1.json`
- Execution manifest: `methods/codability/experiments/concluding_policy_execution_manifest_v1.json`
- Launcher: `methods/codability/experiments/run_concluding_policy_confirmation_sk3.sh`
- Output directory: `notebooks/data/two_faces_20260702/concluding_policy_confirmation_v1/`
- Running breadth job: `notebooks/data/two_faces_20260702/tacit_breadth_confirmation_v3/`

The execution-manifest SHA-256 at final local freeze is
`a3cc3e52129ee9364ec281b698dff60075e834561414929f2cf646c77916910c`.


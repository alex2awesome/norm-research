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

## Execution status

The full calibration-then-lockbox run launched on sk3 physical GPU 5 at
`2026-07-14T02:01:23-07:00`, PID `3178271`.  It runs from the isolated code snapshot
`/lfs/skampere3/0/alexspan/policy_isomorphism_snapshots/concluding_policy_a3cc3e52`, so it cannot
change the code used by the simultaneous breadth job.  The authoritative invocation record and
log are `concluding_policy_confirmation_v1/launch_record.env` and
`concluding_policy_confirmation_v1/logs/full_run.log` under the data directory above.

**v1 outcome (2026-07-14):** calibration completed and certified exactly one arm
(`N_press-releases_35` / `source_definition`), but the lockbox never opened: the release gate
failed on a closure-bookkeeping drift (the manifest declared 15 analysis-implementation files, the
report's runtime self-record listed 12; the 3-file diff is the inert package `__init__.py`s, zero
hash mismatches on real code).  No sealed item was ever read under v1.

## v2 re-execution and SEALED RESULT (2026-07-15)

The closure lists were unified into one canonical `ANALYSIS_IMPLEMENTATION_PATHS` tuple in
`policy_data.py` (commit `451d841`), the execution manifest recompiled as
`concluding_policy_execution_manifest_v2.json` (SHA-256
`a879ccd8bbd6fb68832b649cfe5a012a420a3bc05fc9456d75f3e17814328c4c`; structural diff vs v1 is
exactly the four edited-file hashes plus the v2 output directory), the selection artifact reused
byte-identical, and the full calibration-then-lockbox run re-executed from snapshot
`concluding_policy_a879ccd8` (GPU 5, PID `3567311`, launched `2026-07-15T12:06:26-07:00`,
completed 12:45, no errors).  Both phases scored fresh; nothing was reused from v1.  The batch has
no adaptive step between phases (all arms pre-frozen in the selection artifact), so v1's observed
calibration outcomes could not influence the sealed test.

Sealed 400-item `tacit_breadth_validation` results, simultaneous six-member 99.1667% CIs
(worst-form adverse rho / form-quotient rho), 10,000-draw paired bootstrap, functional floor .70:

| Construct | Arm | Adverse rho [CI] | Quotient rho [CI] | Native 70B−8B gap | Sealed verdict |
|---|---|---|---|---|---|
| `N_press-releases_35` | `source_definition` | **.782 [.717, .829]** | **.796 [.737, .843]** | .573 [.486, .663] | **CERTIFIED functional-ordinal joint substitution** |
| `N_press-releases_35` | `source_full_rubric` | .611 [.518, .690] | .685 [.604, .754] | — | fails floor |
| `N_humor_23` | `source_definition` | .648 [.556, .728] | .681 [.592, .756] | .490 [.407, .575] | fails floor (articulation gain certified) |
| `N_humor_23` | `source_full_rubric` | .630 [.535, .712] | .682 [.593, .756] | — | fails floor (articulation gain certified) |
| `N_humor_11` | `source_definition` | .676 [.584, .743] | .710 [.628, .775] | .390 [.322, .463] | fails floor |
| `N_humor_11` | `source_full_rubric` | .554 [.449, .645] | .624 [.527, .707] | — | fails floor |

The gi35 definition arm passes every certified gate of the joint claim on sealed data:
articulation gain over the 8B name-only baseline, control superiority
(`small_sparse_adverse_rank_below_functional_floor`), fixed-target floor, direct-endpoint floor,
and a simultaneous-certified native scale gap — the same claim grade as H49 (`functional_ordinal`;
near-identity correctly not claimed).  Calibration-to-sealed shrinkage is small
(adverse .801 → .782), i.e., the effect replicates on never-before-read items.  This is the
second sealed certified construct, in a different domain (press-releases) and different construct
family (checkable-claims specificity) than H49 (humor).  Report construct-by-construct; "1 of 3"
is a selection-biased existence count, never a prevalence.

Artifacts: `notebooks/data/two_faces_20260702/concluding_policy_confirmation_v2/`
(`lockbox_report.json`, `calibration_report.json`, `calibration_release.json`,
`launch_record.env`, `logs/full_run.log`).  Release chain verified: the release binds the exact
calibration-report SHA (`8a883e20…`) and manifest SHA (`a879ccd8…`), and the lockbox report's
partition authorization records the same manifest and selection SHAs.

## Cross-version capacity ladder — EXPLORATORY ONLY (2026-07-15)

Llama-3.2-1B and Llama-3.2-3B executors were run against the same frozen Llama-3.1-70B name-only
target, same frozen arms, same open 400-item `tacit_breadth_search` partition (never the sealed
partition).  Llama-3.1 has no 1B/3B checkpoint, so these rungs cross model versions and are
**exploratory observations, never confirmatory**: cross-version name-only baselines are not
monotone in size (the 3.2-3B name baseline tracks the 70B target *better* than 3.1-8B's on gi35,
native gap .401 vs .547, but *worse* on both humor cells), so no same-family scaling claim may be
built on them.  Manifests: `concluding_policy_capacity_{3b,1b}_exploratory_manifest_v1.json`
(status `frozen-before-cross-version-exploratory-capacity-calibration-outcomes`); reports under
`notebooks/data/two_faces_20260702/concluding_policy_capacity_{3b,1b}_exploratory_v1/`.

Definition arm, worst-form adverse rho vs the frozen target (point [95% simultaneous CI]):

| Construct | 1B (3.2) | 3B (3.2) | 8B (3.1, same partition) |
|---|---|---|---|
| gi35 press-releases | .268 [.139, .385] | .721 [.648, .776] | .801 [.745, .844] |
| H23 humor | −.041 [−.174, .085] | .533 [.422, .631] | .671 [.580, .747] |
| gi11 humor | .074 [−.057, .203] | .548 [.443, .635] | .693 [.611, .760] |

Observed pattern: a hard executability floor between 1B and 3B — at 1B every articulation arm is
inert (humor at zero; rubric arm equally dead), while at 3B gi35's definition already functions
near its 8B level (−.08) and the humor constructs recover only about half their 8B fidelity
(−.14/−.15).  The capacity needed to *execute* an explicit articulation of a larger model's
policy is construct-dependent: gi35's threshold sits between 1B and 3B; the humor constructs'
sits at or above 8B.  Native 70B-minus-executor name-only gaps at 1B are extreme (gi35 .756,
gi11 .857, H23 1.072 — the 1B name baseline is negatively correlated with the target on H23).
Ladder ordering is monotone within every construct.

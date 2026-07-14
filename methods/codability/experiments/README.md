# Target-indexed scale--articulation experiments

Additive experiments implementing
`notes/2026-07-12__target-indexed-articulation-frontier-and-duality.md`. Frozen legacy artifacts and
preregistrations remain available; the integrated runner and reports below use explicit, separately
versioned schemas.

This directory is the model-to-model prompt-articulation stream. It does not use compiler outputs
or artifacts from `methods/metric_seam`, which is a separate prompt-versus-code project.

Two evidence states coexist and must not be blurred:

- The canonical retrospective reports are the immutable policy/source/crossfold **v4** artifact
  family generated from saved public shards. Their positive results are observed and retrospective;
  no v4 scale/direct/joint grade is interval-certified.
- The current prospective runner emits policy/source and crossfold **v5**. It adds a
  content-specific joint grade for the frozen same-version Llama-3.1 8B-to-70B confirmation. H49
  completed on 2026-07-13: the source-definition arm is a union-family-simultaneously certified
  content-specific functional substitution on 1,500 sealed items; the full rubric is observed but
  not certified. Neither arm is near-identical or two-sided equivalent. The 990-cell breadth
  confirmation has a CPU-validated GPU-0-only v2 search manifest but no model outcomes yet; it is
  not needed for this bounded result.

Version 5 can read a homogeneous v4 family for audit, but v4 and v5 source reports cannot be mixed
inside one join. Never overwrite a v4 filename with v5 output.

## Integrated experiment core

New reconstruction methods should be configurations or small strategy adapters around these
modules, not new end-to-end loaders and analyzers:

| File | Shared responsibility |
|---|---|
| `policy_data.py` | Declared partition scopes, calibration-release gating, frozen environment/provenance checks, authenticated source groups, shard validation, repetition averaging, orbit construction, and duplicate-safe hash alignment |
| `policy_isomorphism.py` | Fixed-target and direct larger-endpoint identity geometry, paired item/source-group bootstrap certificates, one- and two-sided scale-step loss comparisons, and ordinal/quotient-vector articulation multi-realization sets |
| `run_policy_isomorphism.py` | Configurable direct executor-to-target evaluation over an authorized partition, optional larger-executor comparator, matched controls, v5 union-family content-specific joint grades, ordinal and quotient-vector fibers, calibration release, and crossfold/pooled joins |
| `paired_policy_frontier.py` | Cross-fit challenger/incumbent comparisons on the public folds |
| `compile_adjacent_scale_isomorphism_bank.py` | Shared `compile_scale_pair_bank` configuration path; the upper-scale compiler is a compatibility wrapper |
| `prompt_portfolio_crossfit.py` | Supplementary multi-execution strategy over the same validated policies and certificate; not a single-prompt claim |

Older compile/score/analyze entry points are retained to reproduce already-frozen banks and
artifacts. Their generic data operations now route through `policy_data.py`. New search generations
must reuse that core and add an entry point only when the model readout itself is genuinely new
(for example, pairwise rather than pointwise elicitation).

Portfolio artifact `prompt_portfolio_crossfit_v1.json` is diagnostic-only because controls and
duplicate name policies entered its optimizer. `v2` removes them but is reciprocal public
development. `v3_independent` and `v4_rank` are strict fold transfers over fold-independent text;
neither passes the complete near-identity frontier in both directions. `v5_functional` adds the
retrospective `.70` ordinal tier: two system portfolios pass direct endpoint **fidelity** at the
point level in both directions, and zero pass the lower-confidence-bound tier in both. They are not
v5 endpoint substitutions because the 3B name baseline already exceeds `.70` against that 8B
endpoint.

## Modules

| File | Role |
|---|---|
| `target_articulation_manifest_v1.json` | Target views, articulation channels, treatment boundary, gates, and claim grades |
| `target_articulation_frontier.py` | Fixed-target `R/T`, polarity, adverse-form robustness, held-out substitution, debt/frontier helpers |
| `fixed_target_name_substitution.py` | Experiment N: larger-reader name-only target, development selection, held-out test |
| `common_target_ladder.py` | Reject moving targets/incommensurable costs; validate crossfold ladders and exact same-fold executor response surfaces |
| `gestalt_substitution.py` | Experiment G: combiner interaction, composition gap, span residual, and gestalt substitution |
| `fixed_target_surface.py` | Persist a reader's full held-out articulation surface and paired draws |
| `surface_comparison.py` | Cellwise and finite-bank-simultaneous comparisons from aligned surfaces |
| `name_surface_atlas.py` | Frozen within-family surface atlas driver |
| `cross_family_surface_atlas.py` | Reciprocal target-family/executor-family atlas driver |
| `surface_frontier_report.py` | Rung/channel/domain/scale articulation-gain report |
| `validate_surface_atlas.py` | Cross-file hashes, array alignment, joins, and aggregate integrity certificate |
| `residual_teaching_manifest_v1.json` | Fresh source-telling vs residual-teaching vs fitted-optimizer confirmation protocol |
| `gestalt_execution_manifest_v1.json` | Fresh model-gestalt and social-practice target protocol beyond construct names |
| `fresh_item_partition_manifest_v1.json` | Exact fresh partition sizes, visibility boundaries, and holdout grades |
| `build_fresh_item_partitions.py` | Hash-frozen item/target separation with legacy exclusion and source-group guards |
| `validate_fresh_item_partitions.py` | Sealed-safe hash, leakage, source-split, alignment, and label integrity certificate |
| `fresh_target_view_manifest_v1.json` | Executed Qwen/Gemma name targets; its unexecuted pilot G entries are superseded below |
| `score_fresh_target_views.py` | Integrity-gated, resumable target scoring with per-form soft signatures |
| `fresh_target_score_report.py` | Target informativeness, form reliability, and independent-launch test-retest audit |
| `fresh_gestalt_target_manifest_v1.json` | Clean G-only holistic-question instrument without the generic criterion wrapper |
| `fresh_llama70_name_target_manifest_v1.json` | Clean Llama-70B name-only priority-cell execution job |
| `practice_target_health.py` | Aggregate-only health/provenance report for sealed archival P proxies |
| `shard_fresh_score_artifact.py` | Immutable per-partition target/executor shards for selector/lockbox isolation |
| `fresh_name_arm_selection.py` | Public-shard-only bootstrap selection and frozen lockbox selection artifact |
| `run_fresh_public_queue_sk3.sh` | Guarded target/development/shard/selection queue that stops before lockbox |
| `compile_fresh_name_arm_bank.py` | Source-only definition/rule/example arms plus exact-length wrong/inert controls |
| `fresh_name_execution_manifest_v1.json` | Frozen executor jobs, public/lockbox phases, hashes, and version caveat |
| `same_version_upper_item_partition_manifest_v1.json` | Frozen 400-item calibration/1,500-item sealed same-version sentinel partition contract |
| `same_version_upper_partition_integrity_v1.json` | Sealed-safe joint certificate for all 1,900 item/group identities, predecessor exclusions, and absence of practice targets |
| `same_version_upper_selection_v1.json` | Frozen declarative/procedural two-arm family, matched controls, thresholds, and decision rule |
| `same_version_upper_execution_manifest_v1.json` | Exact Llama-3.1 BF16 checkpoints, teacher-forced readout, phase gates, and GPU resource ceiling |
| `score_fresh_name_arms.py` | Integrity-gated target/executor scoring; opens only the named phase files, enforces the frozen selection, and supports teacher-forced declared-label probabilities |
| `tacit_breadth_item_partition_manifest_v2.json` | Frozen 11-task, 8,200-item open-search/sealed-validation packet contract with label-projection grades |
| `run_tacit_breadth_search_sk3.sh` | Hash-pinned, account-cap-checking launcher for the open breadth target/executor, sharding, and analysis steps; contains no validation command |
| `concluding_policy_construct_panel_v1.json` | Three exact prior-selected legacy constructs for the existence-only concluding batch; two humor constructs share one panel |
| `concluding_policy_selection_v1.json` | Frozen two-content/four-control family per construct; explicitly excludes the batch from prevalence claims |
| `concluding_policy_execution_manifest_v1.json` | Same-version Llama-3.1 8B-to-70B hashes, calibration-release barrier, per-construct correction, and GPU 5/6/7 resource policy |
| `run_concluding_policy_confirmation_sk3.sh` | Thin batched configuration over the shared scorer/sharder/analyzer; runs calibration and then the authenticated lockbox without adding a new analysis stack |

## Non-negotiable distinctions

- A name target, model-internal gestalt target, social-practice target, and operational rubric target
  are different random variables.
- Unsigned mutual information is not verdict isomorphism: policy inversion has the same MI. The
  primary frontier therefore orients `R/T` by covariance and retains a direct signature gate.
- Candidate-form robustness uses the adverse candidate form against a target form quotient.
- Failure to reach the target is right-censored debt, never a large finite cost.
- Legacy word counts are description lengths, not certified articulation units.
- Fixed-target functional ordinal reconstruction requires both adverse-form and form-quotient rho
  `>= .70`. A fixed-target reconstruction, even with a smaller-name gap, is not by itself a scale
  substitution: the paired larger sparse endpoint must also be supplied and tested directly.
  Point-level and bootstrap-certified results are never conflated.
- Crossfold joins report threshold-free rank capacity and a fixed sensitivity profile. Optional
  pooling uses a fold-stratified bootstrap and reports both nominal per-arm and Bonferroni arm-bank
  intervals; pooled public items improve precision but do not become a new holdout.
- PR/CW and any other grouped source use an authenticated source-group bootstrap that retains every
  member item when a group is drawn; IID item intervals are not valid for those panels. Humor's
  singleton groups reduce to the ordinary paired item bootstrap.
- Retrospective fiber summaries exclude controls, distinguish component-minimal articulations from
  passing supersets, audit whether added components improve or interfere, and report
  `.70/.80/.85/.90` point-only mutual-rank sensitivity. The prospective v5 sentinel instead has
  two preselected atomic route arms because its v1 bank declares no decomposed inventories; it
  requires distinct declarative/procedural channels, surface distance, a multiplicity-adjusted
  mutual-rank interval, and at least 99% valid rank-bootstrap draws. This is convergent ordinal
  reconstruction, not numerical or per-form policy equality.
- A paired scale-step certificate uses one shared bootstrap for the fixed target, small name,
  small articulation, and larger name. It separately reports one-sided target-relative
  noninferiority, two-sided equal target loss, direct candidate-to-larger endpoint fidelity at
  `.70`, direct near-identity, fixed-target fidelity, and their joint grades. Direct functional
  endpoint substitution requires adverse-form and form-quotient candidate-to-larger rho `>= .70`,
  the predeclared adverse-rank small-name baseline below `.70`, and direct MAE improvement. The
  joint functional grade additionally requires fixed-target functional reconstruction; the joint
  equivalent grade also requires two-sided target-relative rank/MAE loss equivalence. The
  target-self-band near-identity grade is stricter still. Neither one-sided noninferiority nor equal
  target loss alone is item-policy isomorphism.
- In frozen v4 reports, matched-control superiority is a separate certificate and does not
  retroactively enter a joint scale grade. The prospective v5 primary `H_J` grade intersects the
  joint fixed-target/direct-endpoint functional substitution with superiority on **both rank and
  MAE** to every required inert-length and wrong-construct control. Its simultaneous tier uses one
  union-family Bonferroni denominator over all eligible scale candidates and all matched
  source-control pairs in the cell. Missing either control provenance makes `H_J` ineligible.
- `H_J^eq` is the secondary anti-overshoot refinement: it adds two-sided robust fixed-target rank
  and MAE equal-loss intervals to `H_J`. Near identity is a separate, still stricter secondary
  target-self-band grade. Neither may be substituted for a failed primary `H_J` result.
- The prospective same-version readout teacher-forces each declared first-token continuation
  (`YES`, `NO`) and normalizes their log likelihoods. It does not depend on a top-k list containing
  both labels and does not use constrained generation, avoiding the earlier missing-label and
  artificial 0/1 failure modes. Runtime continuation IDs must equal the frozen tokenizer IDs, and
  a missing actual-token log probability is fatal rather than imputed from another token.
- The shard validator authenticates job IDs, item/prompt/readout alignment, and score hashes; it
  does not infer family membership or parameter ordering from a checkpoint registry. Current
  "same-family scale step" language is therefore additionally bound to the frozen execution
  manifests/model sidecars. The same-version sentinel binds every job ID to exact checkpoint,
  revision, family/version, parameter count, precision, tokenizer, and role in its execution
  manifest; broader future ladders should do the same in one hashed ladder manifest.
- Both a matched inert arm and a matched wrong-construct arm are required for a prospective v5
  articulation-specific joint result.
- The combiner, composition, and span quantities locate gestalt inside the DPI bracket; none is an
  all-prompt ceiling.

## Prospective same-version sentinel (v5; completed)

The next confirmatory experiment is a one-construct same-version test, not another retrospective
re-read. It keeps the objective unsupervised: the Llama-3.1-70B name-only three-form policy is both
the fixed top target `Q` and the larger sparse endpoint `E`; no human label, compiler output,
practice outcome, or third-model judgment defines success. Fixed-target and direct-endpoint
fidelity are logically redundant in this two-rung sentinel, but both are emitted so the result uses
the same vocabulary as later multi-rung panels.

| Frozen element | Declaration |
|---|---|
| Construct | `N_humor_49`, **Wordplay quality and clarity** |
| Smaller executor | `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`, BF16 |
| Larger target/endpoint | `meta-llama/Llama-3.1-70B-Instruct`, revision `1605565b47bb9346c5515c34102e054115b4f98b`, BF16 |
| Candidate arms | `source_definition` (declarative) and `source_full_rubric` (procedural) |
| Matched controls | One exact-count inert and one exact-count wrong-construct arm for each candidate |
| Items | 400 calibration items plus 1,500 sealed final items |
| Readout | Three prompt forms, two engine repetitions, teacher-forced declared-label `P(YES)/(P(YES)+P(NO))` |
| Primary endpoint | Union-family-simultaneous content-specific `H_J` |
| Secondary endpoints | `H_J^eq`, near identity, the declarative/procedural mutual-rank fiber, and its strict quotient-vector `H_fiber^vec` subset |
| Execution status | Completed 2026-07-13; authenticated report at `notebooks/data/two_faces_20260702/same_version_upper_confirmation_v1/lockbox_report.json` |

The two candidates were fixed from the prior best public declarative and procedural routes before
any Llama-3.1-70B outcome. They are atomic arms in the v1 bank: this confirmation does not claim
independent component inventories, component minimality, or component-set incomparability. Each
candidate is paired with both control species. The confirmatory union
family therefore contains six inferential members: two candidate scale certificates and four
matched source-control contrasts. The name arm is still required as the sparse baseline, but it is
not an additional searched articulation. No dossier, example stack, outcome-conditioned rewrite,
or post-freeze recombination may enter this experiment.

For an arm `a`, primary `H_J(a)` requires all of the following on the untouched final panel at the
confidence-bound grade:

1. a genuine native 8B-to-70B sparse rank/MAE gap;
2. an 8B articulation rank/MAE gain over 8B name-only;
3. adverse-form and form-quotient Spearman at least `.70` to both `Q` and `E`, direct MAE gain, and
   the predeclared adverse-rank baseline exclusion;
4. superiority on both rank and MAE to every matched inert and wrong-construct control under the
   one six-member union-family correction.

`H_J^eq` additionally requires the complete paired intervals for the robust fixed-target rank and
MAE loss differences between articulated 8B and sparse 70B to lie inside the frozen two-sided
`.05/.02` margins. Near identity instead asks for the full target-self MAE/rank/flip/bias band and
remains secondary. If both declarative and procedural arms pass `H_J`, their mutual held-out
Spearman lower bound must reach `.90` for the primary equal-but-different fiber (`.85` is the frozen
sensitivity analysis), and at least 99% of paired bootstrap rank draws must be finite. Both
inferential families use central intervals. Ordinal `H_fiber` consumes the rank lower edge;
`H_fiber^vec` is an intersection-union test over that edge plus the MAE/flip/bias upper edges, so
the coordinates need no within-pair alpha split. Pair multiplicity and the separate content/control
family together retain a `.05` composite FWER bound. The ordinal fiber conclusion is
highly concordant item ordering, not equality of probabilities, threshold decisions, calibration,
or individual prompt-form behavior.
The nested `H_fiber^vec` secondary additionally requires the mutual quotient-policy MAE,
binary-flip, and absolute-bias upper bounds all to be at most `.02`. It is the direct test of a
stronger equal-but-different small-model policy, while remaining agnostic about matched-form and
semantic equality.

Version 5 does not upgrade any retrospective claim. A read-only v5 re-evaluation has zero
content-specific joint members in existing data for a design reason: the positive upper-pair v4
bank has no scored matched controls, while the controlled 3B cells have no joint direct-endpoint
substitution. `H_J` is therefore a genuinely prospective endpoint.

The packet contains no practice-target files and excludes all 2,200 predecessor-packet
item/source identities plus the 300 reconstructed legacy probes. Because humor source identity is
only recoverable by content hash, the declared holdout is deduplicated item-disjoint, not
author-disjoint. Calibration opens only the named 400-item file; it cannot open the lockbox and
filter it afterward. Calibration may test execution integrity and power only—no new articulation
content may be authored from its outcomes. The 1,500-item panel remains inaccessible until the
exact frozen runner emits an authenticated production-only calibration report and release artifact;
fake scores cannot release it. Arms, controls, checkpoints, readout, thresholds, and multiplicity
must remain unchanged.

Frozen provenance:

- item protocol: `same_version_upper_item_partition_manifest_v1.json`, SHA-256
  `2e2dedf1c89691cda880da52826fa67f91310eda2568f2c3cd538a1159323955`;
- packet manifest:
  `notebooks/data/two_faces_20260702/same_version_upper_item_partitions_v1/packet_manifest.json`,
  SHA-256 `54a96684baeb0e873860a960468fc53cda456303f8a1b5e27271e8b05d2bba36`;
- two-arm selection: `same_version_upper_selection_v1.json`, SHA-256
  `cbbb4ed06079bb768436d10856895bd7419a0a049bd50520f65080402a673765`;
- joint partition-integrity certificate: `same_version_upper_partition_integrity_v1.json`, SHA-256
  `e9f161e62bed87d3b742611046a149ef2740f6b5003b7d36d48019dc442cd509`;
- arm bank: `notebooks/data/two_faces_20260702/fresh_name_arm_bank_v1.json`, SHA-256
  `2358800875b276317e41d64dbd7cf02886d80e41713850fcacba17bc4c29961d`;
- exact checkpoint, tokenizer, readout, phase, and resource bindings:
  `same_version_upper_execution_manifest_v1.json`, SHA-256
  `f69c906cb8b7f013fdd27cadbfda608cd7012bba2cf73426eb2cfb10b93f6460`.

The execution manifest declares sequential TP=1 processes, at most one GPU for any one job and at
most four in total. It also freezes `spawn` and PCI-bus CUDA ordering after the unrelated breadth
TP=2 communicator failure. H49 was subsequently executed and completed under this contract; its
authenticated lockbox report is linked in the table above.
The same-version BF16 pair removes the retrospective Llama-3.1-to-3.3 and FP8-target confounds, but
it remains one prior-selected construct in one family; it cannot establish construct prevalence or
a universal scale law by itself.

### Breadth replication queue

The same runner absorbs breadth as configuration, not as another pipeline. The current order is:

1. finish the frozen 990-cell same-version 8B→70B open search and canonical arm selection;
2. score the independently sealed validation fold and estimate R1/R2/R3 substitution prevalence,
   vector residuals, address-dose response, and equal-but-different articulation fibers;
3. retain H49 as a focused one-construct sentinel rather than using it as the breadth gate; and
4. use Qwen2.5 1.5B/3B/7B later as a secondary same-version falsification family.

The legacy 267-metric, nine-domain grids remain discovery-only because they lack the new hashes and
controls; contaminated CW→70B joins and deprecated atlas outputs do not enter confirmation. Every
new endpoint remains unsupervised model-to-model reconstruction, never a human, compiler, or
community-practice target.

### Prospective 990-cell breadth confirmation

Scope boundary: the live experiment writes only below
`notebooks/data/two_faces_20260702/tacit_breadth_confirmation_v3/` and reads its frozen packet from
`tacit_breadth_item_partitions_v2/` plus the sibling `tacit_breadth_metric_panel_v3.json` and
`tacit_breadth_arm_bank_v3.json`.  Its transitive code closure is enumerated and hashed under the
execution manifest's `implementation` block; although those files live in this shared codability
package, `methods/metric_seam/` and compiler-execution comparisons are outside the experiment.
Historical `fresh_*`, `residual_*`, and earlier `policy_isomorphism_*` data directories are lineage
artifacts, not inputs to this confirmation.

The source-frozen panel contains exactly 30 metrics for each of 11 tasks at each of R1, R2, and
R3: 990 cells total. Cross-round raw-rubric dependence is recorded explicitly; the panel has 812
task-global provenance components, and task/construct names are unique. The arm bank has 28,314
arms spanning the name baseline, declarative definitions, procedural rules, ostensive leaf
signals, compositions, deterministic address-prefix doses, full text, and word-count-matched inert
and wrong-construct controls.

Frozen open-search anchors:

| Artifact | SHA-256 |
|---|---|
| `tacit_breadth_metric_panel_v3.json` | `ea34fddad96558ad5261455b394b6aba7378b737b3f1d326a2f47d1abcdce479` |
| `tacit_breadth_arm_bank_v3.json` | `e61999c68eb04d582893ec2bc2a19ee02a8ced79bb80615300707aee89dd1d32` |
| `tacit_breadth_item_partitions_v2/packet_manifest.json` | `2bdadf79072155587f5c1a03eb30cee14ea76b78d8b4c6260f51037d284225ea` |
| `tacit_breadth_item_partitions_v2/partition_integrity.json` | `0e0b5359511ec7bede656d58f2bb9877f0b50e0d69db5fb1c50c550ecb003d3a` |
| `tacit_breadth_confirmation_v3/search_execution_manifest_v2.json` | `e7290eddd725257ae8dec6b2f1ed82160e3d2129f10341f74f1a9f132315ad30` |

`search_execution_manifest.json` (SHA-256 `9e01e75b...`) is retained only as the failed attempt-1
record: it requested TP=2 over physical GPUs 0 and 2 and stopped at communicator initialization
before weights or outcomes. The v2 manifest supersedes it. V2 forces TP=1, one process at a time,
and physical GPU 0 only; GPUs 1--4 are explicitly forbidden with no fallback. Its 26-file runtime
closure is copied byte-for-byte under
`tacit_breadth_confirmation_v3/frozen_implementation_v2/`, so execution need not overwrite the
shared working code used by other campaigns.

Local and sk3 regeneration is byte-identical for both panel and bank. Raw-source validation matched
8,200/8,200 packet items. Item files contain only identity, source-group/split, text, and text hash;
all 22 target paths are null. Ten tabular sources exclude outcome columns at parser projection;
legal JSONL necessarily decodes a row and immediately retains only projected keys. No outcome is
retained, emitted, selected on, or used.

The v2 manifest was compiled off the data host. At freeze time it recomputed all packet structure,
item files, item hashes, partitions, groups, and protocol bindings locally, then authenticated the
already-frozen raw-source membership certificate (11/11 domains, 8,200/8,200 matched rows) because
the code-review and creative-writing source files are server-only. This mode is explicit in
`source_validation_at_freeze`; it does not claim a second raw-source scan on the laptop.

The real 8B/B200 scheduling smoke rejected eight-row teacher-forced batching before outcomes:
71/72 probabilities were bit-identical to scalar, the maximum deterministic batch-shape delta was
`3.06e-5`, and there were no decision flips. Explicit row-batch one was 72/72 bit-identical, so it
is the frozen production schedule. The manifest records the full smoke hashes and diagnostics.
The search target is Llama-3.1-70B BF16 name-only; the executor is exact Llama-3.1-8B BF16. The
functional endpoint remains adverse-form and form-quotient Spearman `>= .70` plus direct MAE gain
and matched-control specificity. The validation partition remains scorer-inaccessible until the
search report, canonical selection, validation manifest, and production-only release all validate.
The current execution status is ready-but-not-launched: the frozen-root fake scorer smoke passed,
the full codability CPU suite is the launch gate, and no breadth v2 GPU process or model outcome
exists yet.

## Retrospective Experiment N

The first method-validation pass uses the legacy humor and math raw grids. Example:

```bash
python -m methods.codability.experiments.fixed_target_name_substitution \
  --domains humor,math \
  --small-tag Llama-3.2-3B-Instruct \
  --big-tag Llama-3.1-8B-Instruct \
  --target-tag Llama-3.1-8B-Instruct \
  --n-boot 500 \
  --seed 1207 \
  --out notebooks/data/two_faces_20260702/fixed_target_name_substitution_3b_8b_v1.json
```

The common-target ladder must keep `--target-tag` fixed while changing the two comparison readers.
Validate it with:

```bash
python -m methods.codability.experiments.common_target_ladder \
  --artifacts <1B-to-3B-target8B.json>,<3B-to-8B-target8B.json>,<1B-to-8B-target8B.json> \
  --labels 1B_to_3B,3B_to_8B,1B_to_8B \
  --out <ladder.json>
```

For direct policy-isomorphism reports, the same module validates target-shard, item, prompt-hash,
and readout identity before comparing executor scales:

```bash
python -m methods.codability.experiments.common_target_ladder \
  --policy-crossfold-artifacts <fixed70-1B-crossfold.json>,<fixed70-8B-crossfold.json> \
  --labels Llama-3.2-1B,Llama-3.1-8B \
  --out <fixed70-executor-ladder.json>
```

When one exact executor rung exists on only one fold, the same module builds a strictly point-level
response surface without pretending it is crossfold confirmation:

```bash
python -m methods.codability.experiments.common_target_ladder \
  --policy-fold-artifacts <fixed70-1B-fold.json>,<fixed70-3B-fold.json>,<fixed70-8B-fold.json> \
  --labels 1B,3B,8B \
  --out <fixed70-executor-response-surface.json>
```

That report separates descriptive target-relative adverse-envelope loss comparisons against the
larger sparse executor, matched-form sensitivity, four-coordinate sensitivity, `.70` fixed-target
reconstruction, and genuine floor rescue. It does not compute direct candidate-to-larger policy
fidelity or paired two-sided target-loss equivalence. Its closure ratios and
executor-by-articulation differences remain descriptive until the paired scale-step certificates
below are run.

Supply the larger sparse executor directly to run those paired certificates inside the same
source report (controls remain in their own contrasts and cannot enter the candidate family):

```bash
python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root <small-shards> \
  --target-shard-root <fixed-top-target-shards> \
  --scale-comparator-shard-root <larger-executor-shards> \
  --scale-comparator-job <larger-job> \
  --scale-comparator-arm-id name \
  --arm-bank <shared-bank.json> --partition <authorized-public-fold> \
  --small-job <small-job> --big-job <fixed-target-job> --target-arm-id target \
  --out <policy-report.json>
```

When the fixed target is itself the larger sparse endpoint, use
`--scale-comparator-use-target`. The frozen canonical source reports use schema
`policy_isomorphism_experiment/v4`; their paired scale-step object is v2, crossfold joins are v4,
pooled fixed-target precision reports are v3, and ladder/response reports are v4. Earlier reports
predating direct endpoint fidelity and anti-overshoot equivalence must not be mixed into that
family. The prospective runner now emits `policy_isomorphism_experiment/v5` and
`crossfold_policy_isomorphism_fibers/v5`, adding the content-specific union-family grades above.
It retains read compatibility with a homogeneous v4 family for audit, not permission to combine v4
and v5 folds.

### Canonical CPU reanalysis runbook (2026-07-12)

The following is the exact canonical-v4 CPU invocation record over authenticated saved score
shards. It does not execute a model or use a GPU. Byte-identical v4 regeneration requires the
frozen implementation hashes below: runner
`d9972ba23e241f5862b66dd06596cf2adbc78f2839666ab45e50363d3e545484`, policy geometry
`79e0b05b2476db8ab8a9de8ae64f301b4262f49f7a9b0ba0212a8313425f12fa`, policy data
`5acbf7bf3ec1bdeff0f7fb81b34011bf3c0499a4b8cbe18059a049b5990a0c52`, and common-target ladder
`05c6bf8bd035ccb8c65fa8e9f8ace3efc75d5a51c3560fadd516b28036766565`. The current v5 runner must
not be pointed at these v4 output filenames; use a new v5 family instead.

The exact 3B compatible bank exists only on the prompt-selection fold, so neither the 3B exact
response surface nor the direct 3B-to-8B re-read is crossfold. A filename counter before
`_policy_v4` denotes the bank/run revision; for example, `*_v3_policy_v4.json` is a v4 report whose
first counter records the third run revision.

```zsh
D=notebooks/data/two_faces_20260702
BANK=$D/upper_scale_isomorphism_bank_v1.json
MBANK=$D/fresh_name_arm_bank_v1.json
PACKET=$D/fresh_item_partitions_v1_local
PM=$PACKET/packet_manifest.json
S70=$D/fresh_name_target_score_shards_v1
S8=$D/upper_scale_isomorphism_score_shards_v1
S1=$D/adjacent_scale_isomorphism_score_shards_v1
S3=$D/policy_isomorphism_score_shards_v1
SM=$D/fresh_name_arm_score_shards_v1
COMMON=(--packet-root "$PACKET" --packet-manifest "$PM" --seed 1207 --mae-margin .02 \
  --rho-margin .05 --flip-margin .02 --bias-margin .02 --functional-rho-floor .70 \
  --confidence .95)

UP=$D/upper_scale_isomorphism_prompt_v7_policy_v4.json
UU=$D/upper_scale_isomorphism_unit_v7_policy_v4.json
UCF=$D/upper_scale_functional_fiber_crossfold_v7_policy_v4_pooled.json
P1=$D/fixed70_llama1_prompt_v6_policy_v4.json
U1=$D/fixed70_llama1_unit_v6_policy_v4.json
CF1=$D/fixed70_llama1_crossfold_v6_policy_v4_pooled.json
P3=$D/fixed70_llama3_exact_upper_prompt_v3_policy_v4.json
D38=$D/direct_llama3_to_llama8_exact_prompt_v3_policy_v4.json
MP=$D/fixed70_llama3_multicell_controls_prompt_v5_policy_v4.json
MU=$D/fixed70_llama3_multicell_controls_unit_v5_policy_v4.json
MCF=$D/fixed70_llama3_multicell_controls_crossfold_v7_policy_v4_pooled.json
LAD=$D/fixed70_executor_ladder_v5_policy_v4.json
SURF=$D/fixed70_exact_executor_response_surface_v4_policy_v4.json

python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$S8" --target-shard-root "$S70" --arm-bank "$BANK" \
  --partition residual_prompt_selection --small-job llama8_upper --big-job llama70_n_target \
  --target-arm-id target --scale-comparator-use-target --n-boot 5000 "${COMMON[@]}" --out "$UP"
python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$S8" --target-shard-root "$S70" --arm-bank "$BANK" \
  --partition residual_unit_certification --small-job llama8_upper --big-job llama70_n_target \
  --target-arm-id target --scale-comparator-use-target --n-boot 5000 "${COMMON[@]}" --out "$UU"
python -m methods.codability.experiments.run_policy_isomorphism \
  --join-report "$UP" --join-report "$UU" --pool-fold-items --pooled-n-boot 10000 \
  --pooled-seed 1217 --packet-root "$PACKET" --packet-manifest "$PM" \
  --functional-rho-floor .70 --confidence .95 --out "$UCF"

python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$S1" --target-shard-root "$S70" \
  --scale-comparator-shard-root "$S8" --scale-comparator-job llama8_upper \
  --scale-comparator-arm-id name --arm-bank "$BANK" --partition residual_prompt_selection \
  --small-job llama1_adjacent --big-job llama70_n_target --target-arm-id target \
  --n-boot 5000 "${COMMON[@]}" --out "$P1"
python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$S1" --target-shard-root "$S70" \
  --scale-comparator-shard-root "$S8" --scale-comparator-job llama8_upper \
  --scale-comparator-arm-id name --arm-bank "$BANK" --partition residual_unit_certification \
  --small-job llama1_adjacent --big-job llama70_n_target --target-arm-id target \
  --n-boot 5000 "${COMMON[@]}" --out "$U1"
python -m methods.codability.experiments.run_policy_isomorphism \
  --join-report "$P1" --join-report "$U1" --pool-fold-items --pooled-n-boot 10000 \
  --pooled-seed 1217 --packet-root "$PACKET" --packet-manifest "$PM" \
  --functional-rho-floor .70 --confidence .95 --out "$CF1"

python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$S3" --target-shard-root "$S70" \
  --scale-comparator-shard-root "$S8" --scale-comparator-job llama8_upper \
  --scale-comparator-arm-id name --arm-bank "$BANK" --partition residual_prompt_selection \
  --small-job llama3_isomorphism --big-job llama70_n_target --target-arm-id target \
  --n-boot 10000 "${COMMON[@]}" --out "$P3"
python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$S3" --target-shard-root "$S8" --arm-bank "$BANK" \
  --partition residual_prompt_selection --small-job llama3_isomorphism --big-job llama8_upper \
  --target-arm-id name --scale-comparator-use-target --n-boot 10000 "${COMMON[@]}" --out "$D38"

MCELLS=(--cell-id N_humor_23 --cell-id N_humor_49 --cell-id N_pr_8)
python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$SM" --target-shard-root "$S70" \
  --scale-comparator-shard-root "$SM" --scale-comparator-job llama8_big_sparse \
  --scale-comparator-arm-id name --arm-bank "$MBANK" --partition residual_prompt_selection \
  --small-job llama3_small --big-job llama70_n_target --target-arm-id target --include-controls \
  "${MCELLS[@]}" --n-boot 5000 "${COMMON[@]}" --out "$MP"
python -m methods.codability.experiments.run_policy_isomorphism \
  --executor-shard-root "$SM" --target-shard-root "$S70" \
  --scale-comparator-shard-root "$SM" --scale-comparator-job llama8_big_sparse \
  --scale-comparator-arm-id name --arm-bank "$MBANK" --partition residual_unit_certification \
  --small-job llama3_small --big-job llama70_n_target --target-arm-id target --include-controls \
  "${MCELLS[@]}" --n-boot 5000 "${COMMON[@]}" --out "$MU"
python -m methods.codability.experiments.run_policy_isomorphism \
  --join-report "$MP" --join-report "$MU" --pool-fold-items --pooled-n-boot 10000 \
  --pooled-seed 1217 --packet-root "$PACKET" --packet-manifest "$PM" \
  --functional-rho-floor .70 --confidence .95 --out "$MCF"

python -m methods.codability.experiments.common_target_ladder \
  --policy-crossfold-artifacts "$CF1,$UCF" --labels "Llama-3.2-1B,Llama-3.1-8B" --out "$LAD"
python -m methods.codability.experiments.common_target_ladder \
  --policy-fold-artifacts "$P1,$P3,$UP" --labels "1B,3B,8B" --out "$SURF"
```

Canonical v4 output SHA-256 values (paths relative to
`notebooks/data/two_faces_20260702/`):

| Artifact | SHA-256 |
|---|---|
| `upper_scale_isomorphism_prompt_v7_policy_v4.json` | `543f03914e6af91a14bd9892ab673ef326206ddf0a6c39a4289a83f43d9da83e` |
| `upper_scale_isomorphism_unit_v7_policy_v4.json` | `91d9bbda7a9879df0e1fb577a0cdc8d8b5536bcf5e73e8ecf4bd2e3112fc7d01` |
| `upper_scale_functional_fiber_crossfold_v7_policy_v4_pooled.json` | `cc4fbaaa18ac7a32ebcc629bac69b757d35d735912e5db0d98a45c2c582983ec` |
| `fixed70_llama1_prompt_v6_policy_v4.json` | `777a966e73b67338969214706a8e1b604a9faf6534b2b5d3d6a4ddc2ce2baf27` |
| `fixed70_llama1_unit_v6_policy_v4.json` | `c82c2bf0642b4b1a1a4ec99d951d703dbc4daf32fb059a384f47c8e326b57f2b` |
| `fixed70_llama1_crossfold_v6_policy_v4_pooled.json` | `415491cb807b7082c23b740414d1acaaf4103da9071e403b8b49f906834caf60` |
| `fixed70_llama3_exact_upper_prompt_v3_policy_v4.json` | `82877b6d93c93823256e0353cb5478ff27aba6fd0698b6b2983680df5aa0242d` |
| `direct_llama3_to_llama8_exact_prompt_v3_policy_v4.json` | `e2b879854e4926d1416adc88170a9b7001d0838825947421a17ca10f2a0c270d` |
| `fixed70_llama3_multicell_controls_prompt_v5_policy_v4.json` | `ee8f21393d22645ace60e9bc9161e9276b3240a1ba8634b9b782d3aecd521273` |
| `fixed70_llama3_multicell_controls_unit_v5_policy_v4.json` | `2524f1877c5c3c1903f3ac8e14450c374ba3d2fd62e290096776bc61aa3e5d49` |
| `fixed70_llama3_multicell_controls_crossfold_v7_policy_v4_pooled.json` | `617cfcec30796ce4147784762caec94e5c198e54e78b21cf2246b52bf3084b84` |
| `fixed70_executor_ladder_v5_policy_v4.json` | `012cc1546ddc82adf2d3d91c3f277f6de8c7071732bf99953c7478d54af2d722` |
| `fixed70_exact_executor_response_surface_v4_policy_v4.json` | `2b64ff610da51910713c17d23bffb675ac9f2a2f03eb8ef57f4847c34ac63360` |

These hashes freeze the retrospective v4 family. Prospective v5 outputs require new filenames and
new hashes even when they read the same saved score shards.

The bank/packet anchors are SHA-256
`290569fb1d9627180ead0eea9efdf5a2990344fbf497a1f45b86467a76787954`,
`2358800875b276317e41d64dbd7cf02886d80e41713850fcacba17bc4c29961d`, and
`827eb0962b41826761ef56cd3e43d5ebba0d9b4bbd816a193c8175bd173f90ab`
for the upper bank, multicell bank, and packet manifest respectively. Output hashes and the
scientific interpretation are recorded in
`notes/2026-07-12__isomorphism-first-tacit-policy-reconstruction.md`.

The earlier `fixed_target_name_substitution` legacy grids—not the canonical v4 policy reports
above—lack probe hashes, matched controls, certified units, and a preregistered version of this
target view. Their claim grade is diagnostic even though candidate selection and evaluation are
split. Results and exact hashes are in
`notes/2026-07-12__fixed-target-name-substitution-first-pass.md`.

The scaled all-task atlas, reciprocal family results, multiplicity audit, and frontier interpretation
are in `notes/2026-07-12__all-task-fixed-target-name-surface-atlas.md`.
The fresh N/G/P partition freeze, target-health results, controls, and public-only execution queue are
tracked in `notes/2026-07-12__fresh-name-gestalt-practice-confirmation.md`.

Validate both persisted atlases and write a joint integrity certificate with:

```bash
python -m methods.codability.experiments.validate_surface_atlas \
  notebooks/data/two_faces_20260702/target_surface_atlas_v1 \
  notebooks/data/two_faces_20260702/target_surface_cross_family_v1 \
  --out notebooks/data/two_faces_20260702/target_surface_atlas_integrity_v1.json
```

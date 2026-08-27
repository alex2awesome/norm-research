# mathlib — formal-library review norms (Lean / mathlib4)

Status 2026-06-10: **design revisit** after the 2026-06-02 audit dropped the
naive merge-vs-not framing. Data collection stalled at 5,000 PR stubs; the
label is being redefined before resuming the fetch.

## 1. Why mathlib at all

mathlib4 is the largest formal mathematics library in the world. Every merged
contribution **type-checks by construction** — the Lean kernel is the hardest
verifiability floor in any of our tasks. Therefore everything reviewers still
argue about (and they argue a lot) is, by construction, in the
articulable-or-taste band. That makes mathlib the project's cleanest
instrument for separating A from T *above* a saturated V floor:

- **V** = does it compile, does it pass mathlib's own linters and style rules
  (the community has unusually thin rules: naming grammar, layout, banned
  idioms — see §5).
- **A** = the review discourse: API duplication, generality level, naming
  semantics, proof idiom. Review threads are written-down norms — the math
  analog of codereview.SE reviewer prose.
- **T** = whatever predicts review friction beyond both ("is this lemma worth
  having", aesthetic golfing preferences).

## 2. What the 2026-06-02 audit found (and why merge-vs-not is dead)

From `project_code_dataset_pivot_2026_06_02.md`, confirmed against the sk3
data (`datasets/math/mathlib/prs_list.jsonl`, 5,000 closed PRs):

1. **Bors mechanics**: mathlib merges via the Bors bot. `merged_at` is null
   for Bors-merged PRs; the merge fact lives in the `[Merged by Bors] - `
   title prefix and the `ready-to-merge` label. Raw title → label LR AUC
   0.978; 0.794 even after stripping.
2. **Reject class is mostly abandonment**: 56%+ of closed-unmerged PRs carry
   merge-conflict / awaiting-author / WIP / blocked-by-other-PR states.
   Truly content-rejected PRs are <10% of rejects.
3. **Author-status leak**: `author_association = NONE` → ~0% merge.
4. Same broader pattern as GitHub code PRs: merge-vs-not conflates quality +
   author identity + process state + reviewer availability + time.

The 2026-06-05 GitHub-PR corrections (strip-don't-drop, within-repo balance,
CI-as-feature; `github-pr-within-repo-balance-2026-06-05`) fix #1 and #3 but
**not #2** — there is no within-repo balancing trick that turns an
abandonment-dominated negative class into a quality signal.

## 3. Reframed labels (proposed 2026-06-10)

Keep the spirit of "y = did the community take it" but move the contrast to
where the community actually expresses judgment:

### 3a. Primary: review friction among *merged* PRs

Direct analog of patents Task A ("first-draft acceptance"):

- **y = 1**: merged with no substantive changes requested — e.g. carries the
  `easy` label, zero CHANGES_REQUESTED reviews, short author-revision count.
- **y = 0**: merged only after substantive review rounds (≥1
  CHANGES_REQUESTED review, or ≥N author force-pushes/revision commits after
  first review).

Both classes compile, both got merged, both authors stuck around → kills the
abandonment confound, the Bors leak (label no longer derived from merge
markers — strip the `[Merged by Bors]` prefix from all titles anyway), and
most of the author-status leak (can additionally balance within
`author_association` strata). What remains is: *given that this was good
enough to land, did reviewers have to articulate objections first?*

Confounds to control, mirrored from the other tasks: PR size (lines/decls
changed — bin-match), subject area (`t-algebra`, `t-topology`, … labels —
balance within), author experience (stratify), calendar era (Lean4 migration
waves; restrict or stratify by year).

### 3b. Within-PR revision pairs (A-track gold)

For PRs with review-driven revisions: (first-pushed version, final merged
version) of the same declaration set, paired with the review comments that
demanded the change. Same author, same mathematical content → the delta is
*exactly what the community's norms required*, and the comment thread is the
articulation. This is the math analog of the N&C response-to-comment and the
CR.SE reviewer-prose corpora, and it feeds the norm-commentary track
directly (review comments → norm extraction).

### 3c. Continuous auxiliary: time-in-review, n review comments

`days_open` analog from `project_code_review_modeling.md`. Cheap, dense,
noisy; use as a secondary regression target / sanity check, not the headline.

### 3d. Hard test set only: genuine content rejections

The <10% of closed-unmerged PRs that were actually rejected on content
(maintainer comment says why; no staleness labels). Too small and too skewed
to train on; valuable as a held-out "the community said no and said why" eval.

## 4. Data plan

What exists on sk3 (`/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/`):
`prs_list.jsonl` (5,000 most-recent closed PRs, full REST objects, ~105 MB),
`prs.parquet` (flattened), `pr_details.jsonl` (7 rows — stalled),
`pr_files.jsonl` (empty). Highest PR number seen: ~40,159 → full history is
~35–40K PRs.

Fetch plan (resume-friendly, authenticated GraphQL preferred over REST for
the per-PR fan-out):

1. **Full closed-PR list** (~35K): number, title, labels, author_association,
   created/closed timestamps, base/head SHAs. (REST list endpoint, ~7× what
   we have.)
2. **Per-PR review data** (the label source): reviews (state =
   APPROVED/CHANGES_REQUESTED/COMMENTED), review comments (text + file/line),
   issue comments, timeline events (force-pushes, label additions —
   `ready-to-merge` timestamp gives the bors-queue moment). GraphQL: one
   query per PR page, ~35K PRs ≈ a few hours within rate limits.
3. **Diff reconstruction** for 3a features and 3b pairs: clone mathlib4 once;
   first-push state from the PR's earliest head SHA (GraphQL
   `commits(first:1)`), merged state from `merge_commit_sha`. Avoids
   per-file API fetches entirely. Caveat: force-pushed-over first commits are
   unreachable from the final branch — fetch PR head refs
   (`refs/pull/N/head`) and rely on timeline `HeadRefForcePushedEvent`
   before/after SHAs.
4. **Label derivation**: merged := title starts with `[Merged by Bors]` OR
   (`ready-to-merge` label ∧ closed). Then strip the prefix from `title`
   before the text ever enters a model.
5. **Closed-unmerged taxonomy** (for 3d): classify via labels
   (`merge-conflict`, `awaiting-author`, `WIP`, `please-adopt`, `stale`) +
   last-comment heuristics, manually validate a 100-PR sample
   (`feedback_validate_before_scaling`).

## 5. Metric inventory (from the already-collected online-rubrics)

`datasets/math/stackexchange/online-rubrics/claude-parsed/lean_mathlib_style_guide.md`
is the seed; mathlib also ships executable norms (CI linters), which is the
whole point — this task has the highest density of *genuinely code-checkable*
style norms of anything in the project.

**V (mechanically checkable, mostly via existing mathlib tooling):**

| Metric | Checker |
|---|---|
| type-checks / builds | `lake build` (floor: true for all merged — see note below) |
| **first-push CI history** | n failed CI runs / first-push build status from timeline — the V signal that *varies* among merged PRs |
| mathlib lint suite | `#lint` / `scripts/lint-style` (env linters: docstring presence, unused args, simp-lemma form) |
| naming grammar `A_of_B_of_C` vs statement | parse statement AST, compare to name tokens |
| layout: ≤100 cols, indent, `by` placement, calc alignment | textual linter (mathlib CI has one) |
| banned idioms: `erw`, non-terminal `simp`, `λ`/`=>` instead of `fun ↦`, squeezed-simp policy | grep/AST over proof source |
| docstrings on defs | linter |
| import minimality | `mk_all` / unused-import script |
| proof length, term-vs-tactic mode ratio | trivial |

**Where V discrimination actually lives.** Strict correctness is near-constant among
merged PRs (bors only merges green CI; maintainers generally require compilation even to
get review) — so correctness-as-V has ~zero variance and ~zero discrimination on any
within-merged label. That is a *feature* of the reframing (it isolates A+T above a
saturated V floor), but it means discriminative V must come from: (a) the metric suite
computed on the **first-push state**, not the merged state — the merged version has been
polished to pass review, the first push hasn't; (b) first-push CI failures; (c) style-lint
deltas between first push and merged state.

**Label-mechanics findings (2026-06-10, from the first ~8K PRs of the full review fetch,
2021-2023 era):**
- **CHANGES_REQUESTED is dead in mathlib**: 99.4% of merged PRs have zero — reviewers
  don't use GitHub's formal review states. The friction signal lives in **review
  threads** (line comments): P(zero threads) = 0.624 overall, stable by era (0.61-0.62
  in 2022/2023) → natural 62/38 binary split, no class-rarity problem.
- Size confound on the threads label: P(zero threads) = 0.774 in the smallest size
  quartile vs 0.529 in the largest → bin-match on size as planned (§3a).
- Author association is mild and non-monotonic (MEMBER 0.665, CONTRIBUTOR 0.621,
  COLLABORATOR 0.564) — stratify, but it's not the leak it was for merge-vs-not.
- `easy`-labeled PRs are 83.6% zero-thread — good concordance; threads-based label
  subsumes it.
- **First-commit statusCheckRollup is None for ~100% of older PRs** (GitHub GC'd old
  check runs) — first-push CI is NOT recoverable from GraphQL for the back catalog.
  First-push V must be computed by us: clone mathlib4 (running on sk3 at
  `datasets/math/mathlib/mathlib4_repo/`), reconstruct first-push trees from
  `refs/pull/N/head` + force-push events, run the style linters ourselves.

**Empirical grounding (2026-06-10, from the 5,000 most-recent closed PRs on sk3):**
83.5% merged. Median days-to-close: `easy`-labeled merged 0.9 vs non-easy merged 5.6
(q75: 3.7 vs 19.0) — review friction is real and wide-spread, so the 3a/3c labels have
signal to find. `easy` covers only 433/4,174 merged → y=1 needs the zero-CHANGES_REQUESTED
definition from the review fetch, not the label alone. `author_association=NONE` occurs
*only* in unmerged (120/826) — stratify by association among merged and the leak is gone.
Modern artifact: 141 unmerged PRs carry an `LLM-generated` label — a brand-new community
norm, worth keeping as its own category. 

**Friction dataset state (2026-06-10 evening).** Built from the completed 37,249-PR
fetch: 32,787 merged, P(frictionless)=0.513 full-history. `friction_dataset_v2.csv.gz`
(16,884 rows, 50/50, matched within size×prefix×year×assoc×**primary-topic** cells) is
canonical; v1 (19,356, no topic matching) kept for comparison. `friction_full_v2.csv.gz`
carries unbalanced rows + continuous targets. Audit (`friction_audit/REPORT.md`): title
TF-IDF 0.627 test (legitimate task-type signal — `theorem`/`define`/`tactic` draw
review), metadata-only 0.509, exact cell balance. **Banned model inputs**: label
components (`n_review_threads`, `n_reviews`, `n_changes_requested`), post-treatment
(`days_open`, `n_force_pushes`, `n_commits`, `n_issue_comments`, `closed_at`), and
label-adjacent (`easy`, raw `labels`). Final-state `size`/`changed_files` OK as strata
only — recompute first-push size for models.

**A (LLM-judge over statement + proof ± library context):**

- duplicate / near-duplicate of existing API ("this is `Foo.bar`") — most
  common substantive review comment class, semi-retrievable with embeddings
  over mathlib's decl docstrings.
- stated at the right typeclass generality (Monoid vs Group vs the ambient
  one used in the proof).
- missing-API smell (`erw`/`rfl` workarounds signal absent simp lemmas — the
  style guide says so explicitly).
- naming *semantics* (grammar can be checked; whether the name is the
  idiomatic English reading is a judgment).
- placement: right file/module for the declaration.
- proof idiom quality / golfing appropriateness.

**T (residual):** "is this lemma worth having", generality-vs-readability
tradeoffs, aesthetic golfing. Measured as dense-model ceiling minus V+A bank,
as everywhere else.

## 6. Expected V/A/T signature (hypothesis to test)

Because correctness is machine-checked, V should explain almost none of the
*review-friction* label variance (it's near-constant among merged PRs) —
i.e. mathlib should show the **smallest C−B articulability gap** of any task
if the thin-rule story is right: this community has spent a decade thinning
its thick rules into linters. The interesting outcome either way:

- small C−B → supports "mature norm communities thin their rules"
  (verifiability-cycle story, `project_refactoring_algorithm_idea.md`).
- large C−B despite max thinning → strongest possible evidence for the
  rubric critique: even the most rule-codified human evaluation community
  leaves an inarticulable residue.

## 7. Ground-truth lint/warning probe, 2026-06-11

Upgrade of the first-push V metrics from regex proxies to actual tooling, over
the 504 lake-probed PRs (462 built + 42 build_failed in
`lake_probe_results.csv`). Script: `scripts/lake_lint_probe.py` (sk3). Per PR:
re-checkout first-push worktree, `lake exe cache get <files>`, delete the
touched modules' own build artifacts (forces genuine re-elaboration; deps stay
cache-hot), `lake build <mods>` with FULL output captured, then
`lake exe runLinter <Module>` per touched module (300s/module, 600s/PR budget).
Output: `lake_lint_results.csv`; full raw logs in `lint_probe_logs/pr{N}.txt`.

**Status: PARTIAL (N=76/504, run in flight on sk3).** Throughput ~2.5-4
min/PR (runLinter on simp-heavy modules is the bottleneck; ~20% of PRs hit the
lint budget), full sweep lands in roughly a day. Resume:
`HOME=/lfs/skampere3/0/alexspan nohup ~/envs/norm-scraper/bin/python
scripts/lake_lint_probe.py >> lake_lint.log 2>&1 &` (resume-safe: skips PR
numbers already in the output CSV; year-round-robin order keeps partials
balanced).

### Findings at N=76 (interpret as directional, CIs are wide)

**1. Build-time warnings are a dead feature.** 0 warnings across all 69
rebuilt states; only 2 `warning:` lines in 77 full logs, both lake
manifest-out-of-date noise. 2022-23 era built with `-DwarningAsError=true`
(warnings would have been build failures), and modern first-push states of
*eventually-merged* PRs are warning-clean. All n_warn_* AUCs = 0.500 exactly.
The regex-proxy warning metrics in METRICS.md have no ground-truth signal to
recover here.

**2. The genuine linter suite runs almost everywhere (incl. 2022).**
`lake exe runLinter <Module>` worked in every era we touched
(std4/batteries exe; lint_available 80% — misses are lint *timeouts*, not
unavailability). 12.7% of lintable first-push states have >=1 lint error;
linter breakdown is dominated by `simpNF` (103 hits) then `docBlame` (21).
n_lint_errors AUC vs y = 0.45 (n=55) — no positive signal yet, slightly
inverted; consistent with CI already gating lint at PR time so lint errors at
first push mostly flag *stale-cache/era* quirks, not future review friction.

**3. t_rebuild (cache-hot wall time of the touched modules) is the only live
feature so far:** log t_rebuild AUC = 0.555 (n=69), per-file 0.514. Weak
proof-weight proxy, as expected.

**4. Revised "first drafts wrong" headline.** Full error text now separates
infra from genuine elaboration errors:

| class | n | share of 76 | note |
|---|---|---|---|
| lean_error (genuine) | 2 | 2.6% | unknown identifier / unknown constant |
| port artifact ("unknown package 'Mathbin'") | 2 | 2.6% | 2022-23 mathport era, not author error |
| infra (binary-package / lake-layout) | 3 | 3.9% | 2022 toolchain + 2024 layout |

The old conflated number (~8% build-failure) splits roughly in half: genuine
first-draft elaboration errors are ~2-4%, the rest is era infrastructure.
By y (still tiny n): lean_error y=0 3/38 vs y=1 1/38 — no evidence yet that
high-friction PRs start more broken; if anything inverted.

**5. Ops notes.** runLinter is the cost center (simpNF on simp-heavy modules
can exceed 10 min; budget caps it). `lake exe` spawns a child binary that
survives a naive timeout kill — the script kills the whole process group
(start_new_session + killpg). 15/69 built PRs recorded >=1 lint timeout.

Analysis script: `scripts/analyze_lake_lint.py` (run on sk3 against
`lake_lint_results.csv`; rerun when the sweep finishes for the final table).

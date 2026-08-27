# Mathlib review-friction dataset — leakage/confound audit

Dataset: `friction_dataset.csv.gz`, 19356 rows (train 15434 / eval 1966 / test 1956), pos rate 0.500. Label: y=1 ⟺ zero review threads among merged PRs. All models fit on train, evaluated on test only.

## 1. Title-only model

- **Bors-prefix check: FAIL (minor build bug)** — 18 titles (0.09%) still contained `[Merged by Bors]`. Root cause (verified against raw `pr_reviews_mathlib4.jsonl` on sk3): these PRs were re-landed through Bors twice and have a **doubled** prefix; the anchored regex in `build_friction_dataset.py` strips only one copy. Labels among them are mixed (10 y=0 / 8 y=1) and every row in the dataset is merged, so this is **not a label leak** for the friction label — but their `conv_prefix` was mis-binned as `OTHER`, so they were matched in the wrong cell. Fix: replace `BORS_PREFIX` with `^(\[Merged by Bors\]\s*-\s*)+`. Residual prefixes were stripped before the title model below.
- TF-IDF (word 1–2 grams, min_df=3, 7952 features) + LR: **train AUC 0.805, test AUC 0.627**.
- Top ±30 features in `title_top_features.csv`. Top 15 each direction:

| → y=1 (frictionless) | coef | → y=0 (friction) | coef |
|---|---|---|---|
| `align` | +1.96 | `theorem` | -1.55 |
| `port probability` | +1.70 | `define` | -1.49 |
| `conv` | +1.63 | `part` | -1.43 |
| `feat more` | +1.59 | `comment` | -1.40 |
| `revert` | +1.58 | `mk_all` | -1.34 |
| `data mvpolynomial` | +1.55 | `grouptheory groupaction` | -1.26 |
| `grind` | +1.52 | `to use` | -1.23 |
| `split` | +1.49 | `argument` | -1.23 |
| `protect` | +1.44 | `tactic` | -1.22 |
| `feat make` | +1.41 | `categorytheory functor` | -1.22 |
| `port ringtheory` | +1.40 | `principle` | -1.21 |
| `port geometry` | +1.39 | `feat ennreal` | -1.19 |
| `concretecategory` | +1.38 | `inv` | -1.19 |
| `port analysis` | +1.36 | `extract` | -1.16 |
| `chore classify` | +1.35 | `atoms` | -1.16 |

## 2. Single-feature test AUCs (rank AUC, no fitting)

| feature | test AUC (raw) | AUC (oriented) | post-treatment? |
|---|---|---|---|
| n_force_pushes | 0.241 | 0.758 | **YES — descriptive only** |
| n_commits | 0.324 | 0.676 | **YES — descriptive only** |
| days_open | 0.329 | 0.671 | **YES — descriptive only** |
| n_issue_comments | 0.333 | 0.667 | **YES — descriptive only** |
| easy | 0.520 | 0.520 | no |
| new_contributor | 0.481 | 0.519 | no |
| changed_files | 0.483 | 0.517 | no |
| year | 0.490 | 0.510 | no |
| size | 0.504 | 0.504 | no |

Post-treatment variables (`days_open`, `n_force_pushes`, `n_issue_comments`, `n_commits`) accrue **during** review and are partially downstream of the label (a review thread causes revision commits, force-pushes, comments, and longer time-to-merge). They must never be model inputs — reported here only to describe the label's footprint.

## 3. Residual cell balance (full dataset, by construction should be ≈0.5)

**size_bin** (max |dev| = 0.0000):

| value | P(y=1) | n |
|---|---|---|
| 0 | 0.5000 | 3636 |
| 1 | 0.5000 | 5414 |
| 2 | 0.5000 | 5022 |
| 3 | 0.5000 | 5284 |

**conv_prefix** (max |dev| = 0.0000):

| value | P(y=1) | n |
|---|---|---|
| OTHER | 0.5000 | 268 |
| chore | 0.5000 | 6386 |
| ci | 0.5000 | 84 |
| doc | 0.5000 | 430 |
| docs | 0.5000 | 46 |
| feat | 0.5000 | 10452 |
| fix | 0.5000 | 670 |
| golf | 0.5000 | 6 |
| move | 0.5000 | 6 |
| perf | 0.5000 | 130 |
| refactor | 0.5000 | 800 |
| style | 0.5000 | 78 |

**year** (max |dev| = 0.0000):

| value | P(y=1) | n |
|---|---|---|
| 2021 | 0.5000 | 56 |
| 2022 | 0.5000 | 528 |
| 2023 | 0.5000 | 5376 |
| 2024 | 0.5000 | 5008 |
| 2025 | 0.5000 | 5748 |
| 2026 | 0.5000 | 2640 |

Joint (size_bin × conv_prefix × year × association) cells: 365 cells; 0 have P(y=1) ≠ 0.5 exactly (max dev 0.0000). Construction guarantees exact within-cell balance.

## 4. Metadata-only model

LR on size_bin/conv_prefix/year/association one-hots + log(changed_files): **train AUC 0.524, test AUC 0.509**.

## 5. `easy` label

P(y=1 | easy=True) = **0.602** (n=1884); P(y=1 | easy=False) = 0.489 (n=17472).

The raw concordance (README label-mechanics: 83.6% of `easy` PRs are zero-thread) is mostly absorbed by the cell balancing (easy PRs are small → small size_bin cells are downsampled toward 0.5), leaving single-feature test AUC ≈0.52 here. But the mechanism is the problem, not the magnitude: maintainers apply `easy` partly *because* review turned out to be trivial, i.e. it is label-adjacent / post-treatment-ish. **Recommendation: ban `easy` as a model input.**

## 6. Scrutiny of title-feature classes (full dataset)

| token | n | P(y=1 \| token) |
|---|---|---|
| `port` | 3017 | 0.611 |
| `split` | 500 | 0.628 |
| `move` | 421 | 0.606 |
| `grind` | 146 | 0.596 |
| `revert` | 32 | 0.750 |
| `golf` | 525 | 0.482 |
| `deprecate` | 287 | 0.484 |
| `theorem` | 291 | 0.395 |
| `define` | 170 | 0.259 |
| `tactic` | 408 | 0.407 |

| topic label (NOT in matching cells) | n | P(y=1) |
|---|---|---|
| t-algebra | 3630 | 0.447 |
| t-analysis | 1122 | 0.434 |
| t-topology | 1088 | 0.469 |
| t-category-theory | 1374 | 0.556 |
| t-number-theory | 362 | 0.387 |
| t-combinatorics | 303 | 0.360 |
| t-meta | 781 | 0.421 |
| t-order | 902 | 0.473 |
| t-measure-probability | 820 | 0.504 |
| t-data | 796 | 0.469 |
| t-set-theory | 216 | 0.537 |
| (no topic label) | 7845 | 0.554 |

Classification of the top title features:

- **Quality / task-type signal (legitimate):** `theorem`, `define`,
  `definition`, `formula`, `constructors`, `tactic` → friction (new
  mathematical content and new metaprograms genuinely draw review);
  `revert`, `align`, `import`, `doc fix`, `deprecation`, `update mathlib`,
  `mathlib dependencies` → frictionless (mechanical maintenance). This is
  exactly the signal the label is supposed to carry.
- **Stratification residual (finer-grained than the matched cells):**
  `port *` (P(y=1)=0.611, n=3017), `split` (0.628), `move` (0.606), `grind`
  (0.596). `split`/`move` are conv_prefixes and balanced *at title start*,
  but the tokens recur mid-title under `chore:`/`feat:` prefixes; `port` is
  the Lean3→Lean4 porting wave — pre-reviewed in mathlib3, hence
  frictionless. Borderline-legitimate (ports really are lower-risk work)
  but a model can use them as PR-type shortcuts.
- **Math-area confound (NOT balanced — gap vs README §3a plan):** the build
  matched on (size × conv_prefix × year × association) but **not** on
  `t-*` topic labels, and areas are imbalanced: t-combinatorics 0.360,
  t-number-theory 0.387, t-meta 0.421, t-analysis 0.434, t-algebra 0.447 vs
  t-category-theory 0.556, t-set-theory 0.537, no-label 0.554. This is why
  area tokens (`feat numbertheory`, `feat linearalgebra`, `ringtheory
  ideal`, `computability`, `measuretheory`, `euclidean`) appear in the top
  coefficients. Plausibly part-genuine (different reviewer cultures per
  area) and part-confound (reviewer availability per area). Recommend:
  include topic labels in the matching cells on the next rebuild, or always
  report per-area AUCs.
- **Leakage: none found.** No `bors`/`easy`/`merge`/process tokens in the
  vocabulary's top coefficients; `golf` (0.482) and `deprecate` (0.484) are
  ~neutral, contrary to the prior worry.

## 7. Verdict and banned columns

**Verdict: the friction label is clean enough to model against first-push
code features.** Metadata-only test AUC 0.509 confirms the cell matching
removed the size/prefix/year/association confounds; exact 0.5 balance holds
in every joint cell; no leak tokens in the title model; the title signal
that remains (test AUC ~0.63 vs train 0.80 — heavily memorization-limited)
decomposes into legitimate task-type signal plus two named residuals
(port/refactor mid-title tokens, math-area imbalance). The post-treatment
variables behave exactly as post-treatment variables should (oriented AUC
0.67–0.76), confirming the label has a real process footprint.

**Banned columns for downstream modeling** (never as model inputs):

| column | reason |
|---|---|
| `n_review_threads` | label definition (y = [threads == 0]) |
| `n_reviews`, `n_changes_requested` | direct review-process counts (label-adjacent) |
| `days_open`, `closed_at` | post-treatment: review friction extends time-to-merge |
| `n_force_pushes` | post-treatment: revisions after review (oriented AUC 0.76) |
| `n_commits` | post-treatment: review-driven revision commits (0.68) |
| `n_issue_comments` | post-treatment: discussion accrues during review (0.67) |
| `labels` (raw string) | contains post-review process labels (`ready-to-merge`, `maintainer-merge`, `awaiting-review`, …) |
| `easy` | label-adjacent: applied partly *because* review was trivial |
| `head_oid`, `merge_commit_oid` | final post-review state (use `first_commit_oid` for first-push reconstruction) |
| `state`, `merged` | constant among rows |

**Caution (stratifiers / keys only, not features):** `additions`,
`deletions`, `size`, `changed_files`, `size_bin` are measured on the
**final merged** state, which includes review-driven revisions — mildly
post-treatment. They are fine as matching strata (and are ~0.5 AUC after
balancing) but first-push size should be recomputed from the cloned repo
for any size-aware model. `number`, `first_commit_oid` are join keys.

**Allowed inputs:** `title` (Bors-prefix-stripped — after fixing the
doubled-prefix bug), first-push code/diff features reconstructed from
`first_commit_oid`, `conv_prefix`, `year`, `author_association`,
`topic_labels`, `new_contributor`, `llm_generated`, `created_at`.

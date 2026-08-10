# Mathlib + homepage corrective dense runs (user-directed, 2026-08-08)

Context: `notes/2026-07-27__vat-run-registry.md`, entry **"2026-08-08 — CORRECTION (user
catch): MATHLIB AND HOMEPAGE 'TERMINAL' VERDICTS RETRACTED AS OVERSTATED"**. Two cells had
been written off as instrument failures. In both cases the failure was a *design* failure,
not a *cell* failure, and this note runs the two corrective designs.

- **JOB 1 — mathlib**, chapter `notes/2026-08-08__scaleupA_dense_reruns.md` §Job 1.
  The class-weighted area-grouped run on the canonical de-confounded slice returned eval
  .5643 (3-seed mean, spread .126) / test .4670 while a class-weighted TF-IDF on the
  *identical* rows scored .6774/.7856. A bag-of-words beating an 8B LoRA on the same rows
  means the training run was starved, not that the cell is dead: n=7,956 at 94.3% positive
  leaves ~450 negatives, ~363 of them in the train fold.
  **Correction: TRAIN-BIG / EVAL-CANONICAL.**
- **JOB 2 — homepage**, chapter `notes/2026-08-08__scaleupC_builds.md` §Build 4.
  The outlet-held-out design (8 outlets, 1 held out for eval, 1 for test) gave eval
  .4322/.4590/.4393 and test .7361/.7429/.7534 — the two held-out outlets disagree in
  *sign*. That is grouping variance at k=2, not proof of no transfer.
  **Correction: STORY-GROUPED T on the same scored population.**

Conventions honoured: sk3 (`HOME=/lfs/skampere3/0/alexspan`), GPU ledger claimed before
launch and both jobs on ONE card (GPU=5, stacked processes), only own PIDs touched,
`latex/` untouched, no prior artifact overwritten (both builds write to new directories,
both result JSONs are new filenames).

---

## JOB 1 — mathlib TRAIN-BIG / EVAL-CANONICAL

### Build

`datasets/math/mathlib/build_dense_bigtrain.py` → `datasets/math/mathlib/dense_standard_bigtrain/`

| fold | rows | source | areas | pos rate |
|---|---:|---|---|---:|
| train | 29,324 | pre-audit population, held-out area groups removed | 37 | .9078 |
| eval | 932 | **verbatim copy** of `dense_standard_cw/split/eval.csv` | Analysis | .9442 |
| test | 795 | **verbatim copy** of `dense_standard_cw/split/test.csv` | CategoryTheory, Control | .9497 |

Pre-audit population = `accept_reject_dataset.csv.gz` (35,796 rows) joined to the
status-200 diffs in `pr_diffs.jsonl` (35,619), minus 42 rows whose diff is empty (no
artifact to read), = 35,577 usable; minus the three held-out area groups → 29,324.

Text is `diff_noauth` (author-stripped diff, no title) and group is the top-level
`Mathlib/<Area>/` path, both computed with the canonical C-leg functions
(`save_deconf.py` / `mathlib_remeasure2.py`), identically on all three folds.

### Area-grouped holdout integrity — assertion evidence

Both assertions are executed in `build_dense_bigtrain.py` (the build aborts if either
fails) and their text is copied into `dense_standard_bigtrain/manifest.json`:

1. **Zero group overlap / zero row overlap.**
   `train areas (37) INTERSECT held areas ['Analysis','CategoryTheory','Control'] = EMPTY;
   train row_ids INTERSECT eval+test row_ids = EMPTY — PASS`
2. **Provenance.** Every one of the 1,727 canonical eval+test rows reproduces
   **byte-identically — both its text and its area label** — when rebuilt from
   `accept_reject_dataset.csv.gz` + `pr_diffs.jsonl` through the canonical
   `strip_author`/`area` functions. So the train fold and the canonical eval folds come
   out of one text pipeline; the only thing that differs between them is which rows the
   de-confounding audit kept.

A useful cross-check falls out of the strata census: the number of train-big rows that
pass the full canonical filter is **6,229 — exactly the canonical train fold's row count**.
The canonical train fold is, precisely, the audit-surviving subset of this train fold.

### What the audit removed, and how that bounds interpretation

The canonical slice is the pre-audit population through two filters:

| filter | script | what it drops | why |
|---|---|---|---|
| regime | `save_deconfounded.py` | everything that is not `conv_prefix=='feat'` **and** `year>=2025` | mathlib3 port era (`#align`) + change-type confound; title-only AUC collapses .638 → .566 there, i.e. that signal *was* confound |
| hygiene | `finalize_slice.py` | `additions==0` or `>1000`; rejects with `n_review_threads==0` | size confound; and "abandoned" PRs, which are y=0 **without a reviewer decision** |

Train-big re-admits all of it. Census of the 29,324 train rows (manifest
`readmitted_confound_strata`): 22,673 fail the regime filter, 990 are size outliers,
**1,635 are abandoned rejects**, and 6,229 pass everything. Negatives go from **363**
(canonical train) to **2,705** — a 7.5× increase, which is the entire point of the fix.

The third stratum is the one that changes *meaning*: for 1,635 train rows y=0 records
"the author walked away", not "the reviewers said no". So the train and eval folds do not
share one label semantics. That is tolerable here — and only here — because **every
evaluated row is a canonical de-confounded row** and the areas are disjoint, so no
confounded row can leak into the readout. The cost is a train→eval distribution shift,
which makes this T a **conservative lower bound** on the canonical-row dense ceiling
rather than a best estimate of it.

That shift is not hypothetical; it is measurable. A class-weighted linear TF-IDF, scored
on the canonical eval/test rows throughout, varying only which training rows it sees
(`outputs/tfidf_ablation_mathlib_bigtrain.json`):

| train fold | n | negatives | eval AUC | test AUC |
|---|---:|---:|---:|---:|
| A. full train-big | 29,324 | 2,705 | .6489 | .6415 |
| B. − regime filter failures (feat ∧ ≥2025 kept) | 6,651 | 780 | **.6754** | **.7913** |
| C. − size outliers | 28,334 | 2,513 | .6458 | .6400 |
| D. − abandoned rejects | 27,689 | 1,070 | .6643 | .6966 |
| E. full canonical filter (= the canonical train fold) | 6,229 | 363 | .6796 | .7883 |
| F. random 6,229-row subsample of train-big | 6,229 | 554 | .6016 | .6829 |

Reading (descriptive): the whole distribution-shift cost is carried by the **regime**
filter, not by size or by size-of-training-set. Arm B keeps only regime-matched rows and
recovers essentially all of the canonical arm's AUC (.675/.791 vs .680/.788) while
carrying 2.1× its negatives; arm F, the same *number* of rows drawn at random from the
shifted pool, does not (.602/.683). Abandoned rejects cost about a third of the gap
(D vs A: +.015 eval / +.055 test). For a bag-of-words, then, training big is *worse* on
canonical rows than training small-and-matched — the question the dense arm answers is
whether an 8B reader trades that shift for the 7.5× negatives profitably.

### Result

<!--RESULTS_MATHLIB-->

---

## JOB 2 — homepage STORY-GROUPED T

### Build

`datasets/news-homepages/build_dense_storygrouped.py` →
`datasets/news-homepages/va/dense_standard_storygrouped/`

Population is the scale-up-C scored A/V population **unchanged**: n=12,998, pos-rate
.5006, 1,229 snapshots, 8 outlets (asserted against `homepage_curation_ledger.json` to
float precision). The bank is not touched — its coherent-vs-scrambled failure (.387) is a
separate A-instrument issue.

### Which grouping unit, and why — state it plainly

**The population has no story/article ID, so the grouping unit is the `snapshot_id`: one
outlet's homepage capture at one moment, i.e. a date-block key.** It is also the historic
unit — the registry's `T .824 groupsplit prov` came from a snapshot-grouped sweep — and
the unit of wave C's own secondary readout.

The two obvious alternatives were rejected for stated reasons:

- **Article-only grouping** (11,592 distinct normalised headlines) splits *every* snapshot
  across folds. Each item's text carries a `CONTEXT` field byte-identical for all rows of
  a snapshot, and the label is a *within-snapshot* contrast (top vs bottom half of the
  top-30% zone), so the model would see ~80% of each eval snapshot's labelled rows during
  training. That is a direct leak and would inflate T.
- **Union of the two** (connected components of the snapshot × headline bipartite graph)
  removes both leaks but chains through persistent wire stories: the largest component is
  **3,186 rows = 24.5% of the corpus, spanning three outlets**. It would force most of
  four outlets into a single fold.

Snapshot grouping alone leaks the other way — 800 normalised headlines recur across more
than one snapshot (2,199 rows), because a story sits on the homepage across successive
captures. That leak is closed by **de-duplication on the train side**: any train row whose
normalised headline also appears in eval or test is dropped (630 rows, 300 distinct
headlines). Dropping from train rather than from eval/test keeps the readout folds intact
and representative.

Snapshots are packed into folds **within outlet** with the campaign's standard
`stable_hash_bucket_map` (deterministic greedy + hill-climb, row-count and pos-rate
matched, no seeded shuffle). Packing globally instead lands eval/test at ~36% Guardian —
Guardian's captures are the smallest, so the packer fills the small buckets with them —
and the resulting T would describe a Guardian-heavy subsample rather than the population.

Asserted in the build: zero snapshot overlap between folds; zero train↔held-out article
overlap; post-dedup fractions inside the trainer's 80/10/10 ±2pp gate.

| fold | rows | fraction | snapshots | pos rate |
|---|---:|---:|---:|---:|
| train | 9,737 | .7873 | 912 | .4983 |
| eval | 1,313 | .1062 | 132 | .5004 |
| test | 1,318 | .1066 | 178 | .5008 |

**Outlet composition (share of each fold)** — the point of the within-outlet packing:

| fold | bbc | cnn | guardian | latimes | nytimes | reuters | wapo | wsj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | .140 | .137 | .138 | .129 | .123 | .131 | .134 | .070 |
| eval | .131 | .131 | .130 | .130 | .130 | .133 | .137 | .079 |
| test | .128 | .130 | .130 | .124 | .140 | .128 | .127 | .094 |

All eight outlets appear in all three folds at essentially the population mix (WSJ is
thinner everywhere because only 61 of its captures resolve — flagged in the wave-C build).

### Result

<!--RESULTS_HOMEPAGE-->

---

## Runs

| job | dir | recipe | seeds |
|---|---|---|---|
| mathlib train-big | `datasets/math/mathlib/dense_standard_bigtrain/` | dense standard + `--class_weight_auto`, select-on-eval | 42 (+1,2 if wall-clock permits) |
| homepage story-grouped | `datasets/news-homepages/va/dense_standard_storygrouped/` | dense standard, select-on-eval (population balanced, no class weighting) | 42, 1, 2 |

Runners: `methods/dense/run_corrections_mathlib_homepage.sh` (job 1 + a job-2 block that
no-ops on the RUN_DONE sentinels) and `methods/dense/run_homepage_storygrouped.sh`
(job 2, stacked as a second process on the same card once job 1 measured out at
~6.5 h/seed). Scoring is the unmodified `methods/dense/score_eval_dense_v4.py`.

`methods/dense/run_bigtrain_eval_canonical.py` is a new 40-line harness for job 1 only.
`train_reward_model.get_or_create_fixed_split()` hard-asserts that an on-disk split is
80/10/10 ±2pp, which a train-big design (.944/.030/.026) can never satisfy. The harness
imports the trainer, sets its three fraction constants to the observed on-disk fractions
so that assertion is a no-op for this run, and calls `train_reward_model.train()`
otherwise **completely unmodified** — deliberately, so that the shared trainer other
agents are running concurrently is not edited.

## Artifacts

- `datasets/math/mathlib/build_dense_bigtrain.py`, `datasets/math/mathlib/dense_standard_bigtrain/{data.csv,split/,manifest.json}`
- `datasets/news-homepages/build_dense_storygrouped.py`, `datasets/news-homepages/va/dense_standard_storygrouped/{data.csv,split/,manifest.json}`
- `methods/dense/run_bigtrain_eval_canonical.py`, `methods/dense/run_corrections_mathlib_homepage.sh`, `methods/dense/run_homepage_storygrouped.sh`
- `methods/taste_decomposition/results/samerows_T_mathlib_bigtrain.json`
- `methods/taste_decomposition/results/samerows_T_homepage_storygrouped.json`
- `outputs/tfidf_comparators_corrections.json`, `outputs/tfidf_ablation_mathlib_bigtrain.json` (sk3)
- GPU ledger: `GPU=5` claimed 2026-08-08T17:25:41Z (`agent=claude-mathlib-homepage-corrections`,
  nvidia-smi 0 MiB / 0% before claim), job 2 stacked as a second process on the same card
  at 17:33Z.

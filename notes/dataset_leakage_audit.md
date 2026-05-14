# Dataset leakage and confounder audit

Running log of every leakage path, confounder, and shortcut feature we've found
across the 12 modeling datasets, plus the cleanup status for each. Update this
whenever a new leak is identified or a fix is shipped.

**Conventions**

- *Confounder*: a feature that is correlated with the label but not causally tied
  to the construct (e.g. outlet identity for newsworthiness). Cleanup =
  balancing or removing.
- *Leak*: train-time information that should not be available at inference,
  most often caused by a row-level split that lets correlated rows cross the
  train/val boundary. Cleanup = group-aware split.
- *Status*: ✅ fixed and verified, 🛠 fix in flight, ⚠ identified but not yet
  fixed, ❓ not yet audited.

**Default training pipeline assumption**: `methods/dense/train_reward_model.py`
uses 80/10/10 row-level `train_test_split` unless `--group_split_column` is
passed (added 2026-05-12).

---

## news-homepages — newsworthiness

Task: predict whether an article ranks in the top half of a homepage's
top-30% zone.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Outlet identity (NYT vs WSJ vs CNN ...) was a near-perfect shortcut. AUC ~0.99 before fix. | ✅ | Per-outlet 50/50 balancing in `build_homepage_dataset.py`. |
| 2 | Position-in-CONTEXT exposed the label (CONTEXT was ordered by page position). | ✅ | `process_snapshot` shuffles `other_headlines` before joining as CONTEXT. |
| 3 | Topic imbalance (covid-era stories dominated label=1; lifestyle/sports dominated label=0). | ✅ | LDA k=50 + within-topic balancing in `topic_balance.py`. AUC dropped to ~0.75 after deconfounding. |
| 4 | **Snapshot-level leakage via CONTEXT field**: every row's CONTEXT contains the other headlines from the same homepage snapshot. 99% of focal headlines also appear in some other row's CONTEXT. With row-level split, train sees a snapshot's headlines as foci and val sees the rest of the same snapshot — model gets to learn the news cycle / outlet style from training. | 🛠 | Found 2026-05-12. Sweep `homepage_newsworthiness_sweep_llama8b` hit AUC 0.86 (well above the 0.75 deconfounded estimate) → leakage signature. Fix: derived `snapshot_id` via md5 hash of sorted headline-set per row (`scripts/add_snapshot_id_homepages.py`) → 21,951 unique snapshots, mean 8.4 rows/snapshot. New dataset: `homepage_newsworthiness_topic_balanced_groupsplit.csv.gz`. Patched `train_reward_model.py` to support `--group_split_column`. New sweep launched: `runs/homepage_newsworthiness_groupsplit_sweep_llama8b` on GPU 5. |

---

## creative-writing — WritingPrompts (r/WritingPrompts via Arctic Shift)

Task: predict whether a story scored ≥10 upvotes on its prompt.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Near-duplicate stories (cross-posts / reformatted reposts) inflated effective dataset size and could leak across split. 42.6% dupe rate in original creative-writing pipeline. | ✅ | MinHash LSH on char 9-grams (num_perm=128, threshold=0.5) reduced 87K → 50K, then expanded to 100K from new Arctic Shift pull. New pipeline checked: 243 exact dupes / 100K (0.24%). |
| 2 | Length bias: high-scoring stories were 1.30× longer on average. | ✅ | Per-length-bucket balancing (`build_writingprompts_balanced.py`). Final ratio 1.004×. |
| 3 | **Prompt-level leakage**: 15,391 prompts have ≥2 stories (42.1% of rows in multi-story groups). Within those, MI(prompt, label) = 0.318 nats ≈ **48% of H(label)**. With row-level split, the model can memorize per-prompt scoring patterns from training and use the prompt text (which is present in input) to predict in val. | ✅ | Found 2026-05-12. Fixed by adding `prompt_id = md5(prompt)` and using `--group_split_column prompt_id`. New sweep `creative_writing_groupsplit_sweep_llama8b` on GPU 7. Split: 56,361 train / 7,046 eval / 7,046 test prompt-groups, zero overlap. |
| 4 | **Reddit moderator-bot residue**: low-score "stories" are often the mod-removal template (`Hi u/<user>, this submission has been removed. ... reddit.com/r/writingprompts/...`). Trigram `reddit com r` appears 10,834 times in label=0 vs 2,126 in label=1; `com r writingprompts` 10,243 vs 708. | ✅ | Found 2026-05-12. Fixed by filtering rows matching regex `this submission has been removed` (1,957 rows, all label=0; zero false positives in label=1). New dataset: `writingprompts_modeling_clean.csv.gz` (96,080 rows, 48,040 per class, length ratio 1.001). Builder: `datasets/creative-writing/rebuild_writingprompts_clean.py`. |
| 5 | `EDIT:` / `Edit:` marker imbalance: 1,439 in label=1 vs 866 in label=0 (high-scoring posts get more author edits because they get more attention). | ⚠ | Found 2026-05-12. Not yet stripped — comparing AUC with/without filtering will tell us if the leakage is material. Leave for now; revisit if the group-split run still overperforms. |
| 6 | Style residues in pos-leaning trigrams: `the dark lord`, `the dragon s`, `a thousand years`, `thanks for reading` are likely fanfic / serial-writing markers. If the same authors appear across train and val, this leaks author identity. | ❓ | Author id not currently tracked in the dataset. Hard to audit without re-linking to Reddit usernames. |

---

## peer-review — accept/reject from review text + paper

Task: predict accept vs reject from paper text (with reasoning column = review
text in the `_with_reasoning` variant).

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Venue-level confounder: ICLR / NeurIPS oral vs spotlight vs poster vs reject base rates differ by venue. | ❓ | 70,192 rows across multiple venues (ICLR, NeurIPS, eLife, others). Not balanced across venues. |
| 2 | Year leakage: paper-style and topic drift over time means model can pick up "looks like 2023 NeurIPS" as a proxy. | ❓ | Year column is present; not used as split key. |
| 3 | Paper-id duplication: same paper across multiple venues / versions could appear in both splits. | ❓ | `paper_id` column exists — would be the natural group key for a leak-safe split. |
| 4 | Review-text leakage in `_with_reasoning` variant: reviews explicitly state accept/reject reasoning, so they shouldn't be in the *input* unless task is "predict given review." | ✅ | By design — separated into `_with_reasoning.csv.gz` and only used for explanation experiments, not for the main accept/reject classifier. |

---

## code-review — GitHub PR accept/reject

Task: predict whether a PR is merged (or `days_open` as alternative label).

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Severe class imbalance (86 accept / 14 reject) handled via class weights, but no other deconfounding. | 🛠 | Class weights set. |
| 2 | Repo-level confounder: some repos merge everything, some reject many. Repo identity is an outlet-style shortcut. | ❓ | 141K PRs across many repos. No per-repo balancing currently. |
| 3 | Author-level leakage: same author's PRs across train/val could leak (some authors have known acceptance rates). | ❓ | Not split-aware. |
| 4 | Time leakage: PR styles and review norms drift; row-level split mixes years. | ❓ | Not split-aware. |
| 5 | Review-comment leakage in `_with_reasoning` variant: comments often explicitly negotiate the merge decision. | ✅ | Separated into `_with_reasoning.csv.gz`. |

---

## press-releases — newsworthy / placement

Task: predict whether a press release got coverage / placement.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Cleaned dataset uploaded as `press_release_modeling_dataset_clean.csv.gz`. | ✅ | Replaces the original. |
| 2 | Outlet/industry confounder, length, year. | ❓ | Not audited end-to-end in this audit pass. |
| 3 | Per memory `project_press_release_results.md`: both methods achieved ~0.5 AUC, suggesting the task is genuinely hard OR a signal is missing — but not obviously leaky either way. | ✅ | Low AUC is a *negative* leakage indicator. |

---

## math-stackexchange — answer quality

Task: predict whether an answer is highly upvoted / accepted.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Question-level grouping: same question has many answers. Row-level split → leak. | ❓ | `Posts.xml` extracted; not yet built into a modeling dataset. |
| 2 | Time / topic drift. | ❓ | Not audited. |
| 3 | Comment text (7.1M comments in `Comments.xml`) is candidate "reasoning" feature; if added to input, it directly states quality. | ⚠ | Don't put comments in the input for the prediction task; use only in `_with_reasoning` variant. |

---

## notice-and-comment — substantive vs procedural / accept-into-rule

Task: predict whether a public comment was treated as substantive.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Agency-level confounder: different agencies have different acceptance norms. | ❓ | Per memory `project_nc_pipeline_state.md`: dataset is `notice_and_comment_len_balanced.csv.gz`. AUC 0.614 was reported. Per-agency balancing not confirmed. |
| 2 | Rule-level grouping: comments on the same proposed rule are correlated; row-level split leaks rule context. | ❓ | Rule_id may exist in raw data — would be natural group key. |
| 3 | Length balancing already applied (`_len_balanced` suffix). | ✅ | |
| 4 | Topic balancing via LDA on `_with_topics` variant. | 🛠 | Topic file exists; not yet used as a group key. |

---

## patents — first-draft / final-outcome (with/without applicant cites, with/without examiner cites)

Task: 4 binary tasks per memory `project_patents_first_draft_prediction.md`.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Examiner-cite leakage: examiner citations are added *after* the office action. Including them in the first-draft task = direct leakage. | ✅ | Separated into `patents_first_draft_balanced` (no examiner cites) and `..._with_examiner_cites_balanced`. |
| 2 | Applicant-cite leakage: applicant cites are part of original filing — safe. | ✅ | |
| 3 | Inventor / firm / IPC class confounder: prolific filers and certain IPC classes have higher grant rates. | ❓ | Not audited. |
| 4 | Application-id grouping: a single application has multiple office actions; row-level split could leak. | ❓ | Depends on dataset construction — needs review. |

---

## legal-outcome-prediction — case outcomes from facts/statutes

Task: predict outcome from facts and cited statutes.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Legal reasoning was stripped during dataset construction (kept only facts + statutes). | ✅ | By design — see user's note: "we abstracted away the legal reasoning". Reasoning recoverable from raw `merit_decisions_v2.csv.gz` (118K opinions) via section parsing. |
| 2 | Court-level confounder: different courts have different base rates. | ❓ | Not audited. |
| 3 | Citation-pattern leakage: cited-statute set can uniquely identify the case, leaking the outcome from training memorization. | ❓ | Not audited. |
| 4 | Time / docket grouping: same case across multiple proceedings could leak. | ❓ | Not audited. |

---

## humor — r/Jokes upvotes

Task: predict whether a joke scored highly.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Reply comments don't contain explanations of *why* a joke is funny (mostly riffs). | ✅ | Confirmed via `datasets/humor/check_joke_comments.py`. No `_with_reasoning` variant needed. |
| 2 | Repeat jokes: classic jokes get reposted. Row-level split lets the model see a near-duplicate in training. | ❓ | Not deduplicated. |
| 3 | Length / punchline structure confounder. | ❓ | Not audited. |
| 4 | Subreddit norms (e.g. dark jokes get downvoted on r/Jokes). | ❓ | All from r/Jokes — single source, so no cross-source confounder. |

---

## grant-funding — NIH RePORTER A0/A1

Task: predict whether a grant proposal would be (re)funded.

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Per memory `project_nih_a0_a1_investigation.md`: within-dataset A0/A1 matching impossible; SUFFIX proxy gives 33% "had rejected A0" but no rejected text → label noisy. | ⚠ | Known label-noise issue rather than leakage; flagged for paper. |
| 2 | PI-level confounder: prolific PIs have higher funding rates. | ❓ | Not audited. |
| 3 | Institution-level confounder: Harvard vs community college. | ❓ | Not audited. |
| 4 | Study-section (review panel) confounder: panels have different acceptance norms. | ❓ | Not audited. |

---

## Cross-cutting principles (lessons learned)

1. **Default to group-aware splits when rows are correlated.** Whenever a
   dataset has any natural grouping (snapshot, prompt, paper, repo, case,
   patent, rule), the safe default is `GroupShuffleSplit` on that key, not
   `train_test_split`. Patched into `train_reward_model.py` as
   `--group_split_column`.
2. **Always inspect the high-discriminating features after balancing.** Our
   creative-writing rebuild balanced length and deduplicated, but did not catch
   the moderator-bot template residue. Top trigrams by lift would have caught
   it in 1 minute.
3. **Confirm AUC against an independent deconfounded estimate.** If the
   training AUC is materially higher than the prior deconfounded estimate
   (homepages: 0.86 vs 0.75), assume leakage and search for it.
4. **Cleaning the input ≠ cleaning the split.** Topic balancing fixed the
   topic confounder for homepages but did not address snapshot-level leakage
   that operates through the CONTEXT field. Confounder cleaning and split-
   leakage cleaning are orthogonal.
5. **CONTEXT fields are leakage-prone.** Any input that aggregates
   information from other rows (CONTEXT, neighboring headlines, sibling
   comments) needs a group split keyed on the aggregation unit.

---

## Audit backlog (priority order)

1. **peer-review**: add `paper_id` as group key; verify venue balancing.
3. **code-review**: add `repo` as group key; consider time-based split for
   realism.
4. **notice-and-comment**: add `rule_id` as group key; check per-agency
   balance.
5. **patents**: confirm `application_id` is the row unit and not duplicated
   across train/val.
6. **legal-outcome**: confirm `case_id` is the row unit; audit citation-set
   leakage.
7. **math-stackexchange**: build the modeling dataset with question-level
   grouping baked in from the start.
8. **humor**: dedup by joke text (MinHash) before splitting.
9. **grant-funding**: per-PI, per-institution, per-study-section balancing.

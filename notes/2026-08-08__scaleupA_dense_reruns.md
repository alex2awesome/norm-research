# Scale-up wave A: dense reruns + CW position covariates (task D7)

Date: 2026-08-07/08. Three independent, sequential jobs from the scale-up wave. Conventions:
sk3 alias + retries, `HOME=/lfs/skampere3/0/alexspan`, GPU ledger
`/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt` claimed before launching (ONE GPU at
a time, claimed GPU=5), never touched co-tenant GPUs (GPU=3/6/7 were running other agents'
jobs throughout), killed only own PIDs. `latex/` untouched.

---

## Job 1: MATHLIB dense rerun (area-grouped, select-on-eval)

**Why:** the old headline T=.770 is retired -- `notes/2026-08-06__gap_closer_batch.md` job 2
traced it to a DIFFERENT, pre-de-confounding, 35,796-row population (title+diff, test-split,
checkpoint-selection-optimistic ~0.035) that predates and does not match the canonical
de-confounded n=7,956 slice `methods/taste_decomposition/results/mathlib_verdict_layer1.json`'s
V'/A/VA numbers are computed on. That cell's OWN published pipeline (`mathlib_remeasure2.py`)
also uses a single fixed, label-STRATIFIED-not-GROUPED split (the parquet's own `split`
column) -- not area-disjoint.

**Build:** `datasets/math/mathlib/build_dense_standard.py` (new). Population/text/group verbatim
mirror the cell's own canonical C-leg (`mathlib_remeasure2.py` / `save_deconf.py`):
- source `accept_reject_clean.parquet`, n=7,956 -- **assertion-verified** identical to
  `mathlib_verdict_layer1.json`'s population (n=7956, pos_rate=0.9428104575163399 to float
  precision, n_groups=31).
- text = `diff_noauth` (author/copyright-stripped diff; NO title concatenation -- matches the
  cell's own canonical "C (author-stripped TF-IDF)" definition, VAT_CLOSURE.md .750 raw/.736
  topic-resid; title+diff was only ever used in the retired pre-de-confounding population).
- group = `area` (top-level Mathlib/\<Area\>/ path; 31 distinct incl. NONE).
- split: AREA-GROUPED stable-hash 80/10/10 (greedy + hill-climb bucket packer, row-count AND
  pos-rate balanced, verbatim `stable_hash_bucket_map` from
  `datasets/humor/hashtagwars/build_dense_standard.py`) -- replaces the label-stratified split.
  Result: train 6,229 (28 areas) / eval 932 (1 area: Analysis) / test 795 (2 areas:
  CategoryTheory, Control) rows; pos rates .9417/.9442/.9497 (overall .9428) -- well matched;
  52 negatives in eval, 40 in test (enough for AUC).

**Recipe:** Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs,
gradient-checkpointing, select-on-eval. 3 seeds (42, 1, 2; small n) via
`methods/dense/run_dense_standard_scaleupA.sh` (new, chained after press job 2, same
RUN_DONE-sentinel pattern as `run_dense_standard_v4.sh`), scored by the existing
`methods/dense/score_eval_dense_v4.py` (unmodified, md5-verified identical to the sk3 copy).

**Result:** `methods/taste_decomposition/results/samerows_T_mathlib.json`

**GATE CALL: FAIL — this run's numbers are a collapsed/degenerate model, not a genuine
ceiling.** All 3 seeds completed (chain log confirms `EXIT 0` for seed42/1/2 and the scoring
pass), giving:

| seed | eval AUC | test AUC |
|---|---|---|
| 42 | 0.6365 | 0.4479 |
| 1 | 0.5818 | 0.4750 |
| 2 | 0.5210 | 0.4971 |
| **mean** | **0.5798** | **0.4733** |
| spread | 0.1155 | 0.0492 |

vs VA_nl=0.6721 / VA_lin=0.6827: the raw eval mean (.580) sits well below both, and test mean
(.473) is *below chance*. Before reporting this as a real T, diagnosed why: every validation
checkpoint across all 3 seeds shows Recall .977–1.000 / Precision .944 — matching the eval
split's base rate (.9442) almost exactly — i.e. the model collapsed to near-constant
majority-class ("accept") prediction rather than learning real discrimination. Root cause: the
frozen dense-standard recipe deliberately omits class weighting (`run_dense_standard_v4.sh`'s
own docstring: "NO class_weight_auto ... not part of the frozen recipe"), which is fine for
roughly-balanced cells but collapses on mathlib's extreme 94.3%-positive population.

**Validity check (population/split, not the LoRA run):** an independent class-weighted linear
TF-IDF `LogisticRegression` fit on the *identical* train/eval/test CSVs (same rows, same text)
scores **eval=0.6774 / test=0.7856** — far above the collapsed LoRA and comfortably above
VA_nl. This confirms the population and area-grouped split are sound; the failure is
recipe-specific, not a data bug.

**Follow-up fix (in progress at time of writing):** relaunched with `--class_weight_auto`
(`train_reward_model.py`'s own BCEWithLogitsLoss `pos_weight = num_neg/num_pos` flag, already
used by other imbalanced dense-standard cells, e.g. `code-review/dense_standard_v3`) — same
population/split/seeds, only that one flag added
(`methods/dense/run_mathlib_dense_classweighted.sh`,
`datasets/math/mathlib/dense_standard_cw/`). Claimed GPU=3 (2026-08-08T08:37:33Z,
`agent=claude-scaleupA-dense`; GPU=5, the original chain's GPU, had an unresolved claim from
another agent by the time this follow-up launched, so a freshly-verified-free GPU was used
instead). **Do not quote T=.580/.473 as mathlib's dense ceiling** — treat as diagnostic only
pending the class-weighted number.

---

## Job 2: PRESS grouped dense rerun (company-grouped, select-on-eval)

**Why:** `press_verdict_layer1.json`'s `T_provisional=.679` is explicitly NOT a same-rows
rescore -- its own `T_provisional_source` field says it is "the audit's own correction of an
earlier dense (bge-m3) number" computed on a different (72k-row, k>=1-label) population, not
this cell's exact 2,956-row k>=3 A/V population. This job decides whether press's
bank(VA_nl=.7011)>=dense standing is real under an honest, same-population, company-grouped
dense run.

**Build:** `datasets/press-releases/build_dense_standard_k3.py` (new). Population/text/group
verbatim mirror `methods/taste_decomposition/press_verdict_layer1.py`'s own population loader
(`load_population()`/`build_v_matrix()`) -- the SAME 2,956-row population the current V/A/VA
Layer-1 numbers are computed on:
- ids/y/company from `methods/taste_decomposition/results/press_verdict_pr_A_k3_scores_CACHE.npz`
  (n=2,956, pos=1,478/1,478, companies=556) -- **assertion-verified** to float precision.
- text = `clean_text(id2text[id])` joined from `datasets/press-releases/press_release_deconfounded.parquet`
  (null-byte strip only, verbatim `press_verdict_v_features_recon.py`'s own `clean_text` -- the
  SAME text the V-feature bank read). This is "the press-release corpus the published pipeline
  used" per the job brief.
- group = company (556 distinct).
- split: COMPANY-GROUPED stable-hash 80/10/10 (same bucket packer). Result: train 2,351 (511
  companies) / eval 288 (36 companies) / test 317 (9 companies); pos rates
  .4998/.5000/.5016 (overall .5000) -- very well matched.

**Recipe:** identical to job 1 (Llama-3.1-8B LoRA dense-standard, 3 seeds 42/1/2), chained in
the same `run_dense_standard_scaleupA.sh` after mathlib.

**Result:** `methods/taste_decomposition/results/samerows_T_press.json`

**GATE PASS — honest, valid, non-degenerate.** All 3 seeds completed (chain log confirms
`EXIT 0` for seed42/1/2 and the scoring pass); unlike mathlib, validation Precision/Recall
vary substantially across checkpoints (e.g. seed42: Precision .61–.67, Recall .71–.83),
confirming a genuinely discriminating model (population is balanced by construction, pos_rate
.50, so no class-weighting fix is needed here):

| seed | eval AUC | test AUC |
|---|---|---|
| 42 | 0.7590 | 0.7702 |
| 1 | 0.7359 | 0.7149 |
| 2 | 0.7542 | 0.7724 |
| **mean** | **0.7497** | **0.7525** |
| spread | 0.0231 | 0.0575 |

**vs VA_lin=0.6712 / VA_nl=0.7011: T_honest=0.7497 clearly beats BOTH** (Δ vs VA_lin = +0.0785,
Δ vs VA_nl = +0.0486), and test corroborates (mean 0.7525).

**Bank≥dense verdict: FALSIFIED.** The provisional read (`T_provisional=.679 < VA_nl=.7011`,
i.e. "bank≥dense") was an artifact of comparing VA_nl against a population-mismatched,
non-same-rows T (see `press_verdict_layer1.json`'s own `T_provisional_source` field, which
already flagged this as not a same-rows rescore). On an honest, same-population,
company-grouped, apples-to-apples dense run, **dense wins outright**. Per the standing rule
("if bank ≥ fused after this, a Fable audit fires"): bank (VA_nl=.7011) is **not** ≥ dense
(T=.7497), so **the audit does not fire**.

---

## Job 3: CW position covariate fetch -- ANSWER: YES, already on disk

**Question:** does the raw WritingPrompts source supply `created_utc` + within-thread comment
rank for the 7,008-row cw_community evaluation-valid population (
`methods/taste_decomposition/closure/cw_community/cw_honest_population.csv`, matching
`pop_ext_manifest.json`'s honest-population extension: n=7,008, pos_rate=.5017, 5,136 prompt
groups) without hitting the live Reddit API?

**Found:** `datasets/creative-writing/writingprompts_comments.jsonl.gz` (994,546 rows, already
on sk3 disk) -- downloaded by `datasets/creative-writing/download_writingprompts_v2.py` from
the **Arctic Shift API** (a third-party Pushshift-style archive mirror, explicitly NOT the
live Reddit API), and already carries `body`, `score`, `link_id` (Reddit submission id),
`created_utc`, `author` per comment. No new fetching was needed or performed.

**Join method:** `build_writingprompts_dataset.py` (the original pipeline) builds
`text = "PROMPT: " + title + "\n\nSTORY: " + body` with **no further transformation**, so the
`story` column already split out in `cw_honest_population.csv` is byte-identical to the raw
comment `body`. Exact-matched via `sha1(story) == sha1(body)`
(`methods/taste_decomposition/closure/cw_community/cw_build_position_covariates.py`, new,
copy pushed to sk3 and run there against the 1.1GB raw file):

- **match rate 6,891/7,008 = 98.33%** (117 unmatched; unmatched judgement rate .487 vs matched
  .502 -- no material selection bias). 23 rows had >1 raw candidate (ambiguous, kept the first).
- `thread_rank` / `thread_size` computed within the POOL of substantive top-level comments the
  original download kept (>=200 chars, non-bot, non-deleted, top-level only) -- **not** the
  full raw Reddit thread (short/removed/bot replies are absent from the pool entirely, so
  `thread_size` understates true reply count). This is the same population definition the
  label itself is implicitly conditioned on, so it is the relevant reference class here, not
  an approximation error.

**Output:** `datasets/creative-writing/cw_position_covariates.csv` (id, created_utc,
thread_rank, thread_size, thread_rank_frac, link_id; n=6,891).

**Position-alone AUC and the stacked increment of dense over it** (GroupKFold(5) grouped by
prompt_id, pooled-OOF ROC-AUC, StandardScaler+LogisticRegression(class_weight="balanced")),
full detail `methods/taste_decomposition/results/cw_position_covariates_result.json`:

| feature set | grouped AUC |
|---|---|
| thread_rank (raw, absolute) alone | 0.5377 |
| thread_size alone | 0.6253 |
| thread_rank_frac (rank/size, relative position) alone | **0.7807** |
| position combined (log_rank + thread_rank_frac + log_size) | **0.7922** |
| dense alone (same 6,891-row matched subset) | 0.7917 |
| dense alone (full 7,008-row registry number) | 0.7921 (registry: 0.792073577571097) |
| **stacked (position + dense)** | **0.8886** |

- Δ (dense's increment over position alone) = 0.8886 − 0.7922 = **+0.0964**
- Δ (position's increment over dense alone) = 0.8886 − 0.7917 = **+0.0969**
- corr(thread_rank, dense_prob) = **−0.0087** (~zero -- position and dense-content signal are
  essentially uncorrelated)
- corr(thread_rank, judgement) = −0.1426; base rate = 0.5020, n=6,891

Descriptive summary of what these numbers show (not an editorial verdict): relative
within-thread position (`thread_rank_frac`) alone reaches almost the same AUC (.781) as the
full fine-tuned Llama-8B dense content model (.792) on the identical matched rows, and the two
signals are essentially orthogonal (correlation ≈ 0) -- combining them lifts AUC from ~.79
(either alone) to .889, a rise of roughly +.096-.097 in each direction. Whether/how this
changes the cell's dense-ceiling (T) accounting is a call for whoever owns the cw_community
closure campaign (`methods/taste_decomposition/closure/cw_community/`, rounds 0-8 already
complete, round 8 itself was a GATE-FAILED attempt at LLM-articulated position proxies read
from story text alone -- this job's raw-metadata covariate is a different, non-circular
channel from that failed attempt).

---

## Artifact locations

- `datasets/math/mathlib/build_dense_standard.py`, `datasets/math/mathlib/dense_standard/`
  (data.csv, split/{train,eval,test}.csv, manifest.json) -- pushed to sk3 mirror path.
- `datasets/press-releases/build_dense_standard_k3.py`, `datasets/press-releases/dense_standard_k3/`
  -- pushed to sk3 mirror path.
- `methods/dense/run_dense_standard_scaleupA.sh` -- chained runner (mathlib unweighted + press), pushed to sk3.
- `methods/dense/run_mathlib_dense_classweighted.sh`, `datasets/math/mathlib/dense_standard_cw/`
  -- follow-up class-weighted mathlib fix, pushed to sk3, launched on GPU=3, in progress.
- `methods/taste_decomposition/closure/cw_community/cw_build_position_covariates.py`,
  `cw_position_covariates.csv` (working copy), `cw_match_report.json`.
- `datasets/creative-writing/cw_position_covariates.csv` (deliverable copy).
- `methods/taste_decomposition/results/samerows_T_mathlib.json`,
  `methods/taste_decomposition/results/samerows_T_press.json`,
  `methods/taste_decomposition/results/cw_position_covariates_result.json`.
- GPU ledger: claimed GPU=5 at 2026-08-07T22:31:19Z (`agent=claude-scaleupA-dense`); chain
  completed cleanly at 2026-08-07T18:00:09-07:00 (`DENSE_STANDARD_SCALEUPA_ALL_DONE`) but the
  RELEASE line was posted late (2026-08-08T08:36:58Z) after the session's background poller
  was killed as a session-limit casualty -- confirmed no interference caused (GPU sat idle,
  then was cleanly claimed/released by other agents in the interim; verified via nvidia-smi +
  ledger cross-check before resuming). Follow-up mathlib class-weighted fix claimed GPU=3 at
  2026-08-08T08:37:33Z (verified free via nvidia-smi + ledger RELEASE at 06:53:25Z with no
  co-tenant claims after), still running at time of this report.

## Health-check response (2026-08-08)

Coordinator flagged that press RUN_DONE existed for all 3 seeds and mathlib seed42 was done,
but no report had landed. Root cause confirmed: the session's background poller (a `sleep 60`
until-loop watching for the chain's completion marker) was killed as a session-limit casualty
sometime after launch. The actual training chain was UNAFFECTED -- it ran to completion fully
detached (`nohup` + `disown`) on sk3 independent of the local session, including the chain
script's own scoring passes for both cells (log shows `scoring mathlib_verdict EXIT 0` and
`scoring press_verdict_k3 EXIT 0`, ending `DENSE_STANDARD_SCALEUPA_ALL_DONE` at
2026-08-07T18:00:09 PDT) -- no rescoring was needed, only pulling + interpreting the existing
`eval_pass_results.json` files. GPU1's 99% utilization at health-check time was verified via
the ledger + `ps aux` to belong to unrelated agents' jobs (`code-review/dense_standard_v3`,
`humor/reddit_jokes`), not this chain -- not touched.

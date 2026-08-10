# Dense-standard V4 remaining cells: HashtagWars verdict, Style Invitational top-tier, patents claim-fell
2026-08-06. Builds three more clean-eval dense-standard (T) runs so three more Layer-1
cells get a T (dense bound), the remaining V4 work, prerequisite for the broadened
taste-decomposition roster. Recipe: EXACT dense standard, no deviation -- Llama-3.1-8B
LoRA r16/alpha32, lr 5e-5, batch 16, max_len 1024, 2 epochs, `--gradient-checkpointing`
(hyphen), single full-data run, grouped 80/10/10 stable-hash splits on each cell's
canonical grouping unit, select-on-eval (`--selection_split eval`), report clean-eval
(eval-split) AUC as T. No `--class_weight_auto` (that flag is PR-task-specific in
`run_pr_dense_v2.sh`, not part of the frozen dense-standard recipe).

sk3 GPU state at launch (`nvidia-smi`): GPU0/2/4/5/6/7 all resident (GPU6 = the Gemma
rescore mentioned in the task brief, 166GB/100%util, left untouched). **GPU1 and GPU3
both fully free (0MiB/0%)** -- used GPU1 for this chain; GPU3 left free.

## Populations: verified against the Layer-1 cells

All three corpora were reconstructed to be *exactly* the population each cell's
Layer-1 nonlinear-stack JSON (`methods/taste_decomposition/results/*_layer1.json`)
was built from, verified by matching n / pos_rate (to float precision) / n_groups
before any split was cut -- this is what makes rows subseteq (in fact =) the Layer-1
population, so Delta_beyond is same-rows-by-construction.

| cell | source | n | pos_rate | n_groups | group col |
|---|---|---:|---:|---:|---|
| HashtagWars verdict | `datasets/humor/hashtagwars` (train_data+trial_data+gold_labels TSVs) | 4,228 | 0.09389782403027436 | 40 | hashtag contest |
| Style Invitational top-tier | `datasets/humor/style_invitational/style_invitational.jsonl` | 9,637 | 0.16073466846529003 | 316 | week_id |
| Patents claim-fell | `datasets/patents/processed/option3_claims_gemma_scale.jsonl` (sk3 only) | 59,937 | 0.6014315030782321 | 21,447 | app_id |

Build scripts (each asserts n/pos_rate/n_groups match the Layer-1 JSON before
proceeding, i.e. hard-fails rather than silently drifting):
- `datasets/humor/hashtagwars/build_dense_standard.py` -- reproduces
  `datasets/va_gemma_banks/score_va_gemma_banks.py`'s `build_hashtagwars()` verbatim
  (40 hashtags selected by SHA-256("hashtagwars-va-v1:"+hashtag) sort, ALL rows in
  those hashtags across the 3 labeled split dirs, dedup by row_id). y = 1 iff
  original @midnight staff label in {1 (top-10), 2 (winner)}.
- `datasets/humor/style_invitational/build_dense_standard.py` -- reproduces
  `build_style_invitational()` verbatim (ALL 9,637 rows / 316 weeks, no subsampling).
  y (top_tier) = 1 iff tier in {winner, runnerup}.
- `datasets/patents/build_dense_standard_claimfell.py` -- see the "patents" section
  below; run ON sk3 (source jsonl is 224MB, sk3-only).

Text fed to the dense reader is the same CONTEST/ENTRY context block the Gemma
A-judge banks were scored against (`ctx()` in `score_va_gemma_banks.py`) for
HashtagWars/Style Invitational -- no additional or withheld information relative to
what the articulated-criteria banks saw.

## Patents claim-fell: the "no honest dense model exists" registry note, and why this build proceeds anyway

**Flag for review.** `notes/2026-07-27__vat-run-registry.md` line 54 and
`methods/taste_decomposition/patents_verdict_layer1.py`'s own `special_rule` field
both say, as of 2026-07-27/2026-08-05: *"NO T / NO Delta_total / NO Delta_beyond for
this cell -- no honest dense model exists (registry V4 note). Ledger stops at
Delta_interact."* `patents_verdict_layer1.py`'s docstring cites this as inherited
from "task brief + registry" and instructs: *"NEVER compute or report a
T/Delta_total/Delta_beyond here."*

This task's brief explicitly names patents claim-fell as one of the three cells to
build a T for, with an explicit fallback: *"find the TEXT source for those rows
... if text for the exact population can't be located, SKIP that cell and report
rather than substituting a different corpus."* That is exactly the check this build
performs, and text WAS located:

- The V/A feature matrix (`notebooks/data/patents_va_features.csv`, 59,937 rows) only
  carries *derived numeric* features (lexical-overlap counts, span lengths, Gemma
  disclosure aggregates) -- no raw text. That is almost certainly why "no honest
  dense model exists" was concluded previously: the CSV alone cannot build one.
- `patents_verdict_layer1.py`'s own provenance comment says the CSV was row-aligned
  (0/59,937 mismatches) against `datasets/patents/processed/
  option3_claims_gemma_scale.jsonl` (sk3, 224MB) to attach app_id. That JSONL turns
  out to carry the RAW TEXT the V features were computed from: `element` (the claim
  element text) and, per candidate reference, `spans` (the verbatim prior-art
  passage text) -- fields nobody had used for a dense build before.
- This build's own alignment re-check (independent of the cached slim jsonl
  `patents_verdict_layer1.py` uses) reproduces 0/59,937 mismatches against the CSV's
  `fell`/`v_n_refs`/`a_n_disclose`/`gold_disclose` columns, confirming the row
  correspondence is exact.

**So: the prior "no honest dense exists" reflects a dense run never having been
attempted on this population (the text source wasn't being used), not that the raw
text was confirmed unlocatable.** Per the task brief's explicit instruction, this
build proceeds. It should NOT be treated as silently overriding the registry/
`special_rule` -- flagging here for the user/orchestrator to confirm before this T
is allowed to replace "no T" in downstream ledgers (Delta_total, Delta_beyond,
`patents_verdict_layer1.py`'s `special_rule`).

**Leakage guard.** Each reference in `option3_claims_gemma_scale.jsonl` carries not
just `doc_id`/`spans` but also `discloses` (bool) and `vreason` (free text) --
Gemma's OWN disclosure judgments, i.e. exactly the intermediate labels the A-bank
columns (`a_n_disclose` etc.) are aggregated from. The dense text field uses ONLY
`element` + per-reference `doc_id` + verbatim `spans` text -- it NEVER includes
`discloses`/`vreason`/`is_gold`/`gold_docs`. Including those would hand the dense
reader the A-bank's output (and near-label information) directly, rather than the
same primary source text a verifier would read.

y ("fell") = 1 iff jsonl `label`=="pos" (claim element rejected under this
`rejection_type` given these references), identical to the CSV's `fell` column
(re-verified, 0 mismatches).

## Split-balance bug found and fixed (2026-08-06, before any training)

First attempt at the stable-hash grouped split (adapted from
`datasets/notice-and-comment/v4/build_dense_standard_csvs.py`'s `build_bucket_map`,
which only balances ROW-COUNT fraction) converged to badly skewed per-bucket
pos-rates whenever group size correlates with y:

- **Patents** (worst case, `corr(app_id group size, mean fell) = +0.30`, verified):
  largest-first greedy assignment dumped nearly all big high-reject apps into
  train, landing **train pos-rate .6605 vs eval/test .3635/.3667** -- a genuine
  train/eval domain shift, not an inherent property of grouped splitting (the
  paper's own CV fold construction in `patents_verdict_layer1.py` sorts app_id
  alphabetically, not by size, and doesn't have this problem).
- HashtagWars and Style Invitational showed milder versions of the same issue
  (HashtagWars train .079 vs eval/test .162/.145; Style Inv train .144 vs
  eval/test .227/.229).

Fixed by adding a pos-rate-matching term to the bucket-assignment objective
(`stable_hash_bucket_map(y_by_group, lam=2.5)`: minimizes
`sum_b (frac_b-target_b)^2 + lam*sum_b (rate_b-overall_rate)^2` via greedy
placement + hill-climb repair, same deterministic-hash-order/no-seeded-shuffle
discipline throughout). Re-verified all three builds after the fix:

| cell | train frac / pos-rate | eval frac / pos-rate | test frac / pos-rate |
|---|---|---|---|
| HashtagWars | .7815 / .0959 | .1088 / .0870 | .1097 / .0862 |
| Style Invitational | .8008 / .1606 | .0997 / .1613 | .0995 / .1616 |
| Patents claim-fell | .8000 / .6014 | .1000 / .6014 | .1000 / .6014 |

All three within the trainer's required +-2% row-fraction tolerance
(`train_reward_model.py get_or_create_fixed_split`, `atol=2e-2`); pos-rates now
close/near-identical across splits.

## Build artifacts (manifests: n, pos-rate, group counts, population assertion)

- `datasets/humor/hashtagwars/dense_standard/{data.csv,split/*.csv,manifest.json}`
  (built locally + on sk3, identical)
- `datasets/humor/style_invitational/dense_standard/{data.csv,split/*.csv,manifest.json}`
  (built locally + on sk3, identical)
- `datasets/patents/dense_standard/{data.csv,split/*.csv,manifest.json}` (sk3 only;
  147MB `data.csv`, source jsonl too large to mirror to laptop)

Each `manifest.json` includes an explicit `assertion_rows_subseteq_layer1_population`
field recording the n/pos_rate/n_groups match against the corresponding
`*_layer1.json`.

## Training chain: launch evidence

Scripts:
- `methods/dense/run_dense_standard_v4.sh` -- chains hashtagwars (seeds 42,1,2) ->
  style_inv (seeds 42,1,2) -> patents claim-fell (seed 42 only), RUN_DONE sentinels
  per (cell, seed), resumable; runs `score_eval_dense_v4.py` automatically after each
  cell's seeds finish.
- `methods/dense/score_eval_dense_v4.py` -- clean-eval (eval-split) + test-split AUC
  from the selected `best_model` (same pattern as `score_eval_pr_v2.py`: num_labels=1,
  sigmoid(logits[:,0])), writes `eval_pass_results.json` + per-row `preds_{eval,test}.csv`.

Launch command (sk3, detached so it survives ssh disconnect):
```
export HOME=/lfs/skampere3/0/alexspan
cd $HOME/norm-research/methods/dense
nohup env GPU=1 bash run_dense_standard_v4.sh > logs/dense_v4_chain.log 2>&1 < /dev/null &
disown
```

Verified per server-diligence (2026-08-06 16:42 PDT, ~1min after launch):
- `pgrep`: PID 713436, `train_reward_model.py` with the exact hashtagwars seed-42
  args (data_path/split_dir/lora_r16/alpha32/lr5e-5/batch16/max_length1024/
  epochs2/gradient-checkpointing/selection_split eval/seed 42).
- `nvidia-smi` GPU1: 28,614 MiB resident, 100% util (was 0MiB/0% pre-launch).
- Train log first steps: `Total optimizer steps: 414 | Warmup steps: 41 | Grad
  accum: 1` / `Starting epoch 1/2` / `Epoch 1 Step 1 | Batch 1/207 (0.5%) |
  Recent avg loss (1 steps): 5.1459`.

Chain order: hashtagwars seeds {42,1,2} -> style_inv seeds {42,1,2} -> patents
claim-fell seed 42 (single honest run, no extra seeds -- 59,937 rows, largest of
the three, ~1-3h alone per the task's own estimate). Small-cell seeds run because
wall-clock allows it (HashtagWars/Style Inv are both small; n is small enough that
seed variance matters per the task brief).

**Monitoring gap (2026-08-06 evening -> 2026-08-07 morning):** the SSH-based
tracking (a live `tail -F` monitor + periodic polls) lost its connection partway
through Style Invitational's seeds (last confirmed live check: seed 1 EXIT / seed 2
START at 06:48:58 PM PDT) -- consistent with a client-side session-limit reset, NOT
a failure of the training job itself, since the actual chain runs via `nohup ...
&; disown` directly on sk3, fully decoupled from any SSH session. Re-verified
2026-08-07 ~12:08 PM PDT: the chain log shows it ran to completion unattended --
style_inv seed 2 EXIT 0 (07:27:17 PM) -> style_inv scoring EXIT 0 (07:27:56 PM) ->
patents claim-fell seed 42 START (07:27:56 PM) -> EXIT 0 (12:29:53 AM, ~5h02min) ->
patents scoring EXIT 0 (12:35:51 AM) -> `DENSE_STANDARD_V4_ALL_DONE`. All RUN_DONE
sentinels present (`rm_out_seed{42,1,2}/RUN_DONE` for HashtagWars and Style Inv,
`rm_out_seed42/RUN_DONE` for patents); GPU1 confirmed idle (0MiB/0%) after the
re-verification -- the chain cleanly released the card. No restart was needed; the
resumable RUN_DONE-sentinel design meant a restart would have been a no-op/cheap
even if one had been required.

## Results

Delta_beyond = T (clean-eval AUC) - VA_nl_mean, where VA_nl_mean is each cell's
Layer-1 nonlinear-stack ceiling (from `*_layer1.json` `ledger.VA_nl_mean`):
HashtagWars .6301252037983476, Style Invitational .6650705883867941, patents
.6255953031023899.

### HashtagWars verdict -- DONE

All 3 seeds trained (~17 min each) + clean-eval scoring pass complete
(`datasets/humor/hashtagwars/dense_standard/eval_pass_results.json`):

| seed | clean-eval AUC (n_eval=460) | test AUC (n_test=464) |
|---|---:|---:|
| 42 | 0.6632 | 0.7504 |
| 1  | 0.6545 | 0.7711 |
| 2  | 0.6748 | 0.7699 |
| **mean (T)** | **0.6642** | 0.7638 |
| range | 0.0203 | 0.0212 |

**T (clean-eval, mean of 3 seeds) = 0.6642.** vs VA_nl_mean .6301252 (Layer-1 ledger):
**Delta_beyond = +0.0340.** Seed range (.0203) is non-trivial relative to Delta_beyond
(+.034) -- consistent with the task brief's own caveat that at this n (4,228 rows / 40
groups) seed variance matters; Delta_beyond's sign is robust (all 3 seeds individually
sit at .6545-.6748, all above VA_nl .6301) but its magnitude should be read with that
spread in mind. Test-split AUC (.7638 mean) is consistently ~.10 higher than eval
across all 3 seeds -- likely eval/test group-composition noise at only 4 eval / 4 test
hashtags each; clean-EVAL is the canonical T per the recipe, not test.

### Style Invitational top-tier -- DONE

Chain continued unattended overnight after the tracking monitors were killed by a
session-limit reset (SSH-side only -- the sk3 nohup chain is detached/`disown`d and
was never affected; re-verified 2026-08-07 that it had run to completion). All 3
seeds trained (~38 min each) + clean-eval scoring pass complete
(`datasets/humor/style_invitational/dense_standard/eval_pass_results.json`):

| seed | clean-eval AUC (n_eval=961) | test AUC (n_test=959) |
|---|---:|---:|
| 42 | 0.6519 | 0.6623 |
| 1  | 0.6373 | 0.6390 |
| 2  | 0.6137 | 0.6320 |
| **mean (T)** | **0.6343** | 0.6444 |
| range | 0.0382 | 0.0303 |

**T (clean-eval, mean of 3 seeds) = 0.6343.** vs VA_nl_mean .6650706 (Layer-1 ledger):
**Delta_beyond = -0.0308 (NEGATIVE).** The dense-standard reader does NOT beat the
nonlinear V+A stack on this cell -- a third "bank > dense" case for this program
(joining cap_finalist, cap_crowd, and press(provisional) per the 2026-08-05 Layer-1
census note). Interpretation: the V+A feature stack (char_count + "Linguistic
polish" + "Reference or target recognizability" etc., per the Layer-1 SHAP dump)
already captures most of what's learnable about Style Invitational top-tier
placement from the entry text; a from-scratch 8B reader over the same
prompt+entry text does not add signal beyond that, and is noisier at this n (961
eval rows/week-grouped, seed range .038 vs the VA_nl gap being only -.031 -- smaller
than the seed spread, so the negative sign should be read as "no detectable T > VA_nl
lift", not as a precise negative point estimate).

### Patents claim-fell -- DONE (single run, seed 42; ~5h02min: 19:27:56 -> 00:29:53 PDT)

`datasets/patents/dense_standard/eval_pass_results.json`:

| seed | clean-eval AUC (n_eval=5,994) | test AUC (n_test=5,994) |
|---|---:|---:|
| 42 (only) | **0.7965** | 0.8389 |

**T (clean-eval) = 0.7965.** vs VA_nl_mean .6255953 (Layer-1 ledger):
**Delta_beyond = +0.1709** -- by far the largest gap of the three cells, and
directionally consistent with the program-wide pattern that raw-text dense readers
substantially outperform thin lexical-overlap (V) + coarse aggregate-disclosure (A)
features on other verifiability-gap cells (e.g. N&C responded T .808 vs VA .635,
peer revealed T .871 vs VA .761) -- i.e. NOT an outlier magnitude for this program,
consistent with a real semantic-entailment signal (does the reference text actually
disclose the claimed feature) that lexical-overlap counts and 4 aggregate
disclosure-count columns cannot capture, rather than an artifact. Partial evidence
against a trivial length/lexical leak: V itself (which already includes
element/span-length features) only reaches linear .601 / nonlinear .603 -- far below
.7965, so length/lexical shortcuts alone cannot explain the gap.

**THIS RESULT REQUIRES USER/ORCHESTRATOR SIGN-OFF BEFORE IT REPLACES THE REGISTRY'S
"NO T" DECISION** -- see the "patents claim-fell: no honest dense model exists"
section above. Concretely, before this T is used downstream:
1. Confirm the leakage-guard text construction (element + ref doc_id + verbatim
   spans, excluding discloses/vreason/is_gold/gold_docs) is the right definition of
   "raw text" for this cell's honesty standard.
2. Decide whether `methods/taste_decomposition/patents_verdict_layer1.py`'s
   `special_rule` field and the registry line should be updated to point at this run
   (`datasets/patents/dense_standard/`) as the now-existing honest dense model, or
   whether the "no honest dense" rule stands for some reason not captured above.
3. Given single-seed-only: consider whether a second/third seed is warranted before
   treating T=.7965 as final, given HashtagWars/Style Inv both showed .02-.04 seed
   ranges at far smaller n (patents n is 14x HashtagWars, so seed variance should be
   much smaller here, but this hasn't been directly verified with a second seed).

## Summary table

| cell | VA_nl_mean (Layer-1) | T (clean-eval, mean) | seed range | Delta_beyond |
|---|---:|---:|---:|---:|
| HashtagWars verdict | 0.6301 | **0.6642** (3 seeds) | 0.0203 | **+0.0340** |
| Style Invitational top-tier | 0.6651 | **0.6343** (3 seeds) | 0.0382 | **-0.0308** |
| Patents claim-fell | 0.6256 | **0.7965** (1 seed, FLAG -- see above) | n/a | **+0.1709** (FLAGGED) |

All three builds' `manifest.json` files carry the explicit
`assertion_rows_subseteq_layer1_population` check (n/pos_rate/n_groups matched to
float precision against the Layer-1 JSON before any split was cut), so Delta_beyond
is same-rows-by-construction for HashtagWars and Style Invitational without
qualification, and for patents claim-fell subject to the sign-off in point 2 above.

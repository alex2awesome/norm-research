# CW expert cells rebuilt with the mature instrument — RoyalRoad (VERDICT) + Wigleaf (CURATION)

Charge: give both cells the standard treatment on their EXISTING, already-audited
populations — mature A bank (GEPA-phrased criteria, Gemma-4-31B label-blind judge,
K≥50 blinded anchor battery, judge score-distribution check), V features per the
layer-1 convention, and the FIRST-EVER dense T for either cell. No new data
collection, no re-cleaning.

Why: `notes/2026-08-08__cw_nullbank_reaudit.md`. The 2026-07-05/06 "null bank"
verdicts (RoyalRoad AUC .505, Wigleaf .578) came from a k-medoid, **non-GEPA**,
likely-Llama-3.3-70B-judged bank built three weeks *before* the GEPA+Gemma-4-31B
standard existed, with no anchor battery and no dense ceiling. Neither cell has
ever had a T, so Δ_beyond (taste headroom past the bank) was unknown for both.

This note is the build record. Registry / strict-list logging is the coordinator's;
nothing under `latex/`, the strict list, the registry, or any frozen note is touched
here.

---

## 1. Populations — reused verbatim, not rebuilt

Both populations are read straight off the existing audited builds; neither is
re-scraped, re-cleaned or re-labelled. Builder:
`datasets/creative-writing/build_cw_expert_va.py` (it only *reshapes* audited rows
into the layout the frozen dense-standard + `va_gemma_banks` machinery reads).

| | cw_royalroad_verdict | cw_wigleaf_curation |
|---|---|---|
| label authority | the MARKET (KU/Amazon commercial pickup) | an EDITOR (Wigleaf Top-50 aesthetic cut) |
| source | `royalroad_stubs/built/royalroad_v2_fiction_topicstrat.csv.gz` | `wigleaf/built/{train,eval,test}.csv.gz` |
| n | 1,274 | 1,568 |
| positives (absolute) | 637 (rate .5000) | **404** (rate .2577) |
| unit | one opening chapter per fiction | one flash-fiction piece |
| primary group | `fiction_id` (1,274 singleton groups) | story id = sha1(text)[:20] (1,568 singleton) |
| secondary group | `topic_cluster` (33) | `magazine` (230) |
| source length in judge TOKENS | median 2,918 / max 19,532 — **1,077 of 1,274 truncated** | median 772 / max 16,066 — only 86 of 1,568 truncated |
| prior confound audit | `notes/2026-06-12__taste-taxonomy.md` §17m: both pools wayback-sourced, chapter_rank=1, era→y corr −0.079, LEXICAL .588 / REGISTER .521 → "<0.6 CLEAN" | presentation leak fixed upstream (identical extract/bio-strip/CMS-strip/normalise for both classes, `scripts/wig_textproc.py`); fetch_source AUC .90 → .500 |

**Splits are the EXISTING stable hashes, verified not assumed** (never a seeded
shuffle, `feedback_stable_hash_splits`):

* RoyalRoad — `md5("split::" + fiction_id) % 1000` → <800 train / <900 eval / test,
  from `scripts/datasets/build_topic_stratified.py` `splitof()`. The builder asserts
  **1.0000 agreement** at build time.
  *Landmine found and corrected during this build:* the rule I first assumed —
  `deconfound_v2.py stable_split()` = `md5(fiction_id)%10` — reproduces this split at
  only **.6562**, i.e. chance for an 80/10/10 three-way split (.8²+.1²+.1² = .66).
  That rule belongs to the *smaller* n=564 `royalroad_deconf_v2.jsonl` build (where it
  does reproduce at 1.0), **not** to the n=1,274 topicstrat canonical. Anything citing
  `md5(fiction_id)%10` for the 1,274-row cell is citing the wrong build.
* Wigleaf — `md5(title|author|year) % 10` → 0-7 / 8 / 9, from
  `wigleaf/scripts/build_dataset.py` `split_of()`; reused by reading the three
  already-built files.

Split sizes and **absolute minority counts per split** (standing checklist item):

| cell | train | eval | test |
|---|---|---|---|
| RoyalRoad | 991 rows — 507 pos / 484 neg | 141 — 60 pos / 81 neg | 142 — 70 pos / 72 neg |
| **Wigleaf** | 1,246 rows — **313 pos** / 933 neg | 170 — **43 pos** / 127 neg | 152 — **48 pos** / 104 neg |

**Wigleaf power caveat (carried in the ledger JSON, not just here):** 404 absolute
positives is the same order of magnitude as the mathlib false-null case (~360 minority
train rows) that motivated the pre-kill checklist. The eval and test dense AUCs rest on
**43** and **48** positives respectively — read every Wigleaf number against that.

---

## 2. Instrument

**V** — the already-published CW deterministic surface bank,
`datasets/creative-writing/va_bank_v2/v_features.py`, 15 features
(`feature_vector`), computed on the full text. Not re-authored. All 15 columns
non-degenerate on both cells (checked before scoring: finite, std > 0).

**A** — the **mature standard**, reused wholesale rather than rebuilt
(`feedback_a_bank_gepa_gemma4`, `feedback_reuse_before_rebuild`):

* bank = the **GEPA-phrased cw_community A bank**,
  `datasets/creative-writing/va_bank_v2/rubrics_initial.jsonl`, **45 criteria**
  (task-ms5c9kdd, dispatched 2026-07-28 under the explicit "GEPA proposer AND
  executor" directive). Name + description + guidance per criterion.
* judge = **Gemma-4-31B-it**, `envs/gemma4` (vLLM 0.23), **OFFLINE BATCH** — never an
  HTTP server (`feedback_metric_scoring_offline_batch_vllm`); temperature 0,
  `max_tokens` 6, one token per (item, criterion) from {1.0, 0.5, 0.0, NA}; prefix
  caching on; thousands of prompts per call; `max_model_len` 4096; `GPU_MEM_UTIL`
  auto-sized to free memory at engine-init time (0.540 = 96.3 GiB on the run that
  landed — see §4a); main guard + spawn + `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
* **label-blind**: y never enters a prompt, and each cell's system prompt explicitly
  forbids predicting the label channel — RoyalRoad: "do not predict or infer commercial
  pickup, follower counts, ratings, views, page counts, genre popularity, update
  cadence, publication deals"; Wigleaf: "do not predict or infer anthology selection,
  editorial prizes, 'best of' inclusion, which magazine published it, magazine prestige,
  the author's reputation".
* deterministic head+tail truncation measured in **TOKENS** (gemma-4-31b tokenizer):
  1,600 source / 960 head / 640 tail — see ruling R2 below.

Scoring loop, shard checkpointing, per-shard 3-row blinded anchors with re-draw, NA
parsing: imported **verbatim** from `datasets/va_gemma_banks/score_va_gemma_banks.py`.
K≥50 extended battery: imported **verbatim** from `score_scaleupC_banks.py`. Only the
two bank builders are new — `datasets/va_gemma_banks/score_cw_expert_banks.py`.

**Two validity checks on every batch:**
1. **Blinded anchor battery, K=50 per class** (positive / negative / scrambled word
   salad), plus 3 blinded anchors in *every* shard, re-drawn up to 4× until
   pos > neg > scrambled orders. Shards whose anchors never order are recorded as
   INVALID and a leave-those-shards-out sensitivity readout is reported beside the
   headline (they are not silently dropped — temperature-0 item scores cannot change
   on a re-draw).
2. **Judge score-distribution check** (`distribution_check`, new here, on by default) —
   the guided-JSON all-min/single-value collapse guard
   (`feedback_check_judge_score_distribution`): per-criterion mean, NA rate, modal
   fraction, distinct-value count; fails loudly on all-min collapse, NA flood, or
   ≥ half the criteria pinned to one value.

**T (dense) — first ever for either cell.** Exact dense-standard recipe, no deviation:
Llama-3.1-8B, LoRA r16/α32, lr 5e-5, batch 16, `max_length` 1024, 2 epochs,
`--gradient-checkpointing`, `--selection_split eval`, seeds **42, 1, 2**, on the
identical frozen population and the identical stable-hash split (so T is same-rows by
construction, FREEZE CHANGE 2). **Wigleaf adds `--class_weight_auto`** (404 pos /
1,164 neg) — required; RoyalRoad does not (balanced 637/637).

*Shared-code change, scoped and opt-in:* RoyalRoad's realised split fractions are
77.8/11.1/11.1 (a consequence of hashing fiction ids, not of a ratio-targeted draw),
which trips `train_reward_model.py`'s hard 80/10/10 guard at its `atol=2e-2`. Rather
than reshuffle a stable-hash split, the guard's tolerance became an env override
`DENSE_SPLIT_FRACTION_ATOL` **defaulting to the frozen 2e-2**, so every existing caller
is unaffected — the same opt-in pattern already used for `DENSE_SCORE_MAXLEN` in
`score_eval_dense_v4.py`. The chain exports `0.03` for the RoyalRoad leg only and
unsets it before Wigleaf. Verified: RoyalRoad loads at 0.03, still **fails** at the
default 2e-2, Wigleaf passes at the default untouched.

**Layer-1 stack** — frozen protocol, machinery imported (never re-typed) from
`layer1_gemma_cells.py` / `scaleupC_layer1.py` via
`methods/taste_decomposition/cw_expert_layer1.py`:
linear = family1 (`SimpleImputer(median, add_indicator)` + `StandardScaler` +
`LogisticRegression(C=1, liblinear, max_iter 2000, rs 20260728)`), GroupKFold(5);
nonlinear = HistGradientBoosting, frozen grid {15,31} leaves / lr .06 / max_iter 400 +
early stopping, grid picked by inner GroupKFold(3) **inside each outer train fold
only**, per-fold imputation identical to the linear leg. **VA_nl / V_nl = mean over
seeds {0,1,2}** with spread reported (FREEZE CHANGE 1) — the seed spread the 2026-07
campaign never ran, which is part of why we are here. Δ_interact CI = **group-level**
bootstrap (FREEZE CHANGE 3).

Both cells are **FIRST-FIT**: no prior V+A stack of this construction exists, so the
linear leg *is* the first fit and there is no external reproduction gate. In its place
the ledger carries an **assembled-order gate**: the sharded matrices are independently
re-assembled, the OOF vectors are re-keyed by item id, the rows are randomly permuted,
and every headline AUC is recomputed — required to agree to **< 1e-9**. OOF arrays are
saved **with their ids vector** (`results/<cell>_oof.npz`: `ids`, `groups`, `y`,
`secondary_groups`, and every OOF column), not as bare positional `.npy`.

---

### 2a. Two rulings applied (2026-08-10)

**R1 — the collapse gate is ENFORCED inside `clean_fit`, not merely reported.**
Any criterion whose modal value covers > 98% of its finite entries is *dropped before
fitting*. Two properties make it a gate rather than a cosmetic filter: the mask is
computed on the **TRAIN fold only, inside each outer fold** (no leakage into held-out
rows), and the **identical mask is applied to the linear and the nonlinear leg**, so
Δ_interact stays a like-for-like contrast (design-note protocol point 2). Implemented as
`collapse_mask` / `linear_oof_gated` / `gbm_oof_gated` in `cw_expert_layer1.py`; the
per-fold drop counts are recorded in each ledger under `collapse_gate`.

This gate bites hardest exactly where Wigleaf is weakest — see §5, where the bank sits at
mean .899 with 83% of all responses at 1.0.

**R2 — truncation is measured in TOKENS, not characters.** The old CW constants (6,000
source chars → 3,600 head / 2,400 tail) make the budget depend on text density, so
dialogue-heavy and prose-dense stories receive different amounts of actual content. The
judge context is now cut with the **gemma-4-31b tokenizer** at **1,600 source / 960 head /
640 tail**, the token-exact analogue of the old char budget on the same 60/40 split.

This is not cosmetic for RoyalRoad: **1,077 of 1,274** opening chapters exceed the budget
(source tokens median **2,918**, max **19,532**), so essentially the whole cell is
truncation-sensitive and the character-based cut was giving denser prose systematically
less story. The 2026-08-08 character-truncated scoring is preserved, not deleted, at
`outputs/va_gemma_banks_cw_expert_CHARTRUNC_20260808/`; all headline numbers below come
from the token-truncated rescore. The dense arm was already token-truncated
(`max_length=1024` via the HF tokenizer), so **T did not need recomputing** — only the
A-bank pass was redone.

## 3. Comparison discipline

The old numbers (.505 RoyalRoad / .578 Wigleaf) are carried in both ledger JSONs as
`prior_instrument` **context only**. They are a *different instrument* — different bank
construction (k-medoid coverage selection over a mined pool vs GEPA-phrased), different
judge (likely Llama-3.3-70B vs Gemma-4-31B), no anchor battery, no T. A difference
between the new A_lin and .505/.578 is **not** a same-instrument delta and is never
reported as one; it is a bank-and-judge swap on a fixed population.

Also carried forward from the re-audit: "NULL BANK" was a mislabel for Wigleaf. Its
.578 was the single highest craft-rankability number in the whole CW leg, and its
"0 kept" was a **saturation** finding (no *new* mined metric added bits beyond the
existing 37-criterion bank), not a chance-level bank. Only RoyalRoad's .505 was at
chance.

---

## 4. Artifacts

| what | path |
|---|---|
| population builder | `datasets/creative-writing/build_cw_expert_va.py` |
| RoyalRoad population | `datasets/creative-writing/royalroad_stubs/va/population.csv.gz` (+ `population_manifest.json`) |
| Wigleaf population | `datasets/creative-writing/wigleaf/va/population.csv.gz` (+ `population_manifest.json`) |
| dense-standard inputs | `datasets/creative-writing/{royalroad_stubs,wigleaf}/dense_standard/{data.csv,split/,manifest.json}` |
| A-bank scorer | `datasets/va_gemma_banks/score_cw_expert_banks.py` |
| scored matrices | `outputs/va_gemma_banks_cw_expert/<cell>_shard{0..3}.npz`, `<cell>_meta.json` |
| anchor battery (K=50/class) | `outputs/va_gemma_banks_cw_expert/anchor_battery.json` |
| judge distribution check | `outputs/va_gemma_banks_cw_expert/distribution_check.json` |
| dense T runs | `datasets/creative-writing/{royalroad_stubs,wigleaf}/dense_standard/rm_out_seed{42,1,2}/`, `eval_pass_results.json` |
| layer-1 driver | `methods/taste_decomposition/cw_expert_layer1.py` |
| **ledgers** | `methods/taste_decomposition/results/cw_royalroad_verdict_ledger.json`, `.../cw_wigleaf_curation_ledger.json` |
| OOF arrays (with ids) | `methods/taste_decomposition/results/<cell>_oof.npz` |
| run chain | `methods/dense/run_cw_expert_chain.sh`; GPU claim/poll wrapper `scripts/tools/cw_expert_gpu_claim_and_launch.sh` |
| logs | `logs/cw_expert/{launcher.log,chain.log,stage1_gemma_score.log,layer1.log}` |
| superseded char-trunc scoring (kept) | `outputs/va_gemma_banks_cw_expert_CHARTRUNC_20260808/` + `logs/cw_expert/stage1_gemma_score.CHARTRUNC.log` |

Scale: 1,274×45 = 57,330 + 1,568×45 = 70,560 = **127,890 judge prompts**, plus
2 × 150 × 45 = 13,500 battery prompts.

### 4a. Operational landmine — a 0-MiB claim is not a reservation

Worth recording because it cost a cycle and will recur for anyone running on this box
while the full-grid campaigns are up. The launcher verified GPU3 at **0 MiB / 0% util**,
posted its CLAIM at 19:20:52, and started vLLM — which then died 20 s later:

> `ValueError: Free memory on device cuda:0 (41.24/178.35 GiB) on startup is less than
> desired GPU memory utilization (0.93, 165.87 GiB)`

Another agent stacked ~137 GiB onto the card **inside vLLM's init window**. The claim
was released cleanly (rc=1, no orphan process). Nothing was crowded and no co-tenant
was touched, but two things had to change:

1. **`--auto-util`** — size `gpu_memory_utilization` from `torch.cuda.mem_get_info()`
   *at engine-init time*, capped by `--util`, with `--min-gib 80` (Gemma-4-31B bf16 is
   ~62 GiB of weights plus KV cache) and `--headroom-gib 6`. Under the floor it aborts
   with exit 4 rather than crowding the card. This mirrors the V7-patents precedent in
   the ledger ("util=0.434 sized to 83569MiB free").
2. **launcher retry** (`ATTEMPTS=12`) — a lost race re-polls for another free card
   instead of killing the campaign. Safe because stage 1 is shard-checkpointed and
   stage 2 is `RUN_DONE`-sentinel resumable, so a retry *resumes*, never restarts.

Generalisation: on a contended box, free memory at claim time and free memory at
engine-init time are different numbers. Size the engine to the latter.

---

## 5. Results

Ran 2026-08-10 on GPU0 (claimed ledger-free, 102.3 GiB free, auto-util 0.540 = 96.3 GiB;
released rc=0). 127,890 judge prompts + 13,500 battery prompts. Layer-1 CPU.

### 5.1 Ledgers

| | **RoyalRoad** VERDICT | **Wigleaf** CURATION |
|---|---|---|
| n / positives | 1,274 / 637 (.500) | 1,568 / **404** (.258) |
| V_lin | .5545 | .5927 |
| V_nl (mean seeds 0-2) | .5379 | .5728 |
| **A_lin** | **.5628** [.530, .595] | **.5407** [.509, .573] |
| VA_lin | .5662 [.534, .597] | .5923 [.560, .625] |
| **VA_nl** (mean seeds 0-2) | **.5558** | **.6051** |
| VA_nl seeds | .5599 / .5534 / .5540 (spread **.0065**) | .6005 / .6060 / .6087 (spread **.0083**) |
| **T** (dense eval, seed-mean) | **.4994** | **.6054** |
| T per seed (eval) | .4822 / .485 / .531 | .6042 / .5556 / .6563 |
| T per seed (test) | .5224 / .5387 / .5728 | .575 / .5943 / .6162 |
| Δ_interact | −.0104 (seed0 −.0062, CI [−.043,+.029], P(>0)=.36) | +.0128 (seed0 +.0082, CI [−.024,+.039], P(>0)=.68) |
| Δ_total | −.0668 | +.0131 |
| **Δ_beyond** | **−.0564** (see 5.3 — not a usable headroom) | **+.0003** |
| old instrument | .505 | .578 |

Gates: **assembled-order gate PASS on both, max\|diff\| = 0.00e+00.** Collapse gate
(R1) dropped 1 A-criterion in 3 of 5 RoyalRoad folds, 0 on Wigleaf. Shard 2 was
anchor-INVALID on both cells; leave-it-out sensitivity moves nothing materially
(RoyalRoad n=968: V .571 / A .568 / VA .578; Wigleaf n=1,198: V .584 / A .560 / VA .604).
Seed spreads (.0065, .0083) are smaller than every Δ discussed — the spread the 2026-07
campaign never ran.

### 5.2 Instrument QC — the two cells diverge sharply

| | RoyalRoad | Wigleaf |
|---|---|---|
| distribution check | **PASS** mean .7232, NA .0118, hist 0.0:4,678 / 0.5:22,009 / 1.0:29,964 | **PASS** mean **.8991**, NA .0261, hist 0.0:2,021 / 0.5:9,820 / **1.0:56,874** |
| K=50 battery | pos **.7750** > neg **.6919** > scram **.0000** | pos **.8798** < neg **.9016** > scram .0698 |
| pos-vs-neg AUC | **.658** | **.498 — at chance** |
| coherent-vs-scrambled AUC | 1.000 | .993 |
| ordering holds | **YES → certified** | **NO → NOT certified for the pos/neg contrast** |

**RoyalRoad's A bank is certified. Wigleaf's is not.** The Gemma judge plainly executes
the criteria on Wigleaf — it separates real prose from word salad at .993 — but it cannot
separate Top-50 from longlist: the battery ordering inverts, and **83% of all Wigleaf
responses are 1.0** (mean .899). That is range restriction, and it is substantively
sensible: both Wigleaf classes are *already-published literary flash fiction*, so a
generic craft bank saturates and the editor's cut is a distinction it cannot see. Any
Wigleaf A-number must be read behind that flag. This replicated identically under both
character- and token-truncation, so it is a property of the cell, not of the cut.

### 5.3 RoyalRoad's T is at chance — and that is a power statement, not a ceiling

All three dense seeds land at eval .4822 / .485 / .531 (mean **.4994**) while the bank
sits at VA_lin .5662, so Δ_beyond is **negative**. A negative Δ_beyond is not "negative
taste headroom" — it means **the dense arm is not a valid ceiling for this cell** and
Δ_beyond should not be quoted as a headroom estimate. Pre-kill checklist item (b), run on
this exact split rather than quoted from an older pooled run:

| same-split baseline | RoyalRoad eval / test | Wigleaf eval / test |
|---|---|---|
| TF-IDF + logistic | **.4348 / .6149** | .5631 / .5194 |
| log-length only | .5828 / .4547 | **.6801** / .5740 |
| dense (seed-mean) | .4994 / .5446 | .6054 / .5952 |

On RoyalRoad the simple baseline does **not** reliably beat chance either — TF-IDF is
*below* chance on eval (.435) and above on test (.615), and length-only flips the other
way (.583 / .455). So dense-at-chance is **not** the checklist's "baseline > big model =
training-run failure" signature; it is consistent with a genuinely weak text signal read
through 141-row eval and 142-row test splits. The honest statement: **at n=1,274 with
~141-row evaluation splits, RoyalRoad cannot support a trustworthy T**, and the earlier
".588 LEXICAL" figure (a differently-pooled evaluation) should not be carried as this
split's floor.

Wigleaf's dense arm behaves properly by contrast (.605 > TF-IDF .563 on eval). But
**log-length alone reaches .680 on the Wigleaf eval split** — above the entire V+A stack
— then falls to .574 on test. Length is doing much of the observable work and the
splits are too small (43 and 48 positives) to pin it down. Length lives inside V by
design (`v_log_chars`, `v_log_words`), so this is not a leak, but it does mean Wigleaf's
V_lin .5927 should be read as "largely a length/size effect," not as craft.

### 5.4 Against the old-instrument numbers

The swap moves the two cells in **opposite** directions:

| cell | old bank (2026-07-06) | new mature A_lin | change |
|---|---|---|---|
| RoyalRoad | .505 (called a "genuine null") | **.5628** | **+.058** |
| Wigleaf | .578 (best craft-rankability in the CW leg) | **.5407** | **−.037** |

Both are *instrument swaps* — different bank construction (GEPA-phrased vs k-medoid),
different judge (Gemma-4-31B vs likely Llama-3.3-70B), different truncation unit — on a
fixed population. They are **not** same-instrument deltas and must never be differenced
as such. With group-bootstrap CIs of roughly ±.032, neither move is individually
decisive; what is notable is the *direction*.

Read together with 5.2, the reversal has a coherent reading. The re-audit's headline
worry was that RoyalRoad's null might be instrument-limited. It partly was: a mature bank
does lift it off chance (.505 → .563). But the cell's dense ceiling is at chance, so
nothing about RoyalRoad is well-measured at this n. Meanwhile Wigleaf — the cell that
looked *best* under the old instrument — is where the mature instrument does **worse**
and where the anchor battery outright fails to certify the pos/neg contrast. The
"documented NULL BANK" label was wrong for Wigleaf, as the re-audit argued, but the
correction is not "it ranks craft well"; it is **"its bank saturates and the instrument
cannot certify it."**

One thing the rebuild does settle cleanly: on Wigleaf, **VA_nl .6051 ≈ T .6054**, so
Δ_beyond = **+.0003**. The V+A stack reaches the dense ceiling — there is no measurable
taste residual beyond the articulated stack on this cell. Note this is achieved with
A contributing essentially nothing over V (VA_lin .5923 ≈ V_lin .5927; A_lin .5407 is the
weakest leg), i.e. the ceiling is reached by surface features, not by craft criteria.

### 5.5 Standing caveats

* **Wigleaf power caveat** (carried in the ledger JSON): 404 absolute positives; eval and
  test rest on **43** and **48** positives. Same order as the mathlib false-null case.
* **RoyalRoad T caveat**: 141-row eval / 142-row test; all three seeds at chance;
  Δ_beyond unusable as headroom.
* Δ_interact is within the seed spread on both cells (P(>0) = .36 and .68) — **no
  interaction effect on either.**
* Train-vs-OOF overfit gaps are large on both (VA ≈ .33–.41), expected for GBM on
  ~1.3–1.6k rows with 60 columns; the OOF ledger adjudicates, not the train fit.
* Both cells are FIRST-FIT: no external reproduction gate exists, so the assembled-order
  gate stands in its place (both PASS at exactly 0.0).

### Reproduce


```bash
# 1. instrument QC — read BEFORE any AUC
cat outputs/va_gemma_banks_cw_expert/distribution_check.json   # judge collapse guard
cat outputs/va_gemma_banks_cw_expert/anchor_battery.json       # K=50/class, pos>neg>scram
cat datasets/creative-writing/*/dense_standard/eval_pass_results.json  # T per seed

# 2. ledgers (CPU, ~minutes)
python methods/taste_decomposition/cw_expert_layer1.py --cell all
```

Read order matters, and it bit here: both cells PASS the distribution check, but
**Wigleaf fails the battery ordering**, so its A-numbers carry the §5.2 flag everywhere
they appear. RoyalRoad passes both and its A-numbers are certified.

---

## 6. Follow-up package (2026-08-11) — power fix, pairwise instrument, expansion

### 6.1 RoyalRoad 5-fold cross-fit — the power fix CONFIRMS the chance verdict

The single-split T (.4994) rested on a 141-row eval that was *also* the
checkpoint-selection split. Replaced with the SO-template cross-fit: bucket =
`md5("split::"+fiction_id)%1000//100`; fold k trains on 8 tenths, selects on tenth
2k+1, predicts tenth 2k. Honest set = union of the 5 test tenths, **n=651 / 308 pos,
selection-free, 4.6x the old eval**. Bucket rule reproduces the canonical split at
**1.0000**. Seed 42.

| fold | n | pos | AUC |
|---|---|---|---|
| 0 | 132 | 60 | .5061 |
| 1 | 126 | 66 | .4683 |
| 2 | 134 | 65 | .5570 |
| 3 | 118 | 57 | .4781 |
| 4 | 141 | 60 | .4836 |

**T_fold_mean .4986 · T_pooled .4981 [CI .4532, .5399] · T_rank_pooled .4994** —
estimator spread **.0013**, so cross-fold calibration is not an issue. The prior
single-split .4994 is reproduced almost exactly.

**Verdict: T-at-chance was NOT a power artifact.** With 4.6x the rows and a
selection-free honest set, RoyalRoad's dense ceiling is still chance. The fix was
worth running precisely because it converts "we cannot tell" into "we can, and it
is chance".

Same rows (n=651), bank OOF restricted to the honest ids, never re-fit:

| | value |
|---|---|
| V_lin | .5764 |
| A_lin | **.5875** |
| VA_lin | .5946 |
| VA_nl (mean 3 seeds) | .5718 |
| **T (fold mean)** | **.4986** |
| Δ_beyond (T − VA_nl) | **−.0732** |
| Δ vs A_lin | −.0889 |

**The bank beats the dense arm by ~.09 on identical rows.** This trips the standing
"dense/fused upper bound fails to beat the bank" rule and should not be read as
"no residual": T is a *lower* bound on the ideal model. Two mechanisms are live and
must be named before anyone quotes this: (i) **view asymmetry** — the dense arm reads
the first 1,024 tokens while the A judge reads head 960 + tail 640, i.e. it *sees the
ending*, and RoyalRoad chapters run to a median 2,918 tokens, so dense sees ~35% of
the chapter and never the close; (ii) an 8B LoRA on ~1k rows is a weak ceiling. The
clean test of (i) is a head+tail-view dense arm on the same folds — not run.

### 6.2 Wigleaf pairwise probe — the instrument fix WORKS

200 matched pairs (130 same-magazine / 70 same-year), 40 flipped replicates, 30
scrambled anchors, 45 bank criteria + a holistic question, judged comparatively by
**gpt-5.6-sol** via `codex exec`. Coverage **270/270**.

**Validity gates (read first):**

| gate | result |
|---|---|
| anchors (real vs scrambled) | **pick-real .9963**, holistic 1.000 |
| order consistency (flipped replicates) | criteria **.8406**, holistic **.90** |
| position | A .4297 / B .5426 / **TIE .0278** |

The tie rate is the tell: **2.8%**, against 83% of absolute responses pinned at 1.0.
The comparative frame restores the headroom the 3-point absolute scale had lost.

**Separation:**

| readout | AUC | CI95 |
|---|---|---|
| composite (majority of 45) | **.610** | [.5425, .6775] |
| holistic (overall stronger) | **.610** | [.54, .68] |
| mean per-criterion | .5678 | — |
| same-magazine stratum | **.6231** | — |
| same-year stratum | .5857 | — |
| *absolute bank, for context* | A_lin .5407; battery pos-vs-neg **.498** | — |

32/45 criteria exceed .55; **23/45 have a CI excluding .50**. Crucially the
**same-magazine** stratum is the *strongest* (.6231), so venue is not driving it.

**Answer to the probe question: yes.** Comparative judging separates the editor's
cut where absolute scoring could not — the battery went from **.498 (chance)** to a
composite of **.610** with anchors at .996 and order-consistency at .84.

What the editor's cut rewards, per-criterion — endings and voice, not tidiness:

| top | AUC | | bottom | AUC |
|---|---|---|---|---|
| Ending resonance | .6325 | | Dynamic relationships | .5250 |
| Distinctive narrative voice | .6175 | | Exposition is integrated | .5175 |
| Ending is earned | .6150 | | **Prose economy** | **.4600** |
| Tonal control | .6125 | | **Causal narrative progression** | **.4575** |

The two below-chance criteria are substantive, not noise: longlist pieces are
*more* economical and *more* causally tidy. The Top-50 cut rewards a resonant close
and a distinctive voice and mildly penalises conventional neatness.

**Estimand caveat:** pairwise AUC is measured on matched pairs under forced choice;
A_lin .5407 is an unpaired grouped-OOF AUC over all 1,568 rows. These are different
estimands — the comparison licenses "comparative judging separates the cut", never a
like-for-like ".610 − .5407" delta.

### 6.3 RoyalRoad expansion — GO, n=1,742 at lexical .5759

Sweep over k-means topic granularity (bge-large, 2,367-fiction usable pool), taking
the largest topic×era-matched subsample under a .58 lexical margin:

| k | n | lexical | register | era abs r | new rows | |
|---|---|---|---|---|---|---|
| 6 | 1,952 | .6028 | .5575 | .000 | 827 | over |
| 16 | 1,872 | .5976 | .5623 | .000 | 791 | over |
| 20 | 1,828 | .5814 | .5497 | .000 | 761 | over |
| **24** | **1,742** | **.5759** | .5382 | .000 | **719** | **CLEAN** |
| 32 | 1,746 | .5883 | .5427 | .000 | 719 | over |
| 40 | 1,688 | .5715 | .5263 | .000 | 591 | CLEAN |

**Chosen k=24: n=1,742 (+468 over the cell of record), lexical .5759, era-y corr
.000** — clears the .58 margin and the 1,450 floor, so the build proceeds. 719 new
rows, 1,023 carried; note the expanded set is *not* a superset — 251 of the old rows
fall out when the matching is recomputed at k=24.

Split (per-fiction stable hash, so growth moves no existing row): train 1,374 (702
pos) / eval 181 (82) / test 187 (87). Bank rescoring runs on the **719 new rows
only**; carried rows keep their token-truncated scores and are never re-judged.

---

## 7. Second follow-up (2026-08-11/12)

### 7.1 rr_v2_k24 — SENSITIVITY DESIGN, not the cell of record

The hole was fillable; the fill loses certification. rr_v1 (n=1,274) stays canonical.

| | rr_v1 (canonical) | rr_v2_k24 (sensitivity) |
|---|---|---|
| n | 1,274 | 1,742 |
| lexical floor | **.524** | .5759 |
| K=50 battery pos-vs-neg | **.658 PASS** | **.445 FAIL** (pos .7157 < neg .7577) |
| A_lin | .5628 | .5627 |
| VA_lin | .5946 (honest rows) | .5628 |
| VA_nl | .5718 (honest rows) | .5505 |
| T | .4986 (cross-fit, n=651) | .5112 (n=1,742) |
| Δ_beyond | −.0732 | −.0393 |

Two things worth keeping. **(a) The two-batch seam is clean**: carried-row mean A
.7262 vs new-row .7270, and batch membership predicts y at .5098 (chance), so
merging the rr_v1 and expansion judging batches introduced no artifact. **(b) n was
never the binding constraint on RoyalRoad's T** — going 1,274 → 1,742 moved T from
.4994/.4986 to .5112, still chance. That makes the head+tail view hypothesis the
live one, not sample size.

### 7.2 Wigleaf pairwise at n=600 — the size channel is the whole story

Wave 2 added 400 pairs (253 same-magazine, no combination reuse, 11% anchors);
coverage 484/484. Merged n=600.

Gates hold at scale: anchors **.9982**, order consistency **.8331**, position
A .4682 / B .5062, tie rate 2.6%.

But the separation regresses hard from the 200-pair pilot:

| readout | n=200 | **n=600** |
|---|---|---|
| composite | .610 | **.5517** [.5117, .5917] |
| holistic | .610 | **.5417** |
| same-magazine | .6231 | **.5796** |
| same-year | .5857 | .5023 |

**P2 pairwise-native Layer-1 (600 pairs, antisymmetry enforced, group = pair):**

| | full | size-matched (300) | size-divergent (300) |
|---|---|---|---|
| V_lin | .6655 | **.6140** | .6773 |
| **A_lin** | .5933 | **.4986 — EXACTLY CHANCE** | .6300 |
| VA_lin | .6566 | .5659 | .6518 |
| VA_nl | .6621 (spread .0073) | .5999 | .6875 |
| length-only | .6191 | .5361 | .6703 |

**The ruling-3 test returns a clean verdict: the craft bank vanishes.** Hold
|Δ log-size| below its median and the 45 GEPA-phrased criteria score **.4986** —
chance to three decimals. Every bit of A's apparent pairwise signal (.5933 overall,
.6300 among size-divergent pairs) rides on the size channel.

What does survive size-matching is **V at .6140** — and *not* because V is size:
length-only falls to .5361 in the same stratum. So the signal is non-size surface
style (type-token ratio, sentence-length variability, dialogue fraction, readability),
not articulated craft.

**Wigleaf's honest story is editorial scope plus surface style, not craft.** Every
Wigleaf pairwise number travels with this. The earlier "endings and voice" reading
from the 200-pair pilot does not survive n=600 — at scale the top criterion is
Ending resonance .5842, and Causal narrative progression is still below chance
(.4658), but the whole per-criterion spread now sits inside the size channel's shadow.

T_pair is still pending the cross-fitted Wigleaf dense arm (queued); the P2 script
computes it automatically once those folds land.

---

## 8. Handoff — future work, and what is running

### 8.1 Trainer improvement to make OUTSIDE any live ledger run

`methods/dense/train_reward_model.py` tokenises with **`padding="max_length"`**
(lines ~251 and ~279), so every row costs the full budget regardless of its true
length. On the RoyalRoad V3 full-text arm this is the dominant cost: the median
chapter is 2,918 tokens against a 16,384 budget, so roughly **five sixths of the
compute is spent on padding**. At 16384 a row costs 10.2x what it costs at 1600,
which is why the 10 fold-arms run ~15 h instead of ~3 h.

**Proposed change (NOT made here):** switch the collator to dynamic padding
(pad to the longest sequence in the batch). With dynamic padding a 16384-cap run
would be *cheaper* than today's 8192-cap run, because almost every batch would pad
to a few thousand tokens rather than the cap.

**Adoption protocol, deliberately strict** — this edits a file every dense ledger in
the program depends on:
1. make the change on a branch, with no live chain invoking the trainer;
2. re-run ONE small completed fold (e.g. `dense_crossfit/fold0`, seed 42) under both
   collators and require a **byte-identity check** on `preds_test.csv` — dynamic vs
   max_length padding must produce identical predictions, since padding is masked and
   should be numerically inert;
3. only if identity holds, adopt it and record the equivalence in the registry.
If identity does NOT hold, that is itself a finding (it would mean padding is leaking
into the head) and the change must not be adopted silently.

Frozen-recipe discipline outranks wall-clock: this was explicitly deferred rather
than taken during the live queue.

### 8.2 Lane state at handoff

| arm | design id | status |
|---|---|---|
| rr_v1 cross-fit (honest T) | `dense_crossfit` | done — T .4986 |
| judge-view audit | `dense_crossfit_judgeview` | done — T **.5846** (+.0860) |
| rr_v2 expansion | `rr_v2_k24` | done — SENSITIVITY only |
| Wigleaf cross-fit | `wigleaf/dense_crossfit` | done — T **.5589** |
| **V3 full text, arms a+b** | **`v3_aug_fulltext`** | ▶ running, ~15 h, 16384, 4/1274 truncated |
| decomposition arm | `decomp1024` (614+410 @1024) | queued (phase 2) |
| Stage-A gate + zero-shot ×2 | `cw_transfer_v1` | queued (phase 2) |
| Stage B fine-tunes | — | not built |
| final CW ledger | — | not built; triggers the fused-must-beat-bank auto-audit, log registry-first |

### 8.3 Numbers that are superseded and must not be re-quoted

* RoyalRoad **Δ_beyond −.0732** — superseded. On the fair (judge) view it is **+.0128**.
* RoyalRoad single-split **T .4994** on a 141-row eval — superseded by the cross-fit
  (.4986) and then reframed by the view audit; the standard-1024 T is a VIEW artifact
  pending the decomposition arm's split of view vs budget.
* Wigleaf **T .6054** — never-quote (select-on-eval optimism). Item-level number of
  record is **.5589**; **T_pair .5618**.
* Wigleaf pairwise **composite .610** from the 200-pair pilot — superseded by
  **.5517** at n=600.
* rr_v2_k24 numbers are never canonical (A-bank battery failed at .445).

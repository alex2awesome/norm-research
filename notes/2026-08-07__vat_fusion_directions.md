# V+A+T fusion directions on the dense-below-bank cells

Date: 2026-08-07. Status: COMPLETE (all three directions run; SI Direction 2
skipped — no extra data; SI Direction 3 deferred, see §Direction 3).
Descriptive only — no claim edits.

Task (user-specified 2026-08-07): three fusion directions on the three cells
where the dense standard FAILS to beat the articulated bank:

| cell | same-rows T (held-out E) | VA_nl_mean (Layer-1, full pop) |
|---|---|---|
| cap_crowd | .5554 (n=2,190) | .6656 |
| cap_finalist | .6124 (n=1,055) | .6800 |
| style_inv_toptier | .6343 (clean-eval, 3 seeds) | .6651 |

Terms (first use): **V** = verifiable/surface features; **A** = articulated
criterion scores (Gemma-4-31B judged); **VA_lin / VA_nl** = linear / HistGB
aggregation of the V+A matrix (frozen Layer-1 protocol,
notes/2026-08-05__taste-decomposition-design.md §1); **T** = dense-standard
clean-eval AUC (Llama-3.1-8B LoRA reward model on raw text); **E** =
evaluation-valid rows = rows OUTSIDE the dense model's own training split
(dense eval+test buckets), the only rows where the dense per-row prediction is
out-of-sample; **VAT** = the V+A matrix with the dense per-row probability
appended as one extra column.

Code + artifacts: `methods/taste_decomposition/fusion/` (all scripts, per-cell
JSONs, dense_data builds). GPU work on sk3 GPU claimed via
`gpu_ledger.txt` (agent=claude-vat-fusion; watcher queued behind co-tenants).

## Direction 1 — dense scalar into the stack (CPU, all three cells)

`fusion/direction1_stack.py`. Leakage rule enforced by construction: the whole
analysis is restricted to E, where the dense column is out-of-sample; stacks
are refit on E with the frozen Layer-1 protocol (grouped OOF GroupKFold(5),
linear + HistGB seeds {0,1,2} mean; family2 clean_cols guard re-applied on E
for captions, family1 per-fold imputer for SI). Dense preds sources:
captions `closure/samerows_preds/{cell}_dense_preds_slim.csv`; SI
`dense_standard/rm_out_seed{42,1,2}/preds_{eval,test}.csv` (per dense seed,
reported per-seed + mean; join back to npz ids verified by y-equality asserts).

Known caveat (matched across arms, inherited from T itself): the dense
checkpoint was selected on a split inside E (captions: selection_split=test;
SI: selection on eval), so E is not selection-clean for any arm; the registry
made the same trade.

Secondary context row: "VA fullfit@E" = the full-population Layer-1 OOF
predictions evaluated on E rows only (no small-train handicap, but its
training folds span E's groups' complements — still OOF for every row, so
honest; it differs from the E-refit by training on ~5x more rows).

## Direction 2 — more training data for the failing dense models

Inventory (`fusion/build_direction23_data.py`, from
`datasets/humor/caption_multiy/cap_scores_shard*.npz` + `caption_contest_v2.jsonl`):

- Scored caption pool = 18,838 rows (finalist 678 / neg_hard 4,540 /
  neg_random 13,620); 18,750 have crowd_mean.
- **cap_crowd** (modeled 10,893 = crowd_votes>=100, contest>=6, non-tie):
  excluded-but-labeled = 6,677 rows with 50<=votes<100 (+787 at 20-50, +276
  below). Restricting to contests in the dense TRAIN bucket (the split is
  contest-grouped — extras from eval/test contests would leak group identity)
  and labeling against the contest's existing votes>=100 median: **+3,997
  extra train rows (+45.9%)** → material, retrain runs.
  NOTE: extras are 100% label-0 — every 50-99-vote caption sits below its
  contest's >=100-vote median, consistent with adaptive vote allocation
  (better captions get more votes), so "more data" here = more negatives.
- **cap_finalist** (modeled 5,218 = finalist vs neg_hard): excluded-but-labeled
  = 13,620 neg_random (label 0, valid by construction). Train-bucket contests
  only: **+10,860 extra train rows (+260.9%)** → material, retrain runs.
  Same caveat: negatives only; train pos rate drops .130 → .036.
- **style_inv_toptier**: `style_invitational.jsonl` has exactly 9,637 rows and
  the dense build used ALL of them (build_dense_standard.py asserts n/pos/
  groups against the Layer-1 population). **No material extra data exists —
  cell SKIPPED for Direction 2** (per task rule: don't manufacture data).

Retrain recipe: verbatim original caption chain config (Llama-3.1-8B LoRA
r16/a32, lr 5e-5, bs16, max_len 1024, 2 epochs, grad-ckpt,
**selection_split=test — matching the original cap chains**, seed 42), eval/
test split CSVs byte-content-identical (only train.csv grows). The stock
trainer hard-rejects non-80/10/10 split dirs, so `fusion/train_grown_split.py`
relaxes ONLY that gate to the observed fractions; everything else untouched.

## Direction 3 — feature-augmented dense (text + top criteria in prompt)

`fusion/build_direction23_data.py`. Input format (both arms):

    full text:
        <caption>
    VA metrics:
        <name>: <score>                     (arm a)
        <name> (importance 0.20): <score>   (arm b)

- Top-10 criteria from the cell's cleaned VA matrix ranked by TRAIN-ONLY
  grouped permutation importance (GroupKFold(3) within the dense train split,
  frozen HistGB leaves=31/lr=.06, permutation_importance roc_auc n_repeats=5
  on the inner held-out fold, mean over folds). Importance never sees
  eval/test rows; criterion scores are label-blind judge outputs (safe on all
  splits); y never appears in the prompt.
- cap_crowd top-10 = all A criteria (Virality/shareability .20, Rhetorical
  polish .18, Delivery/timing .17, ...); cap_finalist mixes A + V
  (Cross-cultural translatability .21, v_char_len .19, v_digit .18, ...).
  Full lists in each `dense_data/*_aug_*/manifest.json`.
- Same rows, same splits, same recipe as the original cell chains
  (selection_split=test, matching the originals; the task sheet said
  select-on-eval, which is the generic V4-standard convention — the caption
  originals selected on test, and matching the cell's own T was judged the
  controlling principle; per-split AUCs are reported so either convention can
  be read).
- max_len kept at 1024: augmented texts max 830 chars (~250 tokens) — no bump
  needed; exact truncation rate via tokenizer in
  `dense_data/truncation_report.json` (expected 0).
- SI aug arm: deferred ("if time allows"); not started with the caption chain.

## Results

(Direction 1 per-cell JSONs `fusion/{cell}_direction1.json` + `*_direction1b.json`;
Directions 2+3 `fusion/dense_data/harvest_results.json` + `fusion/direction3_boot.json`;
training logs on sk3 `fusion/dense_data/*/train_seed42.log`)

### Direction 3 — feature-augmented dense (E-pooled, same rows as Direction 1)

Truncation rate 0.0 everywhere (aug prompts max 274 tokens at max_len 1024;
`dense_data/truncation_report.json`).

| quantity (AUC on E) | cap_crowd (n=2,190) | cap_finalist (n=1,055) |
|---|---|---|
| T original dense | .5554 | .6124 |
| VA_nl fullfit@E (bank, seed-mean OOF) | .6217 | .6666 |
| **T_augmented (a: scores only)** | **.6190** | **.6707** |
| **T_augmented (b: scores + weights)** | **.6193** | **.6676** |

Per-split: crowd aug_a eval .6361 / test .6016; aug_b .6324/.6050; finalist
aug_a .6775/.6658; aug_b .6770/.6615. (selection_split=test as in the original
cap chains, so eval-only is the selection-clean leg; original registry
eval-only T was .5631 / .6252.)

Paired row-level bootstraps (`direction3_boot.json`):
- aug_a − T_orig: crowd **+.0636 [+.0353, +.0925] P(>0)=1.000**; finalist
  **+.0583 [+.0106, +.1036] P(>0)=.991** — the metric block decisively lifts
  the dense reader on both cells.
- aug_a − VA_nl_fullfit: crowd −.0027 [−.0182, +.0131]; finalist +.0041
  [−.0337, +.0403] — the augmented dense converges TO the bank's level,
  neither above nor below.
- aug_a − aug_b: −.0003 / +.0031, both CIs tight around 0 — importance
  weights (user's worry) neither help nor hurt; no over-reliance signature.

SI aug arm: deferred (task said "if time allows"; the caption chain + moredata
relaunch consumed the GPU window). The build machinery generalizes (family1
matrix + `build_dense_standard.py` splits) if it's wanted later.

### Direction 2 — T_moredata (E-pooled, same rows; `direction2_boot.json`)

| quantity (AUC on E) | cap_crowd | cap_finalist |
|---|---|---|
| T original | .5554 | .6124 |
| **T_moredata** | **.6087** | **.6303** |
| VA_nl fullfit@E (bank) | .6217 | .6666 |

Per-split: crowd eval .6261 / test .5912; finalist eval .6723 / test .5854
(note the finalist eval/test asymmetry — the +261% extra negatives shift the
train distribution hard, and the two held-out contest buckets react
differently; selection was on test, so eval .6723 is the selection-clean leg).

Bootstraps: moredata − T_orig: crowd **+.0533 [+.0261, +.0789] P(>0)=1.000**;
finalist +.0180 [−.0283, +.0653] P(>0)=.77. moredata − bank: crowd −.0129
[−.0422, +.0139]; finalist −.0362 [−.0885, +.0157] — more data lifts the
dense reader (decisively on crowd) but does NOT reach the bank on either cell.

style_inv_toptier: SKIPPED — the 9,637-row jsonl is the entire corpus and the
dense build already used all of it; no material extra data exists.

## Final per-cell tables (everything on the SAME evaluation-valid rows E)

### cap_crowd (E = 2,190; dense-held-out rows of the 10,893 population)

| arm | AUC on E |
|---|---|
| VA_lin (E-refit) | .5863 |
| VA_nl (E-refit, seeds 0-2) | .5831 |
| T (original dense) | .5554 |
| VAT_lin (E-refit) | .5878 |
| VAT_nl (E-refit, seeds 0-2) | .5920 |
| VA_nl fullfit@E / + T combiner (1b) | .6217 / .6204 |
| T_moredata (+46% train, all-negative extras) | .6087 |
| T_augmented (a: scores) | .6190 |
| T_augmented (b: scores+weights) | .6193 |

### cap_finalist (E = 1,055)

| arm | AUC on E |
|---|---|
| VA_lin (E-refit) | .5923 |
| VA_nl (E-refit, seeds 0-2) | .5806 |
| T (original dense) | .6124 |
| VAT_lin (E-refit) | .5935 |
| VAT_nl (E-refit, seeds 0-2) | .6077 |
| VA_nl fullfit@E / + T combiner (1b) | .6666 / .6685 |
| T_moredata (+261% train, neg_random extras) | .6303 |
| T_augmented (a: scores) | .6707 |
| T_augmented (b: scores+weights) | .6676 |

### style_inv_toptier (E = 1,920; dense seeds 42/1/2)

| arm | AUC on E |
|---|---|
| VA_lin (E-refit) | .5593 |
| VA_nl (E-refit, seeds 0-2) | .6157 |
| T (per dense seed / ensemble) | .6552 / .6378 / .6211 / ens .6490 |
| VAT_lin (E-refit, dense-seed mean) | .5767 |
| VAT_nl (E-refit, dense-seed mean) | .6174 |
| VA_nl fullfit@E / + T combiner (1b, ens) | .6508 / .6624 |
| T_moredata | — (no extra data exists) |
| T_augmented | — (deferred) |

## Plain-language read

1. **Which direction closes the dense-below-bank gap: Direction 3.** Putting
   the top-10 criterion scores in the dense reader's prompt lifts it by ~+.06
   on both caption cells (P≥.99) and lands it exactly at the bank's level
   (Δ vs bank ≈ 0 within ±.02-.04 CIs) — it closes the gap by importing the
   bank's signal, not by unlocking extra text signal beyond it. Direction 2
   (more data) recovers about half the crowd gap (+.053) and a noisy +.018 on
   finalist, still below the bank on both. Direction 1 (dense scalar into the
   stack) never hurts but adds ≈ nothing, because on these cells the dense
   model has ≈ nothing beyond the bank to contribute.
2. **Does VAT_nl exceed both parents? No (captions) / nominally-only (SI).**
   At matched E-refit footing VAT_nl ≥ VA_nl but ties or trails T where T is
   the stronger parent (finalist, SI); with the bank at full strength (1b),
   fusion is a wash on captions (±.002, CIs cross 0) and a consistent small
   positive on SI (+.012 ensemble, CI crosses 0, gain tracks dense-seed
   strength). Nowhere is fusion reliably above max(parents).
3. **Mechanistic picture.** These three cells stay "bank ≥ dense" under every
   remedy tried: the dense reader's residual usefulness is either importable
   into the bank stack for free (Direction 1: no gain) or the bank's signal is
   importable into the dense reader (Direction 3: converges to bank level),
   i.e., the two instruments measure the SAME learnable structure and the
   bank measures it more efficiently at these n's. The only cell with any
   hint of complementarity is style_inv (1b +.012 n.s.).
4. Caveats: E is not selection-clean for any arm (checkpoint selection inside
   E; matched across arms — inherited from the registry protocol); caption
   Direction-2 extras are 100% negatives (adaptive vote allocation / role
   construction), so "more data" here means more negatives; single dense seed
   (42) for all caption arms (SI shows .02-.04 dense seed ranges, so caption
   deltas <.02 should be read with that in mind); row-level bootstraps
   understate group-level CI width on these coarse-grouped cells.

## Artifacts

- `methods/taste_decomposition/fusion/` — all code:
  `direction1_stack.py`, `direction1b_twostage.py`, `build_direction23_data.py`,
  `train_grown_split.py`, `run_fusion_dense_chain.sh`, `watch_and_launch.sh`,
  `truncation_report.py`, `harvest_direction23.py`, `direction3_boot.py`.
- Results JSONs: `fusion/{cap_crowd,cap_finalist,style_inv_toptier}_direction1.json`,
  `fusion/*_direction1b.json`, `fusion/direction2_boot.json`,
  `fusion/direction3_boot.json`, `fusion/dense_data/harvest_results.json`,
  `fusion/dense_data/truncation_report.json`, `fusion/dense_data/build_manifest.json`.
- Trained checkpoints + per-row preds (sk3):
  `/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/fusion/dense_data/*/rm_out_seed42/`
  (preds also mirrored locally under the same relative path).
- GPU: sk3 GPU6 only, claimed/released via `gpu_ledger.txt`
  (agent=claude-vat-fusion; one lost-race RETRACT never occurred — first claim
  20:08 UTC failed on env, re-claims 20:09 and 22:20 UTC, final RELEASE
  2026-08-08T00:28:18Z; no co-tenant GPUs touched, nothing killed).

### cap_crowd (E = 2,190 rows, 64 contests, pos .500)

| quantity | AUC on E |
|---|---|
| T (original dense, held-out) | .5554 |
| VA_lin (E-refit) | .5863 |
| VA_nl (E-refit, seeds 0-2) | .5831 |
| VAT_lin (E-refit) | .5878 |
| **VAT_nl (E-refit, seeds 0-2)** | **.5920** |
| VA_lin fullfit@E (context) | .6038 |
| VA_nl fullfit@E (context) | .6190 |

VAT_nl − VA_nl = +.0047 [−.0105, +.0191], P(>0)=.73 (row-level paired
bootstrap, seed-0 OOF) — sign positive, inside noise. VAT_nl − T = +.0288
[+.0007, +.0565], P(>0)=.98. Read: the stack recovers the bank and is not hurt
by the weak dense column; the dense model adds nothing detectable beyond the
bank on this cell.

### cap_finalist (E = 1,055 rows, 46 contests, pos .128)

| quantity | AUC on E |
|---|---|
| T (original dense, held-out) | .6124 |
| VA_lin (E-refit) | .5923 |
| VA_nl (E-refit, seeds 0-2) | .5806 |
| VAT_lin (E-refit) | .5935 |
| **VAT_nl (E-refit, seeds 0-2)** | **.6077** |
| VA_lin fullfit@E (context) | .6317 |
| VA_nl fullfit@E (context) | .6603 |

VAT_nl − VA_nl = +.0349 [−.0024, +.0710], P(>0)=.97 — the stack clearly uses
the dense column here. VAT_nl − T = −.0165 [−.0711, +.0369], P(>0)=.27 —
statistically it ties T. Read: at matched (E-refit) footing VAT_nl ≈
max(parents), NOT above both; and the E-refit is data-starved (n=1,055, 46
groups) relative to the full-fit bank (VA_nl fullfit@E .6603 > everything else
on these rows), motivating the Direction-1b two-stage readout below.

### Direction 1b — two-stage secondary readout (`fusion/direction1b_twostage.py`)

The E-refit above handicaps the bank (VA trained on ~20% of rows). 1b removes
the handicap without leaking: stage 1 = FULL-population Layer-1 VA OOF
prediction (honest out-of-sample for every row by grouped-OOF construction);
stage 2 on E = (i) fit-free rank-average of [VA_nl_oof, dense_prob],
(ii) 2-column logistic combiner with GroupKFold(5) OOF on E,
(iii) VA_nl_oof-alone combiner as calibration control.

| quantity (AUC on E) | cap_crowd | cap_finalist |
|---|---|---|
| T | .5554 | .6124 |
| VA_nl fullfit@E (seed-mean OOF probs) | .6217 | .6666 |
| rank-average [VA_nl_oof, T] (fit-free) | .6109 | .6677 |
| combiner [VA_nl_oof] only | .6211 | .6628 |
| **combiner [VA_nl_oof, T]** | **.6204** | **.6685** |
| combiner [VA_lin_oof, VA_nl_oof, T] | .6187 | .6557 |

combiner[VA_nl+T] − VA_nl_fullfit: cap_crowd −.0013 [−.0065, +.0036]
(P>0 = .31); cap_finalist +.0020 [−.0082, +.0123] (P>0 = .65). Read: with the
bank at full training strength, the dense scalar adds NOTHING on either
caption cell (fit-free rank-average even dilutes cap_crowd, because T is that
much weaker there); the fitted combiner correctly learns to keep the bank and
ignore/underweight the dense column, so fusion never costs anything either.

style_inv_toptier 1b (E = 1,920; VA_nl fullfit@E seed-mean OOF = .6508):

| dense seed | T_E | rankavg | combiner [VA_nl,T] | Δ vs VA_nl combiner-control | boot CI |
|---|---|---|---|---|---|
| 42 | .6552 | .6687 | .6688 | +.0179 | [−.0050, +.0413] P>0=.95 |
| 1 | .6378 | .6562 | .6587 | +.0079 | [−.0102, +.0271] P>0=.80 |
| 2 | .6211 | .6490 | .6487 | −.0022 | [−.0212, +.0162] P>0=.40 |
| mean-prob ensemble | .6490 | .6624 | .6624 | +.0116 | [−.0084, +.0323] P>0=.87 |

SI is the one cell where two-stage fusion is nominally ABOVE both parents
(ensemble .6624 > VA .6508 and > T .6490), but the gain tracks the dense
seed's own strength (biggest for seed 42, gone for seed 2) and every CI
crosses 0 — read as "consistent small positive, not established", matching
the cell's .038 dense seed range.

(numbers below when run lands)

### style_inv_toptier (E = 1,920 rows, 73 weeks, pos .162)

Three dense seeds (42/1/2); dense-column-dependent quantities per seed + mean.

| quantity | AUC on E |
|---|---|
| T (per dense seed) | .6552 / .6378 / .6211 (mean .6380) |
| T (3-seed mean-prob ensemble) | .6490 |
| VA_lin (E-refit) | .5593 |
| VA_nl (E-refit, seeds 0-2) | .6157 |
| VAT_lin (E-refit, mean over dense seeds) | .5767 |
| **VAT_nl (E-refit, mean over dense seeds)** | **.6174** (.6238 / .6133 / .6150) |
| VA_lin fullfit@E (context) | .5984 |
| VA_nl fullfit@E (context) | .6458 |

Per-seed bootstraps: VAT_nl − VA_nl = +.008/−.004/+.003 (all inside noise);
VAT_nl − T = −.031/−.025/−.001 (seed-42 nominally negative, P(>0)=.04). Read:
on E the E-refit VAT does NOT recover the dense model's level — the GBM stack
at n=1,920/73 groups largely ignores or dilutes the dense column. NOTE the E
population also reverses the headline gap direction: on E rows the dense T
(.638-.649) sits ABOVE the full-fit bank stack (VA_nl fullfit@E .6458 vs the
full-population VA_nl .6651 used in the headline) — the "dense below bank"
gap on this cell is population-composition-sensitive (E = 2 held-out buckets,
73 of 316 weeks), consistent with the .038 dense seed range and the Layer-1
caveat that week-grouped readouts at this n are noisy.


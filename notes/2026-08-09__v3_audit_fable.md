# V3 "criteria-in-prompt" dense arm — audit + maximization (Fable)

Date: 2026-08-08 (filed under the assigned 08-09 name). Descriptive only.
Follows: `notes/2026-08-07__vat_fusion_directions.md` (Direction 3 = the V3
baseline: Llama-3.1-8B LoRA fine-tuned on [full text + top-10 named criterion
scores], which closed the dense-below-bank gap on both caption cells).
Code + artifacts: `methods/taste_decomposition/fusion/v3_audit/` (build,
harvest, bootstrap, importance rankings, results JSONs); new dataset dirs +
checkpoints under `methods/taste_decomposition/fusion/dense_data/` (sk3 +
local mirrors of preds).

Terms, spelled out: **V3 / aug_a** = the criteria-in-prompt dense arm (the
2026-08-07 Direction-3 baseline, k=10 criteria, "name: score" lines appended
after the caption); **E** = evaluation-valid rows = the dense model's own
eval+test buckets pooled (n=2,190 cap_crowd / 1,055 cap_finalist), the only
rows where every dense arm is out-of-sample; **bank** = the cell's full
cleaned V+A criterion matrix under the frozen Layer-1 aggregators (VA_nl =
seed-mean HistGB, VA_lin = logistic); **fullfit@E** = full-population grouped
OOF prediction read on E rows; **T_big** = the original 77K-row wp_clean CW
dense model; **MONITOR/TEST** = the CW closure campaign's held-out splits.
All caption arms: selection_split=test (matching the original cap chains and
the V3 baseline — eval is the selection-clean leg). CW arms:
selection_split=eval (MONITOR), so TEST is the selection-clean leg there.

Frozen constants respected: Llama-8B LoRA recipe (r16/a32, lr 5e-5, bs16,
max_len 1024, 2 epochs, seed 42), grouped splits byte-identical to the
originals, importance rankings computed on training folds only (GroupKFold(3)
permutation importance, frozen HistGB, reproduces the shipped top-10 exactly —
asserted in `build_v3_audit_data.py`), criterion scores are label-blind
Gemma-4-31B judge outputs, y never in the prompt.

## 1. What caps V3 today (code + data audit, before any retrain)

1. **The k=10 block is information-starved, not mis-rendered.** A-criterion
   scores are 3-level {0, 0.5, 1}. The crowd top-10 block collapses to
   **1,194 distinct score-vectors across 10,893 rows** (~9 rows per pattern);
   2-3 of the 10 picks are >=95% modal (e.g. "Late-night talk/variety format"
   is 98.9% zeros — near-zero information), and the picks are a redundant
   clique: 15/45 pairs correlate at rho>=.5, max .84 (Punch placement vs
   Ending buttons). Effective dimensionality ~3-4. Same pattern on finalist
   (4 of 10 picks >=94% modal). At k=40 the block becomes 8,691/8,703
   distinct vectors — the information is in the bank, top-10 just doesn't
   carry it.
2. **Prompt-format / truncation audit: captions clean, long-text cells have a
   landmine.** Caption arms truncate 0% up to k=40 (max 597 tokens at
   max_len 1024). But the trainer + scorer truncate the RIGHT side, and the
   V3 caption format appends the metric block AFTER the text — on CW, where
   35.6% of stories exceed 1024 tokens, that format would silently delete the
   block on exactly the long rows. The CW mirror arms therefore PREPEND the
   block (quantified cost: raw 29.3% rows truncated, +k20 block 46.7%,
   +k40 70.5% — k20 chosen; the block always survives, story tail pays).
3. **No staleness / leakage found.** Scores in the prompt are the same
   single-pass label-blind judge outputs the bank uses (no OOF alignment
   issue — they are not model predictions); eval/test row sets byte-identical
   across arms (asserted); y never appears in any prompt; extras in the
   D2xV3 arm are all scored rows from dense-TRAIN-bucket contests only
   (identical to the shipped Direction-2 extras — asserted).
4. **Score rendering is NOT the cap.** Raw "0 / 0.5 / 1" already is a coarse
   verbal scale; percentile/verbal re-rendering cannot add information to a
   3-level score (dimension 3 of the audit was deprioritized on this measured
   ground, and the definitions arm below confirms the reader doesn't need
   semantic help).
5. **(Observation, recipe frozen.) Selection traces still rising at epoch 2**
   in every caption arm (e.g. k20 best test AUC .604 -> .621 epoch 1 -> 2);
   a longer budget might add a little, untested by design.

## 2. Sweep results (each arm one seed-42 retrain, scored on the SAME rows)

### cap_crowd (sandbox; E = 2,190; baseline V3 = aug_a .6190 on E)

| arm | eval | test | E | Δ vs V3 [95% CI], P(>0) | Δ vs full bank (.6217) |
|---|---:|---:|---:|---|---|
| T original (no block) | .5631 | .5476 | .5554 | — | −.066 |
| V3 baseline aug_a (k10) | .6361 | .6016 | .6190 | — | −.0027, P=.35 |
| **k20** | **.6402** | **.6199** | **.6303** | **+.0113 [+.0007, +.0217], P=.98** | **+.0086, P=.87** |
| k40 | .6385 | .6159 | .6271 | +.0082 [−.0013, +.0181], P=.95 | +.0054, P=.77 |
| k10 + definitions | .6255 | .6091 | .6171 | −.0019 [−.0109, +.0071], P=.34 | −.0046, P=.30 |
| augmd (D2xV3, k10 block + 46% more train) | .6464 | .6080 | .6274 | +.0085 [−.0030, +.0201], P=.93 | +.0057, P=.74 |
| moredata (D2 alone, context) | .6261 | .5912 | .6087 | −.0102, P=.22 | −.0129, P=.17 |

Mechanism control (`bank_topk_oof.json`): the SAME top-k columns under the
bank's own aggregator (fullfit@E, frozen protocol — reproduces the published
full-bank .6217 exactly at k=364) score **.6021 / .6029 / .6173** (nl,
k=10/20/40). The dense reader with the same scores in-prompt gets .6190 /
.6303 / .6271 — **+.017 / +.027 / +.010 above the columns' own carrying
capacity**. The 8B is fusing text signal with the hints, not just importing
the bank.

### cap_finalist (confirm cell; E = 1,055; baseline V3 = aug_a .6707 on E)

| arm | eval | test | E | Δ vs V3 [95% CI], P(>0) |
|---|---:|---:|---:|---|
| T original | .6252 | .6015 | .6124 | — |
| V3 baseline aug_a (k10) | .6775 | .6658 | .6707 | — |
| **k20 (confirm)** | CONFIRM_EVAL | CONFIRM_TEST | CONFIRM_E | CONFIRM_BOOT |

(full bank fullfit@E .6666; top-k-columns-alone controls: k10 .6476,
k20 .6540, k40 .6666)

### Readout conventions

Paired row-level bootstraps (2,000 draws) on identical row sets; single dense
seed 42 everywhere (matched to the V3 baseline; SI showed .02-.04 dense seed
ranges, so single-arm deltas <.02 carry that caveat); E is not
selection-clean for any caption arm (selection inside E, matched across arms,
inherited from the registry protocol) — eval-only is the selection-clean
caption leg and agrees in sign on every claim above.

## 3. D2xV3 interaction: sub-additive

The two 08-07 lifts on cap_crowd were +.064 (V3, block) and +.053 (D2, +46%
all-negative extras). Together (augmd): **.6274 on E = V3 +.0085 (P=.93)**,
nowhere near additive (naive stack would predict ~.67). Against D2 alone it
is +.0187 (P=.97). Read: both interventions import overlapping,
bank-adjacent signal; once the block is present, the extra negatives add
almost nothing (and augmd ties k20, which costs no extra data at all).
Production implication: prefer k20 over data-growing.

## 4. The CW mirror test (dense-STRONG cell): the plateau holds

Setup: CW community honest population (7,008 rows, all held out by T_big),
terminal 144-column bank incl. the closure campaign's surviving mined
criteria (round7_state), FIT+MINE->train (4,794) / MONITOR->eval (1,114,
selection) / TEST->test (1,100, selection-clean). FIT+MINE-only importance;
top-20 block PREPENDED (see §1.2). Both arms same recipe, matched rows.
Note: TEST here is the campaign's twice-read split — this is its third
model-comparison exposure, flagged as the campaign note requires; MONITOR
numbers carry the selection caveat instead. Neither compromises the PAIRED
arm-vs-arm reads, which are what the mirror needs.

| readout | MONITOR | TEST (selection-clean) |
|---|---:|---:|
| T_big (77K-row original) | .7950 | .8048 |
| terminal bank VA_nl (pop_nl, trained on the same 4,794 rows) | .6869 | .7018 |
| top-20 columns alone (fullfit OOF, nl/lin) | .6715 / .6750 | .6734 / .6833 |
| cw_raw (matched-n text-only control) | .7043 | .6772 |
| **cw_augk20 (criteria-in-prompt)** | **.6989** | **.6912** |
| aug − raw, paired boot | −.0055 [−.0330, +.0215], P=.35 | +.0140 [−.0141, +.0417], P=.83 |
| aug − T_big, paired boot | −.0962 [−.1245, −.0668], P=.00 | −.1136 [−.1429, −.0838], P=.00 |
| aug − terminal bank | +.0119, P=.76 | −.0106, P=.28 |

**Verdict: the .805 plateau is robust to articulated hints.** Injecting the
strongest 20 criteria of the terminal (post-campaign) bank into the dense
reader's prompt does not reliably lift it even over its own matched-n
text-only control (+.014 TEST / −.005 MONITOR, both CIs cross 0), and leaves
it .11 below the 77K-row plateau with P(below)=1.00 on both legs. What the
block does do at matched n is close the dense-below-bank gap (raw is −.0246
under the bank on TEST, P=.07 below; aug is −.0106, indistinguishable) —
i.e., the caption result replicates ON THE DENSE-STRONG CELL: criteria-in-
prompt imports the bank's signal into a data-starved dense reader, and does
not unlock text signal beyond what raw-text training at scale already found.
The text-predictability bound does not rise; the taste residual
(Δ_plateau = +.103 on TEST) does not shrink.

Corollary the mirror adds: the CW dense advantage is a TRAINING-SCALE
effect, not an architecture effect. At 4,794 rows the dense reader (.6772
TEST) is BELOW the same-rows bank (.7018) — CW at matched n is a
bank-above-dense cell exactly like the captions; the .805 plateau emerges
only from the 77K-row corpus that the bank never sees.

## 5. Production V3 recipe (recommendation)

One paragraph: keep the V3 architecture exactly as shipped (frozen 8B LoRA
recipe, "full text + name: score" block, label-blind scores, importance
ranked on training folds only) and change three things. (1) **k=20, not
k=10** — the one intervention that survives the bootstrap (+.0113 on E,
P=.98, confirmed direction on the eval-only selection-clean leg), taking the
dense arm nominally above the full bank; k=40 buys nothing more (saturation
between 20 and 40), and if a cell's block is redundancy-heavy prefer
dropping >=95%-modal columns before extending k. (2) **Names only — skip
definitions** (inert, −.002) **and skip score re-rendering** (scores are
already 3-level; there is nothing to re-render). (3) **On long-text cells,
PREPEND the block** — right-side truncation otherwise deletes it on exactly
the long documents (35.6% of CW rows), and budget the block against the
text window (k=20 ≈ 230 tokens is the practical ceiling at max_len 1024).
Do not stack with Direction-2 data growth (sub-additive; k20 alone ties it
without the extra data). And scope the claim honestly: V3 is a
bank-importer for data-starved dense readers — on caption cells it lifts
them to (nominally past) the bank; on a dense-strong cell it neither lifts
the plateau nor shrinks the taste residual.

## Artifacts

- Build/results: `methods/taste_decomposition/fusion/v3_audit/`
  (`build_v3_audit_data.py`, `build_cw_mirror_data.py`, `bank_topk_oof.py` +
  `.json`, `save_bank_oof_probs.py`, `harvest_v3_audit.py`,
  `v3_audit_results.json`, `importance_full_{cap_crowd,cap_finalist,cw_community}.json`,
  chain + confirm runners).
- Datasets + preds: `methods/taste_decomposition/fusion/dense_data/
  {cap_crowd_{k20,k40,k10defs,augmd},cap_finalist_{k20,k40,k10defs},cw_raw,cw_augk20,cw_augk40}/`
  (cw_augk40 built, not trained — 70.5% truncation made it dominated a
  priori; checkpoints on sk3, preds mirrored locally).
- Truncation audit: `fusion/dense_data/truncation_report.json` (regenerated,
  now covers all new arms).
- GPU: sk3 GPU5 only, one GPU throughout, ledger-claimed
  (agent=claude-v3-audit; GPU6 claim RETRACTED before use when a co-tenant
  appeared; chain stacked behind an unledgered 7GB co-tenant process, nothing
  killed).

# Same-rows dense (T) rescore batch — freeze requirement #2

Date: 2026-08-06. Status: DONE for the 6 cells assigned this batch (CW community,
N&C responded/outcome/agree, cap crowd/finalist). Peer-verdict was already done
in the pilot (`methods/taste_decomposition/closure/rescore_dense_same_rows.py`
+ `peer_verdict_dense_preds.report.json`) and is excluded here.

Terms unpacked on first use: **V** = verifiable/surface feature block; **A** =
articulated-criterion (LLM-judged rubric) block; **VA** = V+A concatenated;
**VA_nl** = the nonlinear (gradient-boosting) aggregation of the VA matrix,
mean over seeds {0,1,2} per FREEZE CHANGE 1
(`notes/2026-08-05__taste-decomposition-design.md` S6); **T** = dense-standard
clean-eval AUC (Llama-3.1-8B LoRA reward model, sigmoid head, on raw text);
**Δ_beyond** = T − VA_nl, the only part of the taste residual eligible to be
called taste; **dense_split** = the train/eval/test partition the dense model
itself was fit and selected on; **in_dense_train** = a row fell in the dense
model's own training data, so its per-row score is in-sample/contaminated.

## 0. Motivation (FREEZE CHANGE 2)

The registry's T numbers were computed on each dense model's *own* eval (and
sometimes test) split — not on the exact row population the V/A instrument
scored. Comparing T (one population) against VA_lin/VA_nl (a different,
usually larger population) confounds Δ_beyond with a population mismatch. This
batch rescores each cell's dense `best_model` on the **exact same rows** as its
A/V population, so Δ_beyond is finally an apples-to-apples subtraction.

## 1. Method

Per cell: (1) reconstruct the exact A/V-scored row population from the
matrix/loader the Layer-1 result actually used; (2) join those rows back to
the dense model's own `data.csv` + `split/{train,eval,test}.csv` (same text,
verified byte-identical on samples) to recover `dense_split`; (3) score with
`methods/dense/score_eval_pr_v2.py`'s pattern — Llama-3.1-8B + PEFT LoRA
`best_model`, `num_labels=1`, `sigmoid(logits[:,0])`, batch 16, `max_length`
matching each cell's training recipe (1024 for all six — confirmed from
`train.log` "Training configuration" for the three N&C cells, and from
`methods/dense/score_eval_pass_r2.py`'s hardcoded `MAXLEN=1024` for CW/cap,
which reproduces the registry T for CW to machine precision as a gate).

Population reconstruction per cell (all verified to match the Layer-1 JSON's
`n` exactly, 0 unmatched rows):

- **cw_community**: 2,000 ids from `outputs/va_gemma_banks/creative_writing_shard*.npz`
  (`id = f"{prompt_id}_{sha1(text)[:10]}"`, per
  `datasets/va_gemma_banks/score_va_gemma_banks.py::build_creative`). Joined
  back to `datasets/creative-writing/writingprompts_modeling_clean.csv.gz`
  (96,080-row canonical source) via the same hash scheme to recover
  text/judgement, then to `writingprompts_modeling_clean/{train,eval,test}.csv`
  (same hash scheme) for `dense_split`. Model:
  `datasets/creative-writing/wp_clean_rm_out/best_model`.
  (Note: `datasets/creative-writing/va_bank_v2/sample_manifest.csv.gz`, a
  *different*, unrelated "reconstruction" artifact with non-matching doc-id
  scheme, was correctly NOT used — the Layer-1 JSON's own `matrix` field is
  the authority.)
- **cap_finalist / cap_crowd**: ids from `role_by_id`/`crowd_ids` /
  `hardneg_ids` built by
  `datasets/humor/caption_multiy/aggregate_captions_multiy.py` (`did =
  f"{contest}_{sha1(text)[:12]}"`), joined by `(contest, text)` — no
  collisions — to
  `datasets/caption_contest/dense_llama/{finalist,crowd}/data.csv` +
  `split/*.csv`. n matched the dense model's own total (data.csv) exactly:
  5,218 / 10,893 rows.
- **nc_responded / nc_outcome / nc_agree**: rows are literally
  `datasets/notice-and-comment/v4/dense_llama/{responded,outcome,agree}/data.csv`
  (9,521 / 7,084 / 5,046 rows — exactly the Layer-1 `n`). `doc_id` recovered
  by joining `text` to `nc_vat_sample.jsonl` ∪ `nc_unmatched_sample.jsonl`
  (0 unmapped). `dense_split` via `(docket, text)` join to each cell's own
  `split/*.csv`.

Landmine avoided: `datasets/humor/caption_multiy/` and most of
`datasets/notice-and-comment/v4/` (jsonl/npz sources) exist **only on the
local Mac clone**, not on sk3 — sk3 only has `dense_llama/**` for those two
tasks. All population reconstruction that needed the jsonl/npz sources was
done locally (byte-identical file hashes confirmed against sk3 for the N&C
`dense_llama` CSVs before joining), then just the resulting (id, text,
judgement, dense_split) population CSVs were shipped to sk3 for GPU scoring.

GPU: sk3, one card only. Picked GPU1 (0 MiB / 0% util, re-verified
immediately before launch) via `nvidia-smi`; job ran start-to-finish (6
cells, ~14 min total) without touching any other card. Nothing killed.

## 2. Headline table

Per-row `in_dense_train` is set for every row; the headline same-rows T is the
**dense-held-out subset only** (`dense_split ∈ {eval, test}`), n reported.
"registry T" is the number each cell previously quoted (computed on the dense
model's own eval-only, or eval+test, population — NOT the same rows as V/A).

| cell | registry T | same-rows T (held-out) | n held-out | n population | train-overlap |
|---|---|---|---|---|---|
| cw_community | .7801 (eval-only, n=9,573) | **.7967** | 408 | 2,000 | .796 |
| cap_finalist | .6252 (eval-only, n=528) | **.6124** | 1,055 | 5,218 | .798 |
| cap_crowd | .5631 (eval-only, n=1,098) | **.5554** | 2,190 | 10,893 | .799 |
| nc_responded | eval .808 / test .825 | **.8167** | 1,904 | 9,521 | .800 |
| nc_outcome | eval .622 / test .623 | **.6238** | 1,417 | 7,084 | .800 |
| nc_agree | eval .566 / test .639 | **.6034** | 1,009 | 5,046 | .800 |

Skip list: **none** — all 6 assigned cells resolved (model + row-mapping both
located, gate-quality join: 0 unmatched rows on every cell). peer_verdict was
already done pre-batch and is excluded per the assignment.

## 3. Δ_beyond on matched populations (context, not requested but load-bearing)

`Δ_beyond_samerows = T_heldout_samerows − VA_nl_mean` (VA_nl_mean = mean over
seeds {0,1,2}, per FREEZE CHANGE 1, pulled from each `*_layer1.json`):

| cell | VA_nl_mean | T_heldout_samerows (n) | Δ_beyond_samerows |
|---|---|---|---|
| cw_community | .6207 | .7967 (408) | **+.176** |
| cap_finalist | .6800 | .6124 (1,055) | **−.068** |
| cap_crowd | .6656 | .5554 (2,190) | **−.110** |
| nc_responded | .7244 | .8167 (1,904) | **+.092** |
| nc_outcome | .6102 | .6238 (1,417) | **+.014** |
| nc_agree | .5844 | .6034 (1,009) | **+.019** |

cap_finalist and cap_crowd remain negative-Δ cells (VA_nl exceeds T on the
matched population too — consistent with cap_crowd's pre-registered role as
the negative-Δ control, design doc S4). nc_outcome/nc_agree's residuals are
small (~.01-.02) and inside VA_nl's per-seed spread reported in their
Layer-1 JSONs — read these as "no reliable Δ_beyond," not zero. cw_community
and nc_responded clear the Layer-3 gate (Δ_beyond > .02) on the matched
population too, same conclusion as the pre-freeze numbers.

## 4. Caveats

- **N&C `selection_split="test"`**: all three N&C dense chains selected their
  checkpoint on `test`, not `eval` (confirmed in each cell's `train.log`
  "Training configuration" line). So for N&C, `eval` is the *only* clean
  (selection-uncontaminated) held-out leg; `test` was used for model
  selection. The held-out numbers above pool eval+test per the task spec, but
  the per-split breakdown (`auc_eval_only` / `auc_test_only`) is in each
  cell's JSON for anyone who wants the eval-only, fully-clean number.
- CW/cap held-out populations here are **larger** than the historic
  registry-T eval-only populations (they include `test` too, and the A/V
  population itself is a different draw from the full dense dataset) — the
  registry T and same-rows T are not expected to match exactly, and don't
  (e.g. cw_community .7801 vs .7967); both numbers are reported, not
  reconciled.
- `train_overlap_frac` ≈ .80 uniformly (all six dense chains used a ~80/10/10
  split) — 20% of each population is genuinely held out.

## 5. Artifacts

- Per-cell reports (this batch): `methods/taste_decomposition/results/samerows_T_{cw_community,cap_finalist,cap_crowd,nc_responded,nc_outcome,nc_agree}.json`
  (each has `auc_all_CONTAMINATED`, `auc_dense_heldout`, `n_dense_heldout`,
  `auc_{train,eval,test}_only`, `train_overlap_frac`, `preds_path_sk3`,
  `preds_path_local`).
- Consolidated: `methods/taste_decomposition/results/samerows_T_all_cells.json`.
- Per-row predictions (id/docket-or-contest-or-prompt_id/judgement/
  dense_split/in_dense_train/dense_prob, text column stripped for size):
  `methods/taste_decomposition/closure/samerows_preds/{cell}_dense_preds_slim.csv`.
- Full per-row predictions with text, on sk3 only (not copied down, large):
  `/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/samerows_scratch/{cell}_dense_preds.csv`.
- Population-reconstruction CSVs + the driver script, on sk3:
  `/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/samerows_scratch/`
  (`{cell}_population.csv`, `score_samerows_dense.py`, `samerows_run.log`).
- Peer-verdict (already done, excluded from this batch):
  `methods/taste_decomposition/closure/rescore_dense_same_rows.py` +
  `peer_verdict_dense_preds.report.json`.

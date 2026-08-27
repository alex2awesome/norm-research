# patents claim-fell — round-0 audit runbook

Campaign status: **STOPPED at round 0** (leak post-mortem). See
`notes/2026-08-07__closure_patents.md` for the verdict and every number.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` (FREEZE DECLARATION + ADDENDA 1–3).

Everything here is CPU-only except the two ablation passes. All paths are sk3 paths;
`export HOME=/lfs/skampere3/0/alexspan` first, every time.
Python: `/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python`.

## Exact command sequence (as run, 2026-08-07)

```bash
export HOME=/lfs/skampere3/0/alexspan
cd $HOME/norm-research/methods/taste_decomposition/closure/patents
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python

# --- CPU legs (minutes each) ------------------------------------------------
$PY round0_audit_cpu.py                   # -> round0_audit_cpu.json, round0_manual_read.json
$PY round0_gap_decomposition.py           # -> round0_gap_decomposition.json, round0_gap_scores.npz
$PY round0_ydef_probe.py                  # -> round0_ydef_probe.json
$PY round0_mixed_decomposition.py         # -> round0_mixed_decomposition.json
$PY round0_fixed_label_counterfactual.py  # -> round0_fixed_label_counterfactual.json

# --- GPU legs (claim a free card on gpu_ledger.txt FIRST) -------------------
$PY round0_swap_ablation.py  --gpu 3 --split eval --batch 32   # the FAIR mechanism test
$PY round0_ablation_score.py --gpu 3 --split eval --batch 48   # deletion ablation (confounded)

# --- dense seeds 1-2 (~5h each, chained, resumable) -------------------------
cd $HOME/norm-research/methods/dense
nohup env GPU=3 bash run_patents_seeds12.sh > logs/patents_seeds12.log 2>&1 < /dev/null &
disown
```

`round0_audit_cpu.py` must run first: the other CPU legs assume the same
jsonl-to-split alignment it asserts (0 unmatched rows, 0 label mismatches).

## Prerequisites the CPU legs assume

| input | path |
|---|---|
| source corpus (224 MB, sk3 only) | `datasets/patents/processed/option3_claims_gemma_scale.jsonl` |
| dense build + splits + seed-42 predictions | `datasets/patents/dense_standard/` |
| V/A feature matrix | `notebooks/data/patents_va_features.csv` |
| technology class (CPC), 100% app_id coverage | `/lfs/skampere3/0/alexspan/tmp/appid_probe/app_cpc.json` |
| USPC class, art unit, filing date | `/lfs/skampere3/0/alexspan/tmp/appid_probe/patex_join.json` |
| exact filing year | `datasets/patents/processed/labels.parquet` |

The two `tmp/appid_probe/*.json` caches are derived files. To rebuild: join
`app_id → pgpub_id` through `datasets/patents/raw/patentsview_pg/pg_published_application.tsv.zip`,
then `pgpub_id → CPC` through `pg_cpc_current.tsv.zip` keeping `cpc_type == "inventional"`
and the lowest `cpc_sequence`; USPC/art-unit come straight from
`datasets/patents/raw/patex/application_data.csv` on `application_number`.

## Standing landmines on this corpus

1. **`rejection_type` is the label.** `NEG` ⇔ y = 0 (alone-AUC .988). It is a sidecar
   column in `dense_standard/split/*.csv`. Never expose it to a model, a judge prompt,
   a proposer context, or a feature block.
2. **`gold_disclose` is construction-conditional.** It is 0 for every negative by
   construction; its .659 alone-AUC is arithmetic, not validity. Keep it out of
   `A_ONLY_COLS`.
3. **The candidate-reference set is label-conditional.** Positives carry the examiner's
   gold reference (99.66%, in the last slot 86.6% of the time); negatives carry none
   (0.00%). Any "identify the disclosing reference among K" evaluation on this corpus is
   invalid without re-randomising slot order and constructing negatives symmetrically.
4. **`claim_num` and the "of claim N" preamble** carry .754/.751 alone. Any criterion,
   feature, or judge prompt that can see claim position is reading the dominant nuisance
   channel.
5. **12.2% exact-duplicate rows** exist within applications (0 across splits — verified
   here). Dedup before any counting statistic; grouped splits already contain them.

## If the cell is ever revived

Do all four before mining, not one:

1. restrict the label to §102/§103 (necessary, **not sufficient** — see §4b of the note);
2. rebuild the candidate sets symmetrically across labels, with randomised slot order;
3. build a real A bank from `datasets/patents/online-rubrics/` — the current bank is
   **one** concept in four aggregations;
4. bank claim ordinal position as a declared Track-B nuisance channel at round 0, and
   quote Δ over V + A + STRUCT, never over V + A.

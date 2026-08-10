# DEPRECATED: nc_outcome_va_nl_oof_{seed0,mean3}.npy row order is unrecoverable

Date flagged: 2026-08-10 (registry LANDMINE). Regenerated: 2026-08-08 (see below).

## Why these two files are frozen in place, never overwritten, never deleted

`nc_layer1_stack.py`'s `NCData.rows_for("outcome")` built the row order by
iterating `self.valid_out` -- a **Python set** (`{did for did, y in
self.y_out_by_id.items() if y in (0, 1)}`) -- directly. CPython randomizes
string hashing per process (`PYTHONHASHSEED`), so a `set`'s iteration order
is not stable across separate process launches even though the set's
*contents* are identical every time. No ids vector was saved alongside
`nc_outcome_va_nl_oof_seed0.npy` / `nc_outcome_va_nl_oof_mean3.npy` at save
time, so there is no way to recover which row belongs to which `doc_id`
after the fact. Confirmed empirically: running `nc_outcome` through its own
canonical adapter (`closure/maps_batch1/cells.py`, which already used
`sorted(valid_out)`) and gating `AUC(y, oof)` against the published
nonlinear VA seed-0 AUC (.6102) scored .5083 (seed0 array: .5093) --
indistinguishable from a random permutation, not a near-miss.

Full incident record: `notes/2026-07-27__vat-run-registry.md`, entry
"2026-08-10 -- LANDMINE (permanent): N&C LAYER-1 OOF ROW ORDER UNRECOVERABLE",
and the discovery narrative in `notes/2026-08-10__vat_fullgrid.md` ("LANDMINE
FOUND" section).

## Per the standing rule (never delete data), these files are NOT deleted

They stay on disk exactly as produced. Do not attempt to re-derive their row
order. Do not join them to any external table by position.

## Replacement (regenerated, ids-carried, gate-certified)

`nc_layer1_stack.py`'s `rows_for()` is patched to iterate `sorted(ids)`
(never a raw set/dict) -- the same deterministic-keyed-order convention
`closure/{maps_batch1,peer_curation_ext,peer_revealed}/cells.py::_load_nc`
already used for this exact cell. The regenerated, row-order-safe artifacts
are:

  - `nc_outcome_va_nl_oof_seed{0,1,2}_v2.npy` -- per-seed OOF probabilities
  - `nc_outcome_oof_ids_v2.npy` -- doc ids, same row order, aligned 1:1
  - `nc_outcome_layer1_v2.json` -- full run output (linear gate vs
    `datasets/notice-and-comment/v4/nc_multiy_results.json`, nonlinear seed
    AUCs = the new gateable reference)
  - master summary: `methods/taste_decomposition/results/nc_oof_regeneration.json`

Use the `_v2` files for any future row-wise join. See
`methods/taste_decomposition/results/nc_oof_regeneration.json` for the
verification record (AUC-reproduction assert + 3-row spot check) and the
consumer audit of the old files (verdict: no exposed consumers found --
every reader either recomputes VA_nl in-process with a deterministic sorted
row order, reads only pooled/aggregate AUC scalars, or gates the row-order
reproduction and refuses to difference on failure).

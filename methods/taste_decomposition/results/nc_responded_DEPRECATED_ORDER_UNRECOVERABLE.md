# DEPRECATED: nc_responded_va_nl_oof_{seed0,mean3}.npy row order is unrecoverable

Date flagged: 2026-08-10 (registry LANDMINE). Regenerated: 2026-08-08 (see below).

## Why these two files are frozen in place, never overwritten, never deleted

`nc_layer1_stack.py`'s `NCData.rows_for("responded")` originally built the
"responded" cell's `X_u` (unmatched) universe iteration straight off a Python
dict built from `X_u`/`X_m` insertion order and, more importantly, the
`outcome`/`agree` sibling cells' `valid_out`/`valid_agr` **Python sets**
directly -- the whole `rows_for()` method shared one un-audited assumption
that "iterate the collection I built" gives a stable row order. It does not:
CPython randomizes string hashing per process (`PYTHONHASHSEED`), so the
`responded` cell's array order is not certified stable across the two
`np.save` calls that produced `nc_responded_va_nl_oof_seed0.npy` and
`nc_responded_va_nl_oof_mean3.npy` versus any later process that tries to
reconstruct "the same" order to join these arrays to an external id list.
No ids vector was saved alongside either file at the time, so there is no
way to recover, after the fact, which row of either array belongs to which
`doc_id`. Reordering by guesswork (five candidate orderings were tried
program-wide across the three N&C cells) gates at ~.50 AUC against the
published nonlinear VA seed-0 AUC instead of reproducing it -- i.e. every
guess is statistically indistinguishable from a random permutation.

Full incident record: `notes/2026-07-27__vat-run-registry.md`, entry
"2026-08-10 -- LANDMINE (permanent): N&C LAYER-1 OOF ROW ORDER UNRECOVERABLE".

## Per the standing rule (never delete data), these files are NOT deleted

They stay on disk exactly as produced. Do not attempt to re-derive their row
order. Do not join them to any external table by position.

## Replacement (regenerated, ids-carried, gate-certified)

`nc_layer1_stack.py`'s `rows_for()` is patched to iterate `sorted(ids)`
(never a raw set/dict) -- the same deterministic-keyed-order convention
`closure/{maps_batch1,peer_curation_ext,peer_revealed}/cells.py::_load_nc`
already used. The regenerated, row-order-safe artifacts are:

  - `nc_responded_va_nl_oof_seed{0,1,2}_v2.npy` -- per-seed OOF probabilities
  - `nc_responded_oof_ids_v2.npy` -- doc ids, same row order, aligned 1:1
  - `nc_responded_layer1_v2.json` -- full run output (linear gate vs
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

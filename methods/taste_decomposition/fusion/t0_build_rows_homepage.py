#!/usr/bin/env python3
"""T0 (UNTRAINED-T) ARM, step 1 for the homepage cell -- a POST-HOC ADDITION.

The standing 16-cell builder (`t0_build_rows.py`) rebuilds each cell's E population by
importing that cell's own master-ledger loader and asserts n_E against
`results/vat_fullgrid_<cell>.json`. The homepage cell has no `vat_fullgrid_*` entry (it
was never in the full-grid battery), so this dedicated builder exists. It writes the
IDENTICAL file layout that `t0_score_vllm.py` consumes -- `t0_rows/<cell>.npz`,
`<cell>.texts.jsonl.gz`, `<cell>.meta.json` -- with the same `uid` convention
(`{position:06d}|{natural_id}`) and the same "row ORDER is the contract" rule.

DEVIATIONS FROM THE STANDING BUILDER, all deliberate and recorded in the meta:
  * no `vat_fullgrid_*` assertion (no such file for this cell); in its place the builder
    asserts (a) the dense preds/split alignment gate, (b) that every E row id is present
    in the A-bank matrix, and (c) that T recomputed on E matches the value recorded in
    `results/samerows_T_homepage_storygrouped.json`.
  * E = the dense arm's OWN held-out rows (split in {eval,test}) of
    `dense_standard_storygrouped`, i.e. 1,313 + 1,318 = 2,631 rows.

TEXT is the document the trained dense model read, taken verbatim from the dense chain's
own split CSVs -- not reassembled.

CPU only, read-only w.r.t. every existing result file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
REPO = TD.parents[1]
OUT = HERE / "t0_rows"
OUT.mkdir(exist_ok=True)
RESULTS = TD / "results"

sys.path.insert(0, str(TD))
import homepage_v2_layer1 as HL  # noqa: E402  (bank + dense loaders, imported not copied)

CELL = "homepage_curation_storygrouped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    npz_p = OUT / f"{CELL}.npz"
    if npz_p.exists() and not a.force:
        print(f"[t0-rows] {npz_p} exists; use --force to rebuild")
        return

    meta, A, V, groups, shard_of, ids = HL.load_bank()
    y_all = np.array(meta["ys"]["top_half_placement"], dtype=int)
    id_pos = {d: i for i, d in enumerate(ids)}

    dense, checks = HL.load_dense_rows()
    assert all(c["n_match"] and c["group_seq_match"] and c["y_seq_match"]
               for c in checks), "dense preds/split alignment gate FAILED"

    rows = []
    for split in ("eval", "test"):
        sdf = pd.read_csv(HL.DENSE_DIR / "split" / f"{split}.csv")
        for i, rid in enumerate(sdf["row_id"].astype(str)):
            rows.append({"row_id": rid, "split": split,
                         "text": str(sdf["text"].iloc[i]),
                         "y": int(sdf["judgement"].iloc[i]),
                         "dense": float(dense[split]["mean3"][i])})

    missing = [r["row_id"] for r in rows if r["row_id"] not in id_pos]
    assert not missing, f"{len(missing)} E rows absent from the A-bank matrix"

    pos = np.array([id_pos[r["row_id"]] for r in rows])
    y = np.array([r["y"] for r in rows], dtype=int)
    assert (y == y_all[pos]).all(), "y disagrees between the dense split and the bank"

    d_col = np.array([r["dense"] for r in rows], dtype=float)
    VA_raw = np.column_stack([V[pos], A[pos]])
    grp = np.array([str(groups[i]) for i in pos], dtype=object)
    texts = [r["text"] for r in rows]
    uids = [f"{i:06d}|{r['row_id']}" for i, r in enumerate(rows)]

    T_here = float(roc_auc_score(y, d_col))
    sr = json.loads((RESULTS / "samerows_T_homepage_storygrouped.json").read_text())
    T_eval_recorded = sr["eval_mean"]
    T_eval_here = float(np.mean([
        roc_auc_score(dense["eval"]["y"], dense["eval"]["per_seed"][s])
        for s in HL.DENSE_SEEDS]))
    assert abs(round(T_eval_here, 4) - T_eval_recorded) < 5e-5, (
        f"T on eval {T_eval_here:.6f} disagrees with the recorded {T_eval_recorded}")

    np.savez_compressed(npz_p, ids=np.array([r["row_id"] for r in rows], dtype=object),
                        uids=np.array(uids, dtype=object), y=y, groups=grp,
                        dense=d_col, VA_raw=VA_raw,
                        split=np.array([r["split"] for r in rows], dtype=object))
    with gzip.open(OUT / f"{CELL}.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for u, t in zip(uids, texts):
            fh.write(json.dumps({"uid": u, "text": t}) + "\n")

    ids_sha = hashlib.sha256("\n".join(r["row_id"] for r in rows).encode()).hexdigest()
    txt_sha = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    metaj = {
        "cell": CELL, "n_E": len(rows), "n_groups_E": int(len(set(grp.tolist()))),
        "pos_rate_E": float(y.mean()),
        "family": "impute_perfold",
        "group_column": "snapshot_id (STORY-GROUPED)",
        "population": "dense-held-out rows (split in {eval,test}) of "
                      "datasets/news-homepages/va/dense_standard_storygrouped; joined to "
                      "the A bank BY row_id",
        "n_features_VA_raw": int(VA_raw.shape[1]),
        "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1])},
        "T_recomputed_on_E_pooled_eval_plus_test": T_here,
        "T_eval_recomputed": T_eval_here,
        "T_eval_recorded_in_samerows_json": T_eval_recorded,
        "dense_column": f"mean of seeds {','.join(HL.DENSE_SEEDS)}",
        "split_counts": {s: int(sum(1 for r in rows if r["split"] == s))
                         for s in ("eval", "test")},
        "n_duplicate_natural_ids": len(rows) - len(set(r["row_id"] for r in rows)),
        "ids_sha256": ids_sha, "texts_sha256": txt_sha,
        "n_empty_texts": sum(1 for t in texts if not t.strip()),
        "text_char_len": {"min": int(min(len(t) for t in texts)),
                          "median": float(np.median([len(t) for t in texts])),
                          "max": int(max(len(t) for t in texts))},
        "text_source": str(HL.DENSE_DIR / "split/{eval,test}.csv"),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deviations_from_standing_builder": [
            "no vat_fullgrid_* assertion -- this cell has no full-grid ledger entry",
            "E asserted instead against the dense preds/split alignment gate, A-bank id "
            "containment, and the T recorded in samerows_T_homepage_storygrouped.json"],
        "dense_alignment_checks": checks,
    }
    (OUT / f"{CELL}.meta.json").write_text(json.dumps(metaj, indent=1))
    print(f"[t0-rows] {CELL}: n_E={len(rows)} groups={metaj['n_groups_E']} "
          f"pos={y.mean():.4f} T_on_E={T_here:.4f} (eval {T_eval_here:.4f})")
    print(f"[t0-rows] wrote {npz_p}, texts, meta")


if __name__ == "__main__":
    main()

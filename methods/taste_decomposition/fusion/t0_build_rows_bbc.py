#!/usr/bin/env python3
"""T0 (UNTRAINED-T) ARM, step 1 for the bbc_mostread cell -- a POST-HOC ADDITION.

Same role as `t0_build_rows_homepage.py`: the standing 16-cell builder asserts n_E
against `results/vat_fullgrid_<cell>.json`, which this cell (closure campaign
2026-08-13, terminal by cap) does not have.  This builder writes the IDENTICAL file
layout `t0_score_vllm.py` consumes -- `t0_rows/bbc_mostread.npz`,
`.texts.jsonl.gz`, `.meta.json` -- with the same uid convention
(`{position:06d}|{natural_id}`) and the same "row ORDER is the contract" rule.

DEVIATIONS, all deliberate and recorded in the meta:
  * no `vat_fullgrid_*` assertion; in its place (a) round0_bbc.dense_join's own
    order-join gate (judgement+group sequence match, shuffled counterfactual),
    (b) every E row id present in the scaleupC A-bank matrix, (c) y agreement
    between the dense split CSVs and the population.
  * E = the dense arm's OWN held-out rows (eval + test) of
    datasets/bbc-mostread/va/dense_standard_bbc_mostread.

TEXT is the document the trained dense model read, verbatim from the dense chain's
own split CSVs.  CPU only, read-only w.r.t. every existing result file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
OUT = HERE / "t0_rows"
OUT.mkdir(exist_ok=True)
BBC = TD / "closure" / "bbc_mostread"
sys.path.insert(0, str(BBC))
sys.path.insert(0, str(TD))
import round0_bbc as R0  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

CELL = "bbc_mostread"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    npz_p = OUT / f"{CELL}.npz"
    if npz_p.exists() and not a.force:
        print(f"[t0-rows] {npz_p} exists; use --force to rebuild")
        return

    pop = pd.read_csv(R0.VA_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    id_pos = {r: i for i, r in enumerate(pop.row_id)}

    dj = R0.dense_join(pop)  # raises on any order-join failure (gate with teeth)
    assert dj["report"]["passes"]
    hold_ids = dj["row_ids"]
    leg = dj["leg"]
    dense_mean = np.mean([dj["dense_per_seed"][s] for s in R0.SEEDS], axis=0)

    meta_b, A, V, groups_b, shard, ids_b = SC.load_scaleupC_bank(CELL, out=R0.BANK_OUT)
    ids_b = [str(i) for i in ids_b]
    assert ids_b == pop.row_id.tolist(), "bank rows not aligned with population"
    VA_all = np.column_stack([V, A])

    texts_by_id, y_split = {}, {}
    for lg in ("eval", "test"):
        sdf = pd.read_csv(R0.DENSE / "split" / f"{lg}.csv")
        for rid, tx, yy in zip(sdf["row_id"].astype(str), sdf["text"], sdf["judgement"]):
            texts_by_id[rid] = str(tx)
            y_split[rid] = int(yy)

    missing = [r for r in hold_ids if r not in id_pos]
    assert not missing, f"{len(missing)} E rows absent from the population/bank"
    pos = np.array([id_pos[r] for r in hold_ids])
    y = pop.judgement.astype(int).values[pos]
    assert all(y_split[r] == yv for r, yv in zip(hold_ids, y)), \
        "y disagrees between the dense split CSVs and the population"
    grp = pop.group.astype(str).values[pos]
    VA_raw = VA_all[pos]
    texts = [texts_by_id[r] for r in hold_ids]
    uids = [f"{i:06d}|{r}" for i, r in enumerate(hold_ids)]

    T_pooled = float(roc_auc_score(y, dense_mean))
    print(f"[t0-rows] {CELL}: n_E={len(y)} T(seedmean, pooled eval+test)={T_pooled:.4f}")

    np.savez_compressed(npz_p, ids=np.array(hold_ids, dtype=object),
                        uids=np.array(uids, dtype=object), y=y,
                        groups=np.array(grp, dtype=object),
                        dense=dense_mean, VA_raw=VA_raw,
                        split=np.array(leg, dtype=object))
    with gzip.open(OUT / f"{CELL}.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for u, t in zip(uids, texts):
            fh.write(json.dumps({"uid": u, "text": t}) + "\n")

    metaj = {
        "cell": CELL, "n_E": int(len(y)),
        "n_groups_E": int(len(set(grp.tolist()))),
        "pos_rate_E": float(y.mean()),
        "family": "impute_perfold",
        "group_column": "capture_day",
        "population": ("dense-held-out rows (eval+test) of datasets/bbc-mostread/va/"
                       "dense_standard_bbc_mostread; joined to the scaleupC A-bank "
                       "BY row_id; dense = seed-mean over rm_out_seed{42,1,2}"),
        "n_features_VA_raw": int(VA_raw.shape[1]),
        "T_recomputed_on_E_pooled": T_pooled,
        "dense_join_gate": dj["report"]["legs"],
        "ids_sha256": hashlib.sha256("\n".join(hold_ids).encode()).hexdigest(),
        "texts_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        "deviations": ("post-hoc cell (no vat_fullgrid entry); gates = dense_join "
                       "order-proof + bank-id coverage + split-vs-population y equality"),
    }
    (OUT / f"{CELL}.meta.json").write_text(json.dumps(metaj, indent=1))
    print(f"[t0-rows] wrote {npz_p.name}, texts, meta")
    print("T0_ROWS_BBC_DONE")


if __name__ == "__main__":
    main()

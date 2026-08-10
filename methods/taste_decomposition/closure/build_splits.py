#!/usr/bin/env python3
"""Layer-3 articulation-closure, Stage 1: splits + population export (peer VERDICT).

Prereg: notes/2026-08-05__layer3-closure-prereg.md
  * Population = the 6,030-row peer-verdict A/V evaluation population (vat_3y),
    i.e. rows of datasets/peer-review/vat_3y/verdict.jsonl whose `ntitle` is in
    union_scores.npz and whose `judgement` is a valid 0/1 -- IDENTICAL row set and
    row ORDER to methods/taste_decomposition/layer1_stack.py::load_cell("verdict").
  * FIT+MINE (80%) / MONITOR (20%): stable sha256 hash of the group key (`ntitle`),
    threshold at .80.  No seeded shuffles (BEST-PRACTICES stable-hash rule).
  * Mining slice M = FIT+MINE rows that are ALSO held-out for the dense model
    (dense eval or test split), so the dense score on them is honest.

Writes:
  closure/peer_verdict_splits.json   split map + summary counts
  closure/peer_verdict_population.csv  text/judgement/ntitle for the dense rescore
                                       (uploaded to sk3; freeze change #2 same-rows T)

CPU only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VAT = REPO / "datasets" / "peer-review" / "vat_3y"
DENSE = VAT / "dense_llama" / "verdict" / "split"

THRESH = 0.80


def hash_unit(key: str) -> float:
    """Stable sha256 -> [0,1). Never a seeded shuffle."""
    h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(h, 16) / float(1 << 256)


def _valid_y(r) -> bool:
    j = r.get("judgement")
    try:
        return int(float(j)) in (0, 1)
    except (TypeError, ValueError):
        return False


def load_population():
    z = np.load(VAT / "union_scores.npz", allow_pickle=True)
    nt_set = {str(s) for s in z["ntitle"]}
    rows = [json.loads(l) for l in open(VAT / "verdict.jsonl") if l.strip()]
    R = [r for r in rows if str(r.get("ntitle")) in nt_set and _valid_y(r)]
    return R


def main():
    R = load_population()
    ntl = [str(r["ntitle"]) for r in R]
    y = np.array([int(float(r["judgement"])) for r in R])

    # dense-model split membership, keyed by the dense `group` column == ntitle
    dense_of = {}
    for s in ("train", "eval", "test"):
        df = pd.read_csv(DENSE / f"{s}.csv")
        for g in df["group"].astype(str):
            dense_of[g] = s

    hv = np.array([hash_unit(k) for k in ntl])
    split = np.where(hv < THRESH, "fit_mine", "monitor")
    dsplit = np.array([dense_of.get(k, "unmapped") for k in ntl])
    dense_heldout = np.isin(dsplit, ["eval", "test"])
    mining = (split == "fit_mine") & dense_heldout

    recs = [
        {
            "i": int(i),
            "id": R[i].get("id"),
            "ntitle": ntl[i],
            "split": str(split[i]),
            "dense_split": str(dsplit[i]),
            "in_mining_slice": bool(mining[i]),
        }
        for i in range(len(R))
    ]

    summary = {
        "population_n": len(R),
        "n_groups": int(len(set(ntl))),
        "pos_rate": float(y.mean()),
        "hash": "sha256(ntitle) / 2**256 < 0.80 -> fit_mine",
        "counts": {
            "fit_mine": int((split == "fit_mine").sum()),
            "monitor": int((split == "monitor").sum()),
        },
        "groups": {
            "fit_mine": int(len({k for k, s in zip(ntl, split) if s == "fit_mine"})),
            "monitor": int(len({k for k, s in zip(ntl, split) if s == "monitor"})),
        },
        "pos_rate_by_split": {
            s: float(y[split == s].mean()) for s in ("fit_mine", "monitor")
        },
        "dense_split_counts": {
            s: int((dsplit == s).sum()) for s in ("train", "eval", "test", "unmapped")
        },
        "dense_heldout_n": int(dense_heldout.sum()),
        "mining_slice_n": int(mining.sum()),
        "monitor_and_dense_heldout_n": int(((split == "monitor") & dense_heldout).sum()),
        "monitor_and_dense_train_n": int(((split == "monitor") & (dsplit == "train")).sum()),
    }

    out = {"summary": summary, "rows": recs}
    (HERE / "peer_verdict_splits.json").write_text(json.dumps(out, indent=1))

    pd.DataFrame(
        {
            "i": np.arange(len(R)),
            "text": [r["text"] for r in R],
            "judgement": y,
            "ntitle": ntl,
            "split": split,
            "dense_split": dsplit,
        }
    ).to_csv(HERE / "peer_verdict_population.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

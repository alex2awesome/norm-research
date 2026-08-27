#!/usr/bin/env python3
"""Write cap_crowd_population.csv -- the row file score_gemma_maps.py reads on sk3.

IDENTICAL rows and row ORDER to maps_batch1/cap_crowd_population.csv (asserted here,
so rounds 1-2 and rounds 3+ are scoring the same population in the same order), with one
column ADDED: `desc`, the cartoon description.  That column is the ITEM-VIEW FIX -- see
cells.py, "ITEM-VIEW DEFECT".

CPU only.  Usage: python build_population.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import cells as C

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "maps_batch1" / "cap_crowd_population.csv"


def main():
    d = C.load("cap_crowd")
    sp = json.loads((HERE / "cap_crowd_splits.json").read_text())
    rows_sp = sp["rows"]
    assert len(rows_sp) == len(d["y"])

    out = []
    for i in range(len(d["y"])):
        assert str(rows_sp[i]["id"]) == str(d["ids"][i]), f"split row {i} id mismatch"
        out.append({
            "i": i, "id": d["ids"][i], "text": d["texts"][i], "desc": d["descs"][i],
            "judgement": int(d["y"][i]), "group": d["groups"][i],
            "split": rows_sp[i]["split"], "dense_split": d["dense_split"][i],
        })

    # ---- continuity assertion against the round-1/2 population file ------------
    with open(LEGACY, newline="") as fh:
        legacy = list(csv.DictReader(fh))
    assert len(legacy) == len(out), f"row count {len(legacy)} != {len(out)}"
    for a, b in zip(legacy, out):
        for k in ("i", "id", "text", "judgement", "group", "split", "dense_split"):
            assert str(a[k]) == str(b[k]), f"row {a['i']} column {k}: {a[k]!r} != {b[k]!r}"
    print(f"continuity OK: {len(out)} rows identical to maps_batch1 population, "
          f"+1 column `desc`")

    p = HERE / "cap_crowd_population.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    nod = sum(1 for r in out if not r["desc"])
    L = [len(r["desc"]) for r in out if r["desc"]]
    print(f"wrote {p.name}: desc present on {len(out) - nod}/{len(out)} rows; "
          f"desc chars median {int(np.median(L))} max {max(L)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LEAVE-ONE-PROPOSER-OUT jackknife of the Good-Turing missing mass ON THE MERGED
SPECIES -- i.e. on the figure of record.

`species.py` computes the LOPO jackknife on the tau-only clustering.  The FREEZE
DECLARATION's identity rule makes the STRICT blind-pairwise merge the figure of record,
so the width around THAT number has to be the quoted one.  This script recomputes the
merge from `<tag>_bmerge_key.json` + the judge verdict files (never from the already
merged `<tag>_species.json`, which it does not touch), drops one proposer at a time, and
reports the missing mass each time.

Dropping a proposer removes that proposer's proposals AND every merge edge that touched
them, so a species held together only by the dropped proposer's phrasing correctly falls
apart -- which is the point of the jackknife.

TWO-TIER RULE: only TIER-S (sealed) proposals enter, exactly as in species.py.

CPU only.
Usage: python3 merged_jackknife.py --cell jokes_community --round 1 \
         --verdicts <judgeA>.json,<judgeB>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def merged_species(P, key_items, maps, track, keep_idx):
    """Union-find over keep_idx only; returns species sizes."""
    pos = {i: k for k, i in enumerate(keep_idx)}
    par = list(range(len(keep_idx)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for it in key_items:
        if it.get("track", "B") != track:
            continue
        i, j = it["_i"], it["_j"]
        if i not in pos or j not in pos:
            continue
        if all(m.get(it["pair_id"]) == "SAME" for m in maps):
            a_, b_ = find(pos[i]), find(pos[j])
            if a_ != b_:
                par[a_] = b_
    sizes = {}
    for k in range(len(keep_idx)):
        sizes[find(k)] = sizes.get(find(k), 0) + 1
    return np.array(sorted(sizes.values()))


def gt(sizes, n):
    f1 = int((sizes == 1).sum())
    f2 = int((sizes == 2).sum())
    return {"S_obs": int(len(sizes)), "f1": f1, "f2": f2, "N": int(n),
            "good_turing_missing_mass": f1 / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--verdicts", required=True)
    a = ap.parse_args()
    tag = f"{a.cell}_r{a.round}"

    key = json.loads((HERE / f"{tag}_bmerge_key.json").read_text())
    pool = json.loads((HERE / f"{tag}_proposals_fleet.json").read_text())["proposals"]
    pool = [p for p in pool if str(p.get("tier", "S")).upper() != "D"]
    maps = [{v["pair_id"]: v["verdict"].upper()
             for v in json.loads(Path(p).read_text())["verdicts"]}
            for p in a.verdicts.split(",")]

    out = {"tag": tag, "identity_rule": "STRICT blind pairwise (all judges SAME)",
           "n_judges": len(maps), "tier": "S",
           "note": "LOPO jackknife ON THE MERGED species -- the figure of record. "
                   "species.py's jackknife is on the tau-only clustering and is kept "
                   "beside this one, never in place of it.",
           "tracks": {}}
    for track in ("A", "B"):
        P = [p for p in pool if p["track"] == track]
        if not P:
            continue
        full = gt(merged_species(P, key["items"], maps, track, list(range(len(P)))), len(P))
        proposers = sorted({p["proposer"] for p in P})
        jack = []
        for drop in proposers:
            keep = [i for i, p in enumerate(P) if p["proposer"] != drop]
            g = gt(merged_species(P, key["items"], maps, track, keep), len(keep))
            jack.append({"dropped": drop, **g})
        vals = [j["good_turing_missing_mass"] for j in jack]
        out["tracks"][track] = {
            "full": full,
            "jackknife_LOPO_missing_mass": {
                "values": [round(v, 4) for v in vals],
                "min": float(min(vals)), "max": float(max(vals)),
                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
            },
            "per_drop": jack,
        }
    (HERE / f"{tag}_merged_jackknife.json").write_text(json.dumps(out, indent=1))
    for t, b in out["tracks"].items():
        j = b["jackknife_LOPO_missing_mass"]
        print(f"[{tag}] track {t}: merged S_obs={b['full']['S_obs']} f1={b['full']['f1']} "
              f"M_hat={b['full']['good_turing_missing_mass']:.3f} "
              f"LOPO [{j['min']:.3f}, {j['max']:.3f}] mean {j['mean']:.3f} sd {j['sd']:.3f}")


if __name__ == "__main__":
    main()

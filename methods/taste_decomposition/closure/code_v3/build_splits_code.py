#!/usr/bin/env python3
"""FIT+MINE / MONITOR split for the code_v3 closure cell, and the split-geometry
report that the WITHIN-REPO protocol adaptation rests on.

Frozen rule (prereg): stable-hash on the GROUP key, threshold .80.  Group key here
is the REPOSITORY, which is also the dense chain's split key and the fold key of every
stack in the cell -- so FIT+MINE and MONITOR are repo-disjoint by construction, and
neither contains a repository the dense model trained on.

Because the whole population is dense-held-out (see cells_code.py), MONITOR needs no
intersection with a dense-held-out set: it is T-honest automatically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))

import cells_code as C                                     # noqa: E402
from closure_core import hash_unit                          # noqa: E402

THRESH = 0.80


def build():
    d = C.load()
    repos = sorted(set(d["groups"]))
    h = {r: hash_unit(r) for r in repos}
    fit_repos = {r for r in repos if h[r] < THRESH}
    fitmask = np.array([g in fit_repos for g in d["groups"]])
    monmask = ~fitmask
    assert not (set(d["groups"][fitmask]) & set(d["groups"][monmask])), "repo leak"

    rep = {"threshold": THRESH, "group_key": "repository",
           "n_rows": int(len(d["y"])), "n_repos": len(repos)}
    for nm, m in (("fitmine", fitmask), ("monitor", monmask)):
        wr = C.within_repo_auc(d["y"], d["dense_seed42"], d["groups"], m)
        rep[nm] = {
            "n_rows": int(m.sum()), "n_repos": int(len(set(d["groups"][m]))),
            "pos_rate": float(d["y"][m].mean()),
            "n_eval_rows": int((m & (d["split"] == "eval")).sum()),
            "n_test_rows": int((m & (d["split"] == "test")).sum()),
            "scored_repos_ge20_2class": wr["n_repos"],
            "scored_rows": wr["n_rows"],
        }
    (HERE / "splits_report.json").write_text(json.dumps(rep, indent=1))
    np.savez_compressed(HERE / "splits.npz", ids=np.array(d["ids"]),
                        fitmask=fitmask, monmask=monmask)
    print(json.dumps(rep, indent=1))
    return d, fitmask, monmask


if __name__ == "__main__":
    build()

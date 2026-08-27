#!/usr/bin/env python3
"""Leave-one-container-out jackknife on Delta_beyond, per bank state.

Coordinator instruction (2026-08-08): HashtagWars' MONITOR is four contests and its
group-clustered bootstrap CI is wide, so every round's Delta must also carry a
GROUP JACKKNIFE over the held-out containers -- drop one contest at a time from the
HONEST population, recompute Delta = AUC(T) - AUC(VA_nl) on the remaining ones, and
quote the spread honestly.

Reported per state: the pooled Delta, the leave-one-out values, the jackknife mean,
the jackknife standard error  SE = sqrt((n-1)/n * sum_i (d_i - d_bar)^2),
the min/max, and the single most influential contest (the one whose removal moves
Delta most).  Also reported for T and VA_nl separately so a swing can be attributed.

CPU only.  Usage: python jackknife.py --cell hashtagwars_verdict [--upto 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L
import stage1_slice as S1

HERE = Path(__file__).resolve().parent


def states_upto(d, upto):
    """[(label, blocks)] for state0, state_d, state1..state<upto>."""
    out = [("state0", [d["V"], d["A"]])]
    blocks = [d["V"], d["A"]]
    for tag, label in [("d", "state_d")] + [(r, f"state{r}") for r in range(1, upto + 1)]:
        f = HERE / f"{d['tag']}_r{tag}_scores.npz"
        rt = HERE / f"{d['tag']}_r{tag}_routing_final.json"
        if not (f.exists() and rt.exists()):
            continue
        z = np.load(f, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        a_ids = [x["blind_id"] for x in json.loads(rt.read_text())["final"]
                 if x["final_route"] == "A"]
        idx = [cids.index(i) for i in a_ids if i in cids]
        if idx:
            blocks = blocks + [z["X"][:, idx]]
        out.append((label, list(blocks)))
    return out


def jack(y, t, v, groups):
    gs = sorted(set(groups.tolist()))
    rows = []
    for g in gs:
        m = groups != g
        if len(set(y[m].tolist())) < 2:
            continue
        rows.append({"dropped_container": str(g), "n_remaining": int(m.sum()),
                     "T": L.auc(y[m], t[m]), "VA_nl": L.auc(y[m], v[m]),
                     "Delta": L.auc(y[m], t[m]) - L.auc(y[m], v[m])})
    d = np.array([r["Delta"] for r in rows])
    n = len(d)
    se = float(np.sqrt((n - 1) / n * ((d - d.mean()) ** 2).sum())) if n > 1 else None
    worst = max(rows, key=lambda r: abs(r["Delta"] - (L.auc(y, t) - L.auc(y, v))))
    return {
        "pooled_Delta": L.auc(y, t) - L.auc(y, v),
        "pooled_T": L.auc(y, t), "pooled_VA_nl": L.auc(y, v),
        "n_containers": n,
        "jackknife_mean_Delta": float(d.mean()),
        "jackknife_SE": se,
        "jackknife_range": [float(d.min()), float(d.max())],
        "jackknife_pseudo_CI95": ([float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]
                                  if se is not None else None),
        "most_influential_container": worst["dropped_container"],
        "Delta_without_it": worst["Delta"],
        "leave_one_out": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="hashtagwars_verdict")
    ap.add_argument("--upto", type=int, default=2)
    a = ap.parse_args()

    d = C.load(a.cell)
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, g, dense = d["y"], d["groups"], d["dense"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])

    out = {"cell": a.cell, "n_HONEST": int(held.sum()),
           "n_held_out_containers": int(len(set(g[held].tolist()))),
           "note": "leave-one-container-out over the dense-held-out containers; VA_nl is "
                   "grouped-OOF inside FIT+MINE and held-out on MONITOR in every state",
           "states": {}}
    for label, blocks in states_upto(d, a.upto):
        r = L.fit_block(blocks, fitm, monm, y, g)
        v = np.full(len(y), np.nan)
        v[fitm] = r["oof_nl_fitmine"]
        v[monm] = r["nl_mon"]
        j = jack(y[held], dense[held], v[held], g[held])
        j["n_features"] = r["n_features"]
        out["states"][label] = j
        print(f"{label:8s} feat={r['n_features']:3d} Delta {j['pooled_Delta']:+.4f} "
              f"jack mean {j['jackknife_mean_Delta']:+.4f} SE {j['jackknife_SE']:.4f} "
              f"range [{j['jackknife_range'][0]:+.4f}, {j['jackknife_range'][1]:+.4f}] "
              f"most influential: {j['most_influential_container']} "
              f"(Delta -> {j['Delta_without_it']:+.4f})")
    (HERE / f"{a.cell}_delta_jackknife.json").write_text(json.dumps(out, indent=1))
    print("wrote", HERE / f"{a.cell}_delta_jackknife.json")


if __name__ == "__main__":
    main()

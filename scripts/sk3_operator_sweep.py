"""Sweep CE/cos combination operators for the clustering affinity.

The 0.5 linear blend under-merges: averaging lets the cosine (which has a
~0.85 ceiling for same-concept / different-wording pairs) drag CE-confident
merges below the cut. This compares CE-favoured operators:

  blend_a : a*CE + (1-a)*cos          linear blend, a = CE weight
  ce      : CE                        (a = 1)
  gate_f  : CE if cos >= f else cos    CE decides; cos only vetoes when it is
                                       low enough to signal genuine unrelated

For each operator: build candidate affinity from cached CE scores, hybrid
distance (1-cos on non-candidate pairs), average-linkage, sweep tau. Reports
pooled held-out judge FP/FN, largest cluster (chaining watch), and total
cluster count per (operator, tau).
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sk3_match_pipeline import FORMS, MATCH_OUT, VERDICTS, load_task

TAUS = [0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80, 0.775, 0.75, 0.70]
VARIANTS = [("blend0.50", "blend", 0.50), ("blend0.75", "blend", 0.75),
            ("ce", "ce", None), ("gate0.50", "gate", 0.50),
            ("gate0.60", "gate", 0.60)]


def affinity(kind, param, ce, cand_cos):
    if kind == "blend":
        return param * ce + (1.0 - param) * cand_cos
    if kind == "ce":
        return ce
    return np.where(cand_cos >= param, ce, cand_cos)  # gate


def main():
    by_task = defaultdict(lambda: defaultdict(list))
    for line in FORMS.open():
        r = json.loads(line)
        by_task[r["task"]][r["bucket"]].append(r)
    ev = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "eval" and v.get("score") in (0, 1, 2):
            ev[v["task"]].append(v)

    acc = {n: {t: [0, 0, 0, 0] for t in TAUS} for n, _, _ in VARIANTS}
    mx = {n: {t: 0 for t in TAUS} for n, _, _ in VARIANTS}
    nc = {n: {t: 0 for t in TAUS} for n, _, _ in VARIANTS}

    for task in sorted(by_task):
        sp = MATCH_OUT / f"scored_{task}.npz"
        if not sp.exists():
            continue
        rows, emb = load_task(task, by_task[task])
        if rows is None:
            continue
        keymap = {r["key"]: g for g, r in enumerate(rows)}
        cos = (emb @ emb.T).astype(np.float64)
        d = np.load(sp)
        ii, jj, ce = d["ii"], d["jj"], d["ce"].astype(np.float64)
        cand_cos = cos[ii, jj]
        base = np.clip(1.0 - cos, 0.0, None)
        np.fill_diagonal(base, 0.0)
        pairs = ev.get(task, [])
        for name, kind, param in VARIANTS:
            aff = affinity(kind, param, ce, cand_cos)
            D = base.copy()
            dd = np.clip(1.0 - aff, 0.0, None)
            D[ii, jj] = dd
            D[jj, ii] = dd
            Z = linkage(squareform(D, checks=False), method="average")
            del D
            for tau in TAUS:
                lab = fcluster(Z, t=1.0 - tau, criterion="distance")
                cnt = Counter(lab.tolist())
                mx[name][tau] = max(mx[name][tau], max(cnt.values()))
                nc[name][tau] += len(cnt)
                mg = bad = s = miss = 0
                for v in pairs:
                    a, b = keymap.get(v["key_a"]), keymap.get(v["key_b"])
                    if a is None or b is None:
                        continue
                    if lab[a] == lab[b]:
                        mg += 1
                        bad += v["score"] != 2
                    else:
                        s += 1
                        miss += v["score"] == 2
                acc[name][tau][0] += mg
                acc[name][tau][1] += bad
                acc[name][tau][2] += s
                acc[name][tau][3] += miss
        print(f"  {task} done", flush=True)

    for name, _, _ in VARIANTS:
        print(f"\n--- {name}  (tau : FP / FN | largest | total clusters) ---")
        for tau in TAUS:
            mgv, bad, s, miss = acc[name][tau]
            fp = bad / mgv * 100 if mgv else 0.0
            fn = miss / s * 100 if s else 0.0
            print(f"  {tau:>5.3f}  {fp:>6.1f} / {fn:<6.1f} | "
                  f"{mx[name][tau]:>6} | {nc[name][tau]:>7}")


if __name__ == "__main__":
    main()

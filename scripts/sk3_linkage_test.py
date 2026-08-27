"""Compare linkage methods for the blend clustering.

Manual inspection found a chunk of FN are not CE errors but complete-linkage
artifacts: pairs the CE scores 0.90-1.00 (clearly same) still split, because
complete-linkage demands the whole merged cluster stay tight. This tests
whether average-linkage recovers them without inflating FP or producing
mega-clusters.

alpha fixed at 0.5 (locked by sk3_blend_sweep). Reuses the cached CE scores
(match_out/scored_<task>.npz) -- no GPU. Reports, per linkage method, the
held-out FP/FN frontier and the largest cluster at each tau (over-merge watch).
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sk3_match_pipeline import (FN_TARGETS, FORMS, MATCH_OUT, TAUS, VERDICTS,
                                load_task)

ALPHA = 0.5
METHODS = ["complete", "average", "weighted"]


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

    acc = {m: {tau: [0, 0, 0, 0] for tau in TAUS} for m in METHODS}
    maxsz = {m: {tau: 0 for tau in TAUS} for m in METHODS}

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
        ii, jj, ce = d["ii"], d["jj"], d["ce"]
        aff = ALPHA * ce + (1.0 - ALPHA) * cos[ii, jj]
        D = np.clip(1.0 - cos, 0.0, None)
        np.fill_diagonal(D, 0.0)
        dd = np.clip(1.0 - aff, 0.0, None)
        D[ii, jj] = dd
        D[jj, ii] = dd
        cd = squareform(D, checks=False)
        pairs = ev.get(task, [])
        for m in METHODS:
            Z = linkage(cd, method=m)
            for tau in TAUS:
                lab = fcluster(Z, t=1.0 - tau, criterion="distance")
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
                acc[m][tau][0] += mg
                acc[m][tau][1] += bad
                acc[m][tau][2] += s
                acc[m][tau][3] += miss
                maxsz[m][tau] = max(maxsz[m][tau],
                                    Counter(lab).most_common(1)[0][1])
        print(f"  {task} done", flush=True)

    def rate(m, tau):
        mg, bad, s, miss = acc[m][tau]
        return (bad / mg * 100 if mg else 0.0, miss / s * 100 if s else 0.0)

    print(f"\n{'=' * 72}")
    print("linkage frontier -- min FP at each FN ceiling (alpha=0.5)")
    print(f"{'=' * 72}")
    print(f"{'FN<=':>6} | " + "  ".join(f"{m:>10}" for m in METHODS))
    for tgt in FN_TARGETS:
        cells = []
        for m in METHODS:
            fps = [rate(m, t)[0] for t in TAUS if rate(m, t)[1] <= tgt * 100]
            cells.append(f"{min(fps):>9.1f}%" if fps else f"{'--':>10}")
        print(f"{tgt * 100:>5.0f}% | " + "  ".join(cells))

    for m in METHODS:
        print(f"\n--- {m}  (tau : FP / FN  | largest cluster) ---")
        for tau in TAUS:
            fp, fn = rate(m, tau)
            print(f"  {tau:>4.2f}  {fp:>6.1f} / {fn:<6.1f} | {maxsz[m][tau]:>5}")


if __name__ == "__main__":
    main()

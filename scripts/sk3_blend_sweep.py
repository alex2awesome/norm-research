"""Sweep the blend weight alpha for the CE-rerank matching pipeline.

The full pipeline (sk3_match_pipeline.py) cached, per task, the candidate
indices and CE scores in match_out/scored_<task>.npz. This re-uses those --
no GPU, no CE inference -- and rebuilds the hybrid affinity for a range of
blend weights:

    affinity = alpha * CE + (1 - alpha) * cos      (on candidate pairs)

alpha=0 is pure LoRA cosine, alpha=1 is pure CE. For each alpha it clusters
(complete-linkage), sweeps tau, and reports the held-out FP/FN frontier, so
the best blend mix can be locked from one cheap CPU pass.
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["XDG_CACHE_HOME"] = "/lfs/skampere3/0/alexspan/.cache"

import json
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from sk3_match_pipeline import (EMB, FN_TARGETS, FORMS, MATCH_OUT, TAUS,
                                VERDICTS, fpfn, load_task)

ALPHAS = [0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0]


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

    acc = {a: {tau: [0, 0, 0, 0] for tau in TAUS} for a in ALPHAS}
    for task in sorted(by_task):
        sp = MATCH_OUT / f"scored_{task}.npz"
        if not sp.exists():
            print(f"  {task}: no cached scores -- skipped", flush=True)
            continue
        rows, emb = load_task(task, by_task[task])
        if rows is None:
            continue
        keymap = {r["key"]: g for g, r in enumerate(rows)}
        cos = (emb @ emb.T).astype(np.float64)
        d = np.load(sp)
        ii, jj, ce = d["ii"], d["jj"], d["ce"]
        cand_cos = cos[ii, jj]
        base = np.clip(1.0 - cos, 0.0, None)
        np.fill_diagonal(base, 0.0)
        for a in ALPHAS:
            aff = a * ce + (1.0 - a) * cand_cos
            D = base.copy()
            dd = np.clip(1.0 - aff, 0.0, None)
            D[ii, jj] = dd
            D[jj, ii] = dd
            Z = linkage(squareform(D, checks=False), method="complete")
            for tau, (mg, bad, s, m) in fpfn(Z, keymap, ev.get(task, [])).items():
                acc[a][tau][0] += mg
                acc[a][tau][1] += bad
                acc[a][tau][2] += s
                acc[a][tau][3] += m
            del D
        print(f"  {task} done", flush=True)

    def rate(a, tau):
        mg, bad, s, m = acc[a][tau]
        return (bad / mg * 100 if mg else 0.0, m / s * 100 if s else 0.0)

    print(f"\n{'=' * 70}")
    print("blend-alpha frontier -- min FP at each FN ceiling "
          "(alpha 0=cos .. 1=ce)")
    print(f"{'=' * 70}")
    print(f"{'FN<=':>6} | " + "  ".join(f"a={a:<5.2f}" for a in ALPHAS))
    for tgt in FN_TARGETS:
        cells = []
        for a in ALPHAS:
            fps = [rate(a, t)[0] for t in TAUS if rate(a, t)[1] <= tgt * 100]
            cells.append(f"{min(fps):>7.1f}" if fps else f"{'--':>7}")
        print(f"{tgt * 100:>5.0f}% | " + "  ".join(cells))

    for a in ALPHAS:
        print(f"\n--- alpha={a:.2f}  (tau : FP / FN) ---")
        for tau in TAUS:
            fp, fn = rate(a, tau)
            print(f"  {tau:>4.2f}  {fp:>6.1f} / {fn:<6.1f}")


if __name__ == "__main__":
    main()

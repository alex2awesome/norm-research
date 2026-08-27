"""Finalize the rubric clustering: average-linkage blend, save assignments.

Locked config (from sk3_blend_sweep + sk3_linkage_test):
  affinity  = 0.5*CE + 0.5*cos on candidate pairs, 1*cos elsewhere
  clustering = average-linkage agglomerative, cut at --tau (default 0.92)
  operating point ~ FP 3.2% / FN 12.4% on held-out pairs.

Per task: rebuild the blend distance matrix from cached CE scores
(match_out/scored_<task>.npz), average-linkage, cut at tau, then save the
dendrogram (Z_avg_<task>.npy) and the cluster assignment
(clusters_<task>.json: key -> cluster id).

Prints a one-line summary for every task and a detailed inspection -- sample
clusters, the single largest cluster, held-out FN/FP cases -- for --detail
tasks, so precision can be re-checked by eye under the looser linkage.
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sk3_match_pipeline import FORMS, MATCH_OUT, VERDICTS, load_task

ALPHA = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.92)
    ap.add_argument("--detail", default="peer-review,creative-writing,code-review")
    args = ap.parse_args()
    detail = set(args.detail.split(","))
    rng = random.Random(0)

    by_task = defaultdict(lambda: defaultdict(list))
    canon = {}
    for line in FORMS.open():
        r = json.loads(line)
        by_task[r["task"]][r["bucket"]].append(r)
        canon[r["key"]] = r["canonical"]
    ev = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "eval" and v.get("score") in (0, 1, 2):
            ev[v["task"]].append(v)

    for task in sorted(by_task):
        sp = MATCH_OUT / f"scored_{task}.npz"
        if not sp.exists():
            print(f"{task}: no cached scores -- skipped")
            continue
        rows, emb = load_task(task, by_task[task])
        if rows is None:
            continue
        keys = [r["key"] for r in rows]
        keymap = {k: g for g, k in enumerate(keys)}
        cos = (emb @ emb.T).astype(np.float64)
        d = np.load(sp)
        ii, jj, ce = d["ii"], d["jj"], d["ce"]
        aff = ALPHA * ce + (1.0 - ALPHA) * cos[ii, jj]
        D = np.clip(1.0 - cos, 0.0, None)
        np.fill_diagonal(D, 0.0)
        dd = np.clip(1.0 - aff, 0.0, None)
        D[ii, jj] = dd
        D[jj, ii] = dd
        Z = linkage(squareform(D, checks=False), method="average")
        lab = fcluster(Z, t=1.0 - args.tau, criterion="distance")

        np.save(MATCH_OUT / f"Z_avg_{task}.npy", Z)
        (MATCH_OUT / f"clusters_{task}.json").write_text(
            json.dumps({k: int(lab[i]) for i, k in enumerate(keys)}))

        clusters = defaultdict(list)
        for i, c in enumerate(lab):
            clusters[c].append(i)
        sizes = sorted((len(v) for v in clusters.values()), reverse=True)
        print(f"{task:<26} {len(keys):>6} forms -> {len(clusters):>5} clusters"
              f"  singletons={sizes.count(1):>5}  max={sizes[0]:>4}")

        if task not in detail:
            continue
        ce_lookup = {(int(a), int(b)): float(s)
                     for a, b, s in zip(ii, jj, ce)}
        print(f"  {'-' * 66}")
        mid = [c for c, v in clusters.items() if 4 <= len(v) <= 9]
        for c in rng.sample(mid, min(4, len(mid))):
            print(f"  -- cluster ({len(clusters[c])}) --")
            for i in clusters[c]:
                print(f"     {(canon[keys[i]] or '')[:100]}")
        big = max(clusters.values(), key=len)
        print(f"  -- LARGEST cluster ({len(big)}), first 40 --")
        for i in big[:40]:
            print(f"     {(canon[keys[i]] or '')[:100]}")

        fns, fps = [], []
        for v in ev.get(task, []):
            a, b = keymap.get(v["key_a"]), keymap.get(v["key_b"])
            if a is None or b is None:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            cev = ce_lookup.get((lo, hi))
            if v["score"] == 2 and lab[a] != lab[b]:
                fns.append((v, cev))
            elif v["score"] != 2 and lab[a] == lab[b]:
                fps.append((v, cev))
        print(f"  FN={len(fns)}  FP={len(fps)}")
        for tag, items in (("FN", fns), ("FP", fps)):
            for v, cev in rng.sample(items, min(5, len(items))):
                cs = f"{cev:.2f}" if cev is not None else "n/a"
                print(f"    {tag}[CE={cs} judge={v['score']}] "
                      f"{(v['canonical_a'] or '')[:84]}")
                print(f"      {(v['canonical_b'] or '')[:84]}")


if __name__ == "__main__":
    main()

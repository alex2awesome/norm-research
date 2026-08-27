"""Manually verify singletons via a full nearest-neighbour search.

A form is only genuinely a singleton if NO other form is the same concept.
For each singleton in --bucket, this lists ALL its neighbours with CE >=
--thresh (not just the best one), so the full extent of any missed merge is
visible. N&C is a full-n^2 task -> the search is exhaustive.

Reports:
  - how many singletons have >=1 high-CE neighbour, and the neighbour-count
    distribution,
  - a connected-components view of the CE>=thresh graph: components that span
    more than one current cluster are fragmented concepts,
  - sample singletons with every high-CE neighbour (CE / cos / which cluster).
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np

from sk3_match_pipeline import FORMS, MATCH_OUT, load_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="notice-and-comment")
    ap.add_argument("--bucket", default="general")
    ap.add_argument("--thresh", type=float, default=0.90)
    args = ap.parse_args()
    task, thr = args.task, args.thresh

    by_bucket = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        if r["task"] == task:
            by_bucket[r["bucket"]].append(r)
    rows, emb = load_task(task, by_bucket)
    n = len(rows)
    canon = [r["canonical"] or "" for r in rows]
    bk = [r["bucket"] for r in rows]
    keys = [r["key"] for r in rows]
    inb = np.array([b == args.bucket for b in bk])

    clusters = json.loads((MATCH_OUT / f"clusters_{task}.json").read_text())
    lab = np.array([clusters[k] for k in keys])
    csize = Counter(lab.tolist())

    d = np.load(MATCH_OUT / f"scored_{task}.npz")
    ii, jj, ce = d["ii"], d["jj"], d["ce"].astype(np.float64)
    mask = ce >= thr
    adj = defaultdict(list)
    edges = []
    for a, b, s in zip(ii[mask].tolist(), jj[mask].tolist(), ce[mask].tolist()):
        if inb[a] and inb[b]:
            adj[a].append((b, s))
            adj[b].append((a, s))
            edges.append((a, b))

    sing = [i for i in range(n) if inb[i] and csize[lab[i]] == 1]
    nb = sum(inb)
    print(f"{task} / {args.bucket}: {nb} forms, {len(sing)} singletons, "
          f"CE>={thr} edges among them: {len(edges)}\n")

    with_n = [i for i in sing if adj.get(i)]
    counts = [len(adj[i]) for i in with_n]
    print(f"singletons with >=1 CE>={thr} neighbour: "
          f"{len(with_n)}/{len(sing)} ({len(with_n) / len(sing) * 100:.0f}%)")
    for lo, hi, tag in [(1, 2, "1"), (2, 3, "2"), (3, 5, "3-4"),
                        (5, 10, "5-9"), (10, 10**9, "10+")]:
        c = sum(1 for x in counts if lo <= x < hi)
        print(f"  {tag:>4} neighbours: {c}")
    span2 = sum(1 for i in with_n
                if len({lab[j] for j, _ in adj[i]} | {lab[i]}) > 1)
    print(f"  ...whose neighbours span >1 current cluster: {span2}")

    # connected components of the CE>=thr graph over bucket forms
    parent = {i: i for i in range(n) if inb[i]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    comps = defaultdict(list)
    for i in parent:
        comps[find(i)].append(i)
    multi = [m for m in comps.values() if len(m) > 1]
    frag = [m for m in multi if len({lab[x] for x in m}) > 1]
    print(f"\nCE>={thr} connected components over {nb} forms: "
          f"{len(comps)} components, {len(multi)} multi-form")
    print(f"  components that merge >1 current cluster (fragmented "
          f"concepts): {len(frag)}")
    extra = sum(len({lab[x] for x in m}) - 1 for m in frag)
    print(f"  -> collapsing them removes ~{extra} clusters "
          f"({len(comps)} would become ~{len(comps) - extra + len(multi) - len(frag)}... )")

    rng = random.Random(2)
    print(f"\n--- sample singletons + ALL CE>={thr} neighbours ---")
    for i in rng.sample(with_n, min(12, len(with_n))):
        print(f"\n  S: {canon[i][:104]}")
        for j, s in sorted(adj[i], key=lambda x: -x[1]):
            cos = float(emb[i] @ emb[j])
            kind = "singleton" if csize[lab[j]] == 1 else f"clust/{csize[lab[j]]}"
            print(f"    CE={s:.2f} cos={cos:.2f} [{kind}] {canon[j][:88]}")


if __name__ == "__main__":
    main()

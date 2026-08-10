"""Manual inspection of the blend clustering at a chosen tau.

Cuts each task's blend dendrogram (match_out/Z_blend_<task>.npy) at --tau and
prints, per task:
  - cluster-size summary,
  - a sample of multi-member clusters with member rubric texts
    (precision eyeball: are these genuinely the same concept?),
  - held-out FN cases (true-same eval pairs the clustering SPLIT) with each
    pair's CE score -- to tell a CE scoring error from a linkage artifact,
  - held-out FP cases (merged eval pairs the judge scored != same).
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
FORMS = WORK / "canon_all_real_forms.jsonl"
VERDICTS = WORK / "all_verdicts.jsonl"
MATCH_OUT = WORK / "match_out"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    tasks = args.only.split(",") if args.only else TASKS
    rng = random.Random(0)

    canon = {}
    for line in FORMS.open():
        r = json.loads(line)
        canon[r["key"]] = r["canonical"]
    ev = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "eval" and v.get("score") in (0, 1, 2):
            ev[v["task"]].append(v)

    for task in tasks:
        zp = MATCH_OUT / f"Z_blend_{task}.npy"
        kp = MATCH_OUT / f"keys_{task}.json"
        if not zp.exists():
            print(f"{task}: missing linkage -- skipped")
            continue
        Z = np.load(zp)
        keys = json.loads(kp.read_text())
        idx = {k: i for i, k in enumerate(keys)}
        lab = fcluster(Z, t=1.0 - args.tau, criterion="distance")
        clusters = defaultdict(list)
        for i, c in enumerate(lab):
            clusters[c].append(i)
        sizes = sorted((len(v) for v in clusters.values()), reverse=True)

        ce_lookup = {}
        sp = MATCH_OUT / f"scored_{task}.npz"
        if sp.exists():
            d = np.load(sp)
            for a, b, s in zip(d["ii"], d["jj"], d["ce"]):
                ce_lookup[(int(a), int(b))] = float(s)

        print(f"\n{'=' * 72}")
        print(f"{task}: {len(keys)} forms -> {len(clusters)} clusters  "
              f"(tau={args.tau})  singletons={sizes.count(1)}  "
              f"multi={sum(s > 1 for s in sizes)}  max={sizes[0]}")

        pick = [c for c, v in clusters.items() if 3 <= len(v) <= 8]
        for c in rng.sample(pick, min(5, len(pick))):
            print(f"  -- merged cluster ({len(clusters[c])} rubrics) --")
            for i in clusters[c]:
                print(f"     {(canon[keys[i]] or '')[:104]}")

        fns = []
        fps = []
        for v in ev.get(task, []):
            a, b = idx.get(v["key_a"]), idx.get(v["key_b"])
            if a is None or b is None:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            ce = ce_lookup.get((lo, hi))
            if v["score"] == 2 and lab[a] != lab[b]:
                fns.append((v, ce))
            elif v["score"] != 2 and lab[a] == lab[b]:
                fps.append((v, ce))

        print(f"  FN (true-same, split): {len(fns)}  |  "
              f"FP (merged, not-same): {len(fps)}")
        for tag, rows in (("FN", fns), ("FP", fps)):
            for v, ce in rng.sample(rows, min(5, len(rows))):
                cs = f"{ce:.2f}" if ce is not None else "not-cand"
                print(f"    {tag} [CE={cs}] [judge={v['score']}] "
                      f"{(v['canonical_a'] or '')[:88]}")
                print(f"               {(v['canonical_b'] or '')[:88]}")


if __name__ == "__main__":
    main()

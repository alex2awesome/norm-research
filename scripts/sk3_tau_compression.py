"""Compression vs tau for the locked average-linkage clustering.

Cuts the saved average-linkage dendrograms (match_out/Z_avg_<task>.npy) at a
range of tau and reports clusters / singletons / largest cluster -- the
compression curve -- plus a per-bucket singleton breakdown at the operating
tau, to see whether residual singletons are the (by-design) hyper-specific
rubrics or genuine under-merge in the general bucket.
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
FORMS = WORK / "canon_all_real_forms.jsonl"
MATCH_OUT = WORK / "match_out"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
TAUS = [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.80, 0.75, 0.70]


def main():
    bucket_of = defaultdict(dict)
    for line in FORMS.open():
        r = json.loads(line)
        bucket_of[r["task"]][r["key"]] = r["bucket"]

    for task in TASKS:
        zp = MATCH_OUT / f"Z_avg_{task}.npy"
        kp = MATCH_OUT / f"keys_{task}.json"
        if not zp.exists():
            print(f"{task}: missing -- skipped")
            continue
        Z = np.load(zp)
        keys = json.loads(kp.read_text())
        print(f"\n=== {task} ({len(keys)} forms) ===")
        print(f"  {'tau':>5} {'clusters':>9} {'singles':>9} "
              f"{'%single':>8} {'max':>5}")
        for tau in TAUS:
            lab = fcluster(Z, t=1.0 - tau, criterion="distance")
            cnt = Counter(lab)
            nsing = sum(1 for v in cnt.values() if v == 1)
            print(f"  {tau:>5.2f} {len(cnt):>9} {nsing:>9} "
                  f"{nsing / len(keys) * 100:>7.0f}% {max(cnt.values()):>5}")

        lab = fcluster(Z, t=1.0 - 0.92, criterion="distance")
        csize = Counter(lab)
        bk = defaultdict(lambda: [0, 0])
        for i, k in enumerate(keys):
            b = bucket_of[task].get(k, "?")
            bk[b][0] += 1
            if csize[lab[i]] == 1:
                bk[b][1] += 1
        print("  per-bucket singletons @ tau 0.92:")
        for b in sorted(bk):
            n, s = bk[b]
            print(f"    {b:<16} {n:>6} forms  {s:>6} singletons  "
                  f"({s / n * 100:>3.0f}%)")


if __name__ == "__main__":
    main()

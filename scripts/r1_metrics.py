"""Aggregate R1 metrics for one or more output directories.

Usage:  python scripts/r1_metrics.py <dir1> [<dir2> ...]

For each dir, prints a per-task table (clusters, families, compression,
multi/singleton counts, large-family counts, max size, dup-name affected
clusters + number of dup names).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path


def metrics(dir_path):
    out = {}
    for f in sorted(glob.glob(f"{dir_path}/r1_families_*.json")):
        task = Path(f).stem.replace("r1_families_", "")
        d = json.load(open(f))
        fams = d["families"]
        sizes = [len(fam["cluster_ids"]) for fam in fams]
        nm = [fam["name"].lower().strip() for fam in fams]
        dups = {k: v for k, v in Counter(nm).items() if v > 1}
        dup_cl = sum(len(fam["cluster_ids"]) for fam in fams
                     if fam["name"].lower().strip() in dups)
        out[task] = {
            "cl": len(d["cluster_to_family"]),
            "fam": len(fams),
            "multi": sum(1 for s in sizes if s > 1),
            "sing": sum(1 for s in sizes if s == 1),
            "ge20": sum(1 for s in sizes if s >= 20),
            "ge30": sum(1 for s in sizes if s >= 30),
            "ge40": sum(1 for s in sizes if s >= 40),
            "max": max(sizes) if sizes else 0,
            "dup_cl": dup_cl,
            "n_dup_names": len(dups),
        }
    return out


def print_table(m, name):
    print(f"\n=== {name} ===")
    print(f"{'task':<26}{'cl':>6}{'fam':>6}{'compr':>7}{'multi':>6}"
          f"{'sing':>6}{'ge30':>5}{'ge20':>5}{'max':>5}{'dupC':>6}{'dupN':>5}")
    print("-" * 82)
    tot = {"cl": 0, "fam": 0, "multi": 0, "sing": 0,
           "ge20": 0, "ge30": 0, "dup_cl": 0, "n_dup_names": 0}
    for task in sorted(m):
        d = m[task]
        compr = d["cl"] / d["fam"] if d["fam"] else 1
        print(f"{task:<26}{d['cl']:>6}{d['fam']:>6}{compr:>6.2f}x"
              f"{d['multi']:>6}{d['sing']:>6}{d['ge30']:>5}{d['ge20']:>5}"
              f"{d['max']:>5}{d['dup_cl']:>6}{d['n_dup_names']:>5}")
        for k in tot:
            tot[k] += d[k]
    print("-" * 82)
    compr = tot["cl"] / tot["fam"] if tot["fam"] else 1
    print(f"{'TOTAL':<26}{tot['cl']:>6}{tot['fam']:>6}{compr:>6.2f}x"
          f"{tot['multi']:>6}{tot['sing']:>6}{tot['ge30']:>5}{tot['ge20']:>5}"
          f"{'':>5}{tot['dup_cl']:>6}{tot['n_dup_names']:>5}")
    if tot["cl"]:
        print(f"  -> dup-cluster rate: {tot['dup_cl']/tot['cl']*100:.1f}% "
              f"of clusters in dup-named families")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        m = metrics(d)
        if not m:
            print(f"\n=== {d} === (no r1_families files found)")
        else:
            print_table(m, d)

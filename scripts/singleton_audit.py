"""Surface near-neighbour pairs for manual same-concept adjudication of SINGLETONS.

A singleton = an R1-refined cluster with exactly one member: HDBSCAN (eps=0)
left it unclustered and the LLM dedup pass merged nothing into it. Two ways a
singleton can be an under-merge (a dedup miss):

  mode ss  singleton-vs-singleton  -- two singletons are the same concept and
                                     should have been merged together.
  mode sc  singleton-vs-cluster    -- a singleton is the same concept as an
                                     existing MULTI-member cluster and should
                                     have been absorbed into it.

The script only SURFACES candidates (nearest neighbour by cosine in the
text-embedding-3-small cache). The same-concept call is made by hand from the
printed text. "Same concept" = same construct / same verdict on real data.

Usage:
  python scripts/singleton_audit.py --mode sc --all --per-band 4
  python scripts/singleton_audit.py --mode ss --tasks grant-funding,legal-...
  python scripts/singleton_audit.py --stats-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
EMB_CACHE = ROOT / "notebooks" / "_explore_cache"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]
BANDS = [(0.92, 1.01, ">=.92"), (0.85, 0.92, ".85-.92"), (0.80, 0.85, ".80-.85"),
         (0.75, 0.80, ".75-.80"), (0.70, 0.75, ".70-.75"), (0.60, 0.70, ".60-.70")]
PRINT_FROM = 0.78   # only print sampled pairs from bands at/above this lo


def enumerate_clusters(task: str):
    """(name, desc, size) per child, in load_cluster_texts() order."""
    d = json.loads((HIER / f"{task}_general_r1_refined.json").read_text())
    rows = []
    for par in d.get("parented_trees", []):
        for ch in par.get("children", []):
            rows.append((ch.get("medoid_name", "") or "",
                         ch.get("medoid_description", "") or "",
                         len(ch.get("rubrics", []))))
    return rows


def task_indices(task: str):
    """Return rows, emb, singleton idxs, multi-member idxs (text-deduped)."""
    rows = enumerate_clusters(task)
    emb = np.load(EMB_CACHE / f"emb_rubric_cluster_{task}.npy")
    assert len(emb) == len(rows), f"{task}: {len(emb)} emb vs {len(rows)} rows"
    seen, idxs = set(), []
    for i, (nm, ds, _) in enumerate(rows):
        if (nm, ds) in seen:
            continue
        seen.add((nm, ds))
        idxs.append(i)
    sing = [i for i in idxs if rows[i][2] == 1]
    multi = [i for i in idxs if rows[i][2] >= 2]
    return rows, emb, sing, multi


def nn_pairs(emb, src_idx, tgt_idx, same_set: bool):
    """Nearest target for each source. Returns list of (src_i, tgt_i, cos)."""
    Ssrc, Stgt = emb[src_idx], emb[tgt_idx]
    sims = Ssrc @ Stgt.T
    if same_set:
        np.fill_diagonal(sims, -1.0)
    nn = sims.argmax(axis=1)
    nn_sim = sims[np.arange(len(src_idx)), nn]
    out = {}
    for a in range(len(src_idx)):
        b = int(nn[a])
        key = (min(a, b), max(a, b)) if same_set else (a, b)
        if key not in out or nn_sim[a] > out[key]:
            out[key] = float(nn_sim[a])
    return [(k[0], k[1], s) for k, s in out.items()]


def audit_task(task, mode, per_band, rng):
    rows, emb, sing, multi = task_indices(task)
    n_uniq = len(sing) + len(multi)
    if mode == "ss":
        pairs = nn_pairs(emb, sing, sing, same_set=True)
        src, tgt = sing, sing
        label = "singleton-vs-singleton"
    else:
        if not multi:
            print(f"== {task}: no multi-member clusters ==")
            return {}
        pairs = nn_pairs(emb, sing, multi, same_set=False)
        src, tgt = sing, multi
        label = "singleton-vs-cluster"

    counts = {lab: 0 for *_, lab in BANDS}
    for *_, s in [(0, 0, p[2]) for p in pairs]:
        for lo, hi, lab in BANDS:
            if lo <= s < hi:
                counts[lab] += 1
    print(f"\n{'#'*80}\n# {task}  [{label}]  "
          f"{len(sing)} singletons / {n_uniq} clusters "
          f"({len(sing)/n_uniq*100:.0f}%)")
    print(f"# NN-pair band counts: " +
          "  ".join(f"{lab}:{counts[lab]}" for *_, lab in BANDS))
    print('#'*80)

    for lo, hi, lab in BANDS:
        if lo < PRINT_FROM:
            continue
        band = [(a, b, s) for a, b, s in pairs if lo <= s < hi]
        if not band:
            continue
        pick = rng.choice(len(band), size=min(per_band, len(band)), replace=False)
        print(f"\n-- BAND {lab}  ({len(band)} pairs) --")
        for j in sorted(pick):
            a, b, s = band[j]
            na, da, _ = rows[src[a]]
            nb, db, szb = rows[tgt[b]]
            tag = f" [cluster size {szb}]" if mode == "sc" else ""
            print(f"\n  [cos={s:.3f}]")
            print(f"  SINGLETON: {na}")
            print(f"     {da[:240]}")
            print(f"  {'CLUSTER' if mode=='sc' else 'SINGLETON'}: {nb}{tag}")
            print(f"     {db[:240]}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ss", "sc"], default="sc")
    ap.add_argument("--tasks", default=None, help="comma-separated subset")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--per-band", type=int, default=4)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20)

    if args.stats_only:
        print(f"{'task':<26} {'clusters':>9} {'singletons':>11} {'singl%':>8}")
        for t in TASKS:
            _, _, sing, multi = task_indices(t)
            n = len(sing) + len(multi)
            print(f"{t:<26} {n:>9} {len(sing):>11} {len(sing)/n*100:>7.0f}%")
        return

    tasks = TASKS if args.all else (args.tasks.split(",") if args.tasks else TASKS)
    for t in tasks:
        audit_task(t, args.mode, args.per_band, rng)


if __name__ == "__main__":
    main()

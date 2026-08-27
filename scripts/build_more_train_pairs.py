"""Collect ~300K MORE labelled training pairs, on top of the first 122K.

Constraints:
  - Train pairs only. Excludes any rubric that appears in eval_pairs.jsonl, so
    the held-out evaluation stays leakage-free.
  - Excludes pairs already sampled into all_train_pairs.jsonl (the NN-based
    candidate sampling is deterministic and would otherwise repeat).
  - Stratified across cosine bands per task; pooled across general/specific/
    hyper_specific buckets (so cross-bucket pairs are included).

Output: more_train_pairs.jsonl  (judged afterward by the v6 LLM judge)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
EMB = ROOT / "notebooks" / "_explore_cache" / "bge"
FORMS = OUT / "canon_all_real_forms.jsonl"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
BANDS = [(0.97, 1.01), (0.93, 0.97), (0.89, 0.93), (0.85, 0.89),
         (0.80, 0.85), (0.70, 0.80), (0.50, 0.70)]
PER_BAND = 7400   # cap per band per task
RNG = np.random.default_rng(202)


def load_task(task):
    rows, embs = [], []
    by_bt = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        if r["task"] == task:
            by_bt[r["bucket"]].append(r)
    for bucket, rs in sorted(by_bt.items()):
        rs.sort(key=lambda r: r["idx"])
        p = EMB / f"emb_bge_{bucket}_{task}.npy"
        if not p.exists() or len(np.load(p)) != len(rs):
            continue
        rows += rs
        embs.append(np.load(p).astype(np.float64))
    if not embs:
        return [], None
    emb = np.vstack(embs)
    return rows, emb / np.linalg.norm(emb, axis=1, keepdims=True)


def main():
    eval_keys = set()
    for line in (OUT / "eval_pairs.jsonl").open():
        r = json.loads(line)
        eval_keys.add(r["key_a"])
        eval_keys.add(r["key_b"])
    existing = set()
    for line in (OUT / "all_train_pairs.jsonl").open():
        r = json.loads(line)
        existing.add(frozenset((r["key_a"], r["key_b"])))
    print(f"excluding {len(eval_keys)} eval-rubric keys, {len(existing)} existing pairs")

    out = open(OUT / "more_train_pairs.jsonl", "w")
    total = 0
    for task in TASKS:
        rows, emb = load_task(task)
        if emb is None:
            continue
        keep = [i for i, r in enumerate(rows) if r["key"] not in eval_keys]
        keep = np.array(keep)
        sub = emb[keep]
        n = len(keep)
        cand = {}
        sims = sub @ sub.T
        np.fill_diagonal(sims, -1.0)
        k = min(18, n - 1)
        nn = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(n):
            for j in nn[i]:
                a, b = sorted((i, int(j)))
                if a != b:
                    cand[(a, b)] = float(sub[a] @ sub[b])
        ri = RNG.integers(0, n, size=80 * n)
        rj = RNG.integers(0, n, size=80 * n)
        for a, b in zip(ri, rj):
            if a != b:
                a, b = sorted((int(a), int(b)))
                cand[(a, b)] = float(sub[a] @ sub[b])
        banded = defaultdict(list)
        for (a, b), c in cand.items():
            ka, kb = rows[int(keep[a])]["key"], rows[int(keep[b])]["key"]
            if frozenset((ka, kb)) in existing:
                continue
            for lo, hi in BANDS:
                if lo <= c < hi:
                    banded[(lo, hi)].append((int(keep[a]), int(keep[b]), c))
                    break
        n_task = 0
        for lo, hi in BANDS:
            pool = banded[(lo, hi)]
            if not pool:
                continue
            pick = RNG.choice(len(pool), size=min(PER_BAND, len(pool)),
                              replace=False)
            for kk in pick:
                a, b, c = pool[kk]
                out.write(json.dumps({
                    "task": task, "split": "train", "idx_a": a, "idx_b": b,
                    "key_a": rows[a]["key"], "key_b": rows[b]["key"],
                    "canonical_a": rows[a]["canonical"],
                    "canonical_b": rows[b]["canonical"], "cos": round(c, 4)}) + "\n")
                n_task += 1
        total += n_task
        print(f"  {task}: {n_task} new pairs")
    out.close()
    print(f"\nwrote {total} new train pairs -> {OUT/'more_train_pairs.jsonl'}")


if __name__ == "__main__":
    main()

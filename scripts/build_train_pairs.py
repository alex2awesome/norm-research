"""Generate per-task labelled pairs to train a per-task LoRA on bge-large.

For each task: pool the task's canonical forms across all three buckets, split
the RUBRICS 85/15 into train/eval (so eval pairs never share a rubric with
train -- no leakage), then sample pairs stratified by current bge cosine so
each split has plenty of hard negatives (mid-cosine related-but-different
pairs). The pairs are judged afterward by the v6 LLM judge on sk3.

Output: train_pairs.jsonl, eval_pairs.jsonl
        {task, split, idx_a, idx_b, key_a, key_b, canonical_a, canonical_b, cos}
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
TRAIN_PER_BAND = 1700   # ~12K train pairs/task
EVAL_PER_BAND = 430     # ~3K eval pairs/task
RNG = np.random.default_rng(11)


def load_task(task):
    """Pool the task's canonical forms + bge embeddings across all buckets."""
    rows, embs = [], []
    by_bt = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        if r["task"] == task:
            by_bt[r["bucket"]].append(r)
    for bucket, rs in sorted(by_bt.items()):
        rs.sort(key=lambda r: r["idx"])
        p = EMB / f"emb_bge_{bucket}_{task}.npy"
        if not p.exists():
            continue
        e = np.load(p).astype(np.float64)
        if len(e) != len(rs):
            continue
        rows += rs
        embs.append(e)
    if not embs:
        return [], None
    emb = np.vstack(embs)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return rows, emb


def sample_pairs(idxs, emb, per_band):
    """Stratified pairs among the given rubric indices."""
    idxs = np.array(idxs)
    sub = emb[idxs]
    n = len(idxs)
    cand = {}
    sims = sub @ sub.T
    np.fill_diagonal(sims, -1.0)
    k = min(6, n - 1)
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    for i in range(n):
        for j in nn[i]:
            a, b = sorted((i, int(j)))
            if a != b:
                cand[(a, b)] = float(sub[a] @ sub[b])
    ri = RNG.integers(0, n, size=12 * n)
    rj = RNG.integers(0, n, size=12 * n)
    for a, b in zip(ri, rj):
        if a != b:
            a, b = sorted((int(a), int(b)))
            cand[(a, b)] = float(sub[a] @ sub[b])
    banded = defaultdict(list)
    for (a, b), c in cand.items():
        for lo, hi in BANDS:
            if lo <= c < hi:
                banded[(lo, hi)].append((int(idxs[a]), int(idxs[b]), c))
                break
    out = []
    for lo, hi in BANDS:
        pool = banded[(lo, hi)]
        if not pool:
            continue
        pick = RNG.choice(len(pool), size=min(per_band, len(pool)), replace=False)
        out += [pool[k] for k in pick]
    return out


def main():
    tr = open(OUT / "train_pairs.jsonl", "w")
    ev = open(OUT / "eval_pairs.jsonl", "w")
    n_tr = n_ev = 0
    for task in TASKS:
        rows, emb = load_task(task)
        if emb is None or len(rows) < 50:
            print(f"  {task}: skipped ({len(rows)} rows)")
            continue
        n = len(rows)
        perm = RNG.permutation(n)
        cut = int(n * 0.85)
        train_idx, eval_idx = perm[:cut], perm[cut:]
        for split, idxs, fh, per_band, ctr in [
                ("train", train_idx, tr, TRAIN_PER_BAND, "n_tr"),
                ("eval", eval_idx, ev, EVAL_PER_BAND, "n_ev")]:
            pairs = sample_pairs(list(idxs), emb, per_band)
            for a, b, c in pairs:
                fh.write(json.dumps({
                    "task": task, "split": split, "idx_a": a, "idx_b": b,
                    "key_a": rows[a]["key"], "key_b": rows[b]["key"],
                    "canonical_a": rows[a]["canonical"],
                    "canonical_b": rows[b]["canonical"], "cos": round(c, 4)}) + "\n")
            if split == "train":
                n_tr += len(pairs)
            else:
                n_ev += len(pairs)
        print(f"  {task}: {n} rubrics -> train+eval pairs sampled")
    tr.close()
    ev.close()
    print(f"\nwrote {n_tr} train pairs, {n_ev} eval pairs")


if __name__ == "__main__":
    main()

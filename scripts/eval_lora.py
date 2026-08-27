"""Held-out FP/FN comparison: baseline bge vs. per-task LoRA bge.

The satisfaction gate. Clusters each task's canonical forms (complete-linkage,
cosine) at a tau sweep, with two embeddings -- stock bge-large and the per-task
LoRA-tuned bge -- and measures FP/FN against the v6 judge on the HELD-OUT eval
pairs (split=="eval", rubrics never seen in LoRA training).

If the LoRA embedding bows the frontier inward (lower FP and FN at the same
tau), the per-task distillation worked.

FP% = of judged eval pairs the clustering MERGED, fraction the judge scored !=2
FN% = of judged eval pairs the clustering SPLIT,   fraction the judge scored ==2
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
EMB_BGE = ROOT / "notebooks" / "_explore_cache" / "bge"
EMB_LORA = ROOT / "notebooks" / "_explore_cache" / "lora"
FORMS = OUT / "canon_all_real_forms.jsonl"
VERDICTS = OUT / "all_verdicts.jsonl"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
TAUS = [0.985, 0.97, 0.955, 0.94, 0.925, 0.91, 0.895, 0.88, 0.86, 0.84]


def pooled(task, emb_dir, prefix):
    """Pool a task's bucket embeddings in the canonical (sorted-bucket) order;
    return (key->row index) map and the stacked embedding."""
    by_bt = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        if r["task"] == task:
            by_bt[r["bucket"]].append(r)
    keymap, mats, off = {}, [], 0
    for bucket in sorted(by_bt):
        rs = sorted(by_bt[bucket], key=lambda r: r["idx"])
        p = emb_dir / f"{prefix}_{bucket}_{task}.npy"
        if not p.exists():
            return None, None
        e = np.load(p).astype(np.float64)
        if len(e) != len(rs):
            return None, None
        for r in rs:
            keymap[r["key"]] = off + r["idx"]
        mats.append(e)
        off += len(rs)
    return keymap, np.vstack(mats)


def task_linkage(emb):
    """One complete-linkage dendrogram per task; cut at any tau with fcluster."""
    if len(emb) < 2:
        return None
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return linkage(pdist(e, metric="cosine"), method="complete")


def main():
    ev = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "eval" and v.get("score") in (0, 1, 2):
            ev[v["task"]].append(v)
    n_ev = sum(len(x) for x in ev.values())
    print(f"{n_ev} held-out eval pairs across {len(ev)} tasks")

    results = {}
    for name, emb_dir, prefix in [("bge", EMB_BGE, "emb_bge"),
                                  ("lora", EMB_LORA, "emb_lora")]:
        # labels[(task,tau)] = cluster labels on the pooled embedding
        labels, kmap = {}, {}
        for task in TASKS:
            km, emb = pooled(task, emb_dir, prefix)
            if emb is None:
                continue
            kmap[task] = km
            Z = task_linkage(emb)
            for tau in TAUS:
                labels[(task, tau)] = (np.zeros(len(emb), dtype=int) if Z is None
                                       else fcluster(Z, t=1.0 - tau,
                                                     criterion="distance"))
        per_tau = {}
        for tau in TAUS:
            merged = merged_bad = split = split_same = 0
            for task, pairs in ev.items():
                km = kmap.get(task)
                lab = labels.get((task, tau))
                if km is None or lab is None:
                    continue
                for v in pairs:
                    ia, ib = km.get(v["key_a"]), km.get(v["key_b"])
                    if ia is None or ib is None:
                        continue
                    same_cluster = lab[ia] == lab[ib]
                    if same_cluster:
                        merged += 1
                        if v["score"] != 2:
                            merged_bad += 1
                    else:
                        split += 1
                        if v["score"] == 2:
                            split_same += 1
            per_tau[tau] = (merged_bad / merged * 100 if merged else 0.0,
                            split_same / split * 100 if split else 0.0)
        results[name] = per_tau

    print(f"\n{'='*66}")
    print("HELD-OUT FP / FN  -  baseline bge  vs  per-task LoRA")
    print(f"{'='*66}")
    print(f"{'tau':>7} | {'bge FP':>8} {'bge FN':>8} | {'lora FP':>9} {'lora FN':>9}")
    for tau in TAUS:
        bfp, bfn = results["bge"][tau]
        lfp, lfn = results["lora"][tau]
        print(f"{tau:>7.3f} | {bfp:>7.1f}% {bfn:>7.1f}% | {lfp:>8.1f}% {lfn:>8.1f}%")


if __name__ == "__main__":
    main()

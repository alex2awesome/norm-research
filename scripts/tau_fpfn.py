"""FP / FN of metric clustering vs. agglomeration threshold tau.

Clusters the canonical forms (nemotron embeddings, complete-linkage cosine) per
(bucket, task) at a range of tau. The judged pair pool (judge_verdicts.jsonl)
gives ground truth. Per tau, per bucket:

  FP% = of judged pairs the clustering MERGED (same cluster),
        the fraction the judge says are DIFFERENT      -> over-merging
  FN% = of judged pairs the clustering SPLIT (diff cluster),
        the fraction the judge says are the SAME       -> under-merging

Expectation: stricter tau -> FN% up, FP% down. The crossover is the operating
point. (Absolute levels are relative to the stratified pool; the trend and
crossover are the deliverable.)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
EMB = ROOT / "notebooks" / "_explore_cache" / "bge"
FORMS = OUT / "canon_all_real_forms.jsonl"
VERDICTS = OUT / "judge_verdicts.jsonl"

TAUS = [0.985, 0.97, 0.955, 0.94, 0.925, 0.91, 0.895, 0.88, 0.86, 0.84]


def main():
    n_by_bt = {}
    for line in FORMS.open():
        r = json.loads(line)
        n_by_bt[(r["bucket"], r["task"])] = n_by_bt.get((r["bucket"], r["task"]), 0) + 1

    verdicts = [json.loads(l) for l in VERDICTS.open()]
    # graded judge: score 2 = same/mergeable, 0/1 = different
    verdicts = [v for v in verdicts if v.get("score") in (0, 1, 2)]
    for v in verdicts:
        v["same"] = (v["score"] == 2)
    buckets = sorted({v["bucket"] for v in verdicts})
    print(f"{len(verdicts)} judged pairs | buckets={buckets}")
    for b in buckets:
        vb = [v for v in verdicts if v["bucket"] == b]
        print(f"  {b}: {len(vb)} pairs, {sum(v['same'] for v in vb)} judged SAME")

    # cluster labels per (bucket, task, tau)
    bt_keys = {(v["bucket"], v["task"]) for v in verdicts}
    labels = {}  # (bucket, task, tau) -> label array
    for (bucket, task) in bt_keys:
        p = EMB / f"emb_bge_{bucket}_{task}.npy"
        if not p.exists():
            continue
        emb = np.load(p).astype(np.float64)
        for tau in TAUS:
            if len(emb) < 2:
                labels[(bucket, task, tau)] = np.zeros(len(emb), dtype=int)
                continue
            m = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                        linkage="complete",
                                        distance_threshold=1.0 - tau)
            labels[(bucket, task, tau)] = m.fit_predict(emb)

    print(f"\n{'='*78}")
    print("FP / FN vs tau   (FP% = wrong merges; FN% = missed merges)")
    print(f"{'='*78}")
    rows_out = []
    for bucket in buckets:
        vb = [v for v in verdicts if v["bucket"] == bucket]
        print(f"\n--- {bucket}  ({len(vb)} pairs) ---")
        print(f"  {'tau':>7} {'merged':>8} {'FP%':>7} {'split':>8} {'FN%':>7}")
        for tau in TAUS:
            merged_same = merged = split_same = split = 0
            for v in vb:
                lab = labels.get((bucket, v["task"], tau))
                if lab is None:
                    continue
                same_cluster = lab[v["idx_a"]] == lab[v["idx_b"]]
                if same_cluster:
                    merged += 1
                    if not v["same"]:
                        merged_same += 1   # merged but judged DIFFERENT
                else:
                    split += 1
                    if v["same"]:
                        split_same += 1    # split but judged SAME
            fp = merged_same / merged * 100 if merged else 0.0
            fn = split_same / split * 100 if split else 0.0
            print(f"  {tau:>7.3f} {merged:>8} {fp:>6.1f}% {split:>8} {fn:>6.1f}%")
            rows_out.append({"bucket": bucket, "tau": tau, "n_merged": merged,
                             "fp_pct": fp, "n_split": split, "fn_pct": fn})

    import pandas as pd
    pd.DataFrame(rows_out).to_parquet(OUT / "tau_fpfn.parquet")
    print(f"\nwrote {OUT/'tau_fpfn.parquet'}")


if __name__ == "__main__":
    main()

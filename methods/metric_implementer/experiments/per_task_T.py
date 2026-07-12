"""Per-task transmission T_TVD over the consistency longtable (ZERO GPU), for the specificity↔T join.

One fixed strong judge (Llama-3.1-8B, by filename) keeps T comparable across tasks. Per clean cell
(metric_id, version_id): s_i = per-item mean score over passes; T = MAD(s) = mean_i|s_i − s̄|. Per task =
median T over non-degenerate cells. Output: per_task_T.json {task: {T_median, n_clean, judge}}.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

COLS = ["task", "metric_id", "version_id", "pass", "item_id", "score", "applicable"]


def main():
    fs = sorted(glob.glob("outputs/metric_implementer_scale/search1/longtable/"
                          "sampled_meta-llama_Llama-3.1-8B-Instruct__*.parquet"))
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_parquet(f, columns=COLS))
        except Exception:
            d = pd.read_parquet(f)
            parts.append(d[[c for c in COLS if c in d.columns]])
    df = pd.concat(parts, ignore_index=True)
    if "applicable" in df.columns:
        df = df[df["applicable"] != False]  # noqa: E712
    print(f"rows={len(df)}  tasks={sorted(df['task'].unique())}")
    out = {}
    for task, g in df.groupby("task"):
        Ts = []
        for _, gc in g.groupby(["metric_id", "version_id"]):
            s = gc.groupby("item_id")["score"].mean().values
            if len(s) < 8 or np.std(s) < 1e-6:
                continue
            Ts.append(float(np.mean(np.abs(s - s.mean()))))
        if Ts:
            out[task] = {"T_median": float(np.median(Ts)), "n_clean": len(Ts)}
    json.dump(out, open("outputs/metric_implementer_scale/per_task_T.json", "w"), indent=1)
    for t, v in sorted(out.items(), key=lambda x: -x[1]["T_median"]):
        print(f"{t:24s} T={v['T_median']:.3f}  clean_cells={v['n_clean']}")


if __name__ == "__main__":
    main()

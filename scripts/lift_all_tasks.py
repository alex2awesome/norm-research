"""Per-task aspect-label lift across all tasks and available judges in cells_v1.

For each (task, judge) pair: load labels from runs/validity_full/v2/<task>/datapoints.json,
load cells from outputs/v2_db/cells_v1/task=<task>/judge=<judge>/data.parquet, then for
each aspect compute delta = mean(score|y=1) - mean(score|y=0) and Welch t-test p.
Reports per-task: number of testable aspects, count surviving Bonferroni, count with
|delta|>.05, max |delta|.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASKS = [
    "peer_review", "math", "notice_and_comment", "press_releases",
    "humor", "news_homepages", "patents", "code_review", "creative_writing",
]
JUDGES = ["qwen_thinking_fp8", "claude"]


def task_lift(task: str, judge: str):
    labels_p = REPO / f"runs/validity_full/v2/{task}/datapoints.json"
    if not labels_p.exists():
        return None
    dps = json.loads(labels_p.read_text())
    labels = pd.Series(
        {d["datapoint_id"]: int(d["judgement"])
         for d in dps if d.get("judgement") is not None}
    )
    f = REPO / f"outputs/v2_db/cells_v1/task={task}/judge={judge}/data.parquet"
    if not f.exists():
        return None
    cells = pd.read_parquet(f)
    cells["score_num"] = cells["score"].where(cells["applicable"], np.nan).astype(float)
    mat = (cells.groupby(["datapoint_id", "aspect_id"])["score_num"]
                .mean().unstack("aspect_id"))
    common = mat.index.intersection(labels.index)
    n_labels = len(labels)
    n_overlap = len(common)
    n_aspects = mat.shape[1]
    if n_overlap < 20:
        return (task, judge, n_labels, n_overlap, n_aspects, 0, 0, 0, 0, 0.0)
    mat_a = mat.loc[common]
    y = labels.loc[common]
    deltas, pvals = [], []
    for a in mat_a.columns:
        col = mat_a[a]
        mask = col.notna()
        if mask.sum() < 10:
            continue
        y_a = y[mask]
        s_a = col[mask]
        s0 = s_a[y_a == 0]
        s1 = s_a[y_a == 1]
        if len(s0) < 5 or len(s1) < 5:
            continue
        deltas.append(float(s1.mean() - s0.mean()))
        try:
            _, p = ttest_ind(s1, s0, equal_var=False)
        except Exception:
            p = np.nan
        pvals.append(float(p))
    if not deltas:
        return (task, judge, n_labels, n_overlap, n_aspects, 0, 0, 0, 0, 0.0)
    deltas_arr = np.array(deltas)
    pvals_arr = np.array(pvals)
    testable = len(deltas_arr)
    bonf = np.clip(pvals_arr * testable, 0, 1)
    return (
        task, judge, n_labels, n_overlap, n_aspects, testable,
        int((bonf < 0.05).sum()),
        int((np.abs(deltas_arr) > 0.05).sum()),
        int((np.abs(deltas_arr) > 0.10).sum()),
        float(np.nanmax(np.abs(deltas_arr))),
    )


def main():
    results = []
    for task in TASKS:
        for judge in JUDGES:
            r = task_lift(task, judge)
            if r is not None:
                results.append(r)

    cols = ["task", "judge", "lbls", "dp_overlap", "aspects",
            "testable", "bonf<.05", "|d|>.05", "|d|>.10", "max|d|"]
    print(f"{cols[0]:<22} {cols[1]:<22} {cols[2]:>5} {cols[3]:>10} "
          f"{cols[4]:>7} {cols[5]:>8} {cols[6]:>8} {cols[7]:>8} "
          f"{cols[8]:>8} {cols[9]:>7}")
    print("-" * 120)
    for r in results:
        print(f"{r[0]:<22} {r[1]:<22} {r[2]:>5} {r[3]:>10} "
              f"{r[4]:>7} {r[5]:>8} {r[6]:>8} {r[7]:>8} "
              f"{r[8]:>8} {r[9]:>7.3f}")


if __name__ == "__main__":
    main()

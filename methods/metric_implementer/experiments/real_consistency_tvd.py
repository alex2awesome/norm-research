"""Rung-1/3 on the REAL consistency longtable, in TVD (zero-GPU, sk3 CPU).

Loads the search1 5-pass parquets -> channel_consistency (now TVD-enabled) -> per-cell same-f
transmission T_tvd = I_TVD(I;V) alongside the Shannon I_V, with:
  * Rung 1  : cap_gap = cap_TVD(½) − T_tvd  — how far real metrics sit below the universal cap.
  * Rung 3  : per (task, tier, metric), the best VERSION by T_tvd and whether its bootstrap CI
              dominates the other versions (certified best-in-set among the optimizer's candidates).
  * descriptive: T_tvd by operator (INIT vs mutated) — does the optimizer raise transmission?

CAVEAT (theory §4.3): this is the *consistency* channel = transmission T, the GAMEABLE leg. It is the
precondition + cap + ranking test, NOT recovery R (which needs the reconstruction LLM pass, #24).

Run on sk3:  PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
             $PY -m methods.metric_implementer.experiments.real_consistency_tvd [task] [n_boot]
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from methods.metric_implementer import vinfo as v

LT_GLOB = ("outputs/metric_implementer_scale/search1/longtable/sampled_*.parquet")
CAP_TVD = 0.5
GROUP = ["task", "judge_model", "metric_id", "version_id", "token_cap"]
COLS = ["task", "judge_model", "metric_id", "version_id", "operator", "token_cap",
        "pass", "item_id", "score", "applicable"]


def load(slice_task=None) -> pd.DataFrame:
    fs = sorted(glob.glob(LT_GLOB))
    dfs = []
    for f in fs:
        try:
            d = pd.read_parquet(f, columns=COLS)
        except Exception:
            d = pd.read_parquet(f)
            d = d[[c for c in COLS if c in d.columns]]
        if slice_task:
            d = d[d["task"] == slice_task]
        if len(d):
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=COLS)


def short_tier(m: str) -> str:
    return m.split("/")[-1].replace("-Instruct", "")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    df = load(task)
    if not len(df):
        print("no rows"); return
    print(f"rows={len(df):,}  tasks={sorted(df.task.unique())}")
    print(f"tiers={[short_tier(t) for t in sorted(df.judge_model.unique())]}")

    cells = v.channel_consistency(df, group_cols=GROUP, n_boot=n_boot, min_items=20)
    op = (df.groupby(GROUP)["operator"].first().reset_index())
    cells = cells.merge(op, on=GROUP, how="left")
    cells = cells[np.isfinite(cells["tvd_t"])].copy()
    cells["cap_gap"] = CAP_TVD - cells["tvd_t"]
    cells["tier"] = cells["judge_model"].map(short_tier)
    cells["clean"] = cells["flags"].astype(str).str.len() == 0
    n_flag = int((~cells["clean"]).sum())
    print(f"\nscored cells: {len(cells)}  (flagged degenerate: {n_flag}; clean: {cells['clean'].sum()})")

    # ---- Rung 1: how far below cap_TVD do real metrics sit, by task (CLEAN cells only) ----
    clean = cells[cells["clean"]]
    print("\n=== Rung 1 — T_tvd vs cap_TVD=0.5, by task (CLEAN cells; flagged/collapsed excluded) ===")
    print(f"{'task':>20} {'clean':>5} {'med_T_tvd':>9} {'med_capgap':>10} {'max_T_tvd':>9} {'med_IV(sh)':>10}")
    for t, g in clean.groupby("task"):
        if not len(g):
            print(f"{t:>20} {0:>5}  (all cells collapsed)"); continue
        print(f"{t:>20} {len(g):>5} {g['tvd_t'].median():>9.3f} {g['cap_gap'].median():>10.3f} "
              f"{g['tvd_t'].max():>9.3f} {g['iv_mm'].median():>10.3f}")

    # ---- Rung 3: per (task,tier,metric,token_cap) best VERSION at fixed budget, CI-dominance ----
    print("\n=== Rung 3 — best version per (task,tier,metric,token_cap); CI-certified best-in-set? ===")
    cert, total = 0, 0
    examples = []
    for (t, tier, mid, tok), g in clean.groupby(["task", "tier", "metric_id", "token_cap"]):
        if g["version_id"].nunique() < 2:
            continue
        total += 1
        g = g.sort_values("tvd_t", ascending=False)
        best = g.iloc[0]
        rivals = g.iloc[1:]
        dominates = bool((best["tvd_t_ci_lo"] > rivals["tvd_t_ci_hi"]).all())
        cert += dominates
        if dominates and len(examples) < 6:
            examples.append((t, tier, mid.split("__")[-1][:24], tok, int(g["version_id"].nunique()),
                             round(best["tvd_t"], 3), round(best["tvd_t_ci_lo"], 3), "CERTIFIED"))
    print(f"(task,tier,metric,budget) cells with >=2 versions: {total};  CI-certified single best: "
          f"{cert} ({100*cert/max(total,1):.0f}%);  rest = top groups (CI overlap)")
    for e in examples:
        print(f"  {e[0]:>14}/{e[1]:<14} {e[2]:<26} tok={e[3]} v={e[4]} bestT={e[5]} lo={e[6]} {e[7]}")

    # ---- descriptive: operator effect on T_tvd ----
    print("\n=== T_tvd by operator (does mutation raise transmission?) ===")
    for opn, g in cells.groupby("operator"):
        print(f"  {str(opn):>12}: n={len(g):>4}  med_T_tvd={g['tvd_t'].median():.3f}")


if __name__ == "__main__":
    main()

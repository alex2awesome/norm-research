#!/usr/bin/env python
"""Task #25/3a closure: FAMILY-VERDICT JOIN.

Rebuilds per-metric per-family agreement-vs-z curves directly from the OSL panels
(mbar285_<exec>.npz for humor, mbar2_<task>_<exec>.npz for creative_writing/peer_review/
math/news_homepages, + mbar2_humor_sup_<exec>.npz humor supplement), for three executor
families:
  llama  = llama1b, llama3b, llama8b, llama70b               (4 rungs)
  qwen25 = qwen25-3b, qwen25-7b, qwen25-14b, qwen25-32b, qwen25-72b   (5 rungs)
  qwen3  = qwen3-1.7b, qwen3-4b, qwen3-8b, qwen3-14b, qwen3-32b       (5 rungs)

Reference convention (per user spec, NOT the 11-exec crowd mean): binary verdict per item
per metric per executor = 1[m_bar(P(YES)) > 0.5]. Frontier = {llama70b, qwen25-72b}
majority verdict per item, TIES DROPPED (item excluded from that metric's frontier
reference whenever llama70b and qwen25-72b disagree). Agreement(exec, metric) = fraction
of frontier-defined items where exec's binary verdict matches the frontier verdict.

SELF-REFERENCE CAVEAT (own design decision, no existing script covers this case): llama70b
and qwen25-72b are themselves both family top-rungs AND the two frontier voters. Naively
scoring "agreement vs frontier" for a frontier member is trivially 1.0 (it always agrees
with itself on the items where it agreed with the other voter). Following the precedent in
outputs/osl_multi/zxa_fit.py (which excludes the scored executor from its own reference set
when e is a frontier member), we score frontier members' own agreement as direct pairwise
agreement with the OTHER frontier member alone (no ties-drop needed with one remaining
voter). This yields llama70b's and qwen25-72b's top-rung points as the SAME non-trivial
number per metric (their mutual pairwise agreement), never a trivial 1.0.

Metrics are restricted to kind=='bank' (excludes the small planted-control set used
elsewhere for capability-reference calibration).

Usage: python3 family_verdict_join_v1.py
"""
import json
import os
import numpy as np

B = "/lfs/skampere3/0/alexspan"
OSL = f"{B}/outputs/osl"
OM = f"{B}/outputs/osl_multi"

FAMILIES = {
    "llama": ["llama1b", "llama3b", "llama8b", "llama70b"],
    "qwen25": ["qwen25-3b", "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b"],
    "qwen3": ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"],
}
ALL_EXECS = sorted({e for fam in FAMILIES.values() for e in fam})
FRONTIER = ["llama70b", "qwen25-72b"]
TASKS = ["humor", "creative_writing", "peer_review", "math", "news_homepages"]

MIN_FRONTIER_ITEMS = 30   # min ties-dropped frontier-defined items to trust a metric's ref
MIN_EXEC_ITEMS = 20       # min valid items to trust one executor's agreement point
MIN_FAMILY_RUNGS = 3      # min rungs with a valid point to attempt a family OLS slope
RISE_THRESH = 0.01        # slope>+.01/z -> RISING; |slope|<=.01 -> FLAT; else FALLING


def zmap_load():
    z = {}
    missing = []
    for e in ALL_EXECS:
        p = f"{OSL}/{e}.json"
        if os.path.exists(p):
            z[e] = json.load(open(p))["battery"]["z"]
        else:
            missing.append(e)
    return z, missing


def load_panel_bank(path):
    d = np.load(path, allow_pickle=True)
    names, kinds, m = d["names"], d["kinds"], d["m_bar"]
    out = {}
    for i, n in enumerate(names):
        if str(kinds[i]) != "bank":
            continue
        out[str(n)] = m[i].astype(float)
    return out


def load_task_panels(task):
    """exec -> {metric_name: m_bar_vector(300,)} restricted to kind=='bank'."""
    panels = {}
    coverage = {}
    if task == "humor":
        for e in ALL_EXECS:
            d = {}
            p1 = f"{OSL}/mbar285_{e}.npz"
            p2 = f"{OM}/mbar2_humor_sup_{e}.npz"
            got = []
            if os.path.exists(p1):
                d.update(load_panel_bank(p1))
                got.append("mbar285")
            if os.path.exists(p2):
                sup = load_panel_bank(p2)
                overlap = set(sup) & set(d)
                if overlap:
                    print(f"  WARN humor {e}: {len(overlap)} name overlap mbar285/sup, "
                          f"keeping mbar285 version")
                    for n in overlap:
                        sup.pop(n)
                d.update(sup)
                got.append("humor_sup")
            if d:
                panels[e] = d
            coverage[e] = got
    else:
        for e in ALL_EXECS:
            p = f"{OM}/mbar2_{task}_{e}.npz"
            if os.path.exists(p):
                panels[e] = load_panel_bank(p)
                coverage[e] = ["mbar2"]
            else:
                coverage[e] = []
    return panels, coverage


def binarize(v):
    b = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    b[ok] = (v[ok] > 0.5).astype(float)
    return b


def agreement_curve(panels, name):
    """Returns {exec: (n_valid, agreement)} for every exec holding this metric."""
    if FRONTIER[0] not in panels or FRONTIER[1] not in panels:
        return None, 0
    if name not in panels[FRONTIER[0]] or name not in panels[FRONTIER[1]]:
        return None, 0
    b_l = binarize(panels[FRONTIER[0]][name])
    b_q = binarize(panels[FRONTIER[1]][name])
    both_fin = np.isfinite(b_l) & np.isfinite(b_q)
    agree_fq = both_fin & (b_l == b_q)          # frontier-defined items (ties dropped)
    n_frontier = int(agree_fq.sum())
    if n_frontier < MIN_FRONTIER_ITEMS:
        return None, n_frontier
    ref = b_l.copy()   # == b_q on agree_fq mask, value irrelevant elsewhere
    # frontier members' own points: pairwise agreement with the OTHER voter (self-excluded)
    pairwise = float(np.mean(b_l[both_fin] == b_q[both_fin])) if both_fin.sum() >= MIN_EXEC_ITEMS else None
    n_pairwise = int(both_fin.sum())

    out = {}
    for e, d in panels.items():
        if name not in d:
            continue
        if e in FRONTIER:
            if pairwise is not None:
                out[e] = (n_pairwise, pairwise)
            continue
        b_e = binarize(d[name])
        mask = agree_fq & np.isfinite(b_e)
        n = int(mask.sum())
        if n < MIN_EXEC_ITEMS:
            continue
        a = float(np.mean(b_e[mask] == ref[mask]))
        out[e] = (n, a)
    return out, n_frontier


def ols_slope(z, y):
    z, y = np.asarray(z, float), np.asarray(y, float)
    if len(z) < 2 or np.std(z) < 1e-9:
        return None
    zc = z - z.mean()
    return float(np.sum(zc * (y - y.mean())) / np.sum(zc * zc))


def classify(slope):
    if slope is None:
        return "INSUFFICIENT"
    if slope > RISE_THRESH:
        return "RISING"
    if slope < -RISE_THRESH:
        return "FALLING"
    return "FLAT"


def family_fit(fam_execs, points, zmap):
    """points: {exec: (n, agree)}. Returns dict or None if <MIN_FAMILY_RUNGS."""
    rungs = [(e, zmap[e], points[e][1], points[e][0]) for e in fam_execs
             if e in points and e in zmap]
    if len(rungs) < MIN_FAMILY_RUNGS:
        return dict(n_rungs=len(rungs), rungs=rungs, slope=None, verdict="INSUFFICIENT")
    rungs_sorted = sorted(rungs, key=lambda r: r[1])   # by z ascending
    z = [r[1] for r in rungs_sorted]
    y = [r[2] for r in rungs_sorted]
    slope = ols_slope(z, y)
    mid_idx = len(rungs_sorted) // 2
    top_y, mid_y = y[-1], y[mid_idx]
    return dict(n_rungs=len(rungs_sorted),
                rungs=[dict(exec=r[0], z=round(r[1], 4), agree=round(r[2], 4), n=r[3])
                       for r in rungs_sorted],
                slope=round(slope, 5) if slope is not None else None,
                top_minus_mid=round(top_y - mid_y, 4),
                top_y=round(top_y, 4), mid_y=round(mid_y, 4),
                verdict=classify(slope))


def main():
    zmap, zmiss = zmap_load()
    print("z map coverage:", {e: round(zmap[e], 3) for e in ALL_EXECS if e in zmap})
    if zmiss:
        print("MISSING z (battery.z) for:", zmiss)

    all_rows = []
    task_coverage = {}
    for task in TASKS:
        panels, coverage = load_task_panels(task)
        task_coverage[task] = {e: coverage.get(e, []) for e in ALL_EXECS}
        present = [e for e in ALL_EXECS if panels.get(e)]
        missing = [e for e in ALL_EXECS if e not in present]
        if missing:
            print(f"[{task}] MISSING panel executors: {missing}")
        if FRONTIER[0] not in panels or FRONTIER[1] not in panels:
            print(f"[{task}] SKIP: frontier pair not both present")
            continue
        names = sorted(set(panels[FRONTIER[0]]) & set(panels[FRONTIER[1]]))
        print(f"[{task}] {len(present)}/{len(ALL_EXECS)} execs present; "
              f"{len(names)} bank metrics with frontier coverage")

        n_skipped_frontier = 0
        n_metrics_scored = 0
        for name in names:
            points, n_frontier = agreement_curve(panels, name)
            if points is None:
                n_skipped_frontier += 1
                continue
            fam_fits = {fam: family_fit(execs, points, zmap) for fam, execs in FAMILIES.items()}
            all_rows.append(dict(task=task, name=name, n_frontier_items=n_frontier,
                                  families=fam_fits))
            n_metrics_scored += 1
        print(f"[{task}] scored {n_metrics_scored} metrics; "
              f"{n_skipped_frontier} skipped (< {MIN_FRONTIER_ITEMS} frontier-defined items "
              f"or frontier pair absent for that metric)")

    print(f"\nTOTAL metric-task rows scored: {len(all_rows)}")

    json.dump(dict(meta=dict(
        generated="family_verdict_join_v1.py", families=FAMILIES, frontier=FRONTIER,
        tasks=TASKS, min_frontier_items=MIN_FRONTIER_ITEMS, min_exec_items=MIN_EXEC_ITEMS,
        min_family_rungs=MIN_FAMILY_RUNGS, rise_thresh=RISE_THRESH,
        self_reference_note=("llama70b/qwen25-72b top-rung points = pairwise agreement "
                              "with the OTHER frontier voter only (self excluded), never "
                              "trivial 1.0 self-agreement"),
        zmap={e: zmap.get(e) for e in ALL_EXECS}, task_coverage=task_coverage),
        rows=all_rows), open(f"{OM}/family_verdict_join_v1_raw.json", "w"), indent=1)
    print(f"-> wrote {OM}/family_verdict_join_v1_raw.json")


if __name__ == "__main__":
    main()

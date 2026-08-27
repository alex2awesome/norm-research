"""Phase 3 + Phase 4 — optimizer comparison table and the prompt-optimality scaling estimators on
BOTH arms' raw proposal logs and rescore matrices. Run with the REPO python (needs scipy), not the
folder venv:  python3 analyze.py

Estimators are imported from the main repo (code import is fine; DATA never crosses the boundary —
all inputs/outputs stay in this folder).

Phase 4 readouts per (dataset x arm):
  * discovery: cumulative distinct candidate prompts over raw proposal-evaluation draws (TRUE draw
    order incl. rejected candidates — no survivor bias) -> Heaps fit + moving-block unit-bootstrap
    CI (fully licensed here, unlike the main repo's registry curve).
  * value (exchangeable over distinct candidates, from rescore matrices):
      best-of-m:  E[max val-accuracy among m uniformly drawn candidates]
      union-of-m: E[fraction of val items solved by >=1 of m drawn candidates]
    + saturating fits and the redundancy read union(all) vs sum of individual coverages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))                      # repo root, code-only import
from methods.metric_implementer.experiments import unseen_value_scaling as uvs  # noqa: E402

DATASETS = ("hover", "hotpotqa", "aime2025")
ARMS = ("official", "inhouse", "unitrecomb")


def _norm_hash(text):
    import hashlib
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def discovery_from_log(plog: Path) -> dict:
    rows = [json.loads(l) for l in open(plog)]
    rows.sort(key=lambda r: r["ts"])
    seen, curve = set(), []
    for r in rows:
        seen.add(_norm_hash(next(iter(r["candidate"].values()))))
        curve.append(len(seen))
    m = np.arange(1, len(curve) + 1, dtype=float)
    fit = uvs.fit_power_law(m, np.asarray(curve, float), n_boot=0)
    inc = np.diff(np.concatenate([[0], curve])).astype(int)
    ub = uvs.heaps_unit_bootstrap_ci(inc, n_boot=500)
    return {"n_draws": len(curve), "n_distinct": curve[-1], "curve": curve,
            "alpha": fit.get("alpha"),
            "alpha_unit_ci": ub.get("alpha_ci") if ub.get("ok") else None}


def value_curves_from_rescore(rpath: Path, *, n_subsets: int = 300, seed: int = 0) -> dict:
    rows, _seen = [], set()
    for l in open(rpath):                       # two resumable rescorers may double-append
        r = json.loads(l)
        if r["hash"] not in _seen:
            _seen.add(r["hash"])
            rows.append(r)
    if len(rows) < 3:
        return {"ok": False, "reason": f"only {len(rows)} rescored candidates"}
    M = np.array([r["item_scores"] for r in rows], float)        # (n_cand, n_items)
    means = M.mean(1)
    rng = np.random.default_rng(seed)
    n = len(rows)
    m_grid = np.unique(np.linspace(1, n, min(10, n)).astype(int))
    best, union = [], []
    for m in m_grid:
        b, u = [], []
        for _ in range(n_subsets):
            idx = rng.choice(n, int(m), replace=False)
            b.append(means[idx].max())
            u.append(M[idx].max(0).mean())
        best.append(float(np.mean(b)))
        union.append(float(np.mean(u)))
    sat_b = uvs.fit_saturating(m_grid.astype(float), np.array(best), n_boot=0)
    sat_u = uvs.fit_saturating(m_grid.astype(float), np.array(union), n_boot=0)
    sum_cov = float(means.sum())                                 # additive analog
    union_all = float(M.max(0).mean())                           # joint analog
    return {"ok": True, "n_candidates": n, "m": m_grid.tolist(),
            "best_of_m": best, "union_of_m": union,
            "sat_best": {k: sat_b.get(k) for k in ("ok", "y_inf", "tau")},
            "sat_union": {k: sat_u.get(k) for k in ("ok", "y_inf", "tau")},
            "seed_val": float(means[0]), "best_val": float(means.max()),
            "union_all_items": union_all, "sum_individual_cov": sum_cov,
            "redundancy_read": (f"union(all {n}) covers {union_all:.2f} of items vs additive sum "
                                f"{sum_cov:.2f} -> overlap is the gap")}


def main():
    summary = {"phase3": [], "phase4": {}}
    for ds in DATASETS:
        for arm in ARMS:
            rd = HERE / "runs" / ds / arm
            if not (rd / "proposals.jsonl").exists():
                continue
            entry = {"dataset": ds, "arm": arm}
            res = rd / "result.json"
            if res.exists():
                r = json.loads(res.read_text())
                entry["final_val"] = r.get("val_score") or (
                    max(r["val_aggregate_scores"]) if r.get("val_aggregate_scores") else None)
                entry["lm_calls"] = r.get("task_lm_calls")
            disc = discovery_from_log(rd / "proposals.jsonl")
            entry["n_draws"], entry["n_distinct_prompts"] = disc["n_draws"], disc["n_distinct"]
            entry["heaps_alpha"], entry["heaps_alpha_unit_ci"] = disc["alpha"], disc["alpha_unit_ci"]
            vc = None
            if (rd / "rescore.jsonl").exists():
                vc = value_curves_from_rescore(rd / "rescore.jsonl")
            summary["phase3"].append(entry)
            summary["phase4"][f"{ds}/{arm}"] = {"discovery": disc, "value": vc}

    out = HERE / "runs" / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))

    lines = ["# prompt-optimality-test — Phase 3/4 summary\n",
             "| dataset | arm | final val | draws | distinct prompts | Heaps α [unit CI] | "
             "best-of-m τ | union τ |", "|---|---|---|---|---|---|---|---|"]
    for e in summary["phase3"]:
        v = (summary["phase4"][f"{e['dataset']}/{e['arm']}"]["value"]) or {}
        tb = v.get("sat_best", {}).get("tau") if v.get("ok") else None
        tu = v.get("sat_union", {}).get("tau") if v.get("ok") else None
        ci = e.get("heaps_alpha_unit_ci")
        a = e.get("heaps_alpha")
        lines.append(f"| {e['dataset']} | {e['arm']} | {e.get('final_val')} | {e['n_draws']} | "
                     f"{e['n_distinct_prompts']} | {f'{a:.2f}' if a is not None else '—'} "
                     f"{[round(x,2) for x in ci] if ci else ''} | "
                     f"{f'{tb:.1f}' if tb else '—'} | {f'{tu:.1f}' if tu else '—'} |")
    (HERE / "runs" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out} and runs/summary.md")


if __name__ == "__main__":
    main()

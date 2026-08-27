"""EXP-GTK-1 tally — the prereg'd P1 gate + battery double-differences, from grids on disk.

Replaces the ad-hoc v1 readout with a reproducible instrument. Statistics follow the frozen
prereg (notes/2026-07-22__exp-gtk-1-prereg.md):

  - per-cell rho = ADVERSE (min over the source's name-arm forms) Spearman vs the target's
    mean-over-forms name vector; mean-over-forms rho reported as a secondary column.
  - humor cells restricted to ITEM-HALF-2 (stable hash, salt `exp_gtk1`); cross-domain
    batteries (n&c, math) use all items (no training exposure).
  - P1 gate: mean over A-cells of (real - fresh) adverse-rho >= +.15. P1 fails -> every
    downstream number is EXPLORATORY.
  - GTK(B) double differences vs BOTH controls, cluster bootstrap over cells (10k, percentile
    CI) — threshold-free readouts.

B3 (n&c) and B4 (math) memberships are derived deterministically from the frozen executor
grids with the same failure-set predicate as the design builder (gap > .10 and best < .70);
full-domain rows are reported alongside.

CPU-only; run from repo root:
  python -m methods.tacit_channels.channels.gtk.tally_gtk1 \
      --scores-version scores_v1c --out outputs/tacit_channels/exp_gtk1/tally_v1c.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import (
    cell_stats, load_grid, spearman, stable_split,
)

BASE = "notebooks/data/two_faces_20260702"
SCORES_ROOT = f"{BASE}/family_scores_qwen25"
TARGET_JOB = "qwen25_72b_name_target"
FRESH_JOB = "qwen25_7b_executor"
PACKET_ROOT = f"{BASE}/tacit_breadth_item_partitions_v2"
EXP_ROOT = "outputs/tacit_channels/exp_gtk1"
ARMS = ("real", "shuffled", "construct_permuted")
DOMAINS = ("humor", "notice-and-comment", "math-stackexchange")
P1_BAR = 0.15
SALT = "exp_gtk1"


def half2_mask(domain: str) -> np.ndarray:
    payload = _apparatus.load_domain_items(
        PACKET_ROOT, domain, partitions=["tacit_breadth_search"])
    hashes = payload.get("hashes") or []
    if not hashes:
        import hashlib
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in payload["texts"]]
    return np.array([stable_split(h, 0.5, salt=SALT) != "train" for h in hashes])


def rhos_for_cell(src: dict, cell: str, t_ref: np.ndarray, mask: np.ndarray):
    """(adverse, mean_forms) Spearman of a source's name-arm forms vs the target reference."""
    forms = [v for (c, a, _f), v in src.items() if c == cell and a == "name"]
    if not forms:
        return None, None
    vals = [spearman(np.asarray(v)[mask], t_ref[mask]) for v in forms]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return None, None
    return float(min(vals)), float(np.mean(vals))


def failure_set(domain: str) -> tuple[list[str], list[str]]:
    """(failure_cells, all_cells) from the frozen grids — same predicate as the builder."""
    tgt, _ = load_grid(SCORES_ROOT, TARGET_JOB, domain)
    exe, emeta = load_grid(SCORES_ROOT, FRESH_JOB, domain)
    cells = sorted({c for (c, a, f) in tgt})
    fails = []
    for c in cells:
        row = cell_stats(tgt, exe, emeta, c)
        if row and (row.get("gap") or 0) > .10 and (
                row.get("best_rho") is None or row["best_rho"] < .70):
            fails.append(c)
    return fails, cells


def battery_summary(rows: list[dict], cells: list[str], rng: np.random.Generator,
                    n_boot: int = 10_000) -> dict | None:
    by_cell = {r["cell_id"]: r for r in rows if r["cell_id"] in set(cells)}
    use = [by_cell[c] for c in cells if c in by_cell
           and all(by_cell[c].get(k) is not None
                   for k in ("adv_fresh", "adv_real", "adv_shuffled",
                             "adv_construct_permuted"))]
    if not use:
        return None
    d_real = np.array([r["adv_real"] - r["adv_fresh"] for r in use])
    d_shuf = np.array([r["adv_shuffled"] - r["adv_fresh"] for r in use])
    d_perm = np.array([r["adv_construct_permuted"] - r["adv_fresh"] for r in use])

    def ci(vals: np.ndarray) -> list[float]:
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means = vals[idx].mean(axis=1)
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    return {
        "n_cells": len(use),
        "delta_real": float(d_real.mean()), "delta_real_ci": ci(d_real),
        "delta_shuffled": float(d_shuf.mean()),
        "delta_permuted": float(d_perm.mean()),
        "gtk_vs_shuffled": float((d_real - d_shuf).mean()),
        "gtk_vs_shuffled_ci": ci(d_real - d_shuf),
        "gtk_vs_permuted": float((d_real - d_perm).mean()),
        "gtk_vs_permuted_ci": ci(d_real - d_perm),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-version", default="scores_v1c",
                    help="subdir of exp_gtk1 holding <arm>/grid_*.npz (scores | scores_v1c)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    design = json.load(open(f"{EXP_ROOT}/design_manifest.json"))
    adapter_root = f"{EXP_ROOT}/{args.scores_version}"
    rng = np.random.default_rng(20260723)

    all_rows: list[dict] = []
    for domain in DOMAINS:
        tgt, _ = load_grid(SCORES_ROOT, TARGET_JOB, domain)
        fresh, _m = load_grid(SCORES_ROOT, FRESH_JOB, domain)
        arm_grids = {}
        for arm in ARMS:
            g, _ = load_grid(adapter_root, arm, domain)
            if g:
                arm_grids[arm] = g
        mask = half2_mask(domain) if domain == "humor" else None
        cells = sorted({c for (c, a, f) in tgt})
        for cell in cells:
            t_forms = [v for (c, a, f), v in tgt.items() if c == cell and a == "name"]
            if not t_forms:
                continue
            t_ref = np.mean(t_forms, axis=0)
            m = mask if mask is not None else np.ones(len(t_ref), dtype=bool)
            row = {"cell_id": cell, "domain": domain,
                   "n_items_eval": int(m.sum())}
            adv, mf = rhos_for_cell(fresh, cell, t_ref, m)
            row["adv_fresh"], row["mf_fresh"] = adv, mf
            for arm, g in arm_grids.items():
                adv, mf = rhos_for_cell(g, cell, t_ref, m)
                row[f"adv_{arm}"], row[f"mf_{arm}"] = adv, mf
            if any(row.get(f"adv_{arm}") is not None for arm in ARMS):
                all_rows.append(row)

    # ---- P1 gate (A-cells, humor, half-2, adverse) --------------------------------
    a_cells = design["A"]
    a_rows = [r for r in all_rows if r["cell_id"] in set(a_cells)
              and r.get("adv_real") is not None and r.get("adv_fresh") is not None]
    gains = np.array([r["adv_real"] - r["adv_fresh"] for r in a_rows])
    p1 = {
        "n": len(a_rows),
        "mean_gain": float(gains.mean()) if len(gains) else None,
        "median_gain": float(np.median(gains)) if len(gains) else None,
        "n_positive": int((gains > 0).sum()),
        "n_ge_bar": int((gains >= P1_BAR).sum()),
        "bar": P1_BAR,
        "pass": bool(len(gains) and gains.mean() >= P1_BAR),
    }

    # ---- batteries ----------------------------------------------------------------
    nc_fail, nc_all = failure_set("notice-and-comment")
    math_fail, math_all = failure_set("math-stackexchange")
    batteries = {
        "A_trained": design["A"],
        "B1_heldout_humor_failure": design["B1"],
        "B2_humor_success": design.get("B2_success", []),
        "B3_nc_failure": nc_fail,
        "B3_nc_all": nc_all,
        "B4_math_all": math_all,
        "B4_math_failure": math_fail,
    }
    summary = {name: battery_summary(all_rows, cells, rng, args.n_boot)
               for name, cells in batteries.items()}

    out = {
        "schema": "exp_gtk1_tally/v1",
        "scores_version": args.scores_version,
        "prereg_sha256": design.get("prereg_sha256"),
        "statistic": "adverse_rho_min_over_forms_vs_target_mean; humor=item_half2",
        "p1": p1,
        "batteries": summary,
        "battery_sizes": {k: len(v) for k, v in batteries.items()},
        "per_cell": all_rows,
    }
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"P1 [{args.scores_version}] n={p1['n']} mean {p1['mean_gain']:+.3f} "
          f"median {p1['median_gain']:+.3f} pos {p1['n_positive']}/{p1['n']} "
          f">=bar {p1['n_ge_bar']} -> {'PASS' if p1['pass'] else 'FAIL'}")
    for name, s in summary.items():
        if s is None:
            print(f"{name:28s} (no scored cells)")
            continue
        print(f"{name:28s} n={s['n_cells']:3d} dReal {s['delta_real']:+.3f} "
              f"CI[{s['delta_real_ci'][0]:+.3f},{s['delta_real_ci'][1]:+.3f}] "
              f"GTKvShuf {s['gtk_vs_shuffled']:+.3f} "
              f"CI[{s['gtk_vs_shuffled_ci'][0]:+.3f},{s['gtk_vs_shuffled_ci'][1]:+.3f}] "
              f"GTKvPerm {s['gtk_vs_permuted']:+.3f} "
              f"CI[{s['gtk_vs_permuted_ci'][0]:+.3f},{s['gtk_vs_permuted_ci'][1]:+.3f}]")
    print(f"tally -> {args.out}")


if __name__ == "__main__":
    main()

"""Thorn (a): the PROMPT-SUBSPACE CAP — an empirical all-articulation upper bound (v0).

Goal: upgrade "no articulation we tried reaches the target" toward "no articulation CAN",
per cell, in OUR setting. The DPI fixed-target cap (M_omega audit line) certifies all-prompt
bounds in the M_omega setting; the executor analog here is built from a different, measurable
structural fact:

  Every articulation induces a score vector s_p = f_exec(template(p, x)) over the SAME items.
  The observed family {s_p} (every arm x form of the bank = 50+ diverse articulations per
  cell) spans a subspace of rank-space behavior. IF that subspace SATURATES as articulations
  accumulate (new prompts add no new directions - measurable), then the reachable set of any
  further articulation is (approximately) inside it, and

      cap(cell) = max corr(z_target, v)  over v in span{ranked s_p}

  bounds what ANY articulation blend can achieve. Assumption to state with every use: unseen
  prompts stay (near) the saturated subspace - falsifiable by any prompt that escapes it;
  GEPA-optimized prompts are the designated escape attempt (Tier 1 search doubles as the
  subspace stress test).

Honesty guards:
  - v0 approximation: we work in RANK space (centered ranks per row); "span" there covers any
    monotone readout of any linear blend of observed behaviors. Label all outputs v0.
  - 90 items invite overfit: coefficients are fit on item-half-1, the reported cap is
    evaluated OUT-OF-SAMPLE on item-half-2 (cap_oos is the quotable number).
  - Saturation curve reported per cell: cap_oos as a function of number of prompts included
    (random order, averaged over permutations) - flat tail = saturated.

CPU-only, existing npz. Usage:
  python -m methods.tacit_channels.channels.eval.prompt_subspace_cap \
      --root notebooks/data/two_faces_20260702/family_scores_qwen25 \
      --target-job qwen25_72b_name_target --exec-job qwen25_7b_executor \
      --domain humor --out outputs/tacit_channels/subspace_cap_q7b_humor.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from methods.tacit_channels.channels.common import (
    _rankdata, load_grid, stable_split, write_jsonl,
)


def _center_rank(v: np.ndarray) -> np.ndarray:
    r = _rankdata(np.asarray(v, float))
    r = r - r.mean()
    n = np.linalg.norm(r)
    return r / n if n > 0 else r


def cap_for_cell(rows: np.ndarray, target: np.ndarray, half1: np.ndarray,
                 half2: np.ndarray, n_perms: int = 8, seed: int = 0) -> dict:
    """rows: (P, n_items) raw executor scores under P articulations; target: (n_items,)."""
    P = len(rows)
    z = _center_rank(target)
    R = np.vstack([_center_rank(r) for r in rows])

    def cap_split(row_subset: np.ndarray) -> float:
        # fit blend on half-1 (least squares of z on subspace), evaluate corr on half-2
        A1, z1 = row_subset[:, half1].T, z[half1]
        coef, *_ = np.linalg.lstsq(A1, z1, rcond=None)
        v2 = row_subset[:, half2].T @ coef
        z2 = z[half2]
        if v2.std() == 0 or z2.std() == 0:
            return float("nan")
        return float(np.corrcoef(v2, z2)[0, 1])

    cap_oos = cap_split(R)
    # in-sample (upper reference only)
    coef_all, *_ = np.linalg.lstsq(R.T, z, rcond=None)
    fit = R.T @ coef_all
    cap_ins = float(np.corrcoef(fit, z)[0, 1]) if fit.std() > 0 else float("nan")
    # best single articulation, out-of-sample framing not needed (no fitting)
    best_single = max(float(np.corrcoef(_center_rank(r), z)[0, 1]) for r in rows)
    # saturation curve: cap_oos vs #prompts, averaged over random inclusion orders
    rng = random.Random(seed)
    grid = sorted({max(2, P // 8), max(3, P // 4), max(4, P // 2),
                   max(5, 3 * P // 4), P})
    curve = []
    for k in grid:
        vals = []
        for _ in range(n_perms):
            idx = rng.sample(range(P), k)
            vals.append(cap_split(R[idx]))
        vals = [v for v in vals if not np.isnan(v)]
        curve.append((k, round(float(np.mean(vals)), 4) if vals else None))
    # effective rank (90% energy) of the rank-behavior family
    sv = np.linalg.svd(R, compute_uv=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    eff_rank = int(np.searchsorted(energy, 0.90) + 1)
    saturated = (len(curve) >= 2 and curve[-1][1] is not None and curve[-2][1] is not None
                 and abs(curve[-1][1] - curve[-2][1]) < 0.02)
    return {"n_prompts": P, "eff_rank_90": eff_rank,
            "cap_oos": round(cap_oos, 4) if not np.isnan(cap_oos) else None,
            "cap_insample": round(cap_ins, 4) if not np.isnan(cap_ins) else None,
            "best_single_rho": round(best_single, 4),
            "saturation_curve": curve, "saturated": bool(saturated)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target-job", required=True)
    ap.add_argument("--exec-job", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--item-salt", default="exp_gtk1")
    ap.add_argument("--cells", default=None, help="comma list; default all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tgt, _ = load_grid(args.root, args.target_job, args.domain)
    exe, emeta = load_grid(args.root, args.exec_job, args.domain)
    cells = sorted({c for (c, a, f) in tgt})
    if args.cells:
        cells = [c for c in cells if c in set(args.cells.split(","))]

    out_rows = []
    for cell in cells:
        t_forms = [v for (c, a, f), v in tgt.items() if c == cell and a == "name"]
        rows = [v for (c, a, f), v in exe.items() if c == cell]
        if not t_forms or len(rows) < 8:
            continue
        target = np.mean(t_forms, axis=0)
        n = len(target)
        # deterministic item halves via the same stable hash used by EXP-GTK-1
        halves = [stable_split(f"{cell}::item{i}", 0.5, salt=args.item_salt)
                  for i in range(n)]
        half1 = np.array([i for i, h in enumerate(halves) if h == "train"])
        half2 = np.array([i for i, h in enumerate(halves) if h != "train"])
        res = cap_for_cell(np.vstack(rows), target, half1, half2)
        out_rows.append({"cell_id": cell, "domain": args.domain,
                         "exec_job": args.exec_job, **res})
    write_jsonl(args.out, out_rows)

    caps = [r["cap_oos"] for r in out_rows if r["cap_oos"] is not None]
    sat = sum(1 for r in out_rows if r["saturated"])
    below = sum(1 for c in caps if c < 0.70)
    print(json.dumps({
        "n_cells": len(out_rows), "saturated_cells": sat,
        "median_cap_oos": round(float(np.median(caps)), 4) if caps else None,
        "cells_cap_below_.70": below,
        "median_eff_rank": float(np.median([r["eff_rank_90"] for r in out_rows]))
        if out_rows else None}, indent=2))


if __name__ == "__main__":
    main()

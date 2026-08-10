"""W1a v0 readouts from the variant grids — exclusion leak, NOT-gap, composition, holistic.

v0 references (declared in the W1a addendum): negation reference = rank-reversal of the
target's name vector; composition reference = elementwise min of rank-z-scored member target
vectors (AND semantics). The 72B target pass upgrades both to the target's OWN vectors (v1).

Statistics per (config, cell), adverse (min) over forms unless stated:
  tf_rho        — adverse Spearman(tf, target name mean-forms)
  leak_self     — median over forms of Spearman(exclusion_f, tf_f): +1 = policy leaks
                  through the suppress instruction unchanged; -1 = full instructed inversion
  not_gap       — tf_rho - adverse Spearman(negated, REVERSED target) : 0 = perfect NOT use
  comp_rho      — adverse Spearman(composed, min-z-blend of member targets), per pair
  holistic OOS R^2 — ridge (alpha=1, standardized) of the holistic vector on the config's
                  own 90 name tf vectors; fit item-half-1, R^2 on item-half-2 (salt exp_gtk1)

Adapter-involved rows are restricted to item-half-2 throughout. CPU-only, local grids.

  python -m methods.tacit_channels.battery.tally_w1a \
      --out outputs/tacit_channels/battery_w1/tally_w1a_v0.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import (
    _rankdata, load_grid, spearman, stable_split,
)

BASE = "notebooks/data/two_faces_20260702"
SCORES_ROOT = f"{BASE}/family_scores_qwen25"
TARGET_JOB = "qwen25_72b_name_target"
PACKET_ROOT = f"{BASE}/tacit_breadth_item_partitions_v2"
W1_ROOT = "outputs/tacit_channels/battery_w1"
CONFIGS = {  # config -> (grid dir, half2-restricted?)
    "qwen25_7b_base": ("qwen25_7b_base", False),
    "qwen25_7b_real": ("qwen25_7b_real_n8192c", True),
    "qwen25_14b_base": ("qwen25_14b_base", False),
}
SALT = "exp_gtk1"


def half2_mask() -> np.ndarray:
    payload = _apparatus.load_domain_items(
        PACKET_ROOT, "humor", partitions=["tacit_breadth_search"])
    hashes = payload.get("hashes")
    return np.array([stable_split(h, 0.5, salt=SALT) != "train" for h in hashes])


def zrank(v: np.ndarray) -> np.ndarray:
    r = _rankdata(np.asarray(v, float))
    return (r - r.mean()) / (r.std() or 1.0)


def by_variant(grid: dict):
    """{(cell, variant, form): vec} from a W1a grid keyed (cell, arm_id, form)."""
    out = {}
    for (c, a, f), v in grid.items():
        variant = {"name": "tf", "name_exclusion": "exclusion", "name_negated": "negated",
                   "name_composed": "composed", "holistic": "holistic"}.get(a)
        if variant:
            out[(c, variant, f)] = v
    return out


def adverse(vecs: list, ref: np.ndarray, mask: np.ndarray) -> float | None:
    vals = [spearman(np.asarray(v)[mask], ref[mask]) for v in vecs]
    vals = [x for x in vals if not np.isnan(x)]
    return float(min(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default="outputs/tacit_channels/exp_gtk1/design_manifest.json")
    args = ap.parse_args()

    design = json.load(open(args.design))
    a_cells, b2 = set(design["A"]), set(design.get("B2_success", []))
    tgt, _ = load_grid(SCORES_ROOT, TARGET_JOB, "humor")
    t_ref = {}
    for (c, a, f), v in tgt.items():
        if a == "name":
            t_ref.setdefault(c, []).append(v)
    t_ref = {c: np.mean(vs, axis=0) for c, vs in t_ref.items()}
    h2 = half2_mask()
    all_mask = np.ones(len(h2), dtype=bool)

    rows, holistic = [], {}
    for cfg, (d, restrict) in CONFIGS.items():
        grid, _ = load_grid(W1_ROOT, d, "humor")
        g = by_variant(grid)
        mask = h2 if restrict else all_mask
        cells = sorted({c for (c, _v, _f) in g if _v == "tf" and "&&" not in c})

        forms = lambda c, var: [g[(c, var, f)] for (cc, vv, f) in g
                                if cc == c and vv == var]
        for c in cells:
            if c not in t_ref:
                continue
            tf_vecs = forms(c, "tf")
            row = {"config": cfg, "cell_id": c,
                   "set": "A" if c in a_cells else ("B2" if c in b2 else "other"),
                   "tf_rho": adverse(tf_vecs, t_ref[c], mask)}
            # exclusion leak vs own tf, form-matched
            leaks = []
            for (cc, vv, f), v in g.items():
                if cc == c and vv == "exclusion" and (c, "tf", f) in g:
                    s = spearman(np.asarray(v)[mask], np.asarray(g[(c, "tf", f)])[mask])
                    if not np.isnan(s):
                        leaks.append(s)
            row["leak_self"] = float(np.median(leaks)) if leaks else None
            neg = forms(c, "negated")
            neg_rho = adverse(neg, -t_ref[c], mask)
            row["not_gap"] = (row["tf_rho"] - neg_rho
                              if None not in (row["tf_rho"], neg_rho) else None)
            rows.append(row)
        # composed pairs
        for (c, vv, f), v in sorted(g.items()):
            if vv != "composed" or (c, "composed", f) != (c, vv, f):
                continue
        pair_cells = sorted({c for (c, _v, _f) in g if _v == "composed"})
        for pc in pair_cells:
            ca, cb = pc.split("&&")
            if ca not in t_ref or cb not in t_ref:
                continue
            blend = np.minimum(zrank(t_ref[ca]), zrank(t_ref[cb]))
            comp_rho = adverse(forms(pc, "composed"), blend, mask)
            member = None
            ra = next((r["tf_rho"] for r in rows if r["config"] == cfg
                       and r["cell_id"] == ca), None)
            rb = next((r["tf_rho"] for r in rows if r["config"] == cfg
                       and r["cell_id"] == cb), None)
            if ra is not None and rb is not None:
                member = float(np.mean([ra, rb]))
            rows.append({"config": cfg, "cell_id": pc,
                         "set": "AxA" if ca in a_cells and cb in a_cells else "nonA",
                         "comp_rho": comp_rho, "member_mean_tf_rho": member})
        # holistic OOS R^2 (fit half-1 -> R^2 half-2), own 90 name vectors as span
        hol = forms("HOLISTIC::humor", "holistic")
        if hol:
            y = np.asarray(hol[0], float)
            X = np.column_stack([np.asarray(g[(c, "tf", "canonical")], float)
                                 for c in cells if (c, "tf", "canonical") in g])
            Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
            fit, ev = ~h2, h2
            A = Xz[fit]
            w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y[fit])
            resid = y[ev] - Xz[ev] @ w
            r2 = 1 - resid.var() / y[ev].var()
            holistic[cfg] = {"oos_r2": float(r2),
                             "unnamed_share": float(1 - r2), "n_predictors": X.shape[1]}

    def med(vals):
        vals = [v for v in vals if v is not None]
        return (float(np.median(vals)), len(vals)) if vals else (None, 0)

    summary = {}
    for cfg in CONFIGS:
        cr = [r for r in rows if r["config"] == cfg]
        s = {}
        for name, pred in (("A", lambda r: r.get("set") == "A"),
                           ("B2", lambda r: r.get("set") == "B2"),
                           ("other", lambda r: r.get("set") == "other")):
            sub = [r for r in cr if pred(r) and "leak_self" in r]
            s[f"leak_{name}"], s[f"n_{name}"] = med([r["leak_self"] for r in sub])
            s[f"notgap_{name}"], _ = med([r["not_gap"] for r in sub])
            s[f"tf_{name}"], _ = med([r["tf_rho"] for r in sub])
        for name in ("AxA", "nonA"):
            sub = [r for r in cr if r.get("set") == name and "comp_rho" in r]
            s[f"comp_{name}"], s[f"ncomp_{name}"] = med([r["comp_rho"] for r in sub])
            s[f"compmember_{name}"], _ = med([r["member_mean_tf_rho"] for r in sub])
        s["holistic"] = holistic.get(cfg)
        summary[cfg] = s

    out = {"schema": "battery_w1a_tally/v0",
           "references": "negation=reversed target; composition=min z-blend (v0)",
           "summary": summary, "per_cell": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for cfg, s in summary.items():
        print(f"\n{cfg}:")
        print(f"  tf median A/B2/other: {s['tf_A']} / {s['tf_B2']} / {s['tf_other']}")
        print(f"  leak_self A/B2/other: {s['leak_A']} / {s['leak_B2']} / {s['leak_other']}")
        print(f"  not_gap  A/B2/other: {s['notgap_A']} / {s['notgap_B2']} / {s['notgap_other']}")
        print(f"  comp AxA {s['comp_AxA']} (n={s['ncomp_AxA']}, member {s['compmember_AxA']}) "
              f"| nonA {s['comp_nonA']} (n={s['ncomp_nonA']}, member {s['compmember_nonA']})")
        print(f"  holistic: {s['holistic']}")
    print(f"\ntally -> {args.out}")


if __name__ == "__main__":
    main()

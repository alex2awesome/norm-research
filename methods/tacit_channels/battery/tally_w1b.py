"""W1b readouts — CoT-delta / reason-first interference + verbalized-confidence STAT-1
and the FROZEN P-B5 proxy-validity test.

Per (config, cell), canonical form only (W1b addendum):
  tf_rho      — Spearman(tf canonical, target name mean-forms)
  rf_rho      — Spearman(reason_first, same target)
  cot_delta   — rf_rho - tf_rho  (P-TOK-1: token-dependence; P-INTF-1: interference when <0)
  conf_acc_verbal / conf_acc_logodds — Spearman over items of (confidence, per-item rank
    agreement with target); confidence = verbalized 0-100 (W1b) vs |p_yes-.5| (v0 proxy)
  guess_agreement — mean item agreement on the bottom-confidence quartile (knowledge
    without metacognitive access if >> chance while conf_acc ~ 0)

P-B5 (FROZEN, prereg body): Spearman across cells of conf_acc_verbal vs conf_acc_logodds
>= +.40, else the v0 proxy is retired and W0 STAT-1 rows are flagged. Evaluated per config.
P-W1b-1 (addendum): cot_delta median < 0 for 7B on the articulation-failure strata,
neutral-or-positive for 14B.

Adapter rows restricted to item-half-2. CPU-only.
  python -m methods.tacit_channels.battery.tally_w1b \
      --out outputs/tacit_channels/battery_w1/tally_w1b_v0.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from methods.tacit_channels import _apparatus
from methods.tacit_channels.battery.artifacts import item_agreement
from methods.tacit_channels.channels.common import load_grid, spearman, stable_split

BASE = "notebooks/data/two_faces_20260702"
SCORES_ROOT = f"{BASE}/family_scores_qwen25"
TARGET_JOB = "qwen25_72b_name_target"
PACKET_ROOT = f"{BASE}/tacit_breadth_item_partitions_v2"
W1_ROOT = "outputs/tacit_channels/battery_w1"
CONFIGS = {
    "qwen25_7b_base": ("qwen25_7b_base", False),
    "qwen25_7b_real": ("qwen25_7b_real_n8192c", True),
    "qwen25_14b_base": ("qwen25_14b_base", False),
}
SALT = "exp_gtk1"


def half2_mask() -> np.ndarray:
    payload = _apparatus.load_domain_items(
        PACKET_ROOT, "humor", partitions=["tacit_breadth_search"])
    return np.array([stable_split(h, 0.5, salt=SALT) != "train"
                     for h in payload["hashes"]])


def load_conf(dirname: str) -> dict:
    """{(cell_id): confidence vector} from the confidence npz (canonical rows)."""
    paths = glob.glob(f"{W1_ROOT}/{dirname}/confidence_humor_w1b_*_rep0.npz")
    out = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        scores = np.asarray(d["scores"])
        for i, s in enumerate(d["meta"]):
            m = json.loads(s)
            if m.get("form") == "canonical":
                out[m["cell_id"]] = scores[i]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default="outputs/tacit_channels/exp_gtk1/design_manifest.json")
    args = ap.parse_args()

    design = json.load(open(args.design))
    a_cells, b1, b2 = (set(design["A"]), set(design["B1"]),
                       set(design.get("B2_success", [])))
    strata = lambda c: ("failure" if (c in a_cells or c in b1) else
                        "success" if c in b2 else "other")
    tgt, _ = load_grid(SCORES_ROOT, TARGET_JOB, "humor")
    t_ref = {}
    for (c, a, f), v in tgt.items():
        if a == "name":
            t_ref.setdefault(c, []).append(v)
    t_ref = {c: np.mean(vs, axis=0) for c, vs in t_ref.items()}
    h2 = half2_mask()

    rows = []
    for cfg, (d, restrict) in CONFIGS.items():
        w1a, _m = load_grid(W1_ROOT, d, "humor")   # matches grid_humor_w1variants_*
        conf = load_conf(d)
        mask = h2 if restrict else np.ones(len(h2), dtype=bool)
        # reason_first grid rows live in grid_humor_w1b_reason_* (own npz, same key space)
        rf = {}
        for p in glob.glob(f"{W1_ROOT}/{d}/grid_humor_w1b_reason_*_rep0.npz"):
            z = np.load(p, allow_pickle=True)
            sc = np.asarray(z["scores"])
            for i, s in enumerate(z["meta"]):
                m = json.loads(s)
                if m.get("form") == "canonical":
                    rf[m["cell_id"]] = sc[i]
        for c in sorted(rf):
            if c not in t_ref or (c, "name", "canonical") not in w1a:
                continue
            t = t_ref[c]
            tf_vec = np.asarray(w1a[(c, "name", "canonical")], float)
            rf_vec = np.asarray(rf[c], float)
            tf_rho = spearman(tf_vec[mask], t[mask])
            rf_rho = spearman(rf_vec[mask], t[mask])
            row = {"config": cfg, "cell_id": c, "set": strata(c),
                   "tf_rho": tf_rho, "rf_rho": rf_rho,
                   "cot_delta": rf_rho - tf_rho}
            agree = item_agreement(tf_vec[mask], t[mask])
            lo = np.abs(tf_vec[mask] - 0.5)
            row["conf_acc_logodds"] = spearman(lo, agree)
            if c in conf:
                cv = np.asarray(conf[c], float)[mask]
                ok = np.isfinite(cv)
                if ok.sum() >= 50:
                    row["conf_acc_verbal"] = spearman(cv[ok], agree[ok])
                    q = np.nanquantile(cv[ok], 0.25)
                    row["guess_agreement"] = float(agree[ok][cv[ok] <= q].mean())
                    row["conf_mean"] = float(cv[ok].mean())
            rows.append(row)

    def med(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float)
                                                          and np.isnan(v))]
        return (float(np.median(vals)), len(vals)) if vals else (None, 0)

    summary = {}
    for cfg in CONFIGS:
        cr = [r for r in rows if r["config"] == cfg]
        s = {}
        for st in ("failure", "success", "other"):
            sub = [r for r in cr if r["set"] == st]
            s[f"cot_delta_{st}"], s[f"n_{st}"] = med([r["cot_delta"] for r in sub])
        s["conf_acc_verbal_med"], _ = med([r.get("conf_acc_verbal") for r in cr])
        s["conf_acc_logodds_med"], _ = med([r.get("conf_acc_logodds") for r in cr])
        s["guess_agreement_med"], _ = med([r.get("guess_agreement") for r in cr])
        s["conf_mean_med"], _ = med([r.get("conf_mean") for r in cr])
        both = [(r["conf_acc_verbal"], r["conf_acc_logodds"]) for r in cr
                if r.get("conf_acc_verbal") is not None
                and r.get("conf_acc_logodds") is not None
                and not np.isnan(r["conf_acc_verbal"])
                and not np.isnan(r["conf_acc_logodds"])]
        if len(both) >= 10:
            v, l = zip(*both)
            s["p_b5_proxy_corr"] = spearman(np.array(v), np.array(l))
            s["p_b5_n"] = len(both)
            s["p_b5_pass"] = bool(s["p_b5_proxy_corr"] >= 0.40)
        summary[cfg] = s

    out = {"schema": "battery_w1b_tally/v0", "summary": summary, "per_cell": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for cfg, s in summary.items():
        print(f"\n{cfg}:")
        print(f"  cot_delta med fail/succ/other: {s['cot_delta_failure']} "
              f"(n={s['n_failure']}) / {s['cot_delta_success']} / {s['cot_delta_other']}")
        print(f"  conf_acc verbal/logodds med: {s['conf_acc_verbal_med']} / "
              f"{s['conf_acc_logodds_med']}; guess_agree {s['guess_agreement_med']}; "
              f"conf_mean {s['conf_mean_med']}")
        if "p_b5_proxy_corr" in s:
            print(f"  P-B5 proxy corr {s['p_b5_proxy_corr']:+.3f} (n={s['p_b5_n']}) -> "
                  f"{'PASS' if s['p_b5_pass'] else 'FAIL'} (bar +.40)")
    print(f"\ntally -> {args.out}")


if __name__ == "__main__":
    main()

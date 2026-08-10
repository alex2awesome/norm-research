#!/usr/bin/env python3
"""GEPA PHRASING PASS + COLLAPSE GATE + SIGN-CONTRADICTION RE-AUDIT for the mined
Track-A criteria of this cell.

The FREEZE DECLARATION requires three things of a mined criterion before its number is
quoted as final, and this script is where all three are applied together, because they
interact (a collapsed criterion is also usually a sign-contradicting one, and rephrasing
is the remedy for exactly one of the three).

1. COLLAPSE GATE (prereg step 5, "guided-JSON collapse check on every criterion's score
   distribution before use").  `score_gemma_maps.py` FLAGS collapse
   (`modal_frac > .98`) in the score report, but `closure_core.clean_fit` only drops a
   column when fewer than 5 rows sit off the mode -- at n = 16,000 a criterion at
   modal .988 leaves 190 off-modal rows and survives the screen.  The flag was therefore
   being recorded and not acted on.  This script applies it: a flagged criterion is
   EXCLUDED from the bank and the headline gain is recomputed without it.

2. SIGN-CONTRADICTION RE-AUDIT (freeze: "sign-contradicting criteria -> re-audit
   trigger").  Two-sided noise-scaled band, Hanley-McNeil SE at AUC = .5 on the FIT+MINE
   class counts; the trigger fires only when alone-AUC sits more than 2 SE BELOW chance.
   Criteria inside the band are recorded as `sign_null_band` and kept.

3. GEPA FIDELITY TARGETING (label-blind).  fidelity = .5*(1-modal_share)
   + .3*(1-na_rate) + .2*min(spread/scale_half, 1), scale_half = 5.0 for this 0-10 mined
   scale.  Criteria with modal_share > .75 or na_rate > .20 are TARGETED for rephrasing;
   the rest have PASSED the phrasing pass and their numbers may be quoted.
   Targeting is computed from the score distribution only -- it never sees y.

Stage 1 (`targets`) is what gates quoting.  Stages 2-4 (author variants, probe-score
them, accept a variant only if it beats the incumbent's fidelity on the SAME probe rows
by >= .02) run once at terminal on the union of surviving mined criteria, following
../gepa_{targets,select,finalize}_peer.py.

CPU only.  Usage: python3 gepa_phrasing.py targets --rounds 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
MODAL_TRIGGER, NA_TRIGGER, SCALE_HALF = 0.75, 0.20, 5.0
COLLAPSE_MODAL = 0.98


def fidelity(col):
    fin = np.isfinite(col)
    if fin.sum() < 20:
        return 0.0, {"modal_share": 1.0, "na_rate": 1.0, "spread": 0.0}
    _, cnts = np.unique(col[fin], return_counts=True)
    modal = float(cnts.max() / fin.sum())
    na = float((~fin).mean())
    spread = float(np.std(col[fin]) / SCALE_HALF)
    return (0.5 * (1 - modal) + 0.3 * (1 - na) + 0.2 * min(spread, 1.0),
            {"modal_share": modal, "na_rate": na, "spread": spread})


def hm_se(n1, n0):
    """Hanley-McNeil SE of an AUC at the null value .5."""
    return float(np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0)))


def cmd_targets(a):
    d = C.load()
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm = split == "fit_mine"
    y = d["y"]
    n1, n0 = int(y[fitm].sum()), int((1 - y[fitm]).sum())
    se = hm_se(n1, n0)
    band = 2 * se

    out = {"cell": a.cell, "rounds": a.rounds,
           "sign_band": {"n_pos_fitmine": n1, "n_neg_fitmine": n0,
                         "hanley_mcneil_SE_at_null": se, "two_sided_2SE_band": band,
                         "rule": "trigger fires only when alone-AUC on FIT+MINE is more "
                                 "than 2 SE BELOW .5; inside the band = sign_null_band, kept"},
           "gepa_rule": {"modal_trigger": MODAL_TRIGGER, "na_trigger": NA_TRIGGER,
                         "scale_half": SCALE_HALF,
                         "fidelity": ".5*(1-modal)+.3*(1-na)+.2*min(spread,1)"},
           "collapse_rule": {"modal_frac_gt": COLLAPSE_MODAL,
                             "note": "score_gemma_maps flags it; clean_fit does NOT drop it "
                                     "at n=16,000; this pass EXCLUDES it"},
           "criteria": []}

    for r in a.rounds.split(","):
        r = r.strip()
        z = np.load(HERE / f"{a.cell}_r{r}_scores.npz", allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        routing = json.loads((HERE / f"{a.cell}_r{r}_routing_final.json").read_text())
        nm = {x["blind_id"]: x["name"] for x in routing["final"]}
        A_ids = [x["blind_id"] for x in routing["final"] if x["final_route"] == "A"]
        for cid in A_ids:
            col = z["X"][:, cids.index(cid)]
            f, info = fidelity(col)
            fin = np.isfinite(col[fitm])
            auc = float(roc_auc_score(y[fitm][fin], col[fitm][fin])) if fin.sum() > 50 else None
            rec = {"round": r, "blind_id": cid, "name": nm[cid],
                   "fidelity": f, **info, "alone_AUC_fitmine": auc,
                   "COLLAPSED": bool(info["modal_share"] > COLLAPSE_MODAL),
                   "gepa_targeted": bool(info["modal_share"] > MODAL_TRIGGER
                                         or info["na_rate"] > NA_TRIGGER)}
            if auc is None:
                rec["sign_verdict"] = "no_auc"
            elif auc < 0.5 - band:
                rec["sign_verdict"] = "SIGN_CONTRADICTING_re_audit_trigger"
            elif auc < 0.5:
                rec["sign_verdict"] = "sign_null_band"
            else:
                rec["sign_verdict"] = "ok"
            rec["QUOTABLE"] = bool(not rec["COLLAPSED"] and not rec["gepa_targeted"]
                                   and rec["sign_verdict"] in ("ok", "sign_null_band"))
            out["criteria"].append(rec)

    out["n_criteria"] = len(out["criteria"])
    out["n_collapsed"] = sum(c["COLLAPSED"] for c in out["criteria"])
    out["n_gepa_targeted"] = sum(c["gepa_targeted"] for c in out["criteria"])
    out["n_sign_triggered"] = sum(
        c["sign_verdict"] == "SIGN_CONTRADICTING_re_audit_trigger" for c in out["criteria"])
    out["n_quotable"] = sum(c["QUOTABLE"] for c in out["criteria"])
    out["excluded_ids"] = [f"r{c['round']}:{c['blind_id']}" for c in out["criteria"]
                           if c["COLLAPSED"]]
    (HERE / f"{a.cell}_gepa_targets.json").write_text(json.dumps(out, indent=1))
    print(f"2SE band below chance: {0.5 - band:.4f}  (SE {se:.5f})")
    print(f"{'id':5s} {'fid':>5s} {'modal':>6s} {'na':>6s} {'AUCfm':>6s} {'quot':>5s} "
          f"{'sign':22s} name")
    for c in sorted(out["criteria"], key=lambda x: -x["fidelity"]):
        print(f"{c['blind_id']:5s} {c['fidelity']:5.3f} {c['modal_share']:6.3f} "
              f"{c['na_rate']:6.3f} {(c['alone_AUC_fitmine'] or 0):6.3f} "
              f"{str(c['QUOTABLE']):>5s} {c['sign_verdict']:22s} {c['name'][:44]}")
    print(f"\ncollapsed {out['n_collapsed']} | gepa-targeted {out['n_gepa_targeted']} | "
          f"sign-triggered {out['n_sign_triggered']} | quotable {out['n_quotable']}"
          f"/{out['n_criteria']}")


def cmd_regate(a):
    """Recompute the Track-A closure gain with COLLAPSED criteria excluded."""
    import stage1_slice as S1
    d = C.load()
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y, g = d["y"], d["groups"]
    tgt = json.loads((HERE / f"{a.cell}_gepa_targets.json").read_text())
    collapsed = {(c["round"], c["blind_id"]) for c in tgt["criteria"] if c["COLLAPSED"]}

    r = str(a.round)
    z = np.load(HERE / f"{a.cell}_r{r}_scores.npz", allow_pickle=True)
    cids = [str(s) for s in z["crit_ids"]]
    routing = json.loads((HERE / f"{a.cell}_r{r}_routing_final.json").read_text())
    A_all = [x["blind_id"] for x in routing["final"] if x["final_route"] == "A"]
    A_keep = [c for c in A_all if (r, c) not in collapsed]

    prev_blocks, prev_tags = S1.current_blocks(d, r)
    XA_all = z["X"][:, [cids.index(i) for i in A_all]]
    XA_keep = z["X"][:, [cids.index(i) for i in A_keep]]

    res = {"cell": a.cell, "round": r, "bank_entering": prev_tags,
           "n_A_all": len(A_all), "n_A_after_collapse_gate": len(A_keep),
           "excluded": [c for c in A_all if c not in A_keep]}
    rp = L.fit_block(prev_blocks, fitm, monm, y, g)
    ra = L.fit_block(prev_blocks + [XA_all], fitm, monm, y, g)
    rk = L.fit_block(prev_blocks + [XA_keep], fitm, monm, y, g)
    ymon = y[monm]

    def full(rr):
        v = np.full(len(y), np.nan); v[fitm] = rr["oof_nl_fitmine"]; v[monm] = rr["nl_mon"]
        return v

    for nm, rr in (("prev", rp), ("with_collapsed", ra), ("collapse_gated", rk)):
        res[nm] = {"n_features": rr["n_features"],
                   "VA_nl_MONITOR": L.auc(ymon, rr["nl_mon"]),
                   "VA_nl_HONEST": L.auc(y[held], full(rr)[held])}
    res["gain_MONITOR_with_collapsed"] = (res["with_collapsed"]["VA_nl_MONITOR"]
                                          - res["prev"]["VA_nl_MONITOR"])
    res["gain_MONITOR_collapse_gated"] = (res["collapse_gated"]["VA_nl_MONITOR"]
                                          - res["prev"]["VA_nl_MONITOR"])
    res["gain_HONEST_collapse_gated"] = (res["collapse_gated"]["VA_nl_HONEST"]
                                         - res["prev"]["VA_nl_HONEST"])
    res["gain_ci_MONITOR_collapse_gated"] = L.group_boot_ci(
        ymon, rk["nl_mon"], rp["nl_mon"], np.array([str(x) for x in g[monm]]))
    res["HEADLINE"] = ("the collapse-gated gain is the quotable one; the with-collapsed "
                       "figure is kept beside it so the correction is visible")
    (HERE / f"{a.cell}_r{r}_collapse_gated.json").write_text(json.dumps(res, indent=1,
                                                                       default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--cell", default="jokes_community")
    t.add_argument("--rounds", default="1")
    rg = sub.add_parser("regate")
    rg.add_argument("--cell", default="jokes_community")
    rg.add_argument("--round", default="1")
    a = ap.parse_args()
    {"targets": cmd_targets, "regate": cmd_regate}[a.cmd](a)

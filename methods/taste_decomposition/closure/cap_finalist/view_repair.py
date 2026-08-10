#!/usr/bin/env python3
"""ROUND 3 = VIEW-REPAIR PASS (TIER R).  Registered in
notes/2026-08-09__closure_cap_finalist.md §1.4 before it ran.

WHY.  The A bank scored every caption as `CARTOON: <desc>\\n\\nCAPTION: "<text>"`; the
round-1/round-2 mined criteria were scored on the caption ALONE.  A New Yorker caption is
close to ungradeable without the drawing it captions, so 50 criteria were measured on a
strictly weaker view than the bank they were being added to.  This pass re-measures those
same 50 criteria on the matched view.  Nothing is proposed, nothing is re-routed: it is a
re-measurement, so by the registered 2026-08-08 rule it cannot advance the stopping clock,
and by the two-tier rule it contributes no Good-Turing mass.

  build   -> cap_finalist_r3_species.json  (tier R, 50 criteria, ids R1_*/R2_*)
  split   -> rewrites cap_finalist_r{1,2}_scores.npz from cap_finalist_r3_scores.npz,
             preserving the original crit_ids.  The pre-repair matrices are already
             saved as cap_finalist_r{1,2}_scores.MISMATCHEDVIEW.npz and are never deleted.

CPU only.  Usage:
  python view_repair.py build
  # ... score cap_finalist_r3 on GPU 5 ...
  python view_repair.py split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROUNDS = ("1", "2")


def cmd_build(_):
    selected, prov = [], []
    for r in ROUNDS:
        sp = json.loads((HERE / f"cap_finalist_r{r}_species.json").read_text())
        for c in sp["selected"]:
            selected.append({
                "blind_id": f"R{r}_{c['blind_id']}",
                "name": c["name"],
                "instruction": c["instruction"],
                "track": c["track"],
            })
            prov.append({"repair_id": f"R{r}_{c['blind_id']}", "origin_round": r,
                         "origin_blind_id": c["blind_id"], "name": c["name"],
                         "track": c["track"]})
    out = {
        "tag": "cap_finalist_r3", "cell": "cap_finalist", "round": "3",
        "tier": "R",
        "kind": "VIEW REPAIR -- re-measurement of existing round-1/round-2 criteria on the "
                "cartoon+caption item view that the A bank itself was scored on. No sealed "
                "fleet ran; no new concepts were proposed.",
        "two_tier_rule": "TIER R contributes NO Good-Turing / missing-mass quantity and "
                         "cannot advance the stopping clock (registered 2026-08-08 rule: "
                         "non-PROPOSING rounds are exempt).",
        "composition": {"n_criteria": len(selected),
                        "from_round_1": sum(1 for p in prov if p["origin_round"] == "1"),
                        "from_round_2": sum(1 for p in prov if p["origin_round"] == "2"),
                        "track_A": sum(1 for p in prov if p["track"] == "A"),
                        "track_B": sum(1 for p in prov if p["track"] == "B")},
        "provenance": prov,
        "selected": selected,
    }
    (HERE / "cap_finalist_r3_species.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["composition"], indent=1))
    print(f"wrote cap_finalist_r3_species.json ({len(selected)} criteria, tier R)")


def cmd_split(_):
    z = np.load(HERE / "cap_finalist_r3_scores.npz", allow_pickle=True)
    cids = [str(s) for s in z["crit_ids"]]
    X = z["X"]
    prov = {p["repair_id"]: p for p in
            json.loads((HERE / "cap_finalist_r3_species.json").read_text())["provenance"]}

    report = {"source": "cap_finalist_r3_scores.npz (matched CARTOON+CAPTION view)",
              "rounds": {}}
    for r in ROUNDS:
        old_p = HERE / f"cap_finalist_r{r}_scores.MISMATCHEDVIEW.npz"
        old = np.load(old_p, allow_pickle=True)
        old_ids = [str(s) for s in old["crit_ids"]]
        take = [cids.index(f"R{r}_{c}") for c in old_ids]
        Xnew = X[:, take]
        assert (z["i"] == old["i"]).all(), f"r{r}: row order moved between passes"
        np.savez_compressed(
            HERE / f"cap_finalist_r{r}_scores.npz", X=Xnew,
            crit_ids=np.array(old_ids, dtype=object),
            crit_names=old["crit_names"], i=z["i"], row_id=z["row_id"],
            Xanchor=z["Xanchor"], anchor_tags=z["anchor_tags"], scale="0-10",
            item_view="CARTOON+CAPTION (matched to the A bank) -- VIEW REPAIR, round 3",
            superseded_file=f"cap_finalist_r{r}_scores.MISMATCHEDVIEW.npz")
        # how far did each criterion actually move?
        moved = []
        for k, cid in enumerate(old_ids):
            a, b = old["X"][:, k], Xnew[:, k]
            m = np.isfinite(a) & np.isfinite(b)
            rho = (float(np.corrcoef(np.argsort(np.argsort(a[m])),
                                     np.argsort(np.argsort(b[m])))[0, 1])
                   if m.sum() > 100 else None)
            moved.append({"crit_id": cid, "name": str(old["crit_names"][k]),
                          "track": prov[f"R{r}_{cid}"]["track"],
                          "mean_old": float(np.nanmean(a)), "mean_new": float(np.nanmean(b)),
                          "na_old": float(np.isnan(a).mean()), "na_new": float(np.isnan(b).mean()),
                          "spearman_old_vs_new": rho})
        moved.sort(key=lambda d: (d["spearman_old_vs_new"] if d["spearman_old_vs_new"]
                                  is not None else 9))
        report["rounds"][r] = {"n_criteria": len(old_ids), "per_criterion": moved,
                               "median_spearman_old_vs_new": float(np.nanmedian(
                                   [d["spearman_old_vs_new"] for d in moved
                                    if d["spearman_old_vs_new"] is not None]))}
        print(f"r{r}: {len(old_ids)} criteria rewritten; median rank-correlation "
              f"old-view vs new-view {report['rounds'][r]['median_spearman_old_vs_new']:.3f}")
    (HERE / "cap_finalist_view_repair_report.json").write_text(json.dumps(report, indent=1))
    print("wrote cap_finalist_view_repair_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "split"):
        sub.add_parser(n)
    a = ap.parse_args()
    {"build": cmd_build, "split": cmd_split}[a.cmd](a)

#!/usr/bin/env python3
"""M3 step 5 -- AUC recovery from rediscovered concepts (zero new judging).

For each replicate: take the concepts the sealed fleet rediscovered, re-add ONLY
their ORIGINAL score columns to the depleted bank, refit VA_nl under the frozen
Layer-1 spec, and report what fraction of the depletion drop comes back.

  recovered_fraction = (AUC_recovered - AUC_depleted) / (AUC_full - AUC_depleted)

This is the measurement the design note calls the positive control: it converts
"the fleet named the concept" into "the fleet would have recovered the signal".
No new criterion is scored -- rediscovered concepts are measured with the score
columns the bank already carries.

CPU only.  Usage: python m3_recover.py [--tau 0.79]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(CLOSURE))
import closure_lib as L  # noqa: E402
from stage4_readout import build_blocks, fit_block, group_boot_ci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.79)
    a = ap.parse_args()

    cfg = json.loads((HERE / "m3_concepts.json").read_text())
    dep = json.loads((HERE / "m3_depletion.json").read_text())

    # Rediscovery source of truth = the full-recall adjudicated readout (m3_recall.json).
    # The mechanical tau detector is NOT used: its entire bank-vs-fleet cosine range sits
    # below the tau band, so it returns 0 for held-out and retained concepts alike and
    # carries no information (see m3_detect.py docstring and m3_detection.json
    # aggregate.RANGE_WARNING).
    recall = json.loads((HERE / "m3_recall.json").read_text())
    redisc_by_rep = {rep: set(v) for rep, v in recall["aggregate"]["rediscovered_concepts"].items()}
    print("rediscovery source: m3_recall.json (rule=primary), "
          + ", ".join(f"{k}:{len(v)}" for k, v in redisc_by_rep.items()))

    pop, split, dsplit, *_ = build_blocks()
    y, nt, A, V = pop["y"], pop["ntitle"], pop["A"], pop["V"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    ymon = y[monm]

    def readout(r):
        va = np.full(len(y), np.nan)
        va[fitm] = r["oof_nl_fitmine"]
        va[monm] = r["nl_mon"]
        return {"n_features": r["n_features"],
                "VA_nl_MONITOR_all": L.auc(ymon, r["nl_mon"]),
                "VA_nl_honest_level_heldout1244": L.auc(y[held], va[held])}, va

    print("baseline (full bank) refit for a like-for-like comparison ...", flush=True)
    r_full = fit_block([V, A], fitm, monm, y, nt)
    rep_full, va_full = readout(r_full)
    print(json.dumps(rep_full), flush=True)

    fp = cfg["concept_footprints"]
    out = {"tau": a.tau, "baseline_full": rep_full, "replicates": {}}

    for rep in ("rep1", "rep2", "rep3"):
        if rep not in redisc_by_rep:
            continue
        held_out = cfg["replicates"][rep]
        redisc = set(redisc_by_rep[rep])
        assert redisc <= set(held_out), f"{rep}: rediscovered set escapes the holdout"

        drop_cols = sorted({j for c in held_out for j in fp[c]})
        readd_cols = sorted({j for c in redisc for j in fp[c]})
        keep_dep = [j for j in range(A.shape[1]) if j not in set(drop_cols)]
        keep_rec = sorted(set(keep_dep) | set(readd_cols))

        print(f"\n{rep}: rediscovered {len(redisc)}/{len(held_out)} concepts "
              f"-> re-adding {len(readd_cols)} of {len(drop_cols)} dropped columns", flush=True)
        t = time.time()
        r_rec = fit_block([V, A[:, keep_rec]], fitm, monm, y, nt)
        rep_rec, va_rec = readout(r_rec)
        print(json.dumps(rep_rec), f"({time.time()-t:.0f}s)", flush=True)

        d = dep[rep]
        rec = {
            "n_held_out": len(held_out), "n_rediscovered": len(redisc),
            "rediscovered_concepts": sorted(redisc),
            "not_rediscovered": sorted(set(held_out) - redisc),
            "AUC_full_honest": rep_full["VA_nl_honest_level_heldout1244"],
            "AUC_depleted_honest": d["VA_nl_honest_level_heldout1244"],
            "AUC_recovered_honest": rep_rec["VA_nl_honest_level_heldout1244"],
            "AUC_full_MONITOR": rep_full["VA_nl_MONITOR_all"],
            "AUC_depleted_MONITOR": d["VA_nl_MONITOR_all"],
            "AUC_recovered_MONITOR": rep_rec["VA_nl_MONITOR_all"],
        }
        gap = rec["AUC_full_honest"] - rec["AUC_depleted_honest"]
        rec["depletion_drop_honest"] = gap
        rec["recovery_honest"] = rec["AUC_recovered_honest"] - rec["AUC_depleted_honest"]
        rec["recovered_fraction_honest"] = rec["recovery_honest"] / gap if abs(gap) > 1e-9 else None
        gapm = rec["AUC_full_MONITOR"] - rec["AUC_depleted_MONITOR"]
        rec["depletion_drop_MONITOR"] = gapm
        rec["recovery_MONITOR"] = rec["AUC_recovered_MONITOR"] - rec["AUC_depleted_MONITOR"]
        rec["recovered_fraction_MONITOR"] = rec["recovery_MONITOR"] / gapm if abs(gapm) > 1e-9 else None
        # group-level paired bootstrap of the recovered-vs-depleted AUC delta,
        # using the depleted prediction vectors persisted by m3_deplete.py
        zd = np.load(HERE / "m3_depleted_preds.npz", allow_pickle=True)
        va_dep = np.full(len(y), np.nan)
        va_dep[fitm] = zd[f"oof_fitmine_{rep}"]
        va_dep[monm] = zd[f"nl_mon_{rep}"]
        rec["recovery_ci_honest"] = group_boot_ci(y[held], va_rec[held], va_dep[held], nt[held])
        rec["depletion_ci_honest"] = group_boot_ci(y[held], va_full[held], va_dep[held], nt[held])
        out["replicates"][rep] = rec

    fr = [v["recovered_fraction_honest"] for v in out["replicates"].values()
          if v["recovered_fraction_honest"] is not None]
    tot_gap = sum(v["depletion_drop_honest"] for v in out["replicates"].values())
    tot_rec = sum(v["recovery_honest"] for v in out["replicates"].values())
    out["pooled"] = {
        "mean_recovered_fraction_honest": float(np.mean(fr)) if fr else None,
        "pooled_recovered_fraction_honest": tot_rec / tot_gap if abs(tot_gap) > 1e-9 else None,
        "sum_depletion_drop_honest": tot_gap, "sum_recovery_honest": tot_rec,
        "n_rediscovered_total": sum(v["n_rediscovered"] for v in out["replicates"].values()),
        "n_held_out_total": sum(v["n_held_out"] for v in out["replicates"].values()),
    }
    (HERE / "m3_recovery.json").write_text(json.dumps(out, indent=2))
    print("\n", json.dumps(out["pooled"], indent=1))
    print("wrote", HERE / "m3_recovery.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FINGERPRINT-versus-OBSERVED-ORDINAL check for the FREEZE ADDENDUM 4 channel.

Addendum 4 asks the Track-B fleet to name the textual FINGERPRINT of an item's position
in its container.  This cell is the first in the programme where the observed ordinal
exists as a covariate (`created_utc`, 86.1% matched) AND the fleet named the channel
unanimously, so the two can be put on the same rows and compared directly -- the readout
math.SE round 1 could only do as a null (its conjectured arrival-order fingerprint read
.492 against an observed ordinal of .614).

For every Track-B channel this reports, on HONEST (all dense-held-out rows) and restricted
to rows where the timestamp matched:

  alone_AUC              the judged channel's own AUC against y
  rho_with_observed_era  Spearman of the judged channel against the observed era ordinal
  recovery               |alone_AUC - .5| / |observed_era_AUC - .5|, i.e. how much of the
                         observed ordinal's own predictive signal the fingerprint carries
  stack_era_plus_channel grouped-OOF logistic stack of the observed ordinal + the channel,
                         so a channel that adds signal BEYOND the observed ordinal is
                         visible (a fingerprint of something the ordinal does not capture)

NOTHING HERE FEEDS THE CLOSURE CURVE.  The observed ordinal is never banked, never judged
and never fitted into VA; this file is a validity check on the fleet's Track-B output.

CPU only.  Usage: python3 fingerprint_vs_observed.py --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent


def stack_oof(cols, y, groups):
    X = np.column_stack(cols)
    X = np.nan_to_num(X, nan=0.0)
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="jokes_community")
    ap.add_argument("--round", default="1")
    a = ap.parse_args()
    tag = f"{a.cell}_r{a.round}"

    d = C.load(a.cell)
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm = split == "fit_mine"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y, g = d["y"], d["groups"]

    era = d["created_utc"].astype(float)
    fin = np.isfinite(era)
    both = held & fin

    z = np.load(HERE / f"{tag}_scores.npz", allow_pickle=True)
    X, cids = z["X"], [str(s) for s in z["crit_ids"]]
    routing = json.loads((HERE / f"{tag}_routing_final.json").read_text())
    rows_by = {r["blind_id"]: r for r in routing["final"]}
    B_ids = [r["blind_id"] for r in routing["final"] if r["final_route"] == "B"]

    era_auc = float(roc_auc_score(y[both], era[both]))
    out = {"tag": tag,
           "note": "validity check on the Addendum-4 fingerprint; nothing here feeds the "
                   "closure curve, and the observed ordinal is never banked or judged",
           "n_HONEST": int(held.sum()),
           "n_HONEST_with_timestamp": int(both.sum()),
           "observed_era_AUC_on_matched_HONEST": era_auc,
           "channels": []}

    keep, med = L.clean_fit(X[fitm][:, [cids.index(i) for i in B_ids]])
    Xb = L.clean_apply(X[:, [cids.index(i) for i in B_ids]], keep, med)
    kept_ids = [B_ids[j] for j in keep]

    for k, bid in enumerate(kept_ids):
        col = Xb[:, k]
        auc = float(roc_auc_score(y[both], col[both]))
        rho = float(spearmanr(col[both], era[both]).statistic)
        rec = {
            "channel_id": bid, "name": rows_by[bid]["name"],
            "upstream_parent": rows_by[bid].get("upstream_parent"),
            "mixed": bool(rows_by[bid].get("mixed")),
            "alone_AUC_matched_HONEST": auc,
            "rho_with_observed_era": rho,
            "recovery_of_observed_ordinal": (abs(auc - .5) / abs(era_auc - .5)
                                             if abs(era_auc - .5) > 1e-9 else None),
            "stack_era_plus_channel": stack_oof([era[both], col[both]], y[both], g[both]),
        }
        rec["channel_increment_over_observed_era"] = rec["stack_era_plus_channel"] - era_auc
        out["channels"].append(rec)
    out["channels"].sort(key=lambda r: -abs(r["rho_with_observed_era"]))

    jointB = stack_oof([Xb[both][:, k] for k in range(Xb.shape[1])], y[both], g[both])
    out["joint_B_matched_HONEST"] = jointB
    out["stack_era_plus_all_B"] = stack_oof(
        [era[both]] + [Xb[both][:, k] for k in range(Xb.shape[1])], y[both], g[both])
    out["all_B_increment_over_observed_era"] = out["stack_era_plus_all_B"] - era_auc
    out["observed_era_increment_over_all_B"] = out["stack_era_plus_all_B"] - jointB

    (HERE / f"{tag}_fingerprint_vs_observed.json").write_text(json.dumps(out, indent=1))
    print(f"observed era ordinal AUC (matched HONEST, n={out['n_HONEST_with_timestamp']}): "
          f"{era_auc:.4f}")
    for r in out["channels"]:
        print(f"  rho={r['rho_with_observed_era']:+.3f} alone={r['alone_AUC_matched_HONEST']:.3f} "
              f"recov={r['recovery_of_observed_ordinal']:.2f}  {r['name'][:52]}")
    print(f"joint B {jointB:.4f} | era+allB {out['stack_era_plus_all_B']:.4f} | "
          f"allB increment over era {out['all_B_increment_over_observed_era']:+.4f} | "
          f"era increment over allB {out['observed_era_increment_over_all_B']:+.4f}")


if __name__ == "__main__":
    main()

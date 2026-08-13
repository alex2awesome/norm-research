#!/usr/bin/env python3
"""BBC most-read ROUND 3 readout (generalized from r1) (2026-08-13). Track-A closure step + spurious map.

Reuses round0_bbc.py verbatim for the frozen machinery (dense join gate, fit_va,
swap_pair, hash splits) — instrument identity with the round-0 anchor:
  VA_nl MONITOR .7502 (seedmean-pred) · T MONITOR .8231 · Delta_0 +.0749 · eps .005
  (paired-seed SD .00252 -> resolvable; two-consecutive-rounds rule binds).

Round 1 inputs: bbc_mostread_r1_scores.npz (25 mined criteria scored corpus-wide,
chunked Gemma pass, anchors PASS) + bbc_mostread_r1_routing_final.json (A=16 bank
joiners / B=9 nuisance channels, 4/4 planted probes, arbiter-final).

Readouts
  1. TRACK A   VA_1 = fit_va([V, A_bank, A_routed16]) on FITMINE -> MONITOR;
               Delta_1 = T - VA_1; gain = Delta_0 - Delta_1 vs eps. (Round 1 of the
               2-consecutive-sub-eps rule — the rule cannot fire this round.)
  2. SPURIOUS  per-B-channel alone-AUC on the dense-held-out rows; joint B
               (linear + GBM, grouped-OOF); stacked increments (B+dense vs B;
               B+bank+dense vs B+bank).
  3. SWAP      (C+, C-) with the r1 bank on dense-held-out.
  4. B-side missing mass copied from the species merge.

Run ON sk3 (data lives there):  nohup python readout_r1_bbc.py > readout_r1.log 2>&1 &
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round0_bbc as R0  # noqa: E402
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

OUT = HERE
EPS, D0 = 0.005, None  # Delta_0 read live from round0_results.json

res = {"cell": "bbc_mostread", "round": 3}

# ---- population, splits, bank, dense ----------------------------------------
pop = pd.read_csv(R0.VA_DIR / "population.csv.gz")
pop["row_id"] = pop.row_id.astype(str)
splits = pd.read_csv(HERE / "splits.csv.gz")
splits["row_id"] = splits.row_id.astype(str)
assert (splits.row_id.values == pop.row_id.values).all(), "splits/pop order drift"

meta, A, V, groups, shard, ids = SC.load_scaleupC_bank("bbc_mostread", out=R0.BANK_OUT)
ids = [str(i) for i in ids]
assert ids == pop.row_id.tolist(), "bank rows not aligned with population"
X0 = np.column_stack([V, A])

dj = R0.dense_join(pop)
hold_ids = dj["row_ids"]
dense_mean = np.mean([dj["dense_per_seed"][s] for s in R0.SEEDS], axis=0)
dense_by_id = dict(zip(hold_ids, dense_mean))

# ---- r1 mined scores + routing ----------------------------------------------
z = np.load(HERE / "bbc_mostread_r1_scores.npz", allow_pickle=True)
Xr1 = z["X"]
r1_ids = [str(x) for x in z["row_id"]]
assert r1_ids == pop.row_id.tolist(), "r1 scores not aligned with population"
cids = [str(c) for c in z["crit_ids"]]
routing = json.load(open(HERE / "bbc_mostread_r1_routing_final.json"))
track = {c["blind_id"]: c["final_route"] for c in routing["final"]}  # ARBITER-FINAL (audit_track is pre-arbiter; caught 2026-08-13)
iA = [i for i, c in enumerate(cids) if track.get(c) == "A"]
iB = [i for i, c in enumerate(cids) if track.get(c) == "B"]
res["routing"] = {"n_A": len(iA), "n_B": len(iB)}
A1, B1 = Xr1[:, iA], Xr1[:, iB]
namesA = [str(z["crit_names"][i]) for i in iA]
namesB = [str(z["crit_names"][i]) for i in iB]

y = pop.judgement.astype(int).values
g = pop.group.astype(str).values
fi = (splits.split3 == "FITMINE").values
mi = (splits.split3 == "MONITOR").values

# ---- r2 + r3 mined scores + routing --------------------------------------------
z2 = np.load(HERE / "bbc_mostread_r2_scores.npz", allow_pickle=True)
assert [str(x) for x in z2["row_id"]] == pop.row_id.tolist()
cids2 = [str(c) for c in z2["crit_ids"]]
routing2 = json.load(open(HERE / "bbc_mostread_r2_routing_final.json"))
track2 = {c["blind_id"]: c["final_route"] for c in routing2["final"]}
iA2 = [i for i, c in enumerate(cids2) if track2.get(c) == "A"]
iB2 = [i for i, c in enumerate(cids2) if track2.get(c) == "B"]
res["routing_r2"] = {"n_A": len(iA2), "n_B": len(iB2)}
A2, B2 = z2["X"][:, iA2], z2["X"][:, iB2]
namesB2 = [str(z2["crit_names"][i]) for i in iB2]
z3 = np.load(HERE / "bbc_mostread_r3_scores.npz", allow_pickle=True)
assert [str(x) for x in z3["row_id"]] == pop.row_id.tolist()
cids3 = [str(c) for c in z3["crit_ids"]]
routing3 = json.load(open(HERE / "bbc_mostread_r3_routing_final.json"))
track3 = {c["blind_id"]: c["final_route"] for c in routing3["final"]}
iA3 = [i for i, c in enumerate(cids3) if track3.get(c) == "A"]
iB3 = [i for i, c in enumerate(cids3) if track3.get(c) == "B"]
res["routing_r3"] = {"n_A": len(iA3), "n_B": len(iB3)}
A3, B3 = z3["X"][:, iA3], z3["X"][:, iB3]
namesB3 = [str(z3["crit_names"][i]) for i in iB3]

# ---- 1. TRACK A --------------------------------------------------------------
r2res = json.load(open(HERE / "readout_r2_results.json"))
VA1_prev = r2res["track_A"]["VA2_MONITOR_seedmean_pred"]
T_MON = r2res["track_A"]["T_MONITOR"]
D1 = T_MON - VA1_prev
X1 = np.column_stack([X0, A1, A2, A3])
info1, mon_lin1, mon_nl1 = R0.fit_va(X1[fi], X1[mi], y[fi], g[fi])
va1_pred = np.mean([mon_nl1[s] for s in (0, 1, 2)], axis=0)
VA1 = float(roc_auc_score(y[mi], va1_pred))
res["track_A"] = {
    "VA2_prev_MONITOR": VA1_prev, "T_MONITOR": T_MON, "Delta_2": D1,
    "VA3_MONITOR_seedmean_pred": VA1,
    "VA3_per_seed": {str(s): float(roc_auc_score(y[mi], mon_nl1[s])) for s in (0, 1, 2)},
    "VA3_lin_MONITOR": float(roc_auc_score(y[mi], mon_lin1)),
    "Delta_3": T_MON - VA1, "gain": D1 - (T_MON - VA1), "epsilon": EPS,
    "sub_eps": bool(D1 - (T_MON - VA1) < EPS),
    "rule_note": "r1 +.0084 and r2 +.0067 were both SUPER-eps; a sub-eps gain here is the FIRST of two required",
    "fitmine_oof": info1,
}
print(f"[trackA] VA2 {VA1_prev:.4f} -> VA3 {VA1:.4f} | Delta {D1:+.4f} -> "
      f"{T_MON - VA1:+.4f} | gain {res['track_A']['gain']:+.4f} "
      f"(eps {EPS})", flush=True)

# ---- 2. SPURIOUS map on dense-held-out ---------------------------------------
hh = splits.in_heldout.values.astype(bool)
yh, gh = y[hh], g[hh]
dh = np.array([dense_by_id[r] for r in pop.row_id[hh]])
def alone(col):
    ok = ~np.isnan(col)
    if ok.sum() < 100 or len(set(yh[ok])) < 2:
        return None
    return float(roc_auc_score(yh[ok], col[ok]))
Ball = np.column_stack([B1, B2, B3]); namesBall = namesB + namesB2 + namesB3
res["spurious_map"] = sorted(
    [{"name": n, "alone_auc_heldout": alone(Ball[hh][:, j])} for j, n in enumerate(namesBall)],
    key=lambda r: -abs((r["alone_auc_heldout"] or .5) - .5))
Bh = np.where(np.isnan(Ball[hh]), np.nanmedian(Ball[hh], axis=0), Ball[hh])
def grouped_oof(X_, y_, g_):
    oof = np.zeros(len(y_))
    for tr, te in GroupKFold(5).split(X_, groups=g_):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_[tr], y_[tr])
        oof[te] = clf.predict_proba(X_[te])[:, 1]
    return float(roc_auc_score(y_, oof)), oof
bank_h_pred = grouped_oof(np.where(np.isnan(X1[hh]), np.nanmedian(X1[hh], axis=0), X1[hh]), yh, gh)
jB, oofB = grouped_oof(Bh, yh, gh)
jBd, _ = grouped_oof(np.column_stack([Bh, dh]), yh, gh)
jBbank, oofBbank = grouped_oof(np.column_stack([Bh, np.where(np.isnan(X1[hh]), np.nanmedian(X1[hh], axis=0), X1[hh])]), yh, gh)
jBbankd, _ = grouped_oof(np.column_stack([Bh, np.where(np.isnan(X1[hh]), np.nanmedian(X1[hh], axis=0), X1[hh]), dh]), yh, gh)
res["discount_stack"] = {
    "joint_B_alone": jB, "B_plus_dense": jBd, "dense_increment_over_B": jBd - jB,
    "B_plus_bank_r1_lin": jBbank, "B_plus_bank_plus_dense": jBbankd,
    "dense_increment_over_B_plus_bank": jBbankd - jBbank,
    "note": "linear grouped-OOF stacks on dense-held-out (n=%d)" % int(hh.sum()),
}
print(f"[spurious] joint B {jB:.4f} | dense over B +{jBd-jB:.4f} | "
      f"dense over B+bank +{jBbankd-jBbank:.4f}", flush=True)

# ---- 3. swap + 4. mass --------------------------------------------------------
res["swap"] = R0.swap_pair(yh, dh, oofBbank)
species = json.load(open(HERE / "bbc_mostread_r3_species.json"))
res["missing_mass"] = species.get("blind_merge")

(OUT / "readout_r3_results.json").write_text(json.dumps(res, indent=1, default=float))
print("READOUT_R3_DONE", flush=True)

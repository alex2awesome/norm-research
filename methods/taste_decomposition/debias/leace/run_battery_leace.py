#!/usr/bin/env python3
"""LEACE closed-form-projection pilot -- planted battery V1-V4 + the
utility-distortion frontier.  Successor instrument to the retired GRL
(notes/2026-08-07__debias_audit_fable.md).

Procedure per arm (spec, user-approved 2026-08-10):
  1. pooled representations h from the FROZEN trained model (already extracted:
     B01 planted-vanilla / B00 real-vanilla rep_h; R06 realtok-vanilla,
     R08 v3b-vanilla rep; R06 stripped-text rep via extract_reps_leace.py);
  2. closed-form LEACE eraser for the named concept(s), fit on TRAIN rows only
     (leace.py, verified vs synthetic hand-check AND machine-equal to the
     concept-erasure reference implementation with shrinkage off);
  3. refit the score head -- StandardScaler + LogisticRegression(C=1),
     TRAIN rows only -- on projected reps; baseline = same head on raw reps;
  4. readouts below.  All CIs: paired docket-level bootstrap, 2,000 resamples.

Gates:
  V1  EXPLOIT      linear probe on raw B01 h reads the plant >= .95
  V2  REMOVAL      (a) linear probe on plant-projected B01 h <= .55 (LEACE
                   guarantee; failure = implementation bug)
                   (b) NONLINEAR (2-layer MLP, probe_reps machinery) probe on
                   projected reps: <= .55 full pass, else linear-only erasure
                   with the residue quantified
                   (c) utility cost: refit-head AUC projected vs raw
                   (d) planted-vs-unplanted head jump must vanish
  V3a SPECIFICITY  LEACE the 26 standard nuisances on R06: the unnamed
                   real-signal token must survive (probe + ablation delta)
  V3b SPECIFICITY  LEACE the date channel on the year-balanced R08 arm
                   (date verifiably unused): task cost < .005
  V4  CONSISTENCY  LEACE length on B00: implied influence vs the stacked-
                   increment and matched-sampling instruments recomputed on
                   THIS battery's own raw-head eval preds -- same sign, within ~2x
  FRONTIER         head AUC after erasing 1 / 11 / 22 / 45 mined Track-B
                   channels (+ length-2 and standard-26 context points) on B00

Usage:
  python run_battery_leace.py --reps_root <dir with B00.../reps.npz mirrors>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DEBIAS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEBIAS))
from leace import LeaceEraser, precompute_eig     # noqa: E402
from probe_reps import fit_probe, SEEDS           # noqa: E402

RES = HERE / "results"
RES.mkdir(exist_ok=True)
N_BOOT = 2000
RNG = np.random.default_rng(0)
PROBE_DEVICE = "cpu"


# ------------------------------------------------------------------ utils --
def onehot_bins(Zraw, tr, n_bins=10):
    """Encode a concept matrix CATEGORICALLY so the strong LEACE certificate
    (no linear classifier under any convex loss; Belrose Thm 3.1, one-hot Z)
    applies -- the continuous form is OLS-loss-only (their S4.3; lit review R1a).

    Per column: <= 2 train-distinct values -> kept as the (already categorical)
    binary column; else one-hot over train-DECILE bins (edges = train quantiles
    .1...9, deduplicated; bins empty on train dropped).  The .5 quantile is
    always among the edges, so every median-split probe target used by the
    battery is a UNION of the erased categories -- a linear function of the
    one-hot Z -- and inherits the certificate.  Returns (Z_onehot, cols_per_channel).
    """
    cols, meta = [], []
    for j in range(Zraw.shape[1]):
        v = Zraw[:, j].astype(np.float64)
        if len(np.unique(v[tr])) <= 2:
            cols.append(v[:, None])
            meta.append(1)
            continue
        edges = np.unique(np.quantile(v[tr], np.arange(1, n_bins) / n_bins))
        idx = np.digitize(v, edges)              # 0..len(edges); x == edge -> upper bin
        oh = np.zeros((len(v), len(edges) + 1))
        oh[np.arange(len(v)), idx] = 1.0
        keep = oh[tr].sum(0) > 0
        oh = oh[:, keep]
        cols.append(oh)
        meta.append(int(oh.shape[1]))
    return np.hstack(cols), meta


CERT_CAT = "categorical (train-decile one-hot; Belrose Thm 3.1 any-convex-loss form)"
CERT_CONT = "continuous (OLS-loss-only form, Belrose S4.3) -- secondary readout"


def head_fit_predict(X, y, tr):
    """The refit score head: logistic on (projected) reps, train rows only."""
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X[tr], y[tr])
    return m, m.predict_proba(X)[:, 1]


def linear_probe(X, t, tr, ev):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X[tr], t[tr])
    return float(roc_auc_score(t[ev], m.predict_proba(X[ev])[:, 1]))


def mlp_probe(X, t, tr_idx, va_idx, te_idx):
    """probe_reps.py protocol: standardize on train, 3 seeds, mean eval AUC."""
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xs = ((X - mu) / sd).astype(np.float32)
    aucs = [fit_probe(Xs[tr_idx], t[tr_idx], Xs[va_idx], t[va_idx],
                      Xs[te_idx], t[te_idx], s, PROBE_DEVICE)[0] for s in SEEDS]
    return {"auc_eval_mean": float(np.mean(aucs)), "auc_eval_seeds": aucs}


def probe_splits(split):
    """train/inner-val/eval indices exactly as probe_reps.py builds them."""
    rng = np.random.default_rng(0)
    tr_all = np.flatnonzero(split == "train")
    hold = rng.permutation(len(tr_all))
    va_idx = tr_all[hold[: max(1, len(tr_all) // 10)]]
    tr_idx = tr_all[hold[len(tr_all) // 10:]]
    te_idx = np.flatnonzero(split == "eval")
    return tr_idx, va_idx, te_idx


def boot_diff(y, pa, pb, dockets, split_mask, n=N_BOOT):
    """Docket-level bootstrap of AUC(pa) - AUC(pb) on the masked rows (paired)."""
    yy, a, b, d = y[split_mask], pa[split_mask], pb[split_mask], dockets[split_mask]
    uniq = np.unique(d)
    idx_by = {u: np.flatnonzero(d == u) for u in uniq}
    point = roc_auc_score(yy, a) - roc_auc_score(yy, b)
    out = []
    for _ in range(n):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx_by[u] for u in pick])
        if len(set(yy[ii])) < 2:
            continue
        out.append(roc_auc_score(yy[ii], a[ii]) - roc_auc_score(yy[ii], b[ii]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)], "n_boot": len(out)}


def eraser_diag(er, X, Z, ev):
    return {"proj_rank": er.proj_rank, "cov_rank": er.cov_rank,
            "crosscov_train_max_raw": None,  # filled by caller on train rows
            "crosscov_eval_max_corr": float(er.crosscov_max(X[ev], Z[ev], standardize=True)),
            "distortion_all_rows": er.distortion(X)}


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps_root", required=True)
    ap.add_argument("--skip_mlp", action="store_true", help="debug only")
    args = ap.parse_args()
    reps_root = Path(args.reps_root)
    t00 = time.time()

    nz = np.load(DEBIAS / "build/nuisance.npz", allow_pickle=True)
    nz_ids = np.array([str(s) for s in nz["doc_id"]])
    nz_pos = {d: i for i, d in enumerate(nz_ids)}
    Znames = [str(s) for s in nz["names"]]
    groups = json.loads(str(nz["groups_json"]))
    Zstd = nz["Z"].astype(np.float64)                  # 26 standardized nuisances
    plant_all = nz["plant"].astype(int)
    realtok_all = nz["realtok"].astype(int)
    docket_all = np.array([str(s) for s in nz["docket"]])

    mined = np.load(HERE / "build/mined_channels_nc45.npz", allow_pickle=True)
    assert (np.array([str(s) for s in mined["doc_id"]]) == nz_ids).all()
    MX = mined["X"].astype(np.float64)                 # 9521 x 45
    mrank = mined["rank_by_train_dev"].astype(int)

    def load_arm(tag, rep_key="rep", fname="reps.npz"):
        z = np.load(reps_root / tag / fname, allow_pickle=True)
        ids = np.array([str(s) for s in z["doc_id"]])
        pos = np.array([nz_pos[d] for d in ids])
        arm = {
            "tag": tag, "pos": pos,
            "X": z[rep_key].astype(np.float64),
            "split": np.array([str(s) for s in z["split"]]),
            "y": z["y"].astype(int),
            "model_prob": z["prob"].astype(float),
            "docket": docket_all[pos],
        }
        arm["tr"] = arm["split"] == "train"
        arm["ev"] = arm["split"] == "eval"
        arm["te"] = arm["split"] == "test"
        return arm

    print("[load] arms ...", flush=True)
    B00 = load_arm("B00_vanilla_real", "rep_h")
    B01 = load_arm("B01_vanilla_planted", "rep_h")
    R00 = load_arm("R00_vanilla_real", "rep")     # pooled-arch unplanted baseline (V3a jump)
    R06 = load_arm("R06_vanilla_realtok", "rep")
    R08 = load_arm("R08_vanilla_v3b", "rep")
    stripped_path = reps_root / "R06_vanilla_realtok" / "reps_stripped.npz"
    R06s = load_arm("R06_vanilla_realtok", "rep", "reps_stripped.npz") if stripped_path.exists() else None
    assert (B00["pos"] == B01["pos"]).all() and (B00["pos"] == R06["pos"]).all()
    if R06s is not None:
        assert (R06s["pos"] == R06["pos"]).all()

    out = {"spec": "LEACE pilot battery, notes/2026-08-10__leace_pilot.md",
           "eraser": "closed-form LEACE (Belrose et al. 2023 Thm 4.3), fit on TRAIN rows only",
           "certificate_policy": "PRIMARY erasers use CATEGORICAL one-hot Z (train-decile "
                                 "bins; any-convex-loss guardedness, Belrose Thm 3.1). "
                                 "Continuous-Z erasers (OLS-form only, their S4.3) are "
                                 "reported as secondary pair readouts. Lit review R1a fix; "
                                 "run-1 continuous-only battery preserved as "
                                 "battery_leace_r1_continuousZ.json",
           "verification": "synthetic selfcheck PASS; machine-equal (1.4e-13) to concept-erasure "
                           "LeaceFitter with shrinkage=False",
           "head": "StandardScaler + LogisticRegression(C=1, max_iter=2000), train rows only",
           "n_boot": N_BOOT, "arms": {}, "gates": {}}

    # baseline heads on raw reps ---------------------------------------------
    heads = {}
    for arm in (B00, B01, R00, R06, R08):
        t0 = time.time()
        _, p = head_fit_predict(arm["X"], arm["y"], arm["tr"])
        heads[arm["tag"]] = p
        out["arms"][arm["tag"]] = {
            "n": len(arm["y"]),
            "head_raw_auc_eval": float(roc_auc_score(arm["y"][arm["ev"]], p[arm["ev"]])),
            "head_raw_auc_test": float(roc_auc_score(arm["y"][arm["te"]], p[arm["te"]])),
            "model_head_auc_eval": float(roc_auc_score(arm["y"][arm["ev"]], arm["model_prob"][arm["ev"]])),
            "fit_sec": round(time.time() - t0, 1),
        }
        print(f"[head] {arm['tag']} raw eval AUC "
              f"{out['arms'][arm['tag']]['head_raw_auc_eval']:.4f} "
              f"(model head {out['arms'][arm['tag']]['model_head_auc_eval']:.4f})", flush=True)

    # ======================================================================
    # V1 EXPLOIT -- plant readable from raw planted-vanilla h
    # ======================================================================
    print("[V1] ...", flush=True)
    plant = plant_all[B01["pos"]]
    lin_v1 = linear_probe(B01["X"], plant, B01["tr"], B01["ev"])
    tr_i, va_i, te_i = probe_splits(B01["split"])
    mlp_v1 = None if args.skip_mlp else mlp_probe(B01["X"], plant, tr_i, va_i, te_i)
    jump_raw = boot_diff(B01["y"], heads["B01_vanilla_planted"], heads["B00_vanilla_real"],
                         B01["docket"], B01["ev"])
    out["gates"]["V1"] = {
        "linear_probe_plant_raw_h": lin_v1, "threshold": ">= .95",
        "mlp_probe_plant_raw_h": mlp_v1,
        "head_jump_planted_minus_unplanted_eval": jump_raw,
        "PASS": bool(lin_v1 >= 0.95),
    }

    # ======================================================================
    # V2 REMOVAL -- LEACE the plant indicator on B01
    # ======================================================================
    print("[V2] fitting plant eraser on B01 train h ...", flush=True)
    eig_b01 = precompute_eig(B01["X"][B01["tr"]])
    er_plant = LeaceEraser().fit(B01["X"][B01["tr"]], plant[B01["tr"]].astype(float), eig=eig_b01)
    XP = er_plant.apply(B01["X"])
    d2 = eraser_diag(er_plant, B01["X"], plant.astype(float)[:, None], B01["ev"])
    d2["crosscov_train_max_raw"] = float(er_plant.crosscov_max(
        B01["X"][B01["tr"]], plant[B01["tr"]].astype(float)))

    lin_after = linear_probe(XP, plant, B01["tr"], B01["ev"])
    mlp_after = None if args.skip_mlp else mlp_probe(XP, plant, tr_i, va_i, te_i)
    _, pP = head_fit_predict(XP, B01["y"], B01["tr"])
    util = boot_diff(B01["y"], pP, heads["B01_vanilla_planted"], B01["docket"], B01["ev"])
    util_test = boot_diff(B01["y"], pP, heads["B01_vanilla_planted"], B01["docket"], B01["te"])
    jump_after = boot_diff(B01["y"], pP, heads["B00_vanilla_real"], B01["docket"], B01["ev"])
    jump_after_test = boot_diff(B01["y"], pP, heads["B00_vanilla_real"], B01["docket"], B01["te"])

    # placebo controls, both erased from the SAME B01 reps with the same machinery:
    #   placebo_indep  -- random flag, plant's base rate, independent of y
    #                     (expected cost ~ 0: erasing a concept the reps don't
    #                     carry and y doesn't touch should be nearly free)
    #   placebo_ymatch -- random flag with the plant's exact P(flag|y) rates but
    #                     NO textual presence (isolates the y-correlated-slice
    #                     cost that erasing ANY y-correlated concept pays,
    #                     separate from removing the actually-planted channel)
    prng = np.random.default_rng(7)
    flag_i = (prng.random(len(plant)) < plant.mean()).astype(float)
    p1, p0 = plant[B01["y"] == 1].mean(), plant[B01["y"] == 0].mean()
    flag_y = np.where(B01["y"] == 1, prng.random(len(plant)) < p1,
                      prng.random(len(plant)) < p0).astype(float)
    placebo = {}
    for nm, fl in (("placebo_indep", flag_i), ("placebo_ymatch", flag_y)):
        erp = LeaceEraser().fit(B01["X"][B01["tr"]], fl[B01["tr"]], eig=eig_b01)
        _, pp = head_fit_predict(erp.apply(B01["X"]), B01["y"], B01["tr"])
        placebo[nm] = {
            "certificate_form": "categorical (binary flag)",
            "flag_auc_vs_y": float(roc_auc_score(B01["y"], fl)),
            "cost_eval": boot_diff(B01["y"], pp, heads["B01_vanilla_planted"],
                                   B01["docket"], B01["ev"]),
            "distortion": erp.distortion(B01["X"]),
        }
        print(f"[V2 placebo] {nm} cost {placebo[nm]['cost_eval']['diff']:+.4f}", flush=True)

    mlp_auc = None if mlp_after is None else mlp_after["auc_eval_mean"]
    out["gates"]["V2"] = {
        "certificate_form": "categorical (binary plant indicator is one-hot k=2; "
                            "any-convex-loss form holds as-is)",
        "eraser_diag": d2,
        "a_linear_probe_after": {"auc": lin_after, "threshold": "<= .55",
                                 "PASS": bool(lin_after <= 0.55)},
        "b_mlp_probe_after": {"probe": mlp_after, "threshold": "<= .55 full pass",
                              "nonlinear_residue": (None if mlp_auc is None
                                                    else float(max(0.0, mlp_auc - 0.55))),
                              "FULL_PASS": (None if mlp_auc is None else bool(mlp_auc <= 0.55))},
        "c_utility_cost_eval": util, "c_utility_cost_test": util_test,
        "placebo_controls": placebo,
        "d_jump_after_eval": jump_after, "d_jump_after_test": jump_after_test,
        "d_jump_before_eval": out["gates"]["V1"]["head_jump_planted_minus_unplanted_eval"],
        "d_note": "spec-literal gate |jump_after| <= .005; noise-limited at n=953 "
                  "(CI half-width ~.02) -- CI-covers-0 reported alongside",
        "d_PASS_literal": bool(abs(jump_after["diff"]) <= 0.005),
        "d_PASS_ci": bool(jump_after["ci95"][0] <= 0.0 <= jump_after["ci95"][1]),
    }
    print(f"[V2] lin {lin_after:.4f} mlp {mlp_auc} jump_after {jump_after['diff']:+.4f} "
          f"util {util['diff']:+.4f}", flush=True)

    # ======================================================================
    # V3a SPECIFICITY -- LEACE the 26 standard nuisances on R06 (realtok arm)
    # ======================================================================
    print("[V3a] fitting standard-26 eraser (categorical one-hot) on R06 train h ...", flush=True)
    realtok = realtok_all[R06["pos"]]
    eig_r06 = precompute_eig(R06["X"][R06["tr"]])
    Zs6 = Zstd[R06["pos"]][:, groups["standard"]]
    Zs6_oh, s6_cols = onehot_bins(Zs6, R06["tr"])
    er_std6 = LeaceEraser().fit(R06["X"][R06["tr"]], Zs6_oh[R06["tr"]], eig=eig_r06)
    X6P = er_std6.apply(R06["X"])
    d3 = eraser_diag(er_std6, R06["X"], Zs6_oh, R06["ev"])
    d3["crosscov_train_max_raw"] = float(er_std6.crosscov_max(R06["X"][R06["tr"]], Zs6_oh[R06["tr"]]))
    d3["certificate_form"] = CERT_CAT
    d3["k_raw_channels"], d3["k_onehot_cols"] = int(Zs6.shape[1]), int(Zs6_oh.shape[1])
    # continuous-Z secondary (OLS-form pair readout, lit review R1a "erase both")
    er_std6c = LeaceEraser().fit(R06["X"][R06["tr"]], Zs6[R06["tr"]], eig=eig_r06)
    X6Pc = er_std6c.apply(R06["X"])

    tr6, va6, te6 = probe_splits(R06["split"])
    rt_lin_raw = linear_probe(R06["X"], realtok, R06["tr"], R06["ev"])
    rt_lin_after = linear_probe(X6P, realtok, R06["tr"], R06["ev"])
    rt_mlp_raw = None if args.skip_mlp else mlp_probe(R06["X"], realtok, tr6, va6, te6)
    rt_mlp_after = None if args.skip_mlp else mlp_probe(X6P, realtok, tr6, va6, te6)
    # nuisance kill check (a named channel must be linearly dead after)
    len_bin = (Zstd[R06["pos"], Znames.index("char_len")]
               >= np.median(Zstd[R06["pos"], Znames.index("char_len")][R06["tr"]])).astype(int)
    len_lin_after = linear_probe(X6P, len_bin, R06["tr"], R06["ev"])

    _, p6P = head_fit_predict(X6P, R06["y"], R06["tr"])
    cost6 = boot_diff(R06["y"], p6P, heads["R06_vanilla_realtok"], R06["docket"], R06["ev"])
    _, p6Pc = head_fit_predict(X6Pc, R06["y"], R06["tr"])
    cost6c = boot_diff(R06["y"], p6Pc, heads["R06_vanilla_realtok"], R06["docket"], R06["ev"])
    jump6_raw = boot_diff(R06["y"], heads["R06_vanilla_realtok"], heads["R00_vanilla_real"],
                          R06["docket"], R06["ev"])

    v3a = {"certificate_form": CERT_CAT,
           "eraser_diag": d3,
           "continuous_Z_secondary": {
               "certificate_form": CERT_CONT,
               "proj_rank": er_std6c.proj_rank,
               "probe_realtok_linear_after": linear_probe(X6Pc, realtok, R06["tr"], R06["ev"]),
               "probe_charlen_linear_after": linear_probe(X6Pc, len_bin, R06["tr"], R06["ev"]),
               "head_utility_cost_eval": cost6c},
           "probe_realtok_linear_raw_vs_after": [rt_lin_raw, rt_lin_after],
           "probe_realtok_mlp_raw_vs_after": [None if rt_mlp_raw is None else rt_mlp_raw["auc_eval_mean"],
                                              None if rt_mlp_after is None else rt_mlp_after["auc_eval_mean"]],
           "probe_charlen_linear_after (must be dead)": len_lin_after,
           "head_utility_cost_eval (erasing 26 real nuisances)": cost6}

    if R06s is not None:
        # heads fit on token-present reps, scored on token-present AND stripped reps
        h_raw, p_raw_tok = head_fit_predict(R06["X"], R06["y"], R06["tr"])
        p_raw_strip = h_raw.predict_proba(R06s["X"])[:, 1]
        h_prj, p_prj_tok = head_fit_predict(X6P, R06["y"], R06["tr"])
        p_prj_strip = h_prj.predict_proba(er_std6.apply(R06s["X"]))[:, 1]
        dr = boot_diff(R06["y"], p_raw_tok, p_raw_strip, R06["docket"], R06["ev"])
        dp = boot_diff(R06["y"], p_prj_tok, p_prj_strip, R06["docket"], R06["ev"])
        v3a["ablation_delta_raw_eval"] = dr
        v3a["ablation_delta_projected_eval"] = dp
        v3a["difference_of_deltas"] = float(dp["diff"] - dr["diff"])
        v3a["delta_gate"] = "|delta_proj - delta_raw| < .005"
        v3a["PASS_ablation"] = bool(abs(dp["diff"] - dr["diff"]) < 0.005)
        v3a["model_head_ablation_delta_eval (context)"] = 0.0272
    v3a["jump_vs_R00_raw_eval (context)"] = jump6_raw
    v3a["PASS_probe_survival"] = bool(rt_lin_after >= 0.5 + 0.5 * (rt_lin_raw - 0.5))
    out["gates"]["V3a"] = v3a
    print(f"[V3a] realtok lin {rt_lin_raw:.3f}->{rt_lin_after:.3f} "
          f"charlen after {len_lin_after:.3f} cost {cost6['diff']:+.4f}", flush=True)

    # ======================================================================
    # V3b SPECIFICITY -- LEACE the date channel on R08 (year-balanced arm)
    # ======================================================================
    print("[V3b] fitting date eraser (categorical one-hot) on R08 train h ...", flush=True)
    eig_r08 = precompute_eig(R08["X"][R08["tr"]])
    Zd8 = Zstd[R08["pos"]][:, groups["date"]]
    Zd8_oh, d8_cols = onehot_bins(Zd8, R08["tr"])
    er_date = LeaceEraser().fit(R08["X"][R08["tr"]], Zd8_oh[R08["tr"]], eig=eig_r08)
    X8P = er_date.apply(R08["X"])
    d3b = eraser_diag(er_date, R08["X"], Zd8_oh, R08["ev"])
    d3b["crosscov_train_max_raw"] = float(er_date.crosscov_max(R08["X"][R08["tr"]], Zd8_oh[R08["tr"]]))
    d3b["certificate_form"] = CERT_CAT
    d3b["k_raw_channels"], d3b["k_onehot_cols"] = int(Zd8.shape[1]), int(Zd8_oh.shape[1])
    yr_bin = (Zd8[:, 0] >= np.median(Zd8[R08["tr"], 0])).astype(int)
    yr_lin_raw = linear_probe(R08["X"], yr_bin, R08["tr"], R08["ev"])
    yr_lin_after = linear_probe(X8P, yr_bin, R08["tr"], R08["ev"])
    _, p8P = head_fit_predict(X8P, R08["y"], R08["tr"])
    cost8 = boot_diff(R08["y"], p8P, heads["R08_vanilla_v3b"], R08["docket"], R08["ev"])
    er_datec = LeaceEraser().fit(R08["X"][R08["tr"]], Zd8[R08["tr"]], eig=eig_r08)
    _, p8Pc = head_fit_predict(er_datec.apply(R08["X"]), R08["y"], R08["tr"])
    cost8c = boot_diff(R08["y"], p8Pc, heads["R08_vanilla_v3b"], R08["docket"], R08["ev"])
    out["gates"]["V3b"] = {
        "certificate_form": CERT_CAT,
        "continuous_Z_secondary": {"certificate_form": CERT_CONT,
                                   "proj_rank": er_datec.proj_rank,
                                   "task_cost_eval": cost8c},
        "eraser_diag": d3b,
        "probe_docket_year_linear_raw_vs_after": [yr_lin_raw, yr_lin_after],
        "task_cost_eval": cost8, "threshold": "|cost| < .005",
        "PASS_literal": bool(abs(cost8["diff"]) < 0.005),
        "PASS_ci": bool(cost8["ci95"][0] <= 0.0 <= cost8["ci95"][1]),
    }
    print(f"[V3b] year lin {yr_lin_raw:.3f}->{yr_lin_after:.3f} cost {cost8['diff']:+.4f}", flush=True)

    # ======================================================================
    # V4 CONSISTENCY -- LEACE length on B00 (real corpus)
    # ======================================================================
    print("[V4] fitting length eraser (categorical one-hot) on B00 train h ...", flush=True)
    eig_b00 = precompute_eig(B00["X"][B00["tr"]])
    Zl0 = Zstd[B00["pos"]][:, groups["length"]]
    Zl0_oh, l0_cols = onehot_bins(Zl0, B00["tr"])
    er_len = LeaceEraser().fit(B00["X"][B00["tr"]], Zl0_oh[B00["tr"]], eig=eig_b00)
    X0P = er_len.apply(B00["X"])
    d4 = eraser_diag(er_len, B00["X"], Zl0_oh, B00["ev"])
    d4["crosscov_train_max_raw"] = float(er_len.crosscov_max(B00["X"][B00["tr"]], Zl0_oh[B00["tr"]]))
    d4["certificate_form"] = CERT_CAT
    d4["k_raw_channels"], d4["k_onehot_cols"] = int(Zl0.shape[1]), int(Zl0_oh.shape[1])
    _, p0P = head_fit_predict(X0P, B00["y"], B00["tr"])
    infl = boot_diff(B00["y"], heads["B00_vanilla_real"], p0P, B00["docket"], B00["ev"])
    er_lenc = LeaceEraser().fit(B00["X"][B00["tr"]], Zl0[B00["tr"]], eig=eig_b00)
    X0Pc = er_lenc.apply(B00["X"])
    _, p0Pc = head_fit_predict(X0Pc, B00["y"], B00["tr"])
    infl_c = boot_diff(B00["y"], heads["B00_vanilla_real"], p0Pc, B00["docket"], B00["ev"])
    cl_bin = (Zl0[:, 0] >= np.median(Zl0[B00["tr"], 0])).astype(int)
    cl_lin_raw = linear_probe(B00["X"], cl_bin, B00["tr"], B00["ev"])
    cl_lin_after = linear_probe(X0P, cl_bin, B00["tr"], B00["ev"])
    cl_lin_after_c = linear_probe(X0Pc, cl_bin, B00["tr"], B00["ev"])

    # same-regime instruments on THIS battery's own raw-head eval preds
    print("[V4] same-regime instruments on refit raw head ...", flush=True)
    from analyze_battery import same_regime_length_instruments  # noqa: E402
    preds_df = pd.DataFrame({"doc_id": nz_ids[B00["pos"]], "docket": B00["docket"],
                             "judgement": B00["y"], "split": B00["split"],
                             "prob": heads["B00_vanilla_real"]})
    try:
        inst = same_regime_length_instruments(preds_df)
    except Exception as e:
        inst = {"error": f"{type(e).__name__}: {e}"}
    stack = inst.get("stacked_increment_length_over_T")
    match_drops = [v["drop"] for v in inst.get("matched_sampling", {}).values()] or None

    def consistent(a, b, tol=2.0, null=0.01):
        if a is None or b is None:
            return None
        if abs(a) < null and abs(b) < null:
            return True                      # both null -> consistent
        if a * b <= 0:
            return False
        r = abs(a) / abs(b)
        return bool(1.0 / tol <= r <= tol)

    out["gates"]["V4"] = {
        "certificate_form": CERT_CAT,
        "continuous_Z_secondary": {"certificate_form": CERT_CONT,
                                   "proj_rank": er_lenc.proj_rank,
                                   "implied_influence_eval": infl_c,
                                   "probe_charlen_linear_after": cl_lin_after_c},
        "eraser_diag": d4,
        "probe_charlen_linear_raw_vs_after": [cl_lin_raw, cl_lin_after],
        "implied_influence_eval (head_raw - head_lenprojected)": infl,
        "instruments_on_this_raw_head": inst,
        "vs_stacked_increment": {"stacked": stack,
                                 "consistent_within_2x_or_both_null": consistent(infl["diff"], stack)},
        "vs_matched_sampling": {"matched_drops": match_drops,
                                "consistent_within_2x_or_both_null":
                                    None if match_drops is None else
                                    consistent(infl["diff"], float(np.mean(match_drops)))},
        "frozen_R00_instruments (context)": "results/same_regime_length_instruments.json",
    }
    print(f"[V4] implied influence {infl['diff']:+.4f} stack {stack} "
          f"matched {match_drops}", flush=True)

    # ======================================================================
    # FRONTIER -- erase 1/11/22/45 mined channels (+ length-2, standard-26)
    # ======================================================================
    print("[frontier] ...", flush=True)
    base_auc = out["arms"]["B00_vanilla_real"]["head_raw_auc_eval"]
    MX0 = MX[B00["pos"]]
    sets = [("length_2", Zl0),
            ("standard_26", Zstd[B00["pos"]][:, groups["standard"]]),
            ("mined_top1", MX0[:, mrank[:1]]),
            ("mined_top11", MX0[:, mrank[:11]]),
            ("mined_top22", MX0[:, mrank[:22]]),
            ("mined_45", MX0),
            ("standard26_plus_mined45", np.hstack([Zstd[B00["pos"]][:, groups["standard"]], MX0]))]
    frontier = []
    for name, Zk in sets:
        for form in ("categorical", "continuous"):
            if form == "categorical":
                if name == "length_2":
                    erk, pkP, Zev = er_len, p0P, Zl0_oh
                else:
                    Zev, _ = onehot_bins(Zk, B00["tr"])
                    erk = LeaceEraser().fit(B00["X"][B00["tr"]], Zev[B00["tr"]], eig=eig_b00)
                    _, pkP = head_fit_predict(erk.apply(B00["X"]), B00["y"], B00["tr"])
            else:
                if name == "length_2":
                    erk, pkP, Zev = er_lenc, p0Pc, Zl0
                else:
                    Zev = Zk
                    erk = LeaceEraser().fit(B00["X"][B00["tr"]], Zev[B00["tr"]], eig=eig_b00)
                    _, pkP = head_fit_predict(erk.apply(B00["X"]), B00["y"], B00["tr"])
            a_ev = float(roc_auc_score(B00["y"][B00["ev"]], pkP[B00["ev"]]))
            a_te = float(roc_auc_score(B00["y"][B00["te"]], pkP[B00["te"]]))
            frontier.append({"set": name, "certificate_form": form,
                             "k_raw_channels": int(Zk.shape[1]),
                             "k_concept_cols": int(Zev.shape[1]),
                             "proj_rank": erk.proj_rank,
                             "auc_eval": a_ev, "auc_test": a_te,
                             "delta_vs_raw_eval": float(a_ev - base_auc),
                             "distortion": erk.distortion(B00["X"]),
                             "crosscov_eval_max_corr": float(erk.crosscov_max(
                                 B00["X"][B00["ev"]], Zev[B00["ev"]], standardize=True))})
            print(f"  {name:26s} [{form:11s}] k={Zk.shape[1]:3d}->{Zev.shape[1]:3d} "
                  f"rank={erk.proj_rank:3d} eval {a_ev:.4f} ({a_ev-base_auc:+.4f}) "
                  f"dist {frontier[-1]['distortion']:.3f}", flush=True)
    out["frontier"] = {"baseline_raw_head_auc_eval": base_auc, "points": frontier,
                       "mined_order": "|train AUC - .5| desc (ordering only, declared)",
                       "certificate_note": "categorical = primary (any-convex-loss form); "
                                           "continuous = OLS-form secondary pair readout"}

    out["runtime_sec"] = round(time.time() - t00, 1)
    (RES / "battery_leace.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {RES/'battery_leace.json'} in {out['runtime_sec']}s", flush=True)


if __name__ == "__main__":
    main()

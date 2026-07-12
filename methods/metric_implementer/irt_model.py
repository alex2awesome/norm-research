"""IRT-inspired models for the articulability data — the "model the equations" step (2026-06-14).

Three approaches of increasing structure (the equations from formalization §6/U4):

  (A) Rasch / 2PL on the binarized response matrix  y_{judge, cell} in {0,1}
        P(y=1) = sigma( a_cell * (theta_judge - z_cell) )
      theta_judge = the JUDGE-ABILITY / capability axis (what we want to recover); z = cell difficulty.
      "cell" = (metric_id, item_id). Response = did the judge produce a valid applicable score.

  (B) Beta-IRT on a CONTINUOUS response p in (0,1)  (fidelity / probability-of-label)
        mean = sigma( a*(theta - z) ),  p ~ Beta(mean*phi, (1-mean)*phi)
      The right model once continuous fidelity (U3) + per-tier re-measure exist.

  (C) kappa_r extension (the articulability factor we derived)
        P = sigma( a_cell * kappa_rubric * (theta_judge - z_cell) )
      kappa_rubric scales discrimination per RUBRIC: kappa->0 = rubric carries no resolving power.
      Identifiable only with a crossed judge x rubric design + an anchor (theta ~ N(0,1)); we
      standardize theta each step and report the d_cell <-> kappa identifiability caveat (the
      2PL scale indeterminacy, Noventa 2024).

Fit by Adam MLE in torch (CPU fine). Honest degeneracy handling: judges/cells with no response
variance are dropped and reported — the raw MEASURE matrix is mostly degenerate (only the strong
tier discriminates), so the *informative* capability-axis fit needs the per-tier FIDELITY from
re-measure. Run on whatever data exists; the module is the reusable instrument.
"""
from __future__ import annotations

import argparse
import glob
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------- data loading --------------------------------------------------

def load_measure(longtable_glob: str) -> pd.DataFrame:
    fs = glob.glob(longtable_glob)
    if not fs:
        raise SystemExit(f"no parquet at {longtable_glob}")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["cell"] = df["metric_id"].astype(str) + "@@" + df["item_id"].astype(str)
    return df


def binarize(df: pd.DataFrame, mode: str = "applicable") -> pd.DataFrame:
    """Response y in {0,1}. mode='applicable' -> judge produced a valid applicable score;
    mode='high' -> applicable AND score>=0.5."""
    if mode == "applicable":
        df = df.assign(y=df["applicable"].astype(float))
    else:
        df = df.assign(y=((df["applicable"]) & (df["score"] >= 0.5)).astype(float))
    # collapse passes: mean response per (judge, cell) -> majority
    g = df.groupby(["judge_model", "cell"], observed=True).y.mean().reset_index()
    g["y"] = (g["y"] >= 0.5).astype(float)
    return g


# ----------------------------- IRT fitters (torch) -------------------------------------------

def fit_irt(resp: pd.DataFrame, *, two_pl: bool = True, kappa_by: Optional[str] = None,
            steps: int = 1500, lr: float = 0.05, log=print) -> dict:
    """Fit P(y=1)=sigma(a_cell * kappa_r * (theta_judge - z_cell)) by Adam MLE.
    resp columns: judge_model, cell, y (0/1); optional kappa_by column (e.g. metric_id) for (C).
    Drops all-0/all-1 judges and cells (unidentified). Returns theta per judge + diagnostics."""
    import torch
    # drop degenerate judges/cells (no variance -> separable, theta/z unidentified)
    jvar = resp.groupby("judge_model").y.nunique()
    cvar = resp.groupby("cell").y.nunique()
    keep_j = set(jvar[jvar > 1].index); keep_c = set(cvar[cvar > 1].index)
    dropped_j = sorted(set(jvar.index) - keep_j)
    r = resp[resp.judge_model.isin(keep_j) & resp.cell.isin(keep_c)].copy()
    if r.empty or r.judge_model.nunique() < 2:
        return {"ok": False, "reason": "degenerate: <2 judges with response variance",
                "dropped_judges": dropped_j,
                "judge_applicable_rate": resp.groupby("judge_model").y.mean().round(3).to_dict()}
    judges = sorted(r.judge_model.unique()); cells = sorted(r.cell.unique())
    jix = {j: i for i, j in enumerate(judges)}; cix = {c: i for i, c in enumerate(cells)}
    ji = torch.tensor(r.judge_model.map(jix).values)
    ci = torch.tensor(r.cell.map(cix).values)
    y = torch.tensor(r.y.values, dtype=torch.float32)
    theta = torch.zeros(len(judges), requires_grad=True)
    z = torch.zeros(len(cells), requires_grad=True)
    a = torch.zeros(len(cells), requires_grad=True)        # log-discrimination (2PL); 0 -> a=1
    params = [theta, z] + ([a] if two_pl else [])
    kappa = None; kix = None
    if kappa_by and kappa_by in r.columns:
        ks = sorted(r[kappa_by].unique()); kix = {k: i for i, k in enumerate(ks)}
        kidx = torch.tensor(r[kappa_by].map(kix).values)
        kappa = torch.zeros(len(ks), requires_grad=True)   # log-kappa; 0 -> kappa=1
        params.append(kappa)
    opt = torch.optim.Adam(params, lr=lr)
    for s in range(steps):
        opt.zero_grad()
        disc = torch.exp(a[ci]) if two_pl else torch.ones_like(theta[ji])
        if kappa is not None:
            disc = disc * torch.exp(kappa[kidx])
        logit = disc * (theta[ji] - z[ci])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) \
            + 1e-3 * (theta.pow(2).mean() + z.pow(2).mean())
        loss.backward(); opt.step()
        with torch.no_grad():                              # anchor: theta ~ N(0,1)
            theta -= theta.mean(); theta /= (theta.std() + 1e-6)
    with torch.no_grad():
        th = {j: float(theta[jix[j]]) for j in judges}
        out = {"ok": True, "n_resp": len(r), "n_judges": len(judges), "n_cells": len(cells),
               "final_loss": float(loss), "theta": dict(sorted(th.items(), key=lambda kv: kv[1])),
               "dropped_judges_no_variance": dropped_j,
               "judge_applicable_rate": resp.groupby("judge_model").y.mean().round(3).to_dict()}
        if kappa is not None:
            out["kappa"] = {k: float(torch.exp(kappa[kix[k]])) for k in kix}
        return out


# ----------------------------- a non-degenerate first model on fidelity ----------------------

def model_fidelity(meas_parquet: str, log=print) -> dict:
    """OLS on the SEARCH scorecard fidelity (non-degenerate): fidelity ~ operator + log(cap) + task.
    A first 'does instruction-budget / operator move fidelity' model (Chinchilla-null flavored)
    while the per-tier IRT data is being collected."""
    df = pd.read_parquet(meas_parquet)
    df = df[df.fidelity.notna()].copy()
    df["logcap"] = np.log(df["cap"].fillna(120).clip(lower=1))
    # design: intercept + operator dummies (INIT ref) + logcap + task dummies
    ops = [o for o in sorted(df.operator.unique()) if o != "INIT"]
    tasks = sorted(df.task.unique())[1:]
    X = [np.ones(len(df))]
    names = ["intercept"]
    for o in ops:
        X.append((df.operator == o).astype(float).values); names.append(f"op[{o}]")
    X.append((df.logcap - df.logcap.mean()).values); names.append("logcap_c")
    for t in tasks:
        X.append((df.task == t).astype(float).values); names.append(f"task[{t}]")
    X = np.column_stack(X); yv = df.fidelity.values
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta; r2 = 1 - resid.var() / yv.var()
    return {"n": len(df), "r2": round(float(r2), 3),
            "coefs": {n: round(float(b), 4) for n, b in zip(names, beta)},
            "fidelity_by_operator": df.groupby("operator").fidelity.mean().round(3).to_dict()}


# ----------------------------- CLI -----------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--longtable", default="/lfs/skampere3/0/alexspan/norm-research/outputs/"
                    "metric_implementer_scale/bytier/*/longtable/*.parquet")
    ap.add_argument("--measurements", default="/lfs/skampere3/0/alexspan/norm-research/outputs/"
                    "metric_implementer_scale/measurements_search1.parquet")
    ap.add_argument("--bin-mode", default="applicable", choices=["applicable", "high"])
    args = ap.parse_args(argv)
    import json
    print("===== (A) Rasch/2PL IRT on MEASURE response matrix =====")
    df = load_measure(args.longtable)
    resp = binarize(df, mode=args.bin_mode)
    resp = resp.merge(df[["cell", "metric_id"]].drop_duplicates(), on="cell", how="left")
    fit = fit_irt(resp, two_pl=True, kappa_by="metric_id")
    print(json.dumps(fit, indent=1)[:2500])
    print("\n===== (regression) fidelity ~ operator + logcap + task (scorecards) =====")
    try:
        print(json.dumps(model_fidelity(args.measurements), indent=1))
    except Exception as e:
        print("fidelity model skipped:", type(e).__name__, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

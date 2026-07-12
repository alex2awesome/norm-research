"""OSL fits + pre-registered gates (spec: notes/2026-07-07__osl-executor-scaling-spec.md).

Loads outputs/osl/<executor>.json rows and produces the gated scaling readout:
  G1 instrument reliability (battery item split-half across executors; no-ceiling)
  G2 within-family monotonicity of the declared scalar in log-params
  G3 LOEO prediction of executor-level mean y_dis (the go/no-go for extrapolation talk)
  G4 planted-mech control (asymptote ~1; executor truth-AUC rising in z)
  G5 family-pooling license (permutation test of between-family residual variance)
Fits: saturating logistic y = L/(1+exp(-k(z-z0))) vs non-saturating linear, AICc + LOEO; profile CI
on L. If non-saturating is not rejected, L is reported as a LOWER BOUND, not a point.
Secondary latent (sensitivity arm, never headline): PCA (SVD) + PLS-1 (nested in LOEO) over the
base-measurement vectors; report Spearman agreement with the declared scalar.
"""
from __future__ import annotations

import argparse
import glob
import json
import math

import numpy as np


def _logistic(z, L, k, z0):
    return L / (1 + np.exp(-k * (z - z0)))


def fit_logistic(z, y, L_max=1.2):
    from scipy.optimize import curve_fit
    best = None
    for L0 in (0.5, 0.8, 1.0):
        for k0 in (0.5, 1.5, 4.0):
            try:
                p, _ = curve_fit(_logistic, z, y, p0=[L0, k0, float(np.median(z))],
                                 bounds=([1e-3, 1e-3, min(z) - 5], [L_max, 50, max(z) + 5]),
                                 maxfev=20000)
                rss = float(np.sum((_logistic(z, *p) - y) ** 2))
                if best is None or rss < best[1]:
                    best = (p, rss)
            except Exception:
                continue
    if best is None:
        return None
    p, rss = best
    return {"L": float(p[0]), "k": float(p[1]), "z0": float(p[2]), "rss": rss, "n_par": 3}


def fit_linear(z, y):
    A = np.vstack([np.ones_like(z), z]).T
    w, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    rss = float(np.sum((A @ w - y) ** 2))
    return {"a": float(w[0]), "b": float(w[1]), "rss": rss, "n_par": 2}


def _aicc(rss, n, k):
    if n <= k + 2 or rss <= 0:
        return float("inf")
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def profile_ci_L(z, y, fit, grid=120, L_max=1.2):
    """Profile-likelihood CI on the asymptote L (RSS-ratio with the small-n F threshold;
    asymptote CIs are skewed, so the delta method lies)."""
    from scipy.optimize import curve_fit
    from scipy.stats import f as fdist
    n = len(z)
    rss0 = fit["rss"]
    thr = rss0 * (1 + fdist.ppf(0.95, 1, max(1, n - 3)) / max(1, n - 3))
    lo, hi = fit["L"], fit["L"]
    for L in np.linspace(max(np.max(y) * 0.7, 0.05), L_max, grid):
        try:
            f = lambda zz, k, z0: _logistic(zz, L, k, z0)
            p, _ = curve_fit(f, z, y, p0=[fit["k"], fit["z0"]],
                             bounds=([1e-3, min(z) - 5], [50, max(z) + 5]), maxfev=10000)
            rss = float(np.sum((f(z, *p) - y) ** 2))
            if rss <= thr:
                lo, hi = min(lo, float(L)), max(hi, float(L))
        except Exception:
            continue
    return [lo, hi]


def loeo(z, y, kind="logistic"):
    """Leave-one-executor-out R^2 vs predict-the-mean baseline."""
    preds = np.full(len(z), np.nan)
    for i in range(len(z)):
        tr = np.ones(len(z), bool)
        tr[i] = False
        if kind == "logistic":
            f = fit_logistic(z[tr], y[tr])
            preds[i] = _logistic(z[i], f["L"], f["k"], f["z0"]) if f else np.nan
        else:
            f = fit_linear(z[tr], y[tr])
            preds[i] = f["a"] + f["b"] * z[i]
    ok = np.isfinite(preds)
    sse = float(np.sum((preds[ok] - y[ok]) ** 2))
    sse0 = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return {"r2_loeo": 1 - sse / sse0 if sse0 > 0 else float("nan"),
            "preds": [None if not np.isfinite(v) else float(v) for v in preds]}


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(np.asarray(a)[m], np.asarray(b)[m]).correlation) if m.sum() > 2 else float("nan")


def planted_capability_reference(curves, planted_names, *, top_k=3) -> dict:
    """Empirical planted-control reference at the highest observed capability levels.

    Selection is by ``z`` and must be independent of the outcome ``y``. Selecting the top observed
    y values estimates an order statistic, inflates the reference, and changes whenever a noisy point
    lands. This is an operational reference, not a theorem-level ceiling for arbitrary bank metrics.
    """
    per_control = {}
    members = {}
    for name in planted_names:
        curve = curves.get(name)
        if not curve:
            continue
        z = np.asarray(curve.get("z", []), dtype=float)
        y = np.asarray(curve.get("y", []), dtype=float)
        if z.shape != y.shape:
            continue
        keep = np.isfinite(z) & np.isfinite(y)
        if not keep.any():
            continue
        idx_all = np.flatnonzero(keep)
        idx = idx_all[np.argsort(z[keep])[-min(int(top_k), keep.sum()):]]
        per_control[name] = float(np.mean(y[idx]))
        ex = curve.get("execs") or [str(i) for i in range(len(z))]
        members[name] = [str(ex[i]) for i in idx]
    value = float(np.mean(list(per_control.values()))) if per_control else float("nan")
    return {"value": value, "per_control": per_control, "members": members,
            "selection": f"top-{int(top_k)} by capability z", "is_proved_ceiling": False}


def family_permutation(resid, fams, n_perm=2000, seed=0):
    """G5: between-family variance of residuals vs permuted family labels."""
    fams = np.asarray(fams)
    rng = np.random.default_rng(seed)
    def stat(r, f):
        return float(np.var([r[f == u].mean() for u in np.unique(f)]))
    obs = stat(resid, fams)
    null = [stat(resid, rng.permutation(fams)) for _ in range(n_perm)]
    return {"obs": obs, "p": float((np.sum(np.asarray(null) >= obs) + 1) / (n_perm + 1))}


# -- consensus-agreement y-axis (user decision 2026-07-07) ------------------------------------
def _zscore(v):
    m = np.nanmean(v)
    s = np.nanstd(v)
    return (v - m) / s if s > 1e-9 else None


def consensus_agreement(mbars, top_k=4, battery_z=None):
    """mbars: {executor: {family, m_bar (M,N), per_form (M,4,N), names, kinds}} — aligned metrics.

    y[E,m] = Spearman(m̄_ω(E,m), family-balanced leave-E-out consensus): z-score each executor's
    vector, average WITHIN each family (excluding E), then across families — so a majority family
    cannot dominate the consensus (family-correlated reading errors would otherwise bias it).
    Also: inter-frontier floor per metric = mean pairwise Spearman among the top_k executors by
    battery z (the criterion's intrinsic underdetermination shows up as 1 - that agreement)."""
    from scipy.stats import spearmanr
    execs = sorted(mbars)
    names = list(mbars[execs[0]]["names"])
    M = len(names)
    fams = {e: mbars[e]["family"] for e in execs}
    Z = {e: np.array([_zscore(mbars[e]["m_bar"][i]) if _zscore(mbars[e]["m_bar"][i]) is not None
                      else np.full(mbars[e]["m_bar"].shape[1], np.nan) for i in range(M)])
         for e in execs}
    agree = {e: [] for e in execs}
    for i in range(M):
        for e in execs:
            others = [o for o in execs if o != e and np.isfinite(Z[o][i]).any()]
            if not others or not np.isfinite(Z[e][i]).any():
                agree[e].append(np.nan)
                continue
            by_fam = {}
            for o in others:
                by_fam.setdefault(fams[o], []).append(Z[o][i])
            cons = np.nanmean([np.nanmean(v, axis=0) for v in by_fam.values()], axis=0)
            m = np.isfinite(Z[e][i]) & np.isfinite(cons)
            r = spearmanr(Z[e][i][m], cons[m]).correlation if m.sum() > 20 else np.nan
            agree[e].append(float(r) if np.isfinite(r) else np.nan)
    floor = []
    top = (sorted(execs, key=lambda e: -battery_z.get(e, -9))[:top_k]
           if battery_z else execs[:top_k])
    for i in range(M):
        rs = []
        for a_i in range(len(top)):
            for b_i in range(a_i + 1, len(top)):
                va, vb = Z[top[a_i]][i], Z[top[b_i]][i]
                m = np.isfinite(va) & np.isfinite(vb)
                if m.sum() > 20:
                    rs.append(spearmanr(va[m], vb[m]).correlation)
        floor.append(float(np.nanmean(rs)) if rs else np.nan)
    return {"agree": agree, "frontier_agreement": floor, "frontier_members": top,
            "names": names}


# -- secondary latent (sensitivity arm) --------------------------------------------------------
def pca1(M):
    """First principal component scores of the standardized base-measurement matrix (SVD)."""
    Ms = (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-9)
    Ms = np.nan_to_num(Ms)
    U, s, Vt = np.linalg.svd(Ms, full_matrices=False)
    pc = Ms @ Vt[0]
    return pc, float(s[0] ** 2 / np.sum(s ** 2))


def pls1(M, y):
    """One-component PLS (NIPALS) — supervised; only used nested inside LOEO."""
    Ms = (M - M.mean(0)) / (M.std(0) + 1e-9)
    yc = y - y.mean()
    w = Ms.T @ yc
    w /= (np.linalg.norm(w) + 1e-12)
    return Ms @ w, w


def pls_loeo(M, y):
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        tr = np.ones(len(y), bool)
        tr[i] = False
        t_tr, w = pls1(M[tr], y[tr])
        A = np.vstack([np.ones(tr.sum()), t_tr]).T
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        mi, si = M[tr].mean(0), M[tr].std(0) + 1e-9
        t_i = ((M[i] - mi) / si) @ w
        preds[i] = coef[0] + coef[1] * t_i
    sse = float(np.sum((preds - y) ** 2))
    sse0 = float(np.sum((y - y.mean()) ** 2))
    return 1 - sse / sse0 if sse0 > 0 else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="outputs/osl (executor JSONs)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-executors", type=int, default=4)
    p.add_argument("--mbar-dir", help="dir of mbar_<exec>.npz -> consensus-agreement y-axis")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="executors excluded from fits AND consensus (readout-broken)")
    a = p.parse_args(argv)
    runs = []
    for f in sorted(glob.glob(f"{a.dir}/*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if (isinstance(r, dict) and r.get("executor") and r.get("mean_y_dis_bank") is not None
                and r["executor"] not in a.exclude):
            runs.append(r)
    if len(runs) < a.min_executors:
        print(f"[fit] only {len(runs)} executor runs; need {a.min_executors} — skipping")
        return
    z = np.array([r["battery"]["z"] for r in runs])
    y = np.array([r["mean_y_dis_bank"] for r in runs])
    fams = [r["family"] for r in runs]
    names = [r["executor"] for r in runs]
    rep = {"executors": names, "z": z.tolist(), "y": y.tolist(), "families": fams}

    # G1 instrument reliability + ceiling
    ea = np.array([r["battery"]["auc_even_items"] for r in runs])
    oa = np.array([r["battery"]["auc_odd_items"] for r in runs])
    r_half = float(np.corrcoef(ea, oa)[0, 1]) if len(runs) > 2 else float("nan")
    rep["G1"] = {"split_half_r": r_half,
                 "split_half_sb": float(2 * r_half / (1 + r_half)) if np.isfinite(r_half) else None,
                 "max_auc": float(max(r["battery"]["auc"] for r in runs)),
                 "ceiling_ok": bool(max(r["battery"]["auc"] for r in runs) <= 0.98)}

    # G2 within-family monotonicity of z in log-params
    rep["G2"] = {}
    for fam in sorted(set(fams)):
        idx = [i for i, f in enumerate(fams) if f == fam]
        if len(idx) >= 3:
            lp = np.log([runs[i]["params"] for i in idx])
            rep["G2"][fam] = {"spearman": spearman(lp, z[idx]),
                              "monotone": bool(spearman(lp, z[idx]) > 0.99)}

    # fits + G3
    lg = fit_logistic(z, y)
    ln = fit_linear(z, y)
    n = len(z)
    rep["fit"] = {"logistic": lg, "linear": ln,
                  "aicc_logistic": _aicc(lg["rss"], n, 3) if lg else None,
                  "aicc_linear": _aicc(ln["rss"], n, 2),
                  "saturating_preferred": bool(lg and _aicc(lg["rss"], n, 3) <
                                               _aicc(ln["rss"], n, 2))}
    if lg:
        rep["fit"]["L_profile_ci"] = profile_ci_L(z, y, lg)
        rep["fit"]["asymptote_is_lower_bound_only"] = not rep["fit"]["saturating_preferred"]
    rep["G3"] = {"logistic": loeo(z, y, "logistic")["r2_loeo"],
                 "linear": loeo(z, y, "linear")["r2_loeo"]}

    # G4 planted-mech control
    pl_y, pl_auc = [], []
    for r in runs:
        rows = [m for m in r["metrics"] if m["kind"] == "planted" and not m["excluded"]
                and m.get("y_dis") is not None]
        if rows:
            pl_y.append(float(np.mean([m["y_dis"] for m in rows])))
            pl_auc.append(float(np.nanmean([m.get("truth_auc", np.nan) for m in rows])))
        else:
            pl_y.append(np.nan)
            pl_auc.append(np.nan)
    pm = np.isfinite(pl_y)
    lg_p = fit_logistic(z[pm], np.asarray(pl_y)[pm]) if pm.sum() >= 4 else None
    rep["G4"] = {"planted_mean_y": [None if not np.isfinite(v) else v for v in pl_y],
                 "planted_L": lg_p["L"] if lg_p else None,
                 "truth_auc_vs_z_spearman": spearman(z, np.asarray(pl_auc))}

    # G5 family pooling license
    if lg:
        resid = y - _logistic(z, lg["L"], lg["k"], lg["z0"])
        rep["G5"] = family_permutation(resid, fams)

    # secondary latent (sensitivity)
    keys = sorted(set().union(*[set(r["base_measurements"]) for r in runs]))
    M = np.array([[r["base_measurements"].get(k) if r["base_measurements"].get(k) is not None
                   else np.nan for k in keys] for r in runs], float)
    pc, evr = pca1(M)
    if spearman(pc, z) < 0:
        pc = -pc
    rep["secondary_latent"] = {
        "base_keys": keys, "pca_evr1": evr,
        "spearman_pc1_vs_declared": spearman(pc, z),
        "loeo_r2_on_pc1": loeo(np.asarray(pc), y, "logistic")["r2_loeo"] if len(runs) >= 5 else None,
        "pls_nested_loeo_r2": pls_loeo(np.nan_to_num(M), y) if len(runs) >= 5 else None}

    # -- consensus-agreement y-axis (primary law once mbar vectors exist) ----------------------
    if a.mbar_dir:
        mbars = {}
        for f in sorted(glob.glob(f"{a.mbar_dir}/mbar_*.npz")):
            z_ = np.load(f, allow_pickle=True)
            e = str(z_["executor"])
            if e in a.exclude:
                continue
            mbars[e] = {"family": str(z_["family"]), "names": [str(x) for x in z_["names"]],
                        "kinds": [str(x) for x in z_["kinds"]], "m_bar": z_["m_bar"],
                        "per_form": z_["per_form"]}
        if len(mbars) >= a.min_executors:
            bz = {r["executor"]: r["battery"]["z"] for r in runs}
            ca = consensus_agreement(mbars, top_k=4, battery_z=bz)
            kinds = mbars[next(iter(mbars))]["kinds"]
            bank_i = [i for i, k in enumerate(kinds) if k == "bank"]
            pl_i = [i for i, k in enumerate(kinds) if k == "planted"]
            common = [e for e in ca["agree"] if e in bz]
            zc = np.array([bz[e] for e in common])
            yc = np.array([np.nanmean([ca["agree"][e][i] for i in bank_i]) for e in common])
            fams_c = [mbars[e]["family"] for e in common]
            lg_c = fit_logistic(zc, yc)
            rep["consensus"] = {
                "executors": common, "z": zc.tolist(), "y_bank_mean": yc.tolist(),
                "fit_logistic": lg_c,
                "L_profile_ci": profile_ci_L(zc, yc, lg_c) if lg_c else None,
                "loeo_r2": loeo(zc, yc, "logistic")["r2_loeo"],
                "spearman_z_y": spearman(zc, yc),
                "frontier_members": ca["frontier_members"],
                "frontier_agreement_bank": {ca["names"][i]: ca["frontier_agreement"][i]
                                            for i in bank_i},
                "frontier_agreement_planted": {ca["names"][i]: ca["frontier_agreement"][i]
                                               for i in pl_i},
            }
            if lg_c:
                resid_c = yc - _logistic(zc, lg_c["L"], lg_c["k"], lg_c["z0"])
                rep["consensus"]["family_perm"] = family_permutation(resid_c, fams_c)
            # planted arbitration: does consensus-agreement track truth where truth exists?
            arb = []
            truth = {r["executor"]: {m["name"]: m.get("truth_auc") for m in r["metrics"]
                                     if m["kind"] == "planted"} for r in runs}
            for i in pl_i:
                nm = ca["names"][i]
                ag = np.array([ca["agree"][e][i] for e in common])
                ta = np.array([truth.get(e, {}).get(nm, np.nan) for e in common], float)
                arb.append(spearman(ag, ta))
            rep["consensus"]["planted_arbitration_spearman"] = arb
            per_metric = {ca["names"][i]: [None if not np.isfinite(ca["agree"][e][i])
                                           else round(float(ca["agree"][e][i]), 4)
                                           for e in common] for i in bank_i + pl_i}
            rep["consensus"]["per_metric_agreement"] = per_metric

    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("G1", "G2", "G3", "G4", "G5", "fit")
                      if k in rep}, indent=1, default=float))
    if "consensus" in rep:
        c = rep["consensus"]
        print("[consensus] spearman(z,y)=%.3f loeo=%.3f L=%s ci=%s" % (
            c["spearman_z_y"], c["loeo_r2"],
            round(c["fit_logistic"]["L"], 3) if c["fit_logistic"] else None,
            c["L_profile_ci"]))
    print(f"[fit] n={n} -> {a.out}")


if __name__ == "__main__":
    main()

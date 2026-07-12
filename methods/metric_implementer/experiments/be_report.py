"""Final B_E report for the R3 creative-writing scaling run.

Per metric x executor: B_E (upper), D_obs (lower), coverage, Omega-order/probe variance,
reconstruction accuracy (how well the criteria signatures recover M_i — R^2 + best-single r),
alpha (Heaps), and signature-discrimination (unsupervised diagnostics). Then a cross-executor
scaling table (B_E vs model size) and the bge_pertask signal-matching accuracy (CE recall vs gold).

Inputs: --executors name:dir,name:dir,... (each dir is one executor's *_sigs.npz ckpts).
        --signals-dir  the bge_pertask creative_writing dir (matches + matches_ce + signals).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

from . import alpha_probe as ap


def _recovery(S, Mi, *, k_top=30, n_folds=5, seed=0):
    """Criteria→M_i recovery. ``best`` = max |per-criterion corr| on all data (descriptive only).
    ``multi`` = cross-validated R² of Ridge on the top-``k_top`` criteria — with the top-k selection
    done INSIDE each training fold (selecting on the full data before CV leaks the test fold into the
    feature choice and inflates R²; fixed 2026-07-01)."""
    m = np.isfinite(Mi)
    for j in range(S.shape[0]):
        m &= np.isfinite(S[j])
    if m.sum() < 10:
        return float("nan"), float("nan")
    M = Mi[m]
    X = S[:, m].T
    r = [np.corrcoef(X[:, j], M)[0, 1] if np.std(X[:, j]) > 0 else 0 for j in range(X.shape[1])]
    best = float(np.max(np.abs(r)))
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        r2s = []
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            rt = [np.corrcoef(X[tr, j], M[tr])[0, 1] if np.std(X[tr, j]) > 0 else 0
                  for j in range(X.shape[1])]                      # select on the TRAIN fold only
            top = np.argsort([-abs(x) for x in rt])[: min(k_top, X.shape[1])]
            model = Ridge(alpha=1.0).fit(X[tr][:, top], M[tr])
            pred = model.predict(X[te][:, top])
            ss_res = float(np.sum((M[te] - pred) ** 2))
            ss_tot = float(np.sum((M[te] - M[te].mean()) ** 2))
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
        multi = float(np.nanmean(r2s))
    except Exception:
        multi = float("nan")
    return best, multi


def _per_ckpt(f, cmi):
    z = np.load(f, allow_pickle=True)
    S = np.asarray(z["sigs"], float)
    tags = list(z["tags"])
    Mi = np.asarray(z["M_i"], float) if "M_i" in z.files else None
    name = str(z["name"]) if "name" in z.files else os.path.basename(f)
    gi = int(z["r2_idx"]) if "r2_idx" in z.files else -1
    fin = float(np.isfinite(S).mean())
    cr = ap.conditional_crc_report(S, tags, cmi_thresh=cmi)
    diag = ap.signature_diagnostics(S)
    best_r, rec_r2 = (float("nan"), float("nan"))
    if Mi is not None and np.isfinite(Mi).any():
        best_r, rec_r2 = _recovery(S, Mi)
    try:
        rep = ap.alpha_probe(S, tags, tau=float(z["tau"]) if "tau" in z.files else 0.05,
                             delta=0.05, compute_cmi=False)
        alpha = float(rep["alpha_terminal"])
    except Exception:
        alpha = float("nan")
    return {"gi": gi, "name": name, "fin": fin, "B_E": cr["B_E_upper"],
            "B_E_std": cr["B_E_upper_std"], "D_obs": cr["D_obs_lower"], "coverage": cr["coverage"],
            "f1_over_N": cr.get("f1_over_N", float("nan")),        # Lemma 12.6.0 gate: ≈1 ⇒ α pinned
            "agree": cr["two_method_agree"], "os_D_std": cr["order_stability"]["D_std"],
            "rec_best": best_r, "rec_r2": rec_r2, "alpha": alpha,
            "disc_l1": diag["mean_between_sig_l1"], "frac_const": diag["frac_near_constant_sig"],
            "n_crit": S.shape[0], "n_probes": S.shape[1]}


def _signals_accuracy(sdir):
    """CE top-10 recall vs gold aspects; signal/aspect coverage."""
    out = {}
    try:
        gold_path = os.path.join(sdir, "matches_creative_writing.json")
        ce_path = os.path.join(sdir, "matches_ce_creative_writing.jsonl")
        sig_path = os.path.join(sdir, "signals_creative_writing.jsonl")
        gold = {e["id"]: e.get("aspects", []) for e in json.load(open(gold_path))} if os.path.exists(gold_path) else {}
        ce = {}
        if os.path.exists(ce_path):
            for line in open(ce_path):
                o = json.loads(line)
                ce[int(o["signal_id"])] = o.get("top10", [])
        sigs = {}
        if os.path.exists(sig_path):
            for line in open(sig_path):
                o = json.loads(line)
                sigs[o["i"]] = o.get("s", "")
        # recall@10 / @1 of gold aspects against CE ranking
        hits1 = hits10 = n = 0
        for sid, g in gold.items():
            if not g or sid not in ce:
                continue
            n += 1
            top = ce[sid]
            if any(a in top[:1] for a in g):
                hits1 += 1
            if any(a in top for a in g):
                hits10 += 1
        all_asp = Counter()
        for g in gold.values():
            all_asp.update(g)
        out = {"n_signals": len(sigs), "n_matched": len(gold), "n_with_ce": len(ce),
               "CE_top1_hit_gold": round(hits1 / max(n, 1), 3),
               "CE_top10_recall_gold": round(hits10 / max(n, 1), 3),
               "n_unique_aspects": len(all_asp),
               "aspects_per_signal_mean": round(np.mean([len(g) for g in gold.values()]) if gold else 0, 2),
               "signals_per_aspect_median": float(np.median(list(all_asp.values()))) if all_asp else 0}
    except Exception as e:
        out = {"error": str(e)[:200]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="be_report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--executors", required=True, help="name:dir,name:dir,...")
    p.add_argument("--signals-dir", default="/lfs/skampere3/0/alexspan/data/bge_pertask/creative_writing")
    p.add_argument("--cmi-thresh", type=float, default=0.15)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    execs = [s.split(":", 1) for s in a.executors.split(",") if ":" in s]

    per = defaultdict(dict)  # gi -> executor -> metrics
    print("=" * 100)
    print("B_E SCALING REPORT — R3 creative-writing (cmi=%.2f)" % a.cmi_thresh)
    print("=" * 100)
    for ex, d in execs:
        print(f"\n### executor = {ex}  ({d})")
        print("%-34s %5s %7s %6s %5s %6s %6s %6s %6s %6s" % (
            "metric", "ncrit", "B_E", "+/-", "Dobs", "cov", "recR2", "alpha", "f1/N", "discL1"))
        for f in sorted(glob.glob(os.path.join(d, "*_sigs.npz"))):
            r = _per_ckpt(f, a.cmi_thresh)
            per[r["gi"]][ex] = r
            print("%-34s %5d %7.1f %6.1f %5.0f %6.2f %6.2f %6.2f %6.2f %6.3f" % (
                r["name"][:34], r["n_crit"], r["B_E"], r["B_E_std"], r["D_obs"],
                r["coverage"], r["rec_r2"], r["alpha"], r["f1_over_N"], r["disc_l1"]))

    # scaling table: B_E + recovery across executors per metric
    print("\n" + "=" * 100)
    print("SCALING across executors (B_E / recovery R2 per metric)")
    print("=" * 100)
    hdr = "%-30s" % "metric" + "".join("%14s" % e for e, _ in execs)
    print(hdr)
    agg = {ex: {"B_E": [], "rec": []} for ex, _ in execs}
    for gi in sorted(per):
        if len(per[gi]) < 2:
            continue
        nm = next(iter(per[gi].values()))["name"][:30]
        row = "%-30s" % nm
        for ex, _ in execs:
            r = per[gi].get(ex)
            if r:
                row += "%14s" % ("%.1f/%.2f" % (r["B_E"], r["rec_r2"]))
                agg[ex]["B_E"].append(r["B_E"])
                agg[ex]["rec"].append(r["rec_r2"])
            else:
                row += "%14s" % "-"
        print(row)
    print("%-30s" % "MEAN B_E / recovery:" + "".join(
        "%14s" % ("%.1f/%.2f" % (np.mean(agg[e]["B_E"]) if agg[e]["B_E"] else 0,
                                  np.nanmean(agg[e]["rec"]) if agg[e]["rec"] else 0))
        for e, _ in execs))

    # signals accuracy
    print("\n" + "=" * 100)
    print("bge_pertask creative_writing signal-matching accuracy")
    print("=" * 100)
    sa = _signals_accuracy(a.signals_dir)
    for k, v in sa.items():
        print(f"  {k}: {v}")

    if a.json:
        json.dump({"per_metric": {str(k): v for k, v in per.items()},
                   "mean_per_executor": {e: {"B_E_mean": float(np.mean(agg[e]["B_E"])) if agg[e]["B_E"] else None,
                                             "recovery_mean": float(np.nanmean(agg[e]["rec"])) if agg[e]["rec"] else None}
                                         for e, _ in execs},
                   "signals_accuracy": sa},
                  open(a.json, "w"), indent=1, default=float)
        print(f"\nreport json -> {a.json}")


if __name__ == "__main__":
    main()

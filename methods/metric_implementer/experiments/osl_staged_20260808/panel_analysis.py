"""Generic per-task panel analysis for the v2 multi-task fleet (task #136).
Per task, from mbar2_<task>_<exec>.npz panels (+ humor's mbar285_* base):
  (1) per-metric articulability curves: y = LOFO family-balanced consensus agreement,
      disattenuated by form-orbit split-half rel (per_metric_laws recipe), x = battery z
      (executor-level instrument, task-independent); split-half (even/odd probe) SE per point
      -> curves_<task>.json for the notebook
  (2) knee census: logistic-vs-linear AICc + profile CI on L, classified relative to the
      task's planted-control capability reference (mean y at the top-3 capability-z executors).
      This reference is not a theorem-level ceiling for arbitrary bank metrics.
      -> laws_<task>.json
  (3) external validity: item-level silver AUC per metric per executor (labels = manifest
      judgement, same subsample as probes -> exact join), panel CV-ridge AUC per executor
      (dose-response), MI arm vs cert OPT_Omega bits where a cert exists
      -> silver_<task>.json
Incremental: uses whatever executors have landed; curve verdicts only at n>=8 executors.
Usage: panel_analysis.py <task_dir> [--min-execs N]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr, rankdata

B = "/lfs/skampere3/0/alexspan"
sys.path.insert(0, f"{B}/norm-research")
from methods.metric_implementer.experiments import osl_fit as of

O = f"{B}/outputs/osl_multi"
CFG_TASK = {"creative_writing": "creative-writing", "press_releases": "press-releases",
            "math": "math", "news_homepages": "news-homepages", "peer_review": "peer-review",
            "notice_and_comment": "notice-and-comment", "patents": "patents", "humor": "humor",
            "code_review": "code-review"}
FAM = {"llama1b": "llama", "llama3b": "llama", "llama8b": "llama", "llama70b": "llama",
       "llama405b": "llama",
       "qwen25-3b": "qwen", "qwen25-7b": "qwen", "qwen25-14b": "qwen", "qwen25-32b": "qwen",
       "qwen25-72b": "qwen", "gemma2-9b": "gemma2", "gemma2-27b": "gemma2",
       "mistral7b": "mistral", "mistral-24b": "mistral", "phi4": "phi",
       "qwen35-122b": "qwen35"}

task = sys.argv[1]
min_execs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

# ---- load panels (humor = mbar285 base + humor_sup supplement, unioned on metrics) ----------
def load_panel(path):
    z = np.load(path, allow_pickle=True)
    return dict(names=[str(x) for x in z["names"]], kinds=[str(x) for x in z["kinds"]],
                m=z["m_bar"], pf=z["per_form"])


panels = {}   # exec -> {name -> (m_bar vec, per_form)}
def add(path, ex):
    p = load_panel(path)
    d = panels.setdefault(ex, {})
    for i, n in enumerate(p["names"]):
        d[n] = (p["m"][i], p["pf"][i], p["kinds"][i])


if task == "humor":
    for f in sorted(glob.glob(f"{B}/outputs/osl/mbar285_*.npz")) + \
             sorted(glob.glob(f"{B}/outputs/osl/mbar285c_*.npz")):
        add(f, re.sub(r"^mbar285c?_|\.npz$", "", os.path.basename(f)))
    for f in sorted(glob.glob(f"{O}/mbar2_humor_sup_*.npz")):
        add(f, re.sub(r"^mbar2_humor_sup_|\.npz$", "", os.path.basename(f)))
else:
    for f in sorted(glob.glob(f"{O}/mbar2_{task}_*.npz")):
        add(f, re.sub(rf"^mbar2_{task}_|\.npz$", "", os.path.basename(f)))
panels = {e: d for e, d in panels.items() if e in FAM}
if not panels:
    sys.exit(f"no panels for {task}")
execs = sorted(panels)
all_names = sorted({n for d in panels.values() for n in d})
kinds = {}
for d in panels.values():
    for n, (_, _, k) in d.items():
        kinds[n] = k
bank = [n for n in all_names if kinds[n] == "bank"]
planted = [n for n in all_names if kinds[n] != "bank"]
n_pr = min(v[0].shape[-1] for d in panels.values() for v in d.values())
print(f"[{task}] executors={execs}")
print(f"[{task}] metrics: bank={len(bank)} planted={len(planted)} probes={n_pr}")

zmap = {}
for e in execs:
    p = f"{B}/outputs/osl/{e}.json"
    if os.path.exists(p):
        zmap[e] = json.load(open(p))["battery"]["z"]

# ---- labels (exact join by construction) ----------------------------------------------------
from methods.metric_implementer.manifest import full_manifest, load_corpus_labels
y_lab = None
try:
    entry = next(e for e in full_manifest().datasets if e.task == CFG_TASK[task])
    texts_all, labels, _ = load_corpus_labels(entry, 60 + n_pr, seed=7)
    y_lab = np.array(labels[60:60 + n_pr], float)
    wc_r = rankdata([len(t.split()) for t in texts_all[60:60 + n_pr]]).astype(float)
    print(f"[{task}] label base rate={y_lab.mean():.3f} n={len(y_lab)}")
except (StopIteration, ValueError, FileNotFoundError) as e:
    print(f"[{task}] no silver labels ({e}) — curves/laws only")


def len_resid(v):
    """rank-residualize a score vector on text-length ranks (the CW upvote-length confound)."""
    ok = np.isfinite(v)
    out = np.full_like(v, np.nan, dtype=float)
    if ok.sum() < 10:
        return out
    r = rankdata(v[ok]).astype(float)
    X = np.column_stack([wc_r[ok], np.ones(ok.sum())])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    out[ok] = r - X @ beta
    return out


def zsv(v):
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 1e-9 else np.full_like(v, np.nan)


def relf(pf):
    pf = np.atleast_2d(pf)
    if pf.shape[0] < 2:
        return np.nan
    a, b = np.nanmean(pf[0::2], 0), np.nanmean(pf[1::2], 0)
    m_ = np.isfinite(a) & np.isfinite(b)
    if m_.sum() < 20 or a[m_].std() < 1e-9 or b[m_].std() < 1e-9:
        return np.nan
    r = np.corrcoef(a[m_], b[m_])[0, 1]
    return 2 * r / (1 + r) if r > -0.5 else np.nan


ZS = {e: {n: zsv(panels[e][n][0][:n_pr]) for n in panels[e]} for e in execs}


def consensus_y(e, n, sl):
    by = {}
    for o in execs:
        if o == e or FAM[o] == FAM[e] or n not in ZS[o]:
            continue
        v = ZS[o][n][sl]
        if np.isfinite(v).any():
            by.setdefault(FAM[o], []).append(v)
    if len(by) < 2:
        return np.nan
    cons = np.nanmean([np.nanmean(v, 0) for v in by.values()], 0)
    a = ZS[e][n][sl] if n in ZS[e] else None
    if a is None:
        return np.nan
    m_ = np.isfinite(a) & np.isfinite(cons)
    if m_.sum() < 20:
        return np.nan
    return spearmanr(a[m_], cons[m_]).correlation


sl_all = np.arange(n_pr)
sl_e, sl_o = sl_all[::2], sl_all[1::2]
curves = {}
Yd = {}                                 # (exec, name) -> disattenuated y
for n in all_names:
    zs_, ys_, ses_, exs_ = [], [], [], []
    for e in execs:
        if e not in zmap or n not in panels[e]:
            continue
        r = consensus_y(e, n, sl_all)
        rl = relf(panels[e][n][1][..., :n_pr])
        if not (np.isfinite(r) and np.isfinite(rl) and rl > 0.2):
            continue
        yd = float(np.clip(r / np.sqrt(rl), -1, 1.15))
        re_, ro_ = consensus_y(e, n, sl_e), consensus_y(e, n, sl_o)
        se = float(abs(re_ - ro_) / 2) if np.isfinite(re_) and np.isfinite(ro_) else None
        Yd[(e, n)] = yd
        zs_.append(zmap[e]); ys_.append(yd); ses_.append(se); exs_.append(e)
    if len(ys_) >= 4:
        o = np.argsort(zs_)
        curves[n] = dict(kind=kinds[n], z=[zs_[i] for i in o], y=[ys_[i] for i in o],
                         se=[ses_[i] for i in o], execs=[exs_[i] for i in o])

# Empirical task reference from planted controls. Selection is by capability z, never by outcome y.
planted_ref = of.planted_capability_reference(curves, planted, top_k=3)
ceil = planted_ref["value"]  # Backward-compatible variable/key used by existing notebook loaders.
pl_top = planted_ref["per_control"]
print(f"[{task}] planted capability reference ({planted_ref['selection']} over "
      f"{len(pl_top)} controls) = {ceil:.3f}")

rows = []
if len(execs) >= min_execs:
    for n in bank:
        c = curves.get(n)
        if not c or len(c["y"]) < min_execs:
            rows.append({"name": n, "verdict": "NOISY", "n": 0 if not c else len(c["y"])})
            continue
        z_, y_ = np.array(c["z"]), np.array(c["y"])
        slope = of.spearman(z_, y_)
        lg = of.fit_logistic(z_, y_)
        ln = of.fit_linear(z_, y_)
        a_lg = of._aicc(lg["rss"], len(y_), 3) if lg else np.inf
        a_ln = of._aicc(ln["rss"], len(y_), 2)
        kneed = bool(lg and a_lg < a_ln)
        row = {"name": n, "n": len(y_), "slope": round(float(slope), 3), "kneed": kneed,
               "top_y": round(float(np.mean(y_[np.argsort(z_)[-3:]])), 3)}
        if lg:
            ci = of.profile_ci_L(z_, y_, lg)
            row.update({"L": round(lg["L"], 3), "L_lo": round(ci[0], 3),
                        "L_hi": round(ci[1], 3), "z0": round(lg["z0"], 2)})
        if kneed and lg and np.isfinite(ceil) and ci[1] < ceil:
            row["verdict"] = "BOUNDED"
        elif kneed and lg:
            row["verdict"] = "REACHES"
        elif slope >= 0.35:
            row["verdict"] = "RISING"
        else:
            row["verdict"] = "NOISY"
        rows.append(row)
        curves[n]["verdict"] = row["verdict"]
        for k in ("L", "L_lo", "L_hi"):
            if k in row:
                curves[n][k] = row[k]
    from collections import Counter
    print(f"[{task}] verdicts: {dict(Counter(r['verdict'] for r in rows))}")

# ---- (3) external validity ------------------------------------------------------------------
if y_lab is None:
    json.dump(curves, open(f"{O}/curves_{task}.json", "w"), default=float)
    if rows:
        json.dump({"ceiling": None if not np.isfinite(ceil) else ceil,
                   "ceiling_is_proved": False, "planted_reference": planted_ref,
                   "executors": execs, "rows": rows},
                  open(f"{O}/laws_{task}.json", "w"), indent=1, default=float)
    print(f"[{task}] -> curves/laws saved (no silver arm)")
    sys.exit(0)


def auc(s, l):
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    if l.sum() < 3 or l.sum() > len(l) - 3 or len(np.unique(s)) < 2:
        return np.nan
    r = rankdata(s)
    return float((r[l > 0].mean() - (l.sum() + 1) / 2) / (len(l) - l.sum()))


def panel_cv_auc(X, l, k=5, lam=10.0):
    Xr = np.column_stack([rankdata(c) for c in X.T])
    Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9)
    preds = np.zeros(len(l))
    for f in range(k):
        te = np.arange(len(l)) % k == f
        A, b = Xr[~te], l[~te] - l[~te].mean()
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (l[~te] - l[~te].mean()))
        preds[te] = Xr[te] @ w
    return auc(preds, l)


silver = {"per_exec": {}, "task": task,
          "len_auc": float(a_) if np.isfinite(a_ := auc(wc_r, y_lab)) else None}
print(f"[{task}] length-alone AUC = {silver['len_auc']}")
pm_auc, pm_auc_lc = {}, {}
for e in execs:
    aucs, aucs_lc = {}, {}
    rowsM = []
    for n in bank:
        if n not in panels[e]:
            continue
        v = panels[e][n][0][:n_pr]
        a = auc(v, y_lab)
        if np.isfinite(a):
            aucs[n] = a
            rowsM.append(v)
            al = auc(len_resid(v), y_lab)
            if np.isfinite(al):
                aucs_lc[n] = al
    pm_auc[e], pm_auc_lc[e] = aucs, aucs_lc
    pan = pan_lc = np.nan
    if len(rowsM) > 10:
        Mx = np.vstack(rowsM)
        ok = np.isfinite(Mx).all(0)
        if ok.sum() > 40:
            pan = panel_cv_auc(Mx[:, ok].T, y_lab[ok])
            Mr = np.vstack([len_resid(r) for r in rowsM])
            okr = np.isfinite(Mr).all(0)
            if okr.sum() > 40:
                pan_lc = panel_cv_auc(Mr[:, okr].T, y_lab[okr])
    silver["per_exec"][e] = dict(z=zmap.get(e), n_metrics=len(aucs),
                                 mean_abs=float(np.nanmean([abs(v - .5) for v in aucs.values()]))
                                 if aucs else None,
                                 panel_auc=None if not np.isfinite(pan) else float(pan),
                                 panel_auc_lenctl=None if not np.isfinite(pan_lc) else float(pan_lc))
    if silver["per_exec"][e]["panel_auc"]:
        print(f"[{task}] {e:12s} panel_AUC={pan:.3f} lenctl={pan_lc:.3f} mean|AUC-.5|="
              f"{silver['per_exec'][e]['mean_abs']:.4f} (n={len(aucs)})")

# dose-response across executors (raw + length-controlled)
for key, out in (("panel_auc", "dose_response_rho"), ("panel_auc_lenctl", "dose_response_rho_lenctl")):
    pe = [(v["z"], v[key]) for v in silver["per_exec"].values()
          if v["z"] is not None and v.get(key)]
    if len(pe) >= 5:
        zz, pp = map(np.array, zip(*pe))
        silver[out] = float(spearmanr(zz, pp).correlation)
        print(f"[{task}] dose-response spearman(z, {key}) = {silver[out]:+.3f} (n={len(pe)})")

# MI arm vs cert
cp = f"{B}/outputs/silver_r2/{task}/cert_{task}.json"
if os.path.exists(cp) and execs:
    cert = {r["name"]: r for r in json.load(open(cp))}
    # frontier = MEDIAN over top-3-z executors (single-executor readout was polluted by
    # qwen35-122b's score compression once it became the lone top point)
    e_top3 = sorted((e for e in execs if e in zmap and pm_auc.get(e)),
                    key=lambda e: -zmap[e])[:3]
    for lab, es, src in [("at8B", ["llama8b"], pm_auc),
                         ("frontier", e_top3, pm_auc),
                         ("frontier_lenctl", e_top3, pm_auc_lc)]:
        mi_, av_ = [], []
        for n in bank:
            c = cert.get(n)
            vals = [src[e][n] for e in es if e in src and n in src[e]]
            if not c or c.get("opt_omega_bits") is None or not vals:
                continue
            mi_.append(float(c["opt_omega_bits"]))
            av_.append(abs(float(np.median(vals)) - .5))
        if len(mi_) >= 15:
            r = spearmanr(mi_, av_)
            silver[f"mi_arm_{lab}"] = dict(n=len(mi_), rho=float(r.correlation),
                                           p=float(r.pvalue))
            print(f"[{task}] MI arm {lab}: rho(OPT_bits, |AUC-.5|) = {r.correlation:+.3f} "
                  f"(n={len(mi_)}, p={r.pvalue:.4f})")
            if lab == "frontier":                       # raw pairs for the notebook scatter
                json.dump({"mi": mi_, "auc": av_},
                          open(f"{O}/mi_pairs_{task}.json", "w"), default=float)

json.dump(curves, open(f"{O}/curves_{task}.json", "w"), default=float)
if rows:
    json.dump({"ceiling": None if not np.isfinite(ceil) else ceil,
               "ceiling_is_proved": False, "planted_reference": planted_ref,
               "executors": execs, "rows": rows},
              open(f"{O}/laws_{task}.json", "w"), indent=1, default=float)
json.dump(silver, open(f"{O}/silver_{task}.json", "w"), indent=1, default=float)
print(f"[{task}] -> curves/laws/silver saved under {O}/")

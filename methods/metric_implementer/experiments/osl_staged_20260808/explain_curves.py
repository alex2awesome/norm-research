"""STEP 2 — Explain the curves: what makes a concept articulable?
y-variables (per metric): curve slope, frontier level (top_y), asymptote L, verdict class,
audited class (TACIT-CANDIDATE vs REACHES-OK from bounded_audit), frontier floor.
x-features (all label-free, all PRE-DATING the curves):
  - certificate: opt_omega_bits, H_M, frac_H, n_head, eps_bits
  - CUF unit structure (8B bank): n_units, atom_frac, mean/max delta_M, frac_detect_M,
    mean eps_ctx, mean sign_stability, dead_frac
  - breadth controls: desc length, hierarchy cluster size
Design honors the a-priori-LODO lesson (text-tag models reversed out-of-domain): every
correlation is WITHIN-task partial (| dlen), meta-combined by Stouffer across tasks; the
classifier is evaluated leave-one-domain-out AND within-humor CV, on the honest AUC scale.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr, rankdata, norm as ndist

B = "/lfs/skampere3/0/alexspan"
O = f"{B}/outputs/osl_multi"
CUF = {"humor": "humor", "creative_writing": "creative-writing", "math": "math",
       "press_releases": "press-releases", "news_homepages": "news-homepages",
       "peer_review": "peer-review", "notice_and_comment": "notice-and-comment",
       "patents": "patents"}
HIER = CUF  # same naming for hierarchy files


def unit_features(task):
    p = f"{B}/outputs/unit_cert/bank/{CUF[task]}/llama8b/bank_units.jsonl"
    if not os.path.exists(p):
        return {}
    feats = {}
    for l in open(p):
        try:
            r = json.loads(l)
        except Exception:
            continue
        rows = [u for u in r.get("rows", []) if u.get("level") == 1]
        if not rows:
            continue
        dM = [u.get("delta_M") for u in rows if u.get("delta_M") is not None]
        feats[r["metric"]] = dict(
            n_units=len(rows),
            n_cert=sum(1 for u in rows if u.get("verdict") == "CERTIFIED-UNIT"),
            atom_frac=np.mean([u.get("atom") == "ATOM" for u in rows]),
            mean_dM=np.mean(dM) if dM else np.nan,
            max_dM=np.max(dM) if dM else np.nan,
            frac_detM=np.mean([bool(u.get("detect_M")) for u in rows]),
            frac_detF=np.mean([bool(u.get("detect_free")) for u in rows]),
            mean_epsctx=np.nanmean([u.get("eps_ctx") if u.get("eps_ctx") is not None
                                    else np.nan for u in rows]),
            mean_signstab=np.nanmean([u.get("sign_stability") if u.get("sign_stability")
                                      is not None else np.nan for u in rows]),
            dead_frac=np.mean([u.get("verdict") == "SUBTHRESHOLD" for u in rows]))
    return feats


audit = json.load(open(f"{O}/bounded_audit.json")) if os.path.exists(f"{O}/bounded_audit.json") else {}
FEATS = ["opt_omega_bits", "H_M", "frac_H", "n_head", "eps_bits",
         "n_units", "n_cert", "atom_frac", "mean_dM", "max_dM", "frac_detM", "frac_detF",
         "mean_epsctx", "mean_signstab", "dead_frac", "floor"]
YS = ["slope", "top_y", "L"]

table = []
for task in CUF:
    lp = f"{O}/laws_{task}.json"
    if not os.path.exists(lp):
        continue
    laws = {r["name"]: r for r in json.load(open(lp))["rows"]}
    cert_p = f"{B}/outputs/silver_r2/{task}/cert_{task}.json"
    cert = {r["name"]: r for r in json.load(open(cert_p))} if os.path.exists(cert_p) else {}
    uf = unit_features(task)
    mg = json.load(open(f"{B}/norm-research/outputs/hierarchy/{HIER[task]}_general_r2_expanded.json"))["merged_groups"]
    dlen = {g["merged_name"]: len((g.get("merged_description") or "").split()) for g in mg
            if g.get("merged_name")}
    # floors were computed for BOUNDED/REACHES in the audit; extend via curves file quickly
    fl_map = {x["name"]: x.get("floor") for x in audit.get(task, [])}
    aud_map = {x["name"]: x.get("klass") for x in audit.get(task, [])}
    for n, r in laws.items():
        if r.get("verdict") == "NOISY":
            continue
        c = cert.get(n, {})
        u = uf.get(n, {})
        row = dict(task=task, name=n, verdict=r["verdict"], slope=r.get("slope"),
                   top_y=r.get("top_y"), L=r.get("L"), dlen=dlen.get(n, 0),
                   floor=fl_map.get(n), audited=aud_map.get(n),
                   opt_omega_bits=c.get("opt_omega_bits"), H_M=c.get("H_M"),
                   frac_H=c.get("frac_H"), n_head=c.get("n_head"), eps_bits=c.get("eps_bits"))
        row.update({k: u.get(k) for k in ("n_units", "n_cert", "atom_frac", "mean_dM",
                                          "max_dM", "frac_detM", "frac_detF", "mean_epsctx",
                                          "mean_signstab", "dead_frac")})
        table.append(row)
print(f"metric-feature table: {len(table)} rows over "
      f"{len(set(r['task'] for r in table))} tasks")


def partial_within(rows, feat, y):
    x = np.array([r.get(feat) if r.get(feat) is not None else np.nan for r in rows], float)
    yy = np.array([r.get(y) if r.get(y) is not None else np.nan for r in rows], float)
    dl = np.array([r["dlen"] for r in rows], float)
    ok = np.isfinite(x) & np.isfinite(yy) & np.isfinite(dl)
    if ok.sum() < 12 or x[ok].std() == 0:
        return None
    rx, ry, rd = rankdata(x[ok]), rankdata(yy[ok]), rankdata(dl[ok])
    R = np.column_stack([rd, np.ones(ok.sum())])
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
    ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    r = spearmanr(ex, ey).correlation
    return (float(r), int(ok.sum()))


print("\n== per-feature partial rho (| desc-len), WITHIN task, Stouffer-combined ==")
print(f"{'feature':16s}" + "".join(f" {y:>16s}" for y in YS))
tasks = sorted(set(r["task"] for r in table))
meta = {}
for feat in FEATS:
    line = f"{feat:16s}"
    for y in YS:
        zs = []
        for t in tasks:
            res = partial_within([r for r in table if r["task"] == t], feat, y)
            if res:
                r_, n_ = res
                zs.append((np.arctanh(np.clip(r_, -.99, .99)) * np.sqrt(max(n_ - 3, 1)), n_))
        if zs:
            Z = sum(z for z, _ in zs) / np.sqrt(len(zs))
            p = 2 * (1 - ndist.cdf(abs(Z)))
            star = "**" if p < .01 else ("*" if p < .05 else "  ")
            line += f" {Z:+7.2f}z p={p:.3f}{star}"
            meta[(feat, y)] = (float(Z), float(p))
        else:
            line += f" {'--':>16s}"
    print(line)

# classifier: REACHES vs BOUNDED (+ audited TACIT vs REACHES-OK), LODO + within-humor CV
def clf_eval(rows, pos_label, neg_label, labfield):
    def xy(rs):
        X, Y = [], []
        for r in rs:
            lab = r.get(labfield)
            if lab is None:
                continue
            pos = pos_label in str(lab)
            neg = neg_label in str(lab) and not pos
            if not (pos or neg):
                continue
            X.append([r.get(f) if r.get(f) is not None and np.isfinite(r.get(f) or np.nan)
                      else np.nan for f in FEATS] + [r["dlen"]])
            Y.append(1 if pos else 0)
        return np.array(X, float), np.array(Y, float)
    X, Y = xy(rows)
    if len(Y) < 20 or Y.sum() < 6 or Y.sum() > len(Y) - 6:
        return None
    keep = np.isfinite(X).mean(0) >= 0.5          # drop mostly-NaN features (e.g. signstab)
    X = X[:, keep]
    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)
    Xr = np.column_stack([rankdata(c) for c in X.T])
    Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9)
    n = len(Y)
    preds = np.zeros(n)
    for i in range(n):                                    # LOO ridge-logit approx (linear)
        tr = np.arange(n) != i
        w = np.linalg.solve(Xr[tr].T @ Xr[tr] + 5 * np.eye(Xr.shape[1]),
                            Xr[tr].T @ (Y[tr] - Y[tr].mean()))
        preds[i] = Xr[i] @ w
    r = rankdata(preds)
    aucv = (r[Y > 0].mean() - (Y.sum() + 1) / 2) / (len(Y) - Y.sum())
    return float(aucv), int(len(Y)), int(Y.sum())


print("\n== classifiers (LOO rank-ridge, honest AUC) ==")
res = clf_eval(table, "BOUNDED", "REACHES", "verdict")
if res:
    print(f"BOUNDED vs REACHES (all tasks pooled): AUC={res[0]:.3f} (n={res[1]}, pos={res[2]})")
res = clf_eval([r for r in table if r["task"] == "humor"], "BOUNDED", "REACHES", "verdict")
if res:
    print(f"BOUNDED vs REACHES (humor only):       AUC={res[0]:.3f} (n={res[1]}, pos={res[2]})")
res = clf_eval([r for r in table if r["task"] == "humor"], "TACIT-CANDIDATE", "REACHES-OK",
               "audited")
if res:
    print(f"TACIT-CAND vs REACHES-OK (audited, humor): AUC={res[0]:.3f} (n={res[1]}, pos={res[2]})")

json.dump({"table": table, "meta": {f"{k[0]}|{k[1]}": v for k, v in meta.items()}},
          open(f"{O}/explain_curves.json", "w"), indent=1, default=float)
print(f"\n-> {O}/explain_curves.json")

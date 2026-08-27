#!/usr/bin/env python3
"""y-variable audit for N&C VAT: which structured y carries text signal, and where
does it disappear (docket level vs comment level)?

For each candidate y (from per-label engagement / response_type / outcome axes):
  - pooled docket-grouped AUC for char_len / V / A / V+A / dense (same instruments,
    same rows, OOF predictions)
  - docket purity (share of multi-comment dockets that are single-class)
  - WITHIN-docket AUC: P(score_i > score_j | same docket, y_i=1, y_j=0) over OOF scores
    — the comment-level readout with docket identity fully conditioned out
  - BETWEEN-docket AUC: same over docket-mean scores vs docket-majority y.

Uses the existing Gemma A-score npz shards (label-agnostic).
"""
import glob, json, re
import numpy as np
from collections import defaultdict
from scipy.sparse import hstack as sp_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aggregate_vat_nc import v_features, V_NAMES

rows = {r["doc_id"]: r for r in (json.loads(l) for l in open("nc_vat_sample.jsonl"))}
labs = json.load(open("nc_vat_sample_labels_full.json"))

X_by_id = {}
for p in sorted(glob.glob("nc_scores_shard*.npz")):
    d = np.load(p, allow_pickle=True)
    for i, did in enumerate(d["doc_id"]):
        if str(d["agency"][i]) != "__ANCHOR":
            X_by_id[str(did)] = d["X"][i]
ids = [i for i in rows if i in X_by_id]
A = np.array([X_by_id[i] for i in ids], dtype=float)
V = np.array([[v_features(rows[i]["text"])[n] for n in V_NAMES] for i in ids], dtype=float)
texts = [rows[i]["text"] for i in ids]
docket = np.array([rows[i]["docket"] or ("solo_" + i) for i in ids])
charlen = np.array([[float(len(t))] for t in texts])

SUBST = {"substantive_response"}
AGREE = {"accepted", "agree"}
PARTIAL = {"partially_accepted"}
DISAGREE = {"disagree"}

def maj(vals, pos, neg):
    p = sum(v in pos for v in vals); n = sum(v in neg for v in vals)
    if p == n:
        return -1
    return int(p > n)

def y_defs(doc_id):
    ls = labs.get(doc_id, [])
    eng = [l["engagement"] for l in ls]
    rt = [l["response_type"] for l in ls]
    oc = [l["outcome_collapsed"] for l in ls]
    oc_sub = [l["outcome_collapsed"] for l in ls if l["engagement"] in SUBST]
    return {
        "outcome_maj (baseline)": maj(oc, {"MADE"}, {"NONE"}),
        "outcome_maj|substantive": maj(oc_sub, {"MADE"}, {"NONE"}),
        "any_consider vs NONE": maj(oc, {"MADE", "CONSIDERED"}, {"NONE"}),
        "MADE vs CONSIDERED": maj(oc, {"MADE"}, {"CONSIDERED"}),
        "engagement subst vs proc": maj(eng, SUBST,
            {"procedural_response", "procedural_or_general_discussion", "no_response",
             "no_engagement", "mentioned_without_substantive_response", "acknowledgement"}),
        "agree vs disagree": maj(rt, AGREE, DISAGREE),
        "agree+part vs disagree": maj(rt, AGREE | PARTIAL, DISAGREE),
    }

Y = {k: np.array([y_defs(i)[k] for i in ids]) for k in y_defs(ids[0])}

def oof_scores(M, y, groups, sparse_texts=None):
    oof = np.full(len(y), np.nan)
    k = min(5, len(set(groups.tolist())))
    sgkf = StratifiedGroupKFold(k, shuffle=True, random_state=0)
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        if sparse_texts is not None:
            vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000, sublinear_tf=True)
            Xtr = sp_hstack([vw.fit_transform([sparse_texts[i] for i in tr]),
                             vc.fit_transform([sparse_texts[i] for i in tr])]).tocsr()
            Xte = sp_hstack([vw.transform([sparse_texts[i] for i in te]),
                             vc.transform([sparse_texts[i] for i in te])]).tocsr()
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
        else:
            pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5),
                                 StandardScaler(),
                                 LogisticRegression(max_iter=3000, class_weight="balanced"))
            pipe.fit(M[tr], y[tr])
            oof[te] = pipe.predict_proba(M[te])[:, 1]
    return oof

def within_between(oof, y, groups):
    m = ~np.isnan(oof)
    conc = disc = ties = 0
    by = defaultdict(list)
    for i in np.where(m)[0]:
        by[groups[i]].append((y[i], oof[i]))
    for g, items in by.items():
        pos = [s for yy, s in items if yy == 1]
        neg = [s for yy, s in items if yy == 0]
        for sp in pos:
            for sn in neg:
                if sp > sn: conc += 1
                elif sp < sn: disc += 1
                else: ties += 1
    within = (conc + 0.5 * ties) / max(conc + disc + ties, 1) if (conc + disc + ties) else None
    n_pairs = conc + disc + ties
    gy, gs = [], []
    for g, items in by.items():
        ys = [yy for yy, _ in items]
        mu = np.mean(ys)
        if mu in (0.0, 1.0) or len(items) == 1:
            gy.append(int(round(mu))); gs.append(np.mean([s for _, s in items]))
        elif mu > 0.5:
            gy.append(1); gs.append(np.mean([s for _, s in items]))
        else:
            gy.append(0); gs.append(np.mean([s for _, s in items]))
    between = roc_auc_score(gy, gs) if len(set(gy)) == 2 else None
    return within, n_pairs, between, len(by)

print(f"{'y':28} {'n':>5} {'bal':>11} {'purity':>6} | {'len':>5} {'V':>5} {'A':>5} {'VA':>5} {'dense':>5} | {'w/in(pairs)':>13} {'btw(dockets)':>12}")
results = {}
for name, y in Y.items():
    v = y >= 0
    if v.sum() < 500 or len(np.unique(y[v])) < 2:
        print(f"{name:28} SKIP (n={v.sum()})")
        continue
    yy = y[v]; g = docket[v]
    dk = defaultdict(list)
    for i in range(len(yy)): dk[g[i]].append(yy[i])
    multi = [vv for vv in dk.values() if len(vv) > 1]
    purity = np.mean([len(set(vv)) == 1 for vv in multi]) if multi else np.nan
    r = {}
    line = f"{name:28} {v.sum():>5} {str(np.bincount(yy).tolist()):>11} {purity:>6.2f} |"
    for label_, M, st in (("len", charlen[v], None), ("V", V[v], None), ("A", A[v], None),
                          ("VA", np.hstack([V[v], A[v]]), None),
                          ("dense", None, [texts[i] for i in np.where(v)[0]])):
        oof = oof_scores(M, yy, g, sparse_texts=st)
        m = ~np.isnan(oof)
        auc = roc_auc_score(yy[m], oof[m]) if len(np.unique(yy[m])) == 2 else np.nan
        r[label_] = dict(auc=float(auc))
        line += f" {auc:.3f}"
        if label_ in ("VA", "dense"):
            w, npair, b, ndk = within_between(oof, yy, g)
            r[label_].update(within=w, n_pairs=npair, between=b, n_dockets=ndk)
    va, de = r["VA"], r["dense"]
    line += f" | VA:{va['within'] if va['within'] else float('nan'):.3f}({va['n_pairs']})"
    line += f" de:{de['within'] if de['within'] else float('nan'):.3f}"
    line += f" | VA:{va['between'] if va['between'] else float('nan'):.3f} de:{de['between'] if de['between'] else float('nan'):.3f}({de['n_dockets']})"
    print(line, flush=True)
    results[name] = r
json.dump(results, open("nc_y_audit.json", "w"), indent=2)
print("Y_AUDIT_DONE -> nc_y_audit.json")

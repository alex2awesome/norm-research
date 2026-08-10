#!/usr/bin/env python3
"""Aggregate N&C VAT: V (regex) / A (Gemma rubric scores) / V+A / dense -> AUC table.

Pooled + per-agency, primary y = rule-change outcome (MADE vs NONE), secondary
y = engagement (substantive vs not, unbalanced — AUC is threshold-free).
CV is docket-grouped (StratifiedGroupKFold) — dockets never straddle folds.
Dense = TF-IDF word(1-2) + char_wb(3-5) LR on the same rows, same folds (matched
protocol; no external big-train here).

Primary y = MAJORITY outcome across a comment's MADE/NONE labels (ties dropped) —
the any-MADE union y is confounded by n_labels (comments matched to more responses
mechanically have more chances at MADE: n_labels AUC .695 vs any-MADE, .589 vs
majority). any-MADE kept as sensitivity; n_labels + char_len nuisance baselines
reported alongside.

Usage:
  python3 aggregate_vat_nc.py --sample nc_vat_sample.jsonl --npz 'nc_scores_shard*.npz' \
      --labels nc_vat_sample_label_lists.json --out ../../../notebooks/data/nc_vat.json
"""
import argparse, glob, json, re
from collections import defaultdict
import numpy as np
from scipy.sparse import hstack as sp_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- V features (N&C-specific, stdlib regex) ----------------
NUMTOK  = re.compile(r"\b\d[\d,\.]*\b")
SENT_RE = re.compile(r"[.!?]+")
KW = {
    "v_kw_cfr":        re.compile(r"\bC\.?\s?F\.?\s?R\.?\b|\bU\.?\s?S\.?\s?C\.?\b|\bFed\.?\s?Reg\b", re.I),
    "v_kw_section":    re.compile(r"§|\bsection\s+\d|\bsubpart\b|\bsubsection\b", re.I),
    "v_kw_docket":     re.compile(r"\bdocket\b|\bRIN\b|federal register|proposed rule|\bNPRM\b", re.I),
    "v_kw_legal":      re.compile(r"arbitrary and capricious|administrative procedure act|\bAPA\b|statutor|\bunlawful\b|exceed.{0,20}authority|\bcongress\b", re.I),
    "v_kw_data":       re.compile(r"\bdata\b|\bstudy\b|\bstudies\b|\bresearch\b|\bevidence\b|\bsurvey\b|peer[- ]reviewed|\banalysis\b", re.I),
    "v_kw_econ":       re.compile(r"\bcost\b|\bbenefit\b|economic|\bburden\b|\$\s?\d|compliance cost|small business", re.I),
    "v_kw_alt":        re.compile(r"alternativ|\binstead\b|we recommend|we suggest|should consider|we propose|we urge", re.I),
    "v_kw_support":    re.compile(r"\bsupport\b|\bagree\b|applaud|commend|\bendorse\b", re.I),
    "v_kw_oppose":     re.compile(r"\boppose\b|\bobject\b|disagree|\breject\b|\bwithdraw\b|rescind", re.I),
    "v_kw_org":        re.compile(r"\bassociation\b|\borganization\b|\bcoalition\b|\binstitute\b|on behalf of|\bmembers\b|\bindustry\b", re.I),
    "v_kw_change_req": re.compile(r"should be|must be|\brevise\b|\bamend\b|\bmodify\b|clarif|\bstrike\b|\bdelete\b", re.I),
    "v_kw_specific":   re.compile(r"\bpage\s?\d|\bp\.\s?\d|\bpart\s+\d|paragraph|\btable\s?\d|\bfigure\s?\d|line\s?\d", re.I),
    "v_kw_procedural": re.compile(r"comment period|\bextension\b|\bextend\b|\bdeadline\b|public hearing", re.I),
    "v_kw_attach":     re.compile(r"\battach\b|attached|enclosed|appendix|exhibit|see enclosure", re.I),
    "v_kw_experience": re.compile(r"\bI am a\b|\bmy experience\b|years of experience|\bas a\b|first[- ]hand", re.I),
    "v_kw_hedge":      re.compile(r"\bmay\b|\bmight\b|\bcould\b|\bperhaps\b|\bpossibly\b|\bconcern\b", re.I),
}
def v_features(text):
    t = text or ""
    words = t.split()
    nw = max(len(words), 1)
    sents = [s for s in SENT_RE.split(t) if s.strip()]
    ns = max(len(sents), 1)
    alpha = [c for c in t if c.isalpha()]
    feats = {
        "v_char_len":     float(len(t)),
        "v_word_len":     float(nw),
        "v_sent_count":   float(ns),
        "v_avg_word_len": float(sum(len(w) for w in words) / nw),
        "v_avg_sent_len": float(nw / ns),
        "v_num_density":  float(100.0 * len(NUMTOK.findall(t)) / nw),
        "v_pct_count":    float(t.count("%")),
        "v_question":     float(t.count("?")),
        "v_exclaim":      float(t.count("!")),
        "v_caps_ratio":   float(sum(c.isupper() for c in alpha) / max(len(alpha), 1)),
        "v_first_person": float(100.0 * len(re.findall(r"\b(I|my|me|mine)\b", t)) / nw),
    }
    for name, rgx in KW.items():
        feats[name] = float(len(rgx.findall(t)))
    return feats

V_NAMES = ["v_char_len","v_word_len","v_sent_count","v_avg_word_len","v_avg_sent_len",
           "v_num_density","v_pct_count","v_question","v_exclaim","v_caps_ratio",
           "v_first_person"] + list(KW.keys())

# ---------------- grouped CV AUC ----------------
def grouped_auc(M, y, groups, sparse=False, n_boot=0, seed=0):
    """StratifiedGroupKFold(5) out-of-fold AUC. Returns (auc, lo, hi) with
    bootstrap-over-rows CI on the pooled OOF predictions when n_boot>0."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None, None, None
    if sparse:
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
        def fit_predict(tr, te):
            clf.fit(M[tr], y[tr])
            return clf.predict_proba(M[te])[:, 1]
    else:
        pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5),
                             StandardScaler(),
                             LogisticRegression(max_iter=3000, class_weight="balanced"))
        def fit_predict(tr, te):
            pipe.fit(M[tr], y[tr])
            return pipe.predict_proba(M[te])[:, 1]
    oof = np.full(len(y), np.nan)
    k = min(5, len(set(np.asarray(groups).tolist())))
    sgkf = StratifiedGroupKFold(k, shuffle=True, random_state=0)
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
        if len(te) == 0 or len(tr) == 0 or len(np.unique(y[tr])) < 2:
            continue
        oof[te] = fit_predict(tr, te)
    m = ~np.isnan(oof)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return None, None, None
    auc = float(roc_auc_score(y[m], oof[m]))
    lo = hi = None
    if n_boot:
        rng = np.random.default_rng(seed)
        idx = np.where(m)[0]
        bs = []
        for _ in range(n_boot):
            s = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(y[s])) < 2:
                continue
            bs.append(roc_auc_score(y[s], oof[s]))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return auc, lo, hi

def uni(M, names, y):
    out = []
    for j, nm in enumerate(names):
        col = M[:, j]; mask = ~np.isnan(col)
        if mask.sum() > 30 and len(set(y[mask].tolist())) == 2:
            out.append((str(nm), round(roc_auc_score(y[mask], col[mask]), 3), int(mask.sum())))
    out.sort(key=lambda t: -abs(t[1] - 0.5))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="nc_vat_sample.jsonl")
    ap.add_argument("--labels", default="nc_vat_sample_label_lists.json")
    ap.add_argument("--npz", default="nc_scores_shard*.npz")
    ap.add_argument("--stats", default="v42_stats.json")
    ap.add_argument("--out", default="nc_vat.json")
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()

    rows = {r["doc_id"]: r for r in (json.loads(l) for l in open(a.sample)) }

    # merge shards, keyed by doc_id (anchors excluded via y_out == -1)
    X_by_id, a_names = {}, None
    anchor_report = []
    for p in sorted(glob.glob(a.npz)):
        d = np.load(p, allow_pickle=True)
        a_names = [str(x) for x in d["a_names"]]
        for i, did in enumerate(d["doc_id"]):
            did = str(did)
            if str(d["agency"][i]) == "__ANCHOR":
                anchor_report.append((p.split("/")[-1], did, float(np.nanmean(d["X"][i]))))
                continue
            X_by_id[did] = d["X"][i]
    ids = [i for i in rows if i in X_by_id]
    missing = len(rows) - len(ids)
    print(f"merged A-scores for {len(ids)}/{len(rows)} sample rows ({missing} missing)")
    for sh, did, mu in anchor_report:
        print(f"  [anchor {sh}] {did}: mean={mu:.3f}")

    X = np.array([X_by_id[i] for i in ids], dtype=float)
    V = np.array([[v_features(rows[i]["text"])[n] for n in V_NAMES] for i in ids], dtype=float)
    texts = [rows[i]["text"] for i in ids]
    agency = np.array([rows[i]["agency"] for i in ids])
    docket = np.array([rows[i]["docket"] or ("solo_" + i) for i in ids])
    year = np.array([rows[i]["year"] or 0 for i in ids])
    y_any = np.array([rows[i]["y_out"] for i in ids])
    y_eng = np.array([rows[i]["y_eng"] for i in ids])
    n_labels = np.array([rows[i]["n_labels"] for i in ids], dtype=float)

    labmap = json.load(open(a.labels))
    y_maj = np.full(len(ids), -1)
    for k, i in enumerate(ids):
        ocs = [o for o in labmap.get(i, []) if o in ("MADE", "NONE")]
        nm, nn = ocs.count("MADE"), ocs.count("NONE")
        if nm != nn:
            y_maj[k] = int(nm > nn)
    y_out = y_maj  # primary

    # dense features (fit once on full sample vocabulary is leakage-prone; fit per fold
    # would be ideal — for a matched lower-bound we fit the vectorizer per fold inside
    # grouped_auc via a closure over raw text)
    def dense_grouped_auc(idx, y, groups, n_boot=0):
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return None, None, None
        oof = np.full(len(y), np.nan)
        k = min(5, len(set(np.asarray(groups).tolist())))
        sgkf = StratifiedGroupKFold(k, shuffle=True, random_state=0)
        sub_texts = [texts[i] for i in idx]
        for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
            if len(te) == 0 or len(tr) == 0 or len(np.unique(y[tr])) < 2:
                continue
            vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000,
                                 sublinear_tf=True)
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=200000, sublinear_tf=True)
            Xtr = sp_hstack([vw.fit_transform([sub_texts[i] for i in tr]),
                             vc.fit_transform([sub_texts[i] for i in tr])]).tocsr()
            Xte = sp_hstack([vw.transform([sub_texts[i] for i in te]),
                             vc.transform([sub_texts[i] for i in te])]).tocsr()
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
        m = ~np.isnan(oof)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            return None, None, None
        auc = float(roc_auc_score(y[m], oof[m]))
        lo = hi = None
        if n_boot:
            rng = np.random.default_rng(0)
            w = np.where(m)[0]; bs = []
            for _ in range(n_boot):
                s = rng.choice(w, len(w), replace=True)
                if len(np.unique(y[s])) < 2:
                    continue
                bs.append(roc_auc_score(y[s], oof[s]))
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        return auc, lo, hi

    agency_size = {}
    try:
        stats = json.load(open(a.stats))["per_agency"]
        agency_size = {ag: d.get("long", 0) for ag, d in stats.items()}
    except Exception:
        pass

    def block(idx, y, tag, n_boot):
        idx = np.asarray(idx)
        yy = y[idx]
        g = docket[idx]
        res = {}
        for nm, M, sparse in (("V", V[idx], False), ("A", X[idx], False),
                              ("VA", np.hstack([V[idx], X[idx]]), False)):
            auc, lo, hi = grouped_auc(M, yy, g, n_boot=n_boot)
            res[nm] = dict(auc=auc, ci=[lo, hi])
        auc, lo, hi = dense_grouped_auc(idx, yy, g, n_boot=n_boot)
        res["dense"] = dict(auc=auc, ci=[lo, hi])
        res["n"] = int(len(idx))
        res["balance"] = {int(k): int(v) for k, v in zip(*np.unique(yy, return_counts=True))}
        print(f"[{tag}] n={len(idx)} " +
              " ".join(f"{k}={res[k]['auc']:.3f}" if res[k]["auc"] else f"{k}=NA"
                       for k in ("V", "A", "VA", "dense")), flush=True)
        return res

    out = dict(task="notice_and_comment_v42",
               label_primary="outcome majority-MADE(1) vs majority-NONE(0), ties dropped, docket-grouped CV",
               label_sensitivity="any-MADE union (n_labels-confounded: .695)",
               label_secondary="engagement substantive vs not",
               n_A=X.shape[1], n_V=V.shape[1],
               na_rate=float(np.isnan(X).mean()),
               anchors=[dict(shard=s, id=d, mean=m) for s, d, m in anchor_report])

    valid = y_out >= 0
    allidx = np.where(valid)[0]
    out["pooled_out"] = block(allidx, y_out, "pooled/outcome-majority", a.boot)
    out["pooled_out"]["top_A"] = uni(X[valid], a_names, y_out[valid])[:15]
    out["pooled_out"]["top_V"] = uni(V[valid], V_NAMES, y_out[valid])[:15]
    out["pooled_out"]["nuisance"] = dict(
        n_labels_auc=float(roc_auc_score(y_out[valid], n_labels[valid])),
        char_len_auc=float(roc_auc_score(y_out[valid],
                                         [len(t) for t, v in zip(texts, valid) if v])))
    out["pooled_any"] = block(np.arange(len(ids)), y_any, "pooled/outcome-anyMADE", a.boot)
    out["pooled_any"]["nuisance"] = dict(
        n_labels_auc=float(roc_auc_score(y_any, n_labels)))
    if len(np.unique(y_eng)) == 2:
        out["pooled_eng"] = block(np.arange(len(ids)), y_eng, "pooled/engagement", a.boot)

    out["per_agency"] = {}
    for ag in sorted(set(agency.tolist())):
        idx = np.where((agency == ag) & valid)[0]
        if len(idx) < 80 or len(np.unique(y_out[idx])) < 2:
            continue
        try:
            r = block(idx, y_out, f"{ag}/outcome", a.boot)
        except Exception as e:
            print(f"[{ag}/outcome] SKIPPED ({type(e).__name__}: {e})", flush=True)
            continue
        r["agency_long_count"] = agency_size.get(ag)
        yrs = year[idx][year[idx] > 1990]
        r["year_median"] = float(np.median(yrs)) if len(yrs) else None
        out["per_agency"][ag] = r

    # time axis: pooled by year bucket
    out["by_year"] = {}
    for lo_y, hi_y in ((1990, 2008), (2008, 2014), (2014, 2018), (2018, 2021), (2021, 2027)):
        idx = np.where((year >= lo_y) & (year < hi_y) & valid)[0]
        if len(idx) >= 300 and len(np.unique(y_out[idx])) == 2:
            out["by_year"][f"{lo_y}-{hi_y-1}"] = block(idx, y_out, f"year {lo_y}-{hi_y-1}", 0)

    json.dump(out, open(a.out, "w"), indent=2)
    print("AGG_DONE ->", a.out)

if __name__ == "__main__":
    main()

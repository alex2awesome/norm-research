#!/usr/bin/env python3
"""Tiered claim verification on press releases (the cross-domain flagship).
For each PR claim (from claims_pr.jsonl), verify against FOUR temporally-gated pools:
  T1 own body           (internal consistency)
  T2 coverage articles  (press_release_news_mappings -> news_articles; did journalists echo it?)
  T3 company history    (pr_history.sqlite, year < pr_year; superlative/novelty check)
  T4 wiki               (encyclopedic background; timeless mode, flagged)
Doc metrics: t{1,2,3,4}_support_rate, echo_rate (T2 FULL|PARTIAL), unechoed_self_assertion
(T1-supported but not T2), novelty_repeat_rate (T3 FULL = claim already made before).
NULL twin: T2 with a DIFFERENT PR's coverage articles.
Eval: company-grouped AUC vs k>=3; per-tier + tier-ratio metrics.
Run on sk3: python -m methods.claim_verification.run_tiered_pr [--n 600]"""
import argparse, ast as _ast, csv as _csv, gzip as _gzip, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
_csv.field_size_limit(sys.maxsize)
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, split_head_body, make_passages, verify_claim, _sentences
from claim_verification.evidence_api import EvidenceAPI, clean_evidence_text, chunk_passages

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma", "doc_kind": "press release"}

def load_claims():
    out = {}
    for ln in open(os.path.join(EB, "claims_pr.jsonl")):
        try:
            r = json.loads(ln)
            if r.get("claims"): out[str(r["doc_id"])] = [c["claim"] if isinstance(c, dict) else str(c) for c in r["claims"]]
        except Exception: pass
    return out

def load_meta():
    """pr_id -> (date, n_outlets) + pr->covering article ids."""
    dates, nout, pr2art = {}, {}, {}
    for r in _csv.DictReader(_gzip.open(f"{ROOT}/datasets/press-releases/press_release_modeling_dataset.csv.gz", "rt")):
        pid = str(r["press_release_id"])
        dates[pid] = (r.get("press_release_date") or "")[:10]
        dom = (r.get("news_article_domain") or "").strip()
        try: nout[pid] = len(set(_ast.literal_eval(dom))) if dom else 0
        except Exception: nout[pid] = 0
        aid = (r.get("news_article_id") or "").strip()
        if aid:
            try:
                ids = _ast.literal_eval(aid) if aid.startswith("[") else [aid]
                pr2art[pid] = [str(x) for x in ids][:4]
            except Exception: pass
    return dates, nout, pr2art

def coverage_texts(article_ids, api):
    """Pull coverage article texts from news.sqlite by article_id."""
    out = []
    con = api._con("news")
    for aid in article_ids:
        row = con.execute("SELECT text FROM news WHERE article_id = ? LIMIT 1", (str(aid),)).fetchone()
        if row and row[0]: out.append(clean_evidence_text(row[0]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    api = EvidenceAPI()
    claims = load_claims()
    dates, nout, pr2art = load_meta()
    df = pd.read_parquet(f"{ROOT}/datasets/press-releases/press_release_deconfounded.parquet")
    df["id"] = df.id.astype(str)
    df = df[df.id.isin(claims)]
    df["n_out"] = df.id.map(nout).fillna(0).astype(int)
    pos = df[df.n_out >= 3].sample(min(args.n // 2, (df.n_out >= 3).sum()), random_state=0)
    neg = df[df.n_out == 0].sample(min(args.n // 2, (df.n_out == 0).sum()), random_state=0)
    samp = pd.concat([pos, neg]).sample(frac=1, random_state=0).reset_index(drop=True)
    samp["judgement"] = (samp.n_out >= 3).astype(int)
    print(f"[tiered] {len(samp)} PRs ({samp.judgement.sum()} pos), {samp.company.nunique()} companies", flush=True)
    cache = Cache(f"{ROOT}/outputs/tiered_pr/cache.jsonl")
    rng = np.random.default_rng(0)
    # null-twin coverage: shuffle pr->coverage assignment among sampled POS docs
    pos_ids = [i for i in samp.id if pr2art.get(i)]
    perm = dict(zip(pos_ids, list(np.roll(pos_ids, 1))))
    lock = Lock(); results = {}
    def work(row):
        pid, text, company, date = row.id, row.text, str(row.company), dates.get(row.id, "")
        cl = claims[pid][:4]
        head, body = split_head_body(text)
        pools = {}
        pools["t1"] = make_passages(body)
        cov = coverage_texts(pr2art.get(pid, []), api)
        pools["t2"] = [p for t in cov for p in chunk_passages(t)[:6]][:12]
        year = date[:4] if date else str(row.year or "")
        pools["t3"] = [p["passage"] for c in cl[:2] for p in api.pr_history(c, as_of_year=year or None, company=company)][:12]
        pools["t4"] = [p["passage"] for c in cl[:2] for p in api.wiki(c, allow_timeless=True)][:8]
        nc = coverage_texts(pr2art.get(perm.get(pid, ""), []), api)
        pools["t2null"] = [p for t in nc for p in chunk_passages(t)[:6]][:12]
        out = {}
        for tier, pool in pools.items():
            if not pool:
                out[tier] = None; continue
            vs = [verify_claim(c, pool, CFG, cache) for c in cl]
            full = sum(1 for v in vs if v["verdict"] == "FULL")
            part = sum(1 for v in vs if v["verdict"] == "PARTIAL")
            out[tier] = {"support": full / len(vs), "echo": (full + part) / len(vs), "n": len(vs)}
        with lock:
            results[pid] = out
            if len(results) % 50 == 0: print(f"[tiered] {len(results)}/{len(samp)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, [r for r in samp.itertuples()]))
    # doc metrics
    rows = []
    for r in samp.itertuples():
        o = results.get(r.id, {})
        m = {"id": r.id, "y": r.judgement, "company": str(r.company)}
        for t in ("t1", "t2", "t3", "t4", "t2null"):
            d = o.get(t)
            m[f"{t}_support"] = d["support"] if d else np.nan
            m[f"{t}_echo"] = d["echo"] if d else np.nan
        t1, t2 = m["t1_support"], m["t2_echo"]
        m["unechoed_self_assertion"] = (t1 * (1 - t2)) if (t1 == t1 and t2 == t2) else np.nan
        rows.append(m)
    M = pd.DataFrame(rows)
    M.to_csv(f"{ROOT}/outputs/tiered_pr/tiered_metrics.csv", index=False)
    # eval
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    y = M.y.values; g = M.company.values
    print("\n[tiered] === per-metric univariate AUC (k>=3, n with data) ===", flush=True)
    for c in [c for c in M.columns if c not in ("id", "y", "company")]:
        v = M[c].values.astype(float); mk = ~np.isnan(v)
        if mk.sum() > 80 and np.std(v[mk]) > 0:
            print(f"  {c:28} AUC={roc_auc_score(y[mk], v[mk]):.4f} (n={mk.sum()})", flush=True)
    feats = ["t1_support", "t2_support", "t2_echo", "t3_support", "t4_support", "unechoed_self_assertion"]
    X = M[feats].values.astype(float)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(pipe, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                        groups=g, scoring="roc_auc").mean()
    print(f"\n[tiered] TIERED BANK grouped AUC = {a:.4f}   (refs: V=0.628 A=0.648 dense=0.705)", flush=True)
    print("TIERED_PR_DONE", flush=True)

if __name__ == "__main__":
    os.makedirs(f"{ROOT}/outputs/tiered_pr", exist_ok=True)
    main()

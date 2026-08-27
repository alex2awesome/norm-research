#!/usr/bin/env python3
"""Citation-y study (E3 core contrast): SAME docs, SAME claim metrics, TWO outcome types.
Corpus: openalex_citations_v3 (28,425 NeurIPS/ICML/ICLR papers 2013-2023) — has BOTH
  y_accept  = judgement (institutional)
  y_cite    = citation percentile >= median within venue x year (crowd/impact)
Metrics per paper (claims extracted from abstract text):
  t3 field-corpus support/echo + novelty (earlier abstracts, year-gated FTS)
  t4 wiki support/echo (timeless)
  cq_* claim quality 5 dims (specificity/ambition/surprisingness/falsifiability/elegance)
No t1 (no body text here) — internal consistency is covered by the 2024-25 tiered_peer arm.
H_outcome: quality/style metrics should load on citation-y more than accept-y.
Run on sk3: python -m methods.claim_verification.run_citation_y [--n 900]"""
import argparse, json, os, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, extract_claims, verify_claim
from claim_verification.evidence_api import EvidenceAPI, fts_terms

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CIT = f"{ROOT}/datasets/peer-review/openalex_citations/openalex_citations_v3.csv.gz"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
       "doc_kind": "scientific paper abstract", "max_claims": 4}

def best_prompt():
    try:
        for ln in open(f"{ROOT}/outputs/gepa_extract_results.jsonl"):
            r = json.loads(ln)
            if r["domain"] in ("peer", "peerintro") and r.get("best_prompt"):
                return r["best_prompt"]
    except Exception:
        pass
    return None

def build_fts(df):
    """FTS over ALL abstracts for the year-gated field-corpus tier."""
    path = os.path.join(EB, "citation_abstracts.sqlite")
    if os.path.exists(path): return path
    con = sqlite3.connect(path + ".tmp")
    con.execute("CREATE VIRTUAL TABLE ab USING fts5(pid UNINDEXED, year UNINDEXED, text)")
    rows = [(str(r.id), int(r.year), str(r.text)[:6000]) for r in df.itertuples()
            if isinstance(r.text, str) and len(r.text) > 200]
    con.executemany("INSERT INTO ab VALUES (?,?,?)", rows)
    con.commit(); con.close(); os.rename(path + ".tmp", path)
    print(f"[cit-y] built citation_abstracts.sqlite ({len(rows)})", flush=True)
    return path

class FieldFTS:
    def __init__(self, path):
        import threading
        self.path = path; self._local = threading.local()
    def query(self, claim, year, k=8):
        if getattr(self._local, "con", None) is None:
            self._local.con = sqlite3.connect(self.path)
        q = fts_terms(claim)
        if not q: return []
        try:
            rows = self._local.con.execute(
                "SELECT text FROM ab WHERE ab MATCH ? AND year < ? ORDER BY bm25(ab) LIMIT ?",
                (q, int(year), k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0][:700] for r in rows]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    d = pd.read_csv(CIT, compression="gzip")
    d = d[d.text.astype(str).str.len() > 300].copy()
    d["pct"] = pd.to_numeric(d.percentile, errors="coerce")
    d = d[d.pct.notna() & d.judgement.notna() & (d.year >= 2016)].copy()
    d["grp"] = d.venue.astype(str) + "|" + d.year.astype(int).astype(str)
    print(f"[cit-y] eligible {len(d)}; judgement dist {d.judgement.value_counts().to_dict()}", flush=True)
    # balanced sample: per venue-year cell, half accept/half reject where possible
    per_cell = max(args.n // d.grp.nunique(), 8)
    d2 = d.sample(frac=1, random_state=0)  # groupby.head keeps all columns (apply drops group keys in pandas>=2.2)
    samp = d2.groupby(["grp", "judgement"], group_keys=False).head(max(per_cell // 2, 1))
    samp = samp.sample(min(args.n, len(samp)), random_state=0).reset_index(drop=True)
    print(f"[cit-y] sample {len(samp)} over {samp.grp.nunique()} venue-year cells; "
          f"accept {samp.judgement.mean():.2f}", flush=True)
    bp = best_prompt()
    if bp:
        import claim_verification.prompts as P
        P.CLAIM_EXTRACT = bp
        print("[cit-y] using GEPA-optimized peer prompt", flush=True)
    fts = FieldFTS(build_fts(d)); api = EvidenceAPI()
    from claim_verification.run_claim_quality import judge_claim
    cache = Cache(f"{ROOT}/outputs/citation_y/cache.jsonl")
    cq_cache = Cache(f"{ROOT}/outputs/citation_y/cq_cache.jsonl")
    lock = Lock(); rows = []
    def work(r):
        m = {"id": str(r.id), "grp": r.grp, "year": int(r.year),
             "y_accept": int(r.judgement), "pct": float(r.pct)}
        try:
            cl = extract_claims(str(r.text)[:1800], CFG, cache)[:4]
            cl = [c["claim"] if isinstance(c, dict) else str(c) for c in cl]
            if not cl: raise ValueError("no claims")
            m["n_claims"] = len(cl)
            pools = {"t3": [p for c in cl[:2] for p in fts.query(c, r.year)][:12],
                     "t4": [p["passage"] for c in cl[:2] for p in api.wiki(c, allow_timeless=True)][:8]}
            for tier, pool in pools.items():
                if not pool:
                    m[f"{tier}_support"] = np.nan; m[f"{tier}_echo"] = np.nan; continue
                vs = [verify_claim(c, pool, CFG, cache) for c in cl]
                full = sum(1 for v in vs if v["verdict"] == "FULL")
                part = sum(1 for v in vs if v["verdict"] == "PARTIAL")
                m[f"{tier}_support"] = full / len(vs); m[f"{tier}_echo"] = (full + part) / len(vs)
            m["novelty"] = 1 - m["t3_echo"] if m.get("t3_echo") == m.get("t3_echo") else np.nan
            scores = [judge_claim(c, "scientific paper", cq_cache) for c in cl]
            for dim in ("specificity", "ambition", "surprisingness", "falsifiability", "elegance"):
                vals = [s[dim] for s in scores if s.get(dim) is not None]
                m[f"cq_{dim}"] = float(np.mean(vals)) if vals else np.nan
            allv = [s[dim] for s in scores for dim in
                    ("specificity", "ambition", "surprisingness", "falsifiability", "elegance")
                    if s.get(dim) is not None]
            m["cq_mean"] = float(np.mean(allv)) if allv else np.nan
        except Exception as e:
            m["err"] = str(e)[:60]
        with lock:
            rows.append(m)
            if len(rows) % 50 == 0: print(f"[cit-y] {len(rows)}/{len(samp)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, [r for r in samp.itertuples()]))
    M = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/outputs/citation_y", exist_ok=True)
    M["cite_rank"] = M.groupby("grp").pct.rank(pct=True)
    M["y_cite"] = (M.cite_rank >= 0.5).astype(int)
    M.to_csv(f"{ROOT}/outputs/citation_y/metrics.csv", index=False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    ev = ["t3_support", "t3_echo", "t4_support", "t4_echo", "novelty"]
    cq = [c for c in M.columns if c.startswith("cq_")]
    def auc(cols, y, g, label, frame=None):
        F = M if frame is None else frame
        X = F[cols].values.astype(float)
        mk = ~np.all(np.isnan(X), axis=1) & pd.Series(y).notna().values
        if mk.sum() < 100: print(f"  {label:34} SKIP", flush=True); return
        folds = min(5, len(set(np.asarray(g)[mk])))
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        try:
            a = cross_val_score(pipe, X[mk], np.asarray(y)[mk].astype(int),
                                cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                                groups=np.asarray(g)[mk], scoring="roc_auc")
            print(f"  {label:34} AUC={np.nanmean(a):.4f} (n={int(mk.sum())})", flush=True)
        except Exception as e:
            print(f"  {label:34} ERR {str(e)[:40]}", flush=True)
    g = M.grp.values
    for yname, y in (("ACCEPT (institutional)", M.y_accept.values),
                     ("CITATION pct>=median (crowd)", M.y_cite.values)):
        print(f"\n[cit-y] -- y = {yname} --", flush=True)
        auc(ev, y, g, "evidence (t3+t4+novelty)")
        auc(cq, y, g, "claim quality (A)")
        auc(ev + cq, y, g, "evidence + quality")
    acc = M[M.y_accept == 1]
    if len(acc) > 150:
        ya = (acc.cite_rank >= 0.5).astype(int).values
        print("\n[cit-y] -- y = CITATION pct (ACCEPTED only) --", flush=True)
        auc(ev, ya, acc.grp.values, "evidence (t3+t4+novelty)", frame=acc)
        auc(cq, ya, acc.grp.values, "claim quality (A)", frame=acc)
    print("\n[cit-y] univariate AUC (both y):", flush=True)
    for c in ev + cq:
        v = M[c].values.astype(float); mk = ~np.isnan(v)
        if mk.sum() > 150 and np.nanstd(v[mk]) > 0:
            a1 = roc_auc_score(M.y_accept.values[mk], v[mk])
            a2 = roc_auc_score(M.y_cite.values[mk], v[mk])
            print(f"  {c:20} accept={a1:.4f}  cite={a2:.4f}", flush=True)
    print("CITATION_Y_DONE", flush=True)

if __name__ == "__main__":
    main()

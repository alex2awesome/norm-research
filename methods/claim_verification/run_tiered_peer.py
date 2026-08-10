#!/usr/bin/env python3
"""Tiered claim verification on PEER REVIEW (ICLR) — the contrast arm to run_tiered_pr.
Claims: from claims_peerintro.jsonl (abstract+introduction).
Tiers (all within-paper or temporally-safe):
  T1 paper internals — experiments/results/related-work/conclusion sections (the FactEval move:
     "is this claim supported in the body"; user's peer-review questions)
  T3 field corpus    — OTHER ICLR papers from EARLIER years (novelty/priority check; the
     patents prior-art analog)
  T4 wiki            — encyclopedic background (timeless mode)
y = accept/reject (judgement in peer_review_modeling_dataset). Groups = year (venue-stable).
Both classes have all tiers -> no availability leakage (unlike PR's T2).
Run on sk3: python -m methods.claim_verification.run_tiered_peer [--n 600]"""
import argparse, json, os, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, verify_claim, _sentences
from claim_verification.evidence_api import EvidenceAPI, chunk_passages

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
PDF_DB = os.path.join(ROOT, "datasets/peer-review/peer_review_pdfs.db")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
       "doc_kind": "scientific paper"}
INTERNAL_SECTIONS = ["experiments", "results", "related work", "conclusion", "method", "methods"]

def load_claims():
    out = {}
    for ln in open(os.path.join(EB, "claims_peerintro.jsonl")):
        try:
            r = json.loads(ln)
            if r.get("claims"):
                out[str(r["doc_id"])] = [c["claim"] if isinstance(c, dict) else str(c) for c in r["claims"]]
        except Exception: pass
    return out

class PaperDB:
    """Thread-local access to sections + a tiny FTS over abstracts for the field-corpus tier."""
    def __init__(self):
        import threading
        self._local = threading.local()
        self._build_fts()
    def _con(self):
        if getattr(self._local, "con", None) is None:
            self._local.con = sqlite3.connect(PDF_DB)
        return self._local.con
    def _build_fts(self):
        fts_path = os.path.join(EB, "peer_abstracts.sqlite")
        if os.path.exists(fts_path): return
        d = pd.read_csv(f"{ROOT}/datasets/peer-review/peer_review_modeling_dataset.csv.gz", compression="gzip")
        d = d[d.paper_id.astype(str).str.startswith("iclr")]
        con = sqlite3.connect(fts_path + ".tmp")
        con.execute("CREATE VIRTUAL TABLE ab USING fts5(paper_id UNINDEXED, year UNINDEXED, text)")
        rows = [(str(r.paper_id), str(r.year), str(r.text)[:8000]) for r in d.itertuples()]
        con.executemany("INSERT INTO ab VALUES (?,?,?)", rows)
        con.commit(); con.close(); os.rename(fts_path + ".tmp", fts_path)
        print(f"[peer] built peer_abstracts.sqlite ({len(rows)})", flush=True)
    def internals(self, forum_id, max_passages=14):
        row = self._con().execute("SELECT sections FROM pdf_versions WHERE paper_id=? AND version=0",
                                  (forum_id,)).fetchone()
        if not row or not row[0]: return []
        try: s = json.loads(row[0])
        except Exception: return []
        txt = "\n".join(str(s.get(k, "")) for k in INTERNAL_SECTIONS if s.get(k))
        return chunk_passages(txt, words_per=110, max_passages=max_passages)
    def field_corpus(self, claim, year, k=8):
        from claim_verification.evidence_api import fts_terms
        fts_path = os.path.join(EB, "peer_abstracts.sqlite")
        if getattr(self._local, "fts", None) is None:
            self._local.fts = sqlite3.connect(fts_path)
        q = fts_terms(claim)
        if not q: return []
        try:
            rows = self._local.fts.execute(
                "SELECT text FROM ab WHERE ab MATCH ? AND year < ? ORDER BY bm25(ab) LIMIT ?",
                (q, str(year), k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0][:700] for r in rows]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    api = EvidenceAPI(); pdb = PaperDB()
    claims = load_claims()
    d = pd.read_csv(f"{ROOT}/datasets/peer-review/peer_review_modeling_dataset.csv.gz", compression="gzip")
    d["paper_id"] = d.paper_id.astype(str)
    d = d[d.paper_id.isin(claims) & d.judgement.notna()]
    print(f"[peer] {len(d)} ICLR papers with claims+label", flush=True)
    pos = d[d.judgement == 1].sample(min(args.n // 2, (d.judgement == 1).sum()), random_state=0)
    neg = d[d.judgement == 0].sample(min(args.n // 2, (d.judgement == 0).sum()), random_state=0)
    samp = pd.concat([pos, neg]).sample(frac=1, random_state=0).reset_index(drop=True)
    print(f"[peer] sample {len(samp)} ({int(samp.judgement.sum())} accept)", flush=True)
    cache = Cache(f"{ROOT}/outputs/tiered_peer/cache.jsonl")
    lock = Lock(); results = {}
    def work(row):
        pid = row.paper_id; forum = pid[5:] if pid.startswith("iclr_") else pid
        cl = claims[pid][:4]; year = row.year
        pools = {"t1": pdb.internals(forum),
                 "t3": [p for c in cl[:2] for p in pdb.field_corpus(c, year)][:12],
                 "t4": [p["passage"] for c in cl[:2] for p in api.wiki(c, allow_timeless=True)][:8]}
        out = {}
        for tier, pool in pools.items():
            if not pool: out[tier] = None; continue
            vs = [verify_claim(c, pool, CFG, cache) for c in cl]
            full = sum(1 for v in vs if v["verdict"] == "FULL")
            part = sum(1 for v in vs if v["verdict"] == "PARTIAL")
            out[tier] = {"support": full / len(vs), "echo": (full + part) / len(vs)}
        with lock:
            results[pid] = out
            if len(results) % 50 == 0: print(f"[peer] {len(results)}/{len(samp)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, [r for r in samp.itertuples()]))
    rows = []
    for r in samp.itertuples():
        o = results.get(r.paper_id, {})
        m = {"id": r.paper_id, "y": int(r.judgement), "year": str(r.year)}
        for t in ("t1", "t3", "t4"):
            dd = o.get(t)
            m[f"{t}_support"] = dd["support"] if dd else np.nan
            m[f"{t}_echo"] = dd["echo"] if dd else np.nan
        # novelty = claims NOT already in earlier field corpus (inverse of t3 echo)
        m["novelty"] = (1 - m["t3_echo"]) if m["t3_echo"] == m["t3_echo"] else np.nan
        rows.append(m)
    M = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/outputs/tiered_peer", exist_ok=True)
    M.to_csv(f"{ROOT}/outputs/tiered_peer/tiered_metrics.csv", index=False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    y = M.y.values; g = M.year.values
    print("\n[peer] === availability by class (leak check) ===", flush=True)
    for t in ("t1", "t3", "t4"):
        print(f"  {t}: ", M.groupby("y")[f"{t}_support"].apply(lambda s: round(s.notna().mean(), 3)).to_dict(), flush=True)
    print("[peer] === per-metric univariate AUC (accept/reject) ===", flush=True)
    for c in [c for c in M.columns if c not in ("id", "y", "year")]:
        v = M[c].values.astype(float); mk = ~np.isnan(v)
        if mk.sum() > 80 and np.std(v[mk]) > 0:
            print(f"  {c:16} AUC={roc_auc_score(y[mk], v[mk]):.4f} (n={mk.sum()})", flush=True)
    feats = ["t1_support", "t1_echo", "t3_support", "t3_echo", "t4_support", "t4_echo", "novelty"]
    X = M[feats].values.astype(float)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(pipe, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                        groups=g, scoring="roc_auc").mean()
    print(f"\n[peer] TIERED BANK (year-grouped) AUC = {a:.4f}  (refs: ICLR V=0.611 A=0.676 V+A=0.682)", flush=True)
    print("TIERED_PEER_DONE", flush=True)

if __name__ == "__main__":
    main()

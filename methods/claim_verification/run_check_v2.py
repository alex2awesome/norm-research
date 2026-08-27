#!/usr/bin/env python3
"""Checker v2 for typed claims (failure mode 2 fix + patents prior-art analog):
  1. ADEQUACY check (every claim): ESTABLISHED / ASSERTED_ONLY / ABSENT vs paper internals
     (does the evidence establish the claim at its scope — not mere assertion-echo).
  2. PRIOR-ART check (novelty + performance + contribution claims): year-gated BM25 retrieval
     over earlier ICLR/NeurIPS/ICML abstracts -> localize-then-verify (patents flow):
     ANTICIPATED (prior work discloses substantially the same idea/result) / RELATED / CLEAR.
Outputs: outputs/checks_v2/checks.jsonl (claim level) + paper_metrics.csv (paper level).
Run on sk3 after run_claims_v2: python -m methods.claim_verification.run_check_v2"""
import json, os, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, _post, _parse_json, _key
from claim_verification.evidence_api import fts_terms
from claim_verification.run_tiered_peer import PaperDB
from claim_verification.run_adequacy_mode import ADEQ

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}
PA_TYPES = ("novelty", "performance", "contribution")

PRIOR_ART = """A {year} machine-learning paper makes this CLAIM:
CLAIM: {claim}

Below are abstracts of EARLIER papers (candidate prior art).

{cands}

For EACH candidate, judge whether it anticipates the claim:
- "ANTICIPATED": the earlier paper discloses substantially the same idea, method, or result
- "RELATED": overlaps (same problem, same ingredient) but not the same idea/result
- "CLEAR": unrelated to this claim

Return ONLY JSON: {{"judgments": [{{"idx": 1, "verdict": "ANTICIPATED"|"RELATED"|"CLEAR", "span": "<short quote from the candidate if ANTICIPATED or RELATED, else empty>"}}, ...]}}"""

class PriorFTS:
    """Year-gated candidates from both abstract FTS bases."""
    def __init__(self):
        import threading
        self._local = threading.local()
    def _cons(self):
        if getattr(self._local, "cons", None) is None:
            self._local.cons = [
                (sqlite3.connect(os.path.join(EB, "peer_abstracts.sqlite")),
                 "SELECT text, year FROM ab WHERE ab MATCH ? AND CAST(year AS REAL) < ? ORDER BY bm25(ab) LIMIT 4"),
                (sqlite3.connect(os.path.join(EB, "citation_abstracts.sqlite")),
                 "SELECT text, year FROM ab WHERE ab MATCH ? AND CAST(year AS REAL) < ? ORDER BY bm25(ab) LIMIT 4"),
            ]
        return self._local.cons
    def query(self, claim, year, k=6):
        q = fts_terms(claim)
        if not q: return []
        out, seen = [], set()
        for con, sql in self._cons():
            try: rows = con.execute(sql, (q, float(year))).fetchall()
            except sqlite3.OperationalError: rows = []
            for txt, yr in rows:
                key = txt[:80]
                if key not in seen:
                    seen.add(key); out.append((str(yr)[:4], txt[:900]))
        return out[:k]

def adeq_check(claim, pool, cache):
    k = _key("adeq", CFG["model"], claim, "|".join(pool)[:1500])
    hit = cache.get(k)
    if hit is not None: return hit
    ps = "\n---\n".join(p[:600] for p in pool[:12])
    raw = _post(CFG["base_url"], CFG["model"], ADEQ.format(claim=claim[:400], passages=ps),
                max_tokens=200)
    obj = _parse_json(raw) or {}
    v = str(obj.get("verdict", "")).upper()
    if v not in ("ESTABLISHED", "ASSERTED_ONLY", "ABSENT"): v = "PARSE_FAIL"
    out = {"verdict": v, "reason": str(obj.get("reason", ""))[:200]}
    cache.put(k, out)
    return out

def pa_check(claim, year, cands, cache):
    k = _key("pa", CFG["model"], claim, "|".join(c[1][:60] for c in cands))
    hit = cache.get(k)
    if hit is not None: return hit
    ctext = "\n\n".join(f"[{i+1}] ({y}) {t}" for i, (y, t) in enumerate(cands))
    raw = _post(CFG["base_url"], CFG["model"],
                PRIOR_ART.format(year=year, claim=claim[:400], cands=ctext), max_tokens=500)
    obj = _parse_json(raw) or {}
    js = []
    for j in (obj.get("judgments") or []):
        if isinstance(j, dict):
            v = str(j.get("verdict", "")).upper()
            if v in ("ANTICIPATED", "RELATED", "CLEAR"):
                js.append({"idx": j.get("idx"), "verdict": v, "span": str(j.get("span", ""))[:200]})
    sev = "CLEAR"
    if any(j["verdict"] == "ANTICIPATED" for j in js): sev = "ANTICIPATED"
    elif any(j["verdict"] == "RELATED" for j in js): sev = "RELATED"
    out = {"claim_verdict": sev, "judgments": js}
    cache.put(k, out)
    return out

def main():
    t = pd.read_csv(f"{ROOT}/outputs/tiered_peer/tiered_metrics.csv")
    t["id"] = t.id.astype(str)
    years = {r.id: int(float(r.year)) for r in t.itertuples()}
    claims = {}
    for ln in open(os.path.join(EB, "claims_v2_peer.jsonl")):
        r = json.loads(ln)
        if r.get("claims"): claims[str(r["doc_id"])] = r["claims"]
    print(f"[chk2] {len(claims)} papers, "
          f"{sum(len(v) for v in claims.values())} typed claims", flush=True)
    pdb = PaperDB(); pfts = PriorFTS()
    os.makedirs(f"{ROOT}/outputs/checks_v2", exist_ok=True)
    cache = Cache(f"{ROOT}/outputs/checks_v2/cache.jsonl")
    lock = Lock(); rows = []
    tasks = [(pid, c) for pid, cl in claims.items() for c in cl]
    def work(item):
        pid, c = item
        m = {"paper": pid, "claim": c["claim"], "type": c["type"]}
        pool = pdb.internals(pid.replace("iclr_", ""))
        try:
            m["adequacy"] = adeq_check(c["claim"], pool, cache)["verdict"] if pool else None
        except Exception:
            m["adequacy"] = None
        if c["type"] in PA_TYPES:
            cands = pfts.query(c["claim"], years.get(pid, 2024))
            if cands:
                try:
                    r = pa_check(c["claim"], years.get(pid, 2024), cands, cache)
                    m["prior_art"] = r["claim_verdict"]
                    m["pa_span"] = next((j["span"] for j in r["judgments"]
                                         if j["verdict"] == "ANTICIPATED" and j["span"]), "")
                except Exception:
                    m["prior_art"] = None
            else:
                m["prior_art"] = "NO_CANDS"
        with lock:
            rows.append(m)
            if len(rows) % 300 == 0: print(f"[chk2] {len(rows)}/{len(tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, tasks))
    with open(f"{ROOT}/outputs/checks_v2/checks.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    # paper-level metrics
    F = pd.DataFrame(rows)
    out = []
    for pid, g in F.groupby("paper"):
        m = {"id": pid, "n_claims": len(g)}
        ad = g[g.adequacy.isin(["ESTABLISHED", "ASSERTED_ONLY", "ABSENT"])]
        if len(ad):
            m["est_rate"] = (ad.adequacy == "ESTABLISHED").mean()
            m["asserted_only_rate"] = (ad.adequacy == "ASSERTED_ONLY").mean()
        for ty in ("performance", "novelty", "assumption", "scope", "design_justification"):
            sub = ad[ad.type == ty]
            if len(sub): m[f"est_{ty}"] = (sub.adequacy == "ESTABLISHED").mean()
        pa = g[g.prior_art.isin(["ANTICIPATED", "RELATED", "CLEAR"])]
        if len(pa):
            m["anticipated_rate"] = (pa.prior_art == "ANTICIPATED").mean()
            m["clear_rate"] = (pa.prior_art == "CLEAR").mean()
            nov = pa[pa.type == "novelty"]
            if len(nov): m["anticipated_novelty"] = (nov.prior_art == "ANTICIPATED").mean()
        out.append(m)
    P = pd.DataFrame(out)
    P.to_csv(f"{ROOT}/outputs/checks_v2/paper_metrics.csv", index=False)
    print(f"[chk2] adequacy dist: {F.adequacy.value_counts(normalize=True).round(3).to_dict()}", flush=True)
    print(f"[chk2] prior-art dist: {F.prior_art.value_counts(normalize=True).round(3).to_dict()}", flush=True)
    print(f"[chk2] adequacy by type:", flush=True)
    for ty, g in F[F.adequacy.notna()].groupby("type"):
        print(f"    {ty:20} n={len(g):4d} EST={(g.adequacy=='ESTABLISHED').mean():.3f} "
              f"AO={(g.adequacy=='ASSERTED_ONLY').mean():.3f}", flush=True)
    print("CHECKS_V2_DONE", flush=True)

if __name__ == "__main__":
    main()

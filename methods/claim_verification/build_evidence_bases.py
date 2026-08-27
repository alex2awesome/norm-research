#!/usr/bin/env python3
"""Build the per-domain evidence bases on sk3 (FActScore-style where applicable).

Bases built (all under datasets/evidence_bases/):
  wiki.sqlite       Wikipedia (wikimedia/wikipedia 20231101.en via HF) -> SQLite FTS5 over
                    (title, text). FActScore's base: retrieve passages for a claim's entity/topic.
                    ~6.4M articles; FTS5 gives BM25-ranked lexical retrieval (their GTR-dense
                    step is approximated lexically first; bge-m3 rerank can be added on top).
  news.sqlite       news_articles.csv (37.9M rows, ~68% with text) -> FTS5 over
                    (article_id, date, source, text). Same-story cross-outlet retrieval with
                    date windowing. THE news fact pool (T2/T3 for news + PR-coverage checking).
  pr_history.sqlite press releases indexed by (company, year) -> superlative/novelty checks
                    against the company's own prior releases (T3 for PR).
  peer review       REUSED as-is: peer_review_fullpaper_evidence.jsonl (paper internals).

Usage (sk3):  python -m methods.claim_verification.build_evidence_bases --base wiki|news|pr
Idempotent; each base skipped if its sqlite already exists (delete to rebuild).
"""
import argparse, csv, json, os, sqlite3, sys, time
csv.field_size_limit(sys.maxsize)

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT = os.path.join(ROOT, "datasets", "evidence_bases")
os.makedirs(OUT, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def build_news():
    db = os.path.join(OUT, "news.sqlite")
    if os.path.exists(db): log(f"news.sqlite exists, skip"); return
    con = sqlite3.connect(db + ".tmp")
    con.execute("PRAGMA journal_mode=OFF"); con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE VIRTUAL TABLE news USING fts5(article_id UNINDEXED, date UNINDEXED, source UNINDEXED, text)")
    n, kept = 0, 0
    batch = []
    src = os.path.join(ROOT, "datasets/press-releases/news_articles.csv")
    for r in csv.DictReader(open(src)):
        n += 1
        t = (r.get("news_article_text") or "").strip()
        if len(t) < 500: continue
        batch.append((r.get("news_article_id"), (r.get("news_article_date") or "")[:10],
                      r.get("news_article_source"), t[:20000]))
        kept += 1
        if len(batch) >= 5000:
            con.executemany("INSERT INTO news VALUES (?,?,?,?)", batch); batch = []
            if kept % 500000 == 0: log(f"news: {kept} indexed / {n} scanned")
    if batch: con.executemany("INSERT INTO news VALUES (?,?,?,?)", batch)
    con.commit(); con.close()
    os.rename(db + ".tmp", db)
    log(f"news.sqlite DONE: {kept} articles indexed / {n} scanned")

def build_pr_history():
    db = os.path.join(OUT, "pr_history.sqlite")
    if os.path.exists(db): log("pr_history.sqlite exists, skip"); return
    import pandas as pd
    d = pd.read_parquet(os.path.join(ROOT, "datasets/press-releases/press_release_deconfounded.parquet"))
    con = sqlite3.connect(db + ".tmp")
    con.execute("CREATE VIRTUAL TABLE pr USING fts5(pr_id UNINDEXED, company UNINDEXED, year UNINDEXED, text)")
    rows = [(str(r.id), str(r.company), str(r.year), str(r.text)[:20000]) for r in d.itertuples()]
    con.executemany("INSERT INTO pr VALUES (?,?,?,?)", rows)
    con.commit(); con.close(); os.rename(db + ".tmp", db)
    log(f"pr_history.sqlite DONE: {len(rows)} PRs")

def build_wiki():
    db = os.path.join(OUT, "wiki.sqlite")
    if os.path.exists(db): log("wiki.sqlite exists, skip"); return
    # download via HF datasets (streaming -> sqlite; avoids holding 11GB in RAM)
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    from datasets import load_dataset
    log("wiki: streaming wikimedia/wikipedia 20231101.en ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    con = sqlite3.connect(db + ".tmp")
    con.execute("PRAGMA journal_mode=OFF"); con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE VIRTUAL TABLE wiki USING fts5(title, text)")
    batch, n = [], 0
    for r in ds:
        batch.append((r["title"], r["text"][:24000])); n += 1
        if len(batch) >= 5000:
            con.executemany("INSERT INTO wiki VALUES (?,?)", batch); batch = []
            if n % 500000 == 0: log(f"wiki: {n} articles")
    if batch: con.executemany("INSERT INTO wiki VALUES (?,?)", batch)
    con.commit(); con.close(); os.rename(db + ".tmp", db)
    log(f"wiki.sqlite DONE: {n} articles")

# ---------------- retrieval API (used by verify pipelines) ----------------
def fts_query_terms(claim, max_terms=8):
    import re
    STOP = set("the a an and or of to in on for with as by at is are was were be been said says".split())
    toks = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+|\d[\d,.]*", claim) if w.lower() not in STOP]
    # prefer capitalized (entities) + numbers, then rest
    ents = [t for t in toks if t[0].isupper() or t[0].isdigit()]
    rest = [t for t in toks if t not in ents]
    terms = (ents + rest)[:max_terms]
    return " OR ".join('"' + t.replace('"', "") + '"' for t in terms)

class FTSBase:
    def __init__(self, path, table):
        self.con = sqlite3.connect(path); self.table = table
    def retrieve(self, claim, k=6, where=""):
        q = fts_query_terms(claim)
        if not q: return []
        sql = f"SELECT *, bm25({self.table}) AS rank FROM {self.table} WHERE {self.table} MATCH ? {where} ORDER BY rank LIMIT ?"
        try:
            return self.con.execute(sql, (q, k)).fetchall()
        except sqlite3.OperationalError:
            return []

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=["wiki", "news", "pr", "all"])
    a = ap.parse_args()
    if a.base in ("news", "all"): build_news()
    if a.base in ("pr", "all"): build_pr_history()
    if a.base in ("wiki", "all"): build_wiki()
    print("EVIDENCE_BASES_DONE", flush=True)

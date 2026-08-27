"""Unified, TEMPORALLY-SAFE evidence API over all bases (FActScore-aligned).

Every task can query EVERY base; each retrieve() takes as_of (ISO date or year) and only
returns evidence STRICTLY BEFORE that date:
  news.sqlite        WHERE date < as_of           (per-article dates)
  pr_history.sqlite  WHERE year < as_of_year      (company-scoped optional)
  wiki.sqlite        allowed only if as_of >= WIKI_SNAPSHOT (2023-11-01); else returns []
                     unless allow_timeless=True (caller accepts encyclopedic leak).
  peer internals     per-paper (no cross-doc time issue).

FActScore alignment:
  - passage-level retrieval: articles are chunked to ~100-word passages AT QUERY TIME
    (chunk_passages), BM25 doc-level first, then passage scoring by lexical containment.
  - evidence text is junk-stripped (nav boilerplate) before chunking.
"""
import os, re, sqlite3

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets", "evidence_bases")
WIKI_SNAPSHOT = "2023-11-01"

NAVJUNK = re.compile(
    r"(skip to (main )?content|log in|sign in|subscribe now?|today's paper|advertisement|"
    r"continue reading the main story|watchlist|newsletters?|follow us|share this|"
    r"cookie|privacy policy|terms of (use|service)|©|all rights reserved)", re.I)

def clean_evidence_text(t):
    """Strip nav boilerplate lines/segments from evidence article text."""
    segs = re.split(r"(?<=[.!?])\s+|\s{3,}| {2}", t or "")
    keep = [s for s in segs if len(s.split()) >= 5 and not NAVJUNK.search(s)]
    return " ".join(keep)

def chunk_passages(text, words_per=100, max_passages=40):
    ws = (text or "").split()
    out = []
    for i in range(0, min(len(ws), words_per * max_passages), words_per):
        out.append(" ".join(ws[i:i + words_per]))
    return out

_STOP = set("the a an and or but of to in on for with as by at is are was were be been it its this that from has have had not no said says".split())
def _toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2]
def _containment(claim_toks, passage):
    if not claim_toks: return 0.0
    ps = set(_toks(passage))
    return sum(1 for w in claim_toks if w in ps) / len(claim_toks)

def fts_terms(claim, max_terms=8):
    toks = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+|\d[\d,.]*", claim) if w.lower() not in _STOP]
    ents = [t for t in toks if t[0].isupper() or t[0].isdigit()]
    rest = [t for t in toks if t not in ents]
    return " OR ".join('"' + t.replace('"', "") + '"' for t in (ents + rest)[:max_terms])

import threading

class EvidenceAPI:
    def __init__(self):
        self._local = threading.local()
    def _con(self, name):
        cons = getattr(self._local, "cons", None)
        if cons is None:
            cons = self._local.cons = {}
        if name not in cons:
            cons[name] = sqlite3.connect(os.path.join(EB, f"{name}.sqlite"))
        return cons[name]

    def _passages_from_rows(self, claim, rows, text_idx, k_passages, provenance):
        ct = _toks(claim)
        scored = []
        for row in rows:
            txt = clean_evidence_text(row[text_idx])
            for p in chunk_passages(txt):
                scored.append((_containment(ct, p), p, provenance(row)))
        scored.sort(key=lambda x: -x[0])
        return [{"passage": p, "score": round(s, 3), "src": src}
                for s, p, src in scored[:k_passages] if s > 0.15]

    def news(self, claim, as_of=None, k_docs=6, k_passages=8):
        q = fts_terms(claim)
        if not q: return []
        where = f"AND date < '{str(as_of)[:10]}'" if as_of else ""
        try:
            rows = self._con("news").execute(
                f"SELECT article_id, date, source, text, bm25(news) FROM news "
                f"WHERE news MATCH ? {where} ORDER BY bm25(news) LIMIT ?", (q, k_docs)).fetchall()
        except sqlite3.OperationalError:
            return []
        return self._passages_from_rows(claim, rows, 3, k_passages,
                                        lambda r: f"news:{r[0]}@{r[1]}")

    def pr_history(self, claim, as_of_year=None, company=None, k_docs=6, k_passages=8):
        q = fts_terms(claim)
        if not q: return []
        where = ""
        if as_of_year: where += f" AND year < '{as_of_year}'"
        if company: where += f" AND company = '{str(company).replace(chr(39), '')}'"
        try:
            rows = self._con("pr_history").execute(
                f"SELECT pr_id, company, year, text, bm25(pr) FROM pr "
                f"WHERE pr MATCH ? {where} ORDER BY bm25(pr) LIMIT ?", (q, k_docs)).fetchall()
        except sqlite3.OperationalError:
            return []
        return self._passages_from_rows(claim, rows, 3, k_passages,
                                        lambda r: f"pr:{r[0]}/{r[1]}@{r[2]}")

    def wiki(self, claim, as_of=None, allow_timeless=False, k_docs=4, k_passages=8):
        if as_of and str(as_of)[:10] < WIKI_SNAPSHOT and not allow_timeless:
            return []   # temporal safety: snapshot postdates the document
        q = fts_terms(claim)
        if not q: return []
        try:
            rows = self._con("wiki").execute(
                "SELECT title, text, bm25(wiki) FROM wiki WHERE wiki MATCH ? "
                "ORDER BY bm25(wiki) LIMIT ?", (q, k_docs)).fetchall()
        except sqlite3.OperationalError:
            return []
        return self._passages_from_rows(claim, rows, 1, k_passages,
                                        lambda r: f"wiki:{r[0][:40]}")

    def all_tiers(self, claim, as_of=None, company=None, exclude_doc_id=None):
        """Combined pool with provenance tags; every task gets every base."""
        year = str(as_of)[:4] if as_of else None
        out = []
        out += self.news(claim, as_of=as_of)
        out += self.pr_history(claim, as_of_year=year, company=company)
        out += self.wiki(claim, as_of=as_of)
        if exclude_doc_id:
            out = [p for p in out if exclude_doc_id not in p["src"]]
        out.sort(key=lambda p: -p["score"])
        return out

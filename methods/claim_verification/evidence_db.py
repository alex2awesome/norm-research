"""External evidence database for cross-document claim checking (FactEval-style).

Sources (sk3, datasets/press-releases/):
  news_articles.csv            37.9M coverage articles, ~68% with full text (median 5.6k chars)
  press_release_news_mappings.csv  794k PR->article links (was_covered_by_news_article)

Builds a per-PR evidence set: the covering articles' texts, chunked into passages.
Cross-document metrics (per claim, via core.verify_claim against COVERAGE passages):
  x_support_rate    frac claims FULL-supported by at least one covering article
  x_echo_rate       frac claims at least PARTIAL (repeated) in coverage
  x_attribution     frac coverage-verdicts whose evidence_type is named_source_quote/document
NULL twin: verify against articles covering a DIFFERENT (same-topic) press release.

Usage:
  db = EvidenceDB(pr_dir)              # builds/loads the pr_id -> [article_id] index
  passages = db.coverage_passages(pr_id, max_articles=3, max_passages=12)
"""
import csv, os, sys, json
csv.field_size_limit(sys.maxsize)

class EvidenceDB:
    def __init__(self, pr_dir, cache_dir=None, needed_pr_ids=None):
        self.pr_dir = pr_dir
        self.map_path = os.path.join(pr_dir, "press_release_news_mappings.csv")
        self.art_path = os.path.join(pr_dir, "news_articles.csv")
        self.cache_dir = cache_dir or os.path.join(pr_dir, "evidence_db_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.needed = set(str(x) for x in needed_pr_ids) if needed_pr_ids else None
        self.pr2art = self._load_mapping()
        self.art_texts = {}          # article_id -> text, lazily loaded via _load_articles
        self._articles_loaded = False

    def _load_mapping(self):
        idx_path = os.path.join(self.cache_dir, "pr2art.json")
        if os.path.exists(idx_path) and self.needed is None:
            return json.load(open(idx_path))
        pr2art = {}
        for r in csv.DictReader(open(self.map_path)):
            if str(r.get("was_covered_by_news_article", "")).lower() != "true": continue
            pid = str(r["press_release_id"])
            if self.needed is not None and pid not in self.needed: continue
            pr2art.setdefault(pid, []).append(str(r["news_article_id"]))
        if self.needed is None:
            json.dump(pr2art, open(idx_path, "w"))
        return pr2art

    def _load_articles(self, needed_article_ids):
        """One pass over the 37.9M-row csv, keeping only needed ids (with text)."""
        needed = set(needed_article_ids) - set(self.art_texts)
        if not needed: return
        for r in csv.DictReader(open(self.art_path)):
            aid = str(r.get("news_article_id") or r.get("﻿news_article_id") or "")
            if aid in needed:
                t = (r.get("news_article_text") or "").strip()
                if len(t) > 500: self.art_texts[aid] = t
                needed.discard(aid)
                if not needed: break

    def preload(self, pr_ids, max_articles=3):
        """Batch-load article texts for a set of PRs (single csv pass)."""
        want = []
        for pid in pr_ids:
            want.extend(self.pr2art.get(str(pid), [])[:max_articles])
        self._load_articles(want)

    def coverage_passages(self, pr_id, max_articles=3, max_passages=12, max_chars=600):
        from .core import _sentences
        arts = self.pr2art.get(str(pr_id), [])[:max_articles]
        out = []
        per_art = max(1, max_passages // max(len(arts), 1)) if arts else 0
        for aid in arts:
            t = self.art_texts.get(aid)
            if not t: continue
            sents = _sentences(t)
            for i in range(0, min(len(sents), per_art * 3), 3):
                out.append(" ".join(sents[i:i + 3])[:max_chars])
                if len([p for p in out]) >= max_passages: break
        return out[:max_passages]

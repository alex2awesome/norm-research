#!/usr/bin/env python3
"""Build openalex_citations_v3.csv.gz: single-source (Semantic Scholar) acad-C.

Why v3
------
v2 mixed sources: NeurIPS-2023 was S2, every other cell OpenAlex. That left two
problems:
  1. cross-source inconsistency (one citation source per cell, three across the
     dataset);
  2. the Curran-stub leak survived in NeurIPS-2022 (and latent in 2022/23):
     OpenAlex's official proceedings records carry ~0 cites and no abstract,
     so the bottom quartile was "title-only zero-cite stub" and the top
     quartile was "arXiv twin with abstract" -> a TF-IDF floor of 0.989 that
     is really just an abstract-presence detector.

v3 fixes both: ALL rows come from S2 (cite_source='s2' everywhere), one
citation source throughout. Citation percentiles + top/bottom-quartile labels
are recomputed within every (venue x year) cell from S2 citationCount.

Membership: DBLP per-venue-year accepted-paper lists (authoritative, 29,228).
A DBLP row enters v3 iff it has an ACCEPTED S2 match (reuse cache + fetch
cache). Unmatched rows are omitted (no citation signal), and a cell whose match
coverage drops below 50% has its labels blanked (percentile over a biased
subsample is untrustworthy) -- mirrors join_and_build's guard.

Stable splits: split is a deterministic md5(dblp_key)%10 (0-7 train / 8 eval /
9 test). Keying on dblp_key (the paper's identity, invariant across the
source switch) keeps each paper's split stable from v1/v2 to v3 even though
the row `id` changes from a W-id to an S2 id.

ids: `https://www.semanticscholar.org/paper/{paperId}` (CorpusID fallback),
never colliding with OpenAlex W-ids.

Inputs : dblp_papers.csv (membership)
         s2_reuse_cache.jsonl (records reused from prior caches)
         s2_cache_full.jsonl  (newly fetched records)
Outputs: openalex_citations_v3.csv.gz   (v1 + v2 kept; never overwritten)
"""
import argparse
import hashlib
import json
import re
import unicodedata

import pandas as pd

HERE = "/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/openalex_citations"
DBLP = f"{HERE}/dblp_papers.csv"
REUSE = f"{HERE}/s2_reuse_cache.jsonl"
FETCH = f"{HERE}/s2_cache_full.jsonl"
OUT = f"{HERE}/openalex_citations_v3.csv.gz"

COVERAGE_FLOOR = 0.50


def norm_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-"))
    return s


def norm_title(s):
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_of(key):
    b = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 10
    return "train" if b <= 7 else ("eval" if b == 8 else "test")


def s2_id(rec):
    pid = rec.get("s2_paper_id")
    if pid:
        return f"https://www.semanticscholar.org/paper/{pid}"
    cid = rec.get("s2_corpus_id")
    return f"https://www.semanticscholar.org/CorpusID:{cid}" if cid else None


def s2_doi(rec):
    d = rec.get("doi")
    if d:
        return d if str(d).startswith("http") else f"https://doi.org/{d}"
    ax = rec.get("arxiv")
    return f"https://arxiv.org/abs/{ax}" if ax else None


def load_accepted_by_key(*paths):
    """key -> accepted record (last write wins, so fetch overrides reuse)."""
    d = {}
    for p in paths:
        try:
            f = open(p)
        except FileNotFoundError:
            continue
        with f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not r.get("accepted"):
                    continue
                d[str(r["key"])] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", default=FETCH,
                    help="newly-fetched cache (may not exist yet)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    dblp = pd.read_csv(DBLP)
    dblp["ntitle"] = dblp.title.map(norm_title)
    dblp = dblp.drop_duplicates(["venue", "year", "ntitle"]).copy()
    print(f"DBLP membership (deduped): {len(dblp)}")

    by_key = load_accepted_by_key(REUSE, args.fetch)
    print(f"accepted S2 records (reuse + fetch, by key): {len(by_key)}")

    rows = []
    seen_id = set()
    for m in dblp.itertuples():
        rec = by_key.get(str(m.dblp_key))
        if rec is None:
            continue
        rid = s2_id(rec)
        if rid is None or rid in seen_id:
            continue
        cites = rec.get("citationCount")
        if cites is None:
            continue
        seen_id.add(rid)
        title = norm_text(m.title)
        abstract = norm_text(rec.get("abstract") or "")
        text = (title + "\n\n" + abstract).strip()
        ms = rec.get("matchScore")
        rows.append({
            "id": rid,
            "text": text,
            "judgement": "",
            "percentile": float("nan"),
            "venue": str(m.venue),
            "year": int(m.year),
            "cited_by_count": int(cites),
            "record_kind": "s2_titlematch",
            "match_score": float(ms) if ms is not None else float("nan"),
            "doi": s2_doi(rec),
            "title": title,
            "has_abstract": bool(abstract.strip()),
            "cite_source": "s2",
            "dblp_key": str(m.dblp_key),
            "split": split_of(m.dblp_key),
        })
    df = pd.DataFrame(rows)
    print(f"v3 rows assembled: {len(df)}")

    # coverage per cell vs DBLP membership
    memb = dblp.groupby(["venue", "year"]).size()
    cov = (df.groupby(["venue", "year"]).size() / memb).fillna(0)

    # percentiles + quartile labels WITHIN each cell
    df["percentile"] = (df.groupby(["venue", "year"]).cited_by_count
                          .rank(pct=True, method="average"))
    df["judgement"] = ""
    df.loc[df.percentile >= 0.75, "judgement"] = "1"
    df.loc[df.percentile <= 0.25, "judgement"] = "0"

    low = cov[cov < COVERAGE_FLOOR].index
    if len(low):
        print(f"LOW-COVERAGE cells (labels blanked): "
              f"{[(v, y, round(cov[(v, y)], 2)) for v, y in low]}")
        mask = df.set_index(["venue", "year"]).index.isin(low)
        df.loc[mask, "judgement"] = ""
        df.loc[mask, "percentile"] = float("nan")

    cols = ["id", "text", "judgement", "percentile", "venue", "year",
            "cited_by_count", "record_kind", "match_score", "doi", "title",
            "has_abstract", "cite_source", "dblp_key", "split"]
    df = df[cols]
    df.to_csv(args.out, index=False, compression="gzip")
    print(f"\nwrote {len(df)} rows to {args.out}")

    print("\n=== coverage per venue/year (S2 matched / DBLP) ===")
    print(pd.DataFrame({"dblp": memb,
                        "matched": df.groupby(["venue", "year"]).size(),
                        "cov": cov.round(3)}).to_string())
    print("\n=== abstract coverage per venue/year ===")
    print(df.groupby(["venue", "year"]).has_abstract.mean().round(3).to_string())
    print(f"\noverall abstract coverage: {df.has_abstract.mean():.3f}")
    print("\n=== label counts ===")
    print(df.judgement.value_counts(dropna=False).to_string())
    print("\n=== cite_source ===")
    print(df.cite_source.value_counts().to_string())
    print("\n=== split counts ===")
    print(df.groupby("split").size().to_string())

    print("\n=== NeurIPS-2022 cell (the repair) ===")
    n22 = df[(df.venue == "NeurIPS") & (df.year == 2022)]
    c = n22.cited_by_count
    print(f"rows={len(n22)} median_cites={c.median()} mean={c.mean():.1f} "
          f"zero-frac={(c == 0).mean():.3f} abstract_cov={n22.has_abstract.mean():.3f}")
    topq = n22[n22.judgement == "1"].cited_by_count
    botq = n22[n22.judgement == "0"].cited_by_count
    if len(topq) and len(botq):
        print(f"top-q (y=1) median {topq.median()} min {topq.min()}; "
              f"bot-q (y=0) median {botq.median()} max {botq.max()}")
        print(f"abstract by label: y1 {n22[n22.judgement=='1'].has_abstract.mean():.3f} "
              f"y0 {n22[n22.judgement=='0'].has_abstract.mean():.3f}")


if __name__ == "__main__":
    main()

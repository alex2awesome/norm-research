#!/usr/bin/env python3
"""Build openalex_citations_v2.csv.gz: integrate the S2-sourced NeurIPS 2023 cell.

Single-source-per-cell rule
---------------------------
The label is a citation PERCENTILE within each (venue x year) cell. Mixing S2
and OpenAlex citation counts inside one cell would corrupt the percentile, so we
keep every cell single-source:
  * NeurIPS 2023 was BLANKED in the OpenAlex build (Curran proceedings stubs
    carry ~0 cites; the arXiv-twin repair ran out of OpenAlex budget). We
    REPLACE that whole cell with S2 records (s2_cache.jsonl) -> 100% S2.
  * Every other cell stays exactly as it was (OpenAlex). The 73-79%-covered
    ICLR/ICML 2022-23 cells are LEFT as OpenAlex-only on purpose.

Provenance is recorded in a new `cite_source` column ("openalex" | "s2").

Percentiles + quartile labels are recomputed ONLY for NeurIPS 2023, over its S2
rows. All other cells keep their existing judgement/percentile untouched.

Stable ids: OpenAlex rows keep their W-ids. New S2 rows get id
`https://www.semanticscholar.org/paper/{paperId}` (or CorpusID fallback) so the
downstream id->split hashing is stable and never collides with W-ids.

Inputs : openalex_citations.csv.gz (original, untouched)
         dblp_papers.csv (NeurIPS 2023 membership = canonical row set)
         s2_cache.jsonl (S2 match results for NeurIPS 2023)
Output : openalex_citations_v2.csv.gz   (original kept; never deleted)
"""
import json
import re
import unicodedata

import pandas as pd

ORIG = "openalex_citations.csv.gz"
DBLP = "dblp_papers.csv"
S2 = "s2_cache.jsonl"
OUT = "openalex_citations_v2.csv.gz"

TARGET_VENUE = "NeurIPS"
TARGET_YEAR = 2023


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


def main():
    orig = pd.read_csv(ORIG, dtype={"judgement": "object"})
    print(f"orig rows: {len(orig)}")
    orig["cite_source"] = "openalex"

    # --- isolate and drop the old NeurIPS 2023 cell (the 314 stub rows) ---
    is_n23 = (orig.venue == TARGET_VENUE) & (orig.year == TARGET_YEAR)
    print(f"dropping {int(is_n23.sum())} old OpenAlex NeurIPS-2023 stub rows")
    kept = orig[~is_n23].copy()

    # --- canonical membership for the cell ---
    dblp = pd.read_csv(DBLP)
    memb = dblp[(dblp.venue == TARGET_VENUE) & (dblp.year == TARGET_YEAR)].copy()
    memb["ntitle"] = memb.title.map(norm_title)
    memb = memb.drop_duplicates("ntitle")
    print(f"DBLP NeurIPS-2023 membership: {len(memb)}")

    # --- load accepted S2 matches, keyed by dblp_key ---
    s2_by_key = {}
    with open(S2) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("accepted"):
                continue
            if r.get("target_venue") != TARGET_VENUE or r.get("target_year") != TARGET_YEAR:
                continue
            s2_by_key[r["key"]] = r
    print(f"accepted S2 matches (NeurIPS 2023): {len(s2_by_key)}")

    # --- assemble S2 rows aligned to DBLP membership ---
    rows = []
    seen_id = set()
    matched = 0
    for m in memb.itertuples():
        rec = s2_by_key.get(str(m.dblp_key))
        if rec is None:
            continue  # unmatched membership row: omitted (no citation signal)
        rid = s2_id(rec)
        if rid is None or rid in seen_id:
            continue
        seen_id.add(rid)
        cites = rec.get("citationCount")
        if cites is None:
            continue
        title = norm_text(m.title)          # DBLP title (canonical text)
        abstract = norm_text(rec.get("abstract") or "")
        text = (title + "\n\n" + abstract).strip()
        ms = rec.get("matchScore")
        rows.append({
            "id": rid,
            "text": text,
            "judgement": "",                # set below
            "percentile": float("nan"),     # set below
            "venue": TARGET_VENUE,
            "year": TARGET_YEAR,
            "cited_by_count": int(cites),
            "record_kind": "s2_titlematch",
            "match_score": float(ms) if ms is not None else float("nan"),
            "doi": s2_doi(rec),
            "title": title,
            "has_abstract": bool(abstract.strip()),
            "cite_source": "s2",
        })
        matched += 1
    new = pd.DataFrame(rows)
    print(f"S2 NeurIPS-2023 rows built: {len(new)} "
          f"({len(new)}/{len(memb)} of membership = {len(new)/len(memb)*100:.1f}%)")

    # --- recompute percentile + quartile labels WITHIN the S2 cell only ---
    new["percentile"] = new.cited_by_count.rank(pct=True, method="average")
    new["judgement"] = ""
    new.loc[new.percentile >= 0.75, "judgement"] = "1"
    new.loc[new.percentile <= 0.25, "judgement"] = "0"

    # --- concat; preserve original column order, append cite_source ---
    cols = list(orig.columns)  # already includes cite_source at the end
    new = new[cols]
    out = pd.concat([kept[cols], new], ignore_index=True)

    out.to_csv(OUT, index=False, compression="gzip")
    print(f"\nwrote {len(out)} rows to {OUT}")

    # --- report ---
    print("\n=== NeurIPS 2023 (S2) citation distribution ===")
    c = new.cited_by_count
    print(f"n={len(new)}  min={c.min()} max={c.max()} median={c.median()} "
          f"mean={c.mean():.1f}  zero-frac={ (c==0).mean():.3f}")
    topq = new[new.judgement == "1"].cited_by_count
    botq = new[new.judgement == "0"].cited_by_count
    print(f"top-q (y=1): n={len(topq)} median={topq.median()} min={topq.min()}")
    print(f"bot-q (y=0): n={len(botq)} median={botq.median()} max={botq.max()}")
    print(f"quartile separation clean (min top-q >= max bot-q): "
          f"{topq.min() >= botq.max()}")
    print(f"abstract coverage (NeurIPS 2023 S2): {new.has_abstract.mean():.3f}")

    print("\n=== cite_source counts ===")
    print(out.cite_source.value_counts().to_string())
    print("\n=== label counts (v2 whole) ===")
    print(out.judgement.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

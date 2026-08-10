#!/usr/bin/env python3
"""Build best_papers_labels.csv from the joined awards table.

Applies the project-standard text normalization (NFKC, curly->straight
quotes -- identical to the y=0 pools and Task B) and assembles
text = TITLE\n\nABSTRACT. Award papers are y=1 (judgement=1) by construction;
y=0 candidates come from pools/{venue}.parquet (same venue x year, not an
award; see pull_venue_pools.py).
"""
import unicodedata

import pandas as pd


def norm_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-"))
    return s


def main():
    df = pd.read_csv("best_papers_joined.csv")
    df["title"] = df.title.map(norm_text)
    df["oa_title"] = df.oa_title.map(norm_text)
    df["abstract"] = df.abstract.map(norm_text)
    df["text"] = (df.title + "\n\n" + df.abstract).str.strip()
    df["judgement"] = 1
    df["has_abstract"] = df.abstract.str.len() > 0
    cols = ["venue", "field", "year", "title", "authors_raw", "openalex_id",
            "doi", "oa_title", "oa_year", "oa_type", "oa_source",
            "cited_by_count", "match_score", "matched", "has_abstract",
            "judgement", "text", "abstract"]
    df[cols].to_csv("best_papers_labels.csv", index=False)
    print(f"wrote {len(df)} rows to best_papers_labels.csv")
    print(f"matched: {df.matched.mean():.1%}; "
          f"abstract coverage (matched only): "
          f"{df.loc[df.matched, 'has_abstract'].mean():.1%}")


if __name__ == "__main__":
    main()

"""Extract peer-review SQLite DB into a unified labeled parquet."""
import sqlite3
import re
import os
import pandas as pd
import numpy as np

DB_PATH = "/tmp/peer_review.db"
OUT_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted"
OUT_PARQUET = os.path.join(OUT_DIR, "peer_review_unified.parquet")
SAMPLE_PARQUET = os.path.join(OUT_DIR, "sample_100.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

_LEAD_NUM_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")

def parse_num(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    m = _LEAD_NUM_RE.match(s)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except ValueError:
        return np.nan


def main():
    print(f"Connecting to {DB_PATH} ...")
    con = sqlite3.connect(DB_PATH)

    sql = """
    SELECT
        r.paper_id           AS paper_id,
        p.venue              AS venue_raw,
        p.venue_id           AS venue_id,
        p.year               AS year,
        p.title              AS title,
        p.abstract           AS abstract,
        p.decision           AS decision,
        CAST(r.review_id AS TEXT) AS review_id,
        r.review_text        AS review_text,
        r.score              AS score_raw,
        r.confidence         AS confidence_raw,
        r.recommendation     AS recommendation,
        COALESCE(r.is_meta_review, 0) AS is_meta_review_int
    FROM reviews r
    JOIN papers p USING (paper_id)
    WHERE r.review_text IS NOT NULL
      AND LENGTH(r.review_text) >= 50
    """

    print("Querying ...")
    df = pd.read_sql_query(sql, con)
    con.close()
    print(f"Raw rows after filter: {len(df):,}")

    # Venue normalization: OPENREVIEW with venue_id == TMLR -> TMLR
    df["venue"] = df["venue_raw"].astype(str)
    mask_tmlr = (df["venue"] == "OPENREVIEW") & (df["venue_id"].astype(str).str.upper() == "TMLR")
    df.loc[mask_tmlr, "venue"] = "TMLR"
    df.loc[df["venue"] == "NEURIPS", "venue"] = "NeurIPS"
    # Drop any non-TMLR OPENREVIEW rows (shouldn't happen, but be safe)
    df = df[df["venue"] != "OPENREVIEW"].copy()

    # Numeric parsing
    df["review_score"] = df["score_raw"].map(parse_num).astype("float64")
    df["confidence"] = df["confidence_raw"].map(parse_num).astype("float64")

    # Bool meta-review
    df["is_meta_review"] = df["is_meta_review_int"].astype(int).astype(bool)

    # Types
    df["paper_id"] = df["paper_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["title"] = df["title"].astype("string")
    df["abstract"] = df["abstract"].astype("string")
    df["decision"] = df["decision"].astype("string")
    df["recommendation"] = df["recommendation"].astype("string")
    df["review_text"] = df["review_text"].astype("string")

    cols = [
        "paper_id", "venue", "year", "title", "abstract", "decision",
        "review_id", "review_text", "review_score", "confidence",
        "recommendation", "is_meta_review",
    ]
    df = df[cols]

    print(f"Writing parquet: {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    size_mb = os.path.getsize(OUT_PARQUET) / (1024 * 1024)
    print(f"  -> {len(df):,} rows, {size_mb:.1f} MB")

    # Sample 100
    sample = df.sample(n=100, random_state=42)
    sample.to_parquet(SAMPLE_PARQUET, index=False, compression="zstd")
    print(f"Wrote sample: {SAMPLE_PARQUET} ({len(sample)} rows)")

    # ---------------- Stats ----------------
    print("\n===== Per-venue x year breakdown =====")
    g = (
        df.assign(rev_len=df["review_text"].str.len())
          .groupby(["venue", "year"], dropna=False)
          .agg(reviews=("review_id", "count"),
               median_len=("rev_len", "median"))
          .reset_index()
          .sort_values(["venue", "year"])
    )
    print(g.to_string(index=False))

    print("\n===== Decision distribution per venue =====")
    dec = (
        df.groupby(["venue", "decision"], dropna=False)
          .size()
          .reset_index(name="n")
          .sort_values(["venue", "n"], ascending=[True, False])
    )
    print(dec.to_string(index=False))

    print("\n===== Papers with >= 3 non-meta reviews =====")
    non_meta = df[~df["is_meta_review"]]
    rev_per_paper = non_meta.groupby(["venue", "paper_id"]).size()
    ge3 = (rev_per_paper >= 3).groupby(level=0).sum()
    total_papers = rev_per_paper.groupby(level=0).size()
    summary = pd.DataFrame({"papers": total_papers, "papers_ge3_reviews": ge3})
    summary["pct"] = (summary["papers_ge3_reviews"] / summary["papers"] * 100).round(1)
    print(summary.to_string())
    print(f"\nTotal papers with >=3 non-meta reviews: {int(summary['papers_ge3_reviews'].sum()):,}")

    print("\n===== Per-venue review-text length stats =====")
    ls = (
        df.assign(rev_len=df["review_text"].str.len())
          .groupby("venue")["rev_len"]
          .agg(["count", "min", "median", "mean", "max"])
          .round(1)
    )
    print(ls.to_string())

    print("\n===== Score parse coverage =====")
    sc = df.groupby("venue").agg(
        n=("review_id", "count"),
        score_nonnull=("review_score", lambda s: s.notna().sum()),
        conf_nonnull=("confidence", lambda s: s.notna().sum()),
    )
    sc["score_pct"] = (sc["score_nonnull"] / sc["n"] * 100).round(1)
    sc["conf_pct"] = (sc["conf_nonnull"] / sc["n"] * 100).round(1)
    print(sc.to_string())

    print("\n===== 2 example rows (review_text truncated to 400 chars) =====")
    ex = df.sample(n=2, random_state=7)
    for i, row in ex.iterrows():
        print(f"--- row idx={i} ---")
        for c in cols:
            val = row[c]
            if c == "review_text" and isinstance(val, str):
                val = val[:400] + ("..." if len(row[c]) > 400 else "")
            print(f"  {c}: {val}")

    print("\n===== Anomalies =====")
    empty_by_venue = df.groupby("venue")["review_text"].apply(lambda s: (s.str.len() < 200).sum())
    print("Reviews shorter than 200 chars per venue:")
    print(empty_by_venue.to_string())
    bad_score_venues = sc[sc["score_pct"] < 50]
    if len(bad_score_venues):
        print("\nVenues with <50% parsed score coverage:")
        print(bad_score_venues.to_string())


if __name__ == "__main__":
    main()

"""Build the acad-B (expert curation) task: oral/spotlight vs poster,
conditional on acceptance.

Derived from ../unified_papers.csv.gz `decision_raw` (tiers preserved there;
the canonical accept/reject splits deliberately flatten them — this is a NEW
task, the binary `judgement` of the main peer-review task is untouched).

Label: y=1 oral or spotlight, y=0 poster — a budget-bound (B1) selection by
the program committee, so the label is RELATIVE within venue x year; all
analyses must stratify on (venue, year).

Tier alias table (verified against decision_raw values 2026-06-12):
  oral      <- "Oral", "Talk", "notable-top-5%", "OralPoster" (ICML23)
  spotlight <- "Spotlight", "notable-top-25%", "spotlight poster" (ICML25)
  poster    <- "Poster"
  excluded  <- NeurIPS 2022 (no tiers: bare "Accept"), rejects, NA tiers

Split: stable md5 hash of paper_id (never seeded shuffle — growing lists
reshuffle; cf. v6a incident). 80/10/10.
"""
import gzip
import hashlib
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

def tier(s):
    if not isinstance(s, str):
        return None
    t = s.lower()
    if "oralposter" in t or "oral" in t or "talk" in t or "notable-top-5%" in t:
        return "oral"
    if "spotlight" in t or "notable-top-25%" in t:
        return "spotlight"
    if "poster" in t:
        return "poster"
    return None

def split_of(pid):
    h = int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else ("eval" if h == 8 else "test")

def main():
    df = pd.read_csv(HERE.parent / "unified_papers.csv.gz")
    df = df[df.venue_key.isin(["iclr", "neurips", "icml"])].copy()
    df["tier"] = df.decision_raw.map(tier)
    df = df[df.tier.notna()]
    df = df[df.title.notna() & df.abstract.notna()]
    df = df[df.abstract.str.len().between(200, 6000)]
    df["judgement"] = (df.tier != "poster").astype(int)
    df["text"] = ("TITLE: " + df.title.str.strip() + "\n\nABSTRACT: "
                  + df.abstract.str.strip())
    df["id"] = df.paper_id
    df["year"] = df.year.astype(int)
    df["split"] = df.paper_id.map(split_of)

    out_cols = ["id", "text", "judgement", "tier", "venue_key", "year", "split"]
    df = df[out_cols].drop_duplicates("id")

    print(df.groupby(["venue_key", "year", "tier"]).size().unstack(fill_value=0))
    print("\noverall:", df.judgement.value_counts().to_dict(),
          "| pos rate:", round(df.judgement.mean(), 3))
    print("splits:", df.split.value_counts().to_dict())

    for sp in ["train", "eval", "test"]:
        part = df[df.split == sp].drop(columns="split")
        part.to_csv(HERE / f"{sp}.csv.gz", index=False, compression="gzip")
        print(f"wrote {sp}: {len(part)} rows "
              f"(pos {part.judgement.sum()}, {100*part.judgement.mean():.1f}%)")

if __name__ == "__main__":
    main()

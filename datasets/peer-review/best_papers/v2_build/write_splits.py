#!/usr/bin/env python3
"""Write best_papers/built/best_papers_v2_{train,eval,test}.csv.gz from the
full build, and print final counts / balance / venue-year stats."""
import os

import pandas as pd

SRC = "best_papers_v2_full.csv.gz"
OUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/best_papers/built"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(SRC)
    cols = ["dblp_key", "venue", "year", "label", "title", "abstract", "text",
            "title_raw", "abstract_raw", "cited_by_count", "s2_paper_id",
            "s2_year", "split"]
    df = df[[c for c in cols if c in df.columns]]
    for sp in ["train", "eval", "test"]:
        g = df[df.split == sp]
        path = os.path.join(OUT, f"best_papers_v2_{sp}.csv.gz")
        g.to_csv(path, index=False, compression="gzip")
        print(f"{sp:5s}: {len(g):6d} rows  y=1 {int(g.label.sum())}  "
              f"y=0 {int((g.label==0).sum())}  -> {path}")
    nvy = df.groupby(["venue", "year"]).ngroups
    vy = df.groupby(["venue", "year"]).agg(pos=("label", "sum"),
                                           n=("label", "size")).reset_index()
    usable = vy[(vy.pos >= 1) & (vy.n - vy.pos >= 1)]
    print(f"\nTOTAL {len(df)} rows; {df.label.sum()} awards usable "
          f"(old usable=311); {nvy} venue-years; "
          f"{len(usable)} vy with both classes; "
          f"label balance {df.label.mean():.3f}")
    print("venues present:", sorted(df.venue.unique()))


if __name__ == "__main__":
    main()

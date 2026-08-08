#!/usr/bin/env python3
"""V7 step 1: per-patent forward-citation aggregates for the candidate universe
(US utility patents granted 2005-2015), from the raw PatentsView edge list.

Produces, per cited patent, DISTINCT-citing-patent counts of:
  total / examiner-added / applicant-added, each overall and within a fixed
  5-year post-grant window (window uses the CITING patent's grant date --
  `citation_date` is the CITED patent's date, verified in gt_fwdcites.json).
Also tallies citation_category coverage by CITING YEAR (era-dependence check).
Out: v7_cite_aggregates.parquet, v7_cat_by_year.json
"""
import csv, io, json, zipfile, collections
import numpy as np, pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant/"
OUTD = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/v7_community/"
import os; os.makedirs(OUTD, exist_ok=True)
Y0, Y1 = 2005, 2015
WIN = 5

print("loading g_patent ...", flush=True)
gyear = {}          # every granted patent -> grant year (needed for citing side)
uni_idx = {}        # candidate universe -> row index
uni_rows = []
with zipfile.ZipFile(BASE + "g_patent.tsv.zip") as z, \
     z.open(z.namelist()[0]) as fh:
    r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
    h = next(r)
    i_id, i_ty, i_dt, i_ti, i_nc, i_wd = (h.index("patent_id"), h.index("patent_type"),
        h.index("patent_date"), h.index("patent_title"), h.index("num_claims"), h.index("withdrawn"))
    for row in r:
        if len(row) <= i_wd: continue
        d = row[i_dt]
        if len(d) < 4 or not d[:4].isdigit(): continue
        y = int(d[:4]); pid = row[i_id]
        gyear[pid] = y
        if Y0 <= y <= Y1 and row[i_ty] == "utility" and row[i_wd] != "1":
            uni_idx[pid] = len(uni_rows)
            uni_rows.append((pid, d, row[i_ti], row[i_nc]))
print(f"  all granted: {len(gyear):,}; candidate universe (utility {Y0}-{Y1}): {len(uni_rows):,}", flush=True)

N = len(uni_rows)
uni_year = np.array([int(d[:4]) for _, d, _, _ in uni_rows], dtype=np.int16)
C = {k: np.zeros(N, dtype=np.int32) for k in
     ["tot", "exm", "app", "tot5", "exm5", "app5", "uncat", "uncat5"]}
cat_by_year = collections.defaultdict(collections.Counter)
dup_pairs = 0

print("scanning citations ...", flush=True)
n = 0
cur_citing, seen = None, set()
with zipfile.ZipFile(BASE + "g_us_patent_citation.tsv.zip") as z, \
     z.open(z.namelist()[0]) as fh:
    r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
    h = next(r)
    i_ci, i_cd, i_cat = h.index("patent_id"), h.index("citation_patent_id"), h.index("citation_category")
    for row in r:
        n += 1
        if len(row) <= i_cat: continue
        citing, cited, cat = row[i_ci], row[i_cd], row[i_cat]
        cy = gyear.get(citing)
        cat_by_year[cy][cat] += 1
        if citing != cur_citing:
            cur_citing, seen = citing, set()
        if cited in seen:          # same (citing, cited) pair twice -> count once
            dup_pairs += 1
            continue
        seen.add(cited)
        j = uni_idx.get(cited)
        if j is None: continue
        C["tot"][j] += 1
        inwin = cy is not None and 0 <= cy - uni_year[j] <= WIN
        if inwin: C["tot5"][j] += 1
        if cat == "cited by examiner":
            C["exm"][j] += 1
            if inwin: C["exm5"][j] += 1
        elif cat == "cited by applicant":
            C["app"][j] += 1
            if inwin: C["app5"][j] += 1
        else:
            C["uncat"][j] += 1
            if inwin: C["uncat5"][j] += 1
        if n % 30_000_000 == 0: print(f"  {n:,} edges", flush=True)
print(f"TOTAL {n:,} edges; {dup_pairs:,} duplicate (citing,cited) pairs collapsed", flush=True)

df = pd.DataFrame({"patent_id": [p for p, _, _, _ in uni_rows],
                   "grant_date": [d for _, d, _, _ in uni_rows],
                   "grant_year": uni_year,
                   "title": [t for _, _, t, _ in uni_rows],
                   "num_claims": [c for _, _, _, c in uni_rows],
                   **{k: v for k, v in C.items()}})
df.to_parquet(OUTD + "v7_cite_aggregates.parquet", index=False)
json.dump({str(k): dict(v) for k, v in sorted(cat_by_year.items(), key=lambda kv: (kv[0] is None, kv[0]))},
          open(OUTD + "v7_cat_by_year.json", "w"), indent=2)
print("wrote", OUTD + "v7_cite_aggregates.parquet", df.shape, flush=True)
print(df[["tot", "tot5", "exm5", "app5", "uncat5"]].describe().to_string())
print("\nzero-5y-citation share:", float((df.tot5 == 0).mean()))
print("\nby grant year:\n", df.groupby("grant_year")[["tot", "tot5", "exm5", "app5"]].mean().to_string())
print("V7_AGG_DONE", flush=True)

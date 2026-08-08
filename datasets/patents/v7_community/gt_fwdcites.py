#!/usr/bin/env python3
"""V7 MANDATORY GROUND TRUTH: verify forward_citations.parquet against the raw
PatentsView g_us_patent_citation edge list, by hand, for 20 patents.

Also establishes, from raw data (not assumption):
  * the semantics of `citation_date` (cited-patent grant date vs citing date)
  * the `citation_category` vocabulary (examiner- vs applicant-added)
  * whether a fixed post-grant citation WINDOW is computable (needs citing dates)
Writes gt_fwdcites.json.
"""
import csv, io, json, zipfile, collections, random, sys
import pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant/"
CIT = BASE + "g_us_patent_citation.tsv.zip"
OUT = "/lfs/skampere3/0/alexspan/tmp/gt_fwdcites.json"

# ---- 1. patent grant dates (needed: citation_date is NOT the citing date) ----
print("loading g_patent ...", flush=True)
pdate, ptype = {}, {}
with zipfile.ZipFile(BASE + "g_patent.tsv.zip") as z:
    with z.open(z.namelist()[0]) as fh:
        r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
        h = next(r); i_id, i_ty, i_dt = h.index("patent_id"), h.index("patent_type"), h.index("patent_date")
        for row in r:
            if len(row) > i_dt:
                pdate[row[i_id]] = row[i_dt]
                ptype[row[i_id]] = row[i_ty]
print(f"  g_patent: {len(pdate):,} patents", flush=True)

# ---- 2. pick 20 targets, stratified over the count distribution --------------
fc = pd.read_parquet("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/forward_citations.parquet")
fc["gyear"] = fc.patent_id.map(lambda p: int(pdate[p][:4]) if p in pdate and pdate[p][:4].isdigit() else -1)
pool = fc[(fc.gyear >= 2000) & (fc.gyear <= 2015)]
print(f"  parquet rows in grant-year 2000-2015: {len(pool):,}", flush=True)
rng = random.Random(20260808)
targets = {}
qs = [0, .2, .4, .6, .8, .95, .99, 1.0]
edges = pool.n_forward_cites.quantile(qs).tolist()
for a, b in zip(edges[:-1], edges[1:]):
    sub = pool[(pool.n_forward_cites >= a) & (pool.n_forward_cites <= b)]
    if len(sub) == 0: continue
    for pid in sub.patent_id.sample(3, random_state=42).tolist():
        targets[pid] = int(pool.set_index("patent_id").n_forward_cites.get(pid, -1))
# also 2 patents ABSENT from the parquet (should truly have 0 forward cites)
allp = [p for p, y in ((p, int(pdate[p][:4])) for p in list(pdate)[:400000]
                       if pdate[p][:4].isdigit()) if 2000 <= y <= 2015]
absent = [p for p in allp if p not in set(fc.patent_id)][:2]
for p in absent: targets[p] = 0
print(f"  {len(targets)} targets (incl {len(absent)} claimed-zero)", flush=True)

# ---- 3. ONE full scan: exact recount for targets + global diagnostics --------
tgt = set(targets)
recount = collections.Counter()
by_cat = collections.defaultdict(collections.Counter)
citing_years = collections.defaultdict(collections.Counter)
cat_global = collections.Counter()
date_check = {"match_cited_grant": 0, "match_citing_grant": 0, "checked": 0, "examples": []}
n = 0
with zipfile.ZipFile(CIT) as z:
    with z.open(z.namelist()[0]) as fh:
        r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
        h = next(r)
        i_citing, i_cited = h.index("patent_id"), h.index("citation_patent_id")
        i_date, i_cat = h.index("citation_date"), h.index("citation_category")
        for row in r:
            n += 1
            if len(row) <= i_cat: continue
            cat = row[i_cat]
            cat_global[cat] += 1
            cited = row[i_cited]
            if n <= 3000:   # date-semantics check on a sample
                cd, citing = row[i_date], row[i_citing]
                if cd:
                    date_check["checked"] += 1
                    if cited in pdate and pdate[cited][:7] == cd[:7]:
                        date_check["match_cited_grant"] += 1
                    if citing in pdate and pdate[citing][:7] == cd[:7]:
                        date_check["match_citing_grant"] += 1
                    if len(date_check["examples"]) < 5:
                        date_check["examples"].append(
                            {"citing": citing, "citing_grant": pdate.get(citing),
                             "cited": cited, "cited_grant": pdate.get(cited),
                             "citation_date": cd})
            if cited in tgt:
                recount[cited] += 1
                by_cat[cited][cat] += 1
                cy = pdate.get(row[i_citing], "")[:4]
                citing_years[cited][cy] += 1
            if n % 30_000_000 == 0:
                print(f"  {n:,} edges", flush=True)
print(f"TOTAL {n:,} edges", flush=True)

# ---- 4. verdicts -------------------------------------------------------------
rows = []
for pid, claimed in sorted(targets.items(), key=lambda kv: -kv[1]):
    got = recount[pid]
    gy = pdate.get(pid, "")[:4]
    cy = citing_years[pid]
    w5 = sum(v for k, v in cy.items() if k.isdigit() and gy.isdigit()
             and 0 <= int(k) - int(gy) <= 5)
    rows.append({"patent_id": pid, "grant_year": gy, "type": ptype.get(pid),
                 "parquet_count": claimed, "raw_recount": got, "match": claimed == got,
                 "by_category": dict(by_cat[pid]),
                 "n_in_5y_window": w5,
                 "citing_year_missing": int(sum(v for k, v in cy.items() if not k.isdigit()))})
res = {"n_edges": n, "citation_category_global": dict(cat_global),
       "citation_date_semantics": date_check,
       "n_targets": len(targets),
       "n_match": sum(r["match"] for r in rows), "targets": rows}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "targets"}, indent=2))
print(f"\nHAND CHECK: {res['n_match']}/{len(rows)} exact matches")
for r in rows:
    print(f"  {r['patent_id']:>9s} {r['grant_year']} parquet={r['parquet_count']:>5d} "
          f"raw={r['raw_recount']:>5d} {'OK ' if r['match'] else 'MISMATCH'} "
          f"5y={r['n_in_5y_window']:>4d} cats={r['by_category']}")
print("GT_DONE", flush=True)

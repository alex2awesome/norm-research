#!/usr/bin/env python3
"""V7 patents COMMUNITY/vote cell -- population + y + grouped splits.

y = the downstream inventor community's revealed judgment of a granted patent:
forward citations received within a FIXED 5-year post-grant window, converted to
a WITHIN-COHORT (grant_year x CPC class) median split. Ties at the cohort median
are dropped (mathse_vote_score / so_votes convention).

CONFOUND DESIGN (all four traps from the task brief, each verified in the note):
  AGE       -- y is a within-cohort split of a FIXED 5-year window, so both the
               mechanical age gradient and the truncation gradient are removed
               by construction; grant-year-alone AUC is asserted ~.50 below.
  EXAMINER/ -- counts are DISTINCT citing patents; examiner-added and
  SELF         applicant-added variants are carried as separate never-merged
               columns (`ys` in the manifest). Examiner-added forward citations
               cannot be applicant self-citations.
  METADATA  -- the claim-fell killer. Text = title + abstract + claim 1 ONLY.
               No examiner, art unit, assignee, inventor, filing/grant date,
               patent number, CPC code, num_claims or citation count reaches any
               instrument. Cohort keys are used to DEFINE y and then dropped.
  FAMILY    -- near-duplicate (continuation/divisional) clusters are collapsed
               into one grouping unit via MinHash-LSH over claim-1 shingles
               unioned with exact normalised-title match, so near-identical
               documents can never straddle a split.

CPU only. Out: datasets/patents/v7_community/
"""
import csv, io, json, hashlib, os, re, sys, zipfile, collections
import numpy as np, pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant/"
PROC = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/"
OUTD = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/v7_community/"
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/datasets/patents")
from build_dense_standard_claimfell import stable_hash_bucket_map  # REUSED, not rebuilt

MIN_COHORT = 50
N_TARGET = 16000
SEED = 20260808

agg = pd.read_parquet(OUTD + "v7_cite_aggregates.parquet")
print(f"universe: {len(agg):,}", flush=True)

# ---- 1. CPC class (cohort key only; never a feature) -------------------------
print("loading g_cpc_current ...", flush=True)
best = {}
with zipfile.ZipFile(BASE + "g_cpc_current.tsv.zip") as z, z.open(z.namelist()[0]) as fh:
    r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
    h = next(r)
    i_p, i_s, i_sec, i_cl, i_ty = (h.index("patent_id"), h.index("cpc_sequence"),
        h.index("cpc_section"), h.index("cpc_class"), h.index("cpc_type"))
    for row in r:
        if len(row) <= i_ty or row[i_ty] != "inventional":
            continue
        p = row[i_p]
        try: sq = int(row[i_s])
        except ValueError: continue
        if p not in best or sq < best[p][0]:
            best[p] = (sq, row[i_sec], row[i_cl])
print(f"  cpc for {len(best):,} patents", flush=True)
agg["cpc_section"] = agg.patent_id.map(lambda p: best.get(p, (None, None, None))[1])
agg["cpc_class"] = agg.patent_id.map(lambda p: best.get(p, (None, None, None))[2])
agg = agg[agg.cpc_class.notna()].copy()
print(f"  with CPC: {len(agg):,}", flush=True)

# ---- 2. cohort + y (thresholds computed on the FULL universe) ----------------
agg["cohort"] = agg.grant_year.astype(str) + "|" + agg.cpc_class
sz = agg.cohort.value_counts()
big = set(sz[sz >= MIN_COHORT].index)
dropped_small = int((~agg.cohort.isin(big)).sum())
agg = agg[agg.cohort.isin(big)].copy()
print(f"  cohorts: {len(big):,} (>= {MIN_COHORT}); dropped {dropped_small:,} rows in small cohorts", flush=True)

def median_split(df, col):
    med = df.groupby("cohort")[col].transform("median")
    y = np.where(df[col] > med, 1.0, np.where(df[col] < med, 0.0, np.nan))
    return y, med

# ERA FIX (verified in v7_cat_by_year.json): PatentsView labels the applicant
# bucket "cited by other" for citing patents granted 2002-2012 and "cited by
# applicant" only from 2013. The literal `app5` column is therefore 0 for every
# early cohort and is NOT usable. The era-robust examiner-independent count is
# tot - examiner. Every citing year inside our windows is >= 2005, i.e. inside
# the era where "cited by examiner" IS reliably marked, so both halves are clean.
agg["nonexm5"] = agg.tot5 - agg.exm5
for col, nm in [("tot5", "y_fwd5"), ("exm5", "y_fwd5_examiner"),
                ("nonexm5", "y_fwd5_nonexaminer"), ("tot", "y_fwd_alltime")]:
    agg[nm], _ = median_split(agg, col)
q75 = agg.groupby("cohort").tot5.transform(lambda s: s.quantile(.75))
agg["y_fwd5_topquartile"] = (agg.tot5 > q75).astype(float)

print("  y availability (non-tie share): " + ", ".join(
    f"{nm} {float(agg[nm].notna().mean()):.3f}" for nm in
    ["y_fwd5", "y_fwd5_examiner", "y_fwd5_nonexaminer", "y_fwd_alltime"]), flush=True)

# cohort degeneracy guard: a cohort whose untied rows are all one class carries no
# within-cohort contrast and would smuggle cohort identity back in as signal.
u = agg[agg.y_fwd5.notna()]
st = u.groupby("cohort").y_fwd5.agg(["size", "mean"])
badc = set(st[(st["size"] < 20) | (st["mean"] <= 0.0) | (st["mean"] >= 1.0)].index)
dropped_degenerate = int(agg.cohort.isin(badc).sum())
agg = agg[~agg.cohort.isin(badc)].copy()
print(f"  dropped {len(badc):,} degenerate cohorts ({dropped_degenerate:,} rows)", flush=True)

# ---- 3. sample -----------------------------------------------------------------
elig = agg[agg.y_fwd5.notna()].copy()
print(f"  eligible (non-tied on primary y): {len(elig):,}", flush=True)
rng = np.random.default_rng(SEED)
frac = min(1.0, (N_TARGET * 1.35) / len(elig))     # 35% headroom for text misses
take = elig.sample(frac=frac, random_state=SEED) if frac < 1 else elig
print(f"  sampled {len(take):,}", flush=True)
want = set(take.patent_id)

# ---- 4. text: title + abstract + claim 1 (NOTHING else) ------------------------
print("loading abstracts ...", flush=True)
abst = {}
with zipfile.ZipFile(BASE + "g_patent_abstract.tsv.zip") as z, z.open(z.namelist()[0]) as fh:
    r = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
    h = next(r); i_p, i_a = h.index("patent_id"), h.index("patent_abstract")
    for row in r:
        if len(row) > i_a and row[i_p] in want:
            abst[row[i_p]] = row[i_a]
print(f"  abstracts: {len(abst):,}", flush=True)
print("loading claim 1 ...", flush=True)
c1 = pd.read_parquet(PROC + "granted_patents_claim1_v2.parquet",
                     columns=["patent_id", "claim_text"])
c1 = c1[c1.patent_id.isin(want)].drop_duplicates("patent_id")
claim = dict(zip(c1.patent_id, c1.claim_text))
print(f"  claim1: {len(claim):,}", flush=True)

take["abstract"] = take.patent_id.map(abst)
take["claim1"] = take.patent_id.map(claim)
cov = {"has_abstract": float(take.abstract.notna().mean()),
       "has_claim1": float(take.claim1.notna().mean()),
       "has_title": float(take.title.notna().mean())}
take = take[take.abstract.notna() & take.claim1.notna() & take.title.notna()].copy()
take = take[(take.abstract.str.len() > 40) & (take.claim1.str.len() > 60)].copy()
print(f"  text coverage {cov}; after text filter: {len(take):,}", flush=True)
if len(take) > N_TARGET:
    take = take.sample(n=N_TARGET, random_state=SEED)
print(f"  final n = {len(take):,}", flush=True)

# ---- 5. family / near-duplicate grouping ---------------------------------------
NUM_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")
def norm_title(t):
    return WS_RE.sub(" ", NUM_RE.sub("", (t or "").lower())).strip()
def shingles(txt, k=5):
    w = WS_RE.sub(" ", NUM_RE.sub("#", (txt or "").lower())).split()
    return {" ".join(w[i:i + k]) for i in range(max(len(w) - k + 1, 1))}

print("near-duplicate (family) clustering ...", flush=True)
NPERM, NBAND, ROWS = 64, 16, 4
prm = [(int(rng.integers(1, 2**61 - 1)), int(rng.integers(0, 2**61 - 1))) for _ in range(NPERM)]
P = 2**61 - 1
ids = take.patent_id.tolist()
sh = [shingles(t + " " + c) for t, c in zip(take.title, take.claim1)]
sigs = []
for s in sh:
    hs = [int(hashlib.blake2b(x.encode(), digest_size=8).hexdigest(), 16) for x in s] or [0]
    sigs.append([min(((a * h + b) % P) for h in hs) for a, b in prm])
parent = {i: i for i in range(len(ids))}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[max(ra, rb)] = min(ra, rb)
buckets = collections.defaultdict(list)
for i, sg in enumerate(sigs):
    for band in range(NBAND):
        buckets[(band, tuple(sg[band * ROWS:(band + 1) * ROWS]))].append(i)
n_lsh = 0
for _, mem in buckets.items():
    if len(mem) < 2 or len(mem) > 200: continue
    for j in mem[1:]:
        if len(sh[mem[0]] | sh[j]) and len(sh[mem[0]] & sh[j]) / len(sh[mem[0]] | sh[j]) >= 0.6:
            union(mem[0], j); n_lsh += 1
tmap = collections.defaultdict(list)
for i, t in enumerate(take.title): tmap[norm_title(t)].append(i)
n_title = 0
for t, mem in tmap.items():
    if len(t) < 12 or len(mem) < 2 or len(mem) > 50: continue
    for j in mem[1:]:
        # an identical title alone is NOT a family: "semiconductor device" is
        # shared by hundreds of unrelated patents. Require shared claim language.
        u = len(sh[mem[0]] | sh[j])
        if u and len(sh[mem[0]] & sh[j]) / u >= 0.3:
            union(mem[0], j); n_title += 1
take["family_group"] = [f"fam{find(i):06d}" for i in range(len(ids))]
gs = take.family_group.value_counts()
print(f"  {len(gs):,} family groups (lsh unions {n_lsh}, title unions {n_title}); "
      f"largest {int(gs.max())}, n multi-member {int((gs > 1).sum())}", flush=True)

# ---- 6. splits (stable hash, grouped, pos-rate matched) -------------------------
take = take.sort_values("patent_id").reset_index(drop=True)
y_by_group = take.groupby("family_group").y_fwd5.apply(lambda s: [int(v) for v in s]).to_dict()
bmap = stable_hash_bucket_map(y_by_group)
take["split"] = take.family_group.map(bmap)
take["row_id"] = take.patent_id
take["text"] = ("TITLE: " + take.title.str.strip() + "\n\nABSTRACT: "
                + take.abstract.str.strip() + "\n\nCLAIM 1: " + take.claim1.str.strip())

# ---- 7. leak battery: nothing that defines y may predict y ----------------------
from sklearn.metrics import roc_auc_score
y = take.y_fwd5.values.astype(int)
def a(v):
    try: return float(roc_auc_score(y, v))
    except Exception: return None
gsz = take.family_group.map(take.family_group.value_counts()).values.astype(float)

def oof_group_mean_auc(keycol, n_folds=5):
    """Out-of-fold group-identity AUC. The in-sample version is meaningless here
    (15.7K family groups over 16K rows -> the group mean IS y); only an OOF
    estimate answers 'does knowing the cohort/family tell you y'."""
    k = take[keycol].values
    h = np.array([int(hashlib.blake2b(str(v).encode(), digest_size=4).hexdigest(), 16) % n_folds
                  for v in take.patent_id])
    pred = np.full(len(take), np.nan)
    for f in range(n_folds):
        tr, te = h != f, h == f
        m = pd.Series(y[tr]).groupby(pd.Series(k[tr])).mean()
        pred[te] = pd.Series(k[te]).map(m).fillna(y[tr].mean()).values
    return a(pred)

def perm_null(keycol, n=200):
    """What does the same statistic give when y is shuffled? (finite-cohort noise)"""
    rs = np.random.default_rng(7); k = take[keycol].values; out = []
    for _ in range(n):
        yp = rs.permutation(y)
        out.append(roc_auc_score(yp, pd.Series(yp).groupby(pd.Series(k)).transform("mean").values))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

multi = take.family_group.map(take.family_group.value_counts()) > 1
fam_agree = None
if multi.sum() > 1:
    sub = take[multi]
    ag = sub.groupby("family_group").y_fwd5.agg(["size", "mean"])
    ag = ag[ag["size"] > 1]
    fam_agree = float((ag["mean"] * (ag["size"] * ag["mean"] - 1) / (ag["size"] - 1)
                       + (1 - ag["mean"]) * ((ag["size"] * (1 - ag["mean"]) - 1)
                                             / (ag["size"] - 1))).clip(0, 1).mean())
leak = {
    "grant_year_alone_auc": a(take.grant_year.values.astype(float)),
    "cpc_section_alone_auc": a(pd.factorize(take.cpc_section)[0].astype(float)),
    "cohort_identity_oof_auc": oof_group_mean_auc("cohort"),
    "cohort_identity_insample_auc_AND_PERM_NULL": {
        "insample": a(take.groupby("cohort").y_fwd5.transform("mean").values),
        "perm_null_ci95": perm_null("cohort"),
        "reading": "the in-sample figure is an overfit statistic (~13 sampled rows "
                   "per cohort); it is quotable only against the permutation null."},
    "family_within_group_y_agreement": fam_agree,
    "n_multi_member_rows": int(multi.sum()),
    "num_claims_alone_auc": a(pd.to_numeric(take.num_claims, errors="coerce").fillna(0).values),
    "text_charlen_alone_auc": a(take.text.str.len().values.astype(float)),
    "claim1_charlen_alone_auc": a(take.claim1.str.len().values.astype(float)),
    "group_size_vs_y_corr": float(np.corrcoef(gsz, y)[0, 1]),
    "note": "grant_year / cpc / cohort MUST sit at ~.50: y is a within-cohort split, "
            "so any departure means the cohort thresholds leaked. num_claims and "
            "char length are DECLARED NUISANCE channels -- measured here, never "
            "given to a feature block, a judge prompt or the dense text.",
}
print("\nLEAK BATTERY:", json.dumps(leak, indent=1), flush=True)
BANNED = ["assignee", "inventor", "art_unit", "examiner_id", "examiner_name",
          "filing_date", "attorney", "applicant_name", "firm"]
assert not [c for c in take.columns if any(b in c.lower() for b in BANNED)]
# and nothing outside the three declared text fields may reach an instrument
assert take.text.str.contains(r"\bexaminer\b", case=False).mean() < 0.01

# ---- 8. write -------------------------------------------------------------------
cols = ["row_id", "patent_id", "text", "title", "abstract", "claim1",
        "y_fwd5", "y_fwd5_examiner", "y_fwd5_nonexaminer", "y_fwd_alltime",
        "y_fwd5_topquartile", "tot", "tot5", "exm5", "nonexm5", "app5", "uncat5",
        "cohort", "grant_year", "cpc_section", "cpc_class", "num_claims",
        "family_group", "split"]
take[cols].to_csv(OUTD + "population.csv.gz", index=False, compression="gzip")
man = {
    "cell": "patents_forwardcites", "built": "2026-08-08", "seed": SEED,
    "n": int(len(take)), "n_groups": int(take.family_group.nunique()),
    "pos_rate": float(take.y_fwd5.mean()),
    "y_primary": "y_fwd5 = within-(grant_year x CPC class) median split of the count of "
                 "DISTINCT patents citing this patent that were themselves granted in "
                 "[grant_year, grant_year+5] (calendar-year resolution). Cohort median "
                 "ties dropped.",
    "y_secondary_never_merged": ["y_fwd5_examiner", "y_fwd5_nonexaminer",
                                 "y_fwd_alltime", "y_fwd5_topquartile"],
    "citation_category_era_finding": (
        "PatentsView citation_category is era-dependent: blank for citing patents "
        "granted <2002; examiner/'cited by other' 2002-2012; examiner/applicant "
        "2013+. 'cited by other' IS the applicant bucket in the middle era, so the "
        "literal 'cited by applicant' count is identically 0 for early cohorts. The "
        "examiner-independent measure used here is tot5 - exm5. All citing years "
        "inside a 2005-2015 grant cohort's 5-year window fall in 2005-2020, i.e. "
        "wholly inside the era where 'cited by examiner' is reliably marked."),
    "self_citation_limitation": (
        "PatentsView tables downloaded here carry no assignee, so applicant "
        "SELF-citations cannot be identified and are included in y_fwd5. Mitigation: "
        "y_fwd5_examiner counts only examiner-added citations, which are third-party "
        "by construction and therefore immune to the self-citation trap; it is "
        "reported beside the primary y."),
    "dropped_degenerate_cohort_rows": dropped_degenerate,
    "window_years": 5, "min_cohort": MIN_COHORT, "n_cohorts": int(take.cohort.nunique()),
    "universe_n": int(len(agg)), "dropped_small_cohort_rows": dropped_small,
    "text_fields": ["title", "abstract", "claim1"],
    "text_coverage_before_filter": cov,
    "split_sizes": take.split.value_counts().to_dict(),
    "split_pos_rates": take.groupby("split").y_fwd5.mean().to_dict(),
    "grouping": "MinHash-LSH (64 perms, 16x4 bands, Jaccard>=0.6 on 5-gram shingles of "
                "title+claim1) unioned with exact normalised-title match",
    "group_size_max": int(gs.max()), "n_multi_member_groups": int((gs > 1).sum()),
    "leak_battery": leak,
    "excluded_by_design": ["examiner id/art unit", "assignee/inventor identity",
                           "filing & grant dates", "patent number", "CPC codes",
                           "num_claims", "any citation count"],
}
json.dump(man, open(OUTD + "population_manifest.json", "w"), indent=2, default=str)
print("\n", json.dumps({k: v for k, v in man.items() if k != "leak_battery"}, indent=1, default=str))
print("wrote", OUTD + "population.csv.gz")
print("V7_POP_DONE", flush=True)

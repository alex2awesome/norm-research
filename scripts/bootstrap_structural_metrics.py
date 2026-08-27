"""Honest variance estimation for structural metrics on the locked L0 clustering.

Problem: forms within a source page are correlated (one style guide emits dozens
of rubrics), so bootstrapping *forms* understates the variance of every
structural statistic (singleton rate, Zipf slope, concentration, ...).
Scrape waves also oversampled subtask types, so composition matters.

IMPORTANT: a with-replacement bootstrap is BIASED for these statistics --
duplicated draws of the same page artificially merge (a twice-drawn page turns
its singleton clusters into size-2 clusters), so singleton rate / compression /
Zipf slope shift far from the point estimate (verified: bootstrap CIs excluded
the point entirely). Richness statistics need duplication-free resampling.

Schemes (all duplication-free):
  form  -- half-subsample of forms without replacement, sd scaled by
           sqrt(m/n) (Politis-Romano). The WRONG independence unit; kept as
           the baseline to quantify how much form-level resampling understates.
  page  -- half-subsample of pages without replacement, same scaling (the
           honest within-composition default)
  strat -- half-subsample of pages within each derived subtask type
           (variance holding subtask composition exactly fixed)
  btype -- leave-one-subtask-type-out grouped jackknife: between-subtask
           variance ((G-1)/G * sum (theta_g - mean)^2)
  total sd = sqrt(strat^2 + btype^2)

Subsample spread is used only for VARIANCE; the subsample-level mean is biased
(half the corpus merges less) and is not reported as a level estimate. CIs are
point +/- 1.96 * total sd.

Subtask types: pages.parquet `subtask_short` is near-unique free text, so we
derive K coarse types per task by TF-IDF + k-means over
subtask_short + subtask_keywords. Top terms + samples per type are written to
subtask_types_<task>.txt for manual validation.

Statistics: identical definitions to sk3_structural_metrics.py concentration().

Outputs under outputs/analyses/structural_metrics/bootstrap_v1/:
  boot_<task>.json        -- per-stat mean, sd, 2.5/97.5 pct per scheme
  subtask_types_<task>.txt-- type inspection (top terms, sample subtask_shorts)
  summary.md              -- lean cross-task table
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
SM = ROOT / "outputs/analyses/structural_metrics"
OUT = SM / "bootstrap_v1"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]

B = 1000
SEED = 42
STATS = ["pct_singleton", "compression", "entropy_norm", "zipf_slope", "gini", "top10"]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2 * np.sum(idx * x) / (n * x.sum()) - (n + 1) / n)


def concentration(sizes, n_forms):
    sizes = np.array(sorted(sizes, reverse=True), dtype=float)
    n_cl = len(sizes)
    if n_cl < 3 or n_forms == 0:
        return None
    p = sizes / sizes.sum()
    H = -np.sum(p * np.log(p))
    rank = np.arange(1, n_cl + 1)
    slope = float(np.polyfit(np.log(rank), np.log(sizes), 1)[0])
    cs = np.cumsum(sizes)
    return dict(
        pct_singleton=float((sizes == 1).mean()),
        compression=n_forms / n_cl,
        entropy_norm=float(H / np.log(n_cl)),
        zipf_slope=slope,
        gini=gini(sizes),
        top10=float(cs[min(10, n_cl) - 1] / n_forms),
    )


def replicate_stats(pages_cids, chosen):
    cids = np.concatenate([pages_cids[i] for i in chosen])
    sizes = np.bincount(cids)
    sizes = sizes[sizes > 0]
    return concentration(sizes, len(cids))


def derive_subtask_types(pg_task, rng):
    """K coarse subtask types per task from free-text subtask fields.

    A single k-means pass leaves a giant catch-all type (>50% of pages on
    some tasks), which dominates the two-stage between-type variance. Any
    type holding >20% of pages is recursively re-clustered until no
    catch-all remains (or k_total hits 60).
    """
    txt = (pg_task.subtask_short.fillna("") + " " +
           pg_task.subtask_keywords.fillna("").astype(str)).str.lower()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, stop_words="english",
                          max_features=20000)
    X = vec.fit_transform(txt)
    terms = np.array(vec.get_feature_names_out())
    n = len(pg_task)
    k0 = int(min(25, max(5, n // 50)))
    km = MiniBatchKMeans(n_clusters=k0, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)

    next_label = labels.max() + 1
    for _ in range(6):
        sizes = np.bincount(labels)
        big = [c for c in range(len(sizes)) if sizes[c] > 0.20 * n]
        if not big or len(np.unique(labels)) >= 60:
            break
        for c in big:
            ix = np.where(labels == c)[0]
            ksub = min(8, max(2, int(np.ceil(sizes[c] / (0.10 * n)))))
            sub = MiniBatchKMeans(n_clusters=ksub, random_state=SEED,
                                  n_init=10).fit_predict(X[ix])
            labels[ix] = next_label + sub
            next_label += ksub

    # relabel densely and compute top terms per final type
    uniq = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.array([remap[u] for u in labels])
    top_terms = {}
    for c in range(len(uniq)):
        centroid = np.asarray(X[labels == c].mean(axis=0)).ravel()
        top_terms[c] = terms[np.argsort(-centroid)[:8]].tolist()
    return labels, top_terms, len(uniq)


def run_task(task, pg, rng):
    cl = json.load(open(SM / f"clusters_{task}.json"))
    # form -> page identity
    page_forms = defaultdict(list)  # (source_dir, source_file) -> [cluster_id]
    cid_map = {}
    for key, cid in cl.items():
        t, sd, sf, idx = key.split("::")
        if cid not in cid_map:
            cid_map[cid] = len(cid_map)
        page_forms[(sd, sf)].append(cid_map[cid])

    pg_task = pg[pg.task == task].copy()
    pg_task["pkey"] = list(zip(pg_task.source_dir, pg_task.source_file))
    pg_task = pg_task[pg_task.pkey.isin(page_forms)].drop_duplicates("pkey")
    labels, top_terms, k = derive_subtask_types(pg_task, rng)
    pg_task["stype"] = labels

    # inspection file for manual validation of the derived types
    with open(OUT / f"subtask_types_{task}.txt", "w") as f:
        for c in range(k):
            sub = pg_task[pg_task.stype == c]
            f.write(f"\n=== type {c} (n={len(sub)} pages) terms: {', '.join(top_terms[c])}\n")
            for s in sub.subtask_short.dropna().head(6):
                f.write(f"    {s[:110]}\n")

    pkeys = pg_task.pkey.tolist()
    pages_cids = [np.array(page_forms[p], dtype=np.int64) for p in pkeys]
    stypes = pg_task.stype.to_numpy()
    n_pages = len(pages_cids)
    all_cids = np.concatenate(pages_cids)
    n_forms = len(all_cids)

    point = concentration(np.bincount(all_cids)[np.bincount(all_cids) > 0], n_forms)

    strata = {s: np.where(stypes == s)[0] for s in np.unique(stypes)}
    schemes = {s: defaultdict(list) for s in ["form", "page", "strat"]}

    m_forms = n_forms // 2
    m_pages = n_pages // 2
    scale_form = np.sqrt(m_forms / n_forms)
    scale_page = np.sqrt(m_pages / n_pages)

    for _ in range(B):
        # (a) form half-subsample, no replacement (wrong unit, baseline)
        sizes = np.bincount(rng.choice(all_cids, size=m_forms, replace=False))
        r = concentration(sizes[sizes > 0], m_forms)
        if r:
            for s in STATS:
                schemes["form"][s].append(r[s])
        # (b) page half-subsample, no replacement
        r = replicate_stats(pages_cids,
                            rng.choice(n_pages, size=m_pages, replace=False))
        if r:
            for s in STATS:
                schemes["page"][s].append(r[s])
        # (c) page half-subsample within each subtask type
        chosen = np.concatenate([
            rng.choice(ix, size=max(1, len(ix) // 2), replace=False)
            for ix in strata.values()])
        r = replicate_stats(pages_cids, chosen)
        if r:
            for s in STATS:
                schemes["strat"][s].append(r[s])

    # (d) leave-one-subtask-type-out grouped jackknife (between-type variance)
    jack = defaultdict(list)
    for sx, ix in strata.items():
        keep = np.setdiff1d(np.arange(n_pages), ix)
        r = replicate_stats(pages_cids, keep)
        if r:
            for s in STATS:
                jack[s].append(r[s])
    G = len(strata)
    btype_sd = {s: float(np.sqrt((G - 1) / G * np.sum(
        (np.array(v) - np.mean(v)) ** 2))) for s, v in jack.items()}

    stype_sizes = np.bincount(stypes)
    res = {"task": task, "n_pages": n_pages, "n_forms": n_forms,
           "n_subtask_types": k,
           "max_stype_share": float(stype_sizes.max() / n_pages),
           "B": B, "point": point, "schemes": {}}
    for name, d in schemes.items():
        scale = scale_form if name == "form" else scale_page
        res["schemes"][name] = {
            s: dict(sub_mean=float(np.mean(v)),
                    sd=float(np.std(v) * scale)) for s, v in d.items()}
    res["schemes"]["btype"] = {s: dict(sd=btype_sd[s]) for s in STATS}
    res["schemes"]["total"] = {
        s: dict(sd=float(np.sqrt(res["schemes"]["strat"][s]["sd"] ** 2 +
                                 btype_sd[s] ** 2))) for s in STATS}
    json.dump(res, open(OUT / f"boot_{task}.json", "w"), indent=1)
    return res


def main():
    rng = np.random.default_rng(SEED)
    pg = pd.read_parquet(ROOT / "notebooks/_explore_cache/pages.parquet")
    rows = []
    for task in TASKS:
        res = run_task(task, pg, rng)
        rows.append(res)
        f = res["schemes"]
        print(f"{task}: pages={res['n_pages']} types={res['n_subtask_types']} | "
              f"singleton sd form={f['form']['pct_singleton']['sd']:.4f} "
              f"page={f['page']['pct_singleton']['sd']:.4f} "
              f"btype={f['btype']['pct_singleton']['sd']:.4f} "
              f"total={f['total']['pct_singleton']['sd']:.4f}", flush=True)

    with open(OUT / "summary.md", "w") as out:
        out.write("# Variance of structural metrics (subsampling + grouped "
                  "jackknife, B=%d)\n\n" % B)
        out.write("All schemes duplication-free (with-replacement bootstrap is "
                  "biased for richness stats). form = wrong unit baseline; "
                  "page = honest within-composition; strat = within subtask "
                  "type; btype = leave-one-type-out jackknife (between-subtask); "
                  "total = sqrt(strat^2+btype^2). CI = point +/- 1.96*total.\n")
        for stat in STATS:
            out.write(f"\n## {stat}\n\n")
            out.write("| task | point | form sd | page sd | strat sd | btype sd "
                      "| total sd | page/form | 95% CI |\n"
                      "|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                f = r["schemes"]
                pt = r["point"][stat]
                fs, ps = f["form"][stat]["sd"], f["page"][stat]["sd"]
                ss, bs = f["strat"][stat]["sd"], f["btype"][stat]["sd"]
                ts = f["total"][stat]["sd"]
                out.write(f"| {r['task']} | {pt:.3f} | {fs:.4f} | {ps:.4f} | "
                          f"{ss:.4f} | {bs:.4f} | {ts:.4f} | "
                          f"{ps/max(fs,1e-9):.1f}x | "
                          f"[{pt-1.96*ts:.3f}, {pt+1.96*ts:.3f}] |\n")
    print("\nWrote", OUT / "summary.md")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""V9 -- journalism tweet-engagement cell: population + y build.

THE CELL. Journalism/press field, VOTE/REVEALED (crowd) column of the 3xN
decomposition grid (notes/2026-08-08__vat-3xN-decomposition-grid.md), where it
is listed as "tweets UNLABELED (V9)". y = the crowd's revealed attention to a
news article, measured as Twitter engagement, ranked WITHIN the outlet-day the
article appeared on the outlet's homepage.

WHY THIS y
  y_engagement = 1 if the article's sum_likes is STRICTLY ABOVE the median
                   sum_likes of its own (outlet, first_day) group
                 0 if STRICTLY BELOW
                 undefined (dropped) if it TIES the median.

  This is a verbatim mirror of the two sibling VOTE cells' y -- math.SE
  (datasets/math-stackexchange/v2_va/population_manifest.json) and V6 SO votes
  ("strictly above the median answer Score on its OWN question, ties dropped").
  Using the same rule keeps the vote column commensurable ACROSS fields.

  The within-group framing is not cosmetic, it is the confound control:
    (a) Outlet is constant inside a group, so the enormous outlet-level
        audience-size offset (a Guardian article and a Reuters article do not
        compete for the same crowd) cannot drive the label. Contrast the
        sibling reddit-news arm, where the domain confound is so strong that
        per-domain P(y=1) ranges .008 (youtu.be) to .918 (abcnews.go.com) and
        an explicit propensity-decile deconfounding stage was required.
    (b) Day is constant inside a group, so news-cycle volume (a war week vs a
        quiet week) is differenced out, and topic is partially controlled --
        the competing items are literally the same day's news.
    (c) It matches the CURATION cell in this same grid row, whose y is
        within-snapshot homepage placement. Both are relative-within-a-page
        judgements, so the cross-y contrast runs on the same population with
        the same instruments.

SECONDARY y's carried but NEVER merged into the primary:
  * y_quartile   -- percentile >= .75 -> 1, <= .25 -> 0, middle dropped. The
                    reddit-news sibling's rule; kept as a binarization
                    robustness arm.
  * y_maxlikes   -- median split on max_likes instead of sum_likes. max_likes
                    is the single most-liked tweet, so it is much less
                    sensitive to the 100-tweet retrieval cap than a sum is.
                    This is the censoring robustness arm.

GROUND-TRUTH FINDINGS on the label channel, recorded here because none of them
is visible from the scraper code (the V8 "check the label channel first"
discipline; full detail in notes/2026-08-08__v9_journalism_community_build.md):

  1. 52,112 scraped rows = 47,199 OK + 4,913 errors (4,898 of them HTTP 404).
     The 404s are NOT informative-zero: a genuine "no tweets found" is recorded
     as a successful row with n_tweets=0 (1,393 of those exist). 404s look
     missing-at-random w.r.t. article prominence -- their `appearances`
     distribution and anchor_text length match the OK rows almost exactly --
     so they are DROPPED, not imputed as zero.
  2. ZERO duplicate URLs in the output. The scraper's resume set is keyed on
     url and the corpus was pre-deduped on scheme://netloc/path.
  3. THE CAP. 62.2% of OK rows carry capped=true. The cap is a retrieval limit
     (MAX_PAGES x 20 tweets), not a value ceiling: sum_likes is summed over at
     most ~100 retrieved tweets. It is uniform -- 717 of 731 touched groups cap
     at 100 and exactly ONE group mixes two cap ceilings -- so the censoring is
     a group-constant, which a within-group rank absorbs.
     The cap does NOT collapse the label: capped rate is 75.1% in the top
     within-group tercile vs 41.5% in the bottom, and among capped rows alone
     sum_likes still spans an IQR of [1602, 8056] and correlates rho=.926 with
     max_likes. So the ordering survives the censoring; it is compressed at the
     top, which is exactly the region a top-vs-bottom binarization discards.
  4. THE SEARCH IS type="Latest", NOT "Top". For a capped article we therefore
     hold the ~100 most RECENT tweets at scrape time (2026-06/07) about an
     article published 2025-12..2026-04. The measured quantity is honestly
     named "sustained/trailing Twitter attention", not "launch-day virality".
  5. The engagement facets agree strongly WITHIN group (median Spearman over
     615 groups with >=30 rows): sum_likes vs max_likes .912, vs sum_retweets
     .931, vs sum_views .800, vs sum_bookmarks .801. n_tweets is the outlier at
     .408 -- because it is the facet the cap truncates directly. A single
     latent attention dimension is being measured.
  6. COVERAGE IS THE GOOD NEWS. The scrape is only 8% of the 662,855-URL
     corpus, but that number is irrelevant to a within-group design: the
     scraper walked the corpus in (first_day, outlet) order, so it COMPLETED
     groups rather than sampling them. Median within-group coverage among
     touched groups is 1.000, and 602 groups sit at >=95% coverage.

REUSE (reuse-before-rebuild):
  * Engagement rows 100% reused -- tweet_engagement.jsonl, scraped 2026-06/07.
    The scraper is NOT restarted (it spends paid tweetapi quota).
  * V features 100% reused, imported not forked --
    datasets/news-homepages/va/v_features.py (23 headline features). The item
    here IS a news headline, the same object that bank was written for.
  * A bank 100% reused -- datasets/news-homepages/va/rubrics.jsonl (14
    GEPA-revised news-values criteria), authored for homepage headlines from
    these same outlets. Zero new criteria authored, mirroring V8.
  * Split bucketer 100% reused -- datasets/patents/build_dense_standard_claimfell.py.

POPULATION GATES (each one recorded in the manifest with its row cost):
  * group coverage >= 0.80 (OK rows / corpus rows in that outlet-day)
  * corpus group size >= 20 (so a within-group median is meaningful)
  * English-language outlets only. cnnbrasil is 43% of the raw rows and is
    Portuguese; it is EXCLUDED from the primary population and carried as a
    held-out replication arm, so that (i) the reused English news-values bank
    is scored on the language it was written for and (ii) the outlet set
    matches the homepage CURATION cell it will be contrasted against.
  * bbc dropped: 15 rows total, cannot support a group.

Usage (sk3):
  python3 datasets/journalism-tweets/build_tweets_population.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SALT = "journalism-tweets-v1|"
MIN_GROUP_COVERAGE = 0.80
MIN_CORPUS_GROUP = 20
MIN_OK_GROUP = 10
PRIMARY_OUTLETS = {"nytimes", "washingtonpost", "latimes", "guardian",
                   "cnn", "reuters"}
REPLICATION_OUTLETS = {"cnnbrasil"}


def sha1(s: str) -> str:
    return hashlib.sha1(str(s).encode()).hexdigest()


def load_bucketer(repo: Path):
    """Import stable_hash_bucket_map verbatim from the patents cell."""
    import importlib.util
    p = repo / "datasets/patents/build_dense_standard_claimfell.py"
    spec = importlib.util.spec_from_file_location("_claimfell", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stable_hash_bucket_map


def load_v_features(repo: Path):
    """Import the news-homepages 23-feature headline bank verbatim."""
    import importlib.util
    p = repo / "datasets/news-homepages/va/v_features.py"
    spec = importlib.util.spec_from_file_location("_vf_homepage", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/lfs/skampere3/0/alexspan/norm-research")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = Path(a.repo)
    base = repo / "datasets/news-homepages/twitter_engagement"
    outdir = Path(a.out) if a.out else repo / "datasets/journalism-tweets/va"
    outdir.mkdir(parents=True, exist_ok=True)

    audit: dict = {}

    # ---- denominators: the full homepage article-URL corpus ---------------
    den = Counter()
    for line in open(base / "urls_to_scrape.jsonl"):
        d = json.loads(line)
        den[(d["outlet"], d["first_day"])] += 1
    audit["corpus_urls"] = int(sum(den.values()))
    audit["corpus_groups"] = len(den)

    # ---- scraped engagement rows ------------------------------------------
    ok = defaultdict(list)
    n_err = 0
    err_kinds = Counter()
    for line in open(base / "tweet_engagement.jsonl"):
        d = json.loads(line)
        g = (d["outlet"], d["first_day"])
        if "error" in d:
            n_err += 1
            err_kinds[str(d["error"])[:24]] += 1
            continue
        ok[g].append(d)
    audit["scraped_ok"] = int(sum(len(v) for v in ok.values()))
    audit["scraped_error"] = n_err
    audit["error_kinds_top"] = err_kinds.most_common(4)
    audit["groups_touched"] = len(ok)

    # ---- gate the groups ---------------------------------------------------
    kept_groups, gate_log = [], Counter()
    for g, rs in ok.items():
        outlet = g[0]
        if outlet not in PRIMARY_OUTLETS:
            gate_log["outlet_not_primary"] += 1
            continue
        if den[g] < MIN_CORPUS_GROUP:
            gate_log["corpus_group_too_small"] += 1
            continue
        if len(rs) < MIN_OK_GROUP:
            gate_log["ok_rows_too_few"] += 1
            continue
        if len(rs) / den[g] < MIN_GROUP_COVERAGE:
            gate_log["coverage_below_gate"] += 1
            continue
        kept_groups.append(g)
        gate_log["KEPT"] += 1
    audit["group_gates"] = dict(gate_log)
    audit["coverage_of_kept"] = {
        "median": float(statistics.median(len(ok[g]) / den[g] for g in kept_groups)),
        "min": float(min(len(ok[g]) / den[g] for g in kept_groups)),
    }

    # ---- build rows with within-group y ------------------------------------
    vf = load_v_features(repo)
    rows = []
    tie_drops = Counter()
    for g in kept_groups:
        rs = ok[g]
        likes = np.array([r["sum_likes"] for r in rs], dtype=float)
        maxl = np.array([r["max_likes"] for r in rs], dtype=float)
        med = float(np.median(likes))
        med_max = float(np.median(maxl))
        # percentile rank (average method), used for the quartile arm
        order = pd.Series(likes).rank(pct=True, method="average").to_numpy()
        for i, r in enumerate(rs):
            if likes[i] > med:
                y = 1
            elif likes[i] < med:
                y = 0
            else:
                tie_drops["sum_likes_tie"] += 1
                continue
            if maxl[i] > med_max:
                ymax = 1
            elif maxl[i] < med_max:
                ymax = 0
            else:
                ymax = -1
            p = float(order[i])
            yq = 1 if p >= .75 else (0 if p <= .25 else -1)
            head = str(r.get("anchor_text") or "").strip()
            if len(head.split()) < 3:
                tie_drops["headline_too_short"] += 1
                continue
            rows.append(dict(
                row_id=sha1(r["url"])[:20],
                url=r["url"],
                outlet=r["outlet"],
                day=r["first_day"],
                group=f'{r["outlet"]}|{r["first_day"]}',
                text=f"HEADLINE: {head}",
                raw_headline=head,
                judgement=int(y),
                y_quartile=int(yq),
                y_maxlikes=int(ymax),
                pct_sum_likes=p,
                sum_likes=int(r["sum_likes"]),
                max_likes=int(r["max_likes"]),
                sum_retweets=int(r["sum_retweets"]),
                sum_views=int(r["sum_views"]),
                n_tweets=int(r["n_tweets"]),
                capped=bool(r.get("capped")),
                appearances=int(r.get("appearances") or 1),
                group_n=len(rs),
            ))
    audit["dropped_within_group"] = dict(tie_drops)

    df = pd.DataFrame(rows)
    # groups that lost both classes after tie-dropping are useless downstream
    good = df.groupby("group")["judgement"].nunique()
    df = df[df["group"].isin(good[good == 2].index)].reset_index(drop=True)

    # ---- V matrix (imported bank, headline half only) ----------------------
    v_names = list(vf.V_NAMES)
    V = np.array([vf.vector(vf.headline_of(t)) for t in df["text"]], dtype=float)
    for j, nm in enumerate(v_names):
        df[nm] = V[:, j]

    # ---- grouped stable-hash split (no seeded shuffle) ---------------------
    bucket = load_bucketer(repo)
    y_by_group = {g: sub["judgement"].tolist()
                  for g, sub in df.groupby("group")}
    bmap = bucket(y_by_group, targets={"train": .8, "eval": .1, "test": .1},
                  lam=2.5)
    df["split"] = df["group"].map(bmap)

    # ---- audit + write -----------------------------------------------------
    audit["n_rows"] = int(len(df))
    audit["n_groups"] = int(df["group"].nunique())
    audit["pos_rate"] = float(df["judgement"].mean())
    audit["outlets"] = df["outlet"].value_counts().to_dict()
    audit["day_range"] = [df["day"].min(), df["day"].max()]
    audit["group_size"] = {
        "median": float(df.groupby("group").size().median()),
        "min": int(df.groupby("group").size().min()),
        "max": int(df.groupby("group").size().max()),
    }
    audit["split_rows"] = df["split"].value_counts().to_dict()
    audit["split_pos_rate"] = df.groupby("split")["judgement"].mean().round(4).to_dict()
    audit["split_groups"] = df.groupby("split")["group"].nunique().to_dict()
    audit["capped_rate"] = float(df["capped"].mean())
    audit["capped_by_class"] = df.groupby("judgement")["capped"].mean().round(4).to_dict()
    audit["y_agreement"] = {
        "primary_vs_maxlikes": float((df.loc[df.y_maxlikes >= 0, "judgement"] ==
                                      df.loc[df.y_maxlikes >= 0, "y_maxlikes"]).mean()),
        "primary_vs_quartile": float((df.loc[df.y_quartile >= 0, "judgement"] ==
                                      df.loc[df.y_quartile >= 0, "y_quartile"]).mean()),
        "n_quartile_rows": int((df.y_quartile >= 0).sum()),
    }
    audit["v_names"] = v_names
    audit["built_utc"] = datetime.now(timezone.utc).isoformat()
    audit["gates"] = dict(min_group_coverage=MIN_GROUP_COVERAGE,
                          min_corpus_group=MIN_CORPUS_GROUP,
                          min_ok_group=MIN_OK_GROUP,
                          primary_outlets=sorted(PRIMARY_OUTLETS),
                          replication_outlets=sorted(REPLICATION_OUTLETS))
    audit["y_definition"] = (
        "1 = sum_likes strictly ABOVE the median sum_likes of its own "
        "(outlet, first_day) homepage group; 0 = strictly BELOW; ties dropped. "
        "Mirrors the math.SE and V6 SO-votes vote-cell rule.")

    df.to_csv(outdir / "population.csv.gz", index=False, compression="gzip")
    (outdir / "population_manifest.json").write_text(
        json.dumps(audit, indent=2, default=str))

    print(json.dumps({k: v for k, v in audit.items() if k != "v_names"},
                     indent=2, default=str))
    print(f"\nwrote {outdir/'population.csv.gz'}  rows={len(df)} "
          f"groups={df['group'].nunique()}")


if __name__ == "__main__":
    main()

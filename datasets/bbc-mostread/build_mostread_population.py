#!/usr/bin/env python3
"""BBC most-read cell (journalism community #2): population + y build.

THE CELL. Journalism/press field, VOTE/REVEALED (crowd) column -- the SAME-OUTLET
READERSHIP companion to V9 (datasets/journalism-tweets), which measures
cross-platform amplification. The user directive behind this build: test whether
V9's community signal reflects Twitter platform dynamics or generalises to a
same-outlet readership list. Same field, same item type (a news headline), same
instruments; the only thing that changes is WHICH crowd and HOW it acts --
BBC readers clicking BBC articles vs Twitter users amplifying links.

y = the headline appeared in the BBC News home page's ranked MOST READ module
    (top 10) on that capture; 0 = it appeared elsewhere on the same capture.

Note what makes this contrast clean: the negatives are links from the SAME page
capture, so the comparison is "what readers chose to read" against "what the
editors put in front of them on the same page at the same moment". The taxonomy
note's standing worry about this y -- "lists reflect placement" -- is therefore
controlled by construction rather than needing a placement adjustment.

GROUPING mirrors V9 exactly. V9 groups by outlet x day; BBC is a single outlet,
so outlet x day collapses to DAY. Stable-hash grouped splits over days, no
seeded shuffle.

=============================================================================
GROUND-TRUTH PASS ON THE LABEL CHANNEL (the V8 lesson; the charge requires the
list-capture timing/coverage be verified against the raw captures BEFORE any
instrument work). The prior build at datasets/news-homepages/bbc_mostread/built
is REUSED as the row source, but it is NOT usable as shipped. Three defects,
none visible from its code, all found by counting:

  DEFECT 1 -- CAPTURE TYPE LARGELY DETERMINES THE LABEL. The shipped 82,891-row
  pool mixes four capture strata, and in three of them y is nearly constant:

      parser        n        pos rate
      morph      51,790        .4400
      popular_page 7,520      1.0000     <- every row positive
      react       6,488        .9553     <- 95.5% positive
      (none)     17,093        .0000     <- every row negative

  i.e. 24,613 of 82,891 rows (29.7%) come from strata that supply essentially
  only one class. Any era/lexical signature of those captures is free AUC, which
  is what inflates the shipped manifest's length-matched TF-IDF floor to
  .720 eval / .711 test.

  DEFECT 2 -- THE LENGTH CONFOUND IS AN ARTEFACT OF DEFECT 1. The manifest
  records len_label_corr .2163 and pos/neg mean length 48.8 / 44.3, which reads
  like a real length effect. Within `morph` alone, pos and neg lengths are
  44.83 and 44.76 -- identical. The whole apparent length signal comes from
  popular_page (52.1) and react (59.3) being all-positive strata.

  DEFECT 3 -- THE SHIPPED SPLITS ARE NOT DAY-GROUPED. 3,343 of 3,421 days appear
  in more than one of train/eval/test, so the same news day's story cluster is
  split across fit and evaluation. (Rows themselves are unique -- 82,891 distinct
  headline_ids, no duplicated text -- so this is leakage of the DAY, not of rows.)

THE REPAIR, and why it is a filter rather than a rebuild (reuse-before-rebuild):
restrict to `parser == "morph"`, the one stratum carrying both classes at a
sane rate with no length asymmetry, and re-split day-grouped. The row content
itself is sound -- verified below -- so nothing needs re-scraping or re-parsing.

VERIFICATION AGAINST THE RAW CAPTURES (datasets/news-homepages/bbc_mostread/
raw/captures.jsonl, 7,298 Wayback captures):
  * All 51,790 morph rows re-derive their label from raw: label-matches-raw
    51,790, MISMATCH 0, not-found 0.
  * All 22,787 positive rows re-derive their `rank` exactly: 22,787 match, 0
    mismatched.
  * `most_read` and `others` are disjoint within a capture (0 hrefs in both), so
    the positive/negative assignment is unambiguous.
  * Retention from raw is deliberate downsampling, recorded rather than hidden:
    77.3% of raw morph positives and 33.7% of raw morph negatives survive into
    the pool (the negatives were subsampled to reach a ~44% positive rate).

TIMING -- a real property of this label that constrains what it means. Wayback
captured the BBC home page overwhelmingly just after midnight UTC (hour 00:
39,645 rows; hour 01: 7,017). The most-read module is a rolling window, so a
row labelled day D reports reading that happened mostly on day D-1. The label is
"was among the 10 most-read as of the small hours of day D". 693 of 2,256 days
carry 2 captures, the rest 1.

REUSE (nothing here is re-authored):
  * Rows + labels: datasets/news-homepages/bbc_mostread/built/{train,eval,test}
    .csv.gz, filtered to parser==morph.
  * V features: datasets/news-homepages/va/v_features.py, imported verbatim --
    the same 23 headline features V9 uses.
  * A bank: datasets/news-homepages/va/rubrics.jsonl, the same 14 GEPA-revised
    news-values criteria V9 reused. Population-exact: BBC is one of the outlets
    that bank was authored on, and the item is a headline.
  * Split bucketer: datasets/patents/build_dense_standard_claimfell.py.

Usage (sk3):
  python3 datasets/bbc-mostread/build_mostread_population.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MIN_DAY_ROWS = 10          # a day must support a within-day contrast
KEEP_PARSER = "morph"


def sha1(s: str) -> str:
    return hashlib.sha1(str(s).encode()).hexdigest()


def _load(repo: Path, rel: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, repo / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/lfs/skampere3/0/alexspan/norm-research")
    a = ap.parse_args()
    repo = Path(a.repo)
    src = repo / "datasets/news-homepages/bbc_mostread/built"
    outdir = repo / "datasets/bbc-mostread/va"
    outdir.mkdir(parents=True, exist_ok=True)

    audit: dict = {}
    raw = pd.concat([pd.read_csv(src / f"{s}.csv.gz").assign(orig_split=s)
                     for s in ["train", "eval", "test"]], ignore_index=True)
    audit["shipped_pool"] = {
        "n": int(len(raw)),
        "pos_rate": float(raw.judgement.mean()),
        "by_parser": {str(k): {"n": int(v), "pos_rate": round(float(
            raw[raw.parser.fillna("__none__") == k].judgement.mean()), 4)}
            for k, v in raw.parser.fillna("__none__").value_counts().items()},
        "days_in_more_than_one_shipped_split": int(
            (raw.groupby("day")["orig_split"].nunique() > 1).sum()),
        "n_days_shipped": int(raw.day.nunique()),
    }

    df = raw[raw.parser == KEEP_PARSER].copy()
    audit["after_parser_filter"] = {"n": int(len(df)),
                                    "pos_rate": float(df.judgement.mean()),
                                    "dropped": int(len(raw) - len(df))}

    # --- DEFECT 4: a small link-KIND stratum where the href alone fixes y ----
    # Same failure pattern as DEFECT 1, two orders smaller, and only visible by
    # bucketing the href. Of 51,790 morph rows, 1,184 (2.3%) are not ordinary
    # news articles, and in those buckets the label is near-deterministic:
    #   sport        280 rows  pos rate 1.000
    #   live         104 rows  pos rate 1.000
    #   news-other   232 rows  pos rate 0.931
    #   in-pictures  568 rows  pos rate 0.079
    # A photo gallery or a live page is a different KIND of object from an
    # article, so keeping them lets the instruments learn "is this an article at
    # all" instead of "did readers choose this article". Restrict to ordinary
    # dated news articles.
    kind = df.href.astype(str).str.extract(
        r"(/live/|/in-pictures|/in_pictures|/av/|/sport/|/newsround|/reel)",
        expand=False)
    is_article = kind.isna() & df.href.astype(str).str.match(r"^/news/.*\d{6,}$")
    audit["link_kind_gate"] = {
        "dropped_non_article": int((~is_article).sum()),
        "kept": int(is_article.sum()),
        "dropped_pos_rate": (float(df[~is_article].judgement.mean())
                             if (~is_article).sum() else None)}
    df = df[is_article].copy()

    # --- gates ------------------------------------------------------------
    df["group"] = df.day.astype(str)
    sz = df.groupby("group").size()
    both = df.groupby("group").judgement.nunique() == 2
    keep = set(sz[sz >= MIN_DAY_ROWS].index) & set(both[both].index)
    gate = Counter()
    gate["day_too_small"] = int((sz < MIN_DAY_ROWS).sum())
    gate["day_single_class"] = int((~both).sum())
    gate["KEPT"] = len(keep)
    df = df[df.group.isin(keep)].reset_index(drop=True)
    audit["day_gates"] = dict(gate)

    df["row_id"] = [sha1(f"{t}|{h}")[:20] for t, h in zip(df.timestamp, df.href)]
    assert df.row_id.is_unique, "row_id collision"
    df["raw_headline"] = df.text.astype(str).str.strip()
    df["text"] = "HEADLINE: " + df.raw_headline

    # --- V features (imported bank, identical to V9) -----------------------
    vf = _load(repo, "datasets/news-homepages/va/v_features.py", "vf_homepage")
    v_names = list(vf.V_NAMES)
    V = np.array([vf.vector(vf.headline_of(t)) for t in df["text"]], dtype=float)
    for j, nm in enumerate(v_names):
        df[nm] = V[:, j]

    # --- grouped stable-hash split (repairing DEFECT 3) --------------------
    bucket = _load(repo, "datasets/patents/build_dense_standard_claimfell.py",
                   "_claimfell").stable_hash_bucket_map
    y_by_group = {g: sub["judgement"].tolist() for g, sub in df.groupby("group")}
    bmap = bucket(y_by_group, targets={"train": .8, "eval": .1, "test": .1},
                  lam=2.5)
    df["split"] = df["group"].map(bmap)
    assert df.groupby("group")["split"].nunique().max() == 1, "day split across buckets"

    # --- cross-corpus overlap (for a same-rows contrast where it exists) ---
    ov = {}
    try:
        hp = pd.read_csv(repo / "datasets/news-homepages/va/population.csv.gz")
        hp_h = set(vf.headline_of(t) for t in hp[hp.outlet == "bbc"].text)
        ov["homepage_curation_bbc_rows"] = int((hp.outlet == "bbc").sum())
        ov["headline_overlap_with_homepage_bbc"] = int(
            len(set(df.raw_headline) & hp_h))
    except Exception as e:
        ov["homepage_error"] = str(e)[:120]
    try:
        tw = pd.read_csv(repo / "datasets/journalism-tweets/va/population.csv.gz")
        ov["v9_tweets_rows_total"] = int(len(tw))
        ov["v9_tweets_bbc_rows"] = int((tw.outlet == "bbc").sum())
        ov["headline_overlap_with_v9_tweets"] = int(
            len(set(df.raw_headline) & set(tw.raw_headline.astype(str))))
    except Exception as e:
        ov["tweets_error"] = str(e)[:120]
    audit["cross_corpus_overlap"] = ov

    # --- audit -------------------------------------------------------------
    audit["n_rows"] = int(len(df))
    audit["n_groups"] = int(df.group.nunique())
    audit["pos_rate"] = float(df.judgement.mean())
    audit["group_size"] = {"median": float(df.groupby("group").size().median()),
                           "min": int(df.groupby("group").size().min()),
                           "max": int(df.groupby("group").size().max())}
    audit["within_day_pos_rate_mean"] = float(
        df.groupby("group").judgement.mean().mean())
    audit["day_range"] = [str(df.day.min()), str(df.day.max())]
    audit["years"] = {str(k): int(v) for k, v in
                      df.day.astype(str).str[:4].value_counts().sort_index().items()}
    audit["section"] = df.section.value_counts().head(5).to_dict()
    audit["split_rows"] = df.split.value_counts().to_dict()
    audit["split_groups"] = df.groupby("split").group.nunique().to_dict()
    audit["split_pos_rate"] = df.groupby("split").judgement.mean().round(4).to_dict()
    audit["train_minority_count"] = int(min(
        (df[df.split == "train"].judgement == 0).sum(),
        (df[df.split == "train"].judgement == 1).sum()))
    audit["headline_len_by_class"] = df.assign(
        n=df.raw_headline.str.len()).groupby("judgement").n.mean().round(2).to_dict()
    audit["rank_dist_positives"] = {
        str(int(k)): int(v) for k, v in
        df[df.judgement == 1]["rank"].value_counts().sort_index().items()}
    audit["v_names"] = v_names
    audit["built_utc"] = datetime.now(timezone.utc).isoformat()
    audit["y_definition"] = (
        "1 = headline appeared in the BBC News home page ranked MOST READ module "
        "(top 10) on that Wayback capture; 0 = headline appeared elsewhere on the "
        "SAME capture. Group = capture day. Restricted to parser=='morph'.")

    df.to_csv(outdir / "population.csv.gz", index=False, compression="gzip")
    (outdir / "population_manifest.json").write_text(
        json.dumps(audit, indent=2, default=str))
    print(json.dumps({k: v for k, v in audit.items() if k != "v_names"},
                     indent=2, default=str))
    print(f"\nwrote {outdir/'population.csv.gz'} rows={len(df)} "
          f"groups={df.group.nunique()}")


if __name__ == "__main__":
    main()

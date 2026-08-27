"""
Build the forum-topic <-> wiki-problem join table by text-matching forum first
posts (the problem statement repost) against wiki problem statements.

Why text matching: neither the wiki dump (164 forum links across 4,937 problem
pages) nor the crawled topics (no title/source field in the ajax payload)
carries an explicit id-level mapping. The priority topic ids were harvested
from community Contest Collection categories, whose per-problem structure was
not persisted. First posts of contest-collection threads are verbatim (or
near-verbatim) copies of the problem statement, so high-threshold char-ngram
TF-IDF matching recovers the mapping.

Inputs (laptop):
  wiki/problems.parquet
  forum_priority_posts.jsonl.gz   (pulled from sk3 priority_posts.jsonl.gz)
  wiki/wiki_forum_links.csv       (direct links: used to validate matching)

Output:
  wiki/forum_wiki_join.parquet  columns:
    topic_id, contest, year, variant, problem_n, match_sim, match_margin,
    match_kind ("text" | "text_secondary")
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parent.parent
KEY = ["contest", "year", "variant", "problem_n"]

SIM_HIGH = 0.75            # accept best match outright at/above this cosine
SIM_LOW = 0.55             # in [SIM_LOW, SIM_HIGH) require numeric-literal
NUM_OVERLAP_MIN = 0.5      # overlap >= this (true reposts share their numbers
                           # and answer choices; coincidental phrasing matches
                           # do not). Calibrated by eyeballing the band.
SECONDARY_WINDOW = 0.02    # also accept 2nd/3rd-best within this of best
                           # (AMC10/12 share problems -> one thread, 2 pages)
SECONDARY_MIN_SIM = 0.85   # ...but only when the tie happens at high sim;
                           # near-threshold ties are spurious (distinct geo
                           # problems share phrasing)


def norm_text(s: str) -> str:
    """Markup-robust normalization for wiki wikitext vs forum BBCode."""
    if not s:
        return ""
    t = s
    # legacy AoPS BBCode operator escapes
    t = (t.replace("\\minus{}", "-").replace("\\equal{}", "=")
          .replace("\\plus{}", "+").replace("\\times{}", "\\times")
          .replace("\\div{}", "\\div"))
    # wiki HTML comments / transclusion tags
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"</?(?:onlyinclude|includeonly|noinclude)>", " ", t, flags=re.I)
    # forum BBCode
    t = re.sub(r"\[asy\].*?\[/asy\]", " ", t, flags=re.S | re.I)
    t = re.sub(r"\[/?(?:b|i|u|url|img|quote|hide|list|size|color|center)[^\]]*\]",
               " ", t, flags=re.I)
    # wiki markup
    t = re.sub(r"<asy>.*?</asy>", " ", t, flags=re.S | re.I)
    t = re.sub(r"</?(?:i?math|cmath|br|center|pre|nowiki)\s*/?>", " ", t, flags=re.I)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)   # [[a|b]] -> b
    t = re.sub(r"\{\{[^}]*\}\}", " ", t)
    t = re.sub(r"\[ihtml\].*?\[/ihtml\]", " ", t, flags=re.S | re.I)
    # latex delimiters (keep the latex body itself)
    t = re.sub(r"\$\$?|\\\[|\\\]|\\\(|\\\)", " ", t)
    # image attachments / urls
    t = re.sub(r"https?://\S+", " ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\\^_{}+=<>().,/-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


NUM_PAT = re.compile(r"\d+(?:\.\d+)?")


def num_overlap(a: str, b: str) -> float:
    """Jaccard-on-smaller-set of numeric literals in two normalized texts."""
    na, nb = set(NUM_PAT.findall(a)), set(NUM_PAT.findall(b))
    if not na or not nb:
        return 0.0
    return len(na & nb) / min(len(na), len(nb))


def main() -> int:
    probs = pd.read_parquet(REPO / "wiki" / "problems.parquet")
    posts = pd.read_json(REPO / "forum_priority_posts.jsonl.gz", lines=True)

    first = (posts[posts.post_number == 1]
             .drop_duplicates("topic_id")
             .reset_index(drop=True))
    first["norm"] = first.post_canonical.map(norm_text)
    probs = probs.copy()
    probs["norm"] = probs.problem_statement.map(norm_text)

    p_ok = probs[(probs.norm.str.len() >= 30) & (~probs.is_redirect.fillna(False))
                 ].reset_index(drop=True)
    f_ok = first[first.norm.str.len() >= 30].reset_index(drop=True)
    print(f"wiki problems vectorized: {len(p_ok)}/{len(probs)}   "
          f"forum first posts vectorized: {len(f_ok)}/{len(first)}")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          sublinear_tf=True)
    Xp = normalize(vec.fit_transform(p_ok.norm))
    Xf = normalize(vec.transform(f_ok.norm))

    S = Xf @ Xp.T  # sparse cosine sims (5.7K x 4.8K)
    S = S.tocsr()

    rows = []
    n_band = n_band_pass = 0
    for i in range(S.shape[0]):
        r = S.getrow(i)
        if r.nnz == 0:
            continue
        idx, vals = r.indices, r.data
        order = np.argsort(-vals)
        best = vals[order[0]]
        if best < SIM_LOW:
            continue
        margin = best - (vals[order[1]] if len(order) > 1 else 0.0)
        for rank, o in enumerate(order[:4]):
            if vals[o] < best - SECONDARY_WINDOW or vals[o] < SIM_LOW:
                break
            if rank > 0 and vals[o] < SECONDARY_MIN_SIM:
                break
            j = idx[o]
            nov = num_overlap(f_ok.norm.iloc[i], p_ok.norm.iloc[j])
            if vals[o] < SIM_HIGH:
                n_band += 1
                if nov < NUM_OVERLAP_MIN:
                    if rank == 0:
                        break
                    continue
                n_band_pass += 1
            rows.append({
                "topic_id": int(f_ok.topic_id.iloc[i]),
                **{k: p_ok[k].iloc[j] for k in KEY},
                "match_sim": float(vals[o]),
                "match_margin": float(margin),
                "num_overlap": nov,
                "match_kind": "text" if rank == 0 else "text_secondary",
            })
    join = pd.DataFrame(rows)
    print(f"band [{SIM_LOW},{SIM_HIGH}): {n_band} candidates, "
          f"{n_band_pass} pass numeric-overlap gate")

    # a wiki problem may attract several topics (collections have dup threads);
    # keep them all but flag the best topic per problem
    join["best_for_problem"] = (
        join.groupby(KEY).match_sim.transform("max") == join.match_sim)

    # --- validation against the 57 direct wiki->topic links ----------------
    links = pd.read_csv(REPO / "wiki" / "wiki_forum_links.csv")
    direct = links[links.link_kind == "topic"].rename(columns={"link_id": "topic_id"})
    v = direct.merge(join[["topic_id"] + KEY], on="topic_id", how="inner",
                     suffixes=("_wiki", "_match"))
    if len(v):
        agree = ((v.contest_wiki == v.contest_match) & (v.year_wiki == v.year_match)
                 & (v.problem_n_wiki == v.problem_n_match))
        print(f"validation vs direct wiki links: {agree.sum()}/{len(v)} agree "
              f"({len(direct)} direct links, {len(v)} with a text match)")
        if (~agree).any():
            print(v[~agree].to_string())

    out = REPO / "wiki" / "forum_wiki_join.parquet"
    join.to_parquet(out, index=False)

    # --- coverage report ----------------------------------------------------
    print(f"\njoin rows: {len(join)}  unique topics matched: {join.topic_id.nunique()}"
          f"/{len(f_ok)} first posts")
    matched_probs = join[KEY].drop_duplicates()
    cov = probs[KEY].merge(matched_probs.assign(m=1), on=KEY, how="left")
    print(f"wiki problems with >=1 matched crawled thread: "
          f"{int(cov.m.fillna(0).sum())}/{len(probs)} "
          f"({cov.m.fillna(0).mean():.1%})")
    print(cov.assign(m=cov.m.fillna(0)).groupby("contest").m
          .agg(["mean", "sum", "count"]).round(3).to_string())
    print(f"\nmatch_sim distribution: {join.match_sim.describe().round(3).to_dict()}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

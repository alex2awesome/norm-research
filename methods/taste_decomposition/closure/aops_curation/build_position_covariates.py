#!/usr/bin/env python3
"""Recover the OBSERVED position-in-container covariates for the AoPS curation
cell (FREEZE ADDENDUM 4), rather than trusting a text fingerprint.

The A/V population carries only row_id / problem / statement / body / judgement /
dense_prob / dense_split.  The raw crawl behind it,
`datasets/math/aops/forum_solutions.parquet` (32,120 posts), carries the TRUE
thread ordinals and timestamps:

    post_number      1-based position of the post inside its AoPS topic thread
    post_time        epoch seconds of the post
    topic_id         the thread
    num_edits        how many times the poster edited it
    topic_num_views  thread traffic                        [TOPIC-LEVEL]
    poster_id        the author
    thanks_received  crowd approval  -> NEIGHBOURING OUTCOME, diagnostic only
    match_sim        the upstream similarity score used to build y -> NEVER a
                     covariate in any readout; carried only so the note can say
                     what it is and that it was not used

JOIN.  `problem` = f"{variant}#{problem_n}"; the population's `body` is the
crawl's `body_noquote` up to whitespace normalisation.  Key =
sha1(problem + "|" + normalised body).  Measured coverage: 5,202 / 5,202 = 100%,
and the key is asserted unique on the population side.  The 488 duplicate bodies
that `build_va_population.py` dropped are the reason the crawl side is
deduplicated by first occurrence in (topic_id, post_number) order.

DERIVED, all from observed quantities:
    sol_rank            0-based rank of this row's post_number among the
                        POPULATION rows of the same problem (the container the
                        closure cell actually groups on)
    is_first_solution   sol_rank == 0
    position_pct        sol_rank / (n_sols_group - 1)
    n_sols_group        population rows for this problem      [GROUP-LEVEL]
    n_posts_topic       posts in the whole topic thread       [TOPIC-LEVEL]
    thread_age_days     (post_time - first post_time in the topic) / 86400 --
                        "problem-thread age at the time of posting", the second
                        position axis the brief asks for
    contest_year        parsed from the problem key (e.g. 1959_IMO#3 -> 1959)
    post_year           calendar year of the post
    years_after_contest post_year - contest_year -- how long after the contest
                        this solution was written
    poster_n_posts      posts by this author across the whole crawl (an UPSTREAM
                        author-standing proxy, kept separate from the position
                        family in every model)

Nothing here is ever added to V or A, judged by any LLM, or fitted into anything
that feeds the closure curve.  These columns enter the discount readouts only.

CPU only.  Usage: python3 build_position_covariates.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AOPS = REPO / "datasets" / "math" / "aops"
POP = AOPS / "va" / "population.csv.gz"
CRAWL = AOPS / "forum_solutions.parquet"
OUT = HERE / "aops_curation_position_covariates.csv"

WS = re.compile(r"\s+")
YEAR = re.compile(r"^(\d{4})_")

NUM_COLS = ["post_number", "sol_rank", "n_sols_group", "position_pct", "n_posts_topic",
            "thread_age_days", "post_year", "contest_year", "years_after_contest",
            "num_edits", "topic_num_views", "poster_n_posts", "thanks_received",
            "nothanks_received", "match_sim"]
STR_COLS = ["topic_id", "poster_id", "username", "contest", "match_kind"]


def norm(s):
    return WS.sub(" ", str(s)).strip()


def key(problem, body):
    return hashlib.sha1((str(problem) + "|" + norm(body)).encode("utf-8")).hexdigest()


def main():
    csv.field_size_limit(10 ** 9)
    with gzip.open(POP, "rt") as fh:
        pop = pd.DataFrame(list(csv.DictReader(fh)))
    pop["k"] = [key(p, b) for p, b in zip(pop["problem"], pop["body"])]
    assert pop["k"].is_unique, "population join key is not unique"

    f = pd.read_parquet(CRAWL)
    f["problem"] = f["variant"].astype(str) + "#" + f["problem_n"].astype(str)
    f["k"] = [key(p, b) for p, b in zip(f["problem"], f["body_noquote"])]
    f["post_number"] = pd.to_numeric(f["post_number"], errors="coerce")
    # post_time is datetime64[ns] in the crawl; carry it as epoch seconds
    f["post_time"] = pd.to_datetime(f["post_time"], errors="coerce").astype("int64") / 1e9
    f.loc[f["post_time"] < 0, "post_time"] = np.nan

    # topic-level aggregates BEFORE deduplication (they describe the real thread)
    topic_first = f.groupby("topic_id")["post_time"].min()
    topic_n = f.groupby("topic_id").size()
    poster_n = f.groupby("poster_id").size()

    f = f.sort_values(["topic_id", "post_number"], kind="mergesort")
    f = f[~f["k"].duplicated()].set_index("k")

    hit = pop["k"].isin(f.index)
    coverage = float(hit.mean())
    assert coverage == 1.0, f"join coverage {coverage:.4f} -- refusing partial ordinals"

    sub = f.loc[pop["k"].values]
    out = pd.DataFrame({"row_id": pop["row_id"].values,
                        "problem": pop["problem"].values})
    out["post_number"] = sub["post_number"].values
    out["topic_id"] = sub["topic_id"].astype(str).values
    out["poster_id"] = sub["poster_id"].astype(str).values
    out["username"] = sub["username"].astype(str).values
    out["contest"] = sub["contest"].astype(str).values
    out["num_edits"] = pd.to_numeric(sub["num_edits"], errors="coerce").values
    out["topic_num_views"] = pd.to_numeric(sub["topic_num_views"], errors="coerce").values
    out["thanks_received"] = pd.to_numeric(sub["thanks_received"], errors="coerce").values
    out["nothanks_received"] = pd.to_numeric(sub["nothanks_received"], errors="coerce").values
    out["match_sim"] = pd.to_numeric(sub["match_sim"], errors="coerce").values
    out["match_kind"] = sub["match_kind"].astype(str).values

    pt = sub["post_time"].values.astype(float)
    t0 = topic_first.reindex(sub["topic_id"].values).values.astype(float)
    out["thread_age_days"] = (pt - t0) / 86400.0
    out["n_posts_topic"] = topic_n.reindex(sub["topic_id"].values).values.astype(float)
    out["poster_n_posts"] = poster_n.reindex(sub["poster_id"].values).values.astype(float)
    out["post_year"] = pd.to_datetime(pd.Series(pt), unit="s", errors="coerce").dt.year.values.astype(float)
    out["contest_year"] = [float(m.group(1)) if (m := YEAR.match(str(p))) else np.nan
                           for p in out["problem"]]
    out["years_after_contest"] = out["post_year"] - out["contest_year"]

    # rank WITHIN the closure cell's own container (the problem group), by the
    # observed thread ordinal.  Ties (same post_number, different topics for one
    # problem) broken deterministically by post_time then row_id.
    out = out.assign(_pt=pt)
    order = out.sort_values(["problem", "post_number", "_pt", "row_id"], kind="mergesort")
    order["sol_rank"] = order.groupby("problem").cumcount().astype(float)
    out = out.merge(order[["row_id", "sol_rank"]], on="row_id", how="left").drop(columns="_pt")
    ng = out.groupby("problem")["row_id"].transform("size").astype(float)
    out["n_sols_group"] = ng
    out["position_pct"] = np.where(ng > 1, out["sol_rank"] / (ng - 1), np.nan)

    out = out.set_index("row_id").loc[pop["row_id"].values].reset_index()
    out.to_csv(OUT, index=False)

    rep = {
        "cell": "aops_curation",
        "n": int(len(out)),
        "join": {"source": str(CRAWL.relative_to(REPO)),
                 "key": "sha1(problem + '|' + whitespace-normalised body_noquote)",
                 "coverage": coverage,
                 "crawl_rows": int(len(f)),
                 "note": "post_canonical matches only 4,662/5,202; body_noquote matches "
                         "5,202/5,202, so body_noquote is the join column"},
        "columns_numeric": NUM_COLS, "columns_string": STR_COLS,
        "neighbouring_outcomes_NEVER_FEATURES": ["thanks_received", "nothanks_received",
                                                 "match_sim"],
        "match_sim_note": "match_sim / match_kind are the CRAWLER's problem-statement "
                          "matching scores (how confidently the forum topic was matched to "
                          "the contest problem: 'text' 31,250 / 'text_secondary' 870), NOT "
                          "the solution-vs-editorial similarity that produced y -- that "
                          "label comes from datasets/math/aops/approach_verdicts.jsonl. "
                          "They are dataset-construction provenance and are carried here "
                          "only so the record can say they were never used in any readout, "
                          "model or stratifier",
        "describe": {c: {"finite": int(np.isfinite(out[c]).sum()),
                         "min": float(np.nanmin(out[c])), "median": float(np.nanmedian(out[c])),
                         "max": float(np.nanmax(out[c]))}
                     for c in NUM_COLS},
        "post_number_distribution": {str(k): int(v) for k, v in
                                     out["post_number"].clip(upper=20).value_counts()
                                     .sort_index().items()},
        "n_topics": int(out["topic_id"].nunique()),
        "n_posters": int(out["poster_id"].nunique()),
    }
    (HERE / "position_covariates.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()

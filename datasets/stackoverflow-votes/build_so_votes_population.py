#!/usr/bin/env python3
"""V6 -- StackOverflow answer-votes cell: population + y build.

THE CELL. Software-code field, VOTE/REVEALED (crowd) column of the 3xN
decomposition grid (notes/2026-08-08__vat-3xN-decomposition-grid.md). y =
community endorsement of a StackOverflow ANSWER via votes.

WHY THIS y (rationale is in notes/2026-08-08__v6_stackoverflow_build.md S1;
the short version, because the definition must be readable from the code):

  y_vote = 1 if the answer's raw net vote Score is STRICTLY ABOVE the median
           answer Score on its OWN question
           0 if STRICTLY BELOW
           undefined (dropped) if it TIES the median.

  This is a verbatim mirror of the math.SE vote cell's y
  ("1 = raw vote score strictly above the median answer score on its own
  question, 0 = strictly below; ties at the median dropped" --
  datasets/math-stackexchange/v2_va/population_manifest.json), which is the
  sibling SE vote cell in the same grid. Two reasons the WITHIN-QUESTION
  relative framing beats a raw score threshold:

    (a) Question popularity is the dominant driver of raw answer score. A
        raw-threshold y would mostly rank *questions* (how many people saw
        this thread), exactly the failure the V8 N&C co-signing build
        documented as "the pooled number is docket composition". Conditioning
        on question removes the popularity offset by construction.
    (b) It makes the vote column commensurable with the verdict column
        (acceptance), which is also inherently within-question -- so the
        cross-y contrast is run on identical rows with identical instruments.

  y_accepted (1 = this is the asker's accepted answer) is carried as a
  SEPARATE column and is NEVER merged into y_vote. That is the math.SE lesson
  the charge names: the legacy so_python_v2 pool defined its label as
  "accepted AND Score>=3" vs "not accepted AND Score<=0", which fuses the
  verdict and vote channels into one confounded target and cannot answer
  either column of the grid. This build keeps them orthogonal.

REUSE (reuse-before-rebuild; see the note's reuse log):
  * Raw corpus  100% reused -- so_python_{questions,answers}.parquet on sk3,
    built 2026-06-11 by datasets/stackoverflow_python/so_python_v2/
    ingest_shards_to_python.py from the 58-shard HF StackOverflow posts mirror.
    NOT re-downloaded, NOT re-ingested.
  * Split bucketer 100% reused -- datasets/patents/build_dense_standard_claimfell.py.
  * y construction mirrors math.SE (same rule, re-implemented over SO columns).

GROUND-TRUTH FINDINGS on the raw corpus, recorded because neither is visible
from the upstream code (the V8 "check the label channel" discipline):
  1. `Body` in this mirror is **Markdown, not HTML**. The ingest script's
     docstring asserts "KEEP Body raw (HTML)"; measured over 400K answers,
     <p>/<pre> appear on 0.21% of rows while ``` fences appear on 61.3% and
     inline `code` on 74.4%. Consequence: the legacy pool builder's
     strip_html() -- re.sub(r"<[^>]+>", " ") -- is not a no-op on this corpus,
     it is a CODE SHREDDER: it deletes `List<int>`, `<module>`, `<class 'x'>`
     and any `a < b ... c > d` span. This build therefore keeps Markdown and
     never calls strip_html.
  2. Score is the raw net vote (upvotes - downvotes), range observed
     [-44, 17568], 19.4% exactly 0, 1.6% negative.

Usage (sk3):
  python3 datasets/stackoverflow-votes/build_so_votes_population.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SALT = "so-votes-v1|"
YEAR_MIN, YEAR_MAX = 2016, 2023
MIN_CHARS = 50
N_TARGET = 13000  # mirrors the math.SE sibling cell's target


def sha1(s: str) -> str:
    return hashlib.sha1(str(s).encode()).hexdigest()


def h_order(qid) -> str:
    return hashlib.sha256((SALT + str(qid)).encode()).hexdigest()


# --- split bucketer: imported verbatim from the patents build (no reimpl) ----
def load_bucketer(repo: Path):
    import importlib.util
    p = repo / "datasets/patents/build_dense_standard_claimfell.py"
    spec = importlib.util.spec_from_file_location("cf_build", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cf_build"] = m
    spec.loader.exec_module(m)
    return m.stable_hash_bucket_map


def tags_to_list(tags) -> list:
    if tags is None:
        return []
    try:
        return [t for t in tags if isinstance(t, str)]
    except TypeError:
        return []


FRAMEWORK_TAGS = {
    "django", "flask", "fastapi", "celery", "pyramid", "tornado", "bottle",
    "aiohttp", "django-rest-framework", "django-models", "django-views",
    "django-forms", "django-admin", "flask-sqlalchemy", "starlette",
}
LIB_DATA_TAGS = {
    "pandas", "numpy", "scipy", "matplotlib", "scikit-learn", "sklearn",
    "seaborn", "tensorflow", "pytorch", "keras", "xarray", "dask", "numba",
    "statsmodels", "plotly", "bokeh",
}


def stratum_of_tags(tag_list) -> str:
    """Verifiability stratum, reused verbatim from the legacy SO pool builder
    (build_so_python_pool_v2.py) so the two populations remain comparable."""
    s = set(tag_list)
    if s & FRAMEWORK_TAGS:
        return "framework"
    if s & LIB_DATA_TAGS:
        return "lib_data"
    return "stdlib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/lfs/skampere3/0/alexspan/norm-research/"
                                      "datasets/stackoverflow_python")
    ap.add_argument("--repo", default="/lfs/skampere3/0/alexspan/norm-research")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-target", type=int, default=N_TARGET)
    a = ap.parse_args()

    root = Path(a.root)
    repo = Path(a.repo)
    out_dir = Path(a.out_dir) if a.out_dir else repo / "datasets/stackoverflow-votes/va"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"build_date": datetime.now().isoformat(timespec="seconds"),
             "salt": SALT, "source": {
                 "questions": str(root / "so_python_questions.parquet"),
                 "answers": str(root / "so_python_answers.parquet"),
                 "provenance": "58-shard HuggingFace StackOverflow posts mirror "
                               "-> ingest_shards_to_python.py (2026-06-11); "
                               "REUSED, not re-downloaded"}}

    # ---------------------------------------------------------- questions ---
    print("[1/7] reading questions ...", flush=True)
    qdf = pq.read_table(root / "so_python_questions.parquet",
                        columns=["Id", "Title", "Body", "Tags",
                                 "AcceptedAnswerId", "Score", "ViewCount",
                                 "CreationDate"]).to_pandas()
    qdf["Id"] = qdf["Id"].astype("int64")
    stats["questions_total"] = int(len(qdf))
    print(f"      questions={len(qdf):,}", flush=True)

    # ------------------------------------------------------------ answers ---
    print("[2/7] reading answers ...", flush=True)
    adf = pq.read_table(root / "so_python_answers.parquet",
                        columns=["Id", "ParentId", "Score", "CreationDate",
                                 "Body", "OwnerUserId"]).to_pandas()
    adf["Id"] = adf["Id"].astype("int64")
    adf["ParentId"] = adf["ParentId"].astype("int64")
    stats["answers_total"] = int(len(adf))

    adf["year"] = adf["CreationDate"].astype(str).str[:4].astype(int)
    stats["answer_year_counts_all"] = {str(k): int(v) for k, v in
                                       sorted(Counter(adf["year"]).items())}
    adf = adf[(adf.year >= YEAR_MIN) & (adf.year <= YEAR_MAX)].copy()
    stats["after_year_window"] = int(len(adf))
    print(f"      answers in {YEAR_MIN}-{YEAR_MAX}: {len(adf):,}", flush=True)

    # NOTE: Body is MARKDOWN on this mirror (see docstring). No HTML stripping.
    adf["body"] = adf["Body"].astype(str).str.strip()
    adf = adf[adf.body.str.len() >= MIN_CHARS]
    stats["after_min_chars"] = int(len(adf))

    adf = adf[adf.ParentId.isin(set(qdf.Id))]
    stats["after_parent_exists"] = int(len(adf))

    # ------------------------------------------ >=2 answers on the question --
    print("[3/7] within-question y ...", flush=True)
    sz = adf.groupby("ParentId").Id.transform("size")
    adf = adf[sz >= 2].copy()
    stats["after_multi_answer"] = int(len(adf))
    stats["n_questions_multi_answer"] = int(adf.ParentId.nunique())

    med = adf.groupby("ParentId").Score.transform("median")
    adf["q_median_score"] = med
    adf["y_vote"] = np.where(adf.Score > med, 1.0,
                             np.where(adf.Score < med, 0.0, np.nan))
    stats["tie_at_median_rate"] = float(adf.y_vote.isna().mean())
    stats["n_vote_defined_prefilter"] = int(adf.y_vote.notna().sum())

    # ------------------------------------------------------ verdict channel --
    acc = qdf.set_index("Id").AcceptedAnswerId
    adf["accepted_answer_id"] = adf.ParentId.map(acc)
    adf["y_accepted"] = (adf.Id == adf.accepted_answer_id).astype(int)

    # -------------------------------------------------------- position line --
    adf = adf.sort_values(["ParentId", "CreationDate", "Id"])
    adf["position"] = adf.groupby("ParentId").cumcount() + 1
    adf["n_answers_q"] = adf.groupby("ParentId").Id.transform("size")

    # ---------------------------------------------- questions with BOTH ------
    # mirrors math.SE: sample only questions carrying BOTH signals, so the
    # cross-y contrast runs on identical rows.
    ok_vote = adf.groupby("ParentId").y_vote.transform(lambda s: s.notna().any())
    ok_acc = adf.groupby("ParentId").y_accepted.transform("max") == 1
    both = adf[ok_vote & ok_acc]
    stats["n_questions_both_signals"] = int(both.ParentId.nunique())
    print(f"      questions w/ both signals: {both.ParentId.nunique():,}",
          flush=True)

    # ------------------------------------------------------------- sample ---
    print("[4/7] hash-ordered whole-question sample ...", flush=True)
    qids = sorted(both.ParentId.unique(), key=h_order)
    per_q = both.groupby("ParentId").size().to_dict()
    chosen, tot = [], 0
    for q in qids:
        chosen.append(q)
        tot += per_q[q]
        if tot >= a.n_target:
            break
    pop = both[both.ParentId.isin(set(chosen))].copy()
    stats["sampling"] = (f"whole questions in sha256('{SALT}' + question_id) "
                         f"order until >= {a.n_target} rows, restricted to "
                         "questions carrying BOTH signals (a defined vote y and "
                         "an accepted answer)")
    print(f"      sampled rows={len(pop):,} questions={pop.ParentId.nunique():,}",
          flush=True)

    # ------------------------------------------------------------ context ---
    print("[5/7] attaching question context ...", flush=True)
    qsub = qdf[qdf.Id.isin(set(chosen))].set_index("Id")
    pop["q_title"] = pop.ParentId.map(qsub.Title).astype(str)
    pop["q_body"] = pop.ParentId.map(qsub.Body).astype(str)
    pop["q_score"] = pop.ParentId.map(qsub.Score)
    pop["q_viewcount"] = pop.ParentId.map(qsub.ViewCount)
    pop["q_creation"] = pop.ParentId.map(qsub.CreationDate).astype(str)
    tagmap = {int(i): tags_to_list(t) for i, t in
              zip(qsub.index, qsub.Tags)}
    pop["tags"] = pop.ParentId.map(lambda q: "|".join(tagmap.get(int(q), [])))
    pop["stratum"] = pop.ParentId.map(
        lambda q: stratum_of_tags(tagmap.get(int(q), [])))
    # exposure covariates (nuisance work / Layer 2(b)); NOT features
    pop["age_days_from_q"] = (
        pd.to_datetime(pop.CreationDate, errors="coerce")
        - pd.to_datetime(pop.q_creation, errors="coerce")).dt.total_seconds() / 86400.0

    pop["row_id"] = pop.Id.astype(str)
    pop["group"] = pop.ParentId.astype(str)
    pop["text"] = ("QUESTION: " + pop.q_title.str.strip()
                   + "\n\nANSWER:\n" + pop.body)

    # -------------------------------------------------------------- splits --
    print("[6/7] docket-analog (question) grouped stable-hash splits ...",
          flush=True)
    voted = pop[pop.y_vote.notna()].copy()
    bucketer = load_bucketer(repo)
    y_by_group = {g: d.y_vote.astype(int).tolist()
                  for g, d in voted.groupby("group")}
    bmap = bucketer(y_by_group)
    pop["split"] = pop.group.map(bmap)

    # ------------------------------------------------------------- write ----
    print("[7/7] writing ...", flush=True)
    cols = ["row_id", "group", "split", "text", "body", "q_title", "q_body",
            "y_vote", "y_accepted", "Score", "q_median_score", "position",
            "n_answers_q", "tags", "stratum", "q_score", "q_viewcount",
            "year", "CreationDate", "q_creation", "age_days_from_q",
            "OwnerUserId"]
    pop = pop[cols].sort_values(["group", "position"]).reset_index(drop=True)
    pop.to_csv(out_dir / "population.csv.gz", index=False, compression="gzip")

    v = pop[pop.y_vote.notna()]
    stats["population"] = {
        "n_rows": int(len(pop)),
        "n_questions": int(pop.group.nunique()),
        "n_vote_defined": int(len(v)),
        "y_vote_pos_rate": float(v.y_vote.mean()),
        "y_accepted_pos_rate": float(pop.y_accepted.mean()),
        "tie_dropped_rows": int(pop.y_vote.isna().sum()),
        "median_answers_per_question": float(pop.groupby("group").size().median()),
        "strata": {k: int(x) for k, x in pop.stratum.value_counts().items()},
        "year_counts": {str(k): int(x) for k, x in
                        sorted(pop.year.value_counts().items())},
    }
    stats["splits_vote"] = {
        s: {"rows": int((v.split == s).sum()),
            "questions": int(v[v.split == s].group.nunique()),
            "pos_rate": float(v[v.split == s].y_vote.mean())}
        for s in ["train", "eval", "test"]}
    stats["splits_all_rows"] = {
        s: int((pop.split == s).sum()) for s in ["train", "eval", "test"]}
    # cross-y contrast, on identical rows
    stats["cross_y"] = {
        "n": int(len(v)),
        "accept_rate_when_vote_pos": float(v[v.y_vote == 1].y_accepted.mean()),
        "accept_rate_when_vote_neg": float(v[v.y_vote == 0].y_accepted.mean()),
        "phi": float(np.corrcoef(v.y_vote, v.y_accepted)[0, 1]),
    }
    # position line (observed covariate, the first-answer advantage)
    stats["position_line"] = {
        str(p): {"rows": int(len(d)), "vote_pos_rate": float(d.y_vote.mean()),
                 "accept_rate": float(d.y_accepted.mean())}
        for p, d in v.groupby(v.position.clip(upper=5))}

    (out_dir / "population_manifest.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats["population"], indent=1))
    print(json.dumps(stats["splits_vote"], indent=1))
    print(json.dumps(stats["cross_y"], indent=1))
    print(json.dumps(stats["position_line"], indent=1))


if __name__ == "__main__":
    main()

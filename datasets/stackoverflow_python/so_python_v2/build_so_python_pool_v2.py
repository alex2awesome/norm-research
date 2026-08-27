#!/usr/bin/env python3
"""SO[python] v2 pool build with position matching + verifiability stratum.

Recipe (mirrors CR.SE v2 / Math.SE v3):
  - Inputs: so_python_questions.parquet, so_python_answers.parquet
            (from ingest_shards_to_python.py)
  - Pos label: accepted AND Score >= --pos-min-score (default 3)
  - Neg label: NOT accepted AND Score <= --neg-max-score (default 0 STRICT)
  - Year window: 2016-2023 inclusive (volume + recency; documented per-year counts)
  - Question-disjoint filter
  - Min answer length 50 chars (HTML stripped) AND parent question must exist
  - **Position matching** within
       (primary_tag x q_len_bin x a_len_bin x position{1,2,3+} x year3) per tag
    — matches CR.SE v2 exactly.
  - **Verifiability stratum** is added as a non-matching column:
       stratum in {lib_data, stdlib, framework} (precedence framework > lib_data > stdlib).
    Stratum is NOT in the matching key — we want a single consistently-built
    pool so that downstream comparisons across strata use the same matching.
  - Auto-fallback for neg_max_score if matched pool < --min-pool-rows (50K).

Manifest records:
  - per-attempt stage counts + time-order audit
  - per-stratum row counts (post-matching)
  - per-year counts on labeled answers before matching
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from splits import split_of  # noqa: E402


POSITION_BIN_LABELS = ("1", "2", "3+")


# ---------- text + tag helpers ----------

def strip_html(html_text: str) -> str:
    if not isinstance(html_text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    text = " ".join(text.split())
    return text.strip()


def tags_to_list(tags) -> list[str]:
    """Tags come from the parquet as a numpy array / list / None.

    SO mirror stores them as a pyarrow list<string>, which round-trips to numpy
    array or python list — handle all of these.
    """
    if tags is None:
        return []
    try:
        out = [t for t in tags if isinstance(t, str)]
    except TypeError:
        return []
    return out


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


def stratum_of_tags(tag_list: list[str]) -> str:
    s = set(tag_list)
    if s & FRAMEWORK_TAGS:
        return "framework"
    if s & LIB_DATA_TAGS:
        return "lib_data"
    return "stdlib"


def primary_tag_of(tag_list: list[str]) -> str:
    """Most-informative-NON-python tag = highest-frequency rule:

    We don't know per-corpus tag frequencies here, so use a deterministic
    fallback: first non-'python' tag, falling back to 'python' if only python
    is present. CR.SE used "first tag in the question's tag list" which is
    fine for that site; SO[python] dilutes too quickly with that rule since
    every question has 'python' as a tag. Skip 'python' itself.
    """
    for t in tag_list:
        if t and t != "python" and not t.startswith("python-"):
            return t
    # fallback: a python-* tag (python-3.x, python-2.7, etc.)
    for t in tag_list:
        if t.startswith("python-"):
            return t
    return "python"


# ---------- core build ----------

def build_labeled_pool(qs_parquet: str, ans_parquet: str,
                       pos_min_score: int, neg_max_score: int,
                       min_chars: int, year_min: int, year_max: int,
                       ) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Returns cand, qs_map, stage, time_order_audit."""
    stage: dict = {}
    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] loading {qs_parquet}", flush=True)
    qs = pq.read_table(qs_parquet).to_pandas()
    print(f"  questions: {len(qs):,}", flush=True)
    stage["questions_total"] = int(len(qs))

    print(f"[{datetime.now():%H:%M:%S}] loading {ans_parquet}", flush=True)
    ans = pq.read_table(ans_parquet).to_pandas()
    print(f"  answers: {len(ans):,}", flush=True)
    stage["answers_total"] = int(len(ans))

    # Question side
    print(f"[{datetime.now():%H:%M:%S}] processing questions", flush=True)
    qs["title"] = qs.Title.fillna("")
    qs["body_stripped"] = qs.Body.fillna("").map(strip_html)
    qs["tag_list"] = qs.Tags.map(tags_to_list)
    qs["question_len"] = qs.title.str.len() + 2 + qs.body_stripped.str.len()
    qs["primary_tag"] = qs.tag_list.map(primary_tag_of)
    qs["stratum"] = qs.tag_list.map(stratum_of_tags)
    qs["tags_pipe"] = qs.tag_list.map(lambda L: "|".join(L))
    qs_map_full = qs.set_index("Id")[
        ["title", "body_stripped", "tags_pipe", "question_len",
         "primary_tag", "stratum"]
    ].to_dict(orient="index")

    accepted_ids = set(
        qs[qs.AcceptedAnswerId.fillna(0).astype(np.int64) > 0]
        .AcceptedAnswerId.astype(np.int64).tolist()
    )
    stage["n_accepted_answer_ids"] = len(accepted_ids)

    # Answers: position, year, accepted
    print(f"[{datetime.now():%H:%M:%S}] computing position + year on answers", flush=True)
    ans = ans.sort_values(["ParentId", "CreationDate", "Id"]).reset_index(drop=True)
    ans["answer_position"] = ans.groupby("ParentId").cumcount() + 1
    ans["n_answers_on_question"] = ans.groupby("ParentId")["Id"].transform("count")
    first_dt = ans.groupby("ParentId")["CreationDate"].transform("first")
    ans["_first_dt"] = pd.to_datetime(first_dt, errors="coerce")
    ans["_self_dt"] = pd.to_datetime(ans.CreationDate, errors="coerce")
    ans["answer_age_gap_days"] = (
        (ans["_self_dt"] - ans["_first_dt"]).dt.total_seconds() / 86400.0
    ).round(3).fillna(0.0)
    ans["answer_year"] = pd.to_datetime(ans.CreationDate, errors="coerce").dt.year

    # Per-year answer counts BEFORE filtering (for the manifest)
    year_counts_all = ans.groupby("answer_year").size().to_dict()
    stage["year_counts_all_answers"] = {int(k): int(v) for k, v in year_counts_all.items()
                                          if not pd.isna(k)}

    # Year window filter
    yr_mask = (ans.answer_year >= year_min) & (ans.answer_year <= year_max)
    ans = ans[yr_mask].copy()
    stage["after_year_window"] = int(len(ans))

    # Answer text strip + length
    print(f"[{datetime.now():%H:%M:%S}] stripping answer HTML", flush=True)
    ans["answer_text"] = ans.Body.fillna("").map(strip_html)
    ans["answer_len"] = ans.answer_text.str.len()

    # Label
    ans["accepted"] = ans.Id.astype(np.int64).isin(accepted_ids)
    is_pos = ans.accepted & (ans.Score >= pos_min_score)
    is_neg = (~ans.accepted) & (ans.Score <= neg_max_score)
    stage["labeled_positive_pre_qdisjoint"] = int(is_pos.sum())
    stage["labeled_negative_pre_qdisjoint"] = int(is_neg.sum())

    keep_mask = (is_pos | is_neg) & (ans.answer_len >= min_chars) & (ans.ParentId.isin(qs_map_full))
    cand = ans[keep_mask].copy()
    cand["judgement"] = is_pos[keep_mask].astype(int).values
    stage["after_length_and_parent_filter"] = int(len(cand))

    # Time-order audit BEFORE question-disjoint filter — VECTORIZED.
    # Algorithm: per (qid), count pairs (i in pos, j in neg) where
    #   pos.position[i] < neg.position[j].
    # Vectorize by sorting all answers in the candidate frame by
    # (ParentId, answer_position), then within each group counting
    # "running #pos earlier than each neg" via cumulative-sum tricks.
    print(f"[{datetime.now():%H:%M:%S}] time-order audit (vectorized)", flush=True)
    aud = cand[["ParentId", "answer_position", "judgement"]].copy()
    aud = aud.sort_values(["ParentId", "answer_position"]).reset_index(drop=True)
    # n_pos per group, broadcast to rows; n_neg = size - n_pos.
    grp_obj = aud.groupby("ParentId", sort=False)
    aud["n_pos"] = grp_obj["judgement"].transform("sum")
    aud["n_tot"] = grp_obj["judgement"].transform("size")
    aud["n_neg"] = aud["n_tot"] - aud["n_pos"]
    both = aud[(aud["n_pos"] > 0) & (aud["n_neg"] > 0)].copy()
    # cumulative pos count BEFORE the current row, within group.
    both["pos_seen_before"] = (both.groupby("ParentId", sort=False)["judgement"]
                                .cumsum() - both["judgement"])
    pos_pairs = int(both.loc[both["judgement"] == 0, "pos_seen_before"].sum())
    grp = both.groupby("ParentId", sort=False).agg(
        n_pos=("judgement", "sum"),
        n_tot=("judgement", "size"),
    )
    grp["n_neg"] = grp["n_tot"] - grp["n_pos"]
    n_pairs = int((grp["n_pos"] * grp["n_neg"]).sum())
    n_q_both = int(len(grp))
    audit = {
        "n_questions_with_both_classes": int(n_q_both),
        "n_pos_neg_pairs": int(n_pairs),
        "p_positive_posted_earlier": (pos_pairs / n_pairs) if n_pairs else float("nan"),
        "note": ("Within-question pos/neg pairs by CreationDate-rank. Computed "
                 "BEFORE the question-disjoint filter (which removes exactly "
                 "these questions)."),
    }
    print(f"  audit: n_q_both={n_q_both:,}  n_pairs={n_pairs:,}  p_earlier={pos_pairs/max(1,n_pairs):.4f}", flush=True)

    # Question-disjoint filter
    pos_qids = set(cand.loc[cand.judgement == 1, "ParentId"].tolist())
    neg_qids = set(cand.loc[cand.judgement == 0, "ParentId"].tolist())
    overlap = pos_qids & neg_qids
    cand = cand[~cand.ParentId.isin(overlap)].copy()
    stage["overlap_questions_dropped"] = int(len(overlap))
    stage["after_question_disjoint_pos"] = int((cand.judgement == 1).sum())
    stage["after_question_disjoint_neg"] = int((cand.judgement == 0).sum())

    # Attach per-row question metadata
    def lookup(q, key, default=""):
        rec = qs_map_full.get(q)
        if not rec:
            return default
        return rec.get(key, default)

    # Coerce ParentId to plain int64 BEFORE map: parquet may store it as
    # float64 (because of NaNs in non-answer rows; we filtered those out).
    cand["ParentId"] = cand["ParentId"].astype(np.int64)
    cand["question_tags"] = cand.ParentId.map(lambda q: lookup(q, "tags_pipe"))
    cand["primary_tag"] = cand.ParentId.map(lambda q: lookup(q, "primary_tag", "_none_"))
    cand["stratum"] = cand.ParentId.map(lambda q: lookup(q, "stratum", "stdlib"))
    cand["question_id"] = cand.ParentId.astype(np.int64).astype(str)
    cand["answer_id"] = cand.Id.astype(np.int64).astype(str)
    cand["question_len"] = cand.ParentId.map(lambda q: lookup(q, "question_len", 0))

    return cand, qs_map_full, stage, audit


def position_matched_downsample(cand: pd.DataFrame, n_len_bins: int,
                                seed: int) -> tuple[pd.DataFrame, dict]:
    """Per-(primary_tag) + (q_len_bin x a_len_bin x position{1,2,3+} x year3)
    matched downsampling. Stratum is NOT part of the cell — we want a single
    pool that compares across strata after the fact. Identical algorithm to
    CR.SE v2's position_matched_downsample.
    """
    rng = random.Random(seed)

    pos = cand[cand.judgement == 1]
    neg = cand[cand.judgement == 0]

    pos_by_tag = {t: g for t, g in pos.groupby("primary_tag")}
    neg_by_tag = {t: g for t, g in neg.groupby("primary_tag")}
    all_tags = set(pos_by_tag) | set(neg_by_tag)

    kept_pos_idx: list[int] = []
    kept_neg_idx: list[int] = []
    n_one_class_tags = 0
    cell_counts: list[dict] = []
    cells_with_overlap = 0
    cells_dropped_no_overlap = 0

    for tag in sorted(all_tags):
        tp = pos_by_tag.get(tag)
        tn = neg_by_tag.get(tag)
        if tp is None or tn is None or len(tp) == 0 or len(tn) == 0:
            n_one_class_tags += 1
            continue

        small_is_pos = len(tp) <= len(tn)
        smaller = tp if small_is_pos else tn
        larger = tn if small_is_pos else tp

        q_edges = np.percentile(smaller["question_len"].values,
                                np.linspace(0, 100, n_len_bins + 1))
        a_edges = np.percentile(smaller["answer_len"].values,
                                np.linspace(0, 100, n_len_bins + 1))
        q_edges[-1] += 1
        a_edges[-1] += 1

        def joint_bin_vec(df: pd.DataFrame):
            qb = np.searchsorted(q_edges[1:], df["question_len"].values)
            ab = np.searchsorted(a_edges[1:], df["answer_len"].values)
            pb = np.minimum(df["answer_position"].values, 3)
            yb = df["answer_year"].values // 3
            return list(zip(qb.tolist(), ab.tolist(), pb.tolist(), yb.tolist()))

        smaller = smaller.copy()
        larger = larger.copy()
        smaller["_cell"] = joint_bin_vec(smaller)
        larger["_cell"] = joint_bin_vec(larger)

        small_by_cell: dict = {}
        for cell, g in smaller.groupby("_cell"):
            small_by_cell[cell] = g.index.tolist()
        large_by_cell: dict = {}
        for cell, g in larger.groupby("_cell"):
            large_by_cell[cell] = g.index.tolist()

        for cell, s_idx in small_by_cell.items():
            l_idx = large_by_cell.get(cell)
            if not l_idx:
                cells_dropped_no_overlap += 1
                continue
            cells_with_overlap += 1
            n_take = min(len(s_idx), len(l_idx))
            sampled_s = rng.sample(s_idx, n_take)
            sampled_l = rng.sample(l_idx, n_take)
            if small_is_pos:
                kept_pos_idx.extend(sampled_s)
                kept_neg_idx.extend(sampled_l)
            else:
                kept_neg_idx.extend(sampled_s)
                kept_pos_idx.extend(sampled_l)
            cell_counts.append({
                "tag": tag, "cell": str(cell),
                "small_avail": int(len(s_idx)),
                "large_avail": int(len(l_idx)),
                "kept_per_class": int(n_take),
            })

    matched = pd.concat([cand.loc[kept_pos_idx], cand.loc[kept_neg_idx]])
    matched = matched.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    cell_sizes = [c["kept_per_class"] for c in cell_counts]
    summary = {
        "tags_processed": int(len(all_tags) - n_one_class_tags),
        "tags_skipped_one_class": int(n_one_class_tags),
        "cells_with_overlap": int(cells_with_overlap),
        "cells_dropped_no_overlap": int(cells_dropped_no_overlap),
        "kept_per_class_quantiles": {
            "min": int(min(cell_sizes)) if cell_sizes else 0,
            "p25": float(np.percentile(cell_sizes, 25)) if cell_sizes else 0.0,
            "p50": float(np.percentile(cell_sizes, 50)) if cell_sizes else 0.0,
            "p75": float(np.percentile(cell_sizes, 75)) if cell_sizes else 0.0,
            "p99": float(np.percentile(cell_sizes, 99)) if cell_sizes else 0.0,
            "max": int(max(cell_sizes)) if cell_sizes else 0,
        },
        "total_rows_matched": int(len(matched)),
        "pos_rate": float(matched.judgement.mean()) if len(matched) else float("nan"),
    }
    return matched, summary


def write_pool(matched: pd.DataFrame, qs_map: dict, out_path: Path,
               seed: int) -> dict:
    matched = matched.copy()

    def _make_text(row):
        # question_id is a stringified int; ParentId may be float in source.
        try:
            qid = int(float(row.question_id))
        except (ValueError, TypeError):
            qid = -1
        q = qs_map.get(qid, {})
        title = q.get("title", "")
        body = q.get("body_stripped", "")
        question = (title + "\n\n" + body).strip()
        return f"Question: {question}\n\nAnswer: {row.answer_text}"

    matched["text"] = matched.apply(_make_text, axis=1)
    matched["split"] = matched["question_id"].map(split_of)

    out_cols = [
        "text", "judgement", "split", "question_id", "answer_id",
        "answer_position", "n_answers_on_question", "answer_age_gap_days",
        "answer_year", "score", "accepted", "primary_tag", "stratum",
        "question_tags",
    ]
    out_df = matched.rename(columns={"Score": "score"})[out_cols]
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, compression="gzip")

    splits = {
        s: {
            "rows": int((out_df.split == s).sum()),
            "positive": int(((out_df.split == s) & (out_df.judgement == 1)).sum()),
            "negative": int(((out_df.split == s) & (out_df.judgement == 0)).sum()),
        } for s in ("train", "eval", "test")
    }
    strata = {
        s: {
            "rows": int((out_df.stratum == s).sum()),
            "positive": int(((out_df.stratum == s) & (out_df.judgement == 1)).sum()),
            "negative": int(((out_df.stratum == s) & (out_df.judgement == 0)).sum()),
        } for s in ("stdlib", "lib_data", "framework")
    }
    years = {
        int(yr): int((out_df.answer_year == yr).sum())
        for yr in sorted(out_df.answer_year.dropna().unique().astype(int))
    }
    return {"rows": int(len(out_df)),
            "pos_rate": float(out_df.judgement.mean()),
            "splits": splits, "strata": strata, "year_counts": years}


def _hash_splits() -> str:
    p = Path(__file__).parent / "splits.py"
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--qs-parquet",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/so_python_questions.parquet",
    )
    ap.add_argument(
        "--ans-parquet",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/so_python_answers.parquet",
    )
    ap.add_argument("--pos-min-score", type=int, default=3)
    ap.add_argument("--neg-max-score", type=int, default=0,
                    help="Strict recipe default. Falls back to 1 if matched "
                         "pool < --min-pool-rows.")
    ap.add_argument("--min-pool-rows", type=int, default=50_000,
                    help="If matched pool < this with --neg-max-score=0, "
                         "fall back to --neg-max-score=1.")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--n-len-bins", type=int, default=10)
    ap.add_argument("--year-min", type=int, default=2016)
    ap.add_argument("--year-max", type=int, default=2023)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/pool/so_python_v2_pool.csv.gz",
    )
    ap.add_argument(
        "--manifest",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/pool/so_python_v2_pool.manifest.json",
    )
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    manifest: dict = {
        "args": vars(args),
        "build_date": datetime.now().isoformat(timespec="seconds"),
        "qs_parquet": args.qs_parquet,
        "ans_parquet": args.ans_parquet,
        "splits_module_hash": _hash_splits(),
    }

    attempts: list[dict] = []
    attempt_order = ([0, 1] if args.neg_max_score == 0 and not args.no_fallback
                     else [args.neg_max_score])
    chosen = None
    for attempt_neg_max in attempt_order:
        print(f"\n=== ATTEMPT: neg_max_score = {attempt_neg_max} ===", flush=True)
        cand, qs_map, stage, audit = build_labeled_pool(
            qs_parquet=args.qs_parquet,
            ans_parquet=args.ans_parquet,
            pos_min_score=args.pos_min_score,
            neg_max_score=attempt_neg_max,
            min_chars=args.min_chars,
            year_min=args.year_min,
            year_max=args.year_max,
        )
        matched, matching_summary = position_matched_downsample(
            cand, n_len_bins=args.n_len_bins, seed=args.seed)
        attempt_log = {
            "neg_max_score": attempt_neg_max,
            "labeling": stage,
            "time_order_audit": audit,
            "matching": matching_summary,
            "matched_rows": int(len(matched)),
        }
        attempts.append(attempt_log)
        if len(matched) >= args.min_pool_rows:
            chosen = (cand, qs_map, matched, attempt_log)
            break
        print(f"[fallback] matched={len(matched)} < min_pool_rows={args.min_pool_rows}", flush=True)
    if chosen is None:
        # Use the last attempt anyway
        chosen = (cand, qs_map, matched, attempt_log)
        attempt_log["forced_use_below_min"] = True

    cand, qs_map, matched, attempt_log = chosen
    manifest["attempts"] = attempts
    manifest["chosen"] = {
        "neg_max_score": attempt_log["neg_max_score"],
        "matched_rows": attempt_log["matched_rows"],
        "fallback_triggered": (attempt_log["neg_max_score"] != args.neg_max_score),
    }

    write_stats = write_pool(matched, qs_map, Path(args.out), seed=args.seed)
    manifest["pool"] = write_stats

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] wrote pool {args.out} "
          f"({write_stats['rows']:,} rows; neg_max_score="
          f"{manifest['chosen']['neg_max_score']})", flush=True)
    print(f"  per-stratum rows: {write_stats['strata']}", flush=True)
    print(f"  manifest -> {args.manifest}", flush=True)


if __name__ == "__main__":
    main()

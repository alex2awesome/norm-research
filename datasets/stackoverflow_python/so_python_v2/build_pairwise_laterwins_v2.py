#!/usr/bin/env python3
"""SO[python] v2 pairwise "later answer wins" companion.

Same recipe as CR.SE v2:
  1. chosen_id > rejected_id    (later answer won)
  2. score_diff >= --min-score-diff (default 3)
  3. both answer texts >= --min-chars (default 50)
  4. question-grouped split via splits.py

Inputs: so_python_questions.parquet + so_python_answers.parquet (from Phase 1).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from splits import split_of  # noqa: E402


def strip_html(html_text: str) -> str:
    if not isinstance(html_text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    text = " ".join(text.split())
    return text.strip()


def tags_to_pipe(tags) -> str:
    if tags is None:
        return ""
    try:
        return "|".join(t for t in tags if isinstance(t, str))
    except TypeError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qs-parquet",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/so_python_questions.parquet",
    )
    ap.add_argument(
        "--ans-parquet",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/so_python_answers.parquet",
    )
    ap.add_argument("--min-score-diff", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--year-min", type=int, default=2016)
    ap.add_argument("--year-max", type=int, default=2023)
    ap.add_argument(
        "--out",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/pairwise/so_python_pairwise_laterwins_v2.csv.gz",
    )
    args = ap.parse_args()

    print(f"[{datetime.now():%H:%M:%S}] loading {args.qs_parquet}", flush=True)
    qs = pq.read_table(args.qs_parquet, columns=["Id", "Tags"]).to_pandas()
    qtags = qs.set_index("Id")["Tags"].map(tags_to_pipe).to_dict()

    print(f"[{datetime.now():%H:%M:%S}] loading {args.ans_parquet}", flush=True)
    ans = pq.read_table(args.ans_parquet).to_pandas()

    ans["answer_text"] = ans.Body.fillna("").map(strip_html)
    ans["answer_len"] = ans.answer_text.str.len()
    ans["answer_year"] = pd.to_datetime(ans.CreationDate, errors="coerce").dt.year
    ans = ans[(ans.answer_len >= args.min_chars)
              & (ans.answer_year >= args.year_min)
              & (ans.answer_year <= args.year_max)].copy()

    print(f"[{datetime.now():%H:%M:%S}] forming within-question pairs", flush=True)
    rows = []
    n_q_multi = 0
    for qid, g in ans.groupby("ParentId"):
        if len(g) < 2:
            continue
        n_q_multi += 1
        g_records = g[["Id", "Score", "answer_text"]].to_dict(orient="records")
        for i in range(len(g_records)):
            for j in range(i + 1, len(g_records)):
                a, b = g_records[i], g_records[j]
                if a["Score"] == b["Score"]:
                    continue
                chosen, rejected = (a, b) if a["Score"] > b["Score"] else (b, a)
                rows.append({
                    "question_id": int(qid),
                    "question_tags": qtags.get(int(qid), ""),
                    "chosen_id": int(chosen["Id"]),
                    "chosen_score": int(chosen["Score"]),
                    "chosen_text": chosen["answer_text"],
                    "rejected_id": int(rejected["Id"]),
                    "rejected_score": int(rejected["Score"]),
                    "rejected_text": rejected["answer_text"],
                    "score_diff": int(chosen["Score"] - rejected["Score"]),
                })

    n0 = len(rows)
    print(f"  built {n0:,} pairs over {n_q_multi:,} multi-answer questions", flush=True)

    pairs = pd.DataFrame(rows)
    manifest = {
        "build_date": datetime.now().isoformat(timespec="seconds"),
        "qs_parquet": args.qs_parquet,
        "ans_parquet": args.ans_parquet,
        "year_window": [args.year_min, args.year_max],
        "n_input_multi_answer_questions": int(n_q_multi),
        "n_pairs_before_filters": int(n0),
        "filters": [],
    }

    pairs = pairs[pairs.chosen_id > pairs.rejected_id].copy()
    manifest["filters"].append({"filter": "chosen_id > rejected_id (later wins)",
                                "n_after": int(len(pairs))})
    print(f"  after later-wins: {len(pairs):,}", flush=True)

    pairs = pairs[pairs.score_diff >= args.min_score_diff].copy()
    manifest["filters"].append({"filter": f"score_diff >= {args.min_score_diff}",
                                "n_after": int(len(pairs))})
    print(f"  after score_diff>={args.min_score_diff}: {len(pairs):,}", flush=True)

    pairs = pairs[(pairs.chosen_text.str.len() >= args.min_chars)
                  & (pairs.rejected_text.str.len() >= args.min_chars)].copy()
    manifest["filters"].append({"filter": f"both texts >= {args.min_chars} chars",
                                "n_after": int(len(pairs))})
    print(f"  after min-length: {len(pairs):,}", flush=True)

    pairs["split"] = pairs.question_id.astype(str).map(split_of)
    out_cols = ["question_id", "question_tags", "chosen_text", "rejected_text",
                "chosen_score", "rejected_score", "score_diff", "split"]
    tag_counts = Counter(
        t for ts in pairs.question_tags.fillna("").map(lambda s: s.split("|"))
        for t in ts if t
    )
    manifest["n_final_pairs"] = int(len(pairs))
    manifest["split_counts"] = pairs.split.value_counts().to_dict()
    manifest["tag_distribution_top20"] = dict(tag_counts.most_common(20))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pairs[out_cols].to_csv(args.out, index=False, compression="gzip")
    mpath = str(args.out).replace(".csv.gz", ".manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[{datetime.now():%H:%M:%S}] wrote {args.out} ({len(pairs):,} pairs)", flush=True)


if __name__ == "__main__":
    main()

"""Deep dive task 1a: sample js/sql answers for hand-validation of the
fenced-block code-detection regex (2026-06-12).

Population = the labeled candidates the filter actually gated: year window
2016-2023, STRICT label (accepted & score>=3 | ~accepted & score<=0),
stripped len>=50, parent exists. Sample 50 detected + 50 not-detected per
slice (seed 7), dump answer bodies (truncated 4000 chars) to JSONL for
hand-reading on the laptop.

Run on sk3: python3 dd_task1_sample_jssql.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets")
OUT = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/se_ladder")
CODE_BLOCK_RE = re.compile(r"^[ \t]*(?:```|~~~)", re.M)


def strip_html(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                 ("&quot;", '"'), ("&#39;", "'")]:
        t = t.replace(a, b)
    return " ".join(t.split()).strip()


def run(slice_name, qs_p, ans_p):
    qs = pq.read_table(qs_p, columns=["Id", "AcceptedAnswerId"]).to_pandas()
    accepted = set(qs[qs.AcceptedAnswerId.fillna(0).astype(np.int64) > 0]
                   .AcceptedAnswerId.astype(np.int64))
    qids = set(qs.Id.astype(np.int64))
    ans = pq.read_table(ans_p).to_pandas()
    ans["year"] = pd.to_datetime(ans.CreationDate, errors="coerce").dt.year
    ans = ans[(ans.year >= 2016) & (ans.year <= 2023)]
    ans["acc"] = ans.Id.astype(np.int64).isin(accepted)
    is_pos = ans.acc & (ans.Score >= 3)
    is_neg = (~ans.acc) & (ans.Score <= 0)
    ans = ans[(is_pos | is_neg) & ans.ParentId.astype(np.int64).isin(qids)].copy()
    ans["label"] = (ans.acc & (ans.Score >= 3)).astype(int)
    ans["slen"] = ans.Body.fillna("").map(strip_html).str.len()
    ans = ans[ans.slen >= 50]
    ans["detected"] = ans.Body.fillna("").str.contains(CODE_BLOCK_RE)
    print(f"[{slice_name}] candidates={len(ans)} detected_rate={ans.detected.mean():.4f}")

    rng = np.random.default_rng(7)
    rows = []
    for det in (True, False):
        sub = ans[ans.detected == det]
        take = sub.iloc[rng.choice(len(sub), 50, replace=False)]
        for _, r in take.iterrows():
            rows.append({
                "slice": slice_name, "answer_id": int(r.Id),
                "detected": bool(det), "label": int(r.label),
                "score": int(r.Score), "year": int(r.year),
                "body": str(r.Body)[:4000],
            })
    out = OUT / f"dd_codedetect_sample_{slice_name}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{slice_name}] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    run("js", BASE / "stackoverflow_js/so_js_questions.parquet",
        BASE / "stackoverflow_js/so_js_answers.parquet")
    run("sql", BASE / "stackoverflow_sql/so_sql_questions.parquet",
        BASE / "stackoverflow_sql/so_sql_answers.parquet")

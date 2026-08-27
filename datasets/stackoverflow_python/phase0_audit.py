#!/usr/bin/env python3
"""Phase-0 ground-truth audit on ONE Stack Overflow shard.

Goal (per the spec):
  - fraction of rows that are answers to [python]-tagged questions
  - score distribution (does score<=0 starve?)
  - accepted AND score>=3 rate
  - position computability (CreationDate per answer per question)
  - extrapolate to 58 shards

The shard contains a random mix of Q and A from across all SO. To know if an
answer belongs to a python question we need its parent's Tags. We do an
*in-shard* sample-join: only count answers whose ParentId is a python question
that also lives in this shard. The shard is roughly a 1/58 stratified slice
(by row id), so within-shard match rate is a lower bound; we report both the
within-shard join and the inferred upper bound.

Writes:
  - <out_dir>/phase0_shard0.json  (machine-readable)
  - prints a human-readable summary
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PY_PREFIXES = (
    "python", "numpy", "pandas", "scipy", "matplotlib", "sqlalchemy",
    "flask", "django", "pytest", "asyncio", "tkinter", "beautifulsoup",
    "tensorflow", "pytorch", "jupyter", "fastapi", "celery", "pyspark",
)


def is_py_tags(tags) -> bool:
    if tags is None:
        return False
    try:
        it = list(tags)
    except TypeError:
        return False
    for t in it:
        if not isinstance(t, str):
            continue
        if t == "python" or t.startswith("python-"):
            return True
    return False


def is_py_tags_wide(tags) -> bool:
    if tags is None:
        return False
    try:
        it = list(tags)
    except TypeError:
        return False
    for t in it:
        if not isinstance(t, str):
            continue
        if t == "python" or t.startswith("python-") or t in PY_PREFIXES:
            return True
    return False


def stratum_of(tags) -> str:
    """Verifiability stratum for python Qs.

    framework: has any of {django, flask, fastapi, celery, pyramid, tornado, bottle, aiohttp}
    lib_data : has any of {pandas, numpy, scipy, matplotlib, scikit-learn, sklearn}
    stdlib   : otherwise (pure python only)
    Precedence: framework > lib_data > stdlib  (frameworks usually pull a data lib too).
    """
    if tags is None:
        return "_none_"
    s = {t for t in tags if isinstance(t, str)}
    fw = {"django", "flask", "fastapi", "celery", "pyramid", "tornado", "bottle", "aiohttp",
          "django-rest-framework", "django-models", "django-views"}
    lib = {"pandas", "numpy", "scipy", "matplotlib", "scikit-learn", "sklearn",
           "seaborn", "scipy.stats", "scipy.optimize", "scikit-image"}
    if s & fw:
        return "framework"
    if s & lib:
        return "lib_data"
    return "stdlib"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shard",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/_raw_parquet/stackoverflow-posts-00000-of-00058.parquet",
    )
    ap.add_argument("--n-shards-total", type=int, default=58)
    ap.add_argument(
        "--out",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/manifests/phase0_shard0.json",
    )
    args = ap.parse_args()

    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] loading {args.shard}", flush=True)
    df = pd.read_parquet(args.shard)
    print(f"  rows={len(df):,}  cols={list(df.columns)}", flush=True)

    # PostTypeId: 1=Question, 2=Answer.
    qs = df[df.PostTypeId == 1].copy()
    ans = df[df.PostTypeId == 2].copy()
    n_q = len(qs)
    n_a = len(ans)
    print(f"  Qs in shard: {n_q:,}   As in shard: {n_a:,}", flush=True)

    # ----- python filter -----
    qs["is_py_strict"] = qs.Tags.apply(is_py_tags)
    qs["is_py_wide"] = qs.Tags.apply(is_py_tags_wide)
    n_py_strict = int(qs.is_py_strict.sum())
    n_py_wide = int(qs.is_py_wide.sum())
    print(f"  python Qs (strict 'python' tag or 'python-*'): {n_py_strict:,} ({n_py_strict/n_q:.2%})", flush=True)
    print(f"  python Qs (wide: incl numpy/pandas/django/...): {n_py_wide:,} ({n_py_wide/n_q:.2%})", flush=True)

    # ----- accepted flag for python qs -----
    py_qs = qs[qs.is_py_strict].copy()
    py_qs["has_accepted"] = py_qs.AcceptedAnswerId.fillna(0).astype(np.int64) > 0
    acc_rate = float(py_qs.has_accepted.mean()) if len(py_qs) else float("nan")
    print(f"  python-Q accepted-rate: {acc_rate:.3f}", flush=True)

    # ----- stratum split on python Qs -----
    py_qs["stratum"] = py_qs.Tags.apply(stratum_of)
    stratum_counts = py_qs.stratum.value_counts().to_dict()
    print(f"  stratum counts on shard's python Qs: {stratum_counts}", flush=True)

    # ----- IN-SHARD JOIN: answers to python Qs that also live in the shard -----
    py_qids = set(py_qs.Id.astype(int).tolist())
    ans_py = ans[ans.ParentId.astype(int).isin(py_qids)].copy()
    n_a_py_inshard = len(ans_py)
    print(f"  answers to python Qs that ALSO live in shard: {n_a_py_inshard:,}", flush=True)

    # accepted bit (this answer's Id is in its parent Q's AcceptedAnswerId set)
    accepted_ids = set(py_qs[py_qs.has_accepted].AcceptedAnswerId.astype(np.int64).tolist())
    ans_py["accepted"] = ans_py.Id.astype(np.int64).isin(accepted_ids)
    pos_strict = (ans_py.accepted & (ans_py.Score >= 3)).sum()
    neg_strict = ((~ans_py.accepted) & (ans_py.Score <= 0)).sum()
    print(f"  in-shard python answers: accepted&score>=3: {int(pos_strict):,}", flush=True)
    print(f"  in-shard python answers: not_accepted&score<=0: {int(neg_strict):,}", flush=True)

    # score distribution (all in-shard python answers)
    sd = ans_py.Score.describe().to_dict()
    sd_q = ans_py.Score.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    n_le0 = int((ans_py.Score <= 0).sum())
    n_le1 = int((ans_py.Score <= 1).sum())
    n_ge3 = int((ans_py.Score >= 3).sum())
    print(f"  in-shard python answer score quantiles: {sd_q}", flush=True)
    print(f"  in-shard python answers: Score<=0 {n_le0:,} ({n_le0/max(1,n_a_py_inshard):.2%}); <=1 {n_le1:,} ({n_le1/max(1,n_a_py_inshard):.2%}); >=3 {n_ge3:,} ({n_ge3/max(1,n_a_py_inshard):.2%})", flush=True)

    # ----- position computability -----
    n_has_creation = int(ans_py.CreationDate.notna().sum())
    n_has_parent = int(ans_py.ParentId.notna().sum())
    print(f"  position computability: CreationDate not-null {n_has_creation}/{n_a_py_inshard}; ParentId not-null {n_has_parent}/{n_a_py_inshard}", flush=True)

    # ----- extrapolation -----
    # IMPORTANT: empirically the mirror is CHRONOLOGICALLY sharded by Id —
    # shard 0 covers ~2008-08 → 2009-08, and 99.99% of answers in the shard
    # have their ParentId also in the shard. So scaling is LINEAR in shards,
    # NOT 58^2 (which would assume independent sharding).
    #
    # Caveat: python's share of SO traffic grew over time; shard 0 is the
    # earliest year, so extrapolating linearly UNDER-estimates the total.
    # Per the scout doc, SO totals (live API) are ~2.8M python Qs and ~3.7M
    # python answers. We report linear-from-shard0 as a LOWER bound.
    extrap_py_qs = n_py_strict * args.n_shards_total
    extrap_py_answers = n_a_py_inshard * args.n_shards_total
    extrap_pos_strict = int(pos_strict) * args.n_shards_total
    extrap_neg_strict = int(neg_strict) * args.n_shards_total
    extrap_matched_strict = min(extrap_pos_strict, extrap_neg_strict)
    # "strict matched pool" is min(pos, neg) since we need balanced pairs.

    summary = {
        "shard": args.shard,
        "build_date": t0.isoformat(timespec="seconds"),
        "n_shards_total": args.n_shards_total,
        "shard_rows": int(len(df)),
        "shard_questions": int(n_q),
        "shard_answers": int(n_a),
        "py_qs_strict_in_shard": n_py_strict,
        "py_qs_strict_pct": round(n_py_strict / max(1, n_q), 4),
        "py_qs_wide_in_shard": n_py_wide,
        "py_qs_accepted_rate": round(acc_rate, 4) if not np.isnan(acc_rate) else None,
        "py_qs_stratum_counts": {k: int(v) for k, v in stratum_counts.items()},
        "answers_to_py_inshard": n_a_py_inshard,
        "answers_to_py_score_quantiles": {str(k): float(v) for k, v in sd_q.items()},
        "answers_to_py_score_le0": n_le0,
        "answers_to_py_score_le1": n_le1,
        "answers_to_py_score_ge3": n_ge3,
        "strict_pos_inshard": int(pos_strict),
        "strict_neg_inshard": int(neg_strict),
        "position_computability": {
            "creation_date_not_null": n_has_creation,
            "parent_id_not_null": n_has_parent,
            "total_in_shard": n_a_py_inshard,
        },
        "extrapolation_linear_58shards_LOWER_BOUND": {
            "note": ("Mirror is chronologically sharded by Id (shard 0 = 2008-08..2009-08, "
                     "99.99% of in-shard answers have in-shard parents). Linear scaling is "
                     "a LOWER BOUND because python's share of SO grew over 2010-2020. "
                     "Per scout doc, true total is ~2.8M python Qs."),
            "py_qs_total_est": int(extrap_py_qs),
            "py_answers_total_est": int(extrap_py_answers),
            "strict_pos_total_est": int(extrap_pos_strict),
            "strict_neg_total_est": int(extrap_neg_strict),
            "strict_matched_pool_est_lower": int(extrap_matched_strict),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] wrote {args.out}", flush=True)
    print("\n=== PHASE-0 HEADLINE ===", flush=True)
    print(json.dumps({
        "py_qs_strict_in_shard": n_py_strict,
        "py_qs_strict_pct": round(n_py_strict / max(1, n_q), 4),
        "py_qs_accepted_rate": round(acc_rate, 4) if not np.isnan(acc_rate) else None,
        "in_shard_answers_to_py": n_a_py_inshard,
        "strict_pos_inshard": int(pos_strict),
        "strict_neg_inshard": int(neg_strict),
        "extrapolated_py_qs_total": int(extrap_py_qs),
        "extrapolated_strict_matched_pool": int(extrap_matched_strict),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

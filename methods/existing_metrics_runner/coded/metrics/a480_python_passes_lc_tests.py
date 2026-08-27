"""a480: Python candidate passes its LeetCode tests.

Binary: 1.0 if the candidate solution, wrapped against its LeetCode
problem's test fixture, passes all asserts within the timeout; 0.0 if it
runs but fails / errors; None if no Python file or no precomputed lookup.

Because LeetCode tests are external to the diff, this metric is *not*
self-contained in the way most THIN metrics are. The actual test
execution is performed offline by
`scripts/run_lc_python_runnability.py`, which writes results keyed by
the SHA-1 of the candidate's code text. At score-time we look up the
hash and return the cached pass/fail.

If the lookup table is not present, returns None — this lets the metric
participate cleanly in the discovery pipeline without forcing every
caller to provide LC tests.

Classification: THIN. Cost ~5s per candidate when computing; O(1) at score.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a480"
ASPECT_NAME = "Python passes LeetCode tests"
TIER = 2
TOOLS = ["python3"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

# Lookup parquet: columns `code_hash, a480, a481, a482`.
_LOOKUP_PATHS = [
    os.environ.get("LC_RUNNABILITY_PARQUET", ""),
    "/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/"
    "lc_python_runnability_scores.parquet",
]
_CACHE: Optional[Dict[str, float]] = None
_FIELD = "a480"


def _hash_code(text: str) -> str:
    # Match scripts/run_lc_python_runnability.py canonicalization:
    # strip whitespace + lowercase + replace \r.
    norm = text.replace("\r", "").strip()
    return hashlib.sha1(norm.encode("utf-8", errors="replace")).hexdigest()


def _load_cache() -> Dict[str, float]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _CACHE = {}
    for p in _LOOKUP_PATHS:
        if not p:
            continue
        try:
            path = Path(p)
            if not path.exists():
                continue
            import pandas as pd
            df = pd.read_parquet(path)
            if "code_hash" not in df.columns or _FIELD not in df.columns:
                continue
            _CACHE = {
                str(h): (float(v) if v is not None and v == v else None)
                for h, v in zip(df["code_hash"], df[_FIELD])
            }
            return _CACHE
        except Exception:
            continue
    return _CACHE


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    cache = _load_cache()
    if not cache:
        return None
    vals = []
    for content in by_path.values():
        h = _hash_code(content)
        v = cache.get(h)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))

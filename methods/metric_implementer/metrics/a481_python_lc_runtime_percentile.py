"""a481: Within-problem runtime percentile of a passing LC candidate.

Continuous in [0, 1] (1.0 = fastest). NaN/None when a480=0, when the
problem cell has fewer than 3 passing candidates, or when no lookup
table is present.

Computed offline by `scripts/run_lc_python_runnability.py` and cached
keyed by SHA-1 of normalized code text.

Classification: THIN. Cost ~5s per candidate when computing; O(1) at score.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a481"
ASPECT_NAME = "Python LC runtime percentile (within-problem)"
TIER = 2
TOOLS = ["python3"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_LOOKUP_PATHS = [
    os.environ.get("LC_RUNNABILITY_PARQUET", ""),
    "/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/"
    "lc_python_runnability_scores.parquet",
]
_CACHE: Optional[Dict[str, float]] = None
_FIELD = "a481"


def _hash_code(text: str) -> str:
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

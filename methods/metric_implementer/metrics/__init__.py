"""Per-norm verification metrics for code_review.

Each module defines ASPECT_ID, TIER, TOOLS, CLASSIFICATION,
APPLIES_TO_LANGS, applies(diff_text), score(diff_text).

Discover all loaded metrics via load_all().
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List


def load_all() -> List:
    out = []
    pkg_path = Path(__file__).parent
    for m in pkgutil.iter_modules([str(pkg_path)]):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{m.name}")
        if hasattr(mod, "ASPECT_ID") and hasattr(mod, "score"):
            out.append(mod)
    return out

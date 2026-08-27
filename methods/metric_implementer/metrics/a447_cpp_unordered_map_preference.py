"""a447: C++ unordered_map / unordered_set preference.

For each occurrence of an associative-container *declaration*, count
whether it is the hashed (``unordered_map`` / ``unordered_set``) or the
ordered (``map`` / ``set``) flavour. Returns the fraction that are
hashed.

LC-community style prefers ``unordered_*`` for O(1) lookups; industrial
review style sometimes keeps ``map`` for stable ordering.

Returns NaN when no associative container is declared.

Tier 2. THIN.
"""
from __future__ import annotations
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a447"
ASPECT_NAME = "C++ unordered_map preference"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]

# REGEX_OK: tool_output — surface-syntax declarations of associative
# containers. We require that the type be immediately followed by ``<``
# to avoid catching a variable named ``map`` and to avoid matching
# ``std::map<...>`` twice via the ``std::`` prefix.
RE_UNORDERED = re.compile(r"\b(?:std::)?unordered_(?:map|set)\s*<")
RE_ORDERED = re.compile(r"(?<![A-Za-z0-9_])(?<!unordered_)(?:std::)?(?:multi)?(?:map|set)\s*<")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    n_unordered = 0
    n_ordered = 0
    for content in by_path.values():
        n_unordered += len(RE_UNORDERED.findall(content))
        n_ordered += len(RE_ORDERED.findall(content))
    total = n_unordered + n_ordered
    if total == 0:
        return None
    return n_unordered / total

"""a510: `#include <bits/stdc++.h>` presence.

Olympiad-flavored C++ shortcut. Binary signal:
  1.0 if any added C/C++ file includes `<bits/stdc++.h>`,
  0.0 otherwise. Abstain on non-C++ diffs.

CLASSIFICATION: THIN — exact textual match.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a510"
ASPECT_NAME = "bits/stdc++.h include"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]
_PAT = re.compile(r"#\s*include\s*[<\"]bits/stdc\+\+\.h[>\"]")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    for content in by_path.values():
        if _PAT.search(content):
            return 1.0
    return 0.0

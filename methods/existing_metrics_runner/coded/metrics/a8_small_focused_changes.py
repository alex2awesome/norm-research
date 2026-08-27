"""a8: Small, focused, reviewable changes — PARTIALLY_THIN structural surrogate.

The norm: keep CLs well-scoped and single-purpose; prefer small, incremental
changes that ease review, merging, and rollback. Split large or mixed changes.

True "focus" / "single-purpose-ness" is judgment — two unrelated bug fixes
of three lines each look identical, structurally, to one bug fix of six
lines in two files. What we CAN measure structurally is *size* and *spread*:

  S1. Size penalty.  A small CL is easier to review. Many engineering style
      guides put a soft cap around ~200-400 changed lines; review quality
      drops sharply above that. We use a smooth decay on total changed
      lines (added + removed): size_score = exp(-changed / SIZE_TAU).
      SIZE_TAU = 300 lines (≈ Google's CL-size guidance). 0 lines → 1.0;
      300 → 0.37; 1000 → 0.04.

  S2. File-count penalty.  Touching many files makes a CL harder to keep
      in working memory during review. files_score =
      exp(-n_files / FILES_TAU), FILES_TAU = 8. 1 file → 0.88; 8 → 0.37;
      30 → 0.02.

  S3. Top-level-package focus.  A focused CL touches one or two areas of
      the codebase, not many. We compute the number of distinct top-level
      path segments touched (e.g. `src/`, `tests/`, `docs/` count as 3),
      and decay: focus_score = exp(-(n_top - 1) / TOP_TAU), TOP_TAU = 2.
      1 top-level dir → 1.0; 3 → 0.37; 6 → 0.08.
      (A diff that spans many unrelated packages is the canonical
      "unfocused" signature.)

Combination (each component ∈ [0,1]; 1 = norm satisfied):

    score = 0.50 * size_score + 0.25 * files_score + 0.25 * focus_score

Size is weighted highest because it is the most discussed and most
empirically-grounded reviewability dimension.

Why PARTIALLY_THIN, not THIN:
  - "Single-purpose" is the heart of the norm and is irreducibly judgment:
    one diff can do 12 things in 30 lines and look small, or do one thing
    cleanly across 50 files and look spread out.
  - Top-level-package count is a proxy for spread, not for thematic unity.
    A monorepo refactor that legitimately touches many packages will score
    poorly here even when perfectly focused.
  - Size is structural — bigger diffs ARE harder to review — so that
    component is THIN. The other two are partial proxies for focus.

THICK what we don't measure:
  - Whether the changes share a single purpose.
  - Whether the CL could be split into smaller independent CLs.
  - Whether risky changes are isolated from low-risk changes.

applies(): True iff the diff contains at least one parseable file change.
Pure documentation-only diffs or empty diffs do not pose a size/focus
trade-off the same way; we still apply, but score them by their numbers.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import whatthepatch

ASPECT_ID = "a8"
ASPECT_NAME = "Small, focused, reviewable changes"
TIER = 2
TOOLS = []  # diff parser only (whatthepatch)
APPLIES_TO_LANGS = ["*"]  # language-agnostic
CLASSIFICATION = "PARTIALLY_THIN"

SIZE_TAU = 300.0   # changed lines at which size_score = 1/e ≈ 0.37
FILES_TAU = 8.0    # file count at which files_score = 1/e ≈ 0.37
TOP_TAU = 2.0      # (top_dirs - 1) at which focus_score = 1/e ≈ 0.37


def _parse(diff_text: str) -> List[Tuple[str, int, int]]:
    """Return [(path, n_added, n_removed)] for each file in the diff.

    Skips /dev/null endpoints' partner side (we keep the real path) and
    files with no changes. Tolerates truncation — whatthepatch is robust.
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return []
    try:
        diffs = list(whatthepatch.parse_patch(diff_text[idx:]))
    except Exception:
        return []
    out: List[Tuple[str, int, int]] = []
    for d in diffs:
        if d is None:
            continue
        new_path = d.header.new_path or ""
        old_path = d.header.old_path or ""
        if new_path.startswith("b/"):
            new_path = new_path[2:]
        if old_path.startswith("a/"):
            old_path = old_path[2:]
        # Prefer the non-/dev/null side.
        if new_path and new_path != "/dev/null":
            path = new_path
        elif old_path and old_path != "/dev/null":
            path = old_path
        else:
            continue
        if not path:
            continue
        n_add = 0
        n_rem = 0
        for ch in (d.changes or []):
            if ch.line is None:
                continue
            if ch.old is None and ch.new is not None:
                n_add += 1
            elif ch.new is None and ch.old is not None:
                n_rem += 1
        if n_add == 0 and n_rem == 0:
            continue
        out.append((path, n_add, n_rem))
    return out


def _top_segment(path: str) -> str:
    """First path component; "" if none. Used to count cross-area spread.

    For paths like "src/foo/bar.py" -> "src"; "README.md" -> "" (we treat
    repo-root files as their own group via a special "<root>" tag below).
    """
    # Normalize separators (diffs use forward slash).
    parts = path.split("/")
    if len(parts) <= 1:
        return "<root>"
    return parts[0]


def applies(diff_text: str) -> bool:
    """True iff the diff contains at least one file with parseable changes."""
    return bool(_parse(diff_text))


def score(diff_text: str) -> Optional[float]:
    files = _parse(diff_text)
    if not files:
        return None

    n_files = len(files)
    total_changed = sum(a + r for _, a, r in files)
    top_dirs = {_top_segment(p) for p, _, _ in files}
    n_top = len(top_dirs)

    # S1: size decay.
    size_score = math.exp(-total_changed / SIZE_TAU)
    # S2: file-count decay.
    files_score = math.exp(-n_files / FILES_TAU)
    # S3: top-level-package focus. 1 top-level dir → 1.0.
    focus_score = math.exp(-max(n_top - 1, 0) / TOP_TAU)

    result = 0.50 * size_score + 0.25 * files_score + 0.25 * focus_score
    return float(max(0.0, min(1.0, result)))

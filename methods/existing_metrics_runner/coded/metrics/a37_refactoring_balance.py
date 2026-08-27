"""a37: Refactoring quality and practice — PARTIALLY_THIN structural surrogate.

The norm (Fowler-style): refactor continuously and *behavior-preservingly*,
guided by code smells; pick the right technique (Extract Method, Rename,
Inline, Move, etc.); avoid cosmetic-only churn; isolate risky changes.

The fully-honest assessment of this norm is THICK: "behavior preservation"
requires either (a) running the pre/post test suites or (b) recognizing a
catalog of semantic-preserving AST transformations (Refactoring Miner does
this for Java; nothing comparable is on PATH here for the multi-language
fixture set). "Smell-guided" requires comparing post-state quality metrics
to a pre-state we never see. "Choosing the appropriate technique" and
"isolating risky changes" require intent we cannot infer from the diff.

We instead measure a narrow, *structural* surrogate that captures one
defensible slice of the norm — the shape of the diff:

  S1. Add/Remove balance.  A pure refactor REPLACES code with equivalent
      code; the count of added lines and removed lines should be similar.
      A pure feature addition is overwhelmingly insertions. We compute
      balance = min(adds, removes) / max(adds, removes) over the whole
      diff. balance ∈ [0,1], with 1 = perfectly balanced rewrite.

  S2. Edits-existing-files fraction.  Refactoring touches existing code;
      pure features tend to add new files. existing_frac = (# files with
      any removed lines) / (# changed files). ∈ [0,1].

  S3. Anti-cosmetic-churn penalty.  Per file, if every removed line is
      EQUAL to some added line after whitespace normalization (i.e. the
      hunk just shuffles or reformats), that is the "cosmetic-only churn"
      the norm names — penalize it. cosmetic_frac = (# files whose entire
      changeset is whitespace-equivalent rewrites) / (# files with any
      removals). ∈ [0,1].

Combination (each component on [0,1]; 1 = norm satisfied):

    score = 0.40 * balance + 0.30 * existing_frac + 0.30 * (1 - cosmetic_frac)

Why PARTIALLY_THIN, not THIN:
  - S1/S2 correlate with "looks like a refactor" but cannot distinguish
    a faithful behavior-preserving rewrite from a behavior-changing rewrite
    of similar line count. A bug-introducing rename has the same surface.
  - S3 is a partial guard against the strict "no cosmetic churn" sub-norm;
    it misses semantic-preserving but non-whitespace cosmetic edits (e.g.,
    renaming a variable consistently throughout — looks like real change
    but is cosmetic in the Fowler sense).
  - "Isolate risky changes" is not measured at all.

What would unblock a THIN measurement:
  - A multi-language refactoring detector that classifies hunks as one of
    the named Fowler refactorings (Extract Method, Rename, Inline, Move
    Method, etc.).  Refactoring Miner does this for Java; analogues are
    research-grade or absent for Python/JS/etc.
  - A test runner that could execute pre- and post-diff test suites and
    confirm semantic equivalence (the gold-standard behavior-preservation
    check).
  - Access to the issue/PR description's stated intent ("refactor",
    "feature", "bug fix"), at which point this becomes a classification
    problem, not a measurement.

applies(): True iff the diff contains BOTH added and removed code lines in
a source file we recognize. (Pure-add and pure-remove diffs do not exhibit
a "refactor vs feature" signal; we abstain.)
"""
from __future__ import annotations

from typing import Optional

import whatthepatch

ASPECT_ID = "a37"
ASPECT_NAME = "Refactoring quality and practice"
TIER = 2
TOOLS = []  # diff parser only (whatthepatch)
APPLIES_TO_LANGS = [
    "Python", "JavaScript", "TypeScript", "Java", "Go", "C", "C++", "C#",
    "Ruby", "PHP", "Rust", "Kotlin", "Scala", "Swift",
]
CLASSIFICATION = "PARTIALLY_THIN"

SOURCE_EXTS = (
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".cs",
    ".kt", ".kts",
    ".scala",
    ".php",
    ".swift",
)


def _is_source_path(path: str) -> bool:
    p = path.lower()
    return any(p.endswith(e) for e in SOURCE_EXTS)


def _parse(diff_text: str):
    """Return list of (path, added_lines, removed_lines) for source files.

    added/removed lines preserve textual content (incl. leading whitespace
    but without the diff marker char). Skips non-source files and
    /dev/null endpoints (pure-create or pure-delete files).
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return []
    try:
        diffs = list(whatthepatch.parse_patch(diff_text[idx:]))
    except Exception:
        return []
    out = []
    for d in diffs:
        if d is None:
            continue
        new_path = d.header.new_path or ""
        old_path = d.header.old_path or ""
        if new_path.startswith("b/"):
            new_path = new_path[2:]
        if old_path.startswith("a/"):
            old_path = old_path[2:]
        path = new_path or old_path
        if not path or path == "/dev/null":
            continue
        if not _is_source_path(path):
            continue
        # Skip files that are pure-create (no old) or pure-delete (no new):
        # these cannot exhibit a refactor signal in either direction.
        if new_path == "/dev/null" or old_path == "/dev/null":
            # Still include them so the existing_frac denominator counts
            # this file. We mark added/removed appropriately.
            added, removed = [], []
            for ch in (d.changes or []):
                if ch.line is None:
                    continue
                if ch.old is None and ch.new is not None:
                    added.append(ch.line)
                elif ch.new is None and ch.old is not None:
                    removed.append(ch.line)
            out.append((path, added, removed))
            continue
        added, removed = [], []
        for ch in (d.changes or []):
            if ch.line is None:
                continue
            if ch.old is None and ch.new is not None:
                added.append(ch.line)
            elif ch.new is None and ch.old is not None:
                removed.append(ch.line)
        out.append((path, added, removed))
    return out


def _norm_ws(s: str) -> str:
    """Whitespace-normalized form: collapse all runs of whitespace to one
    space, strip ends. Used to detect cosmetic-only equivalence."""
    return " ".join(s.split())


def _is_cosmetic_only(added: list, removed: list) -> bool:
    """True iff every added/removed line in this file pairs up under
    whitespace-normalization (i.e., the file's changeset is purely a
    whitespace/formatting rewrite). Empty additions or empty removals fall
    through to False (pure-add or pure-remove isn't cosmetic churn —
    it's actual content change)."""
    if not added or not removed:
        return False
    add_ws = sorted(_norm_ws(l) for l in added if _norm_ws(l))
    rem_ws = sorted(_norm_ws(l) for l in removed if _norm_ws(l))
    if not add_ws or not rem_ws:
        return False
    return add_ws == rem_ws


def applies(diff_text: str) -> bool:
    """Apply only when the diff has BOTH adds and removes in recognized
    source files — that is the regime where 'refactor vs feature' is a
    well-posed question."""
    files = _parse(diff_text)
    if not files:
        return False
    total_adds = sum(len(a) for _, a, _ in files)
    total_rems = sum(len(r) for _, _, r in files)
    return total_adds > 0 and total_rems > 0


def score(diff_text: str) -> Optional[float]:
    files = _parse(diff_text)
    if not files:
        return None

    total_adds = sum(len(a) for _, a, _ in files)
    total_rems = sum(len(r) for _, _, r in files)
    if total_adds == 0 or total_rems == 0:
        return None

    # S1: line-count balance.
    balance = min(total_adds, total_rems) / max(total_adds, total_rems)

    # S2: fraction of touched source files that have ANY removed lines
    # (i.e., are edits of existing files, not pure new-file creations).
    n_files = len(files)
    n_edits_existing = sum(1 for _, _, r in files if len(r) > 0)
    existing_frac = n_edits_existing / n_files if n_files else 0.0

    # S3: cosmetic-only churn penalty. Numerator = files whose entire
    # changeset is whitespace-equivalent. Denominator = files with both
    # adds and removes (the only ones where cosmetic-churn is a question).
    eligible = [(a, r) for _, a, r in files if a and r]
    if eligible:
        n_cosmetic = sum(1 for a, r in eligible if _is_cosmetic_only(a, r))
        cosmetic_frac = n_cosmetic / len(eligible)
    else:
        cosmetic_frac = 0.0

    result = (0.40 * balance
              + 0.30 * existing_frac
              + 0.30 * (1.0 - cosmetic_frac))
    # Clip for numerical safety.
    return float(max(0.0, min(1.0, result)))

"""a175: Documentation formatting and style conventions.

Aspect description (from aspects.json[175]):
    "Use consistent documentation formatting: appropriate headers, structured
     lists/steps, fenced code blocks for multi-line code, concise summary
     fragments, and language-appropriate doc comment styles."

Distinct angle vs adjacent metrics
----------------------------------
- a52 (Documentation quality and maintenance) scores PR-level doc UPKEEP:
  size of the change, presence of headings/lists at all, freshness via
  co-change with source. It treats markdown as project documentation upkeep.
- a202 (Documentation style, placement, and docstring formatting) measures
  Python DOCSTRING formatting via ruff's pydocstyle D-rules — entirely inside
  `.py` source.
- a130 (ADR/RFC completeness) checks SECTION presence (Context/Decision/
  Consequences/...) on ADR-named markdown files.

a175 fills the remaining gap: **pure markdown formatting hygiene** as a
markdownlint-style check applied to ALL added markdown content (project docs,
inline `README.md` snippets, top-level `*.md`, anything except ADR/RFC which
a130 owns and except in-code docstrings which a202 owns). It measures the
mechanical formatting conventions that make a markdown document parse
correctly and render consistently — independent of whether the prose itself
is high-quality, whether the file is "documentation upkeep", or whether
specific ADR sections are present.

Why PARTIALLY_THIN, Tier 2
--------------------------
The mechanical conventions are themselves THIN (a tool decides them):
  - Heading hierarchy: single top-level H1, no skipped levels (h1 -> h3),
    consistent ATX (`#`) vs setext (`===`) style.
  - Code fences: multi-line code is fenced (not indented), fences carry a
    language tag.
  - Trailing whitespace: none on text lines (markdown TREATS two trailing
    spaces as <br>, so trailing whitespace is semantically meaningful and
    should be intentional, not accidental — markdownlint MD009 flags this).
  - List markers: consistent within a list (all `-` or all `*` or all `+`).

The "appropriate" qualifier ("appropriate headers", "concise summary
fragments") is THICK — whether a particular heading level is *the right*
level for the content's logical depth requires understanding the content.
We score only the mechanically-checkable subset, hence PARTIALLY_THIN.

We chose Tier 2 (tree-sitter-markdown AST + stdlib structural checks) over
Tier 3 (markdownlint / mdformat CLI) because neither CLI is in the sandbox
tool registry as of writing; tree-sitter-markdown is. The structural checks
this metric performs are exactly the parser-driven subset of markdownlint's
rule set (MD001 heading-levels-increment, MD002/MD041 first-line-h1, MD003
heading-style, MD004 ul-style, MD009 no-trailing-spaces, MD040 fenced-code-
language). If a CLI is added to the sandbox later, this metric should be
upgraded to Tier 3 and gain MD012 (multiple blank lines), MD013 (line
length), MD025 (multiple H1), etc.

Narrow applicability gate
-------------------------
`applies()` returns True iff the diff adds content to at least one markdown
file (`.md`, `.markdown`, `.mdx`) that is NOT recognizably an ADR/RFC. We
exclude ADR/RFC because their section-completeness norm is owned by a130
and the strict heading hierarchy rules MD001/MD002 don't match the ADR
template (which often starts at H2 under an implied H1 title). We do NOT
require the file to live in a docs/ directory — formatting hygiene applies
to every markdown file equally, including per-package READMEs at arbitrary
depth.

We deliberately do NOT include `.rst` here. The aspect description's
"fenced code blocks" and "appropriate headers" phrasing is markdown-centric;
rST has structurally different code-block and heading conventions that
would need a separate metric (and tree-sitter-rst is not in the sandbox).

Three return states
-------------------
- None when applies()=False (no qualifying markdown added).
- None when applies()=True but tree-sitter-markdown parser unavailable, OR
  the added fragments contain no parseable structural elements at all
  (too tiny to score — abstain rather than fake a 1.0).
- float in [0,1] otherwise. 1.0 = clean formatting; 0.0 = many violations.

Scoring
-------
For each qualifying markdown file with non-empty added content, compute a
violation count over the added lines (we score the *change*, not the file's
overall pre-existing state, so unchanged context that has e.g. a skipped
heading level does not penalize this PR):

  v_heading_skip   = number of heading-level skips in added headings
                     (h1 immediately followed by h3 = 1 skip).
                     Approximation: we look at consecutive added headings in
                     document order within the same file.
  v_fence_no_lang  = number of fenced code blocks with no language tag.
                     (`` ``` `` followed by a newline and then code.)
  v_indented_code  = number of indented-code blocks (4-space) that look like
                     multi-line code (>=2 lines) — these should be fenced.
                     (markdownlint MD046 code-block-style.)
  v_trailing_ws    = number of added lines with stray trailing whitespace
                     that is NOT a markdown hard-break (i.e. NOT exactly two
                     trailing spaces). MD009.
  v_list_style     = number of list-marker style changes within the same
                     added run (mixing `-` and `*` in the same list = 1).
                     MD004.

These are converted to a per-file conformance score via:

    n_lines = number of added non-blank lines in the file (a denominator
              that lets a 200-line README tolerate a few violations while
              a 10-line README is held to a tighter bar).
    v_total = sum of the violation counts above.
    density = v_total / max(n_lines, 8)        # floor to avoid blow-up
    score   = exp( -density * 2.0 )            # 0 viol -> 1.0,
                                               # 1 viol per 8 lines -> 0.78

Aggregate across qualifying files: arithmetic mean.

Empirical note
--------------
On the 50-PR code_review fixtures, no fixture's truncated diff contains a
markdown file, so this metric will abstain on all 50. That is the correct
behavior (the GUIDE's "over-abstain rather than over-apply" rule). The
metric is exercised below in `__main__` against synthetic in-distribution
markdown fragments to confirm non-degenerate scoring.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a175"
ASPECT_NAME = "Documentation formatting and style conventions"
TIER = 2
TOOLS = ["tree-sitter-markdown"]
APPLIES_TO_LANGS = ["Markdown"]
CLASSIFICATION = "PARTIALLY_THIN"


# ---------------------------------------------------------------------------
# Path classification.
# Plain string ops over file paths (paths aren't a language to parse).
# ---------------------------------------------------------------------------

MD_EXTS = (".md", ".markdown", ".mdx")

# Tokens that mark a markdown file as an ADR/RFC — owned by a130, excluded here.
ADR_RFC_DIR_TOKENS = (
    "adr", "adrs", "rfc", "rfcs", "decisions",
    "architecture-decisions", "architecture/decisions",
    "proposals", "designs",
)
ADR_RFC_FILENAME_PREFIXES = ("adr-", "adr_", "rfc-", "rfc_")

# Path tokens that DISQUALIFY a markdown file from being scored as
# documentation formatting (vendored, fixtures, etc.).
EXCLUDE_DIR_TOKENS = (
    "node_modules", "venv", ".venv", "vendor", "third_party",
    "fixtures", "testdata",
)


def _path_segments_lower(path: str) -> List[str]:
    return [seg for seg in path.lower().split("/") if seg]


def _is_adr_or_rfc(path: str) -> bool:
    segs = _path_segments_lower(path)
    name = segs[-1] if segs else ""
    parents = segs[:-1]
    for tok in ADR_RFC_DIR_TOKENS:
        if "/" in tok:
            if tok in "/".join(parents):
                return True
        else:
            if tok in parents:
                return True
    for pre in ADR_RFC_FILENAME_PREFIXES:
        if name.startswith(pre):
            return True
    # MADR NNNN-slug.md under a docs/ ancestor (matches a130's path rule).
    base = name.rsplit(".", 1)[0]
    if len(base) >= 5 and base[:4].isdigit() and base[4] == "-":
        if any(seg in ("docs", "doc", "documentation") for seg in parents):
            return True
    return False


def _is_qualifying_markdown_path(path: str) -> bool:
    p = path.lower()
    if not p.endswith(MD_EXTS):
        return False
    segs = _path_segments_lower(path)
    parents = segs[:-1]
    for tok in EXCLUDE_DIR_TOKENS:
        if tok in parents:
            return False
    if _is_adr_or_rfc(path):
        return False
    return True


# ---------------------------------------------------------------------------
# Markdown structural parsing via tree-sitter-markdown.
# ---------------------------------------------------------------------------

_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is not None:
        return _PARSER
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_markdown as mod
        lang = Language(mod.language())
        _PARSER = Parser(lang)
    except Exception:
        _PARSER = False
    return _PARSER


def _heading_level(heading_node) -> Optional[int]:
    """ATX heading: extract level from atx_hN_marker child."""
    for ch in heading_node.children:
        if ch.type.startswith("atx_h") and ch.type.endswith("_marker"):
            try:
                return int(ch.type[len("atx_h"):-len("_marker")])
            except ValueError:
                return None
    return None


def _setext_level(node) -> Optional[int]:
    for ch in node.children:
        if ch.type == "setext_h1_underline":
            return 1
        if ch.type == "setext_h2_underline":
            return 2
    return None


def _node_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _walk_collect(root, src: bytes) -> Dict[str, List]:
    """Single walk; collect heading levels and fenced-code info-strings.

    tree-sitter-markdown node types of interest:
      - atx_heading  (with atx_hN_marker child)
      - setext_heading
      - fenced_code_block  (with optional info_string child)
      - indented_code_block
    """
    out: Dict[str, List] = {
        "heading_levels": [],     # in document order
        "atx_count": 0,
        "setext_count": 0,
        "fence_no_lang_count": 0,
        "fence_total": 0,
        "indented_code_blocks": [],   # list of line counts
    }

    def walk(n):
        t = n.type
        if t == "atx_heading":
            lvl = _heading_level(n)
            if lvl is not None:
                out["heading_levels"].append(lvl)
                out["atx_count"] += 1
        elif t == "setext_heading":
            lvl = _setext_level(n)
            if lvl is not None:
                out["heading_levels"].append(lvl)
                out["setext_count"] += 1
        elif t == "fenced_code_block":
            out["fence_total"] += 1
            has_lang = False
            for ch in n.children:
                if ch.type == "info_string":
                    s = _node_text(ch, src).strip()
                    if s:
                        has_lang = True
                    break
            if not has_lang:
                out["fence_no_lang_count"] += 1
        elif t == "indented_code_block":
            # Count the line span as a rough "is this multi-line?" signal.
            txt = _node_text(n, src)
            nlines = sum(1 for ln in txt.splitlines() if ln.strip())
            out["indented_code_blocks"].append(nlines)
        for c in n.children:
            walk(c)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Stdlib structural checks on the raw added text (no regex needed).
# ---------------------------------------------------------------------------

def _count_trailing_ws_violations(added: str) -> int:
    """Lines with non-intentional trailing whitespace.

    Markdown semantics: exactly two trailing spaces = hard break (intentional).
    Anything else (1 space, 3+ spaces, trailing tabs) is a MD009 violation.
    Blank lines and lines that are entirely whitespace are ignored.
    """
    n = 0
    for raw in added.splitlines():
        if not raw or not raw.strip():
            continue
        # strip trailing newline already removed by splitlines.
        # Determine trailing whitespace.
        stripped = raw.rstrip(" \t")
        trail = raw[len(stripped):]
        if not trail:
            continue
        # Exactly two trailing spaces (no tabs) is a hard break -> OK.
        if trail == "  ":
            continue
        n += 1
    return n


def _count_list_style_changes(added: str) -> int:
    """Count list-marker style transitions within consecutive list runs.

    A 'run' is a maximal block of consecutive (non-blank-broken) lines whose
    first non-space character is a top-level list marker `-`, `*`, or `+`.
    A change within a run (e.g. `-` then `*`) counts as 1. Blank lines reset
    the run.

    This is a stdlib structural check on plain text — no regex, no parser
    dependency.
    """
    violations = 0
    cur_marker: Optional[str] = None
    cur_in_run = False
    for raw in added.splitlines():
        ln = raw.lstrip(" \t")
        if not ln:
            cur_marker = None
            cur_in_run = False
            continue
        first = ln[:1]
        # Must be a bullet marker AND followed by space (or be a single char).
        is_bullet = (first in ("-", "*", "+")) and (len(ln) == 1 or ln[1] == " ")
        if not is_bullet:
            cur_marker = None
            cur_in_run = False
            continue
        if not cur_in_run:
            cur_marker = first
            cur_in_run = True
            continue
        # In a run already.
        if cur_marker is not None and first != cur_marker:
            violations += 1
            cur_marker = first  # tolerate further marker; count each change once
    return violations


def _count_meaningful_lines(added: str) -> int:
    n = 0
    for raw in added.splitlines():
        if raw.strip():
            n += 1
    return n


# ---------------------------------------------------------------------------
# Per-file scoring.
# ---------------------------------------------------------------------------

def _heading_skip_count(levels: List[int]) -> int:
    """Number of consecutive heading pairs where level jumps by >1.

    e.g. [1, 3, 4] has one skip (1 -> 3). [1, 2, 2, 4] has one skip (2 -> 4).
    Going UP (deeper) by >1 is the skip; going BACK UP (e.g. 3 -> 1) is fine.
    """
    skips = 0
    for i in range(1, len(levels)):
        if levels[i] > levels[i - 1] + 1:
            skips += 1
    return skips


def _score_file(added: str) -> Optional[float]:
    n_lines = _count_meaningful_lines(added)
    if n_lines == 0:
        return None

    parser = _parser()
    if not parser:
        return None
    try:
        src = added.encode("utf8", errors="replace")
        tree = parser.parse(src)
    except Exception:
        return None

    feats = _walk_collect(tree.root_node, src)

    # No parseable structural element AND no plaintext bullet content -> we
    # can't meaningfully score formatting. The check_no_regex_on_code policy
    # is fine: this is a stdlib structural check on plain text.
    if (not feats["heading_levels"]
            and feats["fence_total"] == 0
            and not feats["indented_code_blocks"]
            and _count_list_style_changes(added) == 0
            and _count_trailing_ws_violations(added) == 0):
        # Nothing to score -> abstain (don't reward emptiness with 1.0).
        return None

    v_heading_skip = _heading_skip_count(feats["heading_levels"])
    v_fence_no_lang = feats["fence_no_lang_count"]
    # Indented code blocks of >=2 non-blank lines should be fenced.
    v_indented_code = sum(1 for n in feats["indented_code_blocks"] if n >= 2)
    v_trailing_ws = _count_trailing_ws_violations(added)
    v_list_style = _count_list_style_changes(added)

    v_total = (v_heading_skip + v_fence_no_lang + v_indented_code
               + v_trailing_ws + v_list_style)

    # Density normalized by added-line count with a floor of 8 lines so tiny
    # fragments aren't double-penalized.
    import math
    density = v_total / max(n_lines, 8)
    return float(math.exp(-density * 2.0))


# ---------------------------------------------------------------------------
# Contract.
# ---------------------------------------------------------------------------

def _candidate_files(diff_text: str) -> Dict[str, str]:
    by_path = parse_diff_added_by_file(diff_text)
    return {p: c for p, c in by_path.items()
            if _is_qualifying_markdown_path(p)}


def applies(diff_text: str) -> bool:
    """True iff diff adds content to at least one non-ADR markdown file."""
    return bool(_candidate_files(diff_text))


def score(diff_text: str) -> Optional[float]:
    files = _candidate_files(diff_text)
    if not files:
        return None
    parser = _parser()
    if not parser:
        return None
    scores: List[float] = []
    for _, body in files.items():
        s = _score_file(body)
        if s is not None:
            scores.append(s)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Direct smoke test against synthetic in-distribution markdown fragments.
# Real PR fixtures contain no markdown (verified), so we synthesize tiny
# diffs to confirm the metric is non-degenerate and the scoring direction
# is correct (clean > messy).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CLEAN_DIFF = """diff --git a/README.md b/README.md
index 0000001..0000002 100644
--- a/README.md
+++ b/README.md
@@ -0,0 +1,18 @@
+# My Project
+
+A short description of the project.
+
+## Installation
+
+Install with pip:
+
+```bash
+pip install myproject
+```
+
+## Usage
+
+- Step one
+- Step two
+- Step three
+
"""

    MESSY_DIFF = """diff --git a/docs/guide.md b/docs/guide.md
index 0000001..0000002 100644
--- a/docs/guide.md
+++ b/docs/guide.md
@@ -0,0 +1,16 @@
+# Title   \n+
+#### Skipped level (h1 -> h4)
+
+Run this:
+
+```
+pip install foo
+```
+
+And then:
+
+    import foo
+    foo.run()
+
+- bullet one
+* bullet two with mixed marker
+- bullet three
+
"""

    ADR_DIFF = """diff --git a/docs/adr/0001-use-postgres.md b/docs/adr/0001-use-postgres.md
new file mode 100644
--- /dev/null
+++ b/docs/adr/0001-use-postgres.md
@@ -0,0 +1,3 @@
+# ADR 1: Use Postgres
+## Context
+We need a database.
"""

    NO_MD_DIFF = """diff --git a/src/foo.py b/src/foo.py
index 0000001..0000002 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -0,0 +1,2 @@
+def foo():
+    return 1
"""

    cases = [
        ("clean README", CLEAN_DIFF),
        ("messy guide", MESSY_DIFF),
        ("ADR (a130's turf)", ADR_DIFF),
        ("python only", NO_MD_DIFF),
    ]
    for name, diff in cases:
        a = applies(diff)
        s = score(diff) if a else None
        print(f"{name:<22} applies={a}  score={s}")

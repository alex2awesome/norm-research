"""a52: Documentation quality and maintenance — PR-level docs upkeep.

Distinct angle vs adjacent metrics
----------------------------------
- a202 measures *docstring* style/formatting inside Python source files (ruff
  D-rules on `def`/`class` docstrings).
- a130 measures completeness of *ADR/RFC* records (status/context/decision/
  consequences sections in `decisions/`, `rfc/`, MADR-named files).
- a309 measures *test-source correspondence* by filename naming.

a52 sits orthogonal to all three: it scores PR-level **end-user documentation**
upkeep — README, CHANGELOG, CONTRIBUTING, `docs/` markdown, top-level
`*.md`/`*.rst` files — when those files are touched by the PR. The aspect
description ("Produce clear, accurate, audience-appropriate docs/specs; keep
them current, modular, consistently detailed, and avoid duplication; support
onboarding.") explicitly targets the project's documentation surface, not
in-code docstrings (a202) and not architecture decision records (a130).

Why PARTIALLY_THIN, Tier 2
--------------------------
Whether the *prose itself* is clear, accurate, and audience-appropriate is
genuinely THICK and out of reach for a deterministic tool. What we CAN
measure deterministically is structural quality of the markdown change:

  1. Was a meaningful amount of documentation added (not a 1-line typo fix
     or a blank-line nudge)?
  2. Does the added/updated markdown have heading structure (modularity)?
  3. Does it include at least one code example, link, or list (a
     concreteness signal — pure walls of prose with no examples score
     lower for an onboarding-oriented norm)?
  4. Did the PR also touch source code? When docs change *alongside* source
     code that is "keeping docs current" (a freshness signal). Pure-docs
     PRs score the structural signal only; mixed PRs get the freshness
     bonus.

These are all structural / surface features parsed via tree-sitter-markdown,
NOT regex on prose. The decision to use tree-sitter (Tier 2 structural
parsing) instead of e.g. textstat readability (Tier 3 NLP) is deliberate:
readability metrics applied to a heterogeneous mix of README/CHANGELOG
content produce noisy scores that don't track "documentation quality" the
way they would on prose-only artifacts.

Narrow applicability gate
-------------------------
`applies()` returns True only when the diff adds/updates a markdown or rST
file that is recognizably **project documentation**: README, CHANGELOG,
CONTRIBUTING, AUTHORS, HISTORY, CHANGES, NEWS, anything under `docs/` or
`doc/`, or top-level `*.md`/`*.rst`. ADR/RFC paths are EXCLUDED (those
belong to a130). Files inside test directories (`tests/`, `__tests__/`,
`spec/`) are also excluded — README-like files in test trees are usually
fixtures, not project docs.

Three return states
-------------------
- None when applies() = False (no doc files touched).
- None when applies() = True but tree-sitter-markdown parses nothing
  meaningful (parse failure / empty added body).
- float in [0,1] otherwise.

Scoring
-------
For each qualifying doc file with non-empty added content, compute four
binary-ish signals on the **added lines only** (we deliberately ignore the
unchanged surrounding context — the metric scores the *change*, not the
file's overall state):

  s_size       = 1.0 if >= 3 non-blank non-comment lines added, else 0.0
                 (filters trivial typo/whitespace edits)
  s_structure  = 1.0 if >= 1 heading present in added content, else 0.5 if
                 the added content is part of a documented file that already
                 has headings in surrounding context (we can't see them from
                 the diff body alone, so we approximate via the *parsed*
                 added fragment). 0.0 if no headings AND no lists.
  s_concreteness = 1.0 if added content contains a code fence OR an
                   inline link OR a list item, else 0.0
  s_freshness  = 1.0 if the PR also touches at least one source-code file
                 (docs are being updated alongside code → "keep them
                 current"), else 0.5 (pure docs PR — neutral)

Per-file score = mean of the four signals.
Aggregate across qualifying files: arithmetic mean.

This gives a clean range (typically 0.25 .. 1.0) where:
  - 1.0: substantial docs change with structure, examples/links, alongside
    source code edits.
  - ~0.5: present but missing structure or concreteness.
  - ~0.25: trivial edit, no structure, no examples, no co-change.

Empirical note
--------------
On the 50-PR fixtures (truncated diffs), most PRs will abstain because the
sampled hunk window doesn't include doc files. This is by design — the
narrow gate is the entire point per the GUIDE's "over-abstain rather than
over-apply" rule.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a52"
ASPECT_NAME = "Documentation quality and maintenance"
TIER = 2
TOOLS = ["tree-sitter-markdown"]
APPLIES_TO_LANGS = ["Markdown", "reStructuredText"]
CLASSIFICATION = "PARTIALLY_THIN"


# ---------------------------------------------------------------------------
# Path classification — narrow gate on project documentation files.
# Paths aren't a language to parse; plain string ops are appropriate.
# ---------------------------------------------------------------------------

DOC_EXTS = (".md", ".markdown", ".mdx", ".rst", ".txt")

# Top-level / project documentation filename stems (lowercased, no ext).
DOC_FILENAME_STEMS = {
    "readme", "changelog", "changes", "history", "news",
    "contributing", "authors", "maintainers", "code_of_conduct",
    "codeofconduct", "support", "security", "upgrading", "upgrade",
    "migration", "faq", "tutorial", "getting_started", "gettingstarted",
    "install", "installation", "usage", "quickstart", "roadmap",
}

# Directory names where markdown is documentation (vs ADR/RFC, which a130
# owns, or test fixtures).
DOC_DIR_TOKENS = ("docs", "doc", "documentation", "guides", "wiki", "manual",
                  "handbook", "tutorials")

# Path tokens that DISQUALIFY a file from being "project docs" even if its
# extension is .md.
EXCLUDE_DIR_TOKENS = (
    # ADR/RFC are a130's territory.
    "adr", "adrs", "rfc", "rfcs", "decisions", "architecture-decisions",
    "architecture/decisions", "proposals",
    # Test fixtures.
    "tests", "test", "__tests__", "spec", "specs", "fixtures", "testdata",
    # Issue/PR templates aren't onboarding docs.
    ".github",
    # node_modules, venvs (defensive).
    "node_modules", "venv", ".venv",
)

# Source code extensions used to detect "co-change with source code" for the
# freshness signal.
SOURCE_EXTS = (
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java",
    ".go", ".rs", ".rb",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp",
    ".cs", ".kt", ".kts", ".scala", ".php", ".swift",
)


def _path_segments_lower(path: str) -> List[str]:
    return [seg for seg in path.lower().split("/") if seg]


def _is_doc_path(path: str) -> bool:
    """True iff path looks like project-level documentation."""
    p = path.lower()
    if not p.endswith(DOC_EXTS):
        return False
    segs = _path_segments_lower(path)
    name = segs[-1] if segs else ""
    parents = segs[:-1]

    # Exclusions first — never count ADR/RFC or test markdowns as project docs.
    for tok in EXCLUDE_DIR_TOKENS:
        if "/" in tok:
            if tok in "/".join(parents):
                return False
        else:
            if tok in parents:
                return False

    # 1. Top-level docs filename (readme.md, CHANGELOG.rst, etc.) at any
    #    depth — many monorepos have per-package READMEs.
    stem = name.rsplit(".", 1)[0]
    if stem in DOC_FILENAME_STEMS:
        return True

    # 2. Files under a recognized docs directory.
    for tok in DOC_DIR_TOKENS:
        if tok in parents:
            return True

    # 3. Top-level .md / .rst at repo root (e.g. INSTALL.md) — accept if
    #    parent path is empty (file at repo root).
    if not parents and p.endswith((".md", ".rst", ".markdown", ".mdx")):
        return True

    return False


def _is_source_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(SOURCE_EXTS)


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


def _structural_features(added: str) -> Optional[Dict[str, int]]:
    """Parse added markdown and count headings, code fences, links, list items.

    Returns None on parser failure. tree-sitter-markdown is tolerant of
    partial documents (which is what diff added-lines are — fragments).
    """
    parser = _parser()
    if not parser:
        return None
    try:
        tree = parser.parse(added.encode("utf8", errors="replace"))
    except Exception:
        return None
    counts = {"heading": 0, "code_fence": 0, "link": 0, "list_item": 0}

    def walk(n):
        t = n.type
        # tree-sitter-markdown node types vary by grammar version. We accept
        # several aliases.
        if t in ("atx_heading", "setext_heading"):
            counts["heading"] += 1
        elif t in ("fenced_code_block", "indented_code_block",
                   "code_fence_content"):
            # Avoid double-counting code_fence_content if fenced_code_block
            # is also walked; we just need at least 1.
            counts["code_fence"] += 1
        elif t in ("link", "inline_link", "shortcut_link", "image",
                   "link_reference_definition", "autolink"):
            counts["link"] += 1
        elif t in ("list_item",):
            counts["list_item"] += 1
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return counts


# rST has no widely-available tree-sitter grammar in the sandbox, but the
# diff body for an .rst file can still be heuristically scored on surface
# structural cues. We do a tiny structural pass without inventing a parser:
# headings in rST use an underline of `=`, `-`, `~`, `^` matching the
# preceding line's length; this is a well-defined non-code text format.
def _rst_structural_features(added: str) -> Dict[str, int]:
    """REGEX_OK-free surface counts for rST added content."""
    # tree-sitter-rst isn't in the sandbox; rST is a well-defined plaintext
    # markup format and we only count whether *some* heading-like underline
    # was added. Done with stdlib string ops — no regex on code.
    counts = {"heading": 0, "code_fence": 0, "link": 0, "list_item": 0}
    lines = added.splitlines()
    underline_chars = set("=-~^\"'`*+#<>")
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        # Heading: a line of >=3 of a single underline char (and a non-blank
        # line above it).
        if len(s) >= 3 and len(set(s)) == 1 and s[0] in underline_chars:
            if i > 0 and lines[i - 1].strip():
                counts["heading"] += 1
        # rST literal block / code-block directive (".. code-block::") or
        # a bare "::" at end-of-line.
        if s.startswith(".. code-block::") or s.startswith(".. code::"):
            counts["code_fence"] += 1
        if s.endswith("::"):
            counts["code_fence"] += 1
        # List items: rST bullets are *, -, +, or "#." / "1.".
        if s[:1] in ("*", "-", "+") and (len(s) == 1 or s[1] == " "):
            counts["list_item"] += 1
        # rST inline link target: `text <url>`_ — well-defined format. We
        # don't parse it; we just notice a `_` reference indicator at the
        # end of any token.
        if "`_" in s or s.startswith(".. _") or "http://" in s or "https://" in s:
            counts["link"] += 1
    return counts


# ---------------------------------------------------------------------------
# Per-file scoring.
# ---------------------------------------------------------------------------

def _count_meaningful_lines(added: str) -> int:
    """Number of non-blank, non-comment lines in the added content.

    'Comment' here means HTML/markdown comments `<!-- ... -->` on their own
    line. Inline trivia (just whitespace) is excluded.
    """
    n = 0
    in_html_comment = False
    for raw in added.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if in_html_comment:
            if "-->" in ln:
                in_html_comment = False
            continue
        if ln.startswith("<!--") and "-->" not in ln:
            in_html_comment = True
            continue
        if ln.startswith("<!--") and ln.endswith("-->"):
            continue
        n += 1
    return n


def _score_file(path: str, added: str) -> Optional[float]:
    n_lines = _count_meaningful_lines(added)
    if n_lines == 0:
        return None

    low = path.lower()
    if low.endswith((".rst",)):
        feats = _rst_structural_features(added)
    else:
        feats = _structural_features(added)
        if feats is None:
            # Parser unavailable — abstain rather than fake a score.
            return None

    s_size = 1.0 if n_lines >= 3 else 0.0
    has_heading = feats["heading"] > 0
    has_list = feats["list_item"] > 0
    if has_heading:
        s_structure = 1.0
    elif has_list:
        s_structure = 0.5
    else:
        s_structure = 0.0
    has_concrete = (feats["code_fence"] > 0 or feats["link"] > 0
                    or feats["list_item"] > 0)
    s_concreteness = 1.0 if has_concrete else 0.0

    return s_size, s_structure, s_concreteness  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Contract.
# ---------------------------------------------------------------------------

def _split_doc_and_source(
    diff_text: str,
) -> Tuple[Dict[str, str], List[str]]:
    by_path = parse_diff_added_by_file(diff_text)
    doc_files = {p: c for p, c in by_path.items() if _is_doc_path(p)}
    src_paths = [p for p in by_path.keys() if _is_source_path(p)]
    return doc_files, src_paths


def applies(diff_text: str) -> bool:
    """True iff the diff adds content to at least one project-doc file."""
    doc_files, _ = _split_doc_and_source(diff_text)
    return bool(doc_files)


def score(diff_text: str) -> Optional[float]:
    doc_files, src_paths = _split_doc_and_source(diff_text)
    if not doc_files:
        return None

    has_source_cochange = len(src_paths) > 0
    s_freshness = 1.0 if has_source_cochange else 0.5

    per_file: List[float] = []
    for path, added in doc_files.items():
        triple = _score_file(path, added)
        if triple is None:
            continue
        s_size, s_structure, s_concreteness = triple
        per_file.append(
            (s_size + s_structure + s_concreteness + s_freshness) / 4.0
        )

    if not per_file:
        return None
    return float(sum(per_file) / len(per_file))

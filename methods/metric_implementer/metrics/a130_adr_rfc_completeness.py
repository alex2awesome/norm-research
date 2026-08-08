"""a130: Decision records (RFC/ADR) completeness and quality.

Architecture Decision Records (Michael Nygard, 2011) and lightweight RFCs
have a well-known section schema. The standard Nygard template, as adopted
by most ADR tooling (`adr-tools`, MADR, `log4brains`, Joel Parker Henderson's
ADR catalogue, RFC-style internal templates), prescribes:

  - Title (H1, often "ADR <NNNN>: <topic>" or "RFC <NNNN>: <topic>")
  - Status              (Proposed / Accepted / Deprecated / Superseded)
  - Context             (the forces and constraints)
  - Decision            (the chosen approach)
  - Consequences        (positive, negative, and neutral implications)

The aspect description additionally calls out:
  - Alternatives (including rejected ones)
  - Trade-offs / rationale
  - Open questions / follow-up
  - Stakeholders / authors

We measure structural completeness against this template using
tree-sitter-markdown to walk the heading structure of each added ADR/RFC
file. We do not score prose quality (that is genuinely THICK — see GUIDE.md).
The structural check is THIN; the inferred "quality" via section presence
is at best PARTIALLY_THIN.

Detection (applies()):
  A diff file qualifies as an ADR/RFC iff its path matches one of the well-
  known conventions:

    - directory containing `adr`, `adrs`, `architecture/decisions`,
      `architecture-decisions`, `decisions`, `rfc`, `rfcs`, or `proposals`
    - filename starting with `ADR-`, `ADR_`, `RFC-`, `RFC_`, `0001-`, or
      `NNNN-<slug>.md` (the MADR/adr-tools default naming)

  AND the file extension is `.md` or `.markdown` (RFC text-files like
  `.rst`, `.txt` are also recognized for completeness).

Scoring per file:
  We collect the set of H1 + H2 heading texts (normalized to lowercase,
  punctuation stripped) and check presence of:

    Required Nygard sections (weight 1 each, max 4):
      - "context"
      - "decision"
      - "consequences"
      - "status" (or H1 contains an ADR/RFC status word)

    Aspect-bonus sections (weight 1 each, max 4):
      - "alternatives" (or "options" / "considered options" — MADR)
      - "trade-offs"  (or "tradeoffs" / "rationale" / "discussion")
      - "open questions" (or "follow-up" / "follow-ups" / "next steps")
      - "stakeholders" (or "authors" / "deciders" / "participants")

  Final per-file score = average of:
      required_fraction (0..1)
      bonus_fraction    (0..1)
  i.e. equal-weight blend of "is this a complete Nygard record?" and
  "does it also document alternatives/trade-offs/follow-up/stakeholders?".

  Across files in a diff, scores are averaged.

ABSTAIN gracefully:
  Most PRs touch no ADR/RFC files. applies() = False for those.
  When applies() = True but parsing fails or produces zero headings, score()
  returns None (distinguishable from low-score complete record).

Tier 2, PARTIALLY_THIN: structural section presence is thin; whether each
section's prose actually captures rationale/trade-offs is THICK.
"""
from __future__ import annotations

import string
from typing import Dict, List, Optional, Set

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a130"
ASPECT_NAME = "Decision records (RFC/ADR) completeness and quality"
TIER = 2
TOOLS = ["tree-sitter-markdown"]
APPLIES_TO_LANGS = ["Markdown"]
CLASSIFICATION = "PARTIALLY_THIN"


# ---------------------------------------------------------------------------
# Path-based ADR/RFC detection (Tier 1 over file paths — paths aren't a
# language to parse, so plain string ops are appropriate; no regex needed).
# ---------------------------------------------------------------------------

ADR_DIR_TOKENS = (
    "adr",
    "adrs",
    "architecture/decisions",
    "architecture-decisions",
    "decisions",
    "rfc",
    "rfcs",
    "proposals",
    "designs",
)

# Lowercase filename prefixes that strongly imply ADR/RFC.
ADR_FILENAME_PREFIXES = ("adr-", "adr_", "rfc-", "rfc_")

MD_EXTS = (".md", ".markdown", ".rst", ".txt", ".mdx")


def _looks_like_adr_path(path: str) -> bool:
    """Path-only check for ADR/RFC conventions. Cheap; safe for applies()."""
    p = path.lower()
    if not p.endswith(MD_EXTS):
        return False
    parts = p.split("/")
    name = parts[-1]
    parents = "/".join(parts[:-1])

    # 1. Standard directory locations.
    for tok in ADR_DIR_TOKENS:
        # match as a whole path segment, not substring (avoid "/proposals/" vs
        # "/docs/proposal-template.md")
        if "/" in tok:
            if tok in parents:
                return True
        else:
            for seg in parents.split("/"):
                if seg == tok:
                    return True

    # 2. Filename prefix conventions.
    for pre in ADR_FILENAME_PREFIXES:
        if name.startswith(pre):
            return True

    # 3. MADR / adr-tools default: 4-digit number then dash then slug.
    base = name.rsplit(".", 1)[0]
    if len(base) >= 5 and base[:4].isdigit() and base[4] == "-":
        # only count if at least one path segment hints at decisions/docs
        if any(seg in ("docs", "doc", "documentation") for seg in parts[:-1]):
            return True
        # Also accept if file lives at repo root or any depth as long as the
        # NNNN-slug pattern is present — but be conservative: require a docs
        # ancestor to avoid false-positives like `0001-init-migration.sql.md`.
        return False

    return False


# ---------------------------------------------------------------------------
# Markdown parser (tree-sitter-markdown).
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


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _heading_level(heading_node) -> Optional[int]:
    """Return 1..6 for an atx_heading node, or None."""
    for ch in heading_node.children:
        if ch.type.startswith("atx_h") and ch.type.endswith("_marker"):
            # types: atx_h1_marker .. atx_h6_marker
            try:
                return int(ch.type[len("atx_h"):-len("_marker")])
            except ValueError:
                return None
    return None


def _heading_text(heading_node, src: bytes) -> str:
    for ch in heading_node.children:
        if ch.type == "inline":
            return _text(ch, src).strip()
    return ""


def _setext_heading_level(node) -> Optional[int]:
    """tree-sitter-markdown also emits setext_heading (=== / ---)."""
    for ch in node.children:
        if ch.type == "setext_h1_underline":
            return 1
        if ch.type == "setext_h2_underline":
            return 2
    return None


def _setext_heading_text(node, src: bytes) -> str:
    for ch in node.children:
        if ch.type == "paragraph":
            return _text(ch, src).strip().splitlines()[0] if _text(ch, src).strip() else ""
        if ch.type == "inline":
            return _text(ch, src).strip()
    return ""


def _collect_headings(code: bytes) -> List[tuple]:
    """Return [(level, normalized_text, raw_text), ...] in document order."""
    parser = _parser()
    if not parser:
        return []
    try:
        tree = parser.parse(code)
    except Exception:
        return []
    out: List[tuple] = []

    def walk(n):
        if n.type == "atx_heading":
            lvl = _heading_level(n)
            txt = _heading_text(n, code)
            if lvl is not None and txt:
                out.append((lvl, _normalize(txt), txt))
        elif n.type == "setext_heading":
            lvl = _setext_heading_level(n)
            txt = _setext_heading_text(n, code)
            if lvl is not None and txt:
                out.append((lvl, _normalize(txt), txt))
        for c in n.children:
            walk(c)
    walk(tree.root_node)
    return out


# ---------------------------------------------------------------------------
# Section keyword matching.
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _normalize(s: str) -> str:
    """Lowercase, strip leading numbering, collapse whitespace, drop punct."""
    s = s.lower().strip()
    # strip leading "1.", "1)", "1 -" enumeration
    while s and (s[0].isdigit() or s[0] in ".)- "):
        s = s[1:]
    s = s.translate(_PUNCT_TABLE)
    return " ".join(s.split())


# Section keyword groups. Match if any keyword appears as a substring of the
# normalized heading. Each group counts at most once.
REQUIRED_GROUPS: Dict[str, List[str]] = {
    "context": ["context", "background", "problem", "motivation"],
    "decision": ["decision", "decision drivers", "solution", "chosen option"],
    "consequences": ["consequences", "implications", "impact", "outcomes",
                     "results"],
    "status": ["status", "state"],
}

BONUS_GROUPS: Dict[str, List[str]] = {
    "alternatives": ["alternatives", "options considered",
                     "considered options", "options", "other options",
                     "rejected"],
    "tradeoffs": ["trade offs", "tradeoffs", "rationale", "discussion",
                  "pros and cons", "analysis", "reasoning"],
    "open_questions": ["open questions", "follow up", "follow ups",
                       "followups", "next steps", "todo", "future work",
                       "open issues"],
    "stakeholders": ["stakeholders", "authors", "deciders", "participants",
                     "owners", "approvers", "reviewers", "contributors"],
}


def _group_present(heading_norms: Set[str], keywords: List[str]) -> bool:
    for h in heading_norms:
        for kw in keywords:
            if kw in h:
                return True
    return False


def _status_in_title(headings: List[tuple]) -> bool:
    """Some ADR templates put status in the H1 title (e.g. "[Accepted] ADR-3
    Use Postgres"). Treat that as a status signal too."""
    STATUS_WORDS = ("accepted", "proposed", "rejected", "deprecated",
                    "superseded", "draft")
    for lvl, norm, _ in headings:
        if lvl == 1:
            for w in STATUS_WORDS:
                if w in norm:
                    return True
    return False


# ---------------------------------------------------------------------------
# Per-file scoring.
# ---------------------------------------------------------------------------

def _score_file(code: bytes) -> Optional[float]:
    headings = _collect_headings(code)
    if not headings:
        return None
    norms = {norm for _, norm, _ in headings}

    # Required Nygard sections.
    req_hits = 0
    for key, keywords in REQUIRED_GROUPS.items():
        if _group_present(norms, keywords):
            req_hits += 1
        elif key == "status" and _status_in_title(headings):
            req_hits += 1
    req_frac = req_hits / len(REQUIRED_GROUPS)

    # Aspect-bonus sections.
    bonus_hits = sum(1 for kws in BONUS_GROUPS.values()
                     if _group_present(norms, kws))
    bonus_frac = bonus_hits / len(BONUS_GROUPS)

    return 0.5 * req_frac + 0.5 * bonus_frac


# ---------------------------------------------------------------------------
# Contract.
# ---------------------------------------------------------------------------

def _candidate_files(diff_text: str) -> Dict[str, str]:
    by_path = parse_diff_added_by_file(diff_text)
    return {p: c for p, c in by_path.items() if _looks_like_adr_path(p)}


def applies(diff_text: str) -> bool:
    """True iff the diff adds content to at least one ADR/RFC markdown file."""
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
        s = _score_file(body.encode("utf8", errors="replace"))
        if s is not None:
            scores.append(s)
    if not scores:
        return None
    return float(sum(scores) / len(scores))

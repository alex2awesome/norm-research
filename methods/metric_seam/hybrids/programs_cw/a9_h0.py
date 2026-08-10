"""Hybrid metric channel for a9: Plot Coherence, Unity, and Appropriate Magnitude.

Design rationale (from train residuals):
- Judge rewards ONE whole action with a decisive ending; keyword arc-markers
  (baseline, rho=0.109) barely correlate. The construct is tacit -> two
  narrow YES/NO LLM fields carry it; the predicate stays in code.
- Meta contamination ("Edit:", "To be continued", opening preambles) marks
  lows; end-of-story subreddit plugs do NOT (0.9 stories carry them), so
  penalties are targeted, not generic.
- Magnitude: micro-fragments (<~250 words) cannot carry a weighty whole
  action; sprawl is handled by the arc field, not length alone.
"""

import re

LLM_FIELDS = {
    "complete_arc": (
        "Ignoring author notes, edits, and links: does the story tell ONE "
        "complete plot whose situation clearly changes from beginning to "
        "end? Answer YES or NO."
    ),
    "decisive_ending": (
        "Does the story's final scene land a decisive resolution, twist, or "
        "punchline that completes the main action? Answer YES or NO."
    ),
}

_META_PARA = re.compile(
    r"(edit\s*:|http|www\.|\br/[a-z_]+|\]\(|&#x200b|subscribe|"
    r"if you like my work|read more of it|part \d+ (is|here)|"
    r"^[\s~*_\-=]{3,}$)",
    re.IGNORECASE,
)
_OPEN_META = re.compile(
    r"(\[\s*wp\s*\]|\bobligatory\b|gotta contribute|if no one else will post|"
    r"my sacred duty|first (post|attempt|time posting)|long time lurker|"
    r"much\?\s*;\)|don'?t judge|be gentle|never posted)",
    re.IGNORECASE,
)


def _yn(val, default=0.4):
    """Parse a short YES/NO extractor answer into 1.0 / 0.0 / default."""
    try:
        s = (val or "").strip().lower()
        if not s:
            return default
        if s.startswith("yes"):
            return 1.0
        if s.startswith("no"):
            return 0.0
        if "yes" in s:
            return 1.0
        if "no" in s:
            return 0.0
        return 0.5
    except Exception:
        return 0.5


def _paragraphs(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


def _story_paragraphs(paras):
    """Drop trailing author-meta paragraphs (plugs, edit notes, link lists)."""
    out = list(paras)
    while out and _META_PARA.search(out[-1]):
        out.pop()
    return out if out else list(paras)


def _len_band(n_words):
    if n_words <= 0:
        return 0.0
    if n_words < 300:
        return max(0.1, (n_words - 80) / 220.0)
    if n_words <= 900:
        return 1.0
    return max(0.5, 1.0 - (n_words - 900) / 2200.0)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        if not t.strip():
            return 0.0
        low = t.lower()

        ex = extracted if isinstance(extracted, dict) else {}
        arc = _yn(ex.get("complete_arc", ""))
        end = _yn(ex.get("decisive_ending", ""))

        # --- code features ---------------------------------------------
        n_words = len(t.split())
        lb = _len_band(n_words)

        paras = _paragraphs(t)
        story = _story_paragraphs(paras)

        # Punch ending: short, cleanly terminated final story paragraph.
        punch = 0.0
        if story:
            last = story[-1].strip()
            if 15 <= len(last) <= 170 and re.search(r"[.!?\"”')]$", last):
                punch = 1.0

        # --- targeted penalties (lows in train are meta-contaminated) ---
        pen = 0.0
        n_edit = len(re.findall(r"\bedit\s*:", low))
        pen += min(0.15, 0.06 * n_edit)
        if re.search(r"to be continued\W{0,20}$", low.strip()):
            pen += 0.15
        if _OPEN_META.search(low[:250]):
            pen += 0.12
        story_txt = "\n\n".join(story).rstrip()
        if re.search(r"(\.\.\.|…|[-–—])[\"”')]?$", story_txt):
            pen += 0.07  # trails off instead of landing
        pen = min(pen, 0.30)

        # --- combine: arc dominates; ending gated by arc ----------------
        s = (
            0.10
            + 0.38 * arc
            + 0.22 * end * (0.4 + 0.6 * arc)
            + 0.22 * lb
            + 0.08 * punch
            - pen
        )
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

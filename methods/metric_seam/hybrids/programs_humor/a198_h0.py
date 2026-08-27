"""a198: Parody, pastiche, and intertextuality (hybrid v0).

Criterion: imitates, references, or transforms styles/texts with fit and
invention -- moving beyond imitation into purposeful comedic commentary.

Design: regex/topic-word matching (the v0_keyword baseline) cannot tell
whether a joke actually invokes a specific outside text/genre/idiom, nor
whether it merely repeats vs. purposefully transforms that source. Those
are THICK-INPUT constructs, so two short LLM fields ground the code:
what specific source (if any) is being riffed on, and how the joke moves
on it. Code keeps the predicate: it scores based on whether a source was
named, whether the transform looks like genuine commentary (vs. bare
repetition), corroborates with light lexical/structural signals a
regex *can* see (stock joke-genre openers like "walks into a bar" /
"moral of the story" / echoed ad-slogans; overlap between the named
source and the actual text; elevated long-word/formal-register share as
a pastiche-of-officialese cue), and lightly discounts near-duplicate
reposts of stock formulas when no transformation was identified.
"""
import re

LLM_FIELDS = {
    "source_ref": (
        "Name in <=8 words the specific real idiom, ad slogan, genre "
        "convention (e.g. fable, bar-joke, knock-knock, political-joke), "
        "public figure/event, or text this piece imitates or riffs on; "
        "empty if none."
    ),
    "transform_move": (
        "In <=8 words, say HOW it transforms or comments on that source "
        "(e.g. subverts, literalizes, extends, mocks, escalates); empty "
        "if no reference was identified."
    ),
}

# Stock joke-genre / borrowed-form openers: a textual analog of "pastiche of
# a known form" that plain code can actually detect.
_STOCK_GENRE_RE = re.compile(
    r"\bwalks?\s+into\s+a\s+bar\b"
    r"|\bknock,?\s*knock\b"
    r"|\bwhy\s+did\s+the\b"
    r"|\bmoral\s+of\s+(the|this)\s+story\b"
    r"|\bonce\s+upon\s+a\s+time\b"
    r"|\bnothin'?\s+says\b"
    r"|\bin\s+the\s+style\s+of\b",
    re.IGNORECASE,
)

# Weak explicit-naming-of-the-device lexicon (kept low-weight per corpus
# notes: topic/lexical hits are weak proxies on their own).
_POS_KEYWORDS = ("parody", "pastiche", "reference", "allude", "tribute", "spoof", "riff on", "takeoff on")

# Words in the LLM's transform_move answer that indicate genuine
# transformation/commentary rather than bare repetition of the source.
_TRANSFORM_WORDS = (
    "subvert", "invert", "twist", "extend", "transform", "mock", "satir",
    "exaggerat", "literal", "undercut", "recontextualiz", "flip", "invers",
    "escalat", "juxtapos", "commentary", "critiqu", "revers",
)

_NONE_MARKERS = ("", "none", "n/a", "na", "-")

_TOKEN_RE = re.compile(r"[a-z']+")


def _tokens(s):
    return set(_TOKEN_RE.findall((s or "").lower()))


def _is_present(field_val):
    return bool(field_val) and field_val.strip().lower() not in _NONE_MARKERS


def score(text: str, extracted: dict, ops) -> float:
    try:
        extracted = extracted or {}
        raw = text or ""
        if not raw.strip():
            return 0.5

        t = ops.normalize(raw)
        tl = t.lower()

        source_ref = (extracted.get("source_ref") or "").strip()
        transform_move = (extracted.get("transform_move") or "").strip()

        has_source = _is_present(source_ref)
        has_transform = _is_present(transform_move)

        points = 0.05  # floor: plain joke, no identified intertextuality

        if has_source:
            points += 0.30
            if _tokens(source_ref) & _tokens(t):
                points += 0.08  # grounding: named source echoes text vocabulary

        if has_transform:
            tml = transform_move.lower()
            if any(w in tml for w in _TRANSFORM_WORDS):
                points += 0.32  # genuine transformation / purposeful commentary
            else:
                points += 0.12  # names a move but reads like plain imitation

        if _STOCK_GENRE_RE.search(tl):
            points += 0.10  # code-detectable borrowed-form / genre convention

        kw_hits = sum(1 for k in _POS_KEYWORDS if k in tl)
        if kw_hits:
            points += min(0.06, 0.03 * kw_hits)

        # Pastiche-of-register cue (e.g. mock-officialese, fable narration):
        # only counted once an intertextual anchor was actually identified,
        # so it corroborates rather than rewards generic verbosity.
        if has_source or has_transform:
            try:
                stats = ops.sent_stats(t)
                frac_long = None
                if isinstance(stats, dict):
                    frac_long = stats.get("frac_long_words")
                elif isinstance(stats, (tuple, list)) and len(stats) >= 3:
                    frac_long = stats[2]
                if isinstance(frac_long, (int, float)) and frac_long > 0.16:
                    points += 0.05
            except Exception:
                pass

        # Near-duplicate dampening: a repost of a very common stock formula
        # with no identified source/transform shows little invention.
        if not has_source and not has_transform:
            try:
                neighbors = ops.retrieve_similar(t, k=5) or []
                top_sim = 0.0
                for a, b in neighbors:
                    for cand in (a, b):
                        if isinstance(cand, (int, float)) and 0.0 <= float(cand) <= 1.0:
                            top_sim = max(top_sim, float(cand))
                if top_sim > 0.92:
                    points *= 0.7
            except Exception:
                pass

        return max(0.0, min(1.0, points))
    except Exception:
        return 0.5

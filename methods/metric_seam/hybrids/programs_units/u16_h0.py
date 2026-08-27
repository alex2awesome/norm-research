"""u16 Avoiding Over-Explanation: code detects explicit over-explanation cue phrases and a drag-length structural check; LLM fields carry the actual trailing-explanation text and how many extra sentences follow the real punchline."""

import re

LLM_FIELDS = {
    "trailing_explain": (
        "Quote (<=12 words) any sentence AFTER the punchline that merely "
        "explains or restates why the joke is funny (e.g. 'because...', "
        "'get it?'), or say 'none'."
    ),
    "post_punch_count": (
        "How many sentences follow the joke's actual punchline that add "
        "no new comedic content? Answer: zero, one, or two-or-more."
    ),
}

_EXPLAIN_CUE_RE = re.compile(
    r"\b(get it\??|see what i did there|the joke is|the point is|"
    r"in other words|which means|moral of the story|just kidding|jk\b|"
    r"lol\b|lmao\b|haha+)\b",
    re.IGNORECASE,
)

_COUNT_MAP = [
    ("two-or-more", 2), ("two or more", 2), ("2+", 2),
    ("one", 1), ("zero", 0), ("none", 0),
]


def _classify_count(raw):
    if not raw:
        return None
    s = raw.lower()
    for key, val in _COUNT_MAP:
        if key in s:
            return val
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        trailing = str(extracted.get("trailing_explain", "") or "").strip().lower()
        has_trailing = bool(trailing) and trailing not in ("none", "n/a", "")

        count = _classify_count(str(extracted.get("post_punch_count", "") or ""))
        if count is None:
            count = 1 if has_trailing else 0

        # --- primary construct: penalize trailing sentences that explain
        # rather than land; code owns the mapping from the field-grounded
        # count/quote to a score. ---
        if count == 0:
            base = 0.85
        elif count == 1:
            base = 0.45
        else:
            base = 0.15
        if has_trailing:
            base = min(base, 0.4)  # an actual explanatory sentence was quoted

        # --- code-only structural corroboration: explicit over-explanation
        # cue phrases anywhere in the text, and a final sentence unusually
        # long relative to the rest (dragging on past the punch). ---
        cue_hits = len(_EXPLAIN_CUE_RE.findall(norm))
        struct = -min(0.2, 0.08 * cue_hits)

        sentences = [s for s in re.split(r"[.!?]+", norm) if s.strip()]
        if len(sentences) >= 2:
            counts = [len(s.split()) for s in sentences]
            other_mean = sum(counts[:-1]) / max(1, len(counts[:-1]))
            if other_mean > 0 and counts[-1] > 1.3 * other_mean:
                struct -= 0.08  # bloated final sentence

        s = base + struct
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

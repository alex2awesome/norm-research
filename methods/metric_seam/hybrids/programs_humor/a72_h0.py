"""
Hybrid channel for a72: "Respect the audience; avoid over-explaining"
Trust audience intelligence; don't pre-apologize, condescend, or hand-hold
references unless necessary.

Design: the two LLM fields do THICK-INPUT grounding only (quote-or-empty
extraction of hedging / joke-explaining language a paraphrase-robust reader
would notice but rigid regex might miss). All scoring logic (the predicate)
lives in code, driven by:
  (A) apology / disclaimer / self-deprecating-legitimacy hedges
  (B) meta-commentary that explains the joke, telegraphs the punchline, or
      cues the audience to laugh ("get it?", "HEY-OOO", "wait for it", ...)
  (C) mild condescension lexicon ("obviously", "to be clear", ...)
  (D) redundant restatement of the opening line/title (telling the joke twice
      as if the audience needed a second pass)
Each hit category contributes a capped penalty; final score = 1 - penalty,
clamped to [0, 1]. A clean, un-hedged joke (however dark/offensive the
content -- content is not this criterion's business) scores near 1.0.
"""

import re

LLM_FIELDS = {
    "hedge_phrase": (
        "Quote the exact phrase (<=20 words) where the writer apologizes, "
        "disclaims offense, or downplays/undersells the joke before or after "
        "telling it (e.g. pre-apology, 'not racist but', 'sorry if'); else answer NONE."
    ),
    "explain_phrase": (
        "Quote the exact phrase (<=20 words) where the writer explicitly explains "
        "the joke's own punchline/pun or spells out a reference for the audience "
        "instead of trusting them to get it; else answer NONE."
    ),
}

_APOLOGY_PATTERNS = [
    r"\bnot\s+(?:a\s+)?racist\b",
    r"don'?t\s+(?:kill|hate|hurt)\s+me\b",
    r"\bno\s+offense\b",
    r"\bnot\s+trying\s+to\s+be\s+(?:rude|racist|mean|offensive)\b",
    r"\bi'?m\s+sorry\b",
    r"\bsorry\s+(?:for|if)\b",
    r"\bjust\s+kidding\b",
    r"\bjk\b",
    r"\bkeep\s+your\s+expectations\s+low\b",
    r"\bi\s+made\s+this\s+joke\b",
    r"\bi\s+don'?t\s+know\s+where\s+else\s+to\s+(?:post|put)\s+this\b",
    r"\blet\s+me\s+know\s+if\s+i\s+(?:need|should)\b",
    r"\bnot\s+a\s+(?:standard|real)\s+joke\b",
    r"\bfeel\s+free\s+to\s+downvote\b",
    r"\bdon'?t\s+judge\s+me\b",
]

_META_PATTERNS = [
    r"\bthis\s+joke\b.{0,30}\b(?:makes|means|is\s+about|works)\b",
    r"\bthat\s+joke\b.{0,30}\b(?:makes|means|is\s+about|works)\b",
    r"\bget\s+it\s*\?",
    r"\bsee\s+what\s+i\s+did\s+there\b",
    r"\bif\s+you\s+(?:didn'?t|don'?t)\s+(?:get|understand|catch)\s+(?:it|that)\b",
    r"\bin\s+case\s+you\s+(?:didn'?t|don'?t)\s+(?:get|understand|catch)\b",
    r"\byou\s+(?:were\s+probably|probably\s+were)\s+(?:surprised|shocked|expecting)\b",
    r"\bguess\s+(?:the|what)\s+the?\s*punchline\b",
    r"\bi\s+guess\s+you\s+could\s+say\b",
    r"\bwait\s+for\s+it\b",
    r"\bba+\s*dum\s*[\-\s]?(?:tss+|ching)\b",
    r"\bhey+[\-\s]?o+\b",
    r"\blol'?d\b",
    r"\bpun\s+intended\b",
]

_CONDESCEND_PATTERNS = [
    r"\bobviously\b",
    r"\bbasically\b",
    r"\bto\s+be\s+clear\b",
    r"\bfor\s+those\s+(?:of\s+you\s+)?who\s+don'?t\s+know\b",
    r"\bin\s+case\s+you'?re\s+unfamiliar\b",
    r"\bfor\s+context\b",
    r"\ban\s+explanation\s+is\s+in\s+order\b",
]


def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_APOLOGY_RE = _compile(_APOLOGY_PATTERNS)
_META_RE = _compile(_META_PATTERNS)
_CONDESCEND_RE = _compile(_CONDESCEND_PATTERNS)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")


def _norm_sentence(s):
    words = _WORD_RE.findall(s.lower())
    return " ".join(words)


def _count_hits(patterns, t):
    return sum(1 for p in patterns if p.search(t))


def _has_duplicate_opening(text):
    """Detect the opening line/title being restated verbatim right after,
    as if the audience needed the joke told twice (a hand-holding tell)."""
    try:
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        if len(sents) < 2:
            return False
        norm = [_norm_sentence(s) for s in sents[:4]]
        norm = [n for n in norm if len(n) > 8]
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                a, b = norm[i], norm[j]
                if a == b:
                    return True
                # near-duplicate: one contains most of the other
                shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
                if len(shorter) > 12 and shorter in longer:
                    return True
        return False
    except Exception:
        return False


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.5

        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        if not t:
            t = text

        n_apology = _count_hits(_APOLOGY_RE, t)
        n_meta = _count_hits(_META_RE, t)
        n_condescend = _count_hits(_CONDESCEND_RE, t)
        dup_open = _has_duplicate_opening(t)

        hedge_phrase = ""
        explain_phrase = ""
        if isinstance(extracted, dict):
            hedge_phrase = (extracted.get("hedge_phrase") or "").strip()
            explain_phrase = (extracted.get("explain_phrase") or "").strip()

        penalty = 0.0

        if n_apology > 0:
            penalty += 0.45 + 0.15 * (min(n_apology, 3) - 1)

        if n_meta > 0:
            penalty += 0.40 + 0.15 * (min(n_meta, 3) - 1)

        if n_condescend > 0:
            penalty += 0.12 * min(n_condescend, 3)

        if dup_open:
            penalty += 0.15

        # LLM-grounded signals: only add weight if regex didn't already
        # catch the same phrase (avoid double-penalizing one hedge twice),
        # but a distinct LLM-caught instance still counts.
        if hedge_phrase and n_apology == 0:
            penalty += 0.35
        elif hedge_phrase and n_apology > 0:
            penalty += 0.10

        if explain_phrase and n_meta == 0:
            penalty += 0.35
        elif explain_phrase and n_meta > 0:
            penalty += 0.10

        out = 1.0 - penalty
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5

"""u18 Benign Superiority and Moral Violation: code detects heavy-handed delivery cues (shouted text, violent/hateful vocabulary) and repeated multi-perspective sentence patterns as structural proxies; LLM fields carry the semantic judgment of whether the violation reads as benign vs. malicious, and whether the joke uses multi-perspective repetition or proportionate incongruity (the competing sub-criterion)."""

import re

LLM_FIELDS = {
    "violation_tone": (
        "Does the joke's violation or target come across as benign/gentle "
        "mockery, or as malicious/heavy-handed cruelty? Answer 'benign' or "
        "'malicious/heavy-handed'."
    ),
    "repeat_structure": (
        "Does the joke repeat a similar question or exchange across "
        "multiple characters/perspectives before the punchline, or is it "
        "one single-shot line? Answer 'multi-perspective repeat' or "
        "'single-shot'."
    ),
}

_SHOUT_RE = re.compile(r"[A-Z]{5,}")
_HEAVY_RE = re.compile(
    r"\b(kill\w*|murder\w*|torture\w*|brutal\w*|hate\w*|slur\w*|"
    r"humiliat\w*|degrad\w*|savage\w*)\b",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _repeated_openers(t):
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]
    openers = [" ".join(re.findall(r"[A-Za-z']+", s)[:2]).lower() for s in sents]
    openers = [o for o in openers if o]
    if len(openers) < 2:
        return False
    counts = {}
    for o in openers:
        counts[o] = counts.get(o, 0) + 1
    return max(counts.values(), default=0) >= 2


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        heavy = 0.0
        if _SHOUT_RE.search(raw):
            heavy += 0.2
        heavy_hits = len(_HEAVY_RE.findall(t))
        heavy += 0.15 * min(heavy_hits, 3)
        heavy = min(heavy, 0.6)

        repeat_bonus = 0.15 if _repeated_openers(t) else 0.0

        extracted = extracted or {}
        tone = str(extracted.get("violation_tone", "") or "").strip().lower()
        repeat = str(extracted.get("repeat_structure", "") or "").strip().lower()

        if "benign" in tone:
            tone_adj = 0.35
        elif "malicious" in tone or "heavy-handed" in tone:
            tone_adj = -0.4
        else:
            tone_adj = 0.0

        repeat_adj = 0.15 if "multi-perspective" in repeat else 0.0

        s = 0.5 + tone_adj + repeat_adj + repeat_bonus - heavy
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

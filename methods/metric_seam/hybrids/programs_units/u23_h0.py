"""u23 Cross-Media Adaptability: code detects explicit medium-locked references (images/links/AV cues) as a hazard against portability; LLM fields carry the semantic judgment of whether the joke is self-contained in words (medium-independent) and whether a repeated pattern escalates rather than merely repeats (the competing sub-criterion)."""

import re

LLM_FIELDS = {
    "medium_dependence": (
        "Does the joke rely on a specific visual/audio/meme element (an "
        "image, sound, or format) that would not work if only read as "
        "plain text, or is it self-contained in words? Answer "
        "'medium-independent' or 'medium-dependent'."
    ),
    "escalation": (
        "If the joke repeats a pattern across multiple instances, does "
        "each instance escalate/intensify, or does it just repeat flatly? "
        "Answer 'escalates', 'flat-repeat', or 'no repetition'."
    ),
}

_MEDIA_REF_RE = re.compile(
    r"\b(image|picture|pic|photo|gif|video|meme|screenshot|attached|"
    r"see below|watch this|imgur|youtube)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        hazard = 0.0
        if _MEDIA_REF_RE.search(t):
            hazard += 0.35
        if _URL_RE.search(t):
            hazard += 0.35
        hazard = min(hazard, 0.6)

        extracted = extracted or {}
        medium = str(extracted.get("medium_dependence", "") or "").strip().lower()
        escal = str(extracted.get("escalation", "") or "").strip().lower()

        if "medium-independent" in medium:
            medium_adj = 0.3
        elif "medium-dependent" in medium:
            medium_adj = -0.3
        else:
            medium_adj = 0.0

        if "escalates" in escal:
            escal_adj = 0.25
        elif "flat-repeat" in escal:
            escal_adj = -0.15
        else:
            escal_adj = 0.0

        s = 0.5 + medium_adj + escal_adj - hazard
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

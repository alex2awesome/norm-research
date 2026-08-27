"""u27 hybrid: code detects textual analogs of visual/performative content (stage-direction and caption markers, multi-turn quoted dialogue) as corroboration; LLM fields carry the integration judgment and the secondary recurring-celebrity/talk-show check code cannot make from surface text."""

import re

LLM_FIELDS = {
    "visual_verbal_integration": (
        "Does a visual/performative element merge with the verbal "
        "punchline into one device? Answer integrated, verbal-only, or "
        "NONE."
    ),
    "celebrity_guest_recurring": (
        "Does this involve a recurring celebrity guest in a talk-show-"
        "style format? Answer yes, no, or NONE."
    ),
}

_NONE_VALUES = {"", "none", "n/a", "na", "unclear", "unknown"}

_STAGE_PAT = re.compile(
    r"\((?:laughs?|pauses?|grins?|shrugs?|winks?|nods?|sighs?|gasps?|"
    r"smirks?)\)", re.I)
_CAPTION_PAT = re.compile(r"\bcaption\s*:|\bphoto\s*:|\bpic\s*:|\bimage\s*:", re.I)
_QUOTE_PAT = re.compile(r'"[^"]{2,}"')


def _norm_field(v):
    return (v or "").strip().lower().strip(". ")


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        ex = extracted or {}
        integ = _norm_field(ex.get("visual_verbal_integration", ""))
        celeb = _norm_field(ex.get("celebrity_guest_recurring", ""))

        stage_hit = bool(_STAGE_PAT.search(t))
        caption_hit = bool(_CAPTION_PAT.search(t))
        quote_count = len(_QUOTE_PAT.findall(t))
        code_signal = stage_hit or caption_hit or quote_count >= 2

        # --- primary: visual-verbal merge (textual analog code can only partly see) ---
        if integ == "integrated":
            base = 0.85 if code_signal else 0.6
        elif integ == "verbal-only":
            base = 0.2
        else:
            base = 0.55 if code_signal else 0.3

        base = max(0.0, min(0.9, base))  # leave room for the secondary bump

        # --- secondary (small weight): recurring celebrity guest / talk-show format ---
        if celeb == "yes":
            base += 0.1

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

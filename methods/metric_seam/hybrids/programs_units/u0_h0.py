# Hybrid module for humor unit u0: "Avoids Harmful Targeting"
# Construct division: code carries the deterministic hazard lexicon (slurs/
# protected-characteristic group terms, rape/sexual-assault vocabulary, and
# WHERE in the text a hazard term lands, since a hazard sitting in the
# punchline position is structurally worse than one in incidental scene-
# setting); the two LLM fields carry the thick semantic judgment code cannot
# reach: whether the punchline's target is a protected/immutable class or a
# specific real individual, and whether the joke's actual comedic reveal (not
# just incidental phrasing) is degrading material such as rape.

import re

LLM_FIELDS = {
    "target_class": (
        "In <=8 words, name what the punchline's mockery targets: a "
        "protected/immutable characteristic (race, religion, disability, "
        "sexual orientation, gender), a specific named real person, or "
        "neither."
    ),
    "punchline_is_degrading": (
        "In <=10 words: is the joke's comedic reveal itself rape, sexual "
        "assault, or another degrading act (not just mentioned in passing)? "
        "Answer 'yes' or 'no'."
    ),
}

_SLUR_RE = re.compile(
    r"\b(nigger[s]?|nigga[s]?|kike[s]?|spic[s]?|chink[s]?|gook[s]?|"
    r"fagg?ot[s]?|tranny|trannies|retard(?:ed|s)?|cripple[ds]?|midget[s]?|"
    r"beaner[s]?|wetback[s]?|towelhead[s]?)\b",
    re.IGNORECASE,
)

_PROTECTED_GROUP_RE = re.compile(
    r"\b(black people|white people|mexicans?|asians?|jews?|jewish|muslims?|"
    r"christians?|latinos?|immigrants?|gays?|lesbians?|trans(?:gender)?|"
    r"disabled|autis\w*|blind people|deaf people|women\b|old people|elderly)\b",
    re.IGNORECASE,
)

_RAPE_RE = re.compile(r"\b(rape[ds]?|raping|molest\w*|sexual(?:ly)? assault\w*)\b", re.IGNORECASE)

_INDIVIDUAL_MARKER_RE = re.compile(
    r"\b(trump|biden|obama|putin|merkel|musk|kardashian|clinton)\b", re.IGNORECASE
)

_SUICIDE_RE = re.compile(r"\bsuicid\w*\b", re.IGNORECASE)


def _hazard_position_weight(match_end, text_len):
    """Hazard landing in the last third of the text (punchline zone) is worse."""
    if text_len <= 0:
        return 1.0
    frac = match_end / text_len
    return 1.35 if frac >= 0.66 else 1.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        n = max(1, len(t))

        hazard = 0.0

        # --- deterministic code hazards, weighted by position (punchline vs setup) ---
        for m in _SLUR_RE.finditer(t):
            hazard += 0.5 * _hazard_position_weight(m.end(), n)

        for m in _PROTECTED_GROUP_RE.finditer(t):
            # group mention alone is weak; only counts as targeting if paired
            # with a degrading/violent cue nearby (checked via rape/suicide
            # regexes and the LLM field below), so keep this contribution small
            hazard += 0.05

        for m in _RAPE_RE.finditer(t):
            hazard += 0.45 * _hazard_position_weight(m.end(), n)

        for m in _SUICIDE_RE.finditer(t):
            hazard += 0.15 * _hazard_position_weight(m.end(), n)

        if _INDIVIDUAL_MARKER_RE.search(t) and (_RAPE_RE.search(t) or _SLUR_RE.search(t)):
            hazard += 0.2

        # --- LLM-field grounding (thick-input constructs code can't reach) ---
        extracted = extracted or {}
        target_class = str(extracted.get("target_class", "") or "").strip().lower()
        degrading = str(extracted.get("punchline_is_degrading", "") or "").strip().lower()

        if target_class and target_class not in ("neither", "none", "n/a"):
            if any(w in target_class for w in (
                "race", "religio", "disab", "orientation", "gender", "ethnic",
                "protected", "immutable",
            )):
                hazard += 0.4
            elif any(w in target_class for w in ("named", "real person", "individual", "person")):
                hazard += 0.3
            else:
                hazard += 0.1

        if degrading.startswith("yes"):
            hazard += 0.45

        hazard = min(hazard, 1.5)
        s = 1.0 - hazard
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

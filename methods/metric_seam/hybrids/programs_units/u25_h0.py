# Hybrid module for humor unit u25: "Self-Deprecation for Rapport"
# Construct division: code carries the surface-lexical signal of
# first-person self-deprecation (an "I am/did <negative-self-descriptor>"
# pattern -- idiot, clumsy, failed, terrible at) and a lightweight emotion
# lexicon as a proxy for the piece having a legible emotional undercurrent.
# The two LLM fields carry what code cannot judge: whether the joke's butt
# is genuinely the narrator's own flaws (vs. self-deprecation that's
# actually a vehicle for mocking someone else), and what the piece's
# emotional undercurrent actually is.

import re

LLM_FIELDS = {
    "self_deprecating_target": (
        "In <=8 words: is the joke's butt of humor the narrator's own flaws "
        "or incompetence, or someone/something else? Answer 'self' or name "
        "the other target."
    ),
    "emotional_tone": (
        "In <=6 words: what emotional undercurrent drives the piece (e.g. "
        "embarrassment, anger, vulnerability), or is it emotionally flat? "
        "Answer the emotion or 'flat'."
    ),
}

_FIRST_PERSON_RE = re.compile(r"\bI\b|\bI'm\b|\bI've\b|\bmy\b|\bmyself\b", re.IGNORECASE)

_SELF_NEG_DESCRIPTOR_RE = re.compile(
    r"\b(idiot|stupid|dumb|clumsy|incompetent|terrible at|bad at|awful at|"
    r"failed|failure|loser|pathetic|useless|screw(?:ed)? up|messed up|"
    r"can'?t do anything right|bumbling|klutz)\b",
    re.IGNORECASE,
)

_BUMBLE_ACTION_RE = re.compile(
    r"\bI\s+(?:accidentally|forgot|tripped|spilled|dropped|fell|"
    r"embarrassed myself|screwed up|messed up)\b",
    re.IGNORECASE,
)

_GENDER_MICROAGGRESSION_RE = re.compile(
    r"\b(blonde|women (?:always|can'?t)|men (?:always|can'?t)|typical woman|typical man)\b",
    re.IGNORECASE,
)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        # --- code: first-person self-deprecation requires BOTH a first-person
        # frame and a negative self-descriptor near it, not just either alone ---
        fp_present = bool(_FIRST_PERSON_RE.search(t))
        neg_hits = len(_SELF_NEG_DESCRIPTOR_RE.findall(t))
        bumble_hits = len(_BUMBLE_ACTION_RE.findall(t))

        if fp_present and (neg_hits or bumble_hits):
            self_deprecation_score = min(1.0, 0.4 * neg_hits + 0.35 * bumble_hits + 0.25)
        elif neg_hits or bumble_hits:
            # negative-self language present but no clear first-person anchor
            self_deprecation_score = 0.35
        else:
            self_deprecation_score = 0.25

        # --- code: lexicon-flagged gender-microaggression content is a
        # competing sub-criterion here (appropriate handling matters, not
        # blanket avoidance), so it only mildly discounts rather than zeroes ---
        gender_penalty = 0.1 if _GENDER_MICROAGGRESSION_RE.search(t) else 0.0

        # --- LLM-field grounding ---
        extracted = extracted or {}
        target = str(extracted.get("self_deprecating_target", "") or "").strip().lower()
        emotion = str(extracted.get("emotional_tone", "") or "").strip().lower()

        if target == "self" or target.startswith("self"):
            target_component = 1.0
        elif target and target not in ("n/a", "none"):
            target_component = 0.2  # self-deprecation framing diverted onto another target
        else:
            target_component = 0.5

        emotion_bonus = 0.1 if emotion and emotion != "flat" else 0.0

        combined = (
            0.55 * self_deprecation_score +
            0.35 * target_component +
            emotion_bonus
        )
        out = combined - gender_penalty
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5

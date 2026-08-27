"""u22 hybrid: code detects profanity count/position (tail placement = punchline-shaped) and TF-IDF corpus-redundancy as a well-worn-shock-riff proxy; LLM fields carry the purposeful-vs-gratuitous judgment and the comedic role code cannot infer from surface form alone."""

import re

LLM_FIELDS = {
    "profanity_purpose": (
        "Does profane/transgressive content serve the joke's comedic "
        "mechanism, or feel gratuitous/shock-only? Answer purposeful, "
        "gratuitous, or NONE."
    ),
    "device_role": (
        "What comedic role does the transgressive content play: "
        "punchline, escalation, character-voice, or NONE?"
    ),
}

_NONE_VALUES = {"", "none", "n/a", "na", "unclear", "unknown"}
_DEVICE_ROLES = {"punchline", "escalation", "character-voice", "character voice"}

_PROFANITY_PAT = re.compile(
    r"\b(?:fuck\w*|shit\w*|cunt\w*|damn\w*|hell|ass(?:hole)?\w*|bastard\w*|"
    r"crap\w*|bitch\w*)\b", re.I)


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
        purpose = _norm_field(ex.get("profanity_purpose", ""))
        role = _norm_field(ex.get("device_role", ""))

        hits = [m.start() for m in _PROFANITY_PAT.finditer(t)]
        count = len(hits)
        n = max(1, len(t))
        in_tail = any(pos / n >= 0.75 for pos in hits)

        if count == 0 and purpose in _NONE_VALUES:
            base = 0.8  # vacuously fine -- no transgressive content present
        elif purpose == "purposeful":
            base = 0.8
            if in_tail:
                base += 0.1  # positioned as/near the punchline
            if count >= 4:
                base -= 0.1  # heavy repetition cuts against deliberate economy
        elif purpose == "gratuitous":
            base = 0.15
            if count >= 3:
                base -= 0.05
        else:
            # no LLM judgment: fall back to code-only positional heuristic
            if count == 0:
                base = 0.7
            elif count <= 1 and in_tail:
                base = 0.6
            elif count >= 3:
                base = 0.3
            else:
                base = 0.5

        if role in _DEVICE_ROLES:
            base += 0.05

        # --- corroboration: corpus redundancy as a well-worn-shock-riff proxy ---
        if count > 0:
            try:
                neighbors = ops.retrieve_similar(t, k=5) or []
            except Exception:
                neighbors = []
            sims = [s for s, _ in neighbors] if neighbors else []
            if sims:
                avg_sim = sum(sims) / len(sims)
                if avg_sim >= 0.55:
                    base -= 0.05  # heavily-recycled premise, leans gratuitous
                elif avg_sim <= 0.15:
                    base += 0.03  # distinctive deployment, leans purposeful

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

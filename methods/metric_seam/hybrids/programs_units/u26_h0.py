"""u26 Laughter as Success Metric: code detects explicit in-text laughter/reaction tokens and setup/punchline sentence structure as textual analogs of audience engagement; LLM fields carry whether the text reports a genuine reaction and what the perspective shift actually is."""

import re

LLM_FIELDS = {
    "reaction_evidence": (
        "Does the text itself describe or imply a real reader/audience "
        "reaction of genuine laughter (e.g. 'everyone laughed', 'I lol'd', "
        "a crowd's response)? Answer: yes or no."
    ),
    "shift_clarity": (
        "In <=10 words, describe the perspective shift or twist between "
        "the joke's setup and its punchline, or say 'none'."
    ),
}

_LAUGH_TOKEN_RE = re.compile(
    r"\b(lol+|lmao+|rofl|ha(?:ha)+h?a?|everyone laughed|burst out laughing|"
    r"crowd (?:roared|went wild)|cracked (?:everyone|us|me) up)\b",
    re.IGNORECASE,
)


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        reaction = str(extracted.get("reaction_evidence", "") or "").strip().lower()
        has_reaction = reaction.startswith("y")

        shift = str(extracted.get("shift_clarity", "") or "").strip().lower()
        has_shift = bool(shift) and shift not in ("none", "n/a", "")

        # --- primary construct: a reported real reaction is the strongest
        # textual analog of "actual audience engagement/laughter"; code
        # owns the mapping, LLM supplies the contextual read (raw laugh
        # tokens alone are a weak proxy -- they can be in-story content
        # rather than a meta-reaction). ---
        base = 0.75 if has_reaction else 0.35

        # --- code-only structural corroboration: explicit in-text
        # laughter/reaction tokens as a redundant cross-check. ---
        laugh_hits = len(_LAUGH_TOKEN_RE.findall(norm))
        struct = min(0.1, 0.05 * laugh_hits)

        # --- secondary competing criterion: clarity of the perspective
        # shift/setup structure (LLM-grounded thick read, code-corroborated
        # by a discernible setup/punchline sentence split). ---
        sentences = [s for s in re.split(r"[.!?]+", norm) if s.strip()]
        shift_bonus = 0.15 if has_shift else 0.0
        if has_shift and len(sentences) >= 2:
            shift_bonus += 0.05

        s = base + struct + shift_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

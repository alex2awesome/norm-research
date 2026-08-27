"""u21 Recognizable Stock Characters: code carries a stock-archetype lexicon, a literal cliche-opener detector, and corpus-redundancy (evidence op) as a well-worn-riff proxy; LLM fields carry which persona the joke draws on and whether it reads as a genuinely tired cliche."""

import re

LLM_FIELDS = {
    "stock_character": (
        "Name (<=8 words) the recognizable stock character, persona, or "
        "archetype central to this joke (e.g. 'a blonde', 'a genie', 'the "
        "drunk husband'), or say 'none'."
    ),
    "cliche_flag": (
        "Is the joke's target or catchphrase a tired, overused cliche "
        "rather than a fresh use of the character? Answer: yes or no."
    ),
}

_ARCHETYPE_RE = re.compile(
    r"\b(blonde\w*|lawyer\w*|priest\w*|rabbi\w*|imam\w*|nun\w*|genie\w*|"
    r"bartender\w*|mother-in-law|doctor\w*|nurse\w*|engineer\w*|"
    r"programmer\w*|irishman|englishman|scotsman|redneck\w*|cop\w*|"
    r"policeman|teacher\w*|boss\w*|drunk\w*|ghost\w*|mailman|farmer\w*|"
    r"salesman|accountant\w*|politician\w*|waiter\w*|waitress\w*)\b",
    re.IGNORECASE,
)

_CLICHE_OPENER_RE = re.compile(
    r"\b(?:a|an)\s+\w+\s+walks?\s+into\s+a\s+bar\b|"
    r"\bknock,?\s*knock\b|"
    r"\bwhy did the chicken cross the road\b|"
    r"\b(?:a|an)\s+\w+,\s*(?:a|an)\s+\w+,?\s*and\s+(?:a|an)\s+\w+\s+walk\w*\s+into\s+a\s+bar\b",
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

        character = str(extracted.get("stock_character", "") or "").strip().lower()
        has_character = bool(character) and character not in ("none", "n/a")
        cliche_flag = str(extracted.get("cliche_flag", "") or "").strip().lower().startswith("y")

        archetype_hits = set(m.lower() for m in _ARCHETYPE_RE.findall(norm))

        # --- dominant construct: drawing on a recognizable stock character
        # or ensemble is the reward signal; LLM field is primary, code
        # lexicon is a fallback/cross-check. ---
        if has_character:
            base = 0.7
        elif archetype_hits:
            base = 0.55
        else:
            base = 0.25

        # --- code-only structural corroboration: an ensemble cast
        # (multiple distinct archetypes) is a stronger use of the device. ---
        struct = 0.08 if len(archetype_hits) >= 2 else 0.0

        # --- competing criterion: penalize a literal, tired cliche
        # opener/catchphrase and an LLM-confirmed stale target. ---
        penalty = 0.0
        if _CLICHE_OPENER_RE.search(norm):
            penalty += 0.2
        if cliche_flag:
            penalty += 0.3

        # --- evidence op: corpus-redundancy as a well-worn-riff proxy --
        # high similarity to other corpus jokes suggests a widely
        # circulating stock template rather than a fresh character use. ---
        try:
            neighbors = ops.retrieve_similar(norm, k=5)
        except Exception:
            neighbors = []
        sims = []
        for item in neighbors or []:
            try:
                sims.append(float(item[0]))
            except Exception:
                continue
        if sims:
            avg_sim = sum(sims) / len(sims)
            if avg_sim >= 0.6:
                penalty += 0.1

        s = base + struct - penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

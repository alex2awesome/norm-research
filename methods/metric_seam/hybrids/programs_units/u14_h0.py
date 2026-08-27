"""u14 Improvisational Performance Craft: code counts direct-address/dialogue density as the textual analog of crowdwork; LLM fields carry the spontaneous-vs-scripted read and caricature-character detection (stage constructs code cannot see in text alone)."""

import re

LLM_FIELDS = {
    "interaction_style": (
        "Does the text read as unscripted, conversational, or directly "
        "addressing/reacting to a listener (crowdwork-style), or as a fixed "
        "scripted narrative? Answer: interactive or scripted."
    ),
    "caricature_character": (
        "In <=8 words, name the exaggerated, caricature-style character this "
        "joke's comedy is built around, or say 'none'."
    ),
}

_DIRECT_ADDRESS_RE = re.compile(r"\byou(?:'re|'ll|'ve|'d)?\b", re.IGNORECASE)
_IMPERATIVE_OPEN_RE = re.compile(r"^\s*[A-Z][a-z]+\b.*[.!?]\s*$", re.MULTILINE)
_ASIDE_RE = re.compile(r"\([^)]{2,60}\)|\[[^\]]{2,60}\]")


def _classify_interaction(raw):
    if not raw:
        return None
    s = raw.lower()
    if "interactive" in s or "unscripted" in s or "crowdwork" in s or "conversational" in s:
        return "interactive"
    if "scripted" in s:
        return "scripted"
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
        tl = norm.lower()

        # --- code-only textual analog of live improv/crowdwork: density of
        # direct second-person address and parenthetical unscripted-feeling
        # asides. A written joke has no stage, so this is the closest signal
        # code can reach on its own. ---
        n_you = len(_DIRECT_ADDRESS_RE.findall(tl))
        n_aside = len(_ASIDE_RE.findall(norm))
        struct = min(0.15, 0.03 * n_you) + min(0.08, 0.04 * n_aside)

        # --- primary construct: LLM read on spontaneous vs. scripted feel.
        # Because this is fundamentally a stage/live construct, the base
        # rate for a written joke should default toward the low/scripted
        # end absent positive evidence. ---
        style = _classify_interaction(str(extracted.get("interaction_style", "") or ""))
        if style == "interactive":
            base = 0.62
        elif style == "scripted":
            base = 0.22
        else:
            base = 0.3  # weak default: most text jokes are scripted, not live improv

        # --- secondary competing sub-criterion: caricature-based character
        # design, which *does* transfer to text (an exaggerated character
        # sketch is legible on the page). ---
        carr = str(extracted.get("caricature_character", "") or "").strip().lower()
        carr_bonus = 0.12 if (carr and carr not in ("none", "n/a", "")) else 0.0

        s = base + struct + carr_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

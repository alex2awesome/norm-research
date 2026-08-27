"""u24 Caricature and Meme Exaggeration: code detects hyperbole/intensifier density as a structural exaggeration proxy; LLM fields carry the named real-trait grounding and the offense/handling-tone judgment (whether amplification vs. arbitrary distortion)."""

import re

LLM_FIELDS = {
    "exaggerated_trait": (
        "In <=8 words, name the real, widely-recognized trait or "
        "stereotype this joke exaggerates for effect, or say 'none'."
    ),
    "handling_tone": (
        "Is the exaggeration handled with restraint/wit, or is it "
        "gratuitous/mean-spirited? Answer: restrained, gratuitous, or neutral."
    ),
}

_INTENSIFIER_RE = re.compile(
    r"\b(always|never|every single|literally|so damn|constantly|"
    r"absolutely|completely|totally|extremely|the most|nothing but)\b",
    re.IGNORECASE,
)
_ALLCAPS_RE = re.compile(r"\b[A-Z]{4,}\b")


def _classify_tone(raw):
    if not raw:
        return None
    s = raw.lower()
    if "gratuitous" in s or "mean" in s or "harsh" in s or "cruel" in s:
        return "gratuitous"
    if "restrained" in s or "witty" in s or "wit" in s or "clever" in s:
        return "restrained"
    if "neutral" in s:
        return "neutral"
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

        # --- primary construct: caricature/exaggeration must amplify an
        # already-existing, widely recognized trait, not arbitrary
        # distortion. Presence/grounding of a NAMED real trait is the field
        # signal; code just checks it isn't empty/none. ---
        trait = str(extracted.get("exaggerated_trait", "") or "").strip().lower()
        has_trait = bool(trait) and trait not in ("none", "n/a")
        base = 0.72 if has_trait else 0.32

        # --- code-only structural corroboration: hyperbole/intensifier and
        # shouted-emphasis density, a weak but independent signal that the
        # text is leaning on exaggeration as a device at all. ---
        n_int = len(_INTENSIFIER_RE.findall(norm))
        n_caps = len(_ALLCAPS_RE.findall(norm))
        struct = min(0.1, 0.03 * n_int) + min(0.06, 0.02 * n_caps)
        # structural exaggeration cues only count as corroboration when the
        # field also grounded a real trait; on their own they're too weak
        # (topic/emphasis words are known weak proxies in this corpus)
        if not has_trait:
            struct *= 0.3

        # --- secondary competing criterion: sensitivity to timing/
        # offensiveness in how the exaggeration is handled. ---
        tone = _classify_tone(str(extracted.get("handling_tone", "") or ""))
        if tone == "restrained":
            tone_adj = 0.1
        elif tone == "gratuitous":
            tone_adj = -0.15
        else:
            tone_adj = 0.0

        s = base + struct + tone_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

"""u1 Factual Accuracy in Political Comedy: code detects political/journalism topicality via lexicon (with a vacuous-pass default when absent) plus a dated-claim corroboration check; LLM fields carry which real figure/event is invoked and whether its factual claim is distorted."""

import re

LLM_FIELDS = {
    "political_target": (
        "Name the real political figure, party, government body, or news "
        "outlet central to this joke, in <=8 words, or say 'none'."
    ),
    "fact_distortion": (
        "Does the joke state or imply a false/misleading claim about that "
        "real figure/event for its punchline? Answer: accurate, distorted, "
        "or none."
    ),
}

_POLITICAL_RE = re.compile(
    r"\b(president|senator|congress\w*|politician\w*|election\w*|"
    r"campaign\w*|governor|mayor|prime minister|parliament|democrat\w*|"
    r"republican\w*|white house|senate|policy|vote\w*|ballot\w*|trump|"
    r"biden|obama|putin|merkel|clinton|pelosi)\b",
    re.IGNORECASE,
)
_JOURNALISM_RE = re.compile(
    r"\b(journalist\w*|reporter\w*|newspaper\w*|headline\w*|"
    r"press conference|breaking news|news anchor|editorial\w*)\b",
    re.IGNORECASE,
)


def _classify_distortion(raw):
    if not raw:
        return None
    s = raw.lower()
    if s in ("none", "n/a", ""):
        return "none"
    if any(k in s for k in ("distort", "false", "misleading", "inaccurate", "made up", "exaggerat")):
        return "distorted"
    if any(k in s for k in ("accurate", "true", "correct", "factual")):
        return "accurate"
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

        target = str(extracted.get("political_target", "") or "").strip().lower()
        has_target = bool(target) and target not in ("none", "n/a")
        distortion = _classify_distortion(str(extracted.get("fact_distortion", "") or ""))

        code_political = bool(_POLITICAL_RE.search(norm)) or bool(_JOURNALISM_RE.search(norm))
        is_political = code_political or has_target

        if not is_political:
            # Criterion doesn't bite on non-political/non-journalism humor:
            # nothing to hold to a "higher standard", so vacuously fine.
            return 0.65

        # --- primary construct: code owns the mapping from the LLM's
        # accuracy read to a score; a flagged distortion is the strongest
        # negative signal, a confirmed-accurate/no-claim reading is positive. ---
        if distortion == "distorted":
            base = 0.15
        elif distortion in ("accurate", "none"):
            base = 0.8
        else:
            base = 0.5  # political but no clear field signal

        # --- code-only structural corroboration: a specific, dated claim
        # is more checkable (and thus more "held to a higher standard")
        # than a vague, undated one; a named real target adds specificity. ---
        try:
            dates = ops.extract_dates(norm)
        except Exception:
            dates = []
        struct = 0.05 if dates else 0.0
        if has_target:
            struct += 0.05
        struct = min(0.1, struct)

        s = base + struct
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

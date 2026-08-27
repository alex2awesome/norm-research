"""u9 Premise Consistency & Craft: code measures economy/timing shape and alliteration; LLM fields carry internal-consistency judgment and named verbal-device identification."""

import re

LLM_FIELDS = {
    "consistency": (
        "Is the joke's premise/plot internally consistent throughout, with "
        "no contradicted or dropped details? Answer: consistent, minor-issue, or inconsistent."
    ),
    "verbal_device": (
        "In <=6 words, name the verbal device the punchline relies on "
        "(e.g. pun, alliteration, homophone), or say 'none'."
    ),
}

_ALLIT_RE = re.compile(r"\b([a-z])[a-z']*\s+\1[a-z']*\b", re.IGNORECASE)


def _classify_consistency(raw):
    if not raw:
        return None
    s = raw.lower()
    if "inconsistent" in s or "contradict" in s:
        return "inconsistent"
    if "minor" in s:
        return "minor-issue"
    if "consistent" in s:
        return "consistent"
    return None


def _classify_device(raw):
    if not raw:
        return None
    s = raw.lower()
    if s in ("none", "n/a", ""):
        return None
    if any(k in s for k in ("pun", "homophone", "homonym", "alliteration",
                             "wordplay", "double meaning", "rhyme")):
        return s
    return "other"


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(norm)
        except Exception:
            n_sent, mean_wps, frac_long = 1, 10.0, 0.0

        # --- primary: internal consistency (LLM-grounded) ---
        cons = _classify_consistency(str(extracted.get("consistency", "") or ""))
        if cons == "consistent":
            base = 0.75
        elif cons == "minor-issue":
            base = 0.45
        elif cons == "inconsistent":
            base = 0.15
        else:
            base = 0.5  # no field signal: structure-only fallback below

        # --- primary: polished craft/timing, code-only proxy (economy: low
        # rate of long/rare words, and a tight final line relative to the
        # buildup, both read as "precise, polished") ---
        sentences = [s for s in re.split(r"[.!?]+", norm) if s.strip()]
        last_words = len(sentences[-1].split()) if sentences else 0
        craft = 0.0
        if frac_long < 0.12:
            craft += 0.06
        if mean_wps and last_words and last_words < 0.7 * mean_wps:
            craft += 0.06
        craft = min(0.12, craft)

        # --- secondary competing criterion: specific verbal devices
        # (alliteration/homophonic puns). Code confirms alliteration
        # structurally; LLM names puns/homophones code can't detect. ---
        device_bonus = 0.0
        if _ALLIT_RE.search(norm):
            device_bonus += 0.05
        device = _classify_device(str(extracted.get("verbal_device", "") or ""))
        if device in ("pun", "homophone", "homonym", "alliteration", "wordplay",
                      "double meaning", "rhyme"):
            device_bonus += 0.1
        elif device == "other":
            device_bonus += 0.04
        device_bonus = min(0.15, device_bonus)

        s = base + craft + device_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

"""u4 Clear Premise Setup: code scores structural setup/punchline shape; LLM fields carry setup-clarity judgment and venue/format-fit (thick semantic reads code can't make)."""

import re

LLM_FIELDS = {
    "setup_clarity": (
        "Does the joke have one clear line/sentence that sets up a premise "
        "and an expectation before the punchline? Answer: clear, muddled, or absent."
    ),
    "format_fit": (
        "In <=8 words, name the specific event/venue/format this joke is "
        "tailored to (e.g. roast, holiday, contest theme), or say 'none'."
    ),
}

_QUESTION_SETUP_RE = re.compile(r"^[^.!?]{5,140}\?", re.DOTALL)
_TITLE_ECHO_RE = re.compile(r"^([^\n.!?]{5,90})[.\s]+\1", re.IGNORECASE)


def _classify_clarity(raw):
    if not raw:
        return None
    s = raw.lower()
    if any(k in s for k in ("clear", "strong", "well-set", "well set", "explicit")):
        return "clear"
    if any(k in s for k in ("absent", "none", "missing", "no setup")):
        return "absent"
    if any(k in s for k in ("muddled", "vague", "weak", "unclear", "confus")):
        return "muddled"
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

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(norm)
        except Exception:
            n_sent, mean_wps, frac_long = 1, 10.0, 0.0

        # --- code-only structural corroboration: is there a distinguishable
        # setup portion (buildup) separate from a final short punch, or an
        # explicit question-form setup / title-echo pattern common in this
        # corpus? ---
        struct = 0.0
        sentences = [s for s in re.split(r"[.!?]+", norm) if s.strip()]
        last_words = len(sentences[-1].split()) if sentences else 0
        if n_sent >= 2:
            struct += 0.08
        if mean_wps and last_words and last_words < 0.75 * mean_wps:
            struct += 0.06  # buildup longer than the final punch line
        if _QUESTION_SETUP_RE.search(norm.strip()):
            struct += 0.06  # classic Q&A joke setup
        if _TITLE_ECHO_RE.search(norm.strip()):
            struct += 0.03  # title restates as the opening setup sentence
        struct = min(0.2, struct)

        # --- LLM-grounded primary construct: is the setup actually clear? ---
        clarity = _classify_clarity(str(extracted.get("setup_clarity", "") or ""))
        if clarity == "clear":
            base = 0.78
        elif clarity == "muddled":
            base = 0.45
        elif clarity == "absent":
            base = 0.18
        else:
            # no usable field signal: fall back to structure alone, centered
            base = 0.5

        # --- secondary competing criterion: venue/format alignment. Rare
        # in a scraped-joke corpus (no live venue), so it can only add a
        # small bonus, never a penalty. ---
        fmt = str(extracted.get("format_fit", "") or "").strip().lower()
        fmt_bonus = 0.08 if (fmt and fmt not in ("none", "n/a", "")) else 0.0

        s = base + struct + fmt_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

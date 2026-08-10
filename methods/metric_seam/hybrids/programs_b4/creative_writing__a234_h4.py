"""Hybrid metric channel for a234 (4-field) -- "Rasa: Aesthetic-Emotional Flavor".

Judge signal (from train residuals): NOT emotion-keyword density (baseline
rho=.03). The judge rewards a SINGLE distinctive aesthetic-emotional flavor,
sustained with craft; it punishes tonal scatter, amateur prose, and
frame-breaking author chatter (edit notes, apologies, reader address).
Predicate lives in code; two LLM fields supply the thick-input reads that
regex cannot see (is one flavor sustained? is the prose controlled?).

Two new LLM fields target a gap: `rasa_strength` and `prose_craft` grade a
single named flavor and general craft, but neither checks (a) whether that
flavor actually holds to the close of the piece rather than being undercut
by a late joke/tonal swerve -- "evokes... sustained" is a whole-piece claim,
not a peak reading -- or (b) whether the emotion is *embodied* (grounded in
sensory/physical detail, per "a psychology of primary emotions") rather than
merely named/told. `tone_ending` and `emotion_grounding` probe those two
construct-relevant gaps directly from the text.
"""

import re

LLM_FIELDS = {
    "rasa_strength": (
        "Name this story's single dominant emotional flavor (e.g. sorrow, humor, "
        "wonder, dread, tenderness) and rate its evocation WEAK, MODERATE, or STRONG "
        "(e.g. 'melancholy STRONG'); answer NONE if the tone is scattered, flat, or inconsistent."
    ),
    "prose_craft": (
        "Rate this story's prose craftsmanship (control, rhythm, word choice) with "
        "exactly one word: CRUDE (clumsy, error-ridden), PLAIN (competent but flat), "
        "or POLISHED (vivid, controlled)."
    ),
    "tone_ending": (
        "Does the ending INTENSIFY, SUSTAIN, or UNDERCUT (joke/tonal swerve that "
        "deflates it) the story's established emotional tone? Answer in ONE word."
    ),
    "emotion_grounding": (
        "Is the dominant emotion mainly SHOWN through sensory/physical detail, or "
        "just TOLD/named directly? Answer SHOWN, TOLD, or NONE (no clear emotion)."
    ),
}

# --- frame-break / amateur-chatter detectors (positions matter) -------------

# Class A: fiction-frame breaks. Heavy when they intrude at head/middle;
# a short signature in the final stretch of a strong piece is near-benign.
_A_PATTERNS = [
    r"\bedit\s*\d*\s*:",                      # Edit:, Edit 2:, Final Edit:
    r"\[\s*(?:wp|poem|pi|eu|cw|sp|tt)\s*\]",  # prompt tags
    r"(?:^|[\s(])/?r/\w+",                    # subreddit links
    r"\bwritingprompts\b",
    r"\bupvot\w+|\bgild(?:ed|ing)\b|thanks for the gold",
    r"part\s+\d+\s+(?:of\s+\d+|here|in\s+comments)",
    r"end of recording",                       # benign device; excluded below
    r";\)|:\)|:-\)|:p\b|\bxd\b",
]
_A_EXCLUDE = {r"end of recording"}  # placeholder for in-frame devices (unused penalty)

# Class B: author insecurity / reader address -- amateur tells anywhere.
_B_PATTERNS = [
    r"first\s+(?:time\s+)?post(?:er)?\b",
    r"my\s+first\s+(?:ever\s+)?(?:post|story|submission)",
    r"long\s+time\s+lurker",
    r"let\s+me\s+know\s+what\s+you\s+think",
    r"(?:any\s+)?feedback\s+(?:is\s+)?welcome",
    r"don'?t\s+judge",
    r"don'?t\s+tear\s+me\s+apart",
    r"\bpls\b|\bplz\b",
    r"sorry\s+(?:for|about)\s+(?:the\s+)?(?:typos|formatting|grammar|mistakes)",
    r"i\s+guess\s+i\s+gotta|gotta\s+contribute",
    r"hope\s+you\s+(?:guys\s+)?(?:like|enjoy)(?:d)?\s+(?:it|this)",
]

_STRONG_WORDS = ("strong", "powerful", "intense", "masterful")
_MID_WORDS = ("moderate", "medium", "solid")
_WEAK_WORDS = ("weak", "mild", "faint", "slight")
_NONE_WORDS = ("none", "scattered", "flat", "inconsistent", "mixed", "unclear",
               "no single", "no dominant")

_CRAFT_HI = ("polished", "masterful", "vivid", "skillful", "excellent")
_CRAFT_LO = ("crude", "clumsy", "amateur", "rough", "error", "poor", "sloppy")
_CRAFT_MID = ("plain", "competent", "average", "adequate", "serviceable")


def _rasa_signal(ans):
    """Map the rasa_strength field to [0,1]; empty/NONE means scattered -> low."""
    a = (ans or "").strip().lower()
    if not a:
        return 0.10
    if any(w in a for w in _NONE_WORDS):
        return 0.10
    if any(w in a for w in _STRONG_WORDS):
        return 1.00
    if any(w in a for w in _WEAK_WORDS):
        return 0.30
    if any(w in a for w in _MID_WORDS):
        return 0.60
    # Named a flavor but no usable strength word: mid prior.
    return 0.55


def _craft_signal(ans):
    """Map the prose_craft field to [0,1]; empty is uninformative -> 0.5."""
    a = (ans or "").strip().lower()
    if not a:
        return 0.50
    if any(w in a for w in _CRAFT_LO):
        return 0.10
    if any(w in a for w in _CRAFT_HI):
        return 1.00
    if any(w in a for w in _CRAFT_MID):
        return 0.50
    return 0.50


def _meta_penalty(tl):
    """Position-weighted penalty for frame breaks + amateur chatter."""
    n = max(1, len(tl))
    a_sum = 0.0
    tail_a = 0.0
    for pat in _A_PATTERNS:
        if pat in _A_EXCLUDE:
            continue
        for m in re.finditer(pat, tl):
            pos = m.start() / float(n)
            if pos >= 0.85:
                tail_a += 0.4          # signature zone: mild, and capped below
            elif pos <= 0.15:
                a_sum += 1.4           # chatter before the story even starts
            else:
                a_sum += 1.0           # mid-story frame break
    a_sum += min(tail_a, 1.0)          # tail signatures don't stack
    a_sum = min(a_sum, 4.0)

    b_count = 0
    for pat in _B_PATTERNS:
        if re.search(pat, tl):
            b_count += 1
    # Author chatter before the story starts ("guess I'll post", "the prompt...")
    if re.search(r"\b(?:post(?:ing)?|poster|prompt|contribute|lurker)\b", tl[:150]):
        b_count += 1
    b_count = min(b_count, 3)

    return min(0.30, 0.055 * a_sum + 0.075 * b_count)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text if isinstance(text, str) else ""
        if not t.strip():
            return 0.5
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        tl = t.lower()

        ex = extracted if isinstance(extracted, dict) else {}
        rasa = _rasa_signal(ex.get("rasa_strength", ""))
        craft = _craft_signal(ex.get("prose_craft", ""))

        raw = 0.12 + 0.55 * rasa + 0.30 * craft

        raw -= _meta_penalty(tl)

        # New: does the flavor hold to the close, or get undercut/deflated?
        if "tone_ending" in ex:
            te = str(ex.get("tone_ending", "") or "").strip().lower()
            if "undercut" in te:
                raw -= 0.10
            elif "intensify" in te:
                raw += 0.06
            elif "sustain" in te:
                raw += 0.03

        # New: is the emotion embodied (shown) or merely named (told)?
        if "emotion_grounding" in ex:
            egnd = str(ex.get("emotion_grounding", "") or "").strip().lower()
            if "shown" in egnd:
                raw += 0.06
            elif "told" in egnd:
                raw -= 0.04
            elif "none" in egnd:
                raw -= 0.08

        # Extremely thin fragments rarely establish any rasa: soften upward claims.
        try:
            n_words = len(re.findall(r"[A-Za-z']+", tl))
            if n_words < 60:
                raw = min(raw, 0.45)
        except Exception:
            pass

        if raw < 0.0:
            raw = 0.0
        if raw > 1.0:
            raw = 1.0
        return float(raw)
    except Exception:
        return 0.5

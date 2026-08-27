# Hybrid module for humor unit u5: "Consistent Stage Persona"
# Construct division: since these are text jokes rather than live sets, code
# scores the textual analog of a "well-separated, unbroken persona" --
# grammatical-person stability of the narrating voice (no jarring I/he/they
# switches), and the absence of author-intrusion breaks (edit notes,
# disclaimers, meta-apologies) that step outside the persona/character the
# piece has established. The two LLM fields carry what code cannot see:
# whether a distinct persona/character was even established, and whether
# that persona's voice audibly breaks or shifts partway through the piece.

import re

LLM_FIELDS = {
    "persona": (
        "In <=8 words, name the narrator/persona or main character's voice "
        "(e.g. 'wry first-person retiree', 'omniscient narrator', 'grumpy "
        "boss'), or say NONE if no distinct persona is established."
    ),
    "voice_break": (
        "In <=10 words: does the narrator's or a character's voice/persona "
        "shift or break inconsistently partway through the piece? Answer "
        "'yes' or 'no'."
    ),
}

_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'll|I'd|me|my|mine|myself)\b")
_THIRD_PERSON_RE = re.compile(r"\b(he|him|his|she|her|hers|they|them|their)\b", re.IGNORECASE)

_META_INTRUSION_RE = re.compile(
    r"\b(edit\d*\s*:|not (?:a\s+)?racist|please don'?t (?:kill|murder)|"
    r"disclaimer|jk|just kidding|no offense|sorry (?:if|guys)|"
    r"first time posting|repost)\b",
    re.IGNORECASE,
)


def _strip_quotes(t):
    """Remove quoted dialogue so grammatical-person analysis targets narration."""
    return re.sub(r"[\"“”].*?[\"“”]", " ", t, flags=re.DOTALL)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        narration = _strip_quotes(t)
        fp = len(_FIRST_PERSON_RE.findall(narration))
        tp = len(_THIRD_PERSON_RE.findall(narration))
        total_p = fp + tp

        # --- code: grammatical-person consistency of the narrating voice ---
        if total_p == 0:
            person_consistency = 0.6  # no narrating voice detectable either way
        else:
            dominant = max(fp, tp)
            person_consistency = dominant / total_p  # 1.0 = pure 1st or pure 3rd

        # --- code: sentence-length rhythm stability (a persona reads as
        # "polished/confident" when delivery rhythm is not erratic) ---
        try:
            n_sent, mean_wps, _frac_long = ops.sent_stats(t)
        except Exception:
            n_sent, mean_wps = 1, 0.0
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        if len(sents) >= 3:
            lens = [len(re.findall(r"\w+", s)) for s in sents]
            mean_len = sum(lens) / len(lens)
            if mean_len > 0:
                var = sum((l - mean_len) ** 2 for l in lens) / len(lens)
                cv = (var ** 0.5) / mean_len
                rhythm_stability = max(0.3, 1.0 - min(cv, 1.4) / 1.4 * 0.7)
            else:
                rhythm_stability = 0.6
        else:
            rhythm_stability = 0.7

        # --- code: author-intrusion breaks the illusion of a maintained persona ---
        intrusion_hits = len(_META_INTRUSION_RE.findall(raw))
        intrusion_penalty = min(0.4, 0.2 * intrusion_hits)

        # --- LLM-field grounding ---
        extracted = extracted or {}
        persona = str(extracted.get("persona", "") or "").strip().lower()
        voice_break = str(extracted.get("voice_break", "") or "").strip().lower()

        persona_established = 1.0 if persona and persona not in ("none", "n/a", "") else 0.4
        break_penalty = 0.5 if voice_break.startswith("yes") else 0.0

        combined = (
            0.30 * person_consistency +
            0.25 * rhythm_stability +
            0.45 * persona_established
        )
        out = combined - intrusion_penalty - break_penalty
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5

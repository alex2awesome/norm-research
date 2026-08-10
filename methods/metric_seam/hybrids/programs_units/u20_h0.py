# Hybrid module for humor unit u20: "Candid Personal Authenticity"
# Construct division: code carries the surface-lexical signal of
# first-person, vulnerability-toned material (dense "I/me/my" narration
# plus a struggle/embarrassment lexicon) versus the classic deflecting
# opener that signals a generic, impersonal joke ("a guy walks into a
# bar...") and a self-glorifying-boast lexicon that code can spot directly.
# The two LLM fields carry the semantic judgment code cannot make on its
# own: whether the piece genuinely reads as the narrator's own real
# experience (vs. a fictional/impersonal joke), and whether the narrator is
# boasting/self-glorifying rather than being candid about a struggle.

import re

LLM_FIELDS = {
    "is_personal_experience": (
        "In <=8 words: does this read as the narrator's own real personal "
        "experience or struggle, or a generic/fictional joke? Answer "
        "'personal' or 'generic/fictional'."
    ),
    "self_glorifying": (
        "In <=8 words: does the narrator boast or self-glorify rather than "
        "admit a genuine vulnerability? Answer 'yes' or 'no'."
    ),
}

_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'll|I'd|me|my|mine|myself)\b")

_GENERIC_OPENER_RE = re.compile(
    r"^\s*(a\s+(?:man|guy|woman|priest|rabbi|blonde|lawyer|doctor|farmer)\s+"
    r"(?:walks|walked|goes|went)|two\s+\w+\s+(?:walk|walked)|"
    r"a\s+\w+\s+and\s+a\s+\w+\s+(?:walk|walked))",
    re.IGNORECASE,
)

_VULNERABILITY_RE = re.compile(
    r"\b(embarrass\w*|struggl\w*|failed|failure|therapy|therapist|depress\w*|"
    r"anxious|anxiety|divorce[d]?|broke\b|lonely|alone|cried|crying|"
    r"ashamed|insecur\w*|rejected|rejection|regret\w*)\b",
    re.IGNORECASE,
)

_BOAST_RE = re.compile(
    r"\b(i'?m the best|everyone (?:loves|wants) me|so talented|i'?m amazing|"
    r"i always win|i'?m a genius|obviously i|clearly i'?m)\b",
    re.IGNORECASE,
)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        words = re.findall(r"\w+", t)
        nw = max(1, len(words))

        # --- code: first-person density (personal material is narrated as "I") ---
        fp_hits = len(_FIRST_PERSON_RE.findall(t))
        fp_density = min(1.0, fp_hits / (nw / 25.0)) if nw else 0.0  # ~1 fp word per 25 tokens = saturating

        # --- code: generic stock-character opener deflects away from the
        # personal (classic "a guy walks into a bar" is structurally the
        # opposite of candid personal material) ---
        generic_penalty = 0.35 if _GENERIC_OPENER_RE.search(t.strip()) else 0.0

        # --- code: vulnerability/struggle lexicon ---
        vuln_hits = len(_VULNERABILITY_RE.findall(t))
        vuln_score = min(1.0, vuln_hits / 3.0)

        # --- code: self-glorification lexicon ---
        boast_hits = len(_BOAST_RE.findall(t))
        boast_penalty = min(0.4, 0.3 * boast_hits)

        # --- LLM-field grounding ---
        extracted = extracted or {}
        is_personal = str(extracted.get("is_personal_experience", "") or "").strip().lower()
        self_glorifying = str(extracted.get("self_glorifying", "") or "").strip().lower()

        if "personal" in is_personal and "generic" not in is_personal and "fictional" not in is_personal:
            personal_component = 1.0
        elif is_personal:
            personal_component = 0.15
        else:
            personal_component = 0.5  # extractor gave nothing usable

        glorify_penalty = 0.4 if self_glorifying.startswith("yes") else 0.0

        combined = (
            0.20 * fp_density +
            0.20 * vuln_score +
            0.60 * personal_component
        )
        out = combined - generic_penalty - boast_penalty - glorify_penalty
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5

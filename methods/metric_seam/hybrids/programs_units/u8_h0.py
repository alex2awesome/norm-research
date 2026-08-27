"""u8 Authentic, Grounded Premise: code detects narrative-development structure (characters, dialogue, causal connectives) as a proxy for a genuine setup vs. a string of gags; LLM fields carry the semantic judgment of whether the premise is grounded with real conflict, and whether sensitive representation (children/LGBTQ+ intimacy) is handled respectfully."""

import re

LLM_FIELDS = {
    "scenario": (
        "Does the joke set up a specific character and situation with "
        "genuine conflict before the punchline, or is it just a pun/gag "
        "with no real scenario? Answer 'grounded scenario' or 'gag-only'."
    ),
    "sensitive_rep": (
        "If children or LGBTQ+ intimacy are depicted, is the treatment "
        "respectful and non-exploitative, or exploitative? Answer "
        "'respectful', 'exploitative', or 'n/a'."
    ),
}

_CHARACTER_RE = re.compile(
    r"\b(he|she|they|his|her|their|him|them|i|we|you)\b", re.IGNORECASE
)
_DIALOGUE_RE = re.compile(r'["“”].{2,}?["“”]', re.DOTALL)
_CONNECTIVE_RE = re.compile(
    r"\b(because|so|then|but|when|after|until|since|so that)\b", re.IGNORECASE
)
_GAG_ONLY_RE = re.compile(
    r"^\s*(q\s*:|why (?:did|does|do|is|are)\b|what do you call)", re.IGNORECASE
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

        try:
            n_sent, _mean_wps, _frac_long = ops.sent_stats(t)
        except Exception:
            n_sent = t.count(".") + t.count("!") + t.count("?") or 1

        struct = 0.0
        if n_sent >= 3:
            struct += 0.3
        if _DIALOGUE_RE.search(t):
            struct += 0.2
        if len(_CONNECTIVE_RE.findall(t)) >= 1:
            struct += 0.2
        if len(_CHARACTER_RE.findall(t)) >= 2:
            struct += 0.15
        if _GAG_ONLY_RE.search(t.strip()) and n_sent <= 2:
            struct -= 0.2
        struct = max(0.0, min(1.0, struct))

        extracted = extracted or {}
        scenario = str(extracted.get("scenario", "") or "").strip().lower()
        sensitive = str(extracted.get("sensitive_rep", "") or "").strip().lower()

        if "grounded scenario" in scenario:
            llm_adj = 0.35
        elif "gag-only" in scenario:
            llm_adj = -0.3
        else:
            llm_adj = 0.0

        if "exploitative" in sensitive and "non" not in sensitive:
            sens_adj = -0.4
        elif "respectful" in sensitive:
            sens_adj = 0.1
        else:
            sens_adj = 0.0

        s = 0.5 * struct + 0.3 + llm_adj + sens_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

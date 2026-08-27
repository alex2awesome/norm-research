"""u17 hybrid: code owns the predicate mapping (subject-present x acceptable-vs-gratuitous) plus a graphic-density regex corroboration signal; LLM fields carry what the taboo subject is and whether its handling reads as acceptable, which code cannot judge on its own."""

import re

LLM_FIELDS = {
    "taboo_subject": (
        "Name the taboo/sensitive subject present: violence, sexual "
        "content, real-world target, protected group, death, drugs, or "
        "NONE."
    ),
    "harm_assessment": (
        "Is the taboo material handled acceptably (non-gratuitous) or "
        "gratuitously/mean-spiritedly? Answer acceptable, gratuitous, or "
        "NONE."
    ),
}

_NONE_VALUES = {"", "none", "n/a", "na", "unclear", "unknown"}

_GRAPHIC_TERMS = re.compile(
    r"\b(?:fuck\w*|shit\w*|cunt\w*|nigger\w*|faggot\w*|rape\w*|blood\w*|"
    r"gore\w*|corpse\w*|mutilat\w*|torture\w*|murder\w*|kill\w*)\b", re.I)


def _norm_field(v):
    return (v or "").strip().lower().strip(". ")


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        ex = extracted or {}
        subject = _norm_field(ex.get("taboo_subject", ""))
        assess = _norm_field(ex.get("harm_assessment", ""))

        hit_count = len(_GRAPHIC_TERMS.findall(t))

        if subject in _NONE_VALUES:
            base = 0.75  # vacuously fine -- nothing taboo to navigate
        elif assess == "acceptable":
            base = 0.9
            if hit_count >= 4:
                base -= 0.05  # cumulative graphic intensity, slight caution
        elif assess == "gratuitous":
            base = 0.15
        else:
            base = 0.5  # taboo present, no clear judgment available
            if hit_count >= 3:
                base -= 0.15  # heavy graphic density with no acceptable signal

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

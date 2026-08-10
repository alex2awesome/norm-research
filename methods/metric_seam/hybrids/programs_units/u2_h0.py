"""u2 hybrid: code regex-gates on profanity/graphic-content markers and literal warning tags (mismatch = unrated explicit content); LLM fields carry the content-rating category and irreverent/incongruity judgment that need semantic reading."""

import re

LLM_FIELDS = {
    "content_level": (
        "Rate this joke's explicit content level as one word: clean, mild, "
        "or explicit (profanity, sexual, violent material)."
    ),
    "irreverent_incongruity": (
        "Does the joke rely on irreverent tone or absurd/incongruous "
        "juxtaposition for comedic effect? Answer yes, no, or NONE."
    ),
}

_NONE_VALUES = {"", "none", "n/a", "na", "unclear", "unknown"}

_STRONG_PROFANITY = re.compile(
    r"\b(?:fuck\w*|shit\w*|cunt\w*|nigger\w*|faggot\w*|motherfucker\w*|"
    r"cock\w*|pussy\w*|whore\w*|slut\w*)\b", re.I)
_MILD_PROFANITY = re.compile(r"\b(?:damn\w*|hell|ass(?:hole)?|bastard\w*|crap\w*)\b", re.I)
_GRAPHIC = re.compile(
    r"\b(?:blood\w*|gore\w*|rape\w*|corpse\w*|mutilat\w*|torture\w*)\b", re.I)
_WARNING = re.compile(
    r"\b(?:tw|cw)\s*[:\-]|trigger warning|content warning|nsfw|"
    r"not safe for work|18\+|mature (?:content|audiences)|viewer discretion", re.I)


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
        level = _norm_field(ex.get("content_level", ""))
        irrev = _norm_field(ex.get("irreverent_incongruity", ""))

        code_strong = bool(_STRONG_PROFANITY.search(t) or _GRAPHIC.search(t))
        code_mild = bool(_MILD_PROFANITY.search(t))
        warning_present = bool(_WARNING.search(t))

        # fall back to code-only bucket if the LLM field is missing/unusable
        if level not in ("clean", "mild", "explicit"):
            level = "explicit" if code_strong else ("mild" if code_mild else "clean")

        # --- primary: content-rating match + warning-carrying ---
        if level == "clean":
            base = 0.85
        elif level == "mild":
            base = 0.6 if code_strong else 0.75
        else:  # explicit
            base = 0.75 if warning_present else 0.25

        # --- secondary (small weight): irreverent tone / incongruity ---
        if irrev == "yes":
            base += 0.08
        elif irrev == "no":
            base -= 0.03

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

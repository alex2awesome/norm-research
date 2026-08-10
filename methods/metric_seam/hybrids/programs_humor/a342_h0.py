"""
a342 -- Parody/Spoof
Humorous imitation that subverts or critiques the style, genre, or conventions
of a source text or form, able to lampoon while paying homage.

Design notes:
  - Corpus is short reddit-style jokes; the vast majority are plain jokes with
    NO external form being imitated (setup/punchline, "walks into a bar",
    ethnic/blonde-joke templates). Those should score near zero.
  - The rare positive cases in this criterion involve the text visibly
    imitating a RECOGNIZABLE outside genre/format/work (an ad/commercial,
    a fable's "moral of the story" close, a toast/speech, an alphabet-primer
    list, a movie scene, a political-satire convention) and then twisting it.
  - Code alone can reliably detect a handful of concrete textual markers of
    imitated form (fable-moral phrasing, ad-copy phrasing, alphabet-list
    structure, toast/speech cues, jingle-style slogans). It cannot reliably
    tell whether a text is invoking a more abstract/referential convention
    (e.g. a political-joke or movie-parody frame) or whether an imitated
    convention is actually being twisted for comic effect -- those are
    THICK-INPUT judgments handed to the two LLM fields below.
  - Final score = LLM-identified imitation/subversion (dominant weight) +
    small code-marker corroboration bonus - a mild penalty when the text is
    highly generic/formulaic relative to the rest of the corpus (weak
    evidence op signal, not a substitute for judgment).
"""

import re

LLM_FIELDS = {
    "imitated_form": "Name in <=6 words the specific outside genre/format/work being imitated (ad jingle, movie scene, fable moral, political-joke convention, toast/speech, etc.), else NONE.",
    "subversion": "In <=8 words, say how the text twists or mocks that imitated form for comic effect, else NONE.",
}

_NONE_TOKENS = {"none", "n/a", "na", "no", "nothing", "n a"}

_FABLE_RE = re.compile(
    r"\bmoral(?:\s+of\s+(?:the|this)\s+story)?\s*[:\-]|\bonce\s+upon\s+a\s+time\b",
    re.I,
)
_AD_RE = re.compile(
    r"\bhere\s+at\b|\bact\s+now\b|\bcall\s+(?:now|today)\b|\bterms\s+and\s+conditions\b"
    r"|\b\d{1,2}\s*%\s*off\b|\blimited[\s\-]time\s+offer\b|\bcall\s+1[\-\s]?800\b",
    re.I,
)
_ALPHA_LIST_RE = re.compile(r"(?m)^\s*[A-Z]\s+(?:is|for)\s+\S+")
_SLOGAN_RE = re.compile(
    r"\bnothin'?g?\s+says\b|\bwhen\s+you\s+care\s+enough\b|\bbecause\s+you'?re\s+worth\s+it\b",
    re.I,
)
_TOAST_RE = re.compile(
    r"\bladies\s+and\s+gentlemen\b|\bspeech,?\s+speech\b|\bhear,?\s+hear\b", re.I
)


def _present(s):
    if not s or not isinstance(s, str):
        return False
    s2 = s.strip().lower().rstrip(".!")
    if not s2:
        return False
    if s2 in _NONE_TOKENS:
        return False
    return True


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.0

        t = ops.normalize(text)
        if not t or not t.strip():
            return 0.0

        # --- code-detectable imitation markers (corroborating evidence only) ---
        markers = 0
        if _FABLE_RE.search(t):
            markers += 1
        if _AD_RE.search(t):
            markers += 1
        if len(_ALPHA_LIST_RE.findall(t)) >= 3:
            markers += 1
        if _SLOGAN_RE.search(t):
            markers += 1
        if _TOAST_RE.search(t):
            markers += 1
        code_component = min(0.25, 0.08 * markers)

        # --- LLM-grounded imitation/subversion judgment (dominant weight) ---
        extracted = extracted or {}
        has_form = _present(extracted.get("imitated_form", ""))
        has_subv = _present(extracted.get("subversion", ""))

        if has_form and has_subv:
            llm_component = 0.75
        elif has_form:
            llm_component = 0.40
        elif has_subv:
            llm_component = 0.30
        else:
            llm_component = 0.0

        # --- weak corpus-typicality penalty (evidence op): highly generic /
        # formulaic text relative to the corpus is less likely to be a
        # deliberate, distinctive parody than an idiosyncratic one ---
        similarity_penalty = 0.0
        try:
            neighbors = ops.retrieve_similar(text, k=5)
            sims = []
            for item in neighbors or []:
                try:
                    for x in item:
                        if isinstance(x, (int, float)):
                            sims.append(float(x))
                            break
                except TypeError:
                    continue
            if sims:
                avg_sim = sum(sims) / len(sims)
                if avg_sim >= 0.6:
                    similarity_penalty = 0.05
        except Exception:
            similarity_penalty = 0.0

        if llm_component == 0.0 and code_component == 0.0:
            final = 0.04
        else:
            final = llm_component + code_component - similarity_penalty

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5

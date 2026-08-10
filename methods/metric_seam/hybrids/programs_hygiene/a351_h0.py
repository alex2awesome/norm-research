"""
Hybrid metric channel for aspect a351: "Representation ethics and harm minimization".

Criterion: avoid minimizing trauma for laughs; attend to ableist, racist, and gendered
harms; avoid false equivalences and exploitative portrayals; pursue remediation and
sensitivity.

Design notes:
- This criterion is a pure content/ethics judgment, not a craft/structure judgment, so
  no sentence-length or punctuation-density features are used (unlike a generic humor
  "quality" scorer). ops.sent_stats / ops.extract_dates / ops.retrieve_similar are not
  needed for this predicate and are intentionally unused.
- Code carries the PREDICATE: explicit slurs, atrocity/genocide references, explicit
  sexual-violence vocabulary, and lexical co-occurrence patterns for well-known
  exploitative tropes (undressing near a minor with a dismissive punchline, violence
  against a vulnerable/protected target, child-abuse-for-laughs, suicidal-vulnerable-
  person + sexual solicitation).
- Two LLM_FIELDS supply THICK-INPUT grounding for constructs code cannot reliably
  reach from lexicon alone: (a) *which* protected group / trauma type the joke's
  humor depends on demeaning (catches racist "origin myth" jokes, disability-mockery
  via garbled speech, coded slurs, etc. that carry no code-visible keyword), and
  (b) the text's *stance* toward that target (mocking/trivializing vs. self-aware/
  critiquing/neutral), which lets remediation and sensitivity earn credit back rather
  than being flattened to the same penalty as an unqualified slur.
- Final score = 1 - max(code_hazard, llm_hazard) + small remediation bonus, clipped to
  [0, 1]. Using max (not a sum/average) means either channel alone is sufficient to
  flag harm -- recall on harmful content matters more than precision here, since the
  criterion explicitly wants "the criterion, not your taste" to decide, and false
  negatives (missing exploitative content) are worse than a few over-cautious flags.
"""

import re
import math
import statistics
from collections import Counter

LLM_FIELDS = {
    "harm_target": "Name the protected/vulnerable group, trauma type, or atrocity the joke's punchline demeans or trivializes (e.g. child sexual abuse, racial stereotype, disability mockery, domestic violence, genocide), or say NONE.",
    "harm_stance": "In <=20 words: does the text mock/trivialize/endorse harm toward that target, or instead critique/is self-aware/neutral about it, or NONE.",
}

_SLURS = [
    "retard", "retarded", "faggot", "fag", "tranny", "nigger", "nigga",
    "kike", "spic", "chink", "gook", "cripple", "midget", "coon",
]

_GENOCIDE_MARKERS = [
    "mein fuhrer", "mein führer", "final solution", "gas chamber",
    "concentration camp", "holocaust", "zyklon", "auschwitz",
    "ss officer", "ethnic cleansing", "genocide",
]

_SEXUAL_VIOLENCE_WORDS = [
    "rape", "raping", "raped", "molest", "molested",
    "non-consensual", "nonconsensual",
]

_UNDRESS_MARKERS = ["unbuckl", "unbutton", "unzip"]
_MINOR_MARKERS = ["boy", "girl", "kid", "child", "little"]
_DISMISS_MARKERS = [
    "not your day", "not my day", "sorry kid",
    "wait at the bottom", "just not your day",
]

_SUICIDE_CONTEXT = ["jump", "bridge", "cliff", "ledge", "roof"]
_SOLICIT_MARKERS = ["sex", "fuck me", "have sex", "touch"]

_VIOLENCE_VERBS = [
    "kick", "kicked", "kicking", "beat", "beating",
    "punch", "punched", "beaten", "assault",
]
_VULNERABLE_TARGETS = [
    "nun", "homeless", "elderly", "blind", "deaf", "disabled",
    "wheelchair", "baby", "newborn", "infant", "toddler",
]

_CHILD_ABUSE_MARKERS = ["cigarette"]
_CHILD_ABUSE_VICTIMS = ["baby", "newborn", "infant"]

_STEREOTYPE_PHRASES = ["all of them", "those people", "you people", "typical"]

_REMEDIATION_MARKERS = ["content warning", "trigger warning", "cw:"]

_TRIVIALIZE_STANCE = [
    "trivial", "normaliz", "endors", "mock", "demean", "dehuman",
    "punchline", "laugh at", "false equivalen", "glorif",
]
_CRITIQUE_STANCE = [
    "critiqu", "condemn", "self-aware", "self aware", "apolog",
    "sympath", "reflect", "sensitiv", "remediat",
]


def _safe_lower(s):
    try:
        return s.lower()
    except Exception:
        return ""


# HYGIENE PATCH: `_has_any` used a bare `n in hay` substring test over the
# whole document string, so e.g. "girl"/"boy"/"kid" (in _MINOR_MARKERS) fired
# on "girlfriend"/"boyfriend"/"kidding", "touch"/"sex" (in _SOLICIT_MARKERS)
# fired on "touchdown"/"sexist", and "punch" (in _VIOLENCE_VERBS) fired on
# "punchline" -- all unrelated to the intended concepts. Boundary-aware,
# light-inflection-aware matching instead: phrases (space/hyphen inside) match
# verbatim between word boundaries; single words allow a small whitelist of
# regular inflectional suffixes so real inflections ("kicks"/"kicked") still
# match but unrelated host words no longer do.
_PATTERN_CACHE = {}


def _pattern(term):
    cached = _PATTERN_CACHE.get(term)
    if cached is not None:
        return cached
    esc = re.escape(term)
    if " " in term or "-" in term:
        pat = r"\b" + esc + r"\b"
    else:
        pat = r"\b" + esc + r"(?:'s|s|es|ed|ing)?\b"
    compiled = re.compile(pat)
    _PATTERN_CACHE[term] = compiled
    return compiled


def _has_any(hay, needles):
    try:
        return any(_pattern(n).search(hay) for n in needles)
    except Exception:
        return False


def _code_hazard(text):
    """Lexical / co-occurrence hazard detectors -> magnitude in [0, 1]."""
    try:
        tl = _safe_lower(text)
        if not tl.strip():
            return 0.0
        hazards = []

        if _has_any(tl, _SLURS):
            hazards.append(1.0)

        if _has_any(tl, _GENOCIDE_MARKERS):
            hazards.append(0.95)

        if _has_any(tl, _SEXUAL_VIOLENCE_WORDS):
            hazards.append(0.9)

        if _has_any(tl, _UNDRESS_MARKERS) and _has_any(tl, _MINOR_MARKERS):
            hazards.append(0.9)
        if _has_any(tl, _UNDRESS_MARKERS) and _has_any(tl, _DISMISS_MARKERS):
            hazards.append(0.85)

        if _has_any(tl, _SUICIDE_CONTEXT) and _has_any(tl, _SOLICIT_MARKERS):
            hazards.append(0.75)

        if _has_any(tl, _VIOLENCE_VERBS) and _has_any(tl, _VULNERABLE_TARGETS):
            hazards.append(0.8)

        if _has_any(tl, _CHILD_ABUSE_MARKERS) and _has_any(tl, _CHILD_ABUSE_VICTIMS):
            hazards.append(0.9)

        if _has_any(tl, _STEREOTYPE_PHRASES):
            hazards.append(0.4)

        if not hazards:
            return 0.0
        base = max(hazards)
        # mild compounding bump when multiple independent detectors fire
        extra = 0.0
        if len(hazards) >= 2:
            extra = min(0.15, 0.05 * (len(hazards) - 1))
        return max(0.0, min(1.0, base + extra))
    except Exception:
        return 0.0


def _llm_hazard(extracted):
    """LLM-grounded hazard + remediation bonus, from harm_target / harm_stance."""
    try:
        if not isinstance(extracted, dict):
            return 0.0, 0.0
        target = (extracted.get("harm_target") or "").strip()
        stance = (extracted.get("harm_stance") or "").strip()
        if not target or target.upper() in ("NONE", "N/A"):
            return 0.0, 0.0
        stance_l = stance.lower()
        if not stance_l:
            return 0.7, 0.0
        if _has_any(stance_l, _TRIVIALIZE_STANCE):
            return 0.85, 0.0
        if _has_any(stance_l, _CRITIQUE_STANCE):
            return 0.3, 0.05
        # target flagged but stance ambiguous -- moderate, non-zero penalty
        return 0.55, 0.0
    except Exception:
        return 0.0, 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5

        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        if not isinstance(t, str) or not t.strip():
            t = text

        code_pen = _code_hazard(t)
        llm_pen, bonus = _llm_hazard(extracted)

        penalty = max(code_pen, llm_pen)

        tl = _safe_lower(t)
        if _has_any(tl, _REMEDIATION_MARKERS):
            bonus += 0.05

        raw = 1.0 - penalty + bonus
        if raw < 0.0:
            raw = 0.0
        if raw > 1.0:
            raw = 1.0
        return raw
    except Exception:
        return 0.5

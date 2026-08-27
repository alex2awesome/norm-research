"""a0 hybrid: Protected class membership (McDonnell Douglas prong 1).

Criterion asks whether the plaintiff plausibly belongs to a Title VII
protected class implicated by the claim: race, color, religion, sex
(incl. pregnancy, and post-Bostock sexual orientation/gender identity), or
national origin. Age (ADEA) and disability (ADA) are explicitly NOT Title
VII protected classes, even though they are common co-alleged claims in
this corpus.

Baseline (v0_keyword) failure modes seen in train:
  - SUBSTRING BUG: keyword 'race' never fires on 'racial' (r-a-c-i-a-l has
    no 'race' substring), 'religion' never fires on 'religious'. Several
    judge=1.0 documents whose only trait language is "racial discrimination"
    or "religious beliefs" score baseline=0.0 purely from this bug
    (d00667, d00839, d01084).
  - RIGID PHRASING: baseline only matches a fixed list of exact phrases
    ('sex discrimination', 'because of her sex', ...) so paraphrases like
    "discriminates ... based on sex" (d00905) or "sexually harassing her"
    (d00871) score 0 even though the trait is squarely sex.
  - DIVISOR CEILING: baseline needs 4 distinct keyword phrases to reach 1.0;
    real narratives usually name the trait once or twice, so baseline caps
    out at 0.25-0.5 on many judge=1.0 docs (d00532, d00003, d00028, d00382,
    d00861, d00966, d00437) even when the trait is unambiguous.
  - By contrast, baseline correctly scores 0.0 on every pure ADEA/ADA/other
    train doc (no Title VII trait words present at all) -- so the fix is
    to widen recall on the positive side, not to add new negative gating.

Design: fix the surface-pattern bug in code (word-boundary regex over
race/color/religion/sex/pregnancy/orientation/gender-identity/national
origin, bucketed into 4 Title VII categories; age/disability deliberately
excluded from every bucket) and detect causal linkage ("because of",
"based on", "motivated by", ... within a short window of a trait word) as
a code-only quantity-ish signal regex can legitimately see. Two known
corpus-specific false-positive collisions are guarded against explicitly:
'under color of (state) law' (a boilerplate Sec. 1983 phrase, not the
Title VII 'color' trait) and 'race to the courthouse' (limitations-period
idiom). The part regex genuinely cannot do -- confirming the trait belongs
to the PLAINTIFF (not a comparator/supervisor/co-worker) and is actually
asserted as the basis for the claim rather than only age/disability/
retaliation/other -- is routed to one LLM field. Code owns the predicate:
it maps the LLM's short answer to NONE/trait and blends it with the
regex/causal code score rather than trusting the LLM's own yes/no.
"""
import re

LLM_FIELDS = {
    "protected_trait": (
        "Name the single Title VII protected trait (race, color, religion, "
        "sex, pregnancy, sexual orientation, gender identity, or national "
        "origin) that the PLAINTIFF -- not a coworker, comparator, or "
        "supervisor -- alleges was the basis for the challenged treatment; "
        "answer NONE if the claim rests only on age, disability, "
        "retaliation, or other non-Title-VII grounds."
    ),
}

# --- Title VII trait categories, word-boundary regex (fixes baseline's
# substring bug: 'race' now matches 'racial', 'religion' matches
# 'religious'). Age/disability are intentionally never members of any
# bucket here. ---
_RACE = re.compile(r"\brac(?:e|es|ial|ially)\b(?!\s+to\s+(?:the\s+)?courthouse)")
_COLOR = re.compile(r"\bcolou?r(?:s)?\b(?!\s+of\s+(?:state\s+)?law)")
_RELIGION = re.compile(r"\breligio(?:n|ns|us|usly)\b")
_SEX = re.compile(r"\bsex(?:es|ual|ually)?\b")
_GENDER = re.compile(r"\bgender(?:s)?\b")
_PREGNANT = re.compile(r"\bpregnan(?:t|cy|cies)\b")
_ORIENTATION = re.compile(r"\bsexual orientation\b")
_GENDER_IDENTITY = re.compile(r"\bgender identity\b|\btransgender\b")
_NATL_ORIGIN = re.compile(r"\bnational origin\b|\bnationality\b|\bethnic(?:ity)?\b|\bancestry\b")

_CATEGORIES = {
    "race_color": (_RACE, _COLOR),
    "religion": (_RELIGION,),
    "sex": (_SEX, _GENDER, _PREGNANT, _ORIENTATION, _GENDER_IDENTITY),
    "national_origin": (_NATL_ORIGIN,),
}

# --- causal-linkage cue: a code-legible "is the trait the alleged BASIS"
# signal -- look for a cue phrase followed shortly by a category hit. ---
_CAUSAL_CUE = re.compile(r"\b(?:because of|on the basis of|based on|motivated by|due to)\b")
_CAUSAL_WINDOW = 60

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _count_categories(t):
    return sum(1 for pats in _CATEGORIES.values() if any(p.search(t) for p in pats))


def _has_causal_link(t):
    for m in _CAUSAL_CUE.finditer(t):
        window = t[m.end(): m.end() + _CAUSAL_WINDOW]
        if _count_categories(window) > 0:
            return True
    return False


def _code_score(t):
    n_cat = _count_categories(t)
    if n_cat <= 0:
        base = 0.0
    elif n_cat == 1:
        base = 0.5
    else:
        base = 0.8
    if base > 0.0 and _has_causal_link(t):
        base = min(1.0, base + 0.2)
    return base


def _llm_score(extracted):
    """None -> field missing/unavailable (no evidence either way); 0.0/1.0
    -> field present and decisive."""
    if not isinstance(extracted, dict):
        return None
    raw = extracted.get("protected_trait", None)
    if raw is None:
        return None
    val = str(raw).strip()
    if val.lower().strip(". ") in _NONE_VALUES:
        return 0.0
    return 1.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""
        t_lower = t.lower()

        code = _code_score(t_lower)
        llm = _llm_score(extracted)

        if llm is None:
            final = code
        else:
            final = 0.35 * code + 0.65 * llm

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5

# Hybrid module (4-field) for legal_title_vii aspect a5: "replacement outside
# class" (plaintiff replaced by, or the contested job/promotion went to,
# someone outside the protected class -- a classic McDonnell Douglas
# inference-of-discrimination circumstance). "A less-qualified out-class
# replacement is a particularly strong plaintiff signal; an in-class
# replacement [undermines it]." (criterion text).
#
# Baseline diagnosis: the regex baseline only fires on a narrow set of
# surface phrasings -- "replaced X WITH Y", "succeeded by", "position was
# filled/given BY/TO", "hired/promoted/selected INSTEAD/in place" -- and one
# literal trait-contrast phrase ("outside the protected class"). Train
# examples show it silently misses very common variants: "replaced BY a
# white male" (no "with"), "filled the role WITH a less qualified white
# employee" (no "by/to"), "a white male was hired as interim Chief" with no
# trigger adverb nearby, categorical hiring-policy exclusion with no named
# individual at all (Pan Am's females-only cabin-attendant policy), and it
# has NO way to tell whether the replacement's trait actually differs from
# the plaintiff's -- it also false-fires on an explicit NEGATION ("she was
# NOT replaced by a person OUTSIDE her protected class") because its
# trait-contrast regex just checks for the phrase "outside ... protected
# class" regardless of the "not" governing it. Conversely, several
# judge=0 examples mention a specific rival/competitor for a job with NO
# stated trait at all ("preferred a subsequently-hired cashier over him") --
# a case where the EVENT is real but there is no protected-class contrast to
# infer from, so score should stay low.
#
# Design (original 2 LLM fields): keep the baseline's mechanical
# event-detection idea (widened to catch the missed surface forms, plus a
# categorical-policy-exclusion pattern, plus a code-level negation guard) as
# a modest code-only component. ADD two LLM fields that carry the THICK
# judgment code cannot reach on its own: the plaintiff's own protected-class
# trait, and the trait of whoever filled the position/got the job. The
# PREDICATE -- deciding whether those two traits actually differ (out-class),
# overlap entirely (in-class, which the corpus notes says undermines the
# inference), or one side is simply unknown -- is done in code by comparing
# normalized trait-tag sets, never by the LLM.
#
# GAP the original 2 fields plausibly miss (construct grounds only -- no
# judge/eval signal consulted): the criterion text itself singles out
# QUALIFICATION DIRECTION as the thing that makes an out-class replacement
# "particularly strong" evidence vs. merely present. The existing code's
# `_QUAL_RE` regex only detects that qualification LANGUAGE appears
# SOMEWHERE in the text -- it cannot tell WHO the text says was more
# qualified (the plaintiff, or the person who got the job), which is exactly
# the doctrinal-construct judgment the corpus notes warn regex cannot make
# ("legal TERM presence is a weak proxy for the construct being FACTUALLY
# PRESENT"). Second, the criterion is about an INFERENCE of discrimination,
# and a single anecdotal replacement is weaker evidence than a described
# PATTERN of out-class hires/promotions for the same role/employer -- pattern
# vs. single-instance is also a thick read no regex here attempts. New
# fields `qual_direction` and `pattern_count` probe those two gaps directly.
# Both gate on an already-established replacement event (they are no-ops if
# no replacement/selectee trait was found) and both degrade gracefully: if
# absent from `extracted` (not yet extracted), score() reduces exactly to
# the original 2-field formula.

import re
import math

LLM_FIELDS = {
    "plaintiff_trait": (
        "In <=8 words, the plaintiff's own protected-class trait relevant "
        "to this claim (race/sex/age/national origin/religion/disability), "
        "or NONE if not stated."
    ),
    "replacement_trait": (
        "In <=8 words, the protected-class trait of whoever filled the "
        "plaintiff's job or got the specific contested promotion/hire "
        "(the replacement/selectee), or NONE if no such person is named."
    ),
    "qual_direction": (
        "In <=8 words, who does the text say was more qualified for the "
        "job: the plaintiff, the replacement/selectee, or comparable/unclear?"
    ),
    "pattern_count": (
        "In <=8 words: how many similar instances of an out-class person "
        "getting the job/position are described -- one, several/pattern, or none?"
    ),
}


def _sat(x, k):
    return 1.0 - math.exp(-x / max(1e-6, k))


_EVENT_RES = [
    re.compile(r"replaced\s+(?:her|him|them|it)?\s*(?:with|by)\b", re.I),
    re.compile(r"\bsucceeded by\b", re.I),
    re.compile(r"position\s+(?:was\s+|is\s+)?(?:filled|given|awarded)\s+(?:by|to)\b", re.I),
    re.compile(r"(?:filled|awarded)\s+the\s+(?:position|role|job|promotion)\s+(?:to|with|by)\b", re.I),
    re.compile(r"(?:hired|promoted|selected|appointed|chosen)\s+"
               r"(?:instead(?:\s+of)?|over|in\s+(?:her|his|their)\s+place|rather than)\b", re.I),
    re.compile(r"\bin\s+(?:her|his|their)\s+place\b", re.I),
]

_POLICY_RE = re.compile(
    r"only\s+hir(?:e|es|ed|ing)\s+(?:females|males|women|men)|"
    r"bona fide occupational qualification|"
    r"polic(?:y|ies)\s+of\s+(?:only\s+)?hiring", re.I)

_QUAL_RE = re.compile(
    r"(?:less|more|most|better|comparable|equally)\s+qualified|"
    r"lacked (?:the )?(?:experience|qualifications)|no prior experience|"
    r"less experienced|more experienced|plainly superior", re.I)

_OUTCLASS_RE = re.compile(
    r"outside\s+(?:her|his|their|the)\s+protected class", re.I)

_NEGATION_RE = re.compile(
    r"\bnot\b[^.]{0,40}\breplaced\b|\bnever\b[^.]{0,40}\breplaced\b|"
    r"\bnot\b[^.]{0,60}\boutside\b[^.]{0,20}protected class", re.I)


def _code_signal(t):
    negated = bool(_NEGATION_RE.search(t))

    event_hits = sum(len(r.findall(t)) for r in _EVENT_RES)
    policy_hits = len(_POLICY_RE.findall(t))
    qual_hits = len(_QUAL_RE.findall(t))
    outclass_hits = 0 if negated else len(_OUTCLASS_RE.findall(t))

    s = (0.30 * _sat(event_hits, 1.0)
         + 0.15 * _sat(policy_hits, 1.0)
         + 0.15 * _sat(qual_hits, 1.0)
         + 0.15 * _sat(outclass_hits, 1.0))
    return max(0.0, min(0.75, s)), negated


_NONE_WORDS = {"", "none", "n/a", "na", "unknown", "unspecified", "not stated", "unclear"}

# canonical trait keys the LLM's free-text answer gets folded into before
# comparing plaintiff vs. replacement -- keeps the comparison robust to
# synonyms ("African American" vs "black", "Caucasian" vs "white", ...)
# without hardcoding every nationality/religion (those fall through as
# raw leftover content words, which is enough to tell e.g. "Nigerian" from
# "Iranian" apart).
_RACE_MAP = {
    "african american": "black", "black": "black",
    "caucasian": "white", "white": "white",
    "asian american": "asian", "asian": "asian",
    "latino": "hispanic", "latina": "hispanic", "latinx": "hispanic", "hispanic": "hispanic",
    "native american": "native-american",
}
_STOPWORDS = {
    "employee", "employees", "worker", "workers", "supervisor", "manager",
    "person", "position", "job", "born", "old", "years", "year", "none",
    "unknown", "plaintiff", "defendant", "staff", "colleague", "colleagues",
    "coworker", "coworkers", "applicant", "applicants", "candidate",
    "candidates", "individual", "replace", "replaced", "replacement",
    "selectee", "hire", "hired", "promoted", "same", "class", "protected",
}


def _norm_trait(s):
    s = (s or "").strip().lower()
    if s in _NONE_WORDS:
        return set()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    tags = set()
    for k, v in _RACE_MAP.items():
        if k in s:
            tags.add(v)
    if re.search(r"\b(female|woman|women)\b", s):
        tags.add("female")
    if re.search(r"\b(male|man|men)\b", s):
        tags.add("male")
    if re.search(r"disab", s):
        tags.add("disability")
    for n in re.findall(r"\b(\d{2,3})\b", s):
        tags.add("age:" + n)
    if re.search(r"\bolder\b", s):
        tags.add("age:older")
    if re.search(r"\byounger\b", s):
        tags.add("age:younger")

    leftover = s
    for k in list(_RACE_MAP.keys()) + ["female", "woman", "women", "male", "man", "men",
                                        "disability", "disabled", "handicap",
                                        "older", "younger"]:
        leftover = leftover.replace(k, " ")
    for w in leftover.split():
        if len(w) > 3 and w not in _STOPWORDS and not w.isdigit():
            tags.add(w)
    return tags


def _trait_signal(extracted, negated):
    extracted = extracted or {}
    p_tags = _norm_trait(extracted.get("plaintiff_trait", ""))
    r_tags = _norm_trait(extracted.get("replacement_trait", ""))
    if negated:
        r_tags = set()  # trust the mechanical negation guard over a possibly-confused extractor

    if not r_tags:
        return 0.0
    if not p_tags:
        return 0.5  # a replacement/selectee's trait is named but plaintiff's own trait unclear
    extra = r_tags - p_tags
    if extra:
        return 1.0  # differs on at least one protected axis -> out-class confirmed
    return -0.5  # fully overlapping traits -> in-class replacement, weakens the inference


# --- new-field parsing: qualification direction & pattern breadth ----------

_QUAL_NONE_WORDS = {
    "", "none", "unclear", "unknown", "comparable", "similar", "n/a", "na",
    "not stated", "same", "equal", "equally qualified",
}


def _qual_direction_signal(extracted):
    """Who was more qualified, per the text? Refines the raw _QUAL_RE
    keyword hit (which cannot tell direction) with an actual comparison.
    """
    extracted = extracted or {}
    v = str(extracted.get("qual_direction", "") or "").strip().lower()
    if not v or v in _QUAL_NONE_WORDS:
        return 0.0
    if "plaintiff" in v:
        # plaintiff was more qualified than the out-class replacement/selectee
        # -> "particularly strong plaintiff signal" per the criterion text
        return 0.25
    if "replacement" in v or "selectee" in v or "hire" in v or "hired" in v:
        # the out-class replacement/selectee was more qualified -> a
        # legitimate-reason explanation is available, mildly undermining
        # the inference (but the trait-contrast signal itself still stands)
        return -0.12
    return 0.0


_PATTERN_NONE_WORDS = {
    "", "none", "unclear", "unknown", "n/a", "na", "one", "single",
    "just one", "no pattern", "isolated",
}
_PATTERN_MANY_RE = re.compile(
    r"\b(several|multiple|many|pattern|repeat\w*|numerous|two|three|four|"
    r"five|consistently|routinely|history of)\b", re.I,
)


def _pattern_signal(extracted):
    """A described PATTERN of out-class selections is stronger inference
    evidence than a single anecdote; the original fields only establish a
    single event.
    """
    extracted = extracted or {}
    v = str(extracted.get("pattern_count", "") or "").strip().lower()
    if not v or v in _PATTERN_NONE_WORDS:
        return 0.0
    if _PATTERN_MANY_RE.search(v):
        return 0.15
    return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        code_score, negated = _code_signal(t)
        trait_sig = _trait_signal(extracted, negated)

        s = code_score + 0.55 * trait_sig

        # --- NEW: qualification direction and pattern breadth. Both are
        # only meaningful once a replacement/selectee has actually been
        # established (trait_sig != 0.0 means _trait_signal found r_tags);
        # both are no-ops (0.0) if the field is absent from `extracted`
        # (not yet extracted) or the extractor answered a none/unclear
        # token, so score() degrades gracefully to the original formula.
        if trait_sig != 0.0:
            if "qual_direction" in extracted:
                s += _qual_direction_signal(extracted)
            if "pattern_count" in extracted:
                s += _pattern_signal(extracted)

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

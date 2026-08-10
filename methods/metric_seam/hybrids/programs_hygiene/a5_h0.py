# Hybrid module for legal_title_vii aspect a5: "replacement outside class"
# (plaintiff replaced by, or the contested job/promotion went to, someone
# outside the protected class -- a classic McDonnell Douglas inference-of-
# discrimination circumstance).
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
# Design: keep the baseline's mechanical event-detection idea (widened to
# catch the missed surface forms, plus a categorical-policy-exclusion
# pattern, plus a code-level negation guard) as a modest code-only
# component. ADD two LLM fields that carry the THICK judgment code cannot
# reach on its own: the plaintiff's own protected-class trait, and the
# trait of whoever filled the position/got the job. The PREDICATE --
# deciding whether those two traits actually differ (out-class), overlap
# entirely (in-class, which the corpus notes says undermines the
# inference), or one side is simply unknown -- is done in code by comparing
# normalized trait-tag sets, never by the LLM.

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
        if re.search(r"\b" + re.escape(k) + r"\b", s):
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
    # NOTE (hygiene patch): "man"/"men"/"male" must NOT be removed with a
    # bare str.replace() -- that corrupts any unrelated word that happens to
    # contain those letters contiguously (e.g. "employment"/"department"/
    # "harassment" each contain "men"; "management"/"manager"/"manner"/
    # "permanent" each contain "man"), leaving leftover garbage fragments
    # ("employ", "depart", "harass", ...) that get added as spurious trait
    # tags below and can corrupt the plaintiff-vs-replacement trait
    # comparison (verified: "black female in employment department" ->
    # pre-patch tags included bogus "employ"/"depart"). Boundary-anchor
    # just these three words; leave the other removals (race terms,
    # female/woman/women, disability/older/younger) as-is since they were
    # not flagged and are not implicated in this collision.
    leftover = re.sub(r"\b(?:man|men|male)\b", " ", leftover)
    for k in list(_RACE_MAP.keys()) + ["female", "woman", "women",
                                        "disability", "disabled", "handicap",
                                        "older", "younger"]:
        leftover = re.sub(r"\b" + re.escape(k) + r"\b", " ", leftover)
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
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

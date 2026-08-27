"""a0 RECODE: Protected class membership (McDonnell Douglas prong 1).

Goal (seam recode task): h0 was FIELD-DOMINATED (0.35 code / 0.65 llm blend,
where the single LLM field `protected_trait` alone almost perfectly separates
judge=0 from judge=1 on train). This candidate pushes the classification
PREDICATE itself into code so code alone carries most of the separating
power, and uses the LLM field only as a light corrective nudge.

Diagnosis (from inspecting h0's residuals on the 148 train items):
  h0's _code_score already fixes the v0_keyword substring bug, but is too
  COARSE (only 5 distinct values: 0/.5/.7/.8/1.0) and, worse, is not
  ATTRIBUTION-aware: it fires on any Title-VII trait word anywhere in the
  document, so it produces false positives whenever a trait word appears
  but is NOT actually pled as the plaintiff's own basis for the claim:
    - regulatory/administrative docs that use trait words as data-collection
      categories, not discrimination allegations (d00025: EEO-1 pay-data
      reporting rule lists "sex, race, ethnicity" as report fields)
    - trait mentioned about a THIRD PARTY / comparator, not the plaintiff
      (d00872: "reported incidents of racial discrimination against OTHER
      employees"; d00431: "an ethnic slur from one of his associates")
    - trait word used as a descriptor of subject-matter/content, not a
      discrimination basis (d00000: "sexually explicit material" in a First
      Amendment internet-access-restriction case; d00893: "trafficked for
      sex" in a hotel human-trafficking liability suit; d01040: student-on-
      student Title IX sexual-assault dispute, no employer discrimination)
    - the named "Plaintiff" is actually a CORPORATION (d00369: "Plaintiff
      Stroehmann Bakeries" challenging a labor-arbitration award) -- a
      corporate plaintiff cannot satisfy McDonnell Douglas prong 1 at all.

Redesign: code now requires ATTRIBUTION evidence -- a trait-category hit
only counts as "strong" (plaintiff-linked) when, in a local window, it
co-occurs with a possessive pronoun ("his/her/their/my TRAIT"), a causal
cue ("because of", "based on", "motivated by", ...), or discrimination-
predicate language ("discriminat*", "harass*", "hostile work environment",
"protected class") -- AND is NOT immediately adjacent to a comparator/
third-party marker ("other employees", "coworkers", "his associates", ...).
A "corporate plaintiff" guard and a document-level discrimination-context
soft-check further suppress the collision docs above. Score is then a
continuous function of how many DISTINCT trait categories clear the strong
bar (graduated, not saturating at a handful of buckets), which also gives
finer-grained ranking within the large judge=1.0 block instead of h0's
heavy ties.

The LLM field (reused from h0, same name/prompt so it's populated during
iteration) is now used ASYMMETRICALLY: when code is confident (strong hit,
or a hard gate fired), code dominates the blend; only when code lands in
the ambiguous "weak evidence only" zone does the LLM get more weight. No
new LLM_FIELDS declared.
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

# --- Title VII trait categories, word-boundary regex (fixes v0's substring
# bug: 'race' now matches 'racial', 'religion' matches 'religious'). Age and
# disability are intentionally never members of any bucket. ---
_RACE = re.compile(r"\brac(?:e|es|ial|ially)\b(?!\s+to\s+(?:the\s+)?courthouse)")
_COLOR = re.compile(r"\bcolou?r(?:s)?\b(?!\s+of\s+(?:state\s+)?law)")
_RELIGION = re.compile(r"\breligio(?:n|ns|us|usly)\b")
_SEX = re.compile(r"\bsex(?:es|ual|ually)?\b")
_GENDER = re.compile(r"\bgender(?:s)?\b")
_PREGNANT = re.compile(r"\bpregnan(?:t|cy|cies)\b")
_ORIENTATION = re.compile(r"\bsexual orientation\b")
_GENDER_IDENTITY = re.compile(r"\bgender identity\b|\btransgender\b")
_NATL_ORIGIN = re.compile(r"\bnational origin\b|\bnationality\b|\bethnic(?:ity)?\b|\bancestry\b")

# Identity-NAME words (not just the abstract category word). Docs often
# self-identify the plaintiff ("a Black female", "an African-American male")
# without ever using the literal word "race"; missing these was a real
# recall gap. Deliberately excludes "black"/"white" (common surnames/other
# words -- e.g. "White House", "Mr. White" -- too collision-prone once
# lowercased) and "christian" (common first name). Kept list is low-risk.
_RACE_NAME = re.compile(
    r"\bafrican[- ]americans?\b|\bcaucasians?\b|\bhispanics?\b|\blatinos?\b|"
    r"\blatinas?\b|\bnative american\b")
_RELIGION_NAME = re.compile(
    r"\bmuslims?\b|\bjewish\b|\bcatholics?\b|\bbuddhists?\b|\bhindus?\b|"
    r"\bmormons?\b")

_CATEGORIES = {
    "race_color": (_RACE, _COLOR, _RACE_NAME),
    "religion": (_RELIGION, _RELIGION_NAME),
    "sex": (_SEX, _GENDER, _PREGNANT, _ORIENTATION, _GENDER_IDENTITY),
    "national_origin": (_NATL_ORIGIN,),
}

# --- attribution evidence: does this trait hit actually belong to the
# plaintiff as the claimed BASIS, vs. just appearing in the document? ---
_POSSESSIVE_BEFORE = re.compile(r"\b(?:his|her|their|my)\s+$")
# Deliberately NARROW to discrimination-specific causal phrasing. Generic
# connectives ("due to", "as a result of", "related to") were tried and
# dropped -- they fire on ANY nearby causal statement (e.g. "censored as a
# result of the Act"), not specifically a trait-caused-mistreatment claim.
_CAUSAL_CUE = re.compile(
    r"\b(?:because of|on the (?:basis|ground)s? of|based on|motivated by|"
    r"on account of)\b")
_DISCRIM_ROOT = re.compile(r"discriminat|harass|hostile\s+work\s+environment|"
                            r"hostile\s+environment|protected\s+class|"
                            r"protected\s+characteristic")
_WINDOW = 70

# comparator / third-party markers that, near a trait hit, mean the trait
# belongs to someone OTHER than the plaintiff (or is a report about others)
_COMPARATOR = re.compile(
    r"\bother employees\b|\bfellow employees?\b|\bco-?workers?\b|"
    r"\bcolleagues?\b|\bcomparators?\b|\bsimilarly situated\b|"
    r"\bagainst other\b|\bof other\b|\bhis associates\b|\bher associates\b|"
    r"\btheir associates\b|\banother employee\b|\bthird party\b")

# doc-level soft context check: real Title VII discrimination narratives
# almost always surface at least one of these somewhere.
_CTX = re.compile(
    r"discriminat|harass|title vii|eeoc|protected class|retaliat|"
    r"disparate|adverse (?:employment )?action|hostile|prima facie|"
    r"civil rights act|equal employment", re.I)

# corporate-plaintiff guard: a company cannot itself belong to a Title VII
# protected class, so if "Plaintiff" names a corporate entity, prong 1 is
# moot regardless of trait words elsewhere in the document.
_CORP_PLAINTIFF = re.compile(
    r"\b[Pp]laintiffs?,?\s+([A-Z][\w&.'-]*(?:\s+(?:and|&|of|the)?\s*[A-Z][\w&.'-]*){0,5})")
_CORP_SUFFIX = re.compile(
    r"\b(inc\.?|llc|corp\.?|co\.?|company|corporation|bakeries|stores|"
    r"industries|enterprises|services|systems|group|partners|bank|"
    r"hospital|university|airlines|railroad|railway|association|ltd\.?|"
    r"l\.p\.)\b", re.I)

_SELF_ID = re.compile(
    r"\b(?:a|an)\s+(?:\d{1,3}[- ]year[- ]old\s+)?"
    r"(?:african[- ]american|black|white|caucasian|hispanic|latino|latina|"
    r"asian|native american|arab|jewish|muslim|christian|catholic)"
    r"\b[\s,-]*(?:wo?man|male|female)?", re.I)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _has_comparator_nearby(t, start, end):
    # Fixed-width window only. A wider (multi-sentence) comparator scan was
    # tried and dropped: legal narrative sentences run 100-300+ chars, so
    # "current + previous sentence" pulled in unrelated comparator words
    # from a different clause (e.g. a co-worker's discipline mentioned two
    # sentences earlier) and wrongly suppressed genuinely strong evidence
    # elsewhere in the same sentence (cost more true positives than it
    # rescued false positives).
    lo, hi = max(0, start - _WINDOW), min(len(t), end + _WINDOW)
    return bool(_COMPARATOR.search(t[lo:hi]))


def _is_strong(t, m):
    """A category regex match is STRONG plaintiff-attributed evidence if it
    co-occurs with a possessive pronoun, a causal cue, or discrimination-
    predicate language nearby, and is not shadowed by a comparator marker."""
    start, end = m.start(), m.end()
    if _has_comparator_nearby(t, start, end):
        return False
    if _POSSESSIVE_BEFORE.search(t[max(0, start - 15):start]):
        return True
    lo, hi = max(0, start - _WINDOW), min(len(t), end + _WINDOW)
    window = t[lo:hi]
    if _CAUSAL_CUE.search(window):
        return True
    if _DISCRIM_ROOT.search(window):
        return True
    return False


def _corporate_plaintiff(t_orig):
    for m in _CORP_PLAINTIFF.finditer(t_orig):
        if _CORP_SUFFIX.search(m.group(1)):
            return True
    return False


def _code_score(t_lower, t_orig):
    """Returns (score, confident). `confident` marks cases where code has
    actual evidence one way or the other -- a hard negative gate (corporate
    plaintiff) or at least one plaintiff-attributed (strong) trait hit.
    `confident=False` means code found nothing conclusive (no strong hit,
    maybe a bare/weak trait word or nothing at all); the caller should lean
    on the LLM field rather than trust code's low score as a real negative."""
    if _corporate_plaintiff(t_orig):
        return 0.0, True

    n_strong = 0
    n_weak = 0
    n_strong_total_hits = 0
    for pats in _CATEGORIES.values():
        strong_here = False
        weak_here = False
        for p in pats:
            for m in p.finditer(t_lower):
                if _is_strong(t_lower, m):
                    strong_here = True
                    n_strong_total_hits += 1
                else:
                    weak_here = True
        if strong_here:
            n_strong += 1
        elif weak_here:
            n_weak += 1

    if n_strong == 0:
        # no confirmed plaintiff-attributed trait; small hedge if a bare
        # trait word appeared anywhere (could still be relevant, e.g. LLM
        # catches paraphrases code can't window-match), else true zero --
        # either way code is NOT confident here.
        return (0.12 if n_weak > 0 else 0.0), False

    if n_strong == 1:
        base = 0.65
    elif n_strong == 2:
        base = 0.85
    else:
        base = 0.95

    # multiple independent strong mentions across categories (richer
    # factual support, not just one conclusory clause) -> small bump
    if n_strong_total_hits >= 3:
        base = min(1.0, base + 0.05)

    if _SELF_ID.search(t_lower):
        base = min(1.0, base + 0.05)

    if not _CTX.search(t_lower):
        # trait attribution fired but the document never surfaces ANY
        # general discrimination-context language anywhere -- damp rather
        # than zero (rare true positives lack boilerplate framing).
        base *= 0.5

    return max(0.0, min(1.0, base)), True


def _llm_score(extracted):
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

        code, confident = _code_score(t_lower, t)
        llm = _llm_score(extracted)

        if llm is None:
            final = code
        elif confident:
            # code has real evidence either way (a strong plaintiff-
            # attributed trait hit, or the corporate-plaintiff hard gate):
            # trust code, let llm nudge only slightly.
            final = 0.8 * code + 0.2 * llm
        else:
            # code found nothing conclusive (no strong hit -- a bare trait
            # word at best, or nothing at all). Its low score here is NOT
            # confident evidence of absence, so lean on the LLM's semantic
            # read; code's small value still anchors the floor.
            final = 0.25 * code + 0.75 * llm

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5

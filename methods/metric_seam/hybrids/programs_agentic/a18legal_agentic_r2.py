# Hybrid module for legal_title_vii aspect a18: "retaliation materially
# adverse action" -- RECODE PASS (r2). Goal: shift signal from the two
# LLM fields into code. h0 diagnosis (measured directly): ablating the LLM
# fields from h0 drops train rho from 0.80 to 0.56 -- i.e. ~30% of h0's
# discriminative power lives ONLY in the raw binary "both fields fired"
# signal (0.40 weight on a crude present/absent/both llm_signal), which
# blindly trusts field content regardless of what it actually says.
#
# Data-driven redesign (verified against the 138 scoreable TRAIN items,
# not guessed): grouping items by (protected_activity field present,
# post_activity_action field present) gives clean, large-n buckets:
#   (empty,  empty ) n=45 mean=0.001   -- no story at all -> ~0
#   (present,empty ) n=17 mean=0.047   -- activity w/o any action -> ~0
#   (empty,  present) n=17 mean=0.262  -- action w/o named activity -> low/mixed
#   (present,present) n=59 mean=0.700  -- both present, but HUGE variance
# The "both present" bucket is where the real work is. Sub-classifying it
# by what the post_activity_action TEXT actually says (not just whether it
# exists) splits it cleanly:
#   TANGIBLE employment-consequence wording (fired/demoted/suspended/
#     reassigned/denied promotion/probation/negative review/etc.)
#                                    n=50 mean=0.781 median=0.925
#   CONTINUATION wording ("continued disparaging remarks...")
#                                    n=1  mean=0.0   (pre-existing harm
#     persisting is NOT a new retaliatory act -- generalizes: this is the
#     doctrinal distinction between ongoing discrimination and a fresh
#     adverse act taken BECAUSE OF the complaint)
#   pure accommodation-failure wording ("failure to make a reasonable
#     accommodation") with no further downstream consequence -> near 0
#     (it's a failure-to-accommodate claim, a different Title VII theory,
#     not retaliation-adverse-action)
#   everything else (vague, e.g. "assigned burdensome tasks, oppressive
#     environment") n=7 mean=0.321 -- real but much weaker than tangible.
# This TANGIBLE / CONTINUATION / ACCOMMODATION-FAILURE-ONLY / VAGUE
# classifier is implemented in CODE (regex) and is applied uniformly to
# whichever text is available: the LLM field when present, and a
# doc-text-lexicon fallback when it is not (so held-out extraction misses
# degrade gracefully instead of losing the signal entirely). This is the
# core "recode": the classification logic that used to live entirely in
# a flat 0.40-weighted llm_signal now lives in code that reads and grades
# field/document text against empirically-grounded categories.
#
# A second, smaller code-only fix: the activity lexicon's "report(ed)"
# term fires on self-status DISCLOSURE ("Hayes reported her pregnancy to
# the director" -> then fired for the pregnancy itself = discrimination,
# not retaliation) as often as on genuine opposition ("reported the
# harassment"). A disclosure-tail exclusion removes this specific false
# positive without touching the many correctly-firing "reported
# misconduct/harassment/discrimination" matches.
#
# Kept from h0: the code-only temporal-structure signal (causal phrasing +
# ops.extract_dates day-gap arithmetic) and a code-only proximity check
# between activity and action lexicon hits, both as secondary modulators
# on top of the (now much stronger) primary classifier above. The two
# LLM_FIELDS are unchanged (reused, not new) -- they now serve as
# thick-input evidence that CODE grades, rather than a signal code
# blindly trusts.

import re
import math
import datetime

LLM_FIELDS = {
    "protected_activity": (
        "In <=15 words, name the protected activity (internal complaint, "
        "EEOC/agency charge, opposing discriminatory conduct, reporting "
        "misconduct, participating in an investigation) described BEFORE any "
        "adverse action; answer NONE if no such activity is described."
    ),
    "post_activity_action": (
        "In <=15 words, name the specific action taken against the employee "
        "AFTER that protected activity (e.g. fired, demoted, reassigned, "
        "excluded, poor review, contract not renewed); answer NONE if no "
        "post-activity action is described."
    ),
}


def _sat(x, k):
    return 1.0 - math.exp(-x / max(1e-6, k))


_NONE_ANSWERS = {
    "", "none", "n/a", "na", "no", "not stated", "unclear", "unknown",
    "not shown", "not present", "not mentioned", "not applicable",
}


def _is_none(s):
    if not s:
        return True
    s2 = re.sub(r"[.\s]+$", "", s.strip().lower())
    return s2 in _NONE_ANSWERS


_DATE_FORMATS = (
    "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%b. %d, %Y",
    "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
)


def _parse_date(s):
    s = s.strip().strip(",.")
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


# --- activity lexicon (code): who opposed/complained/charged/reported ---
_ACT_VERB_RE = re.compile(
    r"\b(complain\w*|filed a charge|filed an? (?:eeoc )?complaint|report\w*|"
    r"opposed|objected|participat\w* in (?:the|an) investigation|grievance)\b",
    re.I)

# disclosure-of-own-status false positive: "reported {her/his/their} {status}"
# is telling the employer a fact about oneself (pregnancy/disability/religion),
# not opposing/complaining about discriminatory treatment. Firing someone
# right after such a disclosure is DISCRIMINATION (based on the status), not
# RETALIATION (for protected activity) -- a distinct doctrinal theory.
_DISCLOSURE_TAIL_RE = re.compile(
    r"^\s+(?:to\s+\S+(?:\s+\S+){0,3}\s+)?(?:her|his|their|its)?\s*(?:being\s+)?"
    r"(?:pregnan\w*|disabilit\w*|religious\s+(?:belief|conflict|practice)|"
    r"medical\s+condition|diagnos\w*|health\s+condition)",
    re.I)

_STRONG_ACT_RE = re.compile(
    r"\b(EEOC|Equal Employment Opportunity Commission|"
    r"filed\s+(?:a\s+|an\s+)?(?:eeoc\s+|agency\s+|discrimination\s+)?charge|"
    r"opposing\s+discriminat\w*|opposed\s+discriminat\w*|"
    r"participat\w*\s+in\s+(?:the|an)\s+investigation|grievance|"
    r"right[\s-]to[\s-]sue|filed\s+an?\s+(?:eeoc\s+)?complaint|"
    r"believe\w*\s+(?:she|he|they|I)\s+(?:was|were|am)\s+being\s+discriminated\s+against)\b",
    re.I)

# negated activity: "did not make a formal complaint", "never complained",
# "no grievance was filed" describes the ABSENCE of protected activity --
# the bare noun/verb still lexically matches _ACT_VERB_RE, so it must be
# excluded explicitly rather than counted as evidence of activity.
_NEGATION_HEAD_RE = re.compile(
    r"(?:did\s+not|didn.t|does\s+not|doesn.t|never|failed\s+to|without|"
    r"no)\s+(?:\w+\s+){0,3}$", re.I)

# post-hoc court/agency FILING referenced by its procedural role (the
# pleading itself, or a charge described by what it alleges/when it was
# served) rather than as a protected act that PRECEDED and provoked a
# separate adverse action. This is the single largest false-positive
# source for doc-only fallback: nearly every case in this corpus recites
# the plaintiff's own charge/complaint procedurally, almost always
# describing or following the very adverse action being litigated rather
# than preceding a further retaliatory act.
_POSTHOC_FILING_RE = re.compile(
    r"(?:filed\s+(?:a\s+|an\s+|the\s+|his\s+|her\s+|their\s+|this\s+)?"
    r"(?:complaint|charge|grievance|lawsuit|action|suit)\w*|"
    r"\b(?:the|her|his|their)\s+(?:complaint|charge)\b|"
    r"instituted\s+this\s+action|serve\w*\s+(?:the|his|her|their)\s+complaint)"
    r"(?:[^.]{0,90}?)"
    r"(?:alleging|charging|claiming|asserting|"
    r"in\s+(?:the\s+|this\s+)?(?:united\s+states\s+)?(?:district\s+)?court|"
    r"did\s+not\s+serve)", re.I)


def _activity_matches(t):
    """Count activity-lexicon hits, excluding self-status-disclosure hits,
    negated mentions, and post-hoc procedural filing references."""
    n = 0
    for m in _ACT_VERB_RE.finditer(t):
        head = t[max(0, m.start() - 40): m.start()]
        if _NEGATION_HEAD_RE.search(head):
            continue
        if m.group(0).lower().startswith("report"):
            tail = t[m.end(): m.end() + 70]
            if _DISCLOSURE_TAIL_RE.match(tail):
                continue
        if "complaint" in m.group(0).lower() or "charge" in m.group(0).lower():
            span = t[max(0, m.start() - 20): m.end() + 100]
            if _POSTHOC_FILING_RE.search(span):
                continue
        n += 1
    return n


# --- action classification (code): TANGIBLE / CONTINUATION / ACCOMMODATION
#     FAILURE-ONLY / VAGUE, calibrated against train-set means (see header) ---
_TANGIBLE_RE = re.compile(
    r"terminat\w*|fired|discharg\w*|demot\w*|suspend\w*|non-?renew\w*|"
    r"not\s+(?:be\s+)?renew\w*|contract\s+not\s+renewed|probation|"
    r"denied\s+(?:\w+\s+){0,2}promotion\w*|reassign\w*|removed\s+from|"
    r"revoked|reprimand\w*|refus\w*\s+to\s+rehire|not\s+rehired|"
    r"cut\w*\s*(?:her|his|their)?\s*hours|reduc\w*\s*(?:her|his|their)?\s*hours|"
    r"denied\s+(?:\w+\s+){0,2}(?:raise|health insurance)|written\s+warning|"
    r"misbehavior\s+notice|disciplinary\s+action|counseling\s+notice|"
    r"excluded\s+from|relieved\s+of\s+(?:all\s+)?duties|stripped\s+of|"
    r"salary\s+(?:reduced|cut)|lower\w*\s+(?:pay|salary|wage)|"
    r"denied\s+(?:\w+\s+){0,2}access|denied\s+(?:\w+\s+){0,2}transfer|"
    r"forced\s+to\s+sign|severance\s+offer\s+withdrawn|access\s+revoked|"
    r"hypercritical\s+supervision|timesheets\s+altered|withheld\s+gratuities|"
    r"dropped\s+from\s+the\s+apprenticeship|not\s+made\s+a\s+journeyperson|"
    r"non-selection|negative\s+(?:performance\s+)?(?:evaluation|review|rating)|"
    r"performance\s+improvement\s+plan|disqualified\s+from|write-?up|"
    r"increased\s+scrutiny|isolat\w*", re.I)

_CONTINUATION_RE = re.compile(r"\bcontinu(?:ed|ing|es)?\b", re.I)

_ACCOMMOD_FAIL_RE = re.compile(
    r"(?:failure|failed)\s+to\s+(?:make\s+a\s+reasonable\s+)?accommodat\w*|"
    r"denied\s+(?:\w+\s+){0,2}accommodat\w*|"
    r"accommodat\w*\s+(?:\w+\s+){0,2}denied|did\s+not\s+accommodate", re.I)

# Field-corroborated classification is high-confidence: the extractor was
# specifically instructed to find the action taken AFTER the activity, i.e.
# it already did the pre/post-hoc temporal disambiguation for us. Doc-only
# fallback (no field) has NO such disambiguation available: bare lexicon
# proximity cannot tell "filed a charge, THEN was fired" (retaliation) apart
# from "was fired; plaintiff's complaint, filed on Oct 14, 1999, charges
# unlawful termination" (a post-hoc EEOC/court filing challenging the very
# same termination -- structurally present in nearly every case in this
# corpus, since exhausting administrative remedies is a filing PREREQUISITE
# for suit, not a prior protected act). Verified directly: an earlier
# version of this module that gave doc-only "tangible" matches the same
# 0.82 base as field-corroborated ones collapsed train rho from 0.80 to
# 0.57 by scoring exactly this "filed a complaint ... terminated" doc-wide
# co-occurrence pattern at 0.9+ on ~15 judge=0.0 items. Doc-only fallback is
# therefore capped much lower, mirroring the dampened weight h0 itself puts
# on its own code-lexicon signal (0.40 of the total, vs 0.40 on llm_signal).
_ACTION_CLASS_BASE = {
    "disqualified": 0.08,
    "weak_omission": 0.16,
    "vague": 0.38,
    "tangible": 0.82,
    "absent": None,
}
_ACTION_CLASS_BASE_DOC_FALLBACK = {
    "disqualified": 0.05,
    "weak_omission": 0.08,
    "vague": 0.14,
    "tangible": 0.20,
    "absent": None,
}


def _classify_action(field_text, doc_text):
    """Returns (action_class, used_field: bool)."""
    if not _is_none(field_text):
        if _CONTINUATION_RE.search(field_text):
            return "disqualified", True
        if _TANGIBLE_RE.search(field_text):
            return "tangible", True
        if _ACCOMMOD_FAIL_RE.search(field_text):
            return "weak_omission", True
        return "vague", True
    # doc-text fallback (no field extracted): require an explicit lexicon
    # hit -- doc-wide fuzzy "vague" matching is too noisy to be reliable.
    if _TANGIBLE_RE.search(doc_text):
        return "tangible", False
    if _ACCOMMOD_FAIL_RE.search(doc_text):
        return "weak_omission", False
    return "absent", False


# --- proximity + temporal structure (code-only secondary modulators) ---
def _proximity_signal(t):
    prox = 0
    total = 0
    for m in _ACT_VERB_RE.finditer(t):
        total += 1
        window = t[m.end(): m.end() + 450]
        if _TANGIBLE_RE.search(window):
            prox += 1
    if total == 0:
        return 0.0
    return _sat(prox, 1.0)


_CAUSAL_PHRASE_RE = re.compile(
    r"\b(same day|the next day|the following day|shortly (?:after|thereafter)|"
    r"immediately after|less than (?:a|one|two|three|\d+) \w+ later|"
    r"within (?:one|a|two|1|2|three|3) (?:day|days|week|weeks))\b", re.I)


def _temporal_signal(t, ops):
    phrase_hits = 0
    for m in _CAUSAL_PHRASE_RE.finditer(t):
        window = t[max(0, m.start() - 150): m.end() + 150]
        if _TANGIBLE_RE.search(window):
            phrase_hits += 1
    phrase_score = _sat(phrase_hits, 1.0)

    gap_score = 0.0
    try:
        dates = ops.extract_dates(t)
    except Exception:
        dates = []
    parsed = sorted(d for d in (_parse_date(x) for x in (dates or [])) if d is not None)
    if len(parsed) >= 2:
        gaps = [(parsed[i + 1] - parsed[i]).days for i in range(len(parsed) - 1)]
        gaps = [g for g in gaps if g >= 0]
        if gaps:
            min_gap = min(gaps)
            if min_gap <= 14:
                gap_score = 1.0
            elif min_gap <= 30:
                gap_score = 0.5

    return max(0.0, min(1.0, 0.5 * phrase_score + 0.5 * gap_score))


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        extracted = extracted or {}
        act_field = str(extracted.get("protected_activity", "") or "")
        post_field = str(extracted.get("post_activity_action", "") or "")

        doc_act_ct = _activity_matches(t)
        activity_present = doc_act_ct > 0 or not _is_none(act_field)
        # When the field is present, trust ITS wording for strength (the
        # extractor already picked the single relevant activity out of the
        # whole document); a doc-wide search would also catch unrelated
        # EEOC/charge boilerplate elsewhere in the narrative (procedural
        # history, exhaustion recitals) that has nothing to do with the
        # specific activity named. Only fall back to doc-wide search when
        # there is no field to trust.
        if not _is_none(act_field):
            activity_strong = bool(_STRONG_ACT_RE.search(act_field))
        else:
            activity_strong = bool(_STRONG_ACT_RE.search(t))

        if not activity_present:
            # No evidence anywhere of a protected activity: adverse-action
            # language alone (however much) must not drive the score up --
            # that was h0's baseline-inherited false-positive mode. Give a
            # small amount of background credit only.
            bg = _sat(len(_TANGIBLE_RE.findall(t)), 4.0)
            return max(0.0, min(1.0, 0.06 * bg))

        action_class, used_field = _classify_action(post_field, t)
        if action_class == "absent":
            prox = _proximity_signal(t)
            return max(0.0, min(1.0, 0.04 + 0.04 * prox))

        base_table = _ACTION_CLASS_BASE if used_field else _ACTION_CLASS_BASE_DOC_FALLBACK
        base = base_table[action_class]

        # activity evidence is likewise more trustworthy when it came from
        # the field (LLM already read the whole doc) than from bare doc
        # lexicon -- same pre/post-hoc ambiguity as the action side, e.g.
        # "filed a complaint" usually names the LAWSUIT PLEADING itself (a
        # procedural act that necessarily happens AFTER the adverse action
        # being challenged, not a prior protected act), and "complaining
        # about the alleged discrimination" can refer to an entirely
        # separate, earlier lawsuit. When the action base came from a
        # trusted FIELD but the activity is doc-lexicon-only, shrink the
        # action base toward a low floor rather than trusting it at face
        # value (empirically, activity-absent-from-field + action-present
        # is a low-scoring bucket, mean 0.26 vs 0.70 when both fields
        # corroborate).
        activity_field_present = not _is_none(act_field)
        if used_field and not activity_field_present:
            base = 0.15 + 0.35 * base

        if activity_strong and activity_field_present:
            base += 0.08
        elif activity_strong:
            base += 0.03

        prox = _proximity_signal(t)
        temporal = _temporal_signal(t, ops)
        weight = 0.06 if used_field else 0.10
        base += weight * prox + 0.04 * temporal

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

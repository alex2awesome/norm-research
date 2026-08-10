# Hybrid module for legal_title_vii aspect a18: "retaliation materially
# adverse action" (Burlington Northern v. White). The retaliation adverse-
# action standard is BROADER than the discrimination one: it covers any
# action that would dissuade a reasonable worker from complaining, not just
# ultimate employment actions, PROVIDED it followed a protected activity
# (complaint, charge, opposition, report, participation in an investigation).
#
# Baseline diagnosis: the regex baseline scores (a) a protected-activity
# lexicon, (b) an adverse-action lexicon, (c) temporal-connector words, and
# (d) proximity between an activity mention and an action mention within a
# fixed window, then blends them. Reading the train set against it shows two
# concrete failure modes:
#
#   1. False positives -- plain discrimination narratives with NO protected
#      activity at all (a bare "terminated because of his race" story, no
#      complaint/report/charge anywhere) still score 0.3-0.5 because the
#      action-word and temporal-word terms (30%+15% of the weight) fire on
#      their own, unconditioned on any activity evidence. The judge scores
#      these at 0. Retaliation-adverse-action requires the causal PAIR
#      (activity, then action) -- an action term alone proves nothing.
#   2. False negatives -- genuine retaliation chains that use adverse-action
#      synonyms outside the baseline's fixed lexicon (e.g. "discharged"
#      instead of "fired/terminated", a contract "not... renewed" instead of
#      "demoted/suspended") or protected-activity phrasing outside the fixed
#      lexicon (e.g. telling the employer "I believe I'm being discriminated
#      against" as opposition conduct, rather than the literal word
#      "complained") score near 0 even though the judge scores them 1.0.
#      This is exactly the doctrinal-construct trap the corpus notes warn
#      about: term presence is a weak proxy, and no fixed lexicon can cover
#      every way people narrate "I spoke up, then they hit me for it."
#
# Design: keep the baseline's lexical/proximity signal as a code-side
# backstop (it still catches the clean explicit cases), and ADD:
#   (1) two LLM fields carrying the thick judgment code can't reach -- what
#       the protected activity was, and what the post-activity action was --
#       so paraphrased/synonymous activity->action chains the fixed lexicon
#       misses still register;
#   (2) a code-only temporal-structure signal (tight-causation phrasing plus
#       real date arithmetic over ops.extract_dates) that operationalizes
#       "shortly after" with actual day-gap computation instead of keyword
#       counting;
#   (3) a hard gate: if NEITHER code nor the LLM finds any protected-activity
#       evidence at all, the score is capped low regardless of how much
#       adverse-action or temporal language is present -- this directly
#       fixes failure mode #1 without touching the genuine cases.
# The LLM never scores anything itself; all predicate/combination logic
# stays in code.

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
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)  # "6th" -> "6"
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


# --- code layer 1: lexical baseline signal (kept, lexicon gaps patched) ---
_ACTIVITY_RE = re.compile(
    r"\b(complain\w*|filed a charge|filed an? (?:eeoc )?complaint|reported|"
    r"opposed|objected|participat\w* in (?:the|an) investigation|grievance)\b",
    re.I)
_ULTIMATE_RE = re.compile(
    r"\b(terminat\w*|fired|discharg\w*|demot\w*|suspend\w*|"
    r"denied (?:a )?promotion|(?:not|never) (?:be )?renew\w*|non-?renewal)\b",
    re.I)
_BROADER_RE = re.compile(
    r"\b(reassign\w*|exclud\w*|negative (?:performance )?review|"
    r"increased scrutiny|written up|write-?up|shift chang\w*|"
    r"denied (?:a )?transfer|isolat\w*|left off)\b", re.I)
_TEMPORAL_RE = re.compile(
    r"\b(after|following|shortly after|less than (?:a|one|two|three|\d+) \w+ later|"
    r"within (?:a )?(?:\d+|few) (?:days|weeks|months))\b", re.I)
_ACTIVITY_PROX_RE = re.compile(
    r"\b(complain\w*|filed a charge|reported|opposed|objected|grievance)\b", re.I)
_ACTION_PROX_RE = re.compile(
    r"\b(terminat\w*|fired|discharg\w*|demot\w*|suspend\w*|reassign\w*|"
    r"exclud\w*|negative review|scrutiny|write-?up|(?:not|never) renew\w*)\b",
    re.I)


def _lexical_signal(t):
    protected_activity = len(_ACTIVITY_RE.findall(t))
    ultimate_action = len(_ULTIMATE_RE.findall(t))
    broader_action = len(_BROADER_RE.findall(t))
    temporal_connector = len(_TEMPORAL_RE.findall(t))

    prox = 0
    for m in _ACTIVITY_PROX_RE.finditer(t):
        window = t[m.end(): m.end() + 300]
        if _ACTION_PROX_RE.search(window):
            prox += 1

    activity_signal = _sat(protected_activity, 1.5)
    action_signal = _sat(ultimate_action * 1.3 + broader_action * 1.0, 2.5)
    temporal_signal = _sat(temporal_connector, 1.5)
    prox_signal = _sat(prox, 1.0)

    s = (0.20 * activity_signal + 0.30 * action_signal
         + 0.15 * temporal_signal + 0.35 * prox_signal)
    s = max(0.0, min(1.0, s))
    return s, protected_activity, (ultimate_action + broader_action)


# --- code layer 2: temporal structure (phrasing + real date arithmetic) ---
_CAUSAL_PHRASE_RE = re.compile(
    r"\b(same day|the next day|the following day|shortly (?:after|thereafter)|"
    r"immediately after|less than (?:a|one|two|three|\d+) \w+ later|"
    r"within (?:one|a|two|1|2|three|3) (?:day|days|week|weeks))\b", re.I)


def _temporal_signal(t, ops):
    phrase_hits = 0
    for m in _CAUSAL_PHRASE_RE.finditer(t):
        window = t[max(0, m.start() - 150): m.end() + 150]
        if _ACTION_PROX_RE.search(window):
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


# --- LLM layer: thick-input grounding for the activity->action construct ---
def _llm_signal(extracted):
    extracted = extracted or {}
    activity = str(extracted.get("protected_activity", "") or "")
    action = str(extracted.get("post_activity_action", "") or "")
    has_activity = not _is_none(activity)
    has_action = not _is_none(action)
    if has_activity and has_action:
        sig = 1.0
    elif has_activity or has_action:
        sig = 0.3
    else:
        sig = 0.0
    return sig, has_activity, has_action


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        lex, activity_ct, action_ct = _lexical_signal(t)
        llm, llm_activity, llm_action = _llm_signal(extracted)
        temporal = _temporal_signal(t, ops)

        activity_present = activity_ct > 0 or llm_activity
        action_present = action_ct > 0 or llm_action

        if not activity_present:
            # No evidence anywhere of a protected activity: this is a plain
            # discrimination (or unrelated) narrative, not retaliation.
            # Adverse-action / temporal language alone must not drive the
            # score up -- that was the baseline's main false-positive mode.
            return max(0.0, min(1.0, 0.10 * lex))

        temporal_gated = temporal if (activity_present and action_present) else 0.0

        s = 0.40 * lex + 0.40 * llm + 0.20 * temporal_gated
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

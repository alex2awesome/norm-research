# Hybrid module for legal_title_vii aspect a10: "cat's-paw evidence"
# (biased subordinate/coworker influences an ostensibly unbiased decision-maker's
# adverse action; Staub v. Proctor Hospital).
#
# Baseline diagnosis: the regex baseline scores presence of role words
# (supervisor/manager/coworker), action-feed words (report/recommend/complain),
# bias words, and generic "relied on/rubber-stamped" phrasing. It cannot tell
# whether the person doing the biased reporting and the person making the
# adverse decision are actually TWO DISTINCT roles (the doctrinal core of
# cat's-paw) versus the SAME actor being both biased and the decision-maker
# (ordinary direct discrimination/retaliation, not cat's-paw) -- e.g. a
# plaintiff who is herself titled "Foreman" trips the "foreman" keyword with
# no cat's-paw structure at all. It also has no notion of *temporal* proximity
# between the biased act (a report/accusation/altered record) and the adverse
# decision, which is exactly the kind of structural/causal signal code CAN
# reach reliably via date arithmetic and "same day / shortly after" phrasing.
#
# Design: keep the baseline's lexical evidence (down-weighted, since term
# presence is a weak proxy for the construct per the corpus notes) and ADD:
#   (1) two LLM fields that carry the thick judgment code cannot make --
#       whether the text actually names a biased source AND a separate
#       decision-maker, and whether that decision-maker is shown relying on
#       the source's input;
#   (2) a code-only temporal-structure signal (proximity phrases + real date
#       arithmetic over ops.extract_dates) that measures how tightly the
#       biased act and the adverse decision are chained in time -- a hallmark
#       of cat's-paw causation that no keyword count can capture.
# The LLM never scores anything; all predicate logic stays in code.

import re
import math
import datetime

LLM_FIELDS = {
    "cats_paw_actors": (
        "In <=15 words, name the biased person's role and the SEPARATE "
        "decision-maker/body who acted on their input (e.g. 'supervisor "
        "recommended; HR committee fired her'), or say NONE if the biased "
        "person and the decision-maker are the same person or no adverse "
        "decision followed."
    ),
    "reliance_evidence": (
        "In <=15 words, quote or paraphrase how the decision-maker relied "
        "on/adopted the biased person's report, recommendation, or altered "
        "record, or say NONE if no such reliance is shown."
    ),
}


def _sat(x, k):
    return 1.0 - math.exp(-x / max(1e-6, k))


def _is_none(s):
    if not s:
        return True
    s2 = re.sub(r"[.\s]+$", "", s.strip().lower())
    return s2 in ("", "none", "n/a", "na", "no", "not stated", "unclear", "unknown", "not shown", "not present")


_DATE_FORMATS = (
    "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%b. %d, %Y",
    "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
)


def _parse_date(s):
    s = s.strip().strip(",.")
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)  # "3rd" -> "3"
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt)
        except Exception:
            continue
    if re.fullmatch(r"(19|20)\d{2}", s):
        try:
            return datetime.datetime(int(s), 7, 1)  # bare year: mid-year anchor, coarse
        except Exception:
            return None
    return None


_SUBORDINATE_RE = re.compile(
    r"\b(supervisor|manager|coworker|co-worker|team lead|foreman)\b", re.I)
_ACTION_FEED_RE = re.compile(
    r"\b(recommend\w*|report(?:ed)?|write-?up|written up|complain(?:ed|t)?|"
    r"allegation|accus\w*)\b", re.I)
_BIAS_RE = re.compile(
    r"\b(bias(?:ed)?|animus|grudge|dislik\w*|prejudic\w*|resent\w*|retaliat\w*)\b", re.I)
_RELIANCE_RE = re.compile(
    r"\b(relied on|based on (?:the|his|her|their) (?:report|recommendation|investigation)|"
    r"rubber[- ]stamp\w*|adopted (?:the|his|her|their) recommendation|"
    r"without (?:independent|further) investigation|at the (?:request|direction) of)\b", re.I)
_DECISIONMAKER_RE = re.compile(
    r"\b(hr|human resources|director|vp|committee|board|panel|"
    r"upper management|executive|ultimate decision[- ]maker|chief|president)\b", re.I)
_PROX_ACTION_RE = re.compile(
    r"\b(recommend\w*|report(?:ed)?|complain\w*|write-?up)\b", re.I)

_CAUSAL_PHRASE_RE = re.compile(
    r"\b(same day|the next day|the following day|"
    r"within (?:one|a|two|1|2) (?:day|days|week|weeks)|"
    r"less than (?:one|two|three|1|2|3) (?:week|weeks|day|days)|"
    r"shortly (?:after|thereafter)|immediately after)\b", re.I)
_ADVERSE_ACTION_RE = re.compile(
    r"\b(fired|terminat\w*|discharg\w*|denied|dismiss\w*|non-?renew\w*|"
    r"not renewed|demot\w*|suspend\w*)\b", re.I)


def _lexical_signal(t):
    subordinate = len(_SUBORDINATE_RE.findall(t))
    action_feed = len(_ACTION_FEED_RE.findall(t))
    bias_marker = len(_BIAS_RE.findall(t))
    reliance = len(_RELIANCE_RE.findall(t))
    decisionmaker = len(_DECISIONMAKER_RE.findall(t))

    prox = 0
    for m in _SUBORDINATE_RE.finditer(t):
        window = t[m.end(): m.end() + 250]
        if _PROX_ACTION_RE.search(window):
            prox += 1

    s = (0.25 * _sat(subordinate, 2.0)
         + 0.20 * _sat(action_feed, 2.0)
         + 0.15 * _sat(bias_marker, 1.5)
         + 0.25 * _sat(reliance, 1.5)
         + 0.10 * _sat(decisionmaker, 1.5)
         + 0.15 * _sat(prox, 1.0))
    return max(0.0, min(1.0, s))


def _temporal_signal(t, ops):
    # (a) explicit tight-causation phrasing near an adverse-action word
    phrase_hits = 0
    for m in _CAUSAL_PHRASE_RE.finditer(t):
        window = t[max(0, m.start() - 120): m.end() + 120]
        if _ADVERSE_ACTION_RE.search(window):
            phrase_hits += 1
    phrase_score = _sat(phrase_hits, 1.0)

    # (b) real date arithmetic: tight gap between two parseable dates suggests
    # a fast biased-report -> adverse-action chain (classic cat's-paw timing)
    gap_score = 0.0
    try:
        dates = ops.extract_dates(t)
    except Exception:
        dates = []
    parsed = sorted(d for d in (_parse_date(x) for x in dates) if d is not None)
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


def _llm_signal(extracted):
    extracted = extracted or {}
    actors = str(extracted.get("cats_paw_actors", "") or "")
    reliance = str(extracted.get("reliance_evidence", "") or "")

    if _is_none(actors):
        return 0.0
    sig = 0.6
    if not _is_none(reliance):
        sig += 0.4
    return min(1.0, sig)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        lex = _lexical_signal(t)
        llm = _llm_signal(extracted)
        temporal = _temporal_signal(t, ops)

        s = 0.40 * lex + 0.40 * llm + 0.20 * temporal
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

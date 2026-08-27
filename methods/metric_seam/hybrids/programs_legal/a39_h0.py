"""a39 hybrid: discipline postdates protected activity.

Criterion: the FIRST disciplinary/adverse action against the plaintiff that
is central to the retaliation claim is dated AFTER the plaintiff's protected
activity (complaint, agency charge, grievance, opposition, accommodation
request) -- the classic retaliation timing signature.

Baseline (v2_holistic, train_rho=0.268) has three visible flaws in the pack's
train examples:

  1. Textual position is treated as a proxy for chronological order
     (`order_score`: any discipline-keyword occurring AFTER the first
     protected-keyword occurrence IN THE TEXT). A generic closing-summary
     sentence like "...this discipline was imposed in violation of Title VII"
     near the END of a document trips this even when the real chronology is
     reversed -- d00761 (Moore): the suspension plainly PRECEDES the EEOC
     charge in time, yet the baseline scores 0.71 (judge: 0.0) because a
     late "disciplined"/"discipline" mention sits textually after the charge
     sentence, and its `retaliat*` connector regex fires on the word alone
     with no direction check.
  2. Both keyword lists are too narrow. `discipline_kws` misses "terminated",
     "discharged", "non-renewal"/"would not be renewed", "counseling notice"
     etc., and `protected_kws` misses "accommodation/exemption request".
     d01080 (non-renewal after an EEO complaint) and d00795 (termination
     after a religious-exemption request) are judge=1.0 but baseline scores
     0.22 / 0.0 -- pure vocabulary misses, not a construct problem.
  3. Legal-term presence is a weak proxy for the construct being factually
     true (per the corpus notes) -- whether a *given* disciplinary act is
     the one "central" to the retaliation claim, versus an unrelated/earlier
     personnel record, requires reading the narrative, not counting words.

Design: code keeps (broadened) keyword presence as an evidence/gating
signal and does real date arithmetic (nearest calendar date to each
mention, like the baseline, but position-anchored via a home-grown date
finder rather than assuming the ops date list carries positions). The LLM
carries the two things code cannot reach: (a) which act is the *central*
adverse action and whether it comes after protected activity, holistically,
even when phrased as "two months later" / "shortly after" rather than a
bare calendar date; (b) a broad-vocabulary restatement of that act, used as
a presence backstop so the fixed keyword list is no longer the ceiling.
Presence of BOTH a protected activity and a disciplinary act GATES the
score (soft AND) -- cases with only one of the two (or neither) score near
0, which is where several judge=0.0 train cases sit (no discipline
mentioned at all, or discipline with no protected activity anywhere).
"""
import re
import math

LLM_FIELDS = {
    "order": (
        "Does the disciplinary/adverse action AGAINST THE PLAINTIFF that is "
        "central to their retaliation claim (e.g. write-up, warning, "
        "suspension, demotion, PIP, non-renewal, discharge/termination) "
        "occur chronologically AFTER the plaintiff's protected activity "
        "(complaint, agency/EEOC charge, grievance, opposition, "
        "accommodation or exemption request)? Answer exactly one word: "
        "AFTER, BEFORE, or NONE if either event is missing or the order is "
        "genuinely unclear."
    ),
    "discipline_desc": (
        "In <=15 words, name the first specific disciplinary/adverse action "
        "taken against the plaintiff (e.g. write-up, suspension, PIP, "
        "demotion, non-renewal, termination), or say NONE if no such action "
        "against the plaintiff is described."
    ),
}

_NONE_ANSWERS = {"", "none", "n/a", "na", "not stated", "not mentioned",
                 "unclear", "unknown", "not shown", "not described", "no"}

_PROTECTED_KWS = [
    "complain", "complaint", "filed a charge", "filed an eeoc", "reported",
    "opposed", "grievance", "charge of discrimination", "protected activity",
    "accommodation request", "requested a religious exemption",
    "requested an exemption", "exemption request", "requested an accommodation",
    "objected to",
]
_DISCIPLINE_KWS = [
    "written warning", "write-up", "writeup", "reprimand", "disciplinary action",
    "disciplined", "discipline", "suspension", "suspended",
    "performance improvement plan", "pip", "negative evaluation",
    "poor performance review", "terminated", "termination", "fired",
    "discharged", "discharge", "dismissed", "dismissal", "demoted", "demotion",
    "non-renewal", "not be renewed", "would not be renewed", "not renewed",
    "counseling notice", "record of counseling", "final warning",
    "placed on probation", "reassigned to", "transferred to",
]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_ALT = "|".join(_MONTHS.keys())
_DATE_RE = re.compile(
    r"\b(" + _MONTH_ALT + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b|"
    r"\b(" + _MONTH_ALT + r")\.?\s+(\d{4})\b"
)
_DATE_NUM_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

_AFTER_RE = re.compile(
    r"\b(after (?:her|his|the) (?:complaint|charge|grievance|report|filing|"
    r"request)|following (?:her|his|the) (?:complaint|charge)|"
    r"in retaliation for|shortly after (?:filing|complaining)|"
    r"less than \w+ (?:weeks?|days?) after)\b"
)
_BEFORE_RE = re.compile(
    r"\b(before (?:she|he|filing|her complaint|his complaint)|"
    r"prior to (?:her|his) (?:complaint|charge)|"
    r"predat(?:e|ing|ed) (?:her|his|the)? ?complaint)\b"
)


def _sat(x, k):
    return 1.0 - math.exp(-x / max(1e-6, k))


def _is_none(s):
    if not s:
        return True
    return re.sub(r"[.\s]+$", "", s.strip().lower()) in _NONE_ANSWERS


def _find_dates(t):
    """(position, (year, month)) pairs -- home-grown, position-anchored."""
    out = []
    for m in _DATE_RE.finditer(t):
        if m.group(1):
            month, year = _MONTHS[m.group(1)], int(m.group(3))
        else:
            month, year = _MONTHS[m.group(4)], int(m.group(5))
        if 1000 < year < 2100:
            out.append((m.start(), (year, month)))
    for m in _DATE_NUM_RE.finditer(t):
        month, year = int(m.group(1)), int(m.group(3))
        if 1 <= month <= 12 and 1000 < year < 2100:
            out.append((m.start(), (year, month)))
    out.sort(key=lambda x: x[0])
    return out


def _nearest(dates, pos, window=400):
    cand = [d for d in dates if abs(d[0] - pos) < window]
    if not cand:
        return None
    cand.sort(key=lambda d: abs(d[0] - pos))
    return cand[0][1]


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5
        try:
            t = ops.normalize(raw).lower()
        except Exception:
            t = raw.lower()
        ext = extracted or {}

        # --- code: keyword presence (evidence + gating) ---
        protected_pos = sorted(m.start() for kw in _PROTECTED_KWS
                                for m in re.finditer(re.escape(kw), t))
        discipline_pos = sorted(m.start() for kw in _DISCIPLINE_KWS
                                 for m in re.finditer(re.escape(kw), t))
        protected_code = _sat(len(protected_pos), 0.6)
        discipline_code = _sat(len(discipline_pos), 0.6)

        # --- code: real date arithmetic. Earliest dated protected mention
        # vs LATEST dated discipline mention -- the operative adverse action
        # in a retaliation narrative is usually the final/most severe one
        # (a later suspension, the eventual termination), not necessarily
        # the very first personnel record in the case.
        dates = _find_dates(t)
        date_order = None
        if protected_pos and discipline_pos and dates:
            p_date = _nearest(dates, protected_pos[0])
            d_date = _nearest(dates, discipline_pos[-1])
            if p_date and d_date:
                date_order = 1.0 if d_date >= p_date else 0.0

        # --- code: directional temporal-structure phrasing (small nudge,
        # not a standalone score -- unlike the baseline's undirected
        # `retaliat*` connector count that caused the Moore false positive).
        after_hits = len(_AFTER_RE.findall(t))
        before_hits = len(_BEFORE_RE.findall(t))
        connector_adj = _sat(after_hits, 1.0) - _sat(before_hits, 1.0)

        # --- LLM: thick-input grounding of the construct itself ---
        order_raw = str(ext.get("order", "") or "").strip().lower()
        if order_raw.startswith("after"):
            order_llm = 1.0
        elif order_raw.startswith("before"):
            order_llm = 0.0
        else:
            order_llm = 0.5

        disc_desc = str(ext.get("discipline_desc", "") or "")
        disc_llm_present = 0.0 if _is_none(disc_desc) else 1.0

        # --- combine ---
        discipline_signal = max(discipline_code, disc_llm_present)
        presence_gate = min(protected_code, discipline_signal)

        order_signal = order_llm
        if date_order is not None:
            order_signal = 0.6 * order_llm + 0.4 * date_order
        order_signal = order_signal + 0.15 * connector_adj
        order_signal = max(0.0, min(1.0, order_signal))

        s = presence_gate * order_signal
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

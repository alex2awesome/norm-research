"""a23 hybrid: procedural exhaustion and timeliness (Title VII administrative
charge -> EEOC processing -> right-to-sue -> suit).

Baseline diagnosis: v0_keyword (train rho=0.78) is a flat count of ten fixed
phrases ('eeoc', 'charge of discrimination', 'filed a charge', 'right-to-sue'
/'right to sue', '180 days', '300 days', 'exhaust', ...) over hits/4. It does
NO date arithmetic at all despite this being exactly the mechanical,
quantity-driven criterion where real date arithmetic should beat keyword
counting (per corpus notes). Two concrete failure modes in the training data:

  1. False POSITIVE: 'eeoc'/'equal employment opportunity commission' fires
     even when the EEOC is mentioned as a co-litigant/enforcing party (e.g.
     "The EEOC and Vincent Jackson ... contend that Dollar General's use of
     [a medical exam] violated the ADA and GINA") rather than as a step in
     THIS plaintiff's own administrative process. That example scores
     baseline 0.5 vs judge 0.0 -- pure keyword substring match cannot tell
     "EEOC as party" from "EEOC as this plaintiff's charge-processing agency".

  2. False NEGATIVE: genuine, complete exhaustion narratives that don't use
     the private-sector vocabulary at all score 0 under the baseline, e.g.
     federal-sector processing ("contacted an EEO counselor" ... "filed a
     formal complaint" ... "request for a final agency decision"), state/
     local deferral agencies (NEOC, human rights commissions), or a bare
     appellate-style statement like "plaintiffs had complied with the
     administrative and procedural requirements of Title VII". These are
     judged 1.0 (a real, if differently-worded, exhaustion story is present)
     but baseline gives 0.0 because none of its 10 phrases appear.

Design: keep the baseline's keyword mechanism (broadened to cover the
federal-sector / state-agency vocabulary it was missing) as a code-only
floor signal, and ADD:
  (a) real date arithmetic: instead of just sorting every date in the
      document (the naive approach, which can't tell a charge-filing date
      from an unrelated fact date), dates are ANCHORED by proximity to
      charge-phase vs. completion-phase phrases, and only an anchored,
      order-consistent gap between a charge-anchored date and a later
      completion-anchored date counts as evidence of a genuine, computable
      exhaustion timeline;
  (b) two LLM fields for what regex fundamentally cannot see: which
      exhaustion stage (if any) THIS specific plaintiff personally reached
      (catches non-keyword phrasings, and is explicitly instructed to say
      NONE when the EEOC/agency appears only as a party -- fixing failure
      mode 1), and whether a live timeliness dispute is argued (catches
      cases like a defendant's time-bar defense that the judge rewards even
      when unresolved).
The LLM never scores anything directly; all predicate/arithmetic logic and
the false-positive guard stay in code.
"""
import re
import math
import datetime

LLM_FIELDS = {
    "exhaustion_step": (
        "In <=6 words: the LATEST EEOC/agency exhaustion step THIS "
        "plaintiff personally completed before suing (e.g. 'EEO counselor "
        "contacted', 'charge filed', 'right-to-sue received', 'complied "
        "with requirements'). Say NONE if the text doesn't describe this "
        "plaintiff's own administrative process (an agency such as the "
        "EEOC suing or appearing as a party/co-plaintiff does NOT count)."
    ),
    "timeliness_dispute": (
        "Does either side argue that THIS plaintiff's charge or suit was "
        "filed too late / is time-barred, even if unresolved? Answer YES, "
        "NO, or NONE if no such dispute or procedural exhaustion is "
        "discussed at all."
    ),
}


def _sat(x, k):
    return 1.0 - math.exp(-x / max(1e-6, k))


def _is_none(s):
    if not s:
        return True
    s2 = re.sub(r"[.\s]+$", "", s.strip().lower())
    return s2 in ("", "none", "n/a", "na", "unclear", "unknown", "not stated", "not shown")


_CHARGE_KWS = (
    "charge of discrimination", "filed a charge", "filed a sworn charge",
    "charge with the eeoc", "charge at the", "sworn charge",
    "eeo counselor", "contacted the eeo", "formal complaint",
    "administrative complaint",
)
_AGENCY_KWS = (
    "eeoc", "equal employment opportunity commission", "neoc",
    "human rights commission", "civil rights commission",
    "fair employment", "state deferral", "eeo office",
)
_COMPLETION_KWS = (
    "right-to-sue", "right to sue", "notice of right to sue",
    "final agency decision", "administrative remedies", "exhaust",
    "complied with the administrative", "complied with the procedural",
)
_DISPUTE_KWS = (
    "time-barred", "time barred", "untimely", "statute of limitations",
    "equitably tolled", "equitable tolling", "not timely", "out of time",
)

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
    if re.fullmatch(r"(19|20)\d{2}", s):
        try:
            return datetime.datetime(int(s), 7, 1)  # bare year: coarse mid-year anchor
        except Exception:
            return None
    return None


def _dates_with_offsets(normalized, ops):
    """Pair each ops.extract_dates surface string with its position in
    `normalized`, walking forward so repeated date strings map to their
    successive (not first) occurrences."""
    try:
        raw = ops.extract_dates(normalized) or []
    except Exception:
        raw = []
    out, cursor = [], 0
    for ds in raw:
        idx = normalized.find(ds, cursor)
        if idx < 0:
            idx = normalized.find(ds)
        if idx < 0:
            continue
        out.append((idx, ds))
        cursor = idx + len(ds)
    return out


def _nearby(lower_text, idx, span_len, window, phrases):
    lo = max(0, idx - window)
    hi = min(len(lower_text), idx + span_len + window)
    seg = lower_text[lo:hi]
    return any(p in seg for p in phrases)


def _keyword_score(lower_text):
    charge_hits = sum(1 for k in _CHARGE_KWS if k in lower_text)
    agency_hits = sum(1 for k in _AGENCY_KWS if k in lower_text)
    completion_hits = sum(1 for k in _COMPLETION_KWS if k in lower_text)
    total = charge_hits + agency_hits + completion_hits
    return min(1.0, total / 4.0)


def _date_bonus(normalized, lower_text, ops):
    pairs = _dates_with_offsets(normalized, ops)
    if not pairs:
        return 0.0

    charge_anchored = []
    completion_anchored = []
    for idx, ds in pairs:
        dt = _parse_date(ds)
        if dt is None:
            continue
        if _nearby(lower_text, idx, len(ds), 80, _CHARGE_KWS):
            charge_anchored.append((idx, dt))
        if _nearby(lower_text, idx, len(ds), 80, _COMPLETION_KWS):
            completion_anchored.append((idx, dt))

    if not charge_anchored and not completion_anchored:
        return 0.0

    bonus = 0.10  # at least one procedurally-anchored, parseable date exists

    if charge_anchored and completion_anchored:
        # anchor selection: first-mentioned charge date, last-mentioned
        # completion date -- approximates "charge filed" -> "exhaustion
        # completed" rather than blindly diffing arbitrary document dates.
        charge_dt = min(charge_anchored, key=lambda p: p[0])[1]
        completion_dt = max(completion_anchored, key=lambda p: p[0])[1]
        gap_days = (completion_dt - charge_dt).days
        if -5 <= gap_days <= 3650:
            bonus += 0.15  # real, order-consistent computed exhaustion window

    return min(0.25, bonus)


def _step_score(extracted):
    step = (extracted.get("exhaustion_step") or "").strip().lower()
    if _is_none(step):
        return 0.0, True
    if any(w in step for w in ("right-to-sue", "right to sue", "final agency decision",
                                "complied", "exhaust")):
        return 1.0, False
    if any(w in step for w in ("charge", "formal complaint", "administrative complaint")):
        return 0.6, False
    if any(w in step for w in ("counselor", "contacted", "notified")):
        return 0.4, False
    return 0.3, False  # some step described, unrecognized phrasing


def score(text: str, extracted: dict, ops) -> float:
    try:
        ext = extracted or {}
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            normalized = ops.normalize(raw)
        except Exception:
            normalized = raw
        lower = normalized.lower()

        kw = _keyword_score(lower)
        date_bonus = _date_bonus(normalized, lower, ops)
        step_score, is_none_step = _step_score(ext)

        # false-positive guard: LLM says this isn't about this plaintiff's own
        # exhaustion (e.g. EEOC appears only as a co-litigant/party) -> heavily
        # discount the code-side keyword/date signals that likely fired on
        # that same irrelevant mention.
        if is_none_step:
            kw *= 0.2
            date_bonus *= 0.2

        disp = (ext.get("timeliness_dispute") or "").strip().lower()
        dispute_bonus = 0.15 if disp.startswith("yes") else 0.0
        if disp == "" and any(k in lower for k in _DISPUTE_KWS):
            dispute_bonus = max(dispute_bonus, 0.10)

        s = 0.35 * kw + date_bonus + 0.30 * step_score + dispute_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

"""a26 hybrid: Suit filed within 90 days of the right-to-sue letter.

Criterion: the gap between (a) the date the plaintiff received/was mailed the
EEOC (or other agency) right-to-sue letter / notice of rights / final agency
action ending the administrative process, and (b) the date the complaint was
filed in court, is within the ~90-day statutory window.

Baseline (v0_keyword, train_rho=0.482) counts a fixed phrase list
('right-to-sue', '90 days', 'timely filed this action', ...) with no regard
for polarity or for whether a filing date is ever actually established. Train
inspection shows this is exactly backwards for a mechanical, date-driven
criterion:
  - FALSE POSITIVES: documents that merely NAME the right-to-sue letter/notice
    (d00437, d00937, d00186, all judge=0.0) trip 2-3 keywords and score
    0.33-0.67, even though no complaint-filing date is ever given (or, in
    d00186, the notice's very validity is disputed) -- there is no
    established gap at all, just term presence.
  - The one clean win (d00095, judge=1.0, baseline=1.0) happens to state the
    conclusion in words ("timely filed suit ... within 90 days") without
    giving raw calendar dates -- so a pure date-diff approach with no
    textual-assertion fallback would silently drop this case to 0.
  - Per the corpus notes, real date arithmetic beats keyword counting for
    mechanical/quantity criteria like this one; the clear judge=1.0 cases
    (d00905, d00490, d00812, d00570, d00856) all have two explicit calendar
    dates whose difference is well under 90 days, and the keyword baseline
    under- or over-scores several of them (0.33, 0.67) purely from lexical
    variety, not from the actual gap.

Design: two LLM fields carry the THICK grounding code cannot do alone --
identifying WHICH of the (possibly many) dates in the document are the
right-to-sue-trigger date and the complaint-filing date. Code then does the
PREDICATE: parse both into real calendar dates and compute the day gap
(primary signal, per the corpus notes). We keep the baseline's keyword scan,
but repaired -- it is now polarity-aware (untimely/time-barred phrasing
pulls the score down instead of up) and used only as a fallback when the two
anchor dates aren't both extractable, never as a source of false positives
from bare term presence.
"""
import re
import datetime
import calendar

LLM_FIELDS = {
    "rts_date": (
        "Exact calendar date (e.g. 'March 8, 1988') the plaintiff received or was "
        "mailed the right-to-sue letter / notice of right to sue, or the final "
        "agency action/decision ending the administrative process and starting the "
        "suit-filing clock. Answer NONE if no such date is given."
    ),
    "filed_date": (
        "Exact calendar date the complaint or lawsuit discussed in THIS document "
        "was filed/instituted in court (not an EEOC/agency charge-filing date). "
        "Answer NONE if no such date is given."
    ),
}

# ---------------------------------------------------------------- date parse
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))
_RE_MDY = re.compile(r"\b(" + _MONTH_ALT + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I)
_RE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_ALT + r")\.?,?\s+(\d{4})\b", re.I)
_RE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_RE_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

_NONE_ANSWERS = {"", "none", "n/a", "na", "not stated", "not mentioned", "unknown", "unclear"}


def _is_none(s):
    return (not s) or (s.strip().lower() in _NONE_ANSWERS)


def _day_ordinal(y, mo, d):
    if not (1 <= mo <= 12) or not (1000 < y < 3000):
        return None
    try:
        max_day = calendar.monthrange(y, mo)[1]
        d = min(max(d, 1), max_day)  # defensively clamp typo'd days (e.g. "April 31")
        return datetime.date(y, mo, d).toordinal()
    except Exception:
        return None


def _parse_date(s):
    if _is_none(s):
        return None
    t = s.strip()
    m = _RE_ISO.search(t)
    if m:
        return _day_ordinal(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _RE_SLASH.search(t)
    if m:
        return _day_ordinal(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = _RE_MDY.search(t)
    if m:
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            return _day_ordinal(int(m.group(3)), mo, int(m.group(2)))
    m = _RE_DMY.search(t)
    if m:
        mo = _MONTHS.get(m.group(2).lower())
        if mo:
            return _day_ordinal(int(m.group(3)), mo, int(m.group(1)))
    return None


def _diff_score(diff_days):
    """Smooth threshold around the 90-day line (soft margin absorbs off-by-a-
    few-days parse slop instead of a hard cliff at exactly 90)."""
    if diff_days < 0:
        return None  # filed "before" the trigger date -> likely a bad extraction
    if diff_days <= 85:
        return 1.0
    if diff_days <= 100:
        return max(0.0, 1.0 - (diff_days - 85) / 15.0)
    return 0.0


# --------------------------------------------------------- repaired baseline
_BASE_KWS = ("right-to-sue", "right to sue", "notice of right to sue", "90 days",
             "ninety days", "within 90", "timely filed this action", "received the notice")
_POS_PHRASES = ("within 90 days", "within ninety days", "within the 90-day",
                "within the 90 day", "timely filed", "timely instituted",
                "timely commenced", "timely brought", "filed this action within")
_NEG_PHRASES = ("untimely", "not timely", "time-barred", "time barred",
                "exceeded the 90-day", "exceeded the 90 day", "outside the 90-day",
                "outside the 90 day", "more than 90 days", "beyond the 90-day",
                "did not file within", "failed to file within")


def _fallback_signal(t):
    """Used only when the two anchor dates aren't both extractable. Polarity-
    aware (unlike the raw baseline): a bare mention of 'right to sue' with no
    timeliness assertion no longer scores as a false positive."""
    has_pos = any(p in t for p in _POS_PHRASES)
    has_neg = any(p in t for p in _NEG_PHRASES)
    if has_pos and has_neg:
        phrase = 0.4  # both readings argued -> genuinely disputed
    elif has_neg:
        phrase = 0.05
    elif has_pos:
        phrase = 0.9
    else:
        phrase = 0.0

    hits = sum(1 for k in _BASE_KWS if k in t)
    kw = min(1.0, hits / 3.0)
    return max(0.0, min(1.0, 0.75 * phrase + 0.25 * kw))


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
        rts_ord = _parse_date(str(ext.get("rts_date", "") or ""))
        filed_ord = _parse_date(str(ext.get("filed_date", "") or ""))

        if rts_ord is not None and filed_ord is not None:
            diff = filed_ord - rts_ord
            primary = _diff_score(diff)
            if primary is not None:
                return primary

        return _fallback_signal(t)
    except Exception:
        return 0.5

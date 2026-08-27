"""a12 hybrid: code does real date arithmetic against the SSR 24-3p cutoff (2025-01-06); one LLM field names which of the narrative's many dates is the ALJ decision date."""

# Criterion is purely factual: whether the ALJ decision under review issued
# on/after SSR 24-3p's effective date, which disables the mandatory SSR
# 00-4p DOT-conflict duty. Higher score = stronger evidence the decision
# postdates the cutoff.
#
# Design: the LLM field names the decision date in prose (a narrative
# typically contains many OTHER dates -- onset, filing, hearing, prior
# decisions -- that a bare ops.extract_dates() call can't disambiguate);
# code parses the field's answer with a small format table and compares to
# the cutoff via real datetime arithmetic. Per the corpus notes, date
# arithmetic beats keyword counting for this kind of mechanical,
# statutory-quantity criterion. If the field is missing/unparseable, code
# falls back to the LATEST parseable date found anywhere in the text via
# ops.extract_dates as a weaker proxy for the decision date (appeal
# narratives typically close on the most recent procedural date), with a
# documented low-confidence default if no usable date exists at all.
import re
import datetime

LLM_FIELDS = {
    "alj_decision_date": (
        "State the exact date the ALJ issued the decision under review "
        "(not onset/filing/hearing dates), Month Day, Year format, else NONE."
    ),
}

_CUTOFF = datetime.date(2025, 1, 6)

_DATE_FORMATS = (
    "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%b. %d, %Y",
    "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def _is_bare_year(s):
    return bool(re.fullmatch(r"(19|20)\d{2}", s.strip()))


def _parse_date(s):
    s = s.strip().strip(",.")
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except Exception:
            continue
    if _is_bare_year(s):
        try:
            return datetime.date(int(s), 7, 1)  # bare year: mid-year anchor, coarse
        except Exception:
            return None
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        ex = extracted if isinstance(extracted, dict) else {}
        field_val = str(ex.get("alj_decision_date") or "").strip()

        if not _is_none(field_val):
            if _is_bare_year(field_val):
                y = int(field_val)
                if y > 2025:
                    return 1.0
                if y == 2025:
                    return 0.9  # most of 2025 postdates Jan 6
                return 0.0
            d = _parse_date(field_val)
            if d is not None:
                return 1.0 if d >= _CUTOFF else 0.0

        # fallback: latest parseable date anywhere in the narrative
        try:
            raw_dates = ops.extract_dates(text or "")
        except Exception:
            raw_dates = []
        parsed = [p for p in (_parse_date(x) for x in raw_dates) if p is not None]
        if parsed:
            latest = max(parsed)
            return 0.7 if latest >= _CUTOFF else 0.05

        return 0.1  # no usable date evidence anywhere: documented low prior
    except Exception:
        return 0.5

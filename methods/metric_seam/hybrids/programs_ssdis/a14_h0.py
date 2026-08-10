"""a14 hybrid: code looks up a year-indexed SGA dollar table and does the earnings/threshold ratio arithmetic; two LLM fields carry the post-onset monthly-earnings amount and its year."""

# Criterion is purely mechanical: whether reported post-onset monthly
# earnings meet/exceed the SGA (substantial gainful activity) dollar
# threshold for the relevant year. Higher score = stronger evidence
# earnings are at/above the threshold.
#
# Design: per the corpus notes, statutory dollar quantities are genuinely
# load-bearing and real arithmetic beats keyword counting here. A narrative
# is typically full of OTHER dollar figures (benefit amounts, damages,
# pre-onset wages) that a bare regex dollar-scan can't attribute to the
# right time window, so an LLM field extracts the post-onset monthly figure
# specifically; a second field gives the year so code can select the
# correct year-indexed threshold from a public SSA reference table
# (best-effort for the most recent years, documented below). When the year
# field is missing, code falls back to the latest year mentioned anywhere
# in the text via ops.extract_dates. Unparseable/absent earnings default to
# a low score, reflecting that most SSDI claimants by definition are not
# working near the SGA level.
import re

LLM_FIELDS = {
    "post_onset_monthly_earnings": (
        "State the claimant's approximate monthly dollar earnings AFTER "
        "the alleged disability onset date, else NONE."
    ),
    "earnings_year": (
        "What calendar year do those post-onset monthly earnings pertain "
        "to, else NONE."
    ),
}

# Public SSA non-blind SGA monthly thresholds by year (best-effort; recent
# years are approximate/extrapolated where not independently verified).
_SGA_TABLE = {
    2014: 1070, 2015: 1090, 2016: 1130, 2017: 1170, 2018: 1180, 2019: 1220,
    2020: 1260, 2021: 1310, 2022: 1350, 2023: 1470, 2024: 1550, 2025: 1620,
    2026: 1690,
}
_MIN_YEAR, _MAX_YEAR = min(_SGA_TABLE), max(_SGA_TABLE)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def _parse_dollars(s):
    m = re.search(r"\$?\s?([\d,]+(?:\.\d+)?)", s or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def _parse_year(s):
    m = re.search(r"(?:19|20)\d{2}", s or "")
    return int(m.group(0)) if m else None


def _threshold_for_year(y):
    y = max(_MIN_YEAR, min(_MAX_YEAR, y))
    return _SGA_TABLE[y]


def score(text: str, extracted: dict, ops) -> float:
    try:
        ex = extracted if isinstance(extracted, dict) else {}
        earnings_raw = str(ex.get("post_onset_monthly_earnings") or "")
        year_raw = str(ex.get("earnings_year") or "")

        if _is_none(earnings_raw):
            return 0.05  # no reported post-onset earnings: low default

        earnings = _parse_dollars(earnings_raw)
        if earnings is None:
            return 0.05

        year = _parse_year(year_raw) if not _is_none(year_raw) else None
        if year is None:
            try:
                raw_dates = ops.extract_dates(text or "")
            except Exception:
                raw_dates = []
            years = [int(m) for d in raw_dates for m in re.findall(r"(?:19|20)\d{2}", d)]
            year = max(years) if years else _MAX_YEAR

        threshold = _threshold_for_year(year)
        ratio = earnings / threshold

        if ratio >= 1.0:
            final = min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
        else:
            final = 0.75 * ratio

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5

"""a36 hybrid: Same actor hire-to-fire gap.

Criterion: the same named individual is both the hirer/promoter and the
terminator of the plaintiff, and the elapsed time between those two events
is short (a classic pretext-timing signal in Title VII case narratives).

Baseline (v1_structure, train_rho=-0.03, i.e. noise) has two flaws visible
in the pack's train examples:
  1. Its "same actor" detection needs an exact surface pattern ("X hired
     her" / "hired by X" ... "X fired her" / "fired by X"). Real narratives
     almost never use that exact frame — the hirer/terminator relationship
     is usually implicit ("her supervisor was X" ... "X terminated her
     employment"), or the hire/fire is attributed to the COMPANY while a
     named individual is only the supervisor/decision-driver. Regex just
     misses this; it's thick-input grounding, so actor identity goes to an
     LLM field (code still holds the same-surname predicate).
  2. Its gap estimate scans for the PHRASE "N months/days later" rather
     than doing real date arithmetic. Per the corpus notes, calendar dates
     are extractable and load-bearing for temporal-structure criteria —
     "N days after termination" language in the corpus frequently refers to
     something else entirely (e.g. a filing deadline, d01165), which is
     exactly why the baseline's train_rho is ~0. We instead locate, among
     all hire-type and fire-type verb occurrences, the ones that actually
     sit next to a real calendar date, and take the earliest such hire
     date and latest such fire date as the two endpoints of a real
     month-difference.

We KEEP the baseline's explicit-phrase and named-pattern signals as a
code-side backstop/OR-term (same_actor_phrases, hire_names/fire_names) —
they still catch the rare case that writes the relationship out plainly —
but they no longer gate the score alone.

b4 extension (blind, construct-driven -- no eval signal used to pick these):
the criterion has two halves, "same actor" and "short gap", and h0's two
LLM fields (hire_actor/fire_actor) only ground the first half — actor
IDENTITY. Nothing grounds the temporal half beyond what code's regex-anchored
calendar-date search can find, and nothing checks whether the "same actor"
inference is undercut by a documented mid-tenure change of supervisor (a
name-collision risk, e.g. two different people sharing a common surname).
Two new fields target exactly these gaps:
  - tenure_duration: an LLM-read elapsed time (e.g. "8 years", "3 months")
    for narratives that state duration in prose without a parseable
    calendar date near a hire/fire verb — the case code's date-arithmetic
    path silently falls back to gap_score=0.15 (no evidence) on.
  - supervisor_continuity: whether the plaintiff had the SAME direct
    supervisor the whole tenure or supervisors changed — a documented
    change discounts a name-match same-actor signal that may be a
    surname collision rather than true continuity.
"""
import re

LLM_FIELDS = {
    "hire_actor": (
        "Name the specific INDIVIDUAL PERSON (not a company) described as hiring, "
        "promoting, or extending a job offer/role change to the plaintiff into the "
        "position from which the plaintiff was later terminated. Answer NONE if only "
        "a company/agency (not a named person) is credited with the hiring or "
        "promotion, or if no such person is named."
    ),
    "fire_actor": (
        "Name the specific INDIVIDUAL PERSON (not a company) described as firing, "
        "terminating, or deciding to terminate the plaintiff's employment. Answer "
        "NONE if only a company/agency (not a named person) is credited with the "
        "termination decision, or if no such person is named."
    ),
    "tenure_duration": (
        "State the approximate elapsed time between the plaintiff's hire or "
        "promotion into the role and their termination (e.g. '3 months', "
        "'8 years'). Answer UNKNOWN if not stated in the text."
    ),
    "supervisor_continuity": (
        "Did the plaintiff have the SAME direct supervisor for the entire "
        "period between hire and termination, or did supervisors change at "
        "some point? Answer SAME, CHANGED, or UNCLEAR."
    ),
}

_NONE_ANSWERS = {"", "none", "n/a", "na", "not stated", "not mentioned", "unknown"}

_TITLE_RE = re.compile(
    r"^(dr|mr|mrs|ms|prof|professor|judge|officer|chancellor|chief|capt|captain)\.?\s+",
    re.I,
)
_POSSESSIVE_RE = re.compile(r"[’']s\b")

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))
_DATE_FULL_RE = re.compile(
    r"\b(" + _MONTH_ALT + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I
)
_DATE_MY_RE = re.compile(r"\b(" + _MONTH_ALT + r")\.?\s+(\d{4})\b", re.I)

_HIRE_RE = re.compile(
    r"\b(hired|hire|re-?hired|promot(?:ed|ion)|"
    r"offered\s+(?:her|him|the\s+plaintiff|a\s+position|employment)|"
    r"extended\s+an?\s+offer|began\s+(?:his|her|working)|"
    r"started\s+(?:his|her|working)|start\s+of\s+employment|"
    r"reassign(?:ed|ment)|transferred|"
    r"move(?:d)?\s+(?:her|him|the\s+plaintiff)?\s*(?:to|into)\s+(?:a\s+)?(?:new\s+)?"
    r"(?:\w+\s+){0,3}(?:role|position)|"
    r"accepted\s+(?:a|the)\s+(?:new\s+)?(?:position|role|offer))\b",
    re.I,
)
_FIRE_RE = re.compile(
    r"\b(fired|terminat(?:ed|ion)|discharg(?:ed|e)|dismiss(?:ed|al)|"
    r"let\s+go|decided\s+to\s+terminate)\b",
    re.I,
)

_DATE_WINDOW = 250  # chars: how far a calendar date may sit from the verb


def _is_none_answer(s):
    if not s:
        return True
    return s.strip().lower() in _NONE_ANSWERS


def _surname(name):
    if not name or _is_none_answer(name):
        return ""
    n = _TITLE_RE.sub("", name.strip())
    n = _POSSESSIVE_RE.sub("", n)
    parts = re.findall(r"[A-Za-z][A-Za-z\-]*", n)
    return parts[-1].lower() if parts else ""


def _find_dates(text):
    """(position, month-value) pairs, month-value = year*12 + (month-1) [+ day frac]."""
    dates = []
    full_spans = []
    for m in _DATE_FULL_RE.finditer(text):
        full_spans.append((m.start(), m.end()))
        mon = _MONTHS.get(m.group(1).lower())
        year = int(m.group(3))
        if mon and 1000 < year < 2100:
            day = int(m.group(2))
            dates.append((m.start(), year * 12 + (mon - 1) + (day - 1) / 30.0))
    for m in _DATE_MY_RE.finditer(text):
        s, e = m.start(), m.end()
        if any(fs <= s < fe for fs, fe in full_spans):
            continue
        mon = _MONTHS.get(m.group(1).lower())
        year = int(m.group(2))
        if mon and 1000 < year < 2100:
            dates.append((s, year * 12 + (mon - 1) + 0.5))
    dates.sort(key=lambda x: x[0])
    return dates


def _nearest_date_value(dates, pos, window=_DATE_WINDOW):
    best, best_dist = None, window + 1
    for dpos, dval in dates:
        dist = abs(dpos - pos)
        if dist <= window and dist < best_dist:
            best_dist, best = dist, dval
    return best


def _dated_verb_values(regex, text, dates, window=_DATE_WINDOW):
    """Date values attached (within `window` chars) to each verb occurrence.
    A verb keyword can recur several times (a disciplinary history, a
    generic summary sentence at the end, ...) — only the occurrences that
    sit next to an actual calendar date are usable evidence."""
    vals = []
    for m in regex.finditer(text):
        v = _nearest_date_value(dates, m.start(), window)
        if v is not None:
            vals.append(v)
    return vals


def _parse_llm_duration(ans):
    """Parse an LLM free-text elapsed-time answer (e.g. '3 months', '8 years',
    '2 weeks') into months; return None if absent/unstated/unparseable."""
    if not ans:
        return None
    s = str(ans).strip()
    if not s or s.upper() in ("UNKNOWN", "NONE", "N/A", "NA", "UNCLEAR",
                               "NOT STATED", "UNSTATED"):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?", s, re.I)
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("day"):
        return n / 30.0
    if unit.startswith("week"):
        return n / 4.3
    if unit.startswith("month"):
        return n
    if unit.startswith("year"):
        return n * 12.0
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if text else ""
        if not t:
            return 0.0
        ext = extracted or {}

        # --- code layer 1: baseline backstop (explicit surface patterns) ---
        same_actor_phrases = len(re.findall(
            r"same (person|individual|manager|supervisor|actor|decision[- ]?maker)",
            t, re.I))
        hire_names = set(re.findall(
            r"([A-Z][a-z]+)\s+(?:hired|promoted)\s+(?:her|him|the plaintiff)", t))
        hire_names |= set(re.findall(r"hired\s+by\s+([A-Z][a-z]+)", t))
        fire_names = set(re.findall(
            r"([A-Z][a-z]+)\s+(?:fired|terminated|discharged)\s+(?:her|him|the plaintiff)", t))
        fire_names |= set(re.findall(r"(?:fired|terminated|discharged)\s+by\s+([A-Z][a-z]+)", t))
        regex_same = 1.0 if (hire_names & fire_names) else 0.0
        phrase_same = 1.0 if same_actor_phrases > 0 else 0.0

        # --- LLM layer: thick-input actor identity grounding ---
        hire_sur = _surname(ext.get("hire_actor", ""))
        fire_sur = _surname(ext.get("fire_actor", ""))
        llm_same = (1.0 if hire_sur == fire_sur else 0.0) if (hire_sur and fire_sur) else None

        # code layer 2: the actual predicate — same actor, evidence union
        same_actor_signal = max(
            llm_same if llm_same is not None else 0.0,
            regex_same,
            phrase_same,
        )

        # --- b4: supervisor continuity discounts a name-match same-actor
        # signal when the narrative documents a mid-tenure supervisor
        # change (guards against surname collisions / partial-tenure
        # supervision reading as "same actor"). No discount if the field is
        # absent, so a caller without this evidence gets the h0 behavior.
        cont = (ext.get("supervisor_continuity") or "").strip().upper()
        if cont.startswith("CHANGED"):
            same_actor_signal *= 0.6

        # --- code layer 3: real date arithmetic for the elapsed-time half ---
        # A hire/fire keyword can recur (disciplinary history, a generic
        # closing summary, ...); only occurrences with an actual calendar
        # date nearby are usable. Take the EARLIEST dated hire occurrence
        # (the original hire/promotion) and the LATEST dated fire
        # occurrence (the final termination) as the two endpoints.
        dates = _find_dates(t)
        hire_vals = _dated_verb_values(_HIRE_RE, t, dates)
        fire_vals = _dated_verb_values(_FIRE_RE, t, dates)
        hire_val = min(hire_vals) if hire_vals else None
        fire_val = max(fire_vals) if fire_vals else None

        gap_months = None
        if hire_val is not None and fire_val is not None:
            diff = fire_val - hire_val
            if 0 < diff <= 600:
                gap_months = diff

        # --- b4: LLM-read narrative duration as a fallback when no
        # calendar-anchored hire/fire verb pair was found — covers
        # narratives that state elapsed time in prose ("eight years") with
        # no exact date near the verb, where code layer 3 has no evidence.
        if gap_months is None:
            llm_gap = _parse_llm_duration(ext.get("tenure_duration"))
            if llm_gap is not None:
                gap_months = llm_gap

        gap_score = 0.15 if gap_months is None else max(0.0, 1.0 - gap_months / 48.0)

        # same-actor gates the score; a short real gap amplifies it toward 1.0
        val = same_actor_signal * (0.4 + 0.6 * gap_score)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5

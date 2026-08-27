"""a3 hybrid: Inference of discrimination circumstances (McDonnell Douglas
4th prong catch-all).

Criterion asks whether the pleaded FACTS supply circumstances raising an
inference the adverse action was because of the protected trait: (1) a named
comparator hired/promoted/retained instead of plaintiff who is outside the
protected class, (2) a statistical/pattern disparity disfavoring the group,
(3) a deviation from stated policy, (4) a decisionmaker statement referencing
the trait, or (5) suspicious timing between disclosure of the trait/protected
activity and the adverse action.

Baseline (v0_keyword) scans for a fixed phrase list ('similarly situated',
'pretext', 'shortly after', ...). Train inspection shows two failure modes:
  - FALSE POSITIVES: cases that use the buzz-phrase as a bare, uncorroborated
    plaintiff allegation ("treated differently than similarly situated
    Caucasian employees") score high on keywords even when the surrounding
    facts undercut it (e.g. no replacement was ever hired for the vacated
    role) -- judge scored this well below the keyword hit rate.
  - FALSE NEGATIVES: cases with very strong, concrete circumstances score 0
    under keyword-matching because the narrative never uses the baseline's
    exact phrases -- e.g. an explicit "I don't want people over 40 in the
    lab" statement, continued accommodation of comparable pregnant coworkers,
    a named younger/less-qualified hire, or bare hiring-count statistics
    ("148 white ... no black ... drivers"). None of these trip 'similarly
    situated' / 'pretext' / 'shortly after'.

The recurring trap (per corpus notes): legal-term PRESENCE is a weak proxy
for the construct being FACTUALLY present. Also, the boilerplate framing
sentence present in nearly every complaint ("plaintiff alleges X was done
because of trait Y") is the plaintiff's bare legal conclusion, not
"surrounding circumstances" -- so naive trait-near-adverse-action regex would
fire on almost every document regardless of judge score and was deliberately
NOT used here.

Design: keep the baseline keyword scan as a modest code-side backstop (it
still catches clean explicit-term cases), add a code-only numeric/statistical
disparity detector (two distinct count+group-noun mentions -- genuinely
regex-legible and rare enough not to trip on boilerplate), and route the two
constructs that need real reading comprehension -- (a) concrete comparator /
pattern / policy-deviation / statement evidence, and (b) which two events are
the relevant trait-disclosure -> adverse-action pair -- to LLM fields. Code
still owns the predicate: it parses the LLM's short answer for concrete
detail (digits/names) and for the timing field's quantity+unit, converting
that quantity into the proximity score itself.
"""
import re

LLM_FIELDS = {
    "circumstance_evidence": (
        "State any concrete comparator (named person hired/promoted/retained "
        "instead of plaintiff, outside the protected class), statistical "
        "hiring/promotion pattern disfavoring the group, deviation from "
        "stated policy, or decisionmaker statement referencing the protected "
        "trait; else NONE."
    ),
    "timing_gap": (
        "State the time gap between the protected-trait disclosure/complaint "
        "and the adverse action (e.g. '2 weeks', '6 months'); else NONE."
    ),
}

# --- code-side backstop: baseline's explicit legal-phrase signal (kept, down-weighted) ---
_BASELINE_KEYWORDS = [
    'replaced by', 'position remained open', 'position stayed open',
    'filled by', 'similarly situated', 'comparator', 'treated differently',
    'treated less favorably', 'shortly after', 'deviation from policy',
    'contrary to policy', 'outside the protected class', 'no legitimate reason',
    'pretext',
]

# --- code-only statistical/pattern detector: two distinct count+group mentions ---
_GROUP_TERM = (
    r'(?:white|black|african[- ]american|caucasian|hispanic|latino|latina|'
    r'asian|male|female|men|women|older|younger|disabled|non-?disabled|'
    r'pregnant|christian|muslim|jewish|catholic)'
)
_NUM_GROUP_RE = re.compile(r'\b(\d+)\s+(?:\w+\s+){0,3}' + _GROUP_TERM, re.IGNORECASE)

# --- code-side quantity parsing applied to the LLM's short timing answer ---
_TIME_UNIT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(day|week|month|year)s?', re.IGNORECASE)
_IMMEDIATE_RE = re.compile(r'\b(immediately|same day|next day|within (?:a )?day)\b', re.IGNORECASE)

_EMPTY_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _clean_field(extracted, key):
    try:
        v = (extracted.get(key, "") or "").strip()
    except Exception:
        return ""
    if v.lower() in _EMPTY_VALUES:
        return ""
    return v


def _keyword_score(t_lower):
    hits = sum(1 for k in _BASELINE_KEYWORDS if k in t_lower)
    return min(1.0, hits / 3.0)


def _stat_disparity_score(t):
    matches = _NUM_GROUP_RE.findall(t)
    distinct = set(matches)
    if len(matches) >= 2:
        return 1.0 if len(distinct) >= 2 else 0.6
    if len(matches) == 1:
        return 0.3
    return 0.0


def _comparator_score(val):
    if not val:
        return 0.0
    has_digit = bool(re.search(r'\d', val))
    has_name = bool(re.search(r'\b[A-Z][a-z]+\b', val))
    return 1.0 if (has_digit or has_name) else 0.6


def _timing_score(val):
    if not val:
        return 0.0
    if _IMMEDIATE_RE.search(val):
        return 1.0
    m = _TIME_UNIT_RE.search(val)
    if not m:
        return 0.4  # qualitative timing mentioned ("shortly", "soon") but no parseable quantity
    qty = float(m.group(1))
    unit = m.group(2).lower()
    days = qty * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    if days <= 14:
        return 1.0
    if days <= 90:
        return 0.8
    if days <= 365:
        return 0.4
    return 0.15


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""
        t_lower = t.lower()

        kw = _keyword_score(t_lower)
        stat = _stat_disparity_score(t)

        ext = extracted or {}
        comp_val = _clean_field(ext, "circumstance_evidence")
        comp = _comparator_score(comp_val)

        time_val = _clean_field(ext, "timing_gap")
        timing = _timing_score(time_val)

        combined = 0.15 * kw + 0.20 * stat + 0.40 * comp + 0.25 * timing
        return max(0.0, min(1.0, combined))
    except Exception:
        return 0.5

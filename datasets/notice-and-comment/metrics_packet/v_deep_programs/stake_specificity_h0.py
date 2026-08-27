"""stake_specificity hybrid: Specificity of who is affected and how.

Construct: ~1.0 = comment names a specific, sizeable constituency ("representing 4,200
members"), grounds it geographically (named state/county), AND quantifies the impact
("would cost us $50,000/year"); ~0.5 = one of those elements present; ~0.0 = generic "this
affects people" with no membership count, no geography, no quantified impact.

INPUT = comment text. Code sees: membership-count patterns ("representing N members", "our
N employees"), geographic-specificity (named US states/counties), and quantified-impact
statements (dollar/number figures co-occurring with impact vocabulary). Code CANNOT verify
the claimed membership/impact numbers are accurate (needs an external roster) — out of
scope for h0.
"""
import re

LLM_FIELDS = {
    "who_affected": (
        "In <=20 words, who specifically the comment says is affected by the rule (e.g. "
        "'small family farms in Iowa', 'our 400 hourly warehouse employees'). Answer NONE "
        "if the comment names no specific affected group."
    ),
    "how_affected": (
        "In <=20 words, the specific quantified impact the comment states (e.g. 'would cost "
        "us $50,000 per year', 'affects roughly 12,000 patients'). Answer NONE if no "
        "quantified impact is stated."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_MEMBER_RE = re.compile(
    r'\b(?:representing|on behalf of|our)\s+(?:over\s+|more than\s+|approximately\s+|~\s*)?'
    r'([\d,]{1,9})\s+(members?|member (?:institutions|companies|organizations)|employees|'
    r'households|families|patients|farmers|businesses|workers|residents)\b', re.I)

_STATES = (
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|"
    "georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|"
    "massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|"
    "new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|"
    "oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|"
    "texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming"
)
_GEO_RE = re.compile(
    r'\b(' + _STATES + r')\b|\b[A-Z][a-z]+ County\b', re.I)

_MONEY_RE = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)\s*(thousand|million|billion|k|m|b)?\b', re.I)
_IMPACT_VOCAB_RE = re.compile(
    r'\b(cost|lose|losses|save|savings|affect[s]?|jobs|revenue|income|burden|impact[s]?)\b', re.I)
_NUM_RE = re.compile(r'\b[\d,]{2,9}\b')


def _quant_impact_count(t, window=50):
    money_spans = [m.span() for m in _MONEY_RE.finditer(t)]
    num_spans = [m.span() for m in _NUM_RE.finditer(t)]
    impact_spans = [m.span() for m in _IMPACT_VOCAB_RE.finditer(t)]
    all_num_spans = money_spans + num_spans
    if not all_num_spans or not impact_spans:
        return 0
    n = 0
    for n0, n1 in all_num_spans:
        if any(not (i1 < n0 - window or i0 > n1 + window) for i0, i1 in impact_spans):
            n += 1
    return n


def _code_score(t):
    member_matches = _MEMBER_RE.findall(t)
    n_member = len(member_matches)
    geo_hits = set(m.lower() for m in _GEO_RE.findall(t) if isinstance(m, str))
    n_geo = len(geo_hits)
    n_quant = _quant_impact_count(t)

    member_part = min(0.30, 0.15 * n_member)
    geo_part = min(0.30, 0.15 * n_geo)
    quant_part = min(0.40, 0.15 * n_quant)
    return max(0.0, min(1.0, member_part + geo_part + quant_part))


def _llm_score(extracted):
    who = extracted.get("who_affected")
    has_who = isinstance(who, str) and who.strip().lower().strip(". ") not in _NONE
    who_specific = has_who and bool(re.search(r'\d', who))
    who_part = 0.45 if who_specific else (0.25 if has_who else 0.05)

    how = extracted.get("how_affected")
    has_how = isinstance(how, str) and how.strip().lower().strip(". ") not in _NONE
    how_specific = has_how and bool(re.search(r'\d', how))
    how_part = 0.45 if how_specific else (0.20 if has_how else 0.05)

    return max(0.0, min(1.0, who_part + how_part))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

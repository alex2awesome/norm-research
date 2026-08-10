"""Hybrid channel for a115: "Specific, meaningful metrics and performance context".

Construct = concrete, credible numbers tied to MEANINGFUL KPIs, with performance
CONTEXT (comparisons / drivers) — NOT vanity or bare spec-dump numbers.

Design principle (decoupled presence-vs-quality): the score is driven by the
CO-PRESENCE of (a) specific quantified metrics and (b) comparative/driver context.
A doc that merely name-drops "$1 trillion / 85,000 employees" with no comparison
(judge=0.0, d04471) must NOT score high; a doc rich in "up 36% YoY / compared to /
from X to Y" around real numbers (judge=1.0, Airbnb/AAL/Gartner) must. This also
resists a later counterfactual keyword-injection gate: a bare KPI word without an
adjacent number, and a bare number without context, both stay low.

Code-only fallback works with extracted={}; the optional LLM field only nudges.
"""
import re
import math

LLM_FIELDS = {
    "top_metric": "Quote the single most substantive PERFORMANCE metric together with its comparison/context (e.g. 'revenue up 36% year-over-year', 'shipments down 8.6% vs a year earlier'); answer NONE if there is no meaningful, contextualized performance metric.",
}

# --- context vocabularies -------------------------------------------------
_COMPARE = [
    "compared to", "compared with", "versus", " vs ", " vs.", "year-over-year",
    "year over year", "year-on-year", "year on year", "prior year", "prior quarter",
    "previous quarter", "previous year", "up from", "down from", "same period",
    "a year earlier", "a year ago", "year-ago", "year ago", "from a year",
    "quarter to quarter", "quarter-to-quarter", "two-year", "two year",
    "sequential", "than the", "than a", "relative to",
]
_DRIVER = [
    "driven by", "attributed to", "reflects", "reflecting", "reflected", "due to",
    "because of", "as a result of", "fueled by", "spurred by", "resulted in",
    "led by", "on the back of", "thanks to",
]
_SUPERL = [
    "highest", "lowest", "record", "all-time", "all time", "coldest", "hottest",
    "hardiest", "largest", "smallest", "fastest", "strongest", "best ever",
    "most profitable", "biggest", "surpassed", "milestone",
]
_KPI = [
    "revenue", "net income", "profit", "ebitda", "margin", "earnings",
    "shipments", "market share", "sales", "expense ratio", "assets under management",
    "adjusted operating income", "per share", "per diluted share", "gross sales",
    "net flows", "production", "employment", "output", "shipment", "units shipped",
    "guest arrivals", "host earnings", "fee revenue",
]
_VANITY = [
    "impressions", "page views", "pageviews", "social reach", "follower count",
    "followers", "awards won", "mentions", "likes", "retweets", "hashtag",
]
# spec-dump markers (dimension tables etc. — presence of numbers but not KPIs)
_SPEC = ["tine", "linkage", " lb ", " kg ", " mm ", " in ", " oz ", "spacing"]

_ES = [" el ", " la ", " los ", " las ", " para ", " que ", " con ", " una ",
       " del ", " por ", " este ", " sus ", " como ", " mas ", " sobre ", " son "]
_EN = [" the ", " and ", " of ", " to ", " in ", " for ", " with ", " is ",
       " that ", " on ", " as ", " by ", " from "]


def _norm01(x, cap):
    return min(1.0, x / float(cap)) if cap > 0 else 0.0


def score(text, extracted, ops):
    try:
        t = ops.normalize(text or "")
        low = t.lower()
        if not low.strip():
            return 0.0

        # ---- foreign-language damp (Spanish prospectus / localized pages) ----
        es = sum(low.count(w) for w in _ES)
        en = sum(low.count(w) for w in _EN)
        foreign = (es > 12 and es > en * 0.9)

        # ---- specific, quantified metrics (number + meaningful unit) ----
        pct = re.findall(r"\d[\d,\.]*\s?(?:%|percent)", low)
        dollars = re.findall(r"\$\s?\d[\d,\.]*\s?(?:billion|million|thousand|trillion|bn|k)?", low)
        unitnum = re.findall(
            r"\d[\d,\.]*\s?(?:billion|million|thousand|trillion|degrees?|units?|"
            r"employees|associates|countries|customers|shares|jobs|patients|"
            r"nautical\s+miles|miles|tons?|mt|basis\s+points|users|subscribers)",
            low,
        )
        metrics = len(pct) + len(dollars) + len(unitnum)
        spec_norm = _norm01(metrics, 6.0)

        # ---- performance context (comparisons / drivers / change verbs) ----
        change = len(re.findall(
            r"(?:up|down|grew|rose|fell|declined?|increased?|decreased?|gained|"
            r"dropped|rise|fall|grow(?:th)?|higher|lower|jumped|surged)\s+(?:by\s+)?"
            r"\$?\d", low))
        change += len(re.findall(
            r"\d[\d,\.]*\s?(?:%|percent)\s*(?:increase|decrease|decline|growth|"
            r"higher|lower|gain|drop|up|down|more|less)", low))
        fromto = len(re.findall(r"from\s+\$?\d[\d,\.]*.{0,30}?\bto\s+\$?\d", low))
        compare = sum(low.count(k) for k in _COMPARE)
        driver = sum(low.count(k) for k in _DRIVER)
        superl = sum(low.count(k) for k in _SUPERL)
        ctx_raw = 1.5 * change + 2.0 * fromto + 1.0 * compare + 1.0 * driver + 0.35 * superl
        ctx_norm = _norm01(ctx_raw, 5.0)

        kpi = sum(low.count(k) for k in _KPI)
        kpi_norm = _norm01(kpi, 3.0)
        vanity = sum(low.count(k) for k in _VANITY)
        spec_dump = sum(low.count(k) for k in _SPEC)

        # ---- combine: co-presence of specificity AND context is the construct ----
        copresence = math.sqrt(spec_norm * ctx_norm)  # 0 if either missing
        # metric floor kept small: numbers ALONE (no context) must stay low (d04471)
        core = 0.72 * copresence + 0.10 * spec_norm + 0.18 * (kpi_norm * spec_norm)

        # vanity / spec-dump damping (numbers present but not meaningful KPIs)
        if metrics == 0 and spec_dump >= 6:
            core *= 0.5
        core -= 0.04 * min(vanity, 3)

        if foreign:
            core *= 0.30

        core = max(0.0, min(1.0, core))

        # ---- optional LLM thick-input adjustment (absent in code-only mode) ----
        if "top_metric" in extracted:
            em = (extracted.get("top_metric") or "").strip().lower()
            if (not em) or em in ("none", "n/a", "na", "no", "-"):
                core *= 0.55  # extractor found no contextualized metric
            else:
                has_digit = bool(re.search(r"\d", em))
                has_ctx = any(w in em for w in (
                    "%", "percent", " up ", " down ", "compared", "vs", "from ",
                    "year", "growth", "increase", "decrease", "higher", "lower",
                    "record", "than", "billion", "million"))
                boost = 0.18 * has_digit + 0.18 * has_ctx
                core = core + boost * (1.0 - core)

        return max(0.0, min(1.0, core))
    except Exception:
        return 0.5

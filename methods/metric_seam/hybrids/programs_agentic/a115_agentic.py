"""Hybrid channel for a115: "Specific, meaningful metrics and performance context"
(agentic v1, iterating on programs_v2/a115_h0.py).

Construct (from judge behavior on TRAIN, confirmed by full-train bucket checks,
not just the worst residuals): concrete, credible numbers tied to MEANINGFUL KPIs,
with performance CONTEXT (comparisons / drivers) scores highest; rich quantification
ALONE (no context) gets graded partial credit, not zero (Kleindienst "Floating
Venice" launch, judge=0.55, no YoY-style comparison but many concrete scale figures);
survey-style quantifier comparisons ("almost half", "more than three-quarters")
are just as much "context" as "compared to last year" (Aon Hewitt survey release,
judge=0.75) but h0's _COMPARE lexicon didn't recognize that phrasing register;
release-INDEX/listing pages (multiple "Read more" stubs) systematically score near
zero regardless of any numbers embedded in the stubs (a fleet-wide convention,
validated here too: mean judge 0.08 for readmore>=2 vs 0.39 for the rest, on the
FULL 150-item train set, not a residual-only fit).

Code-only fallback works with extracted={}; the optional LLM field only nudges,
and is now GATED by code-side evidence so a single extracted quote (which may
describe an industry-wide statistic rather than the entity's own KPI) cannot by
itself carry a zero-evidence document to a high score.

Changes over h0, each validated against the FULL 150-item train set (not just
the residuals that motivated them):
  1. NEW aggregator/listing-page gate (readmore>=2): mean judge 0.079 vs 0.389
     for the rest (n=7 vs 141) -- reuses the pattern already established for
     a119 on this same corpus. Fixes Fastenal IR release-list index (d01793,
     judge=0.0, h0 predicted 0.27 from the dollar-amount file-size/price noise
     scattered across the release stubs).
  2. EXTENDED _COMPARE lexicon with quantifier-comparison phrasing (survey
     register): "almost/nearly/more than half", "one/two/three-quarters",
     "of respondents", "survey found", etc. Validated: mean judge 0.564 for
     docs containing >=1 such phrase vs 0.322 for the rest (full train, not
     residual-only). Fixes the Aon Hewitt benefits survey (d00850, judge=0.75,
     h0 predicted 0.055) whose percentages ("almost half (49 percent)", "more
     than three-quarters (76 percent)") are real comparative context that h0's
     narrower "compared to / versus / year-over-year" vocabulary missed.
  3. Modest EXTENSION of the unit-noun list (guests, cabins, decks, vessels,
     rooms, aircraft, vehicles, acres, stores, locations, beds, flights,
     passengers, km, meters, feet) -- checked this barely moves the raw
     unit-count vs judge correlation on the full train (rho 0.601 -> 0.602,
     i.e. non-destructive) while recovering signal on documents whose scale
     figures aren't phrased in business-report units (Kleindienst Floating
     Venice launch: 0 -> 5 unit hits).
  4. LLM boost is now EVIDENCE-GATED: the boost from `top_metric` is scaled
     down when the code side found near-zero quantification, so a single
     extracted industry-wide statistic (ExxonMobil "global demand...rising by
     25 percent from 2014 to 2040" -- an industry projection, not the
     company's own performance) can no longer alone carry an otherwise-empty
     document to a mid-range score the way `core + boost*(1-core)` did in h0
     (that formula gives boost its FULL effect precisely when core is 0,
     backwards from "grounding, not driving").
  TRIED AND NOT KEPT: a whole-document "prose fraction" / stopword-density
  chrome gate (the natural generalization of the readmore gate). It looked
  promising on the nav-heavy false positives (Apple stock-ticker chrome,
  ExxonMobil topic-nav, CME nav) but the Aon Hewitt survey release (judge=0.75)
  is ALSO majority nav-chrome by word count (a huge insurance-services mega-menu
  precedes the actual release) and would have been damped just as hard as the
  true chrome negatives -- same trap the a119 run hit with its "fraction of
  lines that are prose" op. Reverted before it ever reached the candidate file.
  Also tested and NOT kept: a "wire-service marker present + zero metrics ->
  small floor" rule to catch a handful of thin-but-genuine releases (Lockheed
  missile test, NTWRK event, Stanford research news, all judge 0.15-0.20 with
  zero detected metrics). Full-train check: docs with (wire marker present,
  zero metrics, not aggregator) have mean judge 0.094, statistically
  indistinguishable from (no wire marker, zero metrics) at 0.084 -- most
  wire-marked zero-metric docs in TRAIN are legitimately judge=0 (litigation/
  policy releases with no data at all), so this would have been a residual-
  only overfit dressed up as a "structural" rule. Left unfixed; see final note.

  Round 2/3 (plateau, both reverted after a full-train rho check): raising the
  metrics-alone weight (0.10->0.16 of `core`) to lift standalone-quantification
  docs like Kleindienst dropped full-train rho (0.869->0.868); re-deriving the
  LLM-boost evidence_gate from post-copresence `core` instead of `spec_norm`
  dropped it further (0.869->0.864, and didn't even fix its target case,
  Apple stock-ticker chrome, since that page's CODE-side evidence is already
  high from genuine-looking regex hits on real digits -- the boost was never
  the main driver of that overshoot). Both reverted; final program is the
  round-1 candidate below.

  Remaining top residual (Apple/AMD/NBC-Sports/ProFootballTalk-style news/blog
  aggregator homepages with a stock ticker, video-timestamp list, or "most
  commented" headline+vote-count widget): tried four more full-train-validated
  structural discriminators to gate these out -- whole-doc stopword/prose
  fraction, bare parenthetical-integer count, "headline text immediately
  followed by (count)" repeated pattern, and mean-words-per-regex-sentence
  (chrome vs prose) -- and a dateline-or-wire-marker requirement. None
  separated them from genuine releases without also damping genuine ones that
  share the same surface property (Aon Hewitt's mega-menu preamble, Airbnb's
  no-dateline blog-style release, Morningstar/d00520's footnote-numbered
  citations). The actual distinguishing fact -- "is this describing the
  page's own entity's performance, or reporting on someone else's stock price/
  pop-culture headlines" -- is semantic, not structural, and the single
  available LLM field is already spent on `top_metric`; this is the
  irreducibly field-dominated part of the residual.
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
    # quantifier-comparison / survey register (validated: full-train mean
    # judge 0.564 for docs with >=1 hit vs 0.322 for the rest) -- this is
    # exactly as much "performance context" as a YoY phrase, just framed as
    # a population breakdown rather than a trend.
    "almost half", "nearly half", "more than half", "less than half",
    "one in ", "out of ", "of respondents", "of those surveyed",
    "survey found", "survey of", "three-quarters", "two-thirds", "one-third",
    "majority of", "percent of respondents", "percent of those",
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

_READMORE = re.compile(r"read\s*more", re.I)


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
            r"nautical\s+miles|miles|tons?|mt|basis\s+points|users|subscribers|"
            r"guests|cabins|decks|vessels|rooms|aircraft|vehicles|acres|stores|"
            r"locations|beds|flights|passengers|km|meters|feet)",
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

        # ---- aggregator/listing-page gate (validated on all 150 train items:
        # mean judge 0.079 for readmore>=2 vs 0.389 otherwise): >=2 "read more"
        # stubs means this page indexes several releases rather than being one,
        # regardless of numbers scattered across the stub previews. ----
        readmore = 0
        for _ in _READMORE.finditer(t):
            readmore += 1
            if readmore >= 3:
                break
        if readmore >= 2:
            core *= 0.30

        # ---- optional LLM thick-input adjustment (absent in code-only mode) ----
        # Evidence-gated: the boost's effective strength scales with how much
        # code-side quantification was already found, so one extracted quote
        # (which may be an industry-wide stat, not the entity's own KPI)
        # cannot alone carry a near-zero-evidence document to a mid score.
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
                evidence_gate = min(1.0, 0.35 + 0.65 * spec_norm)
                core = core + boost * evidence_gate * (1.0 - core)

        return max(0.0, min(1.0, core))
    except Exception:
        return 0.5

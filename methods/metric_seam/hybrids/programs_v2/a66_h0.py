"""a66 hybrid — "Support claims with relevant data and sourcing".

Judged construct: assertions backed by pertinent, CONTEXTUALIZED statistics
(percentages, money figures, baseline comparisons), cited sources/attribution,
and methodological detail.

Design (decoupling presence from quality):
  * Core predicate is CO-OCCURRENCE within one sentence: a number together
    with a baseline-comparison term ("up 14% from", "compared to", "declined")
    or an attribution term ("according to", "said", "estimates"). Bare keyword
    or bare number presence contributes only via low-weight channels, so
    injecting keywords without genuinely supported claims moves the score
    little.
  * Document-level data-mark densities (%, $, comma-grouped figures) are
    counted over deduplicated text so tabular data (portfolio/shipment
    tables) is credited, while repeated About-boilerplate is not
    double-counted.
  * Unsourced magnitude brags ("94,000 miles of cable") get half credit;
    author-year citations "(Smith, 1994)" count as sourcing.
  * Non-content pages (nav chrome) are damped via prose-sentence count, NOT
    via press-release-likeness — the judge rewards data-rich news/analysis
    articles on this criterion too.
"""
import re

LLM_FIELDS = {
    "key_stats": "List up to 3 statistics or figures (with units) this document uses to support its claims, or NONE.",
    "data_source": "Name the cited source of the document's data (organization, study, report, or named expert), or NONE.",
}

# ---------------------------------------------------------------- lexicons
# number token: not part of a hyphenated/alphanumeric code (COVID-19, 3D, 4K)
_NUM = re.compile(r"(?<![\w-])\d[\d,.]*(?!\w)")
_PCT = re.compile(r"\d[\d,.]*\s*(?:%|percent\b|per cent\b)", re.I)
_DOLLAR = re.compile(r"[$£€]\s?\d")
_DOLLAR_MAG = re.compile(r"[$£€]\s?\d[\d,.]*\s*(?:billion|million|trillion)", re.I)
_MAG = re.compile(r"\b\d[\d,.]*\s*(?:billion|million|trillion)\b", re.I)
_COMMA_NUM = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_URL = re.compile(r"(?:https?://|www\.)[^\s\"'<>]{4,}", re.I)
_CITE = re.compile(r"\([^()]{0,60}?(?:19|20)\d{2}\)")     # (Smith & Jones, 1994)
_METH = re.compile(
    r"methodolog|sample (?:size|of \d)|\bn\s*=\s*\d|surveyed\s+[\d,]+|"
    r"margin of error|preliminary (?:results|data|estimates)|poll of|"
    r"census data|regression|confidence interval|testing conducted|"
    r"tests? conducted|\bbenchmarks?\b", re.I)

# Baseline comparison: a statistic contextualized against a prior/other value.
_CMP_RE = re.compile(
    r"\bup from\b|\bdown from\b|\b(?:increas|decreas|declin)\w*|\bgrew\b|"
    r"\bgrowth of\b|\brose\b|\bfell\b|\bjumped\b|\bdropped\b|\bsurpass\w*|"
    r"\boutperform\w*|\bcompared (?:to|with)\b|\bcomparison\b|"
    r"\byear[- ]over[- ]year\b|\ba year earlier\b|\byear earlier\b|"
    r"\bsame period\b|\bprior period\b|\bprevious (?:quarter|year)\b|"
    r"\bversus\b|\bvs\.|\bup by\b|\bdown by\b", re.I)
# Attribution / sourcing language. Word-bounded ('citing' must not hit
# 'exciting') and lowercase-anchored (Title Case nav headers like "Reports"
# or "Crime Statistics" must not fire).
_ATTR_RE = re.compile(
    r"\b[Aa]ccording to\b|\bsaid\b|\bsays\b|\breport(?:ed|s)?\b|\bannounced\b|"
    r"\bestimat(?:e|es|ed)\b|\bsurvey(?:ed|s)?\b|\bstud(?:y|ies)\b|"
    r"\bresearchers\b|\banalysts?\b|\bdata (?:from|show)\b|\bstatistics\b|"
    r"\bcited\b|\bciting\b|\bfigures from\b|\b[Ss]ource:|\bhe added\b|"
    r"\bshe added\b")
# Quoted speech with a speaker — press-quote sourcing.
_QUOTE_RE = re.compile(r"\",?\s*(?:said|says)\b|\bsaid\s+[A-Z][a-z]+|\bsays\s+[A-Z][a-z]+|\bsaid:\s")
# Company-scale brag: number + scale noun, the classic unsourced About-blurb
# ("24 million customers, 85,000 employees, $1 trillion in assets").
_BRAG_RE = re.compile(
    r"\d[\d,.]*(?:\s*(?:billion|million|trillion))?[^.\n]{0,50}?"
    r"\b(?:employees|associates|customers|clients|members|followers|"
    r"subscribers|engineers|staff)\b", re.I)


def _sat(x, k):
    return x / (x + k) if x > 0 else 0.0


def _segments(t):
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    return [p.strip() for p in parts if p.strip()]


def _dedupe(segs):
    """Drop exact repeats of substantial segments (About-boilerplate guard)."""
    seen, out = set(), []
    for s in segs:
        key = s.lower()
        if len(s) >= 20 and key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _prose_sentences(segs):
    out = []
    for s in segs:
        words = re.findall(r"[A-Za-z']+", s)
        if len(words) >= 8 and any(w[0].islower() for w in words):
            out.append(s)
    return out


def _code_score(text, ops):
    t = ops.normalize(text)
    if not t.strip():
        return 0.0

    segs = _dedupe(_segments(t))
    t = "\n".join(segs)
    prose = _prose_sentences(segs)

    # --- sentence-level co-occurrence predicates -------------------------
    ctx_strong = 0      # number + baseline comparison or attribution
    attr_sents = 0      # sourcing/attribution language present
    for s in prose:
        has_num = bool(_NUM.search(s))
        has_cmp = bool(_CMP_RE.search(s))
        has_attr = bool(_ATTR_RE.search(s))
        if has_num and (has_cmp or has_attr):
            ctx_strong += 1
        if has_attr:
            attr_sents += 1
    attr_sents += min(2, len(_CITE.findall(t)))   # author-year citations
    n_quote = len(_QUOTE_RE.findall(t))           # quoted speech w/ speaker

    # --- document-level data-mark densities (tables included) ------------
    # Exclude company-scale brags (unsourced About-blurb figures) unless the
    # segment actually contextualizes them against a baseline.
    tc = "\n".join(s for s in segs
                   if not (_BRAG_RE.search(s) and not _CMP_RE.search(s)))
    n_pct = len(_PCT.findall(tc))
    n_dollar = len(_DOLLAR.findall(tc))
    n_both = len(_DOLLAR_MAG.findall(tc))
    n_mag = len(_MAG.findall(tc))
    money_eff = n_dollar + 0.5 * max(0, n_mag - n_both)   # bare magnitude = half credit
    n_comma = len(_COMMA_NUM.findall(tc))
    n_url = len(_URL.findall(t))
    n_meth = len(_METH.findall(t))
    n_nums = len(_NUM.findall(tc))

    raw = (
        0.26 * _sat(ctx_strong, 2.5)
        + 0.20 * _sat(n_pct, 3.0)
        + 0.16 * _sat(money_eff, 2.5)
        + 0.10 * _sat(n_comma, 4.0)
        + 0.14 * _sat(attr_sents, 2.5)
        + 0.06 * _sat(n_meth, 1.0)
        + 0.04 * _sat(n_url, 2.0)
        + 0.04 * (1.0 if (ctx_strong >= 1 and n_pct + n_dollar >= 1) else 0.0)
        + 0.05 * _sat(n_nums, 40.0)          # raw data density (tables, lists)
        + 0.03 * _sat(len(prose), 12.0)      # doc makes prose claims at all
        + 0.04 * _sat(n_quote, 1.5)          # attributed press quotes
    )

    # --- content damping (chrome pages make no claims to support) --------
    if len(prose) < 3:
        raw *= 0.4

    return raw


def score(text, extracted, ops):
    raw = _code_score(text, ops)

    # --- LLM thick-input adjustments (bounded; code-only must dominate) --
    if "key_stats" in extracted:
        v = (extracted.get("key_stats") or "").strip()
        if v and v.lower() not in ("none", "n/a", "no", "-"):
            n_items = min(3, max(1, len(re.findall(r"\d[\d,.]*", v))))
            raw += 0.03 * n_items          # confirmed supporting stats
        else:
            raw *= 0.7                     # extractor found no supporting data
    if "data_source" in extracted:
        v = (extracted.get("data_source") or "").strip()
        if v and v.lower() not in ("none", "n/a", "no", "-"):
            raw += 0.05                    # confirmed cited source
        else:
            raw *= 0.92

    return max(0.0, min(1.0, 1.4 * raw))

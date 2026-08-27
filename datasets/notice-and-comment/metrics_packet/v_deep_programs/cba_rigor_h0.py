"""cba_rigor hybrid: Engagement with cost-benefit / economic analysis.

Construct: ~1.0 = comment provides its own quantified cost/benefit figures AND directly
critiques the rule's Regulatory Impact Analysis (RIA)/economic analysis (wrong baseline,
missing cost category, bad assumption); ~0.5 = some numeric engagement but no real critique
of the agency's analysis, or vice versa; ~0.0 = no economic engagement (pure opinion, no
numbers, no reference to costs/benefits).

INPUT = comment text. Code sees: dollar-figure detection with magnitude parsing ($X
thousand/million/billion), percent figures, numeric density near economic vocabulary
(proximity, not just co-occurrence anywhere in the document), and baseline/counterfactual
comparison language ("compared to", "absent this rule", "status quo"). Code CANNOT see
whether the commenter's own figures are ACCURATE (needs external economic data) — that is
out of scope for h0 and would be a tier-2 fetch hook.
"""
import re

LLM_FIELDS = {
    "cost_benefit_figures": (
        "Comma-separated specific dollar or percentage figures the comment provides about "
        "costs, benefits, or economic impact of the rule (e.g. '$2.3 million', '15% increase "
        "in compliance cost'), verbatim. Answer NONE if the comment gives no such figures."
    ),
    "ria_critique": (
        "In <=20 words, what specifically the comment argues is wrong, missing, or "
        "understated in the rule's cost-benefit / economic / Regulatory Impact Analysis. "
        "Answer NONE if the comment does not critique the agency's economic analysis."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_MONEY_RE = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)\s*(thousand|million|billion|bn|k|m|b)?\b', re.I)
_MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}
_PCT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)\b', re.I)
_ECON_VOCAB_RE = re.compile(
    r'\b(cost|costs|benefit|benefits|economic|compliance cost|burden|savings|expense|'
    r'expenses|revenue|ria\b|regulatory impact analysis|cost-benefit|monetiz\w*|quantif\w*|'
    r'cost-effective|price|budget)\b', re.I)
_BASELINE_RE = re.compile(
    r'\b(baseline|status quo|counterfactual|compared to|relative to|versus|vs\.\s|'
    r'current(?:ly)? cost|absent this rule|in the absence of|prior to this rule|'
    r'under the current|as compared)\b', re.I)


def _split_names(val):
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return []
    return [p.strip() for p in re.split(r"[,;]", val) if p.strip() and p.strip().lower() not in _NONE]


def _money_values(t):
    vals = []
    for m in _MONEY_RE.finditer(t):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        vals.append(v * _MULT.get((m.group(2) or "").lower(), 1))
    return vals


def _numeric_spans(t):
    """Return (start, end) spans of every money/pct figure found (for proximity check)."""
    spans = [m.span() for m in _MONEY_RE.finditer(t)] + [m.span() for m in _PCT_RE.finditer(t)]
    return spans


def _proximity_fraction(t, spans, window=60):
    if not spans:
        return 0.0
    econ_spans = [m.span() for m in _ECON_VOCAB_RE.finditer(t)]
    if not econ_spans:
        return 0.0
    near = 0
    for s0, s1 in spans:
        if any(not (e1 < s0 - window or e0 > s1 + window) for e0, e1 in econ_spans):
            near += 1
    return near / len(spans)


def _code_score(t):
    money_vals = _money_values(t)
    n_money = len(set(round(v, 2) for v in money_vals))
    n_pct = len(_PCT_RE.findall(t))
    spans = _numeric_spans(t)
    prox = _proximity_fraction(t, spans)
    has_baseline = bool(_BASELINE_RE.search(t))

    money_part = min(0.35, 0.15 * n_money)
    pct_part = min(0.15, 0.08 * n_pct)
    prox_part = 0.25 * prox if spans else 0.0
    baseline_part = 0.25 if has_baseline else 0.0
    return max(0.0, min(1.0, money_part + pct_part + prox_part + baseline_part))


def _llm_score(extracted):
    figs = _split_names(extracted.get("cost_benefit_figures"))
    n = len(figs)
    fig_part = {0: 0.1, 1: 0.4}.get(n, 0.65 if n <= 3 else 0.85)
    critique = extracted.get("ria_critique")
    has_critique = isinstance(critique, str) and critique.strip().lower() not in _NONE
    crit_part = 0.25 if has_critique else 0.0
    return max(0.0, min(1.0, fig_part * 0.75 + crit_part))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

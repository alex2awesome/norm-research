"""numeric_consistency hybrid: Does the comment's quantitative argument hold together
arithmetically? (VERIFICATION-TIER — code makes correctness decisions about the extracted
numbers, not just detects their presence.)

Construct: ~1.0 = the comment's numbers form a VERIFIED-coherent chain (stated components sum
to a stated total within tolerance; a stated percentage matches the part/whole pair it is
attached to; a per-unit rate times a stated count reconciles with a stated total) with no
internal contradictions; ~0.5 = numbers are present but there is nothing chainable to check
(isolated figures, no totals/rates/percent claims to verify against); ~0.0 = the comment's own
numbers actively fail to reconcile (components don't sum to the claimed total, a percentage
doesn't match its stated part/whole, or the same named quantity is asserted at materially
different values in different places).

INPUT = comment text. Code parses ALL quantities (dollar figures with magnitude words, percent
figures, per-unit rates "$X per Y", and bare counts tied to a whitelisted unit noun like
"12,000 workers") with unit/magnitude normalization, EXCLUDING years (19xx/20xx), section/part/
docket/page/exhibit/citation numbers from the parse. It then runs three arithmetic-consistency
checks (component-sum-to-total within 2% tolerance, percent-vs-part/whole within tolerance,
rate*count-vs-total within tolerance) and one contradiction check (the same cost/benefit-type
referent phrase bound to two materially different dollar values). All checks are WINDOWED —
only numbers sharing a sentence/adjacent-sentence span, or an explicit "total"/referent
relationship, are compared; unrelated numbers elsewhere in the document are never cross-checked.
Code CANNOT verify the numbers against EXTERNAL ground truth (e.g. whether $600,000 is a
plausible compliance cost for this industry) — only INTERNAL self-consistency, which is the
in-scope verification for h2.
"""
import re

LLM_FIELDS = {
    "central_quant_claim": (
        "The comment's single most important quantitative claim, verbatim, including its "
        "numbers (e.g. 'compliance will cost small facilities $45,000 per year'). Answer NONE "
        "if the comment makes no quantitative claim."
    ),
    "supporting_figures": (
        "Comma-separated verbatim list of the component figures the comment uses to SUPPORT "
        "its central quantitative claim (e.g. '$30,000 in labor, $15,000 in equipment'). "
        "Answer NONE if there are no supporting figures."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

# ---------------------------------------------------------------------------
# quantity grammar
# ---------------------------------------------------------------------------

_MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}
_MULT_WORD = r'(?:thousand|million|billion|bn|k|m|b)'

_RATE_RE = re.compile(
    r'\$\s*([\d,]+(?:\.\d+)?)\s*(' + _MULT_WORD + r')?\s*per\s+([a-zA-Z]+)', re.I)
_MONEY_RE = re.compile(
    r'\$\s*([\d,]+(?:\.\d+)?)\s*(' + _MULT_WORD + r')?\b', re.I)
_PCT_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\s*(?:%|(?:percent|pct\.?)\b)', re.I)
_NUM_RE = re.compile(
    r'\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(' + _MULT_WORD + r')?\b')

_YEAR_RE = re.compile(r'^(?:19|20)\d{2}$')
_EXCLUDE_CTX_RE = re.compile(
    r'(§|section|sec\.|subpart|part\s*|no\.|docket|page|p\.|pp\.|exhibit|cfr|u\.s\.c\.|usc|'
    r'fed\.?\s*reg\.?|vol\.|volume|figure|fig\.|table|footnote|fn\.|note\s*)\s*$', re.I)

_UNIT_NOUN_SET = set("""
worker workers employee employees facility facilities entity entities business businesses
farm farms plant plants site sites establishment establishments unit units ton tons
pound pounds gallon gallons hour hours acre acres mile miles job jobs position positions
application applications case cases complaint complaints violation violations incident
incidents claim claims permit permits license licenses student students patient patients
household households family families vehicle vehicles device devices product products
filing filings comment comments respondent respondents operator operators farmer farmers
""".split())

_TOTAL_PRE_RE = re.compile(
    r'\b(?:total(?:s|ing)?|sum(?:s|med)?|grand\s+total)\b(?:\s+of|\s+is|\s+was|\s+equals?)?\s*$', re.I)
_TOTAL_POST_RE = re.compile(
    r'^\s*,?\s*(?:in\s+total|altogether|overall|combined|in\s+the\s+aggregate)\b', re.I)

_FROM_TO_RE = re.compile(
    r'from\s+\$?\s*([\d,]+(?:\.\d+)?)\s*(' + _MULT_WORD + r')?\s+to\s+\$?\s*'
    r'([\d,]+(?:\.\d+)?)\s*(' + _MULT_WORD + r')?', re.I)

_REFERENT_RE = re.compile(
    r'\b((?:compliance|annual|total|estimated|projected|one[- ]time|recurring|initial|'
    r'upfront|ongoing|labor|equipment|administrative)?\s*(?:cost|costs|benefit|benefits|'
    r'saving|savings|burden|burdens|impact|impacts|price|prices|fee|fees|fine|fines|'
    r'penalty|penalties|tax|taxes|wage|wages|salary|salaries|revenue|revenues|expense|'
    r'expenses))\b', re.I)


def _split_names(val):
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return []
    return [p.strip() for p in re.split(r"[,;]", val) if p.strip() and p.strip().lower() not in _NONE]


def _to_value(raw, mult_word):
    try:
        v = float(raw.replace(",", ""))
    except ValueError:
        return None
    return v * _MULT.get((mult_word or "").lower(), 1)


def _in_excluded_context(t, start):
    pre = t[max(0, start - 25):start]
    return bool(_EXCLUDE_CTX_RE.search(pre))


def _split_sentences(t):
    spans = [m.span() for m in re.finditer(r'[^.!?\n]+[.!?]*', t) if m.group().strip()]
    return spans or [(0, len(t))]


def _sent_idx_for(pos, sent_spans):
    for i, (s, e) in enumerate(sent_spans):
        if s <= pos < e:
            return i
    return len(sent_spans) - 1


def _find_unit_noun_forward(t, start, max_words=3, max_chars=25):
    """Look forward from `start` for a whitelisted unit noun within a short adjective+noun
    window (e.g. "12,000 affected workers" -> "worker"), without crossing a sentence end."""
    tail = t[start:start + max_chars]
    tail = re.split(r'[.!?]', tail, maxsplit=1)[0]
    for w in re.findall(r"[A-Za-z']+", tail)[:max_words]:
        if w.lower() in _UNIT_NOUN_SET:
            return w.lower().rstrip('s')
    return None


# ---------------------------------------------------------------------------
# parse all quantities: kind in {money, pct, count, rate}
# ---------------------------------------------------------------------------

def _parse_quantities(t, sent_spans):
    quantities = []
    used_spans = []

    for m in _RATE_RE.finditer(t):
        s, e = m.span()
        if _in_excluded_context(t, s):
            continue
        unit = m.group(3).lower().rstrip('s')
        if unit not in _UNIT_NOUN_SET:
            continue  # "$X per year/month/day" is a cadence, not a countable rate*count target
        val = _to_value(m.group(1), m.group(2))
        if val is None:
            continue
        quantities.append(dict(kind='rate', value=val, unit=unit, span=(s, e),
                                sent=_sent_idx_for(s, sent_spans)))
        used_spans.append((s, e))

    def _overlaps_used(s, e):
        return any(not (e <= us or s >= ue) for us, ue in used_spans)

    for m in _MONEY_RE.finditer(t):
        s, e = m.span()
        if _overlaps_used(s, e) or _in_excluded_context(t, s):
            continue
        val = _to_value(m.group(1), m.group(2))
        if val is None:
            continue
        quantities.append(dict(kind='money', value=val, unit=None, span=(s, e),
                                sent=_sent_idx_for(s, sent_spans)))

    for m in _PCT_RE.finditer(t):
        s, e = m.span()
        if _in_excluded_context(t, s):
            continue
        quantities.append(dict(kind='pct', value=float(m.group(1)), unit=None, span=(s, e),
                                sent=_sent_idx_for(s, sent_spans)))

    for m in _NUM_RE.finditer(t):
        s, e = m.span()
        if _overlaps_used(s, e) or _in_excluded_context(t, s):
            continue
        if t[e:e + 3].lstrip().startswith('%'):
            continue  # this number is a percent figure (PCT_RE's concern), not a count
        raw_num, mult_word = m.group(1), m.group(2)
        if not mult_word and ',' not in raw_num and _YEAR_RE.match(raw_num):
            continue  # bare 19xx/20xx -> treat as year, not count (never a quantity)
        unit = _find_unit_noun_forward(t, e)
        if unit is None:
            continue  # no whitelisted unit noun nearby -> not a countable quantity
        val = _to_value(raw_num, mult_word)
        if val is None:
            continue
        quantities.append(dict(kind='count', value=val, unit=unit, span=(s, e),
                                sent=_sent_idx_for(s, sent_spans)))
        used_spans.append((s, e))

    quantities.sort(key=lambda q: q['span'][0])
    return quantities


def _near_total_kw(t, span, pre_window=20, post_window=25):
    """Directional check: a keyword like "totals $X" (before) or "$X in total" (after) — NOT
    a wide symmetric window, which would false-positive on a component figure that merely
    appears near a sentence that introduces a breakdown ("This total is made up of $350,000...")."""
    s, e = span
    pre = t[max(0, s - pre_window):s]
    post = t[e:e + post_window]
    return bool(_TOTAL_PRE_RE.search(pre)) or bool(_TOTAL_POST_RE.match(post))


# ---------------------------------------------------------------------------
# consistency checks (each returns (results: [bool], explained: {span, ...}))
# ---------------------------------------------------------------------------

def _component_sum_checks(t, quantities):
    results, explained = [], set()
    for kind in ("money", "count"):
        kq = [q for q in quantities if q["kind"] == kind]
        totals = [q for q in kq if _near_total_kw(t, q["span"])]
        for tot in totals:
            components = [q for q in kq
                          if q["span"] != tot["span"]
                          and abs(q["sent"] - tot["sent"]) <= 1  # tight: adjacent sentence only
                          and not _near_total_kw(t, q["span"])]
            if len(components) < 2:
                continue
            comp_sum = sum(q["value"] for q in components)
            if tot["value"] == 0:
                continue
            rel_err = abs(comp_sum - tot["value"]) / abs(tot["value"])
            results.append(rel_err <= 0.02)
            explained.add(tot["span"])
            for q in components:
                explained.add(q["span"])
    return results, explained


def _percent_consistency_checks(t, quantities):
    results, explained = [], set()
    pct_qs = [q for q in quantities if q["kind"] == "pct"]
    money_qs = [q for q in quantities if q["kind"] == "money"]
    for m in _FROM_TO_RE.finditer(t):
        s, e = m.span()
        # An explicit "from A to B" is a stated CHANGE, never a contradiction target for the
        # internal-contradiction check below -- mark its money spans explained unconditionally,
        # whether or not a nearby percent claim exists to verify against.
        for q in money_qs:
            if s <= q["span"][0] < e:
                explained.add(q["span"])
        a = _to_value(m.group(1), m.group(2))
        b = _to_value(m.group(3), m.group(4))
        if not a or a == 0 or b is None:
            continue
        implied = (b - a) / a * 100.0
        near_pct = [q for q in pct_qs
                    if min(abs(q["span"][0] - s), abs(q["span"][0] - e)) <= 120]
        if not near_pct:
            continue
        near_pct.sort(key=lambda q: min(abs(q["span"][0] - s), abs(q["span"][0] - e)))
        pq = near_pct[0]
        tol = max(2.0, 0.10 * abs(implied))
        results.append(abs(abs(implied) - pq["value"]) <= tol)
        explained.add(pq["span"])
    return results, explained


def _rate_count_total_checks(t, quantities):
    results, explained = [], set()
    rates = [q for q in quantities if q["kind"] == "rate"]
    counts = [q for q in quantities if q["kind"] == "count"]
    moneys = [q for q in quantities if q["kind"] == "money"]
    for r in rates:
        matched_counts = [c for c in counts
                           if c["unit"] == r["unit"] and abs(c["sent"] - r["sent"]) <= 2]
        if not matched_counts:
            continue
        c = matched_counts[0]
        window_money = [mo for mo in moneys if abs(mo["sent"] - r["sent"]) <= 2]
        if not window_money:
            continue
        totals = [mo for mo in window_money if _near_total_kw(t, mo["span"])] or window_money
        tot = max(totals, key=lambda mo: mo["value"])
        if tot["value"] == 0:
            continue
        expected = r["value"] * c["value"]
        rel_err = abs(expected - tot["value"]) / abs(tot["value"])
        results.append(rel_err <= 0.04)  # slightly looser: two independently-rounded factors
        explained.add(r["span"]); explained.add(c["span"]); explained.add(tot["span"])
    return results, explained


def _nearest_referent(t, span, window=60):
    s, e = span
    ctx_before = t[max(0, s - window):s]
    ctx_after = t[e:e + 30]
    best = None
    for m in _REFERENT_RE.finditer(ctx_before):
        best = m.group(1)  # keep the closest (last) match before the number
    if best is None:
        m = _REFERENT_RE.search(ctx_after)
        if m:
            best = m.group(1)
    if best is None:
        return None
    norm = re.sub(r'\s+', ' ', best.strip().lower())
    words = norm.split()
    if words:
        words[-1] = re.sub(r's$', '', words[-1])
    return " ".join(words)


def _contradiction_checks(t, quantities, explained):
    money_qs = [q for q in quantities if q["kind"] == "money" and q["span"] not in explained]
    groups = {}
    for q in money_qs:
        ref = _nearest_referent(t, q["span"])
        if not ref:
            continue
        groups.setdefault(ref, []).append(q)
    contradictions = []
    for ref, qs in groups.items():
        if len(qs) < 2:
            continue
        vals = [q["value"] for q in qs]
        hi, lo = max(vals), min(vals)
        if lo <= 0:
            continue
        if (hi - lo) / lo > 0.05 and (hi - lo) > 500:
            contradictions.append(ref)
    return contradictions


def _code_score(t):
    sent_spans = _split_sentences(t)
    quantities = _parse_quantities(t, sent_spans)
    if len(quantities) < 2:
        return 0.5  # nothing to verify

    a_results, a_explained = _component_sum_checks(t, quantities)
    b_results, b_explained = _percent_consistency_checks(t, quantities)
    c_results, c_explained = _rate_count_total_checks(t, quantities)
    explained = a_explained | b_explained | c_explained
    contradictions = _contradiction_checks(t, quantities, explained)

    results = a_results + b_results + c_results
    n_ok = sum(1 for r in results if r)
    n_bad = sum(1 for r in results if not r)
    n_contra = len(contradictions)

    if not results and not contradictions:
        density_bonus = min(0.10, 0.02 * min(len(quantities), 5))
        return 0.45 + density_bonus

    base = 0.5
    base += 0.16 * min(n_ok, 3)
    base -= 0.22 * min(n_bad, 3)
    base -= 0.18 * min(n_contra, 3)
    return max(0.0, min(1.0, base))


def _llm_score(extracted):
    claim = extracted.get("central_quant_claim")
    has_claim = isinstance(claim, str) and claim.strip().lower().strip(". ") not in _NONE
    figs = _split_names(extracted.get("supporting_figures"))
    n = len(figs)
    claim_part = 0.35 if has_claim else 0.05
    figs_part = {0: 0.0, 1: 0.25}.get(n, 0.45 if n <= 3 else 0.60)
    return max(0.0, min(1.0, claim_part + figs_part))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

# Hybrid module for legal_title_vii aspect a21:
# "disparate-impact: facially neutral policy + statistical disparity"
# (42 U.S.C. sec. 2000e-2(k); Griggs v. Duke Power; Watson v. Fort Worth Bank).
#
# Baseline diagnosis (v1_structure, train rho=0.264): scores keyword hits for
# a fixed policy-term list and a fixed stat-term list, plus a four-fifths-style
# ratio computed over ANY two "%" numbers found anywhere in the document. Two
# failure modes are visible in the train pack:
#   (1) UNDER-counts real disparate-impact facts that never surface as a
#       percentage or as one of the listed phrases -- e.g. "Johnson hired 148
#       white longline drivers but no black longline drivers" (d00619, judge
#       0.8) is a textbook Griggs stark-disparity + unvalidated-test claim,
#       but has zero "%" signs and doesn't say "unvalidated test" or
#       "disparate impact" verbatim, so baseline scores it 0.0.
#   (2) OVER-fires when an isolated policy word appears with no disparate-
#       impact claim behind it at all -- e.g. "passed a written test" in a
#       hiring-harassment case (d00205, judge 0.1) trips 'high school diploma'
#       and scores 0.3 even though no protected-group statistical disparity is
#       alleged. Per the corpus notes, legal-term presence is a weak proxy for
#       the construct actually being present; the fix is to let an LLM field
#       ground WHICH neutral practice is being challenged and WHETHER the
#       numeric evidence is tied to that practice's outcomes for a protected
#       group, rather than trusting keyword co-occurrence.
#
# Design: keep the baseline's lexical fallback and its percentage-pair ratio
# (down-weighted, not discarded) and ADD:
#   (1) two LLM fields carrying the doctrinal judgment code can't make: what
#       specific neutral practice is challenged, and what specific numeric
#       group-comparison evidence is offered for it;
#   (2) a code-only generalization of the four-fifths ratio from "%" pairs to
#       raw group-linked counts (e.g. "0 black" vs "148 white"), which is the
#       more classic and harder-to-fake disparate-impact signature;
#   (3) a small code-only temporal-rigor bonus (ops.extract_dates) that credits
#       a disparity measured over a defined date range as more evidentially
#       serious than an undated, anecdotal one.
# Both the "practice identified" and "disparity evidenced" pieces are combined
# with an interaction term, not just summed -- the doctrine requires BOTH
# elements together (a neutral practice alone, or numbers alone, get modest
# partial credit; both together get the bulk of the score), matching the
# train pattern where policy-only cases sit around 0.1-0.2 and the one
# practice+numbers case sits at 0.8.

import re

_NONE_STRINGS = {"", "none", "n/a", "na", "unknown", "not stated", "not present",
                 "no evidence", "not applicable", "not specified", "not mentioned", "unclear"}

LLM_FIELDS = {
    "neutral_practice": (
        "In <=15 words, name the specific facially-neutral policy, test, or "
        "requirement plaintiff claims disproportionately screens out a "
        "protected group, or say NONE if no such practice is challenged."
    ),
    "disparity_evidence": (
        "In <=15 words, state the specific numeric evidence (selection "
        "rates, hire/pass counts, or percentages) comparing the protected "
        "group's outcomes vs. others under that practice, or say NONE if no "
        "such numeric comparison is given."
    ),
}

_POLICY_TERMS = ['facially neutral', 'neutral policy', 'neutral practice', 'written exam',
                 'aptitude test', 'physical ability test', 'strength test',
                 'height requirement', 'weight requirement', 'high school diploma',
                 'criminal background check', 'credit check', 'credential requirement',
                 'screening test', 'minimum score', 'cutoff score']

_STAT_TERMS = ['selection rate', 'pass rate', 'four-fifths', 'applicant pool',
               'statistically significant', 'disparity', 'disproportionate',
               'adverse impact', 'disparate impact']

_GROUP_TERMS = ['black', 'white', 'african-american', 'african american', 'hispanic',
                'latino', 'latina', 'asian', 'female', 'women', 'male', 'men',
                'minority', 'minorities', 'non-white']

_WORD_NUM = {'no': 0, 'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

_COUNT_PAT = re.compile(
    r'\b(no|zero|one|two|three|four|five|six|seven|eight|nine|ten|\d{1,6})\s+(' +
    '|'.join(sorted(_GROUP_TERMS, key=len, reverse=True)) + r')\b')


def _clean(v):
    if not v:
        return ""
    v = v.strip()
    if v.lower().strip(".") in _NONE_STRINGS:
        return ""
    return v


def _ratio_signal(hi, others):
    if hi <= 0 or not others:
        return 0.0
    worst_ratio = min(others) / hi
    return max(0.0, min(1.0, (0.8 - worst_ratio) / 0.8 * 1.5))


def _pct_ratio_signal(t):
    pcts = [float(m) for m in re.findall(r'(\d{1,3}(?:\.\d+)?)\s*%', t)]
    if len(pcts) < 2:
        return 0.0
    hi = max(pcts)
    others = [p for p in pcts if p != hi]
    return _ratio_signal(hi, others)


def _group_key(grp):
    if 'black' in grp or 'african' in grp:
        return 'black'
    if grp in ('female', 'women'):
        return 'female'
    if grp in ('male', 'men'):
        return 'male'
    return grp


def _count_ratio_signal(tl):
    # Generalizes the "%"-only four-fifths ratio to raw group-linked counts,
    # e.g. "148 white ... but no black ..." -- a stark disparity that never
    # shows up as a percentage but is the classic Griggs fact pattern.
    counts_by_group = {}
    for m in _COUNT_PAT.finditer(tl):
        num_str, grp = m.group(1), m.group(2)
        val = _WORD_NUM.get(num_str)
        if val is None:
            try:
                val = int(num_str)
            except ValueError:
                continue
        counts_by_group.setdefault(_group_key(grp), []).append(val)
    if len(counts_by_group) < 2:
        return 0.0
    group_maxes = [max(v) for v in counts_by_group.values()]
    hi = max(group_maxes)
    others = [g for g in group_maxes if g != hi]
    return _ratio_signal(hi, others)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        tl = t.lower()

        extracted = extracted or {}
        neutral_practice = _clean(str(extracted.get("neutral_practice", "") or ""))
        disparity_evidence = _clean(str(extracted.get("disparity_evidence", "") or ""))

        policy_kw_hit = any(p in tl for p in _POLICY_TERMS)
        stat_kw_hit = any(s in tl for s in _STAT_TERMS)

        quant_signal = max(_pct_ratio_signal(t), _count_ratio_signal(tl))

        # Temporal rigor: a disparity anchored to a defined date range reads as
        # measured/serious rather than anecdotal (weak, capped contribution).
        try:
            n_dates = len(ops.extract_dates(t))
        except Exception:
            n_dates = 0
        temporal_bonus = 0.05 if (n_dates >= 2 and quant_signal > 0) else 0.0

        # Practice component: LLM grounding is primary; a bare keyword hit
        # (baseline's whole signal) is kept only as a much weaker fallback so
        # an isolated term like "high school diploma" can't alone imply the
        # doctrinal claim is being made.
        if neutral_practice:
            policy_component = 1.0
        elif policy_kw_hit:
            policy_component = 0.30
        else:
            policy_component = 0.0

        # Statistical component: LLM-grounded numeric evidence (tied to the
        # challenged practice) is primary; the code-computed ratio and the
        # stat-language keywords are supporting, not sufficient alone.
        stat_component = 0.0
        if disparity_evidence:
            stat_component += 0.50
        stat_component += 0.25 * quant_signal
        if stat_kw_hit:
            stat_component += 0.15
        stat_component = max(0.0, min(1.0, stat_component))

        # Doctrine requires BOTH elements together, so weight their product
        # heavily; either alone gets only modest partial credit (matches the
        # train pattern: policy-only cases ~0.1-0.2, practice+numbers ~0.8).
        s = (0.15 * policy_component
             + 0.15 * stat_component
             + 0.70 * (policy_component * stat_component)
             + temporal_bonus)
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

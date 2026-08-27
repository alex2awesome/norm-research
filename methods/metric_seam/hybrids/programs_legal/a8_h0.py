# Hybrid module for legal_title_vii aspect a8: "direct evidence or explicit bias"
# (statements/admissions of discriminatory motive tied to a decision-maker;
# lets a plaintiff bypass McDonnell Douglas -- Price Waterhouse / Sec. 2000e-2(m)).
#
# Baseline diagnosis: the regex baseline (rho=0.206) fires on generic litigation
# boilerplate that has nothing to do with a quoted/paraphrased bias remark --
# "stated that" hits routine procedural sentences (EEOC counsel's brief, an
# interview-feedback recap), and "slur"/"discriminatory remark" fire on the
# plaintiff's own bare ALLEGATION that slurs occurred even when the text never
# surfaces what was said or by whom (corpus notes: this is a doctrinal-
# construct criterion where legal-term presence is a weak proxy for the
# construct being factually present). Meanwhile the baseline scores 0 on some
# of the strongest direct-evidence documents in the pack: a facially
# discriminatory hiring policy ("Pan Am hires only females"), a decision-
# maker's own paraphrased admission ("Allen ... stated she wanted to promote
# someone younger and not a minority"), and quoted age-based remarks tied to
# the plaintiff's job performance -- none of which contain any baseline
# keyword.
#
# What code cannot reach: whether the text actually surfaces a SPECIFIC,
# attributable biased statement or facially discriminatory policy (as opposed
# to a bare allegation that bias occurred), and whether that statement's
# speaker was the one who actually held/exercised decision authority over the
# adverse action at the time -- vs. a stray remark by someone no longer in
# that role, or an admission unconnected to any adverse action against the
# plaintiff at all (e.g. a supervisor's admitted racial *suspicion* about
# OTHER staff conflicts, in a case where the plaintiff was never demoted,
# fired, or docked pay/benefits). Both require reading the narrative, so both
# go to two LLM fields; code keeps the deterministic predicate: presence,
# multiplied by an authority/proximity gate, plus a small bonus when the
# extracted statement is verifiably a literal quotation in the source text
# (stronger evidence than a bare paraphrase) -- and a damped keyword floor for
# when no LLM signal is available at all (kept from the baseline so we never
# discard its one real strength: catching unambiguous slur/admission phrasing).

import re

LLM_FIELDS = {
    "bias_stmt": (
        "Quote or closely paraphrase (<=20 words) the single clearest biased "
        "statement, slur, admission of discriminatory/retaliatory motive, or "
        "facially discriminatory policy tied to a protected trait (race, sex, "
        "age, religion, disability, national origin); answer NONE if the text "
        "never surfaces a specific statement/policy, only a bare allegation "
        "that bias occurred."
    ),
    "bias_tied_to_action": (
        "If that statement/policy exists: was its speaker/maker the one who "
        "actually held decision authority over an adverse action (firing, "
        "non-hire, non-promotion, discipline, pay/duty change) taken against "
        "the plaintiff, at the time it was taken? Answer YES, NO (stray "
        "remark, no longer in that role, or unrelated to any adverse action "
        "against the plaintiff), or UNCLEAR; answer NONE if no statement/"
        "policy exists."
    ),
}

_KWS = (
    'slur', 'epithet', 'derogatory', 'told plaintiff', 'said to plaintiff',
    'admitted that', 'stated that', 'explicitly stated', "we don't hire",
    'too old to', 'because you are', 'get rid of',
    'racial comment', 'sexist comment', 'discriminatory remark',
)

_NONE_VALS = ("", "none", "n/a", "na")


def _clean(v):
    v = (v or "").strip()
    return "" if v.lower().rstrip(".") in _NONE_VALS else v


def _word_in(word, hay):
    return re.search(r'\b' + re.escape(word) + r'\b', hay) is not None


def _keyword_signal(tl):
    hits = sum(1 for k in _KWS if k in tl)
    return min(1.0, hits / 3.0)


def _quote_bonus(stmt, t):
    # A literal quotation of the extracted statement in the source text is
    # stronger evidence than a bare paraphrase -- reward it lightly.
    words = re.findall(r"[a-zA-Z']{4,}", stmt.lower())
    if not words:
        return 0.0
    quoted = re.findall(r'"([^"]{3,300})"', t) + re.findall(r"'([^']{3,300})'", t)
    if not quoted:
        return 0.0
    blob = " ".join(q.lower() for q in quoted)
    overlap = sum(1 for w in words if w in blob)
    return 0.05 if overlap >= max(1, len(words) // 3) else 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        tl = t.lower()

        extracted = extracted if isinstance(extracted, dict) else {}
        bias_stmt = _clean(extracted.get("bias_stmt", ""))
        tie = _clean(extracted.get("bias_tied_to_action", "")).lower()

        if not bias_stmt:
            # No grounded construct signal available (extractor found nothing
            # specific, or wasn't run): fall back to the keyword net, but damp
            # it -- term presence alone is a weak proxy for this construct.
            return max(0.0, min(1.0, 0.5 * _keyword_signal(tl)))

        # A concrete, attributable statement/policy was surfaced -- weight it
        # by whether its speaker actually held decision authority over the
        # adverse action against the plaintiff, at the time.
        if _word_in('yes', tie):
            tie_mult = 1.0
        elif _word_in('no', tie):
            tie_mult = 0.12
        elif 'unclear' in tie:
            tie_mult = 0.5
        else:
            tie_mult = 0.45  # field empty/unavailable -- ambiguous, moderate credit

        val = 0.85 * tie_mult + _quote_bonus(bias_stmt, t)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5

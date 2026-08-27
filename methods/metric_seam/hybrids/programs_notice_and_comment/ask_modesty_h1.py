"""ask_modesty hybrid: How modest/implementable the comment's requested change is.

Construct: ~1.0 = the comment's main ask is modest and cheaply implementable within the
existing rule (a clarification, a technical correction, a narrow phase-in, a subgroup
exemption); ~0.5 = a moderate ask (a real adjustment to a requirement, a meaningful delay,
an expansion of scope); ~0.0 = a maximal ask (structural overhaul, reinterpretation of the
agency's statutory authority, full withdrawal/rescission of the rule).

INPUT = comment text. Code sees: a taxonomy of ask-verbs weighted by implementation cost
(clarify/define/extend deadline/exempt = modest; expand/broaden/mandate = moderate;
withdraw/rescind/rewrite/overhaul/restructure = maximal), plus scope quantifiers ("for all",
"industry-wide", "entirely", "from scratch") that push a nominally-modest verb toward the
maximal end. Code CANNOT verify how costly the ask would ACTUALLY be for the agency to
implement (needs domain/operational knowledge) — out of scope for h1.
"""
import re

LLM_FIELDS = {
    "main_ask": (
        "The single main thing the comment asks the agency to do, in <=15 words, verbatim or "
        "near-verbatim. Answer NONE if the comment makes no request."
    ),
    "ask_category": (
        "One word categorizing that main ask from this set: clarification, adjustment, "
        "exemption, delay, expansion, overhaul, withdrawal, none."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_MODEST_VERB_RE = re.compile(
    r'\b(clarify|clarifies|clarification|define|defines|definition of|correct[s]?|'
    r'technical correction|extend the deadline|extend the compliance|phase[- ]in|'
    r'phase in|exempt|exemption|narrow(?:ly)? tailor|small (?:tweak|change)|'
    r'minor (?:change|revision|adjustment))\b', re.I)
_MODERATE_VERB_RE = re.compile(
    r'\b(adjust|adjustment|revise|amend|modify|reduce|increase|raise|lower|'
    r'expand|broaden|extend (?:the )?scope|add (?:a|an|another) requirement|'
    r'delay|postpone|reconsider)\b', re.I)
_MAXIMAL_VERB_RE = re.compile(
    r'\b(withdraw|rescind|scrap|abandon|kill the rule|overhaul|restructure|'
    r'rewrite|start over|from scratch|reinterpret|statutory authority to|'
    r'complete(?:ly)? redo|fundamental(?:ly)? (?:change|rethink|revise))\b', re.I)
_SCOPE_QUANT_RE = re.compile(
    r'\b(for all|industry[- ]wide|entirely|completely|across the board|every '
    r'(?:facility|entity|sector)|nationwide|sector[- ]wide)\b', re.I)


def _code_score(t):
    n_modest = len(_MODEST_VERB_RE.findall(t))
    n_moderate = len(_MODERATE_VERB_RE.findall(t))
    n_maximal = len(_MAXIMAL_VERB_RE.findall(t))
    has_broad_scope = bool(_SCOPE_QUANT_RE.search(t))

    if n_maximal > 0:
        base = 0.10
    elif n_moderate > 0 and n_modest == 0:
        base = 0.45
    elif n_modest > 0:
        base = 0.85 - 0.10 * min(2, n_moderate)
    else:
        base = 0.5  # no clear ask signal at all -> neutral/uninformative

    if has_broad_scope:
        base -= 0.20
    return max(0.0, min(1.0, base))


_CATEGORY_SCORE = {
    "clarification": 0.90,
    "adjustment": 0.55,
    "exemption": 0.75,
    "delay": 0.60,
    "expansion": 0.30,
    "overhaul": 0.10,
    "withdrawal": 0.03,
    "none": 0.5,
}


def _llm_score(extracted):
    ask = extracted.get("main_ask")
    has_ask = isinstance(ask, str) and ask.strip().lower().strip(". ") not in _NONE
    cat = (extracted.get("ask_category") or "").strip().lower()
    cat = re.sub(r'[^a-z_]', '', cat)
    cat_score = _CATEGORY_SCORE.get(cat, 0.5)
    if not has_ask:
        return max(0.0, min(1.0, 0.5 * cat_score + 0.5 * 0.5))
    return cat_score


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

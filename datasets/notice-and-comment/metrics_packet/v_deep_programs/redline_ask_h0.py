"""redline_ask hybrid: Specific, actionable requested change to the rule text.

Construct: ~1.0 = comment names a specific change-verb + object (revise/strike/insert/amend
§ N) AND offers ready-to-adopt replacement text ("strike X, insert Y" / "revise § N to read:
...") ; ~0.5 = a change-verb targeted at a named section but no ready-to-adopt text, or vague
"please reconsider this section"; ~0.0 = no actionable ask, only general opinion/position.

INPUT = comment text. Code sees: change-verb+object co-occurrence with a section-level
target (§ N, Part N, paragraph (a), N.NN), and structural "strike ... insert ..." /
"revise § N to read" patterns that signal camera-ready redline text. Code CANNOT verify the
replacement text is legally coherent (needs domain legal review) — out of scope for h0.
"""
import re

LLM_FIELDS = {
    "specific_ask": (
        "The single most specific requested change to the rule, verbatim as stated in the "
        "comment (e.g. 'extend the compliance deadline from 30 to 60 days', 'delete "
        "paragraph (c)(2)'). Answer NONE if the comment makes no specific requested change."
    ),
    "replacement_text": (
        "Any ready-to-adopt replacement regulatory text offered by the comment (verbatim), "
        "such as 'strike X, insert Y' or specific proposed wording for a section. Answer "
        "NONE if no such replacement text is offered."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_CHANGE_VERB_RE = re.compile(
    r'\b(revise|amend|strike|delete|remove|insert|replace|add|change|modify|clarify|'
    r'extend|reduce|increase|eliminate|withdraw|reconsider|reinstate|narrow|broaden|'
    r'lower|raise)\b', re.I)
_SECTION_REF_RE = re.compile(
    r'§+\s*\d[\d.]*|\bpart\s+\d+\b|\bsubsection\s*\([a-z0-9]+\)|\bparagraph\s*\([a-z0-9]+\)|'
    r'\b\d{1,3}\.\d{1,4}\b', re.I)
_STRIKE_INSERT_RE = re.compile(r'\bstrike\b[^.]{0,100}\binsert\b', re.I | re.S)
_REVISE_TO_READ_RE = re.compile(r'\b(revise|amend)\b[^.]{0,80}\bto read\b', re.I | re.S)


def _targeted_asks(t, window=80):
    """Count of change-verb occurrences that fall within `window` chars of a section ref."""
    verb_spans = [m.span() for m in _CHANGE_VERB_RE.finditer(t)]
    sect_spans = [m.span() for m in _SECTION_REF_RE.finditer(t)]
    if not verb_spans or not sect_spans:
        return 0
    n = 0
    for v0, v1 in verb_spans:
        if any(not (s1 < v0 - window or s0 > v1 + window) for s0, s1 in sect_spans):
            n += 1
    return n


def _code_score(t):
    n_targeted = _targeted_asks(t)
    n_sections = len(_SECTION_REF_RE.findall(t))
    has_structural = bool(_STRIKE_INSERT_RE.search(t) or _REVISE_TO_READ_RE.search(t))

    targeted_part = min(0.40, 0.15 * n_targeted)
    structural_part = 0.35 if has_structural else 0.0
    section_part = min(0.25, 0.08 * n_sections)
    return max(0.0, min(1.0, targeted_part + structural_part + section_part))


def _llm_score(extracted):
    ask = extracted.get("specific_ask")
    if not isinstance(ask, str) or ask.strip().lower().strip(". ") in _NONE:
        return 0.1
    specific = bool(re.search(r'\d|§|["“]', ask))
    base = 0.6 if specific else 0.35
    repl = extracted.get("replacement_text")
    has_repl = isinstance(repl, str) and repl.strip().lower().strip(". ") not in _NONE
    if has_repl:
        base += 0.3
    return max(0.0, min(1.0, base))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

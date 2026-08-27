"""legal_argument hybrid: Asserted legal theory against the rule (APA-grade argument).

Construct: ~1.0 = comment asserts a specific, doctrinally-named legal theory (statutory
conflict, procedural defect, exceeds statutory authority, arbitrary-and-capricious) grounded
in an authority citation or precise APA vocabulary co-occurring with a conflict verb in the
same sentence; ~0.5 = generic "this is illegal"/"unconstitutional" assertion with no
doctrinal grounding; ~0.0 = no legal argument at all (policy opinion only).

INPUT = comment text. Code sees: authority-citation-or-case-name co-occurring with a
conflict verb ("exceeds", "violates", "inconsistent with") within the same sentence, and
precise APA-vocabulary hits (arbitrary and capricious, substantial evidence, 5 U.S.C. § 706,
Chevron, major questions doctrine). Code CANNOT judge whether the legal theory is actually
MERITORIOUS (needs legal expertise) — out of scope for h0.
"""
import re

LLM_FIELDS = {
    "legal_theory": (
        "One phrase naming the specific legal theory the comment asserts against the rule: "
        "'statutory conflict' (conflicts with the enabling statute), 'procedural defect' "
        "(APA notice-and-comment process flaw), 'exceeds authority' (agency lacks statutory "
        "authority for this rule), or 'arbitrary-capricious' (unreasoned/unsupported "
        "decision). Answer NONE if the comment asserts no legal theory."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_CITATION_TOKEN_RE = re.compile(
    r'§|\bU\.?\s*S\.?\s*C\.?\b|\bC\.?\s*F\.?\s*R\.?\b|\bFed\.?\s*Reg\.?\b', re.I)
_CASE_RE = re.compile(r'\b[A-Z][A-Za-z.\']+\s+v\.?\s+[A-Z][A-Za-z.\']+\b')
_CONFLICT_VERB_RE = re.compile(
    r'\b(conflicts? with|exceeds?|violates?|inconsistent with|contrary to|ultra vires|'
    r'beyond (?:the )?(?:agency\'?s )?(?:statutory )?authority|unauthorized by|'
    r'not authorized by|runs? afoul of)\b', re.I)
_APA_VOCAB_RE = re.compile(
    r'\b(arbitrary and capricious|substantial evidence|reasoned decisionmaking|'
    r'notice[- ]and[- ]comment|record for decision|5\s*U\.?\s*S\.?\s*C\.?\s*§?\s*706|'
    r'chevron|major questions doctrine|statutory authority|ultra vires|'
    r'administrative procedure act)\b', re.I)


def _sentences(t):
    return [s for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]


def _grammar_hits(t):
    n = 0
    for s in _sentences(t):
        has_auth = bool(_CITATION_TOKEN_RE.search(s) or _CASE_RE.search(s))
        has_verb = bool(_CONFLICT_VERB_RE.search(s))
        if has_auth and has_verb:
            n += 1
    return n


def _code_score(t):
    n_grammar = _grammar_hits(t)
    apa_terms = set(m.lower() for m in _APA_VOCAB_RE.findall(t))
    has_citation = bool(_CITATION_TOKEN_RE.search(t) or _CASE_RE.search(t))

    grammar_part = min(0.50, 0.25 * n_grammar)
    apa_part = min(0.35, 0.12 * len(apa_terms))
    citation_part = 0.15 if has_citation else 0.0
    return max(0.0, min(1.0, grammar_part + apa_part + citation_part))


def _llm_score(extracted):
    theory = (extracted.get("legal_theory") or "").strip().lower()
    if theory in _NONE:
        return 0.10
    return {
        "statutory conflict": 0.80,
        "procedural defect": 0.75,
        "exceeds authority": 0.85,
        "arbitrary-capricious": 0.80,
        "arbitrary capricious": 0.80,
    }.get(theory, 0.30)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

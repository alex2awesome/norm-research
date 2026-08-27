"""technical_precision hybrid: Engagement with the rule's OWN technical apparatus.

Construct: ~1.0 = comment engages the specific machinery of the proposed rule — cites
preamble sections, docket exhibits, specific proposed section numbers, quotes the agency's
own defined terms, or cites the agency's own data/estimates back at it; ~0.5 = generic
reference to "the proposed rule" or "this regulation" with no specific structural anchor;
~0.0 = no engagement with the rule's own apparatus at all (pure freestanding opinion that
could apply to any rule).

INPUT = comment text. Code sees: proposed-rule-referencing grammar ("proposed § X",
"preamble at N", "the agency estimates", "docket no.", "exhibit N"), and defined-term
detection (capitalized or quoted terms following "as defined in" / "the term ... means").
Code CANNOT verify the comment's characterization of the agency's own data is ACCURATE
(needs the source document) — out of scope for h1.
"""
import re

LLM_FIELDS = {
    "engaged_element": (
        "The most specific element of the proposed rule the comment engages with — a "
        "section number, preamble reference, defined term, docket exhibit, or the agency's "
        "own cited data/estimate — verbatim as stated. Answer NONE if the comment engages no "
        "specific element of the proposal."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_SECTION_REF_RE = re.compile(
    r'\bproposed\s+§+\s*\d[\d.]*|§+\s*\d[\d.]*|\bproposed\s+(?:rule\s+)?(?:section|part)\s+\d+|'
    r'\b\d{1,3}\.\d{1,4}\b', re.I)
_PREAMBLE_RE = re.compile(
    r'\bpreamble\b(?:\s+at\s+(?:page\s+)?\d+)?|\b\d+\s*fed\.?\s*reg\.?\s*\d+|'
    r'\bfederal register\b.{0,20}\bpage\s+\d+', re.I)
_DOCKET_RE = re.compile(
    r'\bdocket\s+(?:no\.?|number)\s*[:#]?\s*[\w-]+|\bexhibit\s+[\dA-Za-z]+|'
    r'\bdocket\s+id\b', re.I)
_AGENCY_DATA_RE = re.compile(
    r'\bthe agency (?:estimates?|found|concluded|stated|projects?|reports?)\b|'
    r'\baccording to (?:the|EPA|FDA|OSHA|the agency\'?s|the Department\'?s) (?:own\s+)?'
    r'(?:data|analysis|estimate|estimates|finding|findings)\b', re.I)
_DEFINED_TERM_RE = re.compile(
    r'\bas defined in\b|\bthe term\s+["“][^"”]{1,40}["”]\s+means\b|'
    r'["“][^"”]{1,40}["”]\s+(?:is defined as|means)\b', re.I)


def _code_score(t):
    n_section = len(_SECTION_REF_RE.findall(t))
    has_preamble = bool(_PREAMBLE_RE.search(t))
    has_docket = bool(_DOCKET_RE.search(t))
    has_agency_data = bool(_AGENCY_DATA_RE.search(t))
    has_defined_term = bool(_DEFINED_TERM_RE.search(t))

    section_part = min(0.35, 0.15 * n_section)
    preamble_part = 0.15 if has_preamble else 0.0
    docket_part = 0.15 if has_docket else 0.0
    agency_data_part = 0.20 if has_agency_data else 0.0
    defined_term_part = 0.15 if has_defined_term else 0.0
    return max(0.0, min(1.0, section_part + preamble_part + docket_part
                         + agency_data_part + defined_term_part))


def _llm_score(extracted):
    el = extracted.get("engaged_element")
    if not isinstance(el, str) or el.strip().lower().strip(". ") in _NONE:
        return 0.10
    specific = bool(re.search(r'\d|§|["“]', el))
    return 0.85 if specific else 0.45


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

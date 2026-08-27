"""citation_validity hybrid: Legal citation precision (MOSTLY CODE).

Construct: ~1.0 = comment cites multiple well-formed, title-valid CFR/USC/Federal-Register
authorities at SECTION-level precision (not just "40 CFR" but "40 CFR 60.14(b)") AND quotes
specific regulatory text; ~0.5 = a citation or two, mostly part-level, no quoted text; ~0.0 =
no legal citation grammar at all (pure opinion, no authorities named).

INPUT = comment text. Code implements a real citation grammar: CFR (title 1-50), USC
(title 1-54), and Federal Register (volume/page) regexes with title-range validity checks,
section- vs part-level precision, and detection of quoted regulatory text (quotes containing
regulation-register vocabulary or sitting near a citation). Code CANNOT verify a citation
actually EXISTS/is correctly quoted (needs an external CFR/USC lookup) — that is a tier-2
fetch hook, out of scope here.
"""
import re

LLM_FIELDS = {
    "quoted_rule_text": (
        "Does the comment quote or closely paraphrase SPECIFIC text/wording from the "
        "proposed rule itself (not just a general reference to it)? If yes, give the quoted "
        "or paraphrased text in <=25 words. Answer NONE if it does not quote specific rule text."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_CFR_RE = re.compile(
    r'\b(\d{1,2})\s*C\.?\s*F\.?\s*R\.?\s*(?:Part\s+)?§{0,2}\s*(\d{1,4})(?:\.(\d{1,4}))?', re.I)
_USC_RE = re.compile(
    r'\b(\d{1,2})\s*U\.?\s*S\.?\s*C\.?\s*(?:§+\s*)?(\d{1,5}[a-zA-Z]?)((?:\([a-zA-Z0-9]{1,4}\))*)', re.I)
_FR_RE = re.compile(r'\b(\d{1,3})\s*Fed\.?\s*Reg\.?\s*(\d{1,6})', re.I)

_REG_VOCAB_RE = re.compile(
    r'\b(shall|shall not|may not|is defined as|means|require[sd]?|prohibit(?:s|ed)?|'
    r'must|is amended to read|the term)\b', re.I)
_QUOTE_RE = re.compile(r'["“]([^"”]{15,400})["”]')


def _valid_cfr(title):
    return 1 <= title <= 50


def _valid_usc(title):
    return 1 <= title <= 54


def _valid_fr(vol):
    return 1 <= vol <= 120


def _authorities(t):
    """Return list of dicts: {type, key(dedup tuple), section_level(bool), span}."""
    out = []
    for m in _CFR_RE.finditer(t):
        title, part, sect = int(m.group(1)), m.group(2), m.group(3)
        if not _valid_cfr(title):
            continue
        out.append(dict(type="CFR", key=("CFR", title, part, sect),
                         section_level=bool(sect), span=m.span()))
    for m in _USC_RE.finditer(t):
        title, sect, subsec = int(m.group(1)), m.group(2), m.group(3)
        if not _valid_usc(title):
            continue
        out.append(dict(type="USC", key=("USC", title, sect),
                         section_level=bool(subsec), span=m.span()))
    for m in _FR_RE.finditer(t):
        vol, page = int(m.group(1)), m.group(2)
        if not _valid_fr(vol):
            continue
        out.append(dict(type="FR", key=("FR", vol, page), section_level=True, span=m.span()))
    return out


def _has_quoted_regulatory_text(t, authorities):
    auth_spans = [a["span"] for a in authorities]
    for m in _QUOTE_RE.finditer(t):
        quote = m.group(1)
        if _REG_VOCAB_RE.search(quote):
            return True
        qs, qe = m.span()
        if any(not (ae < qs - 150 or a0 > qe + 150) for a0, ae in auth_spans):
            return True
    return False


def _code_score(t):
    auths = _authorities(t)
    seen, distinct = set(), []
    for a in auths:
        if a["key"] in seen:
            continue
        seen.add(a["key"])
        distinct.append(a)
    n_valid = len(distinct)
    authority_part = min(0.45, 0.15 * n_valid)

    section_part = 0.0
    if distinct:
        frac_section = sum(1 for a in distinct if a["section_level"]) / len(distinct)
        section_part = 0.25 * frac_section

    quoted_part = 0.30 if _has_quoted_regulatory_text(t, auths) else 0.0
    return max(0.0, min(1.0, authority_part + section_part + quoted_part))


def _llm_score(extracted):
    val = extracted.get("quoted_rule_text")
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return 0.15
    n_words = len(val.split())
    return 0.55 if n_words < 5 else 0.85


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

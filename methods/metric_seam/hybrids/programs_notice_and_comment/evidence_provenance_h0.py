"""evidence_provenance hybrid: Named external evidence + firsthand credentials.

Construct: ~1.0 = comment names specific studies/datasets/reports (Author (Year), et al.,
DOI/URL) AND claims relevant firsthand credentials/experience; ~0.5 = one or the other;
~0.0 = pure unsupported assertion, no named source and no claimed standing/experience.

INPUT = comment text. Code sees: named-source counting via citation-shaped patterns
(Author (Year), et al., DOI, URL) and specific-year density near study/report vocabulary.
Code CANNOT verify the named studies actually exist or say what's claimed (needs an
external literature lookup) — out of scope for h0.
"""
import re

LLM_FIELDS = {
    "named_sources": (
        "Comma-separated named studies, datasets, or reports cited by the comment (e.g. "
        "'EPA 2019 emissions report', 'Smith et al. 2020', 'CDC surveillance dataset'). "
        "Answer NONE if none are named."
    ),
    "firsthand_experience": (
        "In <=20 words, any firsthand credentials or direct experience the commenter "
        "claims that is relevant to the rule (e.g. '20 years as a rural nurse', 'I operate "
        "a 200-acre farm'). Answer NONE if none is claimed."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_AUTHOR_YEAR_RE = re.compile(r'\b[A-Z][A-Za-z\-]+(?:\s+et al\.?)?\s*\(\s*(?:19|20)\d{2}\s*\)')
_ETAL_RE = re.compile(r'\bet al\.?\b', re.I)
_DOI_RE = re.compile(r'\b10\.\d{4,9}/\S+\b')
_URL_RE = re.compile(r'https?://\S+')
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
_STUDY_VOCAB_RE = re.compile(
    r'\b(study|studies|report|survey|data|dataset|finding[s]?|found|showed|research|'
    r'analysis|publication|peer-reviewed)\b', re.I)


def _split_names(val):
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return []
    return [p.strip() for p in re.split(r"[,;]", val) if p.strip() and p.strip().lower() not in _NONE]


def _named_source_count(t):
    matches = set()
    for m in _AUTHOR_YEAR_RE.finditer(t):
        matches.add(("ay", m.group(0).lower()))
    for m in _DOI_RE.finditer(t):
        matches.add(("doi", m.group(0)))
    for m in _URL_RE.finditer(t):
        matches.add(("url", m.group(0)))
    return len(matches)


def _year_density_near_study(t, window=40):
    year_spans = [m.span() for m in _YEAR_RE.finditer(t)]
    study_spans = [m.span() for m in _STUDY_VOCAB_RE.finditer(t)]
    if not year_spans or not study_spans:
        return 0
    n = 0
    for y0, y1 in year_spans:
        if any(not (s1 < y0 - window or s0 > y1 + window) for s0, s1 in study_spans):
            n += 1
    return n


def _code_score(t):
    n_sources = _named_source_count(t)
    n_year_near = _year_density_near_study(t)
    has_etal = bool(_ETAL_RE.search(t))

    source_part = min(0.55, 0.18 * n_sources)
    year_part = min(0.25, 0.08 * n_year_near)
    etal_part = 0.20 if has_etal else 0.0
    return max(0.0, min(1.0, source_part + year_part + etal_part))


def _llm_score(extracted):
    sources = _split_names(extracted.get("named_sources"))
    n = len(sources)
    src_part = {0: 0.1, 1: 0.4}.get(n, 0.65 if n <= 3 else 0.85)
    firsthand = extracted.get("firsthand_experience")
    has_firsthand = isinstance(firsthand, str) and firsthand.strip().lower() not in _NONE
    fh_part = 0.2 if has_firsthand else 0.0
    return max(0.0, min(1.0, src_part * 0.8 + fh_part))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

"""structure_org hybrid: Document organization and structural discipline (CODE ONLY).

Construct: ~1.0 = comment uses legal/markdown-style headers or numbered/lettered structure,
has a clear intro (states who is writing / what rule) and conclusion (closing ask/thanks),
and paragraphs are of regular, deliberate length; ~0.5 = some structure (a list, or an
intro) but not both; ~0.0 = one undifferentiated block of text with no headers, no lists,
no intro/conclusion framing.

INPUT = comment text only. This is a pure-code aspect (no LLM extraction needed — document
structure is directly observable from formatting/punctuation, not semantic content code
can't see). Uses ops.sent_stats to penalize unbroken run-on structure.
"""
import re

LLM_FIELDS = {}

_HEADER_RE = re.compile(
    r'(?m)^\s*(?:[IVX]{1,4}\.\s+\S|[A-H]\.\s+\S|RE:\s|Dear\s|[A-Z][A-Z ]{6,60}$)')
_ENUM_RE = re.compile(
    r'(?m)^\s*(?:\(?[a-hA-H1-9]\)|[1-9]\.\s|first,|second,|third,)', re.I)
_INTRO_RE = re.compile(
    r'\b(i am writing|on behalf of|submit(?:s|ted)? (?:these|this) comment|these comments '
    r'are submitted|re:\s|i am (?:a|an)\s|as a\s)\b', re.I)
_CONCLUSION_RE = re.compile(
    r'\b(thank you|sincerely|respectfully submitted|we urge|for (?:these|the foregoing) '
    r'reasons|in conclusion|we appreciate|please (?:consider|contact))\b', re.I)


def _paragraphs(t):
    return [p.strip() for p in re.split(r'\n\s*\n', t) if p.strip()]


def _paragraph_regularity(paras):
    if len(paras) < 3:
        return 0.0
    lens = [len(p.split()) for p in paras]
    mean = sum(lens) / len(lens)
    if mean <= 0:
        return 0.0
    var = sum((l - mean) ** 2 for l in lens) / len(lens)
    cv = (var ** 0.5) / mean
    return max(0.0, min(1.0, 1.0 - min(1.0, cv)))


def _code_score(t, ops):
    n_headers = len(_HEADER_RE.findall(t))
    n_enum = len(_ENUM_RE.findall(t))
    paras = _paragraphs(t)
    has_intro = bool(_INTRO_RE.search(t[:400]))
    has_conclusion = bool(_CONCLUSION_RE.search(t[-400:])) if len(t) > 20 else False

    header_part = min(0.20, 0.10 * n_headers)
    list_part = min(0.25, 0.08 * n_enum)
    if len(paras) >= 3:
        para_count_part = 0.15
    elif len(paras) == 2:
        para_count_part = 0.07
    else:
        para_count_part = 0.0
    intro_concl_part = (0.10 if has_intro else 0.0) + (0.10 if has_conclusion else 0.0)
    regularity_part = 0.20 * _paragraph_regularity(paras)

    s = header_part + list_part + para_count_part + intro_concl_part + regularity_part

    # penalize unbroken run-on structure (very long avg sentence, few sentence breaks)
    if ops and hasattr(ops, "sent_stats"):
        try:
            n_sent, mean_wps, _ = ops.sent_stats(t)
            if n_sent <= 2 and mean_wps > 60:
                s -= 0.10
        except Exception:
            pass
    return max(0.0, min(1.0, s))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        return _code_score(t, ops)
    except Exception:
        return 0.5

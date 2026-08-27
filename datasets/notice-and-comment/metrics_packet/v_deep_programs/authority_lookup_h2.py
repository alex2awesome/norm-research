"""authority_lookup hybrid: Do the comment's cited legal authorities actually EXIST?
(VERIFICATION-TIER — code looks each citation up against a real eCFR-derived index, not just
checks that it is syntactically well-formed.)

Construct: ~1.0 = the comment's authorities are real (its CFR parts exist in the current Code
of Federal Regulations, ideally at verified section-level precision) AND the load-bearing
authority the argument most depends on is itself real; ~0.5-ish = only USC/Federal-Register
cites (no existence lookup available for those, only syntactic validity) or no authorities at
all; ~0.0 = the comment's cited CFR parts are FABRICATED (don't exist in the index) — this is
the sharpest failure mode, especially when the load-bearing authority is fake.

INPUT = comment text. Code parses CFR (title 1-50, part, optional section), USC (title 1-54,
no lookup possible), and Federal Register (volume/page, with volume<->year coherence check:
FR volume ~= year - 1935 +/- 1 when both appear near each other) citations, using the same
grammar family as citation_validity_h0. CFR parts are looked up against a static index built
from the eCFR structure API (build_cfr_index.py -> cfr_parts_index.json.gz), loaded lazily at
module level; if that index is missing, the module degrades to syntactic-validity-only (never
crashes). Code CANNOT verify USC section text or that a CFR citation is quoted CORRECTLY
(needs full-text lookup, out of scope for h2) — only that the cited PART exists.
"""
import gzip
import json
import pathlib
import re

LLM_FIELDS = {
    "authority_relied_on": (
        "The single legal authority (CFR section/part, USC section, or Federal Register "
        "citation) that the comment's argument most depends on, verbatim. Answer NONE if the "
        "comment does not rely on a specific legal authority."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

# ---------------------------------------------------------------------------
# citation grammar (same family as citation_validity_h0.py)
# ---------------------------------------------------------------------------

_CFR_RE = re.compile(
    r'\b(\d{1,2})\s*C\.?\s*F\.?\s*R\.?\s*(?:Part\s+)?§{0,2}\s*(\d{1,4}[A-Za-z]?)(?:\.(\d{1,4}[a-zA-Z]?))?', re.I)
_USC_RE = re.compile(
    r'\b(\d{1,2})\s*U\.?\s*S\.?\s*C\.?\s*(?:§+\s*)?(\d{1,5}[a-zA-Z]?)((?:\([a-zA-Z0-9]{1,4}\))*)', re.I)
_FR_RE = re.compile(r'\b(\d{1,3})\s*Fed\.?\s*Reg\.?\s*(\d{1,6})', re.I)
_YEAR_NEAR_RE = re.compile(r'\b(19|20)(\d{2})\b')


def _valid_cfr_title(title):
    return 1 <= title <= 50


def _valid_usc_title(title):
    return 1 <= title <= 54


def _valid_fr_vol(vol):
    return 1 <= vol <= 120


def _parse_authorities(t):
    """Return list of dicts: {type, title, part, section, section_level, span, key}."""
    out = []
    for m in _CFR_RE.finditer(t):
        title, part, sect = int(m.group(1)), m.group(2), m.group(3)
        if not _valid_cfr_title(title):
            continue
        out.append(dict(type="CFR", title=title, part=part, section=sect,
                         section_level=bool(sect), span=m.span(),
                         key=("CFR", title, part.lower(), sect)))
    for m in _USC_RE.finditer(t):
        title, sect, subsec = int(m.group(1)), m.group(2), m.group(3)
        if not _valid_usc_title(title):
            continue
        out.append(dict(type="USC", title=title, part=sect, section=subsec or None,
                         section_level=bool(subsec), span=m.span(),
                         key=("USC", title, sect)))
    for m in _FR_RE.finditer(t):
        vol, page = int(m.group(1)), m.group(2)
        if not _valid_fr_vol(vol):
            continue
        out.append(dict(type="FR", title=None, part=None, vol=vol, page=page,
                         section_level=True, span=m.span(), key=("FR", vol, page)))
    out.sort(key=lambda a: a["span"][0])
    return out


def _dedup_authorities(t):
    seen, distinct = set(), []
    for a in _parse_authorities(t):
        if a["key"] in seen:
            continue
        seen.add(a["key"])
        distinct.append(a)
    return distinct


def _fr_year_coherent(t, a, window=60):
    """None = no nearby year to check against; True/False = coherent / incoherent."""
    s, e = a["span"]
    ctx = t[max(0, s - window):e + window]
    years = [int(m.group(0)) for m in _YEAR_NEAR_RE.finditer(ctx)]
    if not years:
        return None
    expected = a["vol"] + 1935
    return any(abs(y - expected) <= 1 for y in years)


# ---------------------------------------------------------------------------
# CFR part-existence index (lazy load; degrade to syntactic-only if missing)
# ---------------------------------------------------------------------------

_INDEX_PATH = pathlib.Path(__file__).resolve().parent / "cfr_parts_index.json.gz"
if not _INDEX_PATH.exists():  # fall back to the source-repo location
    _INDEX_PATH = (pathlib.Path(__file__).resolve().parents[4]
                   / "datasets" / "notice-and-comment" / "v4" / "cfr_parts_index.json.gz")


def _load_cfr_index():
    try:
        with gzip.open(_INDEX_PATH, "rt", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {str(k): {str(p).lower() for p in v} for k, v in raw.items()}
    except Exception:
        return None


_CFR_INDEX = _load_cfr_index()


def _cfr_part_exists(title, part):
    """True/False if the index is loaded, None if the index is unavailable (unknown)."""
    if _CFR_INDEX is None:
        return None
    parts = _CFR_INDEX.get(str(title))
    if parts is None:
        return False
    return str(part).lower() in parts


# ---------------------------------------------------------------------------
# code channel
# ---------------------------------------------------------------------------

def _code_score(t):
    auths = _dedup_authorities(t)
    if not auths:
        return 0.30  # nothing to verify, and the construct is ABOUT authorities existing

    cfr = [a for a in auths if a["type"] == "CFR"]
    usc = [a for a in auths if a["type"] == "USC"]
    fr = [a for a in auths if a["type"] == "FR"]

    index_unavailable = _CFR_INDEX is None and bool(cfr)

    for a in cfr:
        a["verified"] = None if index_unavailable else _cfr_part_exists(a["title"], a["part"])

    n_cfr = len(cfr)
    n_cfr_verified = sum(1 for a in cfr if a.get("verified") is True)
    n_cfr_section_verified = sum(1 for a in cfr if a.get("verified") is True and a["section_level"])

    fr_coherence = [_fr_year_coherent(t, a) for a in fr]
    fr_ok = sum(1 for c in fr_coherence if c is True)
    fr_bad = sum(1 for c in fr_coherence if c is False)

    n_verified_authorities = n_cfr_verified + len(usc) + fr_ok

    if index_unavailable:
        # degrade to syntactic-validity-only (mirrors citation_validity_h0's scoring shape)
        n_total = len(auths)
        authority_part = min(0.45, 0.15 * n_total)
        section_part = 0.0
        if auths:
            frac_section = sum(1 for a in auths if a["section_level"]) / len(auths)
            section_part = 0.25 * frac_section
        return max(0.0, min(1.0, authority_part + section_part))

    existence_part = 0.0
    if n_cfr:
        existence_part = 0.45 * (n_cfr_verified / n_cfr)
        if n_cfr_verified == 0:
            existence_part -= 0.15  # every cited CFR authority is fabricated
    else:
        existence_part = 0.10 * min(len(usc) + len(fr), 2)  # no CFR to check; modest credit

    count_part = min(0.25, 0.06 * n_verified_authorities)
    section_part = 0.20 * (n_cfr_section_verified / n_cfr_verified) if n_cfr_verified else 0.0
    fr_penalty = 0.15 * min(fr_bad, 2)

    return max(0.0, min(1.0, existence_part + count_part + section_part - fr_penalty))


def _llm_score(extracted):
    val = extracted.get("authority_relied_on")
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return 0.15  # no load-bearing authority named
    auths = _dedup_authorities(val)
    if not auths:
        return 0.20  # named "authority" doesn't even parse as a real citation
    a = auths[0]
    if a["type"] == "CFR":
        if _CFR_INDEX is None:
            return 0.55  # can't verify existence; syntactically valid CFR cite
        exists = _cfr_part_exists(a["title"], a["part"])
        if exists:
            return 0.95 if a["section_level"] else 0.80
        return 0.05  # the load-bearing citation is fabricated
    if a["type"] == "USC":
        return 0.60  # syntactically valid title, no existence lookup available
    if a["type"] == "FR":
        coherent = _fr_year_coherent(val, a)
        if coherent is False:
            return 0.10
        return 0.55
    return 0.30


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5

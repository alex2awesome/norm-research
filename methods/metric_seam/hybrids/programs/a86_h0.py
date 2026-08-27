"""a86 Quote quality, sourcing, and usefulness — hybrid channel (h0).

Design: PRESENCE is split from QUALITY so that a generic attributed boilerplate
quote cannot move the score (CF gate). A quoted span only earns credit if it
passes a specificity predicate (numbers / inner proper nouns / length /
discourse connectives) net of an enthusiasm-boilerplate penalty, with a hard
gate: >=2 boilerplate hits and zero concrete content -> ~0 credit. Attribution
(said-verb + Named speaker, 'according to', em-dash sign-off, headline colon)
scales the credit; docs whose only "quotes" are scare quotes / headline
citations (no said-verb attribution) are capped low. Bonuses for multiple
good quotes, distinct speakers, and independent voices (analyst, professor,
customer...). LLM fields supply thick grounding (quote may sit mid-document,
attribution may be verb-free); extracted text is re-scored by the SAME code
predicate, so an extracted boilerplate quote still earns nothing.
"""
import re

LLM_FIELDS = {
    "best_quote_insight": ("Quote verbatim the single most substantive statement "
                           "attributed to a named person in this document, or NONE"),
    "quote_speakers": ("List the named people quoted and their roles, e.g. "
                       "'Jane Doe, CEO; Raj Patel, analyst', or NONE"),
}

# ---------------- lexicons ----------------
_BOILER = re.compile(
    r"\b(?:thrilled|excited|exciting|excitement|delighted|pleased|proud|honou?red|"
    r"humbled|look(?:ing)?\s+forward|journey|milestone|world[- ]class|best[- ]in[- ]class|"
    r"incredible|incredibly|amazing|awesome|fantastic|game[- ]chang\w+|"
    r"couldn'?t\s+be\s+(?:more|proud|happier)|could\s+not\s+be\s+(?:more|prouder|happier))\b",
    re.I)

_CONNECTIVE = re.compile(
    r"\b(?:because|but|however|although|while|instead|rather\s+than|not\s+only|"
    r"so\s+that|which|unless|despite|as\s+though|whereas|compared\s+(?:to|with)|"
    r"means\s+that|without)\b", re.I)

_SPECIFIC_NUM = re.compile(r"\d|\$|%")

_ARTIFACT = re.compile(
    r"https?://|www\.|href|\.html|\.pdf|\.aspx|\.com\b|</|/>|[{}<>\\]|=\"|&\w{2,6};|"
    r"terms\s+of\s+use|all\s+rights\s+reserved|click\s+here|cookie", re.I)

_SAID = (r"(?:said|says|stated|noted|added|explained|commented|remarked|observed|"
         r"warned|wrote|told|argued|concluded|continued|emphasi[sz]ed|state[sd]?)")
_NAME = r"[A-Z][\w'\.\-]*(?:\s+[A-Z][\w'\.\-]*){0,3}"

_TITLE = re.compile(
    r"\b(?:CEO|CFO|CTO|COO|Chief|President|Chairman|Chairwoman|Director|Founder|"
    r"Co-Founder|Manager|VP|Vice\s+President|Head\s+of|Professor|Dr\.|Officer|"
    r"Spokesperson|spokesman|spokeswoman|General\s+Counsel|Secretary|analyst|"
    r"economist|researcher|scientist|engineer|student|trader|partner)\b")

_INDEP = re.compile(
    r"\b(?:analyst|professor|researcher|student|customer|client|scientist|economist|"
    r"trader|expert|consultant|advocate|resident|patient|investor|academic|"
    r"independent)\b", re.I)

_CAP_STOP = {"I", "We", "You", "It", "The", "This", "That", "These", "Those", "Our",
             "My", "Your", "Their", "A", "An", "And", "But", "Or", "In", "On", "At",
             "As", "To", "For", "If", "When", "While", "Until", "Since", "There",
             "What", "Who", "How", "Why", "Let's", "Let", "Every", "All", "No", "Not"}

_A1 = re.compile(r"^\s*[,\.\-–—:]*\s*" + _SAID + r"\s+(?:by\s+)?(" + _NAME + ")")
_A2 = re.compile(r"^\s*[,\.\-–—:]*\s*(" + _NAME + r")(?:\s*,[^\"]{0,80}?)?\s+" + _SAID)
_A3 = re.compile(r"^\s*[-–—]{1,2}\s*(" + _NAME + ")")
_A4 = re.compile(r"according\s+to\s+(" + _NAME + ")", re.I)
_B1 = re.compile(r"(" + _NAME + r")[^\"]{0,60}\b" + _SAID + r"[\s,:]*$")
_B1b = re.compile(r"\b" + _SAID + r"[\s,:]*$")
_B2 = re.compile(r"(" + _NAME + r")\s*:\s*$")


def _words(s):
    return re.findall(r"[A-Za-z']+", s)


def _inner_propers(span):
    """Capitalized tokens that are not sentence-initial and not stop-caps."""
    count, seen = 0, set()
    for m in re.finditer(r"\b([A-Z][\w'\.\-]+|[A-Z]{2,})\b", span):
        tok = m.group(1)
        if tok in _CAP_STOP or len(tok) < 2:
            continue
        j = m.start() - 1
        while j >= 0 and span[j] in " \t\n":
            j -= 1
        if j < 0 or span[j] in '.!?":;(':  # sentence-initial / quote-initial
            continue
        if tok not in seen:
            seen.add(tok)
            count += 1
    return count


def _quality(span, min_words=8):
    """Specificity-net-of-boilerplate quality of a quoted span, in [0,1].
    A generic enthusiasm quote (thrilled/excited/milestone...) with no concrete
    content is hard-gated to ~0 — this is the CF-safety predicate."""
    span = span.strip()
    if _ARTIFACT.search(span) or span.count("\n") > 4:
        return 0.0
    w = _words(span)
    if len(w) < 5:
        return 0.0
    alpha = sum(len(x) for x in w)
    if alpha < 0.5 * max(1, len(span)):
        return 0.0
    spec = 0.0
    if len(w) >= min_words:
        spec += 0.25
    if len(w) >= 2 * min_words:
        spec += 0.15
    if len(w) >= 28:
        spec += 0.05
    has_num = bool(_SPECIFIC_NUM.search(span))
    propers = _inner_propers(span)
    if has_num:
        spec += 0.30
    spec += 0.15 * min(2, propers)
    if _CONNECTIVE.search(span):
        spec += 0.15
    spec = min(1.0, spec)
    hits = len(_BOILER.findall(span))
    if hits >= 2 and not has_num and propers == 0:
        return 0.03  # hard gate: pure enthusiasm boilerplate
    eff = max(0, hits - (1 if (has_num or propers) else 0))
    return max(0.0, spec - 0.25 * eff)


def _attribution(before, after):
    """(weight, name_or_None, strong_bool). strong = said-verb/according-to/em-dash."""
    m = _A1.search(after)
    if m:
        return 1.0, m.group(1), True
    m = _A2.search(after)
    if m:
        return 1.0, m.group(1), True
    m = _B1.search(before)
    if m:
        return 1.0, m.group(1), True
    m = _A4.search(after) or _A4.search(before[-120:])
    if m:
        return 1.0, m.group(1), True
    m = _A3.search(after)
    if m:
        return 0.85, m.group(1), True
    if _B1b.search(before):
        return 0.7, None, True   # '... he said, "..."' (unnamed but verb-attributed)
    m = _B2.search(before)
    if m:
        return 0.55, m.group(1), False  # headline 'Name: "..."' citation
    return 0.3, None, False


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0

        creds = []          # (credit, name, strong, context)
        for m in re.finditer(r'"([^"]{15,600})"', t):
            span = m.group(1)
            q = _quality(span)
            if q <= 0.0:
                continue
            before = t[max(0, m.start() - 140):m.start()]
            after = t[m.end():m.end() + 110]
            wgt, name, strong = _attribution(before, after)
            ctx = before[-120:] + " " + after
            if _TITLE.search(ctx):
                wgt = min(1.0, wgt + 0.1)
            creds.append((q * wgt, name, strong, ctx))

        # ---- thick-input grounding (must degrade gracefully when empty) ----
        ext = extracted or {}
        bq = (ext.get("best_quote_insight") or "").strip()
        if bq and bq.upper() not in ("NONE", "N/A", "NULL"):
            # re-score the extractor's quote with the SAME predicate: an
            # extracted boilerplate quote still earns nothing (CF-safe).
            qq = _quality(ops.normalize(bq), min_words=6)
            if qq > 0.0:
                creds.append((0.9 * qq, None, True, bq))

        goods = [c for c in creds if c[0] >= 0.15]
        if not goods:
            best = max((c[0] for c in creds), default=0.0)
            return round(min(0.10, 0.4 * best), 4)

        top = max(c[0] for c in goods)
        strong_attr = any(c[2] for c in goods)
        speakers = {c[1].split()[-1].strip(".,") for c in goods if c[1]}
        indep = any(_INDEP.search(c[3] or "") for c in goods)

        sp = (ext.get("quote_speakers") or "").strip()
        if sp and sp.upper() not in ("NONE", "N/A", "NULL"):
            for nm in re.split(r"[;\n]+", sp):
                nmm = re.match(r"\s*(" + _NAME + ")", nm)
                if nmm:
                    speakers.add(nmm.group(1).split()[-1].strip(".,"))
            if _INDEP.search(sp):
                indep = True

        s = 0.42 + 0.40 * min(1.0, top)
        s += 0.07 * min(2, len(goods) - 1)
        if len(speakers) >= 2:
            s += 0.06
        if indep:
            s += 0.05
        if not strong_attr:
            s = min(s, 0.35)  # scare-quote / headline-citation docs stay low
        return round(max(0.0, min(1.0, s)), 4)
    except Exception:
        return 0.0

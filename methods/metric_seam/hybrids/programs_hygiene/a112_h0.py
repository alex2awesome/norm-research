"""a112 hybrid: Related links, sources, and navigational coherence.

Predicate: a document scores high when it has SUBSTANTIVE BODY PROSE *and* a
coherent linking apparatus around that prose (explicit URLs, labeled resource
links like webcast/transcript/report/press-kit/download, related-content
sections). Nav-chrome pages (menus, listings, 404s) have links but no prose ->
gated to ~0. Real releases with no links (dense earnings text) have prose but
no link quality -> ~0. Keyword presence alone is capped low: quality comes from
distinct signal families tied to a prose gate, not from raw cue counts.
"""
import re

LLM_FIELDS = {
    "page_kind": "One word: is this document a 'release', 'article', 'navigation', 'listing', or 'error' page?",
    "related_resources": "List up to 3 short phrases where this document links to related/background/source material (related news, reports, downloads); NONE if none.",
}

# --- link-signal patterns (lowercased text) -------------------------------
_URL_STRONG_PAT = re.compile(r"https?://\S+|www\.\S+")
_URL_BARE_PAT = re.compile(r"\b[a-z0-9][\w-]*\.(?:com|org|net|gov|edu)(?:/\S*)?\b")
_URL_CTX_PAT = re.compile(r"(?:\bat|\bvisit\w*|\bgo to|available|\band)\s*$")
_EMAIL_PAT = re.compile(r"\S+@\S+")

_RESOURCE_PATS = [
    re.compile(p) for p in (
        r"\bwebcasts?\b",
        r"\btranscripts?\b",
        r"\bpress kits?\b",
        r"\bfact sheets?\b",
        r"\bwhite ?papers?\b",
        r"\bfull report\b",
        r"\bdownload",
        r"\breplay\b",
        r"\bblog post\b",
        r"\bpdf\b",
        r"\(\d+(?:\.\d+)? ?[km]b\)",                       # labeled file size
        r"\brss feed\b|\bsubscribe to rss\b",
        r"\b(?:buying|strategy|user) guide\b",
        r"\b(?:ver|see) la gu[ií]a\b",
        # action+resource: "read the full report", "get the strategy guide",
        # "view our past GXI reports" -- verb-anchored so an injected bare
        # keyword ("report") does not fire.
        r"\b(?:view|read|download|get|see)\b[\w ,'-]{0,25}\b(?:reports?|guides?|papers?|kits?|filings?)\b",
    )
]
# each match is a distinct topical deep link ("More information about X ...")
_DEEPLINK_PAT = re.compile(
    r"\bmore information (?:about|on)\b"
    r"|\bm[aá]s informaci[oó]n (?:sobre|acerca)\b|\ben savoir plus\b")
# titled external source citation: "Nexusguard's DDoS Threat Report", "United
# Nations' The World's Cities in 2018 Data Booklet" (case-sensitive)
_CITATION_PAT = re.compile(
    r"(?:'s|s') [A-Z][\w ,'&-]{5,60}(?:Report|Index|Booklet|Study|Survey|Whitepaper)\b")

_RELATED_PATS = [
    re.compile(p) for p in (
        r"\brelated (?:news|articles?|links?|releases?|features?|stories)\b",
        r"\bread this next\b",
        r"\bmore news\b",
        r"\b(?:news|press(?: release)?|results?) archives?\b",
        r"\bview all [\w ]{0,15}(?:news|releases|stories)\b",
        r"\bprevious coverage\b",
        r"(?m)^\s*more:\s",        # newsroom "More: <press kit> | <video>" line
    )
]

_GENERIC_PATS = [
    re.compile(p) for p in (
        r"\blearn more\b",
        r"\bread more\b",
        r"\bclick here\b",
        r"\bsee also\b",
        r"\bfind out more\b",
        r"\bfor more information\b",
        r"\bvisit us\b",
    )
]

_NAV_KINDS = ("navigation", "menu", "listing", "index", "error", "home")
_DOC_KINDS = ("release", "article", "story", "blog", "report")
# NOTE (hygiene patch): page_kind is meant to be a short (one-word-ish)
# category answer, but observed extractions are sometimes verbose free text.
# Bare `k in kind` false-fired on unrelated words that merely contain a
# category token as a substring: "error" inside "terrorism"/"terrorist",
# "home" inside "homepage"/"homeland"/"homeowners", "story" inside
# "history"/"historypersonal", "blog" is fine but guarded anyway, "release"
# inside "released"/"unreleased" (also a negation-drift risk). \b-anchor
# with a light plural/adjective whitelist preserves the intended matches.
_NAV_KINDS_RE = re.compile(
    r"\b(?:navigation\w*|menus?|listings?|index(?:es)?|errors?|homes?)\b")
_DOC_KINDS_RE = re.compile(
    r"\b(?:release[sd]?|articles?|stor(?:y|ies)|blogs?|reports?)\b")


def _prose_sentences(t):
    """Count real body sentences: per line (menu items live on their own short
    lines and lack terminal punctuation, so they never concatenate into fake
    prose), >=12 words, >=7 fully-lowercase words (function words -- Title
    Case footer/menu strings fail this), ends with punctuation."""
    n = 0
    for line in t.split("\n"):
        line = line.strip()
        if len(line) < 60:
            continue
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip()
            if not s or s[-1] not in ".!?\"'”)":
                continue
            words = re.findall(r"[A-Za-z][\w'-]*", s)
            if len(words) < 12:
                continue
            lower = sum(1 for w in words if w.islower())
            if lower >= 7:
                n += 1
    return n


def _n_distinct(pats, t):
    return sum(1 for p in pats if p.search(t))


def score(text, extracted, ops):
    try:
        t = ops.normalize(text or "")
        if len(t.strip()) < 40:
            return 0.0
        tl = t.lower()

        # ---- content gate: is there substantive prose to navigate FROM? ----
        prose_n = _prose_sentences(t)
        gate = min(1.0, prose_n / 6.0)
        if tl.count("{{") >= 3:                 # unrendered template junk
            gate *= 0.3
        if len(re.findall(r"[A-Za-z][\w'-]*", t)) < 120:
            gate *= 0.5

        # optional LLM thick-input adjustment (works fine when absent)
        kind = (extracted or {}).get("page_kind", "").strip().lower()
        if kind:
            if _NAV_KINDS_RE.search(kind):
                gate *= 0.3
            elif _DOC_KINDS_RE.search(kind):
                gate = max(gate, 0.7)

        # ---- link quality: distinct families, each capped -----------------
        no_email = _EMAIL_PAT.sub(" ", tl)
        urls = set()
        for m in _URL_STRONG_PAT.finditer(no_email):
            dom = re.sub(r"^(?:https?://)?(?:www\.)?", "",
                         m.group(0).rstrip(".,);"))
            urls.add(dom)
        for m in _URL_BARE_PAT.finditer(no_email):
            # a bare domain counts only as an actual pointer: it has a path,
            # or sits in "visit / at / go to / available" context
            if "/" in m.group(0) or _URL_CTX_PAT.search(
                    no_email[max(0, m.start() - 14):m.start()].rstrip()):
                urls.add(re.sub(r"^www\.", "", m.group(0).rstrip(".,);")))
        u = min(len(urls), 4)
        resource = min(4, _n_distinct(_RESOURCE_PATS, tl)
                       + min(2, len(_CITATION_PAT.findall(t))))
        deeplink = min(3, len(_DEEPLINK_PAT.findall(tl)))
        related = min(2, _n_distinct(_RELATED_PATS, tl))
        generic = 0
        for p in _GENERIC_PATS:
            generic += len(p.findall(tl))
        generic = min(5, generic)

        q = (0.07 * u             # distinct explicit URLs
             + 0.0625 * resource  # labeled resource links / source citations
             + 0.14 * related     # related-content sections
             + 0.075 * deeplink   # topical "more information about X" links
             + 0.02 * generic)    # weak navigational keywords

        rel = (extracted or {}).get("related_resources", "").strip()
        if rel and rel.lower() not in ("none", "n/a", "no", ""):
            items = [s for s in re.split(r"[;,]", rel) if s.strip()]
            q += 0.08 * min(3, len(items))

        return max(0.0, min(1.0, gate * min(1.0, 1.3 * q)))
    except Exception:
        return 0.5

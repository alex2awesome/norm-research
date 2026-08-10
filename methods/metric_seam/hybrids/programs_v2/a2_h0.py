"""a2_h0 — Attribution and sourcing clarity (citations, links, credentials).

Hybrid channel: code predicate + optional LLM thick-extraction fields.
Judge construct: clear citations/links to accessible evidence, speaker
credentials/bios, named attribution; penalize vague sourcing and pure
self-reference ("reliance solely on the press release"). Non-substantive
pages (nav chrome, product/ToS pages) are damped via a prose gate, NOT
via keyword presence.

Design notes:
- PRESENCE != QUALITY: every signal requires structure (quote + "said" +
  Name, Name + comma + title, legal-citation shapes, URL shapes, named
  contact adjacent to a phone/email), so injecting bare keywords like
  "according to" or "said" into a low doc moves the score very little.
- Self-reference check: a single distinct attributed speaker (canned PR
  quote) is worth less than multiple independent speakers/sources; a
  diversity multiplier rewards independent evidence FAMILIES (external
  attribution, formal citations, bylines, rich credentials, multiple
  explicit URLs) over one signal type repeated.
- Works with extracted == {} (code-only). LLM fields only max-boost the
  corresponding sub-signal; they cannot create a high score on their own.
"""
import math
import re

LLM_FIELDS = {
    "external_sources": "List up to 3 named external sources, documents, or datasets this text explicitly cites or links; answer NONE if none.",
    "speaker_credentials": "Give the name plus stated title or credential of one quoted/attributed speaker in the text; answer NONE if none.",
}


def _sat(x, k):
    """Saturating count -> [0,1)."""
    return 1.0 - math.exp(-float(x) / k)


# --- pattern library -------------------------------------------------------
_ATTR_VERBS = r"(?:said|says|added|noted|stated|charged|explained|commented|wrote|told)"
_TITLE_CORE = (
    r"(?:president|chief|director|professor|secretary|analyst|officer|founder|"
    r"head|chair(?:man|woman)?|manager|executive|vice[- ]president|vp|ceo|cto|"
    r"cfo|coo|economist|scientist|engineer|researcher|spokes\w+|dean|partner|"
    r"strategist|dr\b)"
)
_NAME = r"[A-Z][a-zA-Z'.\-]+(?:\s+[A-Z][a-zA-Z'.\-]*\.?){0,3}"

# quote ... " said Name   (capture name token for speaker counting)
_P_QUOTE_SAID = re.compile(
    r'"[^"\n]{15,400}"\s*[,.]?\s*' + _ATTR_VERBS + r"\s+([A-Z][a-zA-Z'.\-]+)")
# said Name, <up to 2 modifiers> Title
_P_SAID_NAME_TITLE = re.compile(
    _ATTR_VERBS + r"\s+(" + _NAME + r")"
    r"\s*,\s*(?:the\s+|a\s+|an\s+)?(?:former\s+)?(?:[a-z]+\s+){0,2}" + _TITLE_CORE,
    re.IGNORECASE)
# Name said/told/wrote (reverse-order, news-style attribution)
_P_NAME_SAID = re.compile(
    r"\b([A-Z][a-zA-Z'.\-]{2,})\s+(?:said|told|wrote|added|noted)\b")

_P_URL_FULL = re.compile(r"https?://[^\s\"'<>)\]]+|\bwww\.[a-z0-9\-]+\.[a-z]{2,}(?:/[^\s\"'<>)\]]*)?")
_P_URL_BARE = re.compile(r"\b[a-z0-9\-]{2,}\.(?:com|org|gov|edu|net)\b(?:/[^\s\"'<>)\]]*)?")

_P_CRED = re.compile(
    r"\bph\.?\s?d\b|\bm\.d\.\b|\bprofessor\b|\bresearch director\b|"
    r"\bmanaging director\b|\bchief \w+ officer\b|"
    r"\bformer\s+(?:united states\s+|u\.?s\.?\s+)?(?:secretary|director|governor|chairman|president)\b|"
    r"\bexecutive vice president\b|\bsenior (?:vice president|analyst|director|fellow)\b|"
    r"\bfellow at\b|\bdr\.\s+[A-Z]|\bgraduate student\b|\bpostdoctoral\b",
    re.IGNORECASE)

_P_EXT = re.compile(
    r"\b(?:according to|as reported by|cited by|reported by|in a report by|"
    r"obtained by|surveyed by|estimates? from|data from|published (?:by|in)|"
    r"told)\s+(?:the\s+)?[A-Z]")
_P_PATH = re.compile(
    r"\b(?:available at|for more (?:information|details),?\s*(?:please\s+)?visit|"
    r"read the full (?:report|study|article)|see (?:the )?full|view the full|"
    r"\(see table\s*\d|download (?:the|a) (?:report|pdf|paper))",
    re.IGNORECASE)

_P_FORMAL = re.compile(
    r"\bPub\.?\s?L\.\s?(?:No\.?)?\s?\d|\bRelease No\.\s?[A-Z0-9\-]+|"
    r"\b\d{1,3}\s+FR\s+\d{2,6}\b|\bpursuant to\b|\bet al\.|"
    r"[a-z]\s\d{1,2}\s\.")  # last: spaced footnote refs e.g. "every year 1 ."

_P_BYLINE = re.compile(
    r"\bBy\s+[A-Z][a-z]+\s+[A-Z][a-z]+|\b(?:Author|Editor)\s*\n?\s*[A-Z][a-z]+\s+[A-Z]")

_P_PHONE = re.compile(r"\(?\b\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b")
_P_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[A-Za-z]{2,}\b")
_P_NAMEISH = re.compile(r"[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+")

_VAGUE = ("experts say", "sources say", "it is believed", "reportedly",
          "rumor", "leak", "insiders say", "sources familiar", "unnamed",
          "anonymous source", "some claim", "insinuated", "speculat")

_NONE_RE = re.compile(r"^\s*(?:none|n/?a|no\b.*)?\s*$", re.IGNORECASE)

# weights (sum to 1.0): external/verifiable evidence outweighs self-sourced
_W = {"url": 0.24, "quote": 0.14, "cred": 0.10, "ext": 0.17,
      "formal": 0.21, "byline": 0.12, "contact": 0.02}
_NORM = 0.55  # typical strong-doc core; rescales composite into [0,1]


def _field(extracted, key):
    """Return a non-trivial LLM answer or ''."""
    v = (extracted or {}).get(key, "") or ""
    return "" if _NONE_RE.match(v.strip()) else v.strip()


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        tl = t.lower()

        # ---- substantive-prose gate (non-release / nav-chrome damping) ----
        lines = [l.strip() for l in t.split("\n") if l.strip()]
        nav = sum(1 for l in lines
                  if len(l.split()) <= 4 and not re.search(r"[.!?]$", l))
        nav_frac = (nav / len(lines)) if lines else 1.0
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        long_sents = sum(1 for s in sents
                         if len(s.split()) >= 15 and re.search(r"[a-z]", s))
        gate = 0.10 + 0.90 * min(1.0, long_sents / 6.0)
        # nav damping only when there is no real prose body behind the chrome
        if nav_frac > 0.55 and long_sents < 10:
            gate *= max(0.45, 1.0 - (nav_frac - 0.55))

        # ---- sourcing signals (structure-required, saturating) ----
        # explicit URLs = accessible evidence paths (bare domains half-weight)
        full_urls = _P_URL_FULL.findall(t)
        bare = [u for u in _P_URL_BARE.findall(tl)
                if not any(u in f.lower() for f in full_urls)]
        u_eff = len(full_urls) + 0.5 * len(bare) + 0.7 * len(_P_PATH.findall(t))
        s_url = _sat(u_eff, 3.0)

        # named attribution; count distinct speakers to catch the canned
        # single-self-quote PR pattern (self-reference is worth less)
        m1 = _P_QUOTE_SAID.findall(t)
        m2 = [m if isinstance(m, str) else m for m in _P_SAID_NAME_TITLE.findall(t)]
        m3 = _P_NAME_SAID.findall(t)
        q = len(m1) + len(m2) + 0.6 * min(4, len(m3))
        speakers = set()
        for name in list(m1) + list(m2) + list(m3):
            last = str(name).split()[-1].strip(".,'\"").lower()
            if len(last) > 2:
                speakers.add(last)
        s_quote = _sat(q, 2.0)
        if len(speakers) <= 1:
            s_quote *= 0.65      # canned single-self-quote PR pattern

        # speaker credentials / bios
        n_cred = len(_P_CRED.findall(t))
        s_cred = _sat(n_cred, 1.5)

        # external-source attribution
        s_ext = _sat(len(_P_EXT.findall(t)), 1.5)

        # formal citations (legal cites, footnote refs)
        s_formal = _sat(len(_P_FORMAL.findall(t)), 1.5)

        # bylines / author-editor credits
        s_byline = _sat(len(_P_BYLINE.findall(t)), 1.0)

        # named contact with a reachable channel (phone/email near a name)
        contact = 0
        for m in list(_P_PHONE.finditer(t)) + list(_P_EMAIL.finditer(t)):
            if _P_NAMEISH.search(t[max(0, m.start() - 90):m.start()]):
                contact += 1
        s_contact = _sat(contact, 1.0)

        # ---- LLM thick-input boosts (optional; capped) ----
        srcs = _field(extracted, "external_sources")
        if srcs:
            n_items = len([p for p in re.split(r"[;,]| and ", srcs) if p.strip()])
            s_ext = max(s_ext, _sat(n_items, 1.5))
        cred_ans = _field(extracted, "speaker_credentials")
        if cred_ans and re.search(r"[A-Z][a-z]+", cred_ans):
            s_cred = max(s_cred, 0.6)
            s_quote = max(s_quote, 0.35)

        # ---- combine ----
        core = (_W["url"] * s_url + _W["quote"] * s_quote +
                _W["cred"] * s_cred + _W["ext"] * s_ext +
                _W["formal"] * s_formal + _W["byline"] * s_byline +
                _W["contact"] * s_contact)

        # diversity across independent evidence families: one repeated signal
        # type (e.g. many self-quotes) cannot reach the top of the scale
        fams = ((s_ext > 0.05) + (s_formal > 0.05) + (s_byline > 0.05) +
                (n_cred >= 2) + (len(full_urls) >= 3))
        core *= 0.75 + 0.125 * min(2, fams)

        # vague-sourcing penalty (multiplicative, floored)
        v = sum(tl.count(w) for w in _VAGUE)
        vague_mult = max(0.55, math.exp(-v / 4.0))

        return float(max(0.0, min(1.0, gate * core * vague_mult / _NORM)))
    except Exception:
        return 0.5

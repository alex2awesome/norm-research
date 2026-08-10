"""a64 "Explain why it matters" — hybrid channel.

Judge tiers observed in the pack:
  ~0.0  : non-releases (nav chrome / FAQ / blogs / directory pages) AND pure-informational
          releases (portfolio data, call logistics). Mission-speak inside "About X"
          boilerplate does NOT count.
  ~0.25 : genuine releases/articles whose benefit framing is weak or corporate-directed
          (clients / shareholders / brand), or only trace public-good language.
  ~0.9  : explicit public/audience benefit ("so what?" answered): benefit predicate +
          public beneficiary in body sentences, usually with stakes/specificity.
          NOTE: judge rewards this even on non-release pages (e.g. patient-assistance
          or accessibility pages), so release-likeness is a mild damper, not a gate.

Design: PRESENCE != QUALITY. A hit requires a benefit predicate AND a beneficiary in
the SAME sentence, position-weighted (boilerplate/About/Contacts discounted), with a
specificity boost (numbers/$/%) so injected bare keywords score poorly.
"""
import math
import re

LLM_FIELDS = {
    "audience_benefit": ("Quote the passage (<=20 words) that explains how the public, patients, "
                         "customers, or audience concretely benefits from this news; NONE if absent."),
    "beneficiary_type": ("Who mainly benefits from the announced news: answer 'public' (patients/"
                         "consumers/travelers/communities), 'business' (clients/investors/the company), or NONE."),
}

# --- lexicons (patterns, not datapoint memorization) -------------------------------
_PUBLIC = re.compile(
    r"\b(people|patients?|consumers?|customers?|travele?rs?|passengers?|families|"
    r"communit(?:y|ies)|the\s+public|americans?|residents?|students?|workers?|"
    r"drivers?|users?|children|kids|seniors?|veterans?|everyone|anyone|"
    r"homeowners?|buyers?|citizens?|women|patients|lives|society|"
    r"blind|visually\s+impaired|disabilit|new\s+yorkers?|"
    r"todos|personas|pacientes|usuarios)\b", re.I)

_CORP = re.compile(
    r"\b(shareholders?|stockholders?|investors?|clients?|brands?|"
    r"the\s+company|our\s+(?:business|portfolio|clients)|enterprises?)\b", re.I)

_BENEFIT = re.compile(
    r"\b(help(?:s|ed|ing)?|enabl(?:e|es|ed|ing)|empower\w*|protect\w*|"
    r"sav(?:e|es|ed|ing)|prevent\w*|improv\w*|benefit\w*|afford(?:able|ability)?|"
    r"easier|safer|cheaper|healthier|"
    r"(?:expand|increas|improv)\w*\s+access|access(?:ible)?\s+(?:to|for)\b|accessib\w+|"
    r"lower(?:s|ed|ing)?\s+(?:the\s+)?(?:costs?|prices?|premiums?|fees?)|"
    r"reduc\w*\s+(?:the\s+)?(?:costs?|prices?|risks?|robber\w*|deaths?|injur\w*|"
    r"incidents?|errors?|exposure|emissions?)|"
    r"brings?\s+more\s+(?:choices?|options?|connections?)|"
    r"opens?\s+up|make\s+life\s+(?:better|easier)|"
    r"free\s+(?:of\s+charge|use)|at\s+no\s+cost|"
    r"safet?y?\b|safeguard\w*|protections?\b|"
    r"serv(?:e|es|ing)\b|rely\s+on|"
    r"ayud\w+|facilit\w+|permite\w*|mejor\w+|beneficio\w*)\b", re.I)

# stakes / problem framing ("so what?" grounding)
_STAKES = re.compile(
    r"\b(die|dies|died|deaths?|struggl\w+|crisis|robber\w*|virus|pandemic|"
    r"uninsured|injur\w*|unfair|danger\w*|threats?|life[- ]saving|lifesaving|"
    r"epidemic|overdose|disease|unsafe|safety|risks?|"
    r"accessib\w*|accesib\w*|braille|impair\w*|disabilit\w*)\b", re.I)

# causal / purpose connectives
_CAUSAL = re.compile(
    r"\b(because|so\s+that|which\s+means|means\s+that|in\s+order\s+to|"
    r"will\s+(?:allow|bring|open|give|make|help)|allow(?:s|ing)?|"
    r"aims?\s+to|designed\s+to|with\s+the\s+goal\s+of|para\s+que)\b", re.I)

_2ND_PERSON = re.compile(r"\b(you|your|te|tu|tus)\b", re.I)
_SPECIFIC = re.compile(r"(\d|%|\$|percent|million|billion)", re.I)

# release-likeness signals
_WIRE = re.compile(
    r"(/PRNewswire/|BUSINESS\s+WIRE|GLOBE\s*NEWSWIRE|FOR\s+IMMEDIATE\s+RELEASE|"
    r"press\s+release|news\s+provided\s+by|newsroom)", re.I)
_ANNOUNCE = re.compile(r"\b(announc(?:es?|ed|ing)|today\s+announced|announced\s+today|launch(?:es|ed)?|unveil\w*)\b", re.I)
_SAID = re.compile(r"(,?[\"']\s*,?\s*said\s+[A-Z]|\bsaid\s+[A-Z][a-z]+|\bsays\s+[A-Z][a-z]+)")

# boilerplate-zone starters (contacts / about / legal live near the END)
_BOILER = re.compile(
    r"(\bAbout\s+[A-Z][\w&.\- ]{2,40}\n|\nAbout\s+|\bContacts?\s*\n|Forward[- ]Looking|"
    r"Safe\s+Harbor|Disclosures?\b|Privacy\s+(?:Policy|Statement)|Terms\s+of\s+Use|"
    r"All\s+rights\s+reserved|Media\s+Contact|Investor\s+Relations\s+Contact)")

_FAQ = re.compile(r"frequently\s+asked\s+questions|\bFAQ\b", re.I)


def _sentences(t):
    """(start_char, sentence) pairs; newline-heavy chrome splits on newlines too."""
    out = []
    for m in re.finditer(r"[^.!?\n]{15,}[.!?]?", t):
        s = m.group(0).strip()
        if len(s.split()) >= 5:
            out.append((m.start(), s))
    return out


def _chrome_frac(t):
    """Nav-menu density over the FIRST 60% of the doc (tails are chrome-y even on
    real releases; what matters is whether the BODY is a nav page)."""
    lines = [ln.strip() for ln in t[: int(0.6 * len(t))].split("\n") if ln.strip()]
    if not lines:
        return 1.0
    short = sum(1 for ln in lines if len(ln.split()) <= 3)
    return short / len(lines)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        n = len(t)
        tl = t.lower()

        # ---- boilerplate boundary: earliest boiler marker in the latter 45% ----
        boiler_at = n
        for m in _BOILER.finditer(t):
            if m.start() > 0.55 * n:
                boiler_at = m.start()
                break
        # "About X" mission-speak anywhere in the last 60% is boilerplate too
        m_about = re.search(r"\nAbout\s+[A-Z]", t)
        about_at = m_about.start() if (m_about and m_about.start() > 0.25 * n) else n

        # ---- global context signals ----
        chrome = _chrome_frac(t)
        damp = 1.0 - 0.85 * max(0.0, chrome - 0.35) / 0.65   # nav-chrome pages -> heavy damp
        release_like = min(1.0, 0.45 * bool(_WIRE.search(t)) +
                           0.30 * bool(_ANNOUNCE.search(t)) +
                           0.30 * bool(_SAID.search(t)))
        doc_stakes = bool(_STAKES.search(tl))
        faq_page = bool(_FAQ.search(t))

        # ---- sentence-level benefit predicate ----
        strong = 0.0   # public-beneficiary benefit statements
        weak = 0.0     # corporate / vague benefit statements
        for start, s in _sentences(t):
            has_b = bool(_BENEFIT.search(s))
            has_c = bool(_CAUSAL.search(s))
            if not (has_b or has_c):
                continue
            pos = start / max(1, n)
            w = 1.0 if pos < 0.55 else (0.45 if pos < 0.8 else 0.2)
            if start >= boiler_at or start >= about_at:
                w = min(w, 0.15)
            spec = 1.35 if _SPECIFIC.search(s) else 1.0
            stak = 1.25 if _STAKES.search(s) else 1.0
            pub = bool(_PUBLIC.search(s))
            corp = bool(_CORP.search(s))
            snd = bool(_2ND_PERSON.search(s))
            if has_b and pub:
                strong += w * spec * stak
            elif has_b and snd and not faq_page and (doc_stakes or release_like > 0.4):
                strong += 0.55 * w * spec * stak
            elif has_b and corp:
                weak += 0.8 * w
            elif has_b:
                weak += 0.5 * w
            elif has_c and (pub or snd):     # causal framing aimed at audience
                weak += 0.4 * w

        S_strong = 1.0 - math.exp(-0.9 * strong)
        S_weak = 1.0 - math.exp(-0.9 * weak)

        # ---- LLM-extracted thick evidence (predicate stays in code) ----
        ab = (extracted or {}).get("audience_benefit", "") or ""
        bt = ((extracted or {}).get("beneficiary_type", "") or "").strip().lower()
        if ab and ab.strip().upper() != "NONE":
            toks = [w for w in re.findall(r"[a-z]{4,}", ab.lower())]
            grounded = toks and sum(1 for w in toks if w in tl) / len(toks) >= 0.6
            if grounded and (_BENEFIT.search(ab) or _PUBLIC.search(ab) or _STAKES.search(ab)):
                bump = 0.55 + (0.15 if _SPECIFIC.search(ab) else 0.0)
                S_strong = max(S_strong, min(1.0, bump))
        if bt.startswith("business") and strong < 1.0:
            S_strong = min(S_strong, 0.45)
        elif bt.startswith("public"):
            S_strong = min(1.0, S_strong + 0.10)

        raw = 0.03 + 0.10 * release_like + 0.32 * S_weak * damp + 0.62 * S_strong * damp
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5

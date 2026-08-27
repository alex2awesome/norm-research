"""Hybrid metric channel for a198 — Magic system design: clarity, consistency, constraints.

Structure of the predicate (kept in code):
  staircase: (1) is there a speculative system at all? (2) is an explicit
  rule/cost/limit stated? (3) are its consequences actually explored?
  plus code-side densities for constraint language / quantified rules and a
  surface-craft adjustment (judge halo: lows are also mechanically sloppy).

LLM fields do THICK-INPUT grounding only (what regex can't see: whether a rule
is established and whether the story works through its implications); the
scoring logic stays here.
"""

import re
import math

LLM_FIELDS = {
    "system_rule": "State the story's most explicit rule, cost, price, or limit governing how its magic/supernatural/speculative premise works; NONE if no such rule is established.",
    "rule_consequences": "List up to three distinct consequences or applications of that rule actually shown in the story, comma-separated; NONE if none.",
}

# --- code-side lexicons -----------------------------------------------------

_CONSTRAINT_PATTERNS = [
    r"\bfor every\b", r"\b(each|every) time\b", r"\bin exchange\b",
    r"\bin return for\b", r"\bat the cost of\b", r"\bcannot\b",
    r"\bcan never\b", r"\bwill never\b", r"\bcan'?t be\b", r"\bcouldn'?t be\b",
    r"\bno matter (how|what|where|who)\b", r"\bunless\b", r"\bas long as\b",
    r"\bthe only way\b", r"\bonly (way|one|thing|those)\b",
    r"\bone in (a|ten|hundred|thousand|million)\b",
    r"\b(un|not )able to\b", r"\bwon'?t be able\b", r"\bwill not be able\b",
    r"\brequires?\b", r"\bforbidden\b", r"\bmust (never|not)\b",
    r"\bnever (able|allowed)\b", r"\bwithout permission\b",
    r"\bthe price\b", r"\bthe cost\b", r"\bequivalent\b", r"\bthe terms\b",
    r"\bthe rules?\b", r"\bthe deal\b", r"\bthe bargain\b",
    r"\bthe contract\b", r"\bthe curse\b", r"\bthe catch\b",
    r"\bin return\b", r"\bbound (by|to)\b", r"\bsworn\b",
]

_SYSTEM_TERMS = [
    "spell", "spells", "magic", "magical", "sorcery", "incantation", "ritual",
    "rune", "runes", "sigil", "sigils", "enchant", "enchanted", "potion",
    "summon", "summoned", "summoning", "curse", "cursed", "immortal",
    "immortality", "fae", "wizard", "witch", "demon", "demons", "angel",
    "angels", "reaper", "soul", "souls", "supernatural", "telekinetic",
    "telekinesis", "superpower", "superpowers", "prophecy", "grimoire",
    "pentagram", "occult", "wish", "wishes", "realm", "gods",
]

_NUMBER_WORDS = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|dozen|"
    r"hundred|thousand|million|billion|half|twice|single)"
)

_PROFANITY = [
    r"\bfuck\w*", r"\bshit\w*", r"\bbitch\w*", r"\bwanker\w*",
    r"\bmasturbat\w*", r"\bphallus\b", r"\bcock\b",
]

_NONE_RX = re.compile(r"^\s*(none|n/?a|no( such)? rule|no|nothing|-)\s*\.?\s*$", re.I)

# words in the extracted rule that indicate a real mechanism, not a vibe
_MECHANISM_RX = re.compile(
    r"(\d|\bevery\b|\beach\b|\bonly\b|\bcannot\b|\bcan'?t\b|\bmust\b|"
    r"\bnever\b|\bunless\b|\bexchange\b|\bcost\w*\b|\bprice\b|\brequir\w+\b|"
    r"\bper\b|\blimit\w*\b|\bforbidden\b|\bwithout\b|\bunable\b|\bloses?\b|"
    r"\bages?\b|\byear\b|\bpermission\b|\bwilling\w*\b|\breturn\w*\b|"
    r"\bpersist\w*\b|\balways\b|\bif\b|\bwhen\b)", re.I)


def _sat(x, half):
    """Saturating map to [0,1): x=half -> 0.5."""
    if x <= 0:
        return 0.0
    return x / (x + half)


def _clean_field(val):
    if not isinstance(val, str):
        return ""
    v = val.strip()
    if not v or _NONE_RX.match(v):
        return ""
    return v


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        tl = t.lower()
        n_words = max(1, len(re.findall(r"[a-zA-Z']+", tl)))
        per_kw = n_words / 1000.0  # densities per 1000 words

        # ---- code feature K: constraint / rule language density ----
        k_hits = 0
        for pat in _CONSTRAINT_PATTERNS:
            k_hits += len(re.findall(pat, tl))
        K = _sat(k_hits / max(per_kw, 0.12), 8.0)  # ~8 hits/1k words -> 0.5

        # ---- code feature S: speculative-system term presence ----
        s_hits = 0
        for term in _SYSTEM_TERMS:
            s_hits += len(re.findall(r"\b" + re.escape(term) + r"\b", tl))
        S = _sat(s_hits / max(per_kw, 0.12), 5.0)

        # ---- code feature Q: quantified rules (numbers in system/rule sentences) ----
        q_hits = 0
        sents = re.split(r"(?<=[.!?])\s+|\n+", tl)
        num_rx = re.compile(r"\b" + _NUMBER_WORDS + r"\b")
        sys_rx = re.compile(r"\b(" + "|".join(_SYSTEM_TERMS) +
                            r"|rule|rules|score|scored|stat|stats|mark|marks|"
                            r"page|deal|trade|luck|power|powers|gift)\b")
        for s in sents:
            if num_rx.search(s) and sys_rx.search(s):
                q_hits += 1
        Q = _sat(q_hits / max(per_kw, 0.12), 5.0)

        # ---- code feature W: surface craft (halo; lows are sloppy) ----
        pen = 0.0
        # lowercase standalone "i" pronoun (typo-level sloppiness)
        lc_i = len(re.findall(r"(?<![\w'])i(?![\w'])", t))
        pen += min(0.5, 0.10 * lc_i)
        # shouted all-caps words (allow a few stylistic ones)
        caps = len(re.findall(r"\b[A-Z]{4,}\b", t))
        pen += min(0.35, 0.07 * max(0, caps - 3))
        # profanity density (mild weight; some mid stories swear)
        prof = 0
        for pat in _PROFANITY:
            prof += len(re.findall(pat, tl))
        pen += min(0.30, 0.08 * max(0, prof - 1))
        # exclamation pileups !!! ?!?
        bangs = len(re.findall(r"[!?]{2,}", t))
        pen += min(0.2, 0.10 * bangs)
        W = max(0.0, 1.0 - pen)

        # ---- code feature L: room to develop (log-length prior, small) ----
        L = max(0.0, min(1.0, (math.log(n_words + 1) - math.log(120)) /
                         (math.log(1600) - math.log(120))))

        # ---- LLM feature R: explicit rule established + mechanism-specific ----
        rule = _clean_field((extracted or {}).get("system_rule", ""))
        conseq = _clean_field((extracted or {}).get("rule_consequences", ""))
        if rule:
            R = 0.55
            # full mechanism credit only for a substantive rule, not a slogan
            if _MECHANISM_RX.search(rule) and len(rule.split()) >= 5:
                R = 1.0
        else:
            R = 0.0

        # ---- LLM feature C: consequences actually explored (count in code) ----
        if conseq:
            items = [p for p in re.split(r"[,;]| and ", conseq) if p.strip()]
            C = min(3, len(items)) / 3.0
        else:
            C = 0.0

        llm_seen = bool(rule) or bool(conseq)
        if llm_seen or (extracted and any(k in extracted for k in LLM_FIELDS)):
            # full hybrid blend
            s_val = (0.22 * R + 0.24 * C + 0.16 * K + 0.08 * S +
                     0.10 * Q + 0.14 * W + 0.06 * L)
            # a story with no established rule per the extractor is capped:
            # the criterion cannot score high without an intelligible rule.
            if not rule:
                s_val = min(s_val, 0.38)
        else:
            # code-only fallback (extractor unavailable)
            s_val = (0.30 * K + 0.16 * S + 0.18 * Q + 0.22 * W + 0.14 * L)

        return float(max(0.0, min(1.0, s_val)))
    except Exception:
        return 0.5

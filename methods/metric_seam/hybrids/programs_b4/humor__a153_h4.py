"""a153: Cross-cultural translation and translatability (hybrid v0).

Design hypothesis (from the pack's own training examples): documents that score
LOW on this criterion are almost always jokes whose entire punchline depends on
an English-only phonetic pun / homophone (e.g. "nickeless" cage, "mine, mine,
mine", "whale cum" -> welcome, "Israel bad" -> "is really bad", "lyre" -> liar)
that cannot survive translation, or that carry a slur/hateful-stereotype
punchline (with or without a self-aware disclaimer) that risks catastrophic
misreading across audiences. Documents that score HIGH are narrative /
situational / structural jokes (irony, escalation, callback, list-format) whose
humor is conceptual rather than phonetic, so the "meaning" travels even if a
literal translation is imperfect. Topic words and profanity are explicitly
flagged in the pack as weak proxies, so this module avoids topic keyword
matching and instead targets the STRUCTURE of the punchline (repetition,
phonetic collision) plus an LLM-grounded judgment of pun-dependence and risky
reference, per the pack's "keep the predicate in code" instruction.

b4 extension (blind, construct-driven -- no eval signal used to pick these):
the criterion text names TWO things — "language/culture-specific references
AND puns" — but h0's LLM fields only ground the pun half (pun_reliance) and
an offense/misreading-risk half (risky_reference). A joke can be entirely
free of wordplay and free of slurs/stereotypes yet still fail to travel
because it hinges on a real person, place, product, or historical/political
event that a foreign or unfamiliar audience simply won't recognize (e.g. a
joke that only lands if you know a specific ad campaign, a head of state, or
a WWII-era phrase). Two new fields target that gap directly:
  - culture_specific_reference: names any such real-world reference the
    punchline depends on, independent of wordplay.
  - self_contained_context: when such a reference exists, whether the text
    supplies enough in-line context for an unfamiliar reader to still
    follow — a reference that explains itself travels much better than one
    that assumes shared background knowledge, so this tempers the penalty.
"""
import re
from collections import Counter

LLM_FIELDS = {
    "pun_reliance": "In <=6 words, name the English-only pun/homophone/wordplay the punchline depends on; write NONE if the humor is conceptual, situational, or narrative rather than wordplay.",
    "risky_reference": "In <=6 words, name any slur, hateful stereotype, or self-aware disclaimer defending offensive content; write NONE if absent.",
    "culture_specific_reference": "In <=6 words, name any real person, place, product, or historical/political event (not wordplay) required to understand the punchline; write NONE if none.",
    "self_contained_context": "If a specific person/place/event is needed to get the joke, does the text itself explain enough for an unfamiliar reader to follow it? Answer YES, NO, or NA if no such reference.",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "been", "it", "its", "he", "she",
    "they", "his", "her", "their", "this", "that", "these", "those", "i",
    "you", "we", "my", "your", "not", "no", "so", "as", "if", "then", "than",
    "just", "said", "says", "say", "did", "do", "does", "had", "has", "have",
    "will", "would", "could", "should", "im", "dont", "cant", "didnt",
    "wasnt", "werent", "ive", "youre", "theyre", "thats", "what", "when",
    "who", "which", "there", "here", "from", "into", "were", "been",
}

_DISCLAIMER_PATTERNS = [
    r"\bnot\s+a\s+racist\b",
    r"\bno\s+offense\b",
    r"\bnot\s+being\s+racist\b",
    r"\bplease\s+don'?t\s+(kill|cancel|hurt)\s+me\b",
    r"\bjust\s+a\s+joke\b",
    r"\bnot\s+trying\s+to\s+be\s+offensive\b",
    r"\bnot\s+offensive\b",
    r"\bdon'?t\s+cancel\s+me\b",
    r"\bi'?m\s+not\s+racist\b",
]

_SLUR_EUPHEMISMS = [
    r"\bn[\-\s]?word\b",
    r"\bracial\s+slur\b",
    r"\bethnic\s+slur\b",
    r"\boffensive\s+term\b",
]


def _soundex(word):
    word = re.sub(r"[^A-Za-z]", "", word or "").upper()
    if not word:
        return ""
    codes = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first = word[0]
    tail_codes = [codes.get(c, "") for c in word[1:]]
    collapsed = []
    prev = codes.get(first, "")
    for c in tail_codes:
        if c != prev and c != "":
            collapsed.append(c)
        prev = c
    code = (first + "".join(collapsed)).ljust(4, "0")[:4]
    return code


def _content_words(words):
    return [w for w in words if len(w) >= 4 and w not in _STOPWORDS]


def _detect_homophone_pair(words):
    """Weak, precision-biased fallback: two distinct content words that share a
    soundex code but differ in their opening letters and are not simple
    substrings of one another (to avoid plural/suffix false positives)."""
    cw = _content_words(words)
    seen = {}
    for w in cw:
        code = _soundex(w)
        if not code:
            continue
        prior = seen.get(code)
        if prior and prior != w and prior[:2] != w[:2] and prior not in w and w not in prior:
            return True
        seen.setdefault(code, w)
    return False


def _detect_repeated_punchline(sentences):
    """Punchlines that are dominated by one repeated short word (e.g.
    "Mine, mine, mine!") are a classic marker of a homophone/idiom pun."""
    if not sentences:
        return False
    last = sentences[-1]
    toks = re.findall(r"[a-z']+", last.lower())
    toks = [t for t in toks if len(t) >= 3]
    if len(toks) < 2:
        return False
    top_word, top_n = Counter(toks).most_common(1)[0]
    return top_n >= 3 and (top_n / len(toks)) >= 0.35


def _regex_any(patterns, text_lower):
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


def _is_none_answer(v):
    v = (v or "").strip().strip(".").upper()
    return v in ("", "NONE", "NO", "N/A", "NA")


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        if not t.strip():
            return 0.5

        tl = t.lower()
        words = re.findall(r"[A-Za-z']+", tl)
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", t) if s.strip()]

        extracted = extracted or {}

        # --- LLM evidence (thick-input grounding for pun-dependence / risky refs) ---
        pun_key_present = "pun_reliance" in extracted
        risk_key_present = "risky_reference" in extracted

        llm_pun = 0.0
        if pun_key_present:
            llm_pun = 0.0 if _is_none_answer(extracted.get("pun_reliance")) else 1.0

        llm_risk = 0.0
        if risk_key_present:
            llm_risk = 0.0 if _is_none_answer(extracted.get("risky_reference")) else 1.0

        # --- code-only structural fallback / corroboration ---
        code_pun = 0.0
        if _detect_repeated_punchline(sentences):
            code_pun += 0.5
        if _detect_homophone_pair(words):
            code_pun += 0.5
        code_pun = min(1.0, code_pun)

        code_risk = 0.0
        if _regex_any(_DISCLAIMER_PATTERNS, tl):
            code_risk += 0.6
        if _regex_any(_SLUR_EUPHEMISMS, tl):
            code_risk += 0.4
        code_risk = min(1.0, code_risk)

        # LLM evidence dominates when available (keeps predicate deterministic
        # in code); code-only heuristics are the sole signal when the field is
        # absent, and a minor corroborator otherwise.
        pun_conf = (0.75 * llm_pun + 0.25 * code_pun) if pun_key_present else code_pun
        risk_conf = (0.75 * llm_risk + 0.25 * code_risk) if risk_key_present else code_risk

        # --- b4: culture-specific reference dependence, independent of
        # wordplay. Discounted (not zeroed) when the text supplies enough
        # in-line context for the reference to be legible without outside
        # knowledge -- a self-explained reference travels better than an
        # assumed one. Contributes nothing when the field is absent, so the
        # score degrades exactly to the h0 formula.
        ref_conf = 0.0
        if "culture_specific_reference" in extracted:
            ref_present = not _is_none_answer(extracted.get("culture_specific_reference"))
            if ref_present:
                sc = (extracted.get("self_contained_context") or "").strip().upper()
                if sc.startswith("YES"):
                    ref_conf = 0.35   # explained in-line: partial portability
                elif sc.startswith("NO"):
                    ref_conf = 1.0    # opaque outside-knowledge dependency
                else:
                    ref_conf = 0.7    # unclear / inconsistent: moderate default

        base = 0.92
        val = base - 0.70 * pun_conf - 0.85 * risk_conf - 0.30 * ref_conf
        val = max(0.03, min(0.97, val))
        return val
    except Exception:
        return 0.5

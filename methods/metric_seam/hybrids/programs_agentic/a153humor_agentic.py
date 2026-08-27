"""a153: Cross-cultural translation and translatability (agentic hybrid, round 1).

LLM_FIELDS are byte-identical to a153_h0.py (same 2 field names + instructions).

Diagnosis from TRAIN data (full pun_reliance / risky_reference vs judge dump,
not just the worst-residual list) that motivates this rewrite:

1. h0 treats ANY non-NONE `pun_reliance` answer as maximal untranslatable-pun
   evidence (llm_pun = 1.0 flat). Sorting all train items by judge score against
   the raw pun_reliance text shows this is too coarse. There are two very
   different kinds of answer the extractor gives:
     (a) an explicit two-sided pairing, "X / Y" or "X vs Y" (e.g. "Thyme / time",
         "Lyre / liar", "Boeing / Be going", "Is real / Israel", "Steaks /
         stakes") -- this is the extractor naming a genuine cross-word phonetic
         collision. In the training data 22/25 such two-sided answers land at
         judge <= 0.60 (most <= 0.30): a near-clean LOW marker.
     (b) a single word/phrase with no pairing (e.g. "Poisonous", "Just a second",
         "Rotated", "deep shit", "Tips", "Monkey", "Bucks") -- often just the
         extractor naming a salient topic word or a loose idiom/double-entendre
         that is NOT phonetically English-only. These single-answer items are
         spread almost uniformly across the whole judge range (many are >= 0.80,
         e.g. d01357 "Poisonous"->1.00, d01762 "Just a second"->1.00, d02656
         "Rotated"->1.00, d02664 "deep shit"->1.00, d03903 "change (diaper vs.
         identity)"->1.00) -- treating them as maximal evidence is a major
         source of h0's error.
   Fix: gate pun severity on whether the answer text itself contains a
   two-sided "A / B" / "A vs B" / "A versus B" pairing (STRONG), vs a bare
   phrase (WEAK). A bare-phrase answer is promoted back to STRONG only if the
   document's own code-level structure corroborates load-bearing wordplay
   (the repeated-punchline or homophone-pair detectors below both fire) --
   this recovers cases like "Mine (possessive vs. occupation)" (the "mine,
   mine, mine!" joke, judge=0.10) where the LLM's parenthetical hides the
   pairing but the repeated punchline is the whole joke.

2. h0 treats ANY non-NONE `risky_reference` answer as near-maximal harm
   evidence (llm_risk dominated at 0.75 weight). The same full sort shows this
   is backwards for MOST risky_reference hits: naming a generic demographic/
   stereotype category (Hill-Billy 0.90, whores 0.95, Eskimos 0.95, "hateful
   stereotype about absent fathers" 0.90, "hateful stereotype (intellectual
   disability)" 1.00, dowry 0.80) does NOT predict low translatability --
   these are portable narrative/situational jokes where the group name is
   incidental set-dressing, not the untranslatable mechanism. Only a narrow
   SEVERE band reliably predicts low: an explicit slur token, an explicit
   "race/racist/racial/ethnic slur" mention in the field text, or a
   self-aware defensive disclaimer detected in the document itself (h0's own
   disclaimer-pattern list; "Not a racist... please don't kill me... didn't
   use the N word" -> judge=0.05). Fix: three-tier severity (SEVERE / WEAK
   default / no signal) instead of one flat constant.

No new LLM fields. Keyword regexes here only GATE severity tiers / corroborate
structural detectors; they do not directly sum into the score.
"""
import re
from collections import Counter

LLM_FIELDS = {
    "pun_reliance": "In <=6 words, name the English-only pun/homophone/wordplay the punchline depends on; write NONE if the humor is conceptual, situational, or narrative rather than wordplay.",
    "risky_reference": "In <=6 words, name any slur, hateful stereotype, or self-aware disclaimer defending offensive content; write NONE if absent.",
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

# Explicit slur / race-explicit markers -- the narrow band that reliably
# predicts LOW translatability among risky_reference hits (see module doc).
_SLUR_WORDS = [
    "nigger", "nigga", "faggot", "fag", "tranny", "kike", "spic", "chink",
    "gook", "coon", "retard", "retarded",
]
_RACE_EXPLICIT_MARKERS = [
    r"\brace\b", r"\bracist\b", r"\bracial\b", r"\bethnic\s+slur\b",
    r"\bn[\-\s]?word\b", r"\bracial\s+slur\b", r"\bhate\s+speech\b",
]

_PUN_SPLIT_RE = re.compile(r"\s*/\s*|\bvs\.?\b|\bversus\b", re.IGNORECASE)


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


def _pun_is_paired(answer):
    """True iff the extractor's short pun_reliance answer itself names a
    two-sided X/Y (or X vs Y) pairing -- the structural marker for a genuine
    cross-word phonetic collision rather than a single topic word or idiom."""
    a = re.sub(r"\([^)]*\)", "", answer or "").strip()
    parts = [p.strip() for p in _PUN_SPLIT_RE.split(a) if p.strip()]
    return len(parts) >= 2


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

        pun_key_present = "pun_reliance" in extracted
        risk_key_present = "risky_reference" in extracted

        pun_answer = (extracted.get("pun_reliance") or "").strip()
        risk_answer = (extracted.get("risky_reference") or "").strip()

        # NOTE (round 2 fix): _detect_homophone_pair fires on 77/150 (51%) of
        # ALL train documents -- a near-coin-flip base rate on ordinary short
        # jokes with several 4+ letter content words, so it carries almost no
        # information and must NOT be used to promote a bare-phrase pun answer
        # to strong. _detect_repeated_punchline fires on only 2/150: a precise,
        # rare structural marker (the whole punchline IS one repeated word,
        # e.g. "Mine, mine, mine!") -- that is the only corroborator trusted
        # here to promote a bare phrase back to strong.
        code_structural_pun = _detect_repeated_punchline(sentences)

        # --- pun channel: tiered severity, not a flat constant ---
        word_count = len(words)
        llm_pun = 0.0
        if pun_key_present and not _is_none_answer(pun_answer):
            if _pun_is_paired(pun_answer):
                # round 5: within the "genuine X/Y pair" tier itself, word
                # count of the DOCUMENT (not the answer) is a real, general
                # discriminator -- checked across all 25 paired-pun train
                # items, Spearman(word_count, judge) = 0.50 within this
                # subgroup. A short document (<=100 words) is usually a single
                # compact wordplay beat where the pun IS the whole joke
                # (nickel-less cage, thyme/time, lyre/liar: all <=100 words,
                # judge <= 0.35). A long document carries more independent
                # narrative structure around the pun, so the pun is more often
                # one clever aside rather than the load-bearing mechanism
                # (Deshawn "Son/Sun" at 122 words -> 0.90, "Sink in" riddle at
                # 152 words -> 0.90, uncle-Bob "Moral/morale" at 188 words ->
                # 0.80). Linearly discount the paired-tier peak as length grows
                # past 100 words, floored so a long doc with a genuine pun
                # still carries some penalty.
                llm_pun = 0.95
                if word_count > 100:
                    llm_pun = max(0.55, 0.95 - 0.40 * min(1.0, (word_count - 100) / 120.0))
            elif code_structural_pun:
                llm_pun = 0.85        # bare phrase, but code corroborates a
                                       # load-bearing repeated-punchline pattern
            else:
                llm_pun = 0.10        # bare phrase / topic word, weak evidence

        code_pun = 0.6 if _detect_repeated_punchline(sentences) else 0.0
        # _detect_homophone_pair is kept only as a very small residual
        # corroborator (its noisy 51% base rate means it must stay near-zero
        # weight; see note above) rather than a full 0.5 contributor.
        if _detect_homophone_pair(words):
            code_pun += 0.1
        code_pun = min(1.0, code_pun)

        pun_conf = (0.85 * llm_pun + 0.15 * code_pun) if pun_key_present else code_pun

        # --- risk channel: severity tiers on the free-text answer ---
        disclaimer_hit = _regex_any(_DISCLAIMER_PATTERNS, tl)

        llm_risk = 0.0
        if risk_key_present and not _is_none_answer(risk_answer):
            risk_l = risk_answer.lower()
            severe = (
                any(re.search(r"\b" + re.escape(w) + r"\b", risk_l) for w in _SLUR_WORDS)
                or _regex_any(_RACE_EXPLICIT_MARKERS, risk_l)
                or disclaimer_hit
            )
            llm_risk = 0.90 if severe else 0.05

        code_risk = 0.0
        if disclaimer_hit:
            code_risk += 0.6
        if any(re.search(r"\b" + re.escape(w) + r"\b", tl) for w in _SLUR_WORDS):
            code_risk += 0.4
        code_risk = min(1.0, code_risk)

        risk_conf = max(llm_risk, code_risk) if risk_key_present else code_risk

        # Combine the two channels with max, not sum: translatability fails if
        # EITHER an untranslatable pun OR a genuinely severe risky reference is
        # present -- an item with both should not be penalized twice as hard as
        # an item with just one (mirrors the fleet convention in the a351
        # reference module: max lets either channel alone be sufficient).
        penalty = max(pun_conf, risk_conf)
        val = 1.0 - penalty
        val = max(0.03, min(0.97, val))
        return val
    except Exception:
        return 0.5

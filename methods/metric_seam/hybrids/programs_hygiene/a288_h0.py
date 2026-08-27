"""a288: Rule of Three / three-beat pattern (hybrid).

Criterion: establish, confirm, then subvert on the THIRD beat. Requires
recognizability of the repeated pattern, proportional escalation across
beats, and a clean, surprising break on the final beat.

Corpus notes handled: short reddit-style jokes (median ~500 chars), possible
mojibake/quote artifacts, topic words (blonde/bar/lawyer) and profanity are
weak proxies -- the real signal is narrative structure (how many parallel
beats are set up, and whether the last one breaks the pattern).

Design:
  - Code detects surface structural cues that are cheap and reliable:
    explicit ordinal/numeric triads in order ("first"..."second"..."third"),
    and repeated reporting-verb clauses (dialogue/action "turns") with a
    length-based check for a shorter, punchier final beat (economy of the
    break). This distinguishes true 3-beat builds from plain 2-beat
    setup/punchline jokes, which are NOT rule-of-three even if the baseline
    keyword-matcher scores them highly.
  - The LLM fields carry the THICK part of the construct that regex cannot
    reliably reach: whether a pattern is actually established twice and
    subverted a final time (independent of surface phrasing), and whether
    that subversion reads as clean/proportional vs. forced/messy.
  - ops.retrieve_similar / ops.extract_dates are not used: retrieve_similar
    returns bare (similarity, id) pairs with no criterion labels attached, so
    it cannot inform a structural judgment at inference time; extract_dates
    is irrelevant to joke structure. ops.sent_stats is used only as a light,
    format-defensive sanity discount (very few sentences -> less likely a
    full 3-beat arc got room to unfold).
"""
import re

LLM_FIELDS = {
    "pattern_break": (
        "Does the text establish a pattern twice then clearly break/subvert "
        "it on a final beat? Answer YES, PARTIAL, or NO."
    ),
    "twist_quality": (
        "If a break/subversion happens, is it clean and proportional, or "
        "forced/messy? Answer CLEAN, MESSY, or NONE."
    ),
}

_ORDINAL_SEQ = ["first", "second", "third"]
_NUMBER_SEQ = ["one", "two", "three"]

_REPORT_VERBS = re.compile(
    r"\b(says?|said|asks?|asked|repl(?:y|ies|ied)|shouts?|shouted|yells?|"
    r"yelled|responds?|responded|whispers?|whispered|exclaims?|exclaimed)\b",
    re.IGNORECASE,
)


def _split_clauses(t: str):
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _in_order_score(positions):
    if any(p < 0 for p in positions):
        return 0.0
    if positions[0] < positions[1] < positions[2]:
        return 1.0
    return 0.3


# HYGIENE PATCH: `.find(w)` on the whole text is a bare substring search, so
# "one" matched inside "money"/"honey"/"someone's"/"mentioned"/etc -- unrelated
# to the number-word sequence this is meant to detect. Word-bounded lookup
# instead (first/second/third and two/three are far less collision-prone but
# are switched too since they share this helper).
def _find_word(t_low: str, w: str) -> int:
    m = re.search(r"\b" + re.escape(w) + r"\b", t_low)
    return m.start() if m else -1


def _ordinal_signal(t_low: str) -> float:
    pos_ord = [_find_word(t_low, w) for w in _ORDINAL_SEQ]
    pos_num = [_find_word(t_low, w) for w in _NUMBER_SEQ]
    ord_score = _in_order_score(pos_ord)
    num_score = _in_order_score(pos_num)
    return max(ord_score, num_score)


def _turn_repetition_signal(clauses) -> float:
    tagged = [c for c in clauses if _REPORT_VERBS.search(c)]
    n = len(tagged)
    if n == 0:
        return 0.0
    if n == 2:
        # Classic 2-beat setup/punchline -- explicitly NOT rule-of-three.
        return 0.25
    if n >= 3:
        lens = [len(c.split()) for c in tagged[:4]]
        if len(lens) >= 2:
            build = lens[:-1]
            avg_build = sum(build) / max(1, len(build))
            last = lens[-1]
            if avg_build > 0 and last < avg_build:
                # Shorter final beat: consistent with a punchy subversion.
                return 1.0
            return 0.65
        return 0.55
    return 0.0


def _map_pattern_break(val: str):
    v = (val or "").strip().lower()
    if v.startswith("yes"):
        return 0.8
    if v.startswith("partial"):
        return 0.45
    if v.startswith("no"):
        return 0.05
    return None


def _map_twist_quality(val: str) -> float:
    v = (val or "").strip().lower()
    if v.startswith("clean"):
        return 0.2
    if v.startswith("messy"):
        return -0.15
    return 0.0


def _sentence_count(t: str, ops) -> int:
    try:
        ss = ops.sent_stats(t)
    except Exception:
        return 0
    try:
        if isinstance(ss, dict):
            return int(ss.get("n_sent") or ss.get("num_sentences") or 0)
        if isinstance(ss, (list, tuple)) and len(ss) > 0:
            return int(ss[0])
    except Exception:
        return 0
    return 0


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if text else ""
        if not t.strip():
            return 0.5
        t_low = t.lower()

        clauses = _split_clauses(t)
        code_signal = 0.5 * _ordinal_signal(t_low) + 0.5 * _turn_repetition_signal(clauses)
        code_signal = max(0.0, min(1.0, code_signal))

        ex = extracted or {}
        pb = _map_pattern_break(ex.get("pattern_break", ""))
        tq_adj = _map_twist_quality(ex.get("twist_quality", ""))

        if pb is None:
            # No usable LLM signal (missing/NONE): fall back to the code-only
            # structural heuristic alone, kept conservative.
            base = 0.15 + 0.7 * code_signal
        else:
            # LLM judgment is primary (it sees the thick, non-lexical
            # structure); code signal is a secondary sanity check/tie-break.
            base = 0.75 * pb + 0.25 * code_signal + tq_adj

        n_words = len(t_low.split())
        strong_yes = pb is not None and pb >= 0.8
        if n_words < 12 and not strong_yes:
            base *= 0.6

        n_sent = _sentence_count(t, ops)
        if n_sent and n_sent < 3 and not strong_yes:
            base *= 0.85

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

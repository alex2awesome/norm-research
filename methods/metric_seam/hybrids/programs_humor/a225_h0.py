# Hybrid module for humor aspect a225: "Callback design and deployment"
# Criterion: place, recontextualize, and time callbacks across a piece to
# culminate strands and create cumulative payoff.
#
# Design: code detects the LEXICAL analog of a callback -- a distinctive,
# non-filler word/phrase planted in the setup (first ~40% of the text) that
# recurs in the payoff (last ~25%), weighted by rarity/specificity, plus a
# small bonus for triadic/sequential structure ("first...second...third") and
# for a punchy short final sentence (timing). The LLM fields supply the
# semantic layer code cannot reach: callbacks that are paraphrased rather than
# verbatim (e.g. a planted contact name paying off via "isolation", or a
# wish/rule reused ironically) and an explicit strand count. Code owns the
# scoring predicate; the LLM only supplies short grounding text that code
# parses defensively.

import re
import math
from collections import Counter

LLM_FIELDS = {
    "setup_echo": "Name in <=6 words the earlier setup word/image/detail (if any) that the ending deliberately reuses or recontextualizes for payoff; answer NONE if there is none.",
    "strand_count": "How many distinct earlier setup elements does the ending meaningfully recontextualize for payoff: answer 0, 1, or 2+.",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in",
    "on", "at", "for", "with", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "it's", "this", "that", "these", "those", "he",
    "she", "they", "them", "his", "her", "their", "him", "i", "you", "your",
    "we", "our", "us", "as", "by", "from", "not", "no", "do", "does", "did",
    "has", "have", "had", "will", "would", "can", "could", "should", "just",
    "up", "out", "about", "into", "over", "after", "before", "than", "too",
    "very", "what", "who", "whom", "which", "when", "where", "why", "how",
    "all", "any", "some", "such", "one", "two", "three", "said", "says",
    "say", "asked", "ask", "replied", "reply", "tells", "told", "man", "guy",
    "woman", "girl", "boy", "went", "get", "got", "go", "goes", "going",
    "come", "came", "look", "looked", "know", "knew", "like", "back", "there",
    "here", "because", "while", "again", "guys", "think", "thought", "see",
    "saw", "really", "also", "even", "still", "much", "more", "other", "day",
    "time", "people", "thing", "things", "way", "around", "off", "down",
    "first", "second", "third", "last", "own", "my", "me", "mine", "don't",
    "didn't", "doesn't", "wasn't", "isn't", "into", "onto",
}

_ORDINAL_MARKERS = ("first", "second", "third", "finally", "again", "next", "another", "once")

_NONE_LIKE = {"none", "n/a", "na", "no", "nothing", "nil", "-"}


def _safe_words(s):
    return re.findall(r"[a-zA-Z']+", (s or "").lower())


def _unpack_sent_stats(stats):
    """Contract says ops.sent_stats -> (n_sent, mean_words_per_sent, frac_long_words),
    but defend against a dict-shaped implementation too."""
    if isinstance(stats, dict):
        return (
            stats.get("n_sent", stats.get("num_sentences", 0)) or 0,
            stats.get("mean_words_per_sent", stats.get("mean_wps", 0.0)) or 0.0,
            stats.get("frac_long_words", 0.0) or 0.0,
        )
    try:
        n_sent, mean_wps, frac_long = stats
        return n_sent or 0, mean_wps or 0.0, frac_long or 0.0
    except Exception:
        return 0, 0.0, 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        extracted = extracted if isinstance(extracted, dict) else {}
        raw = text or ""

        try:
            norm = ops.normalize(raw)
        except Exception:
            norm = raw

        if not norm.strip():
            return 0.5

        words = _safe_words(norm)
        n = len(words)

        code_component = 0.0
        if n >= 8:
            setup_end = max(1, int(n * 0.4))
            payoff_start = min(n - 1, int(n * 0.75))
            setup_words = words[:setup_end]
            payoff_words = words[payoff_start:]

            freq = Counter(words)
            setup_content = {w for w in setup_words if w not in _STOPWORDS and len(w) > 3}
            payoff_content = {w for w in payoff_words if w not in _STOPWORDS and len(w) > 3}
            candidates = setup_content & payoff_content

            strand_weight = 0.0
            for w in candidates:
                total = freq[w]
                if total < 2:
                    continue
                specificity = 1.0 / math.sqrt(total)  # rarer repeats = more distinctive
                length_bonus = min(1.0, len(w) / 8.0)  # longer words = more specific
                strand_weight += 0.5 * specificity + 0.5 * length_bonus

            code_component = 1.0 - math.exp(-0.8 * strand_weight)

            # weak structural bonus: triadic / sequential scaffolding often
            # carries multiple callback strands to a culminating beat
            marker_hits = sum(1 for m in _ORDINAL_MARKERS if m in words)
            if marker_hits >= 2:
                code_component = min(1.0, code_component + 0.08)

            # timing bonus: a short, punchy final sentence relative to the
            # piece suggests a deliberately placed payoff beat
            try:
                stats = ops.sent_stats(norm)
                n_sent, mean_wps, _frac_long = _unpack_sent_stats(stats)
                if n_sent and n_sent >= 3 and mean_wps:
                    sentences = re.split(r"(?<=[.!?])\s+", norm.strip())
                    last_len = len(_safe_words(sentences[-1])) if sentences else 0
                    if 0 < last_len < 0.6 * mean_wps:
                        code_component = min(1.0, code_component + 0.05)
            except Exception:
                pass

        # ---- LLM semantic layer (catches paraphrased / conceptual callbacks) ----
        echo_raw = str(extracted.get("setup_echo") or "").strip().lower()
        echo_present = bool(echo_raw) and echo_raw not in _NONE_LIKE and not echo_raw.startswith("none")

        strand_raw = str(extracted.get("strand_count") or "").strip().lower()
        llm_strands = 0
        if re.search(r"\b(2|two)\b|2\+|multiple|several", strand_raw):
            llm_strands = 2
        elif re.search(r"\b(1|one)\b", strand_raw):
            llm_strands = 1
        elif re.search(r"\b(0|zero|none|no)\b", strand_raw):
            llm_strands = 0
        elif echo_present:
            llm_strands = 1

        llm_component = 1.0 - math.exp(-1.0 * llm_strands)
        if echo_present:
            llm_component = max(llm_component, 0.5)

        base = 0.15
        combined = base + 0.35 * code_component + 0.55 * llm_component
        return max(0.0, min(1.0, combined))
    except Exception:
        return 0.5

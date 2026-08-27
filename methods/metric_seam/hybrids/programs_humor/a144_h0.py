"""
Hybrid metric channel for aspect a144: "Verbal wit, word choice, and sound"
(rhythm, symmetry, comparison, diction, rarity, phonetics -> sharper phrasing,
elevated laughs, staying clear and precise).

Design:
- The frozen code baseline (v1_structure) literally string-matches words like
  "pun"/"rhyme"/"wordplay" appearing IN the text -- that fires almost never on
  real jokes (train_rho=0.062) and would never generalize; we do not repeat
  that pattern.
- True pun/homophone/double-meaning detection ("lyre"/"liar", "celebrate"/
  "celibate", "toonie"->"two f-ing bucks", "Curiosity killed the cat") is a
  THICK construct: it requires semantic/phonetic knowledge regex cannot reach
  even with a phonetic-code table, because the payoff word and its "sounds
  like" target rarely share spelling. That is delegated to an LLM field
  (`pun_pivot`), with the PREDICATE (how much it counts, how it blends with
  everything else) staying in code.
- A second LLM field (`phrasing_sharpness`) grounds the "clear and precise"
  half of the criterion (rambling vs. tight delivery), which is also hard to
  read off surface features alone.
- Code handles what code CAN reach: alliteration (adjacent-initial-letter
  repetition), crude end-of-clause rhyme (suffix match across clause
  boundaries), parallel/refrain structure (repeated bigrams), punchline
  economy (final clause shorter than the setup) and clarity/conciseness
  (via ops.sent_stats), plus a TF-IDF "rarity of phrasing" evidence signal
  (via ops.retrieve_similar) matching the criterion's own word "rarity".
- No topic-word or profanity keying anywhere (both are flagged in the pack
  as weak/misleading proxies for judged craft).
"""

import re
from collections import Counter

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "he", "she",
    "it", "they", "i", "you", "we", "that", "this", "as", "his", "her", "its",
    "their", "my", "your", "them", "him", "so", "if", "then", "than", "not",
    "no", "do", "did", "does", "have", "has", "had", "will", "would", "can",
    "could", "just", "up", "out", "into", "over", "about", "there", "one",
})

POS_STYLE_WORDS = frozenset({
    "sharp", "precise", "tight", "crisp", "clever", "witty", "economical",
    "punchy", "concise", "elegant", "polished", "sparkling", "snappy",
    "dry", "wry",
})
NEG_STYLE_WORDS = frozenset({
    "flat", "rambling", "dull", "wordy", "clunky", "awkward", "vague",
    "bland", "loose", "verbose", "meandering", "clumsy", "muddled",
    "confusing", "weak",
})

LLM_FIELDS = {
    "pun_pivot": (
        "In up to 6 words, name the exact pun/homophone/double-meaning the "
        "joke's phrasing hinges on, or say NONE if there is no such device."
    ),
    "phrasing_sharpness": (
        "In 3-6 words, say whether the wording/delivery is sharp and "
        "economical or flat and rambling."
    ),
}


def _content_words(body: str):
    toks = re.findall(r"[A-Za-z']+", body.lower())
    return [t for t in toks if len(t) > 1 and t not in STOPWORDS]


def _alliteration_score(body: str) -> float:
    cw = _content_words(body)
    if len(cw) < 2:
        return 0.0
    matches = sum(1 for a, b in zip(cw, cw[1:]) if a[0] == b[0])
    return max(0.0, min(1.0, (matches / max(1, len(cw) - 1)) * 5.0))


def _rhyme_score(body: str) -> float:
    clauses = re.split(r"[.!?\n]+", body)
    ends = []
    for c in clauses:
        toks = re.findall(r"[A-Za-z']+", c.lower())
        if toks:
            ends.append(toks[-1])
    if len(ends) < 2:
        return 0.0
    matches = sum(
        1 for a, b in zip(ends, ends[1:])
        if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:] and a != b
    )
    return max(0.0, min(1.0, (matches / max(1, len(ends) - 1)) * 3.0))


def _symmetry_score(body: str) -> float:
    words = re.findall(r"[A-Za-z']+", body.lower())
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    counts = Counter(bigrams)
    if not counts:
        return 0.0
    top = counts.most_common(1)[0][1]
    return max(0.0, min(1.0, (top - 1) / 3.0))


def _sent_stats_safe(ops, body: str):
    """Tolerate both the pack's documented tuple return and a dict variant."""
    try:
        r = ops.sent_stats(body)
    except Exception:
        return (1, 10.0, 0.1)
    try:
        if isinstance(r, dict):
            n_sent = r.get("n_sent", r.get("num_sentences", 1))
            mean_wps = r.get("mean_words_per_sent", r.get("mean_wps", 10.0))
            frac_long = r.get("frac_long_words", r.get("frac_long", 0.1))
            return (n_sent, mean_wps, frac_long)
        if isinstance(r, (tuple, list)) and len(r) >= 3:
            return (r[0], r[1], r[2])
    except Exception:
        pass
    return (1, 10.0, 0.1)


def _economy_score(body: str, ops) -> float:
    n_sent, _mean_wps, frac_long = _sent_stats_safe(ops, body)

    clauses = [c.strip() for c in re.split(r"[.!?\n]+", body) if c.strip()]
    if len(clauses) >= 2:
        last_len = len(re.findall(r"[A-Za-z']+", clauses[-1]))
        prior_lens = [len(re.findall(r"[A-Za-z']+", c)) for c in clauses[:-1]]
        prior_mean = sum(prior_lens) / max(1, len(prior_lens))
        ratio = last_len / max(1.0, prior_mean)
        punch_bonus = 1.0 if ratio <= 0.9 else max(0.0, 1.0 - (ratio - 0.9))
    else:
        punch_bonus = 0.5

    try:
        frac_long = float(frac_long) if frac_long is not None else 0.1
    except Exception:
        frac_long = 0.1
    if 0.03 <= frac_long <= 0.20:
        clarity = 1.0
    elif frac_long < 0.03:
        clarity = 0.6 + (frac_long / 0.03) * 0.4
    else:
        clarity = max(0.0, 1.0 - (frac_long - 0.20) * 2.0)

    try:
        n_sent = int(n_sent) if n_sent else max(1, len(clauses))
    except Exception:
        n_sent = max(1, len(clauses))
    conciseness = 1.0 if n_sent <= 8 else max(0.0, 1.0 - (n_sent - 8) * 0.05)

    return max(0.0, min(1.0, 0.45 * punch_bonus + 0.30 * clarity + 0.25 * conciseness))


def _rarity_score(text: str, ops) -> float:
    """Lexical/phrasing rarity via TF-IDF neighbor similarity (evidence op).
    Lower average similarity to nearest corpus neighbors ~ rarer/more
    distinctive phrasing, which the criterion names directly ("rarity")."""
    try:
        raw = None
        try:
            raw = ops.retrieve_similar(text, k=6)
        except TypeError:
            raw = ops.retrieve_similar(text)
        if not raw:
            return 0.5
        sims = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            a, b = item[0], item[1]
            if isinstance(a, (int, float)):
                sims.append(float(a))
            elif isinstance(b, (int, float)):
                sims.append(float(b))
        if not sims:
            return 0.5
        sims.sort(reverse=True)
        if sims[0] >= 0.999:  # likely self-match in the indexed corpus
            sims = sims[1:]
        if not sims:
            return 0.5
        avg_sim = sum(sims) / len(sims)
        return max(0.0, min(1.0, 1.0 - avg_sim))
    except Exception:
        return 0.5


def _precision_signal(sharp_raw: str) -> float:
    if not sharp_raw:
        return 0.5
    toks = set(re.findall(r"[a-z']+", sharp_raw.lower()))
    pos_hits = len(toks & POS_STYLE_WORDS)
    neg_hits = len(toks & NEG_STYLE_WORDS)
    val = 0.5 + 0.25 * (pos_hits - neg_hits)
    return max(0.0, min(1.0, val))


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.5

        try:
            norm = ops.normalize(text)
            if not norm:
                norm = text
        except Exception:
            norm = text

        body = norm[:4000]  # bound analysis; keeps clear of end-of-doc boilerplate

        alliteration = _alliteration_score(body)
        rhyme = _rhyme_score(body)
        symmetry = _symmetry_score(body)
        code_phonetic = max(
            0.0, min(1.0, 0.45 * alliteration + 0.35 * rhyme + 0.20 * symmetry)
        )

        economy = _economy_score(body, ops)
        rarity = _rarity_score(text, ops)

        pun_raw = (extracted.get("pun_pivot", "") or "").strip()
        pun_norm = pun_raw.lower()
        pun_flag = 0.0 if (pun_norm in ("", "n/a", "na") or pun_norm.startswith("none")) else 1.0

        sharp_raw = (extracted.get("phrasing_sharpness", "") or "").strip()
        precision = _precision_signal(sharp_raw)

        raw_score = (
            0.10 +                     # floor
            0.40 * pun_flag +          # LLM: actual wordplay/double-meaning present
            0.15 * precision +         # LLM: sharp/economical vs flat/rambling
            0.15 * code_phonetic +     # code: alliteration/rhyme/parallel structure
            0.10 * rarity +            # evidence: distinctiveness of phrasing
            0.10 * economy             # code: punchline economy + clarity + conciseness
        )
        return max(0.0, min(1.0, raw_score))
    except Exception:
        return 0.5

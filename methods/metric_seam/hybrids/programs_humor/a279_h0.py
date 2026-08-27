"""
Hybrid metric channel for aspect a279: "Misdirection and reveal design"

Criterion: Primes an initial reading with fair cues and conceals an alternate
interpretation, then flips expectations with a justified reveal.

Design notes:
- The core construct ("does the ending reinterpret an earlier, fairly-planted
  detail, rather than just being a random/shock non-sequitur") is not reliably
  reachable by regex/keyword code -- the frozen baseline's keyword list
  (train_rho ~ -0.02) confirms surface keywords ("twist", "reveal", "but then")
  do not track judged quality here. So the two LLM_FIELDS carry the semantic
  predicate: (1) does a reveal/punchline phrase exist at all, and (2) does
  that reveal reinterpret something planted earlier in the text (the "fair"
  and "justified" parts of the criterion), as opposed to being an ungrounded
  non-sequitur or shock beat.
- Code supplies the structural corroboration that a human reader of comic
  prose would also notice: an abrupt short final line/beat (classic punchline
  delivery), dialogue/quoted-speech framing (most reveals here are delivered
  as a line of dialogue), a mild penalty for "pun-avalanche" texts that just
  chain many puns on one topic word instead of building to a single reveal,
  and light sanity guards (degenerate length, corpus-outlier via TF-IDF
  neighbor similarity, dense date-lists suggesting scraped non-joke content).
- Ordering intentionally makes an UNJUSTIFIED reveal (a twist that doesn't
  reinterpret any earlier cue -- a pure non-sequitur) score BELOW having no
  reveal at all, because the criterion explicitly requires the reveal to be
  "justified"; an ungrounded twist is a direct violation of the criterion,
  not merely an absence of one.
"""

import re
from collections import Counter

LLM_FIELDS = {
    "reveal_phrase": "In under 8 words, name the twist/punchline line that recontextualizes the setup; say NONE if there is no reveal.",
    "reveal_setup_link": "In under 10 words, name the earlier planted detail the reveal reinterprets; say NONE if the ending doesn't fairly connect to anything earlier.",
}

_STOP = set(
    "the that this with have from they were what when your about would "
    "could should there their which where because into some just like "
    "than then them been being does doesn said says will your".split()
)


def _last_sentence_words(norm_text):
    parts = [p.strip() for p in re.split(r"[.!?\n]+", norm_text) if p.strip()]
    if not parts:
        return 0
    last = parts[-1]
    return len(re.findall(r"[A-Za-z']+", last))


def _dialogue_score(norm_text):
    try:
        quotes = norm_text.count('"') + norm_text.count("'")
        verbs = len(
            re.findall(
                r"\b(said|says?|ask(?:s|ed)?|repl(?:y|ies|ied)|yell(?:s|ed)?|"
                r"whisper(?:s|ed)?|shout(?:s|ed)?)\b",
                norm_text.lower(),
            )
        )
        raw = verbs * 0.25 + min(quotes, 10) * 0.03
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.0


def _repetition_penalty(norm_text):
    try:
        toks = [w for w in re.findall(r"[A-Za-z']+", norm_text.lower()) if len(w) >= 4 and w not in _STOP]
        if len(toks) < 15:
            return 0.0
        counts = Counter(toks)
        top_word, top_n = counts.most_common(1)[0]
        ratio = top_n / len(toks)
        if ratio <= 0.05:
            return 0.0
        return min(0.08, (ratio - 0.05) * 1.6)
    except Exception:
        return 0.0


def _length_penalty(norm_text):
    n = len(norm_text)
    if n < 30:
        return 0.10
    if n > 3500:
        return 0.06
    return 0.0


def _date_penalty(ops, norm_text):
    try:
        dates = ops.extract_dates(norm_text)
        if dates and len(dates) >= 2:
            return 0.05
        return 0.0
    except Exception:
        return 0.0


def _outlier_penalty(ops, norm_text):
    try:
        neighbors = ops.retrieve_similar(norm_text, k=5)
        if not neighbors:
            return 0.0
        sims = []
        for item in neighbors:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            a, b = item[0], item[1]
            # contract order is (similarity, datapoint_id); guard against
            # the reversed (datapoint_id, similarity) convention as well.
            if isinstance(a, (int, float)):
                sims.append(float(a))
            elif isinstance(b, (int, float)):
                sims.append(float(b))
        if not sims:
            return 0.0
        best = max(sims)
        if best < 0.03:
            return 0.05
        return 0.0
    except Exception:
        return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.0

        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm:
                norm = text
        except Exception:
            norm = text

        ext = extracted if isinstance(extracted, dict) else {}
        reveal_phrase = str(ext.get("reveal_phrase", "") or "").strip()
        setup_link = str(ext.get("reveal_setup_link", "") or "").strip()

        def _is_none(s):
            return (not s) or s.strip().upper() in ("NONE", "N/A", "NA", "-")

        has_reveal = not _is_none(reveal_phrase)
        has_link = not _is_none(setup_link)

        # Core: presence + justification of the reveal is the heart of the
        # criterion. An unjustified "twist" (non-sequitur/shock beat with no
        # tie to an earlier cue) scores WORSE than no reveal at all, since the
        # criterion requires the flip to be "justified".
        if has_reveal and has_link:
            core = 0.85
        elif (not has_reveal) and (not has_link):
            core = 0.25
        elif has_reveal and not has_link:
            core = 0.05
        else:
            # extractor found a setup-link but no crisp reveal phrase; treat
            # as weak/ambiguous partial credit.
            core = 0.40

        # Structural corroboration.
        try:
            st = ops.sent_stats(norm)
        except Exception:
            st = None

        n_sent, mean_wps, frac_long = 0, 0.0, 0.0
        if isinstance(st, dict):
            n_sent = st.get("n_sent", 0) or 0
            mean_wps = st.get("mean_words_per_sent", 0.0) or 0.0
            frac_long = st.get("frac_long_words", 0.0) or 0.0
        elif isinstance(st, (tuple, list)) and len(st) >= 3:
            n_sent, mean_wps, frac_long = st[0] or 0, st[1] or 0.0, st[2] or 0.0

        abrupt_bonus = 0.0
        if has_reveal and mean_wps and mean_wps > 0:
            last_words = _last_sentence_words(norm)
            if last_words > 0:
                ratio = last_words / mean_wps
                if ratio <= 1.3:
                    abrupt_bonus = 0.15 * max(0.0, min(1.0, (1.3 - ratio) / 1.3))

        dialogue_bonus = 0.05 * _dialogue_score(norm)
        rep_pen = _repetition_penalty(norm)
        len_pen = _length_penalty(norm)
        date_pen = _date_penalty(ops, norm)
        outlier_pen = _outlier_penalty(ops, norm)

        raw = core + abrupt_bonus + dialogue_bonus - rep_pen - len_pen - date_pen - outlier_pen
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5

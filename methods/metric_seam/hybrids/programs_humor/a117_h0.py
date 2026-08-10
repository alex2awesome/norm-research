import re
import collections
import statistics

# Criterion (a117): "Economy, brevity, and clarity" -- plain, tight language;
# clean setups, concise punches, no fat; every line advances joke/character/plot.
#
# Design rationale:
#   - Economy/brevity is primarily a LENGTH signal (soft decay past a
#     joke-appropriate size), plus sentence-level plainness (short sentences,
#     not too many long/complex words).
#   - "No fat" / "every line advances" maps to: (a) exact-duplicate sentences
#     (restating without adding anything), (b) scraped boilerplate / decorative
#     symbol-runs / mojibake-style separators / attribution-credit lines, which
#     are corpus noise that code can detect directly.
#   - "Clean setups, concise punches" maps to a structural check: is the final
#     sentence short relative to the rest of the piece (a crisp landing) rather
#     than an over-explained or padded ending.
#   - The one thing code genuinely cannot see is whether the text contains an
#     off-topic apologetic/meta aside (not part of the joke) and whether there
#     is a single identifiable payoff line at all (vs. rambling/repeating). Both
#     require reading comprehension, so they are delegated to the two LLM
#     fields; the predicate (how much to penalize/reward) stays in code.

LLM_FIELDS = {
    "filler_aside": (
        "Quote (<=12 words) any off-topic apologetic/disclaimer/meta-commentary "
        "aside that is not part of the joke itself (e.g. 'sorry', 'not racist', "
        "'just a thought', edit notes); answer NONE if there is no such aside."
    ),
    "punchline": (
        "Copy the final punchline/payoff line of the joke in <=15 words; answer "
        "NONE if the text has no single clear payoff (it rambles, repeats itself, "
        "or trails off without landing)."
    ),
}


def _sent_split(s):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.5

        raw = text
        norm = ops.normalize(text)
        if not norm:
            norm = text

        tokens = re.findall(r"\b\w+\b", norm)
        nw = max(1, len(tokens))

        # --- 1. Length economy: joke-appropriate size decays gently past ~70 words
        if nw <= 70:
            length_score = 1.0
        else:
            length_score = 1.0 / (1.0 + (nw - 70) / 130.0)
        length_score = max(0.1, min(1.0, length_score))

        # --- 2. Sentence-level plainness/clarity via ops.sent_stats
        # Contract text disagrees slightly on shape (dict vs tuple) across docs;
        # handle both defensively.
        try:
            ss = ops.sent_stats(norm)
        except Exception:
            ss = None
        n_sent, mean_wps, frac_long = 1, float(nw), 0.0
        if isinstance(ss, dict):
            n_sent = ss.get("n_sent", ss.get("num_sentences", 1)) or 1
            mean_wps = ss.get("mean_words_per_sent", ss.get("mean_wps", nw)) or nw
            frac_long = ss.get("frac_long_words", ss.get("frac_long", 0.0)) or 0.0
        elif isinstance(ss, (list, tuple)) and len(ss) >= 3:
            n_sent, mean_wps, frac_long = ss[0], ss[1], ss[2]

        if mean_wps <= 20:
            sent_score = 1.0
        else:
            sent_score = max(0.2, 1.0 - (mean_wps - 20) / 40.0)
        sent_score = max(0.0, sent_score - max(0.0, frac_long - 0.25) * 0.6)
        sent_score = max(0.0, min(1.0, sent_score))

        # --- 3. Redundancy: exact-duplicate sentences restate without advancing
        sents = [s.lower() for s in _sent_split(norm) if len(s) > 8]
        redundancy_penalty = 0.0
        if len(sents) >= 2:
            counts = collections.Counter(sents)
            dup = sum(c - 1 for c in counts.values() if c > 1)
            redundancy_penalty = min(0.35, (dup / max(1, len(sents))) * 0.7)
        non_redundancy = 1.0 - redundancy_penalty

        # --- 4. Clean-setup / concise-punch structure: is the ending crisp?
        raw_sents = _sent_split(norm)
        if len(raw_sents) >= 2:
            last_len = len(re.findall(r"\w+", raw_sents[-1]))
            other_lens = [len(re.findall(r"\w+", s)) for s in raw_sents[:-1]]
            mean_other = statistics.mean(other_lens) if other_lens else last_len
            if mean_other > 0:
                ratio = last_len / mean_other
                if ratio <= 1.1:
                    landing_score = 1.0
                else:
                    landing_score = max(0.3, 1.0 - (ratio - 1.1) * 0.4)
            else:
                landing_score = 0.7
        else:
            landing_score = 0.6

        # --- 5. Scraper boilerplate / decorative separators / mojibake noise
        # (checked against the RAW text, since ops.normalize may already clean
        # some of this away and we want to detect its presence in the source).
        boilerplate_patterns = [
            r"&amp;", r"&#x[0-9a-fA-F]+;", r"\\\*", r"https?://",
            r"\bcredit\b", r"\bedit\d*\s*:", r"\bnsfw\b", r"\bu/\w+", r"\br/\w+",
            r"�",
        ]
        boiler_hits = sum(1 for p in boilerplate_patterns if re.search(p, raw, re.IGNORECASE))
        symbol_runs = len(re.findall(r"[^\w\s]{4,}", raw))
        noise_penalty = min(0.35, 0.08 * boiler_hits + 0.06 * symbol_runs)

        # --- 6. LLM-grounded: off-topic filler/disclaimer aside not advancing the joke
        filler = str(extracted.get("filler_aside", "") or "").strip().upper()
        filler_penalty = 0.0 if filler in ("", "NONE") else 0.35

        # --- 7. LLM-grounded: is there a single identifiable payoff line?
        punch = str(extracted.get("punchline", "") or "").strip().upper()
        punch_score = 1.0 if punch not in ("", "NONE") else 0.45

        combined = (
            0.35 * length_score +
            0.15 * sent_score +
            0.15 * non_redundancy +
            0.20 * punch_score +
            0.15 * landing_score
        )
        out = combined - noise_penalty - filler_penalty
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5

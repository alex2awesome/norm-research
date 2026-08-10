"""
Hybrid channel for a189: "Compressed, quotable phrasing"
Craft brief, precise lines with full meaning that click instantly and travel.

Design intuition (from the general criterion, not the 30 train rows):
- The construct is NOT "how many comic-punctuation marks / topic keywords are
  present" (that's what the frozen baseline does, and it barely beats chance,
  rho=0.149). It is whether the piece delivers its payoff in a short, tight,
  self-contained line that would still land if quoted out of context.
- Two things code cannot see on its own:
    (1) WHICH line in the document is actually the punchline/twist line
        (raw "last sentence" heuristics get fooled by trailing Reddit
        boilerplate: credit lines, /u/ handles, source URLs -- the corpus
        notes flag this explicitly), and
    (2) WHETHER that line is semantically self-contained / portable
        ("clicks instantly and travels") -- a purely structural/length
        measure can't tell a punchline that only works in-context (e.g. a
        repeated-name gag that needs the whole setup) from one that stands
        alone as a quotable line.
  Both are handed to an LLM extractor; code supplies the length/economy
  arithmetic and a boilerplate-aware fallback so the channel degrades
  gracefully if the extractor returns nothing useful.
"""

import re

LLM_FIELDS = {
    "punchline": (
        "Quote verbatim the single shortest line that delivers this piece's "
        "twist, reveal, or payoff; answer NONE if no single line carries it."
    ),
    "lands_standalone": (
        "If that quoted line were read alone with no other context, would it "
        "still be a complete, funny, self-contained line? Answer YES, PARTLY, or NO."
    ),
}

_BOILERPLATE_LINE = re.compile(
    r"""^\s*(
        https?://\S+ |
        /u/\w+ |
        /r/\w+ |
        (source|credit|via)\b.* |
        edit\s*\d*\s*: .* |
        tl;?dr\b.* |
        \S+@\S+\.\S+
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_QUOTE_CHARS = "\"'`‘’“” "


def _strip_boilerplate(text):
    lines = text.splitlines()
    # trim trailing boilerplate/contact/attribution lines (they cluster at the end)
    while lines and (not lines[-1].strip() or _BOILERPLATE_LINE.match(lines[-1].strip())):
        lines.pop()
    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else text


def _dedupe_title_echo(text):
    # some docs open with "<title> <title-repeated-as-first-sentence>"
    head = text[:70].strip()
    if len(head) < 12:
        return text
    rest = text[len(head):len(head) + 220]
    if head[:40] and head[:40].lower() in rest.lower():
        idx = text.lower().find(head[:40].lower(), len(head))
        if idx != -1:
            return text[idx:].strip()
    return text


def _clean_quote(s):
    return s.strip().strip(_QUOTE_CHARS).strip()


def _is_none_answer(s):
    t = re.sub(r"[^a-z]", "", s.lower())
    return t in ("none", "na", "nonegiven", "")


def _word_list(s):
    return re.findall(r"[A-Za-z0-9']+", s or "")


def _verify_present(punchline, core):
    p_words = set(w.lower() for w in _word_list(punchline) if len(w) >= 4)
    if not p_words:
        return True
    core_words = set(w.lower() for w in _word_list(core))
    overlap = len(p_words & core_words)
    return (overlap / len(p_words)) >= 0.6


def _compression_from_wordcount(pw):
    if pw <= 0:
        return 0.3
    if pw <= 3:
        return 0.75
    if pw <= 14:
        return 1.0
    if pw <= 22:
        return 1.0 - (pw - 14) / 16.0
    return max(0.15, 0.5 - (pw - 22) / 40.0)


def _sentence_brevity(mean_wps):
    try:
        m = float(mean_wps)
    except (TypeError, ValueError):
        return 0.5
    if m <= 0:
        return 0.5
    if 4 <= m <= 14:
        return 1.0
    if m < 4:
        return 0.6
    return max(0.2, 1.0 - (m - 14) / 20.0)


def _word_simplicity(frac_long):
    try:
        f = float(frac_long)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, 1.0 - f))


def _length_economy(n):
    if n <= 0:
        return 0.5
    if n <= 60:
        return 1.0
    if n <= 120:
        return 0.85
    if n <= 220:
        return 0.65
    if n <= 350:
        return 0.45
    return 0.25


def _standalone_from_answer(ans):
    a = (ans or "").strip().upper()
    if a.startswith("Y"):
        return 1.0
    if a.startswith("P"):
        return 0.55
    if a.startswith("N"):
        return 0.15
    return 0.5


def _fallback_punchline(core):
    sents = [s.strip() for s in re.split(r"[.!?\n]+", core) if s.strip()]
    for s in reversed(sents):
        if _BOILERPLATE_LINE.match(s):
            continue
        if len(_word_list(s)) >= 2:
            return s
    return ""


def _sent_stats(ops, core):
    try:
        stats = ops.sent_stats(core)
    except Exception:
        return 0, 0.0, 0.0
    try:
        if isinstance(stats, dict):
            return (
                stats.get("n_sent", 0),
                stats.get("mean_words_per_sent", 0.0),
                stats.get("frac_long_words", 0.0),
            )
        n_sent, mean_wps, frac_long = stats
        return n_sent, mean_wps, frac_long
    except Exception:
        return 0, 0.0, 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.0

        extracted = extracted or {}

        norm = ops.normalize(text) if hasattr(ops, "normalize") else text
        if not isinstance(norm, str) or not norm.strip():
            norm = text

        core = _strip_boilerplate(norm)
        core = _dedupe_title_echo(core)
        if not core.strip():
            core = norm

        total_words = len(_word_list(core))
        n_sent, mean_wps, frac_long = _sent_stats(ops, core)

        structural = (
            0.5 * _length_economy(total_words)
            + 0.3 * _sentence_brevity(mean_wps)
            + 0.2 * _word_simplicity(frac_long)
        )

        punchline_raw = str(extracted.get("punchline", "") or "").strip()
        standalone_raw = str(extracted.get("lands_standalone", "") or "").strip()

        punchline = _clean_quote(punchline_raw)
        if punchline and _is_none_answer(punchline):
            punchline = ""

        if punchline:
            verified = _verify_present(punchline, core)
            pw = len(_word_list(punchline))
            compression = _compression_from_wordcount(pw)
            if not verified:
                compression = 0.5 * compression + 0.5 * 0.5
        else:
            fb = _fallback_punchline(core)
            if fb:
                pw = len(_word_list(fb))
                compression = 0.6 * _compression_from_wordcount(pw)
            else:
                compression = 0.3

        standalone = _standalone_from_answer(standalone_raw)

        raw = 0.50 * standalone + 0.30 * compression + 0.20 * structural
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5

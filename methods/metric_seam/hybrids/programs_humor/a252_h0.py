"""
Hybrid metric channel for a252: "Ensemble chemistry, balance, and adaptability"
(Leads and supporting players spark and share spotlight equitably; the group
shows cohesion and can flex between configurations.)

Textual analog for short reddit-style jokes: an "ensemble" moment is a piece
that is built from a genuine multi-voice EXCHANGE (not just a single narrator,
and not just several characters listed one after another with no response to
each other) where:
  - multiple distinct speakers actually get turns (cohesion/chemistry),
  - their turns are roughly balanced in length (shared spotlight), and
  - the turns rhyme with each other structurally -- a repeated template with
    a twist (rule-of-three, mirrored retorts, topping) -- which is the textual
    stand-in for "flexing between configurations."

Code can reliably find dialogue turns, script-style speaker labels, and the
lexical overlap / length balance between turns. It cannot reliably tell
whether characters that co-occur are truly *interacting* versus merely
enumerated, nor whether a punchline *re-casts* the relationship between
characters (rivals turning out to be allies, roles flipping, etc.) -- both
are thick semantic judgments, so they are delegated to the two LLM fields.
"""

import re
import statistics

LLM_FIELDS = {
    "cast_dynamic": "Classify the interaction: SOLO (one voice/narrator), PARALLEL (multiple characters act or speak but never respond to each other), or BANTER (distinct characters trade lines and respond to each other).",
    "reconfigure": "Does the punchline re-cast the relationship between characters (e.g. rivals turn out to be allies, roles reverse)? Answer YES, PARTIAL, or NO.",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "is",
    "was", "were", "are", "i", "you", "he", "she", "it", "we", "they", "that",
    "this", "for", "with", "as", "be", "been", "so", "if", "then", "just",
    "not", "no", "do", "did", "does", "my", "your", "his", "her", "its",
    "our", "their", "me", "him", "them", "us",
}

_NOTE_LABELS = {
    "edit", "update", "note", "tldr", "tl;dr", "eta", "nsfw", "warning",
    "disclaimer", "context", "source", "title",
}

_QUOTE_RE = re.compile(r'["“]([^"”]{2,400})["”]')

_SCRIPT_LINE_RE = re.compile(
    r"(?m)^\s*([A-Za-z][A-Za-z0-9 '\-]{0,20}):\s*(\S.{0,400})$"
)

_ATTR_BEFORE_RE = re.compile(
    r"\b([A-Za-z][a-zA-Z]{1,20})\s+"
    r"(?:said|says|say|ask(?:s|ed)?|repl(?:y|ies|ied)|"
    r"shout(?:s|ed)?|exclaim(?:s|ed)?|answer(?:s|ed)?|respond(?:s|ed)?|"
    r"tells|told)\b",
    re.IGNORECASE,
)

_ATTR_AFTER_RE = re.compile(
    r'["”]\s*,?\s*(?:said|says|asked|replied|shouted|exclaimed|'
    r"answered|responded|told)\s+(?:the\s+)?([a-zA-Z]{2,20})\b",
    re.IGNORECASE,
)


def _words(s):
    try:
        return [w for w in re.findall(r"[a-zA-Z']+", s.lower()) if w not in _STOPWORDS]
    except Exception:
        return []


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    if not union:
        return 0.0
    return len(sa & sb) / union


def _extract_turns(text):
    """Return a list of (speaker_label_or_None, content_str) dialogue turns."""
    turns = []

    for m in _SCRIPT_LINE_RE.finditer(text):
        label = m.group(1).strip().lower()
        content = m.group(2).strip()
        if label in _NOTE_LABELS:
            continue
        if 1 <= len(label.split()) <= 3 and len(content) >= 2:
            turns.append((label, content))
    if len(turns) >= 2:
        return turns

    turns = []
    for sp in _QUOTE_RE.finditer(text):
        content = sp.group(1).strip()
        if len(content) < 1:
            continue
        window_before = text[max(0, sp.start() - 60):sp.start()]
        window_after = text[sp.end():sp.end() + 60]
        label = None
        before_matches = list(_ATTR_BEFORE_RE.finditer(window_before))
        if before_matches:
            label = before_matches[-1].group(1).lower()
        else:
            after_match = _ATTR_AFTER_RE.search(window_after)
            if after_match:
                label = after_match.group(1).lower()
        turns.append((label, content))
    return turns


def _turn_features(turns):
    """Structural features over dialogue turns.

    `mirror` and `balance` are computed only over "substantive" turns (>=2
    content words), so short connective interjections ("Yeah,", "Well,")
    sitting between two mirrored lines don't mask the mirroring. `mirror` is
    each substantive turn's BEST lexical overlap with any other substantive
    turn (not just its neighbor), so a mirrored pair separated by an
    interjection is still detected.
    """
    n_turns = len(turns)
    n_speakers = len({lbl for lbl, _ in turns if lbl})
    if n_turns < 2:
        return {"n_turns": n_turns, "n_speakers": n_speakers, "mirror": 0.0, "balance": 0.0}

    substantive = [(lbl, content, _words(content)) for lbl, content in turns]
    substantive = [t for t in substantive if len(t[2]) >= 2]

    if len(substantive) < 2:
        return {"n_turns": n_turns, "n_speakers": n_speakers, "mirror": 0.0, "balance": 0.0}

    lengths = [len(t[2]) for t in substantive]
    mean_len = sum(lengths) / len(lengths)
    try:
        cv = statistics.pstdev(lengths) / mean_len if mean_len > 0 else 1.0
    except Exception:
        cv = 1.0
    balance = max(0.0, 1.0 - min(1.0, cv))

    best_sims = []
    for i in range(len(substantive)):
        best = 0.0
        for j in range(len(substantive)):
            if i == j:
                continue
            s = _jaccard(substantive[i][2], substantive[j][2])
            if s > best:
                best = s
        best_sims.append(best)
    mirror = sum(best_sims) / len(best_sims) if best_sims else 0.0

    return {"n_turns": n_turns, "n_speakers": n_speakers, "mirror": mirror, "balance": balance}


def _sim_value(pair):
    """Defensively pull the numeric similarity out of a (sim, id) or (id, sim) tuple."""
    try:
        a, b = pair[0], pair[1]
    except Exception:
        return None
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return float(a)
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return float(b)
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.5

        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm.strip():
                norm = text
        except Exception:
            norm = text

        turns = _extract_turns(norm)
        feats = _turn_features(turns)
        n_turns = feats["n_turns"]
        n_speakers = feats["n_speakers"]
        mirror = feats["mirror"]
        balance = feats["balance"]

        try:
            n_sent, _mean_wps, _frac_long = ops.sent_stats(norm)
            n_sent = float(n_sent)
        except Exception:
            n_sent = float(max(1, norm.count(".") + norm.count("!") + norm.count("?")))

        turn_density = min(1.0, n_turns / max(3.0, n_sent * 0.6)) if n_turns else 0.0

        if n_turns >= 2:
            code_score = 0.5 * mirror + 0.3 * balance + 0.2 * turn_density
        elif n_turns == 1:
            code_score = 0.08
        else:
            code_score = 0.02
        code_score = max(0.0, min(1.0, code_score))

        dynamic = str((extracted or {}).get("cast_dynamic", "") or "").strip().upper()
        if "SOLO" in dynamic:
            base = min(code_score, 0.15)
        elif "PARALLEL" in dynamic:
            # Characters co-occur but don't respond to each other: enumeration,
            # not chemistry. Low ceiling regardless of how tidy the list is.
            base = 0.05 + 0.25 * code_score
        elif "BANTER" in dynamic:
            base = code_score
        else:
            # No usable LLM signal (or extractor answered NONE): trust code, discounted.
            base = 0.8 * code_score
            if n_speakers >= 2 and n_turns >= 2:
                base = max(base, 0.10 + 0.4 * code_score)

        reconf = str((extracted or {}).get("reconfigure", "") or "").strip().upper()
        if "YES" in reconf:
            bonus = 0.20
        elif "PARTIAL" in reconf:
            bonus = 0.10
        else:
            bonus = 0.0

        final = base + bonus

        try:
            sims = ops.retrieve_similar(text, k=5)
            if sims:
                vals = [v for v in (_sim_value(p) for p in sims) if v is not None]
                if vals and max(vals) < 0.02:
                    final *= 0.9
        except Exception:
            pass

        final = max(0.0, min(1.0, final))
        return float(final)
    except Exception:
        return 0.5

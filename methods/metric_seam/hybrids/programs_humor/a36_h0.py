import re
import math
import statistics
from collections import Counter

# a36 -- "Writing economy and clarity": prioritize the core setup and payoff;
# include only what serves the joke; keep language/storyline simple and clean.
#
# Design: the baseline (v1_structure) only looks at raw word-count-near-40 and
# a crude filler ratio, which over-rates rambling/meta texts that happen to be
# short (e.g. a two-line joke followed by an unrelated disclaimer) and
# under-rates long-but-tightly-escalating jokes (rule-of-three builds, extended
# suspense before a single payoff). Code alone cannot tell "this aside serves
# the joke" from "this aside is a disclaimer/tangent", and cannot tell "this
# buildup earns its length" from "this buildup is padding" -- both are thick,
# semantic judgments -- so those two constructs are pushed to LLM_FIELDS and
# used as short, code-checked predicates (not raw LLM scores). Regex/stdlib
# handles what's cheaply mechanical: filler-word density, repeated n-grams
# (title echoes, restated phrases -- a concrete, checkable proxy for
# "unnecessary" content), sentence complexity, and a soft length prior.

LLM_FIELDS = {
    "extraneous_note": "In <=8 words, name any disclaimer, apology, or meta-aside that does not serve the joke, else say NONE.",
    "padding_verdict": "Say ECONOMICAL if the setup and payoff are tight and every part earns its place, or PADDED if the buildup feels redundant/bloated.",
}

_FILLERS = {
    "basically", "actually", "literally", "really", "very", "just", "like",
    "sort", "kind", "stuff", "thing", "things", "somehow", "somewhat",
    "totally", "definitely", "probably", "essentially", "honestly",
}

_NONE_MARKERS = {"none", "no", "n/a", "na", "nothing", "-", ""}


def _safe_normalize(text, ops):
    try:
        if ops is not None and hasattr(ops, "normalize"):
            out = ops.normalize(text)
            if isinstance(out, str) and out:
                return out
    except Exception:
        pass
    return text


def _safe_sent_stats(text, ops):
    # Always returns (n_sent, mean_words_per_sent, frac_long_words).
    # Contract says the op returns a 3-tuple, but we defensively also accept
    # a dict (in case the harness wraps it) and fall back to a manual
    # computation if the op is missing/misbehaves.
    try:
        if ops is not None and hasattr(ops, "sent_stats"):
            raw = ops.sent_stats(text)
            if isinstance(raw, dict):
                n_sent = raw.get("n_sent", raw.get("num_sentences", 1)) or 1
                mean_wps = raw.get("mean_words_per_sent", raw.get("mean_wps", 0.0)) or 0.0
                frac_long = raw.get("frac_long_words", 0.0) or 0.0
                return n_sent, mean_wps, frac_long
            if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                return raw[0], raw[1], raw[2]
    except Exception:
        pass
    try:
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        n_sent = max(1, len(sents))
        words = text.split()
        n = max(1, len(words))
        mean_wps = n / n_sent
        long_ct = sum(1 for w in words if len(w.strip(".,!?\"'();:")) > 6)
        frac_long = long_ct / n
        return n_sent, mean_wps, frac_long
    except Exception:
        return 1, 0.0, 0.0


def _safe_similarity(text, ops):
    # Robust to either (similarity, datapoint_id) or (datapoint_id, similarity)
    # tuple ordering.
    try:
        if ops is not None and hasattr(ops, "retrieve_similar"):
            neigh = ops.retrieve_similar(text, k=5)
            sims = []
            for item in neigh or []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                a, b = item[0], item[1]
                if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
                    sims.append(float(a))
                elif isinstance(b, (int, float)) and not isinstance(a, (int, float)):
                    sims.append(float(b))
                elif isinstance(a, (int, float)):
                    sims.append(float(a))
            if sims:
                return statistics.mean(sims)
    except Exception:
        pass
    return None


def _extraneous_sub(extracted):
    try:
        val = (extracted or {}).get("extraneous_note", "")
        v = (val or "").strip().lower().strip(".! ")
        if v in _NONE_MARKERS:
            return 1.0
        return 0.2
    except Exception:
        return 0.6


def _padding_sub(extracted):
    try:
        val = (extracted or {}).get("padding_verdict", "")
        v = (val or "").strip().lower()
        if "econom" in v:
            return 1.0
        if "pad" in v or "redundant" in v or "bloat" in v:
            return 0.2
        return 0.6
    except Exception:
        return 0.6


def _repetition_sub(words):
    # Fraction of repeated word-trigrams: catches title echoes ("Legend
    # tells... Legend tells...") and rambling restatements, a concrete
    # proxy for content that doesn't serve the joke.
    try:
        toks = [re.sub(r"[^a-z0-9']", "", w.lower()) for w in words]
        toks = [t for t in toks if t]
        if len(toks) < 6:
            return 1.0
        trigrams = list(zip(toks, toks[1:], toks[2:]))
        if not trigrams:
            return 1.0
        counts = Counter(trigrams)
        dup = sum(c - 1 for c in counts.values() if c > 1)
        ratio = dup / len(trigrams)
        return max(0.0, 1.0 - ratio * 4.0)
    except Exception:
        return 0.6


def _filler_sub(words):
    try:
        n = len(words)
        if n == 0:
            return 0.6
        fc = sum(1 for w in words if w.lower().strip(".,!?\"'();:") in _FILLERS)
        ratio = fc / n
        return max(0.0, 1.0 - min(1.0, ratio * 3.0))
    except Exception:
        return 0.6


def _length_sub(n):
    # Soft, wide prior -- long-but-earned buildups (e.g. escalating suspense
    # before one payoff) shouldn't be crushed the way a rigid ~40-word target
    # would crush them.
    try:
        return math.exp(-((n - 60) ** 2) / (2 * 150.0 ** 2))
    except Exception:
        return 0.6


def _complexity_sub(mean_wps, frac_long):
    try:
        a = 1.0 - min(1.0, frac_long * 2.0)
        if mean_wps <= 20:
            b = 1.0
        else:
            b = max(0.0, 1.0 - (mean_wps - 20.0) / 40.0)
        return 0.5 * a + 0.5 * b
    except Exception:
        return 0.6


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5

        norm = _safe_normalize(text, ops)
        words = norm.split() if isinstance(norm, str) and norm.strip() else text.split()
        n = len(words)
        if n == 0:
            return 0.5

        n_sent, mean_wps, frac_long = _safe_sent_stats(norm, ops)

        extraneous = _extraneous_sub(extracted)
        padding = _padding_sub(extracted)
        repetition = _repetition_sub(words)
        filler = _filler_sub(words)
        length = _length_sub(n)
        complexity = _complexity_sub(mean_wps, frac_long)

        raw = (
            0.30 * extraneous
            + 0.25 * padding
            + 0.15 * repetition
            + 0.10 * filler
            + 0.10 * length
            + 0.10 * complexity
        )

        # Regularize toward neutral for documents with near-zero lexical
        # overlap with the rest of the corpus (likely corrupted/atypical
        # scrape artifacts rather than genuine jokes) so we don't return an
        # overconfident extreme score on out-of-distribution input.
        avg_sim = _safe_similarity(norm, ops)
        if avg_sim is not None and avg_sim < 0.05:
            raw = 0.7 * raw + 0.3 * 0.5

        if raw < 0.0:
            raw = 0.0
        if raw > 1.0:
            raw = 1.0
        return raw
    except Exception:
        return 0.5

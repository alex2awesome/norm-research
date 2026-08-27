"""
Hybrid metric channel for aspect a216: "Structural/narrative coherence and focus"
(Clear through-line and organized progression; plots cohere, avoid meandering,
edits preserve pace and purpose.)

Design notes (from pack inspection only):
- Baseline (v2_holistic, train rho=0.138) leaned on transition-keyword density and
  a word-count "length normality" curve (targeting ~20-80 words). Its worst errors
  in the train excerpts were: (a) scoring a text that pastes TWO unrelated,
  independently-complete jokes back to back as coherent (0.48 vs judge 0.0), and
  (b) systematically UNDER-scoring many short, tightly-resolved judge=1.0 jokes
  (baseline 0.36-0.68) because they fall outside its "normal" length band.
- Corpus notes say texts are SHORT (median ~500 chars) reddit-style jokes, so
  brevity itself must NOT be penalized. The real construct is: does the text
  commit to ONE story/bit and carry it to a single resolution, versus splicing
  unrelated material together or trailing off/cutting off without a payoff.
- That construct (single throughline vs. spliced/disconnected parts; resolved vs.
  unresolved ending) is exactly the kind of thing code can't reliably detect from
  surface features alone, so it is delegated to two LLM fields. Code supplies
  robustness (degenerate-text handling, a cheap multi-opener/multi-block
  fragmentation heuristic that doesn't depend on length) and a small evidence-based
  tiebreaker from corpus similarity.
"""

import re
import math
import statistics

LLM_FIELDS = {
    "throughline": (
        "Answer ONE_PART or MULTI_PART: does this text commit to a single "
        "coherent story/joke building to one resolution, or does it splice "
        "together multiple unrelated topics, jokes, or tangents?"
    ),
    "resolution": (
        "Answer RESOLVED or UNRESOLVED: does the text reach a clear "
        "punchline/payoff/ending, or does it trail off, ramble, or cut off "
        "without one?"
    ),
}


def _safe_normalize(text, ops):
    try:
        norm = ops.normalize(text)
        if isinstance(norm, str) and norm.strip():
            return norm
    except Exception:
        pass
    return text if isinstance(text, str) else ""


def _safe_sent_stats(text, ops):
    """Return (n_sent, mean_words_per_sent, frac_long_words), tolerating either
    a dict or a tuple/list return shape from ops.sent_stats, and any failure."""
    try:
        r = ops.sent_stats(text)
    except Exception:
        r = None
    try:
        if isinstance(r, dict):
            n_sent = float(r.get("n_sent", r.get("num_sentences", 1)))
            mwps = float(r.get("mean_words_per_sent", r.get("mean_wps", 10.0)))
            flw = float(r.get("frac_long_words", 0.0))
            return (n_sent, mwps, flw)
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            return (float(r[0]), float(r[1]), float(r[2]))
    except Exception:
        pass
    # fallback: derive crude stats directly
    try:
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        n_sent = max(1, len(sents))
        words = text.split()
        mwps = len(words) / n_sent if n_sent else float(len(words))
        long_w = sum(1 for w in words if len(w) >= 7)
        flw = long_w / max(1, len(words))
        return (float(n_sent), float(mwps), float(flw))
    except Exception:
        return (1.0, 10.0, 0.0)


def _sim_value(pair):
    """Extract the numeric similarity from a (similarity, id) or (id, similarity)
    tuple, without assuming a fixed position."""
    try:
        a, b = pair[0], pair[1]
    except Exception:
        return 0.0
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return float(a)
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return float(b)
    return 0.0


def _classify(answer, positive_markers, negative_markers):
    """Map a short LLM extractor answer to +1 / -1 / 0 (neutral/unknown)."""
    if not answer or not isinstance(answer, str):
        return 0.0
    a = answer.strip().lower()
    if not a:
        return 0.0
    for m in negative_markers:
        if m in a:
            return -1.0
    for m in positive_markers:
        if m in a:
            return 1.0
    return 0.0


def _throughline_signal(extracted):
    ans = ""
    try:
        ans = extracted.get("throughline", "") if extracted else ""
    except Exception:
        ans = ""
    val = _classify(
        ans,
        positive_markers=["one_part", "one part", "single", "coheren", "throughline"],
        negative_markers=["multi_part", "multi part", "multiple", "disconnect", "unrelat", "splice", "spliced", "tangent"],
    )
    return 0.5 + 0.5 * val  # -1..1 -> 0..1


def _resolution_signal(extracted):
    ans = ""
    try:
        ans = extracted.get("resolution", "") if extracted else ""
    except Exception:
        ans = ""
    val = _classify(
        ans,
        positive_markers=["resolved", "punchline", "payoff", "clear end", "complete"],
        negative_markers=["unresolved", "trail", "cut off", "cuts off", "no payoff", "incomplete", "ramble"],
    )
    return 0.5 + 0.5 * val


_OPENER_RE = re.compile(
    r"\b("
    r"walk(?:s|ed)? into a bar|"
    r"there (?:was|once was)|"
    r"once upon a time|"
    r"a \w+ (?:and|,) a \w+ (?:and )?a? ?\w* walk|"
    r"knock knock|"
    r"so a \w+ (?:walks|goes|says)"
    r")\b",
    re.I,
)


def _code_structure_signal(norm_text):
    """Cheap, length-agnostic fragmentation heuristic: flag texts that look like
    two or more independently-complete narrative blocks glued together (the
    baseline's clearest failure mode), otherwise stay near-neutral so short,
    clean jokes are not penalized just for being short."""
    try:
        words = norm_text.split()
        n_words = len(words)
        if n_words < 5:
            return 0.35  # too degenerate to have any real structure

        blocks = [b.strip() for b in re.split(r"\n\s*\n", norm_text) if b.strip()]
        substantial = [b for b in blocks if len(b.split()) >= 8]

        opener_hits = 0
        for b in substantial:
            opener_hits += len(_OPENER_RE.findall(b))

        # multiple independent, substantial blocks each carrying their own
        # opener/setup pattern is the clearest fragmentation signature.
        fragmentation_penalty = 0.0
        if len(substantial) >= 2 and opener_hits >= 2:
            fragmentation_penalty = 0.35
        elif len(substantial) >= 3:
            fragmentation_penalty = 0.15

        # mild, length-agnostic flow term: some sentence-length variation is a
        # normal feature of setup->punchline pacing; do not reward or punish
        # sheer length.
        sents = [s for s in re.split(r"[.!?]+", norm_text) if s.strip()]
        try:
            lens = [len(s.split()) for s in sents]
            if len(lens) > 1:
                sd = statistics.pstdev(lens)
                mean_sl = sum(lens) / len(lens)
                cv = sd / mean_sl if mean_sl else 0.0
            else:
                cv = 0.0
        except Exception:
            cv = 0.0
        flow = min(1.0, cv)

        base = 0.70 + 0.15 * flow - fragmentation_penalty
        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5


def _evidence_signal(text, ops):
    try:
        sims = ops.retrieve_similar(text, k=5)
        if not sims:
            return 0.5
        top = max(_sim_value(p) for p in sims)
        # soft rescale: TF-IDF cosine sims over a short-text corpus are
        # typically small; map gently into [0,1] and clip.
        return max(0.0, min(1.0, top * 2.0))
    except Exception:
        return 0.5


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5

        norm = _safe_normalize(text, ops)

        throughline = _throughline_signal(extracted)
        resolution = _resolution_signal(extracted)
        structure = _code_structure_signal(norm)
        evidence = _evidence_signal(norm, ops)

        raw = (
            0.40 * throughline
            + 0.30 * resolution
            + 0.20 * structure
            + 0.10 * evidence
        )
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5

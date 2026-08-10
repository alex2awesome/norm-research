import re

# ---------------------------------------------------------------------------
# a48 "Journal/venue scope and fit" on Math StackExchange Q&A.
#
# The literal criterion (subject scope / genre fit for a target venue) does
# not map cleanly onto SE answers, and the description-compiled baseline
# (keyword lists for "msc/journal/survey/...") scores baseline_train_rho =
# 0.026 -- essentially noise, because those keywords almost never appear in
# SE answers regardless of quality.
#
# Reading the 30 train examples, the judge is overwhelmingly generous
# (25/30 = 1.0) and reserves low scores for answers that fail a much more
# basic "genre fit" test: is this actually a substantive, on-topic piece of
# mathematical exposition, or is it spam/non-answer/tangential filler that
# doesn't belong in this venue at all? Concretely:
#   - a two-sentence non-answer with zero math content -> 0.0
#   - an answer with real LaTeX/equations that nonetheless drifts to an
#     unrelated topic instead of resolving the asked question -> 0.0
#   - a Socratic/partial answer (no math, no resolution, just pushback) -> 0.5
#   - an indirect/abstract reframing that doesn't finish the job -> 0.6
#   - a substantive but hedging/rambling derivation -> 0.65
# The strong majority that engage directly with the question using real
# LaTeX apparatus (equations, proof structure) land at 1.0 regardless of
# length or topic (algebra, analysis, topology, etc. all score equally).
#
# So: code estimates "is this a substantive piece of math exposition"
# (equation density/richness, sentence length, proof-structure markers,
# LaTeX well-formedness) as the base signal -- this replicates almost all
# of the variance (25/30 constant class). The one genuinely hard case is
# an answer that LOOKS mathematical (has equations) but doesn't actually
# resolve the specific question -- that's a THICK judgment (does this
# content match what was asked?) that a regex/parser cannot make from
# surface features alone, so one LLM field grounds it.
# ---------------------------------------------------------------------------

LLM_FIELDS = {
    "answers_question": (
        "In <=8 words: does the answer directly resolve the specific "
        "question asked, or does it drift to an unrelated/speculative "
        "topic? Say 'Yes' or 'No, <why>'."
    ),
}

_ANSWER_SPLIT_RE = re.compile(r"\bAnswer:\s*", re.IGNORECASE)

_OFFTOPIC_MARKERS = (
    "off-topic", "off topic", "tangential", "unrelated", "non-responsive",
    "nonresponsive", "irrelevant", "doesn't answer", "does not answer",
    "avoids the question", "fails to answer", "no direct answer",
    "doesn't address", "does not address", "unrelated topic",
    "speculative", "wrong topic", "different question",
)


def _answer_portion(text):
    parts = _ANSWER_SPLIT_RE.split(text)
    if len(parts) > 1:
        return parts[-1]
    return text


def _safe(fn, default):
    try:
        val = fn()
        return val if val is not None else default
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0

        t = _safe(lambda: ops.normalize(text), text)
        ans = _answer_portion(t)
        words = ans.split()
        n_words = len(words)

        n_display, n_inline, n_numbered, avg_tokens = _safe(
            lambda: ops.equation_stats(ans), (0, 0, 0, 0.0)
        )
        n_display = n_display or 0
        n_inline = n_inline or 0
        avg_tokens = avg_tokens or 0.0

        skeleton = _safe(lambda: ops.proof_skeleton(ans), {}) or {}
        rigor_hits = 0
        for cat in ("theorem", "definition", "proof", "qed", "case", "connective"):
            try:
                rigor_hits += len(skeleton.get(cat, []) or [])
            except Exception:
                pass

        dh = _safe(lambda: ops.delimiter_health(t), {}) or {}
        try:
            bad = sum(abs(v) for v in dh.values() if isinstance(v, (int, float)))
        except Exception:
            bad = 0

        has_math = (n_display + n_inline) > 0

        # Hard floor: short, non-mathematical filler that doesn't engage
        # with the question at all (e.g. "This helped, see example 3.").
        if n_words < 15 and not has_math and rigor_hits == 0:
            return 0.0

        math_presence = min(1.0, (2 * n_display + n_inline) / 6.0)
        richness = min(1.0, avg_tokens / 8.0) if has_math else 0.0
        length_ok = min(1.0, n_words / 40.0)
        rigor_bonus = min(0.15, 0.05 * rigor_hits)
        delim_penalty = min(0.3, 0.07 * bad)

        base = (
            0.35 * math_presence
            + 0.20 * richness
            + 0.30 * length_ok
            + rigor_bonus
            - delim_penalty
        )
        base = max(0.0, min(1.0, base))

        # THICK-input grounding: does the answer actually resolve the
        # question, or is it topically adrift despite surface math content?
        field = ((extracted or {}).get("answers_question") or "").strip().lower()
        if field:
            if field.startswith("no") or any(m in field for m in _OFFTOPIC_MARKERS):
                base = min(base, 0.15)

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

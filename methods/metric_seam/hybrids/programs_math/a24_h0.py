"""
Hybrid metric channel for a24: "Elegance and beauty of proofs"
(brevity+simplicity+inevitability, aesthetic clarity, generality).

Design rationale (see report for detail): the v1_structure baseline (train rho
0.10) fails mainly because ~1/3 of low-judge examples are actually NON-ANSWERS
(off-topic remarks, spam-like non-replies, or answers that solve a different,
easier question than the one asked) which the baseline's length/case-counting
proxy happily scores near 1.0 since they are short. A parser can't tell
"short and on-topic" from "short and refuses to engage" -- that's genuine
thick-input grounding, so it is delegated to the LLM_FIELDS. Everything else
(casework density, equation bloat, raw length) stays in code using the
LaTeX-aware math ops rather than naive keyword counts, since e.g. baseline's
`text.count("case ")` false-fires on words like "showcase".
"""

import re

LLM_FIELDS = {
    "on_topic": (
        "In one word, does this Answer directly attempt to solve the math "
        "question posed above it (yes), or does it refuse, go off-topic, "
        "or dodge the actual problem (no)?"
    ),
    "is_complete": (
        "In one word, does this Answer reach a finished, fully justified "
        "conclusion (complete), or does it stop short as a mere hint, "
        "pointer, or unfinished sketch leaving the reader to finish it (hint)?"
    ),
}

_ANSWER_SPLIT_RE = re.compile(r'(?im)^\s*answer\s*:\s*')


def _split_answer(norm_text: str) -> str:
    """Isolate the answer segment from the 'Question: ... Answer: ...'
    corpus format; falls back to the whole doc if no marker is found."""
    parts = _ANSWER_SPLIT_RE.split(norm_text)
    if len(parts) >= 2:
        answer = parts[-1].strip()
        if len(answer) >= 3:
            return answer
    return norm_text.strip()


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0

        extracted = extracted or {}
        norm = ops.normalize(text)
        answer = _split_answer(norm)

        if len(answer) < 3:
            return 0.05  # nothing to evaluate as a "proof"

        n_chars = len(answer)
        n_words = max(1, len(answer.split()))

        # ---- thick-input gates (semantic judgments a parser can't make) ----
        on_topic_raw = str(extracted.get("on_topic", "") or "").strip().lower()
        complete_raw = str(extracted.get("is_complete", "") or "").strip().lower()
        off_topic = on_topic_raw.startswith("n")
        is_hint = complete_raw.startswith("h")

        # ---- code-side structural signals (math-aware ops, not raw regex) ----
        try:
            n_disp, n_inl, n_num, avg_tok = ops.equation_stats(answer)
        except Exception:
            n_disp, n_inl, n_num, avg_tok = 0, 0, 0, 0.0

        try:
            skeleton = ops.proof_skeleton(answer)
            case_hits = len(skeleton.get("case", []))
        except Exception:
            case_hits = 0

        # minor noise penalty for badly garbled markup (scrape artifacts),
        # kept small since malformation isn't itself an elegance judgment
        noise_mult = 1.0
        try:
            dh = ops.delimiter_health(answer)
            bad = sum(v for k, v in dh.items() if isinstance(v, (int, float)))
            if bad > 6:
                noise_mult = max(0.85, 1.0 - 0.01 * (bad - 6))
        except Exception:
            pass

        # length: reward concision, but don't crush moderate-length answers
        # that use one worked derivation (only truly sprawling ones get hit)
        if n_chars <= 400:
            len_score = 1.0
        elif n_chars <= 1500:
            len_score = 1.0 - 0.30 * (n_chars - 400) / 1100.0
        else:
            len_score = max(0.20, 0.70 - 0.35 * (n_chars - 1500) / 3000.0)

        # casework fights "inevitability"/generality: penalize by density,
        # not raw count, so long-but-single-case answers aren't punished
        case_density = case_hits / (n_words / 100.0)
        case_score = max(0.0, 1.0 - 0.30 * case_density)

        # equation bloat: judge by *total* symbolic mass (avg tokens per span
        # times number of spans), not avg-per-span alone -- a single dense
        # equality chain in one display block (common in a terse, elegant
        # one-liner) has few spans and shouldn't be confused with several
        # independently-huge CAS-derived formulas, which is what actually
        # reads as brute force rather than insight
        total_tokens = avg_tok * max(1, n_disp + n_inl)
        if total_tokens <= 100:
            eq_score = 1.0
        elif total_tokens <= 400:
            eq_score = 1.0 - 0.85 * (total_tokens - 100) / 300.0
        else:
            eq_score = max(0.10, 0.15 - 0.05 * (total_tokens - 400) / 400.0)

        # multiplicative, not additive: elegance needs concision AND minimal
        # casework AND clean equations *together* -- a single badly-failed
        # axis (e.g. a "few pages long" CAS formula) should sink the score
        # even if the other axes look fine, which an additive blend would
        # dilute away
        structural = len_score * case_score * eq_score
        structural = min(1.0, max(0.0, structural)) * noise_mult

        if off_topic:
            # doesn't engage with the question at all -> no proof exists to
            # be elegant; keep a thin, structural-scaled tail instead of a
            # hard constant so within-bucket ordering isn't totally flat
            return float(max(0.0, min(0.12, 0.03 + 0.07 * structural)))

        result = structural
        if is_hint:
            result *= 0.6

        return float(min(1.0, max(0.0, result)))
    except Exception:
        return 0.5

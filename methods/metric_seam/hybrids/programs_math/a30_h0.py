"""
Hybrid metric channel for a30: "Self-containment for the intended audience"
(Math StackExchange answers).

Rationale: the judge does not reward bare keyword presence (baseline_train_rho
was only 0.17) -- it rewards answers that GROUND new notation right where it is
introduced (a "Let $I,J$ be ideals..." beside real math, not a stray mention of
the word "definition"), that reason with explicit connectives/case-splits, and
that do NOT lean on unexplained links/tools or on jargon the question itself
never used. Long unnarrated symbol-manipulation ("computation dumps") and bare
keyword mentions are both penalized rather than rewarded.
"""

import re

LLM_FIELDS = {
    "undefined_jargon": (
        "List, comma-separated, <=20 words: any named theorem/object/technique "
        "the ANSWER uses without defining or glossing it, that the QUESTION's own "
        "wording suggests the asker does not already know. Else answer NONE."
    ),
    "unexplained_pointer": (
        "In <=15 words: does the answer's key explanation depend on a link, "
        "image, other post, or external tool WITHOUT restating the needed "
        "content inline? Name it. Else answer NONE."
    ),
}

_MATH_MARK_RE = re.compile(r"\$|\\\(|\\\[|\\begin\{")
_ANSWER_RE = re.compile(r"\banswer\s*:", re.I)
_EMPTY_VALS = {"", "none", "n/a", "no", "nothing"}


def _answer_segment(text):
    """Corpus docs are typically "Question: ... \n\nAnswer: ...". Structural
    signals must be read off the ANSWER (what is actually being judged for
    self-containment), not off the asker's own question -- otherwise a
    question that says "the definition of X is..." falsely credits the
    answer. Falls back to the full text if no "Answer:" marker is found."""
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end():]


def _is_empty(v):
    if not v:
        return True
    return v.strip().strip(".").lower() in _EMPTY_VALS


def _count_items(v):
    if _is_empty(v):
        return 0
    parts = re.split(r"[,;]|\band\b", v)
    return len([p for p in parts if p.strip()])


def _grounded_hits(hits, math_offsets, window=90):
    """Count (offset, match) proof-skeleton hits that sit near a math marker,
    to separate real in-line definitions ("Let $X$ be...") from bare mentions
    of the word "definition" in ordinary prose (the baseline's failure mode)."""
    if not hits or not math_offsets:
        return 0
    n = 0
    for off, _match in hits:
        if any(abs(off - m) <= window for m in math_offsets):
            n += 1
    return n


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = ops.normalize(text)
        a = _answer_segment(t)
        if not a.strip():
            a = t

        n_sent, mean_wps, frac_long = ops.sent_stats(a)
        n_sent = max(int(n_sent) if n_sent else 1, 1)

        skeleton = ops.proof_skeleton(a) or {}
        definition_hits = skeleton.get("definition", []) or []
        case_hits = skeleton.get("case", []) or []
        connective_hits = skeleton.get("connective", []) or []
        formal_hits = (
            len(skeleton.get("theorem", []) or [])
            + len(skeleton.get("proof", []) or [])
            + len(skeleton.get("qed", []) or [])
        )

        math_offsets = [m.start() for m in _MATH_MARK_RE.finditer(a)]

        grounded_defs = _grounded_hits(definition_hits, math_offsets)
        n_case = len(case_hits)
        n_connective = len(connective_hits)

        scaffold_hits = grounded_defs + n_case + n_connective
        scaffold_density = scaffold_hits / n_sent

        n_disp, n_inline, n_numbered, avg_tok = ops.equation_stats(a)
        n_spans = (n_disp or 0) + (n_inline or 0)
        math_load = n_spans / n_sent

        dump_ratio = 0.0
        if n_spans > 0:
            dump_ratio = math_load / (scaffold_density + 0.2)

        dh = ops.delimiter_health(a) or {}
        markup_problems = sum(v for v in dh.values() if isinstance(v, (int, float)))

        base = 0.30
        base += 0.45 * (scaffold_density / (scaffold_density + 0.5))
        base += 0.05 if formal_hits > 0 else 0.0
        base -= 0.15 * min(dump_ratio / 4.0, 1.0)
        base -= min(0.03 * markup_problems, 0.10)
        base -= 0.04 * min(frac_long or 0.0, 1.0)

        jargon_n = _count_items(extracted.get("undefined_jargon", ""))
        base -= 0.10 * min(jargon_n, 3)

        if not _is_empty(extracted.get("unexplained_pointer", "")):
            base -= 0.15

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5

r"""
Hybrid metric channel for a210: Mathematical notation and typography correctness in LaTeX.

Reading the 30 train examples: judge score is NOT a function of LaTeX density (the
keyword-density baseline gets baseline_train_rho = -0.1 -- actively backwards). Dense-but-
sloppy answers score LOW: d02378 (0.25) mixes bare "ln x"/"lnx" with proper "\ln" elsewhere;
d02176 (0.3) has "\mathbb{R_+}" (subscript trapped inside the mathbb argument) plus dozens of
ad hoc "\ + \ ... \ = \ " manual-spacing hacks; d04471 (0.3) writes "$r1$" then later the same
quantity as "$r_1$" (inconsistent/missing subscript); d04848 (0.3) mixes literal unicode
"exists"/"forall" symbols with "\exists"/"\forall" in the same post; d01747 (0.3) has a string
of adjacent bare inline formulas "$\forall$ $g$ $\in$ $G$" with no separating prose. Meanwhile
several SPARSE, plain-prose answers with almost no markup at all (d00115, d01866, d00557,
d03541, d03563) score in the 0.65-1.0 tier purely because what little notation is there is
correct and consistent. So: score by ABSENCE of parseable defects that map onto the criterion's
own checklist (symbol/alphabet choice, delimiter sizing, spacing, italic/roman convention,
index clarity, deprecated constructs), not by presence/density of macros.

The pack's high tier splits further into 0.65 vs 0.95-1.0, but I could not find any
LaTeX-parseable feature that reliably separates those two sub-tiers (both contain clean,
well-nested `align`-environment proofs and both contain minimal-markup prose; judge_reliability
is only 0.639) -- forcing a 3-way split off 30 points risks fitting noise, so the model only
commits to a coarse "flawed vs clean" separation plus a small monotone penalty scale within the
flawed band, and lets ties sit inside the clean band rather than guessing a spurious tiebreak.
"""
import re

LLM_FIELDS = {
    "notation_issue": (
        "Name ONE specific LaTeX/math notation error, inconsistency, or symbol/subscript "
        "misuse in this answer's math, in <=15 words; say NONE if notation is correct/consistent."
    ),
}

_FUNC_NAMES = [
    "sin", "cos", "tan", "cot", "sec", "csc",
    "ln", "log", "exp", "min", "max", "sup", "inf",
    "det", "dim", "ker", "gcd", "lcm", "arg",
]

_UNICODE_CMD_PAIRS = [
    ("∃", r"\\exists"),
    ("∀", r"\\forall"),
    ("∈", r"\\in\b"),
    ("∪", r"\\cup\b"),
    ("∩", r"\\cap\b"),
    ("≤", r"\\leq\b"),
    ("≥", r"\\geq\b"),
]


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        spans = _safe(lambda: ops.extract_math_spans(text), [])
        if not isinstance(spans, list):
            spans = []
        span_texts = []
        for item in spans:
            try:
                span_texts.append(item[1] if len(item) > 1 else "")
            except Exception:
                continue

        if not span_texts:
            # No parseable math content -> criterion is largely moot; stay neutral.
            return 0.5

        n_spans = max(len(span_texts), 1)

        # 1. Parse/delimiter health: the op does the hard parsing (odd $ parity, unclosed
        #    openers, brace imbalance, \left/\right mismatch, \begin/\end mismatch, naked
        #    commands outside math, adjacent formulas w/ no separating words). Rate-normalize
        #    by span count so long documents aren't punished more than short ones for the
        #    same defect density.
        dh = _safe(lambda: ops.delimiter_health(text), {})
        dh_total = 0
        if isinstance(dh, dict):
            dh_total = sum(v for v in dh.values() if isinstance(v, (int, float)))
        dh_rate = dh_total / n_spans
        dh_penalty = min(0.35, dh_rate * 0.5) + (0.05 if dh_total > 0 else 0.0)

        # 2. Deprecated constructs the criterion names explicitly (eqnarray; old font-switch
        #    declarations \bf/\rm/\it/\tt/\sc used instead of \mathbf{}/\mathrm{}/...).
        deprecated_penalty = 0.0
        if re.search(r"\\begin\{eqnarray\*?\}", text):
            deprecated_penalty += 0.2
        if re.search(r"\\(bf|rm|it|tt|sc)(?![a-zA-Z])", text):
            deprecated_penalty += 0.1
        deprecated_penalty = min(0.3, deprecated_penalty)

        # 3. Bare multi-letter function names inside math mode ("ln x"/"lnx"/"gcd(p,k)"
        #    instead of "\ln x"/"\gcd(p,k)"): each letter italicizes separately and reads as
        #    a product of variables -- the criterion's "italics/roman conventions" clause.
        bare_hits = 0
        for c in span_texts:
            for fn_name in _FUNC_NAMES:
                bare_hits += len(re.findall(r"(?<!\\)\b" + fn_name + r"\b", c))
        bare_penalty = min(0.3, 0.1 * bare_hits)

        # 4. Malformed bare subscripts: a letter directly glued to a digit with nothing (no
        #    "_" or "^") between them, e.g. "r1" instead of "r_1" -- "clear indices/subscripts".
        bad_sub = 0
        for c in span_texts:
            bad_sub += len(re.findall(r"[A-Za-z](\d+)", c))
        subscript_penalty = min(0.2, 0.05 * bad_sub)

        # 5. Repeated manual "\ " spacing hacks instead of relying on proper macros/automatic
        #    math-mode spacing -- the "spacing" clause.
        spacing_hits = 0
        for c in span_texts:
            spacing_hits += len(re.findall(r"\\ ", c))
        spacing_penalty = min(0.2, 0.03 * spacing_hits) if spacing_hits >= 4 else 0.0

        # 6. Mixing a unicode math/logic symbol with its backslash-command equivalent in the
        #    same document (literal "exists"/"forall" glyphs alongside "\exists"/"\forall")
        #    -- directly violates "Consistent ... notation".
        mixed = 0
        for uni_char, cmd_pat in _UNICODE_CMD_PAIRS:
            if uni_char in text and re.search(cmd_pat, text):
                mixed += 1
        mixed_penalty = min(0.2, 0.1 * mixed)

        # 7. Sub/superscript trapped *inside* a font-alphabet command's argument, e.g.
        #    "\mathbb{R_+}" instead of "\mathbb{R}_+" -- "appropriate symbol/operator
        #    commands and alphabets".
        font_hits = len(re.findall(r"\\math(?:bb|cal|rm|frak|it)\{[^{}]*[_^][^{}]*\}", text))
        font_penalty = min(0.15, 0.1 * font_hits)

        code_penalty = (
            dh_penalty + deprecated_penalty + bare_penalty
            + subscript_penalty + spacing_penalty + mixed_penalty + font_penalty
        )

        # LLM-grounded tacit signal: semantic misuse a parser can't see (wrong symbol for the
        # concept, ambiguously overloaded subscript, near-miss symbol confusion). Kept as a
        # modest secondary nudge, not the primary driver, since a Gemma-31B extractor primed
        # to "name an issue" skews toward over-flagging trivia.
        note = ""
        if isinstance(extracted, dict):
            note = (extracted.get("notation_issue") or "").strip()
        llm_penalty = 0.08 if note else 0.0

        total_penalty = code_penalty + llm_penalty

        bonus = 0.0
        if code_penalty == 0.0:
            bonus += 0.15
            if not note:
                bonus += 0.04

        s = 0.55 + bonus - total_penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5

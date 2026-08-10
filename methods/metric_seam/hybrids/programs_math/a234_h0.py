"""
Hybrid metric channel for a234: "Prefer mathematical notation over programming-style
shortcuts" (Math StackExchange answers).

Design rationale (see train-residual analysis in the improver pack):
  - The description-compiled baseline (train rho=0.098) counts raw occurrences of a fixed
    "good operator" vocabulary (\\cdot, \\times, \\langle, ...) over the WHOLE document. That is
    a presence-proxy, not a quality signal: many judge_score=1.0 examples use almost none of
    those specific commands (clean but plain notation), while some judge_score<=0.5 examples
    use them heavily yet still get docked for other issues. So: default score is HIGH (1.0),
    and we only subtract for concretely-detectable violations, rather than reward keyword
    density.
  - The baseline's own '*'-multiplication check uses a naive `\\$[^$]*\\$` regex that cannot
    see inside `$$...$$` display blocks or `\\[...\\]`, and it never looks at plain-prose ASCII
    arithmetic outside any math delimiters at all (a common failure mode: an OP's raw
    calculator trace like "y = -0.019735186 * 100 + ..."; "10^2.689239229"). We use the
    LaTeX-aware `extract_math_spans` to see inside every span kind, and a separate
    "non-mathy line" scan to catch bare ASCII arithmetic outside any markup.
  - A known false-positive trap: '*' is *legitimate* standard notation for free products /
    convolution / general binary operations (e.g. "$\\pi_1(U) * \\pi_1(V)$", "$G * H$") when the
    operands are structured objects (subscripted names, parenthesized expressions). We only
    flag '*' between bare single alnum tokens inside math (the "a*b", "2*x" arithmetic smell),
    not '*' immediately after a closing paren/brace (the free-product smell), which keeps the
    predicate class-general instead of overfit to one example.
  - Some violations are about a mismatch between "what the notation communicates" and "how a
    programmer would write it" (e.g. one train doc renders a two-point distance as
    "||(x,y,z),(u,v,w)||" -- a function-call-with-tuple-args shape wearing valid LaTeX bars).
    A regex predicate can't reliably see this, so one OPTIONAL LLM field does a single
    holistic "does this look like a function call / code-style construct" check, kept at a
    modest fixed weight so a bad extraction can't dominate the score.
  - Corpus items are templated as "Question: ...\n\nAnswer: ..." (verbatim marker present in
    every train example), and residuals show the judge cares much more about ASCII shortcuts
    written by the ANSWERER than ones merely quoted inside the OP's question: a doc whose only
    violations sit in the quoted question but whose actual answer is clean lands around
    judge~0.65, while a doc with the *same* kind of violation inside the answer itself lands
    much lower (~0.5) even with fewer occurrences. So violations are weighted full-strength in
    the answer zone (text after the marker) and down-weighted (not zeroed -- the question still
    leaks into overall notation quality) before it.

LLM_FIELDS: OPTIONAL, used only for the one class of tacit violation code can't parse.
"""
import re

_ANSWER_MARKER = re.compile(r"\n\s*Answer\s*:", re.IGNORECASE)
_ANSWER_ZONE_WEIGHT = 1.0
_QUESTION_ZONE_WEIGHT = 0.4

LLM_FIELDS = {
    "prog_style_flag": (
        "In the ANSWER's math, is any expression written in a programming/function-call "
        "style rather than standard math notation (e.g. norm(a,b) or dist(p,q) style calls, "
        "two comma-separated tuples crammed inside one bracket/bar pair, chained "
        "code-like assignment)? Quote the short offending phrase, else say NONE."
    ),
}

_ASCII_CMP = re.compile(r"!=|<=|>=")
_STRICT_STAR_MATH = re.compile(r"[A-Za-z0-9]\s*\*\s*[A-Za-z0-9(\\]")
_PROSE_STAR = re.compile(r"[A-Za-z0-9)]\s*\*\s*[A-Za-z0-9(]")
_CARET_DIGIT = re.compile(r"\d\s*\^\s*[\d.]")
_X_MULT = re.compile(r"\b\d+\s*[xX]\s*\d+\b")
_BARE_FN = re.compile(
    r"(?<!\\)\b(?:sin|cos|tan|log|ln|exp|lim|max|min|sup|inf|det|gcd)\b"
)
_MATHY_LINE = re.compile(r"\$|\\\[|\\\]|\\\(|\\\)|\\begin|\\end")


def _weighted(count_q, count_a, cap):
    wc = count_a * _ANSWER_ZONE_WEIGHT + count_q * _QUESTION_ZONE_WEIGHT
    return wc if wc < cap else cap


def _span_contents(zone_text, ops):
    try:
        spans = ops.extract_math_spans(zone_text) or []
    except Exception:
        spans = []
    out = []
    for item in spans:
        try:
            _kind, content = item
            if content:
                out.append(content)
        except Exception:
            continue
    return out


def _prose_counts(zone_text):
    star_n = caret_n = 0
    for line in zone_text.splitlines():
        if _MATHY_LINE.search(line):
            continue
        star_n += len(_PROSE_STAR.findall(line))
        caret_n += len(_CARET_DIGIT.findall(line))
    return star_n, caret_n


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if text else ""
        if not t or not t.strip():
            return 0.5

        m = _ANSWER_MARKER.search(t)
        if m:
            q_zone, a_zone = t[: m.start()], t[m.start() :]
        else:
            q_zone, a_zone = "", t

        q_spans = _span_contents(q_zone, ops)
        a_spans = _span_contents(a_zone, ops)

        penalty = 0.0

        # 1. ASCII comparison/logic operators -- never standard in math prose.
        ascii_cmp_q = len(_ASCII_CMP.findall(q_zone))
        ascii_cmp_a = len(_ASCII_CMP.findall(a_zone))
        penalty += _weighted(ascii_cmp_q, ascii_cmp_a, 3) * 0.20

        # 2. ASCII '*' multiplication between bare alnum tokens INSIDE recognized math spans
        #    (excludes '*' preceded by ')'/'}' to spare legitimate free-product/binary-op use).
        star_math_q = sum(len(_STRICT_STAR_MATH.findall(c)) for c in q_spans)
        star_math_a = sum(len(_STRICT_STAR_MATH.findall(c)) for c in a_spans)
        penalty += _weighted(star_math_q, star_math_a, 3) * 0.20

        # 3. Plain-prose ASCII arithmetic on lines with no math markup at all (raw calculator
        #    trace: '*' multiplication and 'digit^digit' exponent shorthand).
        star_prose_q, caret_prose_q = _prose_counts(q_zone)
        star_prose_a, caret_prose_a = _prose_counts(a_zone)
        penalty += _weighted(star_prose_q, star_prose_a, 3) * 0.15
        penalty += _weighted(caret_prose_q, caret_prose_a, 2) * 0.15

        # 4. ASCII 'x' as multiplication between numbers (e.g. "3 x 4" instead of \times).
        x_mult_q = len(_X_MULT.findall(q_zone))
        x_mult_a = len(_X_MULT.findall(a_zone))
        penalty += _weighted(x_mult_q, x_mult_a, 2) * 0.15

        # 5. Bare function names inside math spans that should be LaTeX commands (sin vs \sin).
        bare_fn_q = sum(len(_BARE_FN.findall(c)) for c in q_spans)
        bare_fn_a = sum(len(_BARE_FN.findall(c)) for c in a_spans)
        penalty += _weighted(bare_fn_q, bare_fn_a, 3) * 0.10

        # 6. Semicolons used as ASCII statement separators inside a single math span
        #    (e.g. "$x \\ne 0; y \\ne 0; z \\ne 0$" reads like a code line, not math notation).
        semi_q = sum(c.count(";") for c in q_spans)
        semi_a = sum(c.count(";") for c in a_spans)
        penalty += _weighted(semi_q, semi_a, 3) * 0.12

        # 7. LLM-grounded tacit check for function-call-shaped notation regex can't see
        #    (already scoped to "the ANSWER's math" in the extraction instruction).
        flag = ""
        try:
            flag = (extracted.get("prog_style_flag") or "").strip()
        except Exception:
            flag = ""
        if flag and flag.upper() not in ("NONE", "N/A", "NA"):
            penalty += 0.25

        penalty = min(0.95, penalty)
        return max(0.05, min(1.0, 1.0 - penalty))
    except Exception:
        return 0.5

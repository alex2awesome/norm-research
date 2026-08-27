#!/usr/bin/env python3
"""LLM extraction prompt + output validation for Math.SE claim verification.

Defines:
  SYSTEM_PROMPT                  full system prompt (schema doc + rules + few-shots)
  FEW_SHOT_EXAMPLES              few-shots selected from the pilot's claims.jsonl
  build_user_prompt(q, a)        the per-answer user message
  validate_extraction(raw, source_text=None) -> (claims, errors)

The pilot (../verification_pilot/REPORT.md) found that *silent
mis-formalization* is the dominant risk of LLM extraction, so the prompt
hard-requires per-claim round-trip fidelity fields:
  source_quote   verbatim substring of the answer the claim came from
  fidelity_note  what was changed when formalizing (typo repair,
                 instantiation of quantified statements, notation choices)
validate_extraction() checks that source_quote actually occurs in the answer
(whitespace-normalized) and attaches a `fidelity` dict to each claim; the
harness passes these fields through into every verdict record.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (CLAIM_TYPES, DEFAULT_RELATION,  # noqa: E402
                     validate_claim_schema)

# ------------------------------------------------------------------ schema doc
SCHEMA_DOC = r"""You extract mechanically checkable mathematical claims from Math StackExchange answers and formalize them as sympy expressions. A separate program will verify each claim with sympy (symbolic simplification, then numeric sampling), so every expression must be valid sympy syntax.

OUTPUT FORMAT — respond with EXACTLY one JSON object, no markdown fences, no prose:

{"claims": [ <claim>, <claim>, ... ]}

Each <claim> is an object with these fields:

  "claim_type":   one of EQUALITY | NUMERIC_VALUE | LIMIT | DERIVATIVE |
                  INTEGRAL | SUM | INEQUALITY | BOOLEAN_EQUIV | MODULAR |
                  DIVISIBILITY | COMBINATORIAL_COUNT | MATRIX |
                  RECURRENCE_CHECK | NONE
  "sympy_lhs":    left-hand side as a sympy expression string
  "sympy_rhs":    right-hand side (null only for relation is_finite/is_infinite)
  "relation":     "==" | "!=" (most types); "<=" | "<" | ">=" | ">" (INEQUALITY);
                  "equiv" (BOOLEAN_EQUIV); "is_finite" | "is_infinite"
                  (convergence claims, sympy_rhs = null); "divides" |
                  "not_divides" (DIVISIBILITY); "satisfies" (RECURRENCE_CHECK)
  "free_symbols": list of symbol declarations. "x" declares a real symbol;
                  add kinds after a colon: "n:integer,positive",
                  "k:integer", "m:integer,nonnegative", "x:positive",
                  "c:nonzero", "t:negative", "F:function" (undefined function).
                  Allowed kinds: integer, positive, nonnegative, nonzero,
                  negative, real, function. Declare EVERY symbol you use,
                  including summation/integration bound variables.
  "assumptions":  list of constraint strings. Equality constraints like
                  "p+q+r=0" are substituted into both sides; relational
                  constraints like "x>0", "x!=1", "x<=4" restrict numeric
                  sampling to admissible points.
  "modulus":      (MODULAR only) the modulus, e.g. "7" or "p"
  "recurrence":   (RECURRENCE_CHECK only) the recurrence written with an
                  undefined function a(.), e.g. "a(n+1) = a(n)**2 - 2";
                  sympy_lhs holds the claimed closed form in n
  "rec_var":      (RECURRENCE_CHECK, optional) recurrence index, default "n"
  "load_bearing": true if this claim IS the answer's main result (the thing
                  the answer exists to establish); false for side
                  computations, sanity checks, and intermediate steps
  "source_quote": VERBATIM substring of the ANSWER text (copy the exact
                  characters, LaTeX included) where this claim is stated
  "fidelity_note": what you changed when formalizing this quote. Mention
                  every repair/choice: typo fixes, notation disambiguation,
                  instantiating a quantified parameter at specific values,
                  encoding an unknown function as F:function, choosing a
                  one-sided limit direction. If the formalization is a
                  faithful 1:1 transcription, write "faithful transcription".

CLAIM TYPE NOTES:
  EQUALITY / NUMERIC_VALUE / DERIVATIVE / LIMIT / SUM / INTEGRAL: claim that
    lhs == rhs (or != rhs). Use Integral(f, (x, a, b)), Sum(f, (k, a, b)),
    Limit(f, x, x0, '+') (one-sided!), diff(f, x), Derivative(F(x), x).
    Convergence/divergence claims: relation "is_finite"/"is_infinite" with
    sympy_rhs null (e.g. an improper Integral converges).
  INEQUALITY: lhs <op> rhs at every admissible point.
  BOOLEAN_EQUIV: propositional equivalence, e.g. Not(Or(l, r)) equiv
    And(Not(l), Not(r)).
  MODULAR: lhs == rhs (mod modulus). Example: "2**100 == 6 (mod 10)" gives
    sympy_lhs "2**100", sympy_rhs "6", modulus "10", relation "==".
  DIVISIBILITY: sympy_lhs = the DIVISOR, sympy_rhs = the DIVIDEND, relation
    "divides". Example: "6 divides n**3 - n" gives sympy_lhs "6",
    sympy_rhs "n**3 - n", free_symbols ["n:integer"].
  COMBINATORIAL_COUNT: a counting expression equals a concrete integer,
    checked with exact arithmetic. Example: "the number of ways is 120"
    where the answer computes binomial(10, 3)*2 gives sympy_lhs
    "binomial(10, 3)*2", sympy_rhs "240". No free symbols allowed.
  MATRIX: identity between small CONCRETE matrices (entries may be symbolic),
    e.g. sympy_lhs "Matrix([[1, 2], [3, 4]])**2", sympy_rhs
    "Matrix([[7, 10], [15, 22]])". Scalar matrix facts (determinants, traces)
    are EQUALITY claims using det(Matrix([...])) etc.
  RECURRENCE_CHECK: a claimed closed form satisfies a recurrence. Example:
    "a_{n+1} = a_n**2 - 2 with a_n = 2**(2**n) + 2**(-2**n)" gives sympy_lhs
    "2**(2**n) + 2**(-(2**n))", recurrence "a(n+1) = a(n)**2 - 2",
    free_symbols ["n:integer,positive"].
  NONE: the answer contains no claim a CAS can check (pure prose, abstract
    proofs, topology/category theory, references). Emit a single claim
    {"claim_type": "NONE", "reason": "<why>"} and nothing else.

EXTRACTION RULES:
1. Extract up to 5 of the most checkable, most load-bearing claims per
   answer. Prefer the answer's headline result, then key intermediate steps.
2. Claims must come from the ANSWER (the question is context only). A claim
   stated in the question but endorsed/proved by the answer counts; quote
   the answer's endorsement.
3. Quantified claims ("for all p > 1 ...") cannot be checked directly:
   instantiate at 1-3 concrete parameter values as SEPARATE claims and say
   so in fidelity_note.
4. Use exact sympy values: Rational(1, 2) not 0.5, pi, oo, log(x), sqrt(2),
   exp(x), binomial(n, k), factorial(n), harmonic(n), Catalan, EulerGamma.
5. Repair obvious typos in the source (and document the repair in
   fidelity_note). Do NOT silently change the mathematical content.
6. Declare assumptions the answer relies on (domains, positivity,
   integrality) — an identity that only holds for x > 0 must carry "x>0".
7. If a step needs an undefined function (a cdf F, an arbitrary f), declare
   it as "F:function" and write Derivative(F(x), x) for its derivative.
8. source_quote must be copied character-for-character from the answer text,
   long enough to be unambiguous (>= 20 chars when possible).
"""

# ------------------------------------------------------------ few-shot examples
# Selected from ../verification_pilot/claims.jsonl (claim_ids noted), with
# source_quote tightened to verbatim substrings of the pilot answers in
# sample30.jsonl. Inputs are excerpts of the actual pilot rows.
FEW_SHOT_EXAMPLES = [
    # --- pilot row 0: claims 0 + 2 (INTEGRAL with named constant; EQUALITY with typo repair)
    {
        "question": r"Find the value of: $$\int_0^{+\infty} \dfrac{\ln(x+1)}{x^2+1} \ \mathrm dx$$ I tried using substitution but it doesn't work.",
        "answer": r"A trig substitution will do. Letting $x = \tan \theta$, the integral becomes $\int_0^{\pi/2} \ln(\tan \theta + 1)\, d\theta$, which is the same as $$\int_0^{\pi/2} \ln(\sin \theta + \cos \theta)\, d\theta - \int_0^{\pi/2}\ln(\cos \theta)\, d\theta$$ Since $\sin(\theta + \cos \theta) = \sqrt{2}\sin(\theta + \pi/4)$, [...] The latter integral equals $\int_0^{\pi/4}\tan(\pi/2 - \theta)\, d\theta = \int_0^{\pi/4}\ln \cot \theta = C$, Catalan's constant. Hence, the integral evaluates to $$\frac{\pi}{4}\ln 2 + C$$",
        "output": {"claims": [
            {"claim_type": "INTEGRAL",
             "sympy_lhs": "Integral(log(x+1)/(x**2+1), (x, 0, oo))",
             "sympy_rhs": "pi*log(2)/4 + Catalan",
             "relation": "==", "free_symbols": ["x"], "assumptions": [],
             "load_bearing": True,
             "source_quote": r"Hence, the integral evaluates to $$\frac{\pi}{4}\ln 2 + C$$",
             "fidelity_note": "Integrand and limits taken from the question (the answer evaluates that integral); 'C' resolved to Catalan because the answer says 'C$, Catalan's constant'."},
            {"claim_type": "EQUALITY",
             "sympy_lhs": "sin(t) + cos(t)",
             "sympy_rhs": "sqrt(2)*sin(t + pi/4)",
             "relation": "==", "free_symbols": ["t"], "assumptions": [],
             "load_bearing": False,
             "source_quote": r"Since $\sin(\theta + \cos \theta) = \sqrt{2}\sin(\theta + \pi/4)$",
             "fidelity_note": "Source has a LaTeX typo 'sin(theta + cos theta)'; formalized the clearly intended identity sin(theta) + cos(theta), with theta renamed t."},
        ]},
    },
    # --- pilot row 3: claims 8 + 9 (EQUALITY with det(Matrix); equality-constraint assumption)
    {
        "question": r"If $p+q+r=0$, prove that $\Delta= \begin{vmatrix} pa & qb &rc \\ qc & ra &pb\\ rb& pc & qa \\ \end{vmatrix} =-pqr \begin{vmatrix} a & b &c \\ b & c &a\\ c& a & b \\ \end{vmatrix}$",
        "answer": r"By Sarrus' rule, you have $$\begin{vmatrix}pa&qb&rc\\qc&ra&pb\\rb&pc&qa\end{vmatrix}=prq(a^3+b^3+c^3)-abc(p^3+q^3+r^3).$$ But \begin{align*}p^3+q^3+r^3&=\overbrace{(p+q+r)^3}^{\phantom{0}=0}-3(p q+r q+p r) (\overbrace{p+q+r}^{\phantom{0}=0})+3 p q r\\&=3pqr\end{align*} and therefore your determinant is equal to \begin{align*}pqr(a^3+b^3+c^3-3abc)&=-pqr(3abc-a^3-b^3-c^3)\\&=-pqr\begin{vmatrix}a&b&c\\b&c&a\\c&a&b\end{vmatrix},\end{align*} again by Sarrus' rule.",
        "output": {"claims": [
            {"claim_type": "EQUALITY",
             "sympy_lhs": "det(Matrix([[p*a, q*b, r*c], [q*c, r*a, p*b], [r*b, p*c, q*a]]))",
             "sympy_rhs": "p*q*r*(a**3 + b**3 + c**3) - a*b*c*(p**3 + q**3 + r**3)",
             "relation": "==",
             "free_symbols": ["a", "b", "c", "p", "q", "r"], "assumptions": [],
             "load_bearing": False,
             "source_quote": r"By Sarrus' rule, you have $$\begin{vmatrix}pa&qb&rc\\qc&ra&pb\\rb&pc&qa\end{vmatrix}=prq(a^3+b^3+c^3)-abc(p^3+q^3+r^3).$$",
             "fidelity_note": "3x3 determinant transcribed as det(Matrix(...)); source writes 'prq' which is the same monomial as p*q*r."},
            {"claim_type": "EQUALITY",
             "sympy_lhs": "p**3 + q**3 + r**3",
             "sympy_rhs": "3*p*q*r",
             "relation": "==", "free_symbols": ["p", "q", "r"],
             "assumptions": ["p+q+r=0"],
             "load_bearing": False,
             "source_quote": r"But \begin{align*}p^3+q^3+r^3&=\overbrace{(p+q+r)^3}^{\phantom{0}=0}-3(p q+r q+p r) (\overbrace{p+q+r}^{\phantom{0}=0})+3 p q r\\&=3pqr\end{align*}",
             "fidelity_note": "Faithful transcription; the hypothesis p+q+r=0 from the question is encoded as an equality assumption."},
        ]},
    },
    # --- pilot row 13: claims 34 + 35 (positivity assumptions like x>0; DERIVATIVE)
    {
        "question": r"Derivative of $y = \log_{\sqrt[3]{x}}(7)$. Is $\frac{dy}{dx}= \dfrac{-3\ln(7)}{x(\ln(x))^2}$ correct?",
        "answer": r"Yes, you are right. Simplify as follows $$y=\log_{\sqrt[3]{x}}(7)=\frac{\ln 7}{\ln (\sqrt[3]{x})}=\frac{\ln (7)}{\frac13\ln x}=\frac{3\ln (7)}{\ln x}$$ $$\therefore \frac{dy}{dx}=3\ln (7)\left(\frac{-1}{(\ln x)^2}\frac1x\right)=-\frac{3\ln (7)}{x(\ln x)^2}$$",
        "output": {"claims": [
            {"claim_type": "EQUALITY",
             "sympy_lhs": "log(7)/log(x**Rational(1,3))",
             "sympy_rhs": "3*log(7)/log(x)",
             "relation": "==", "free_symbols": ["x:positive"],
             "assumptions": ["x!=1"],
             "load_bearing": False,
             "source_quote": r"$$y=\log_{\sqrt[3]{x}}(7)=\frac{\ln 7}{\ln (\sqrt[3]{x})}=\frac{\ln (7)}{\frac13\ln x}=\frac{3\ln (7)}{\ln x}$$",
             "fidelity_note": "log base sqrt[3]{x} rewritten via change of base, as the answer does. Domain of the original expression requires x>0, x!=1; added as symbol kind and assumption."},
            {"claim_type": "DERIVATIVE",
             "sympy_lhs": "diff(3*log(7)/log(x), x)",
             "sympy_rhs": "-3*log(7)/(x*log(x)**2)",
             "relation": "==", "free_symbols": ["x:positive"],
             "assumptions": ["x!=1"],
             "load_bearing": True,
             "source_quote": r"$$\therefore \frac{dy}{dx}=3\ln (7)\left(\frac{-1}{(\ln x)^2}\frac1x\right)=-\frac{3\ln (7)}{x(\ln x)^2}$$",
             "fidelity_note": "Faithful transcription of the derivative claim (the answer's main result, endorsing the asker's second form)."},
        ]},
    },
    # --- pilot row 14: claims 37 + 38 (one-sided LIMIT direction choice)
    {
        "question": r"We have function $y = f(x) = |2x + 1|$ ... What happens at $x = -\frac12$?",
        "answer": r"Note that for the limit of a difference quotient to exist, you must have $$\lim_{x\rightarrow x_0^{\color{red}{-}}}\frac{f(x)-f(x_0)}{x-x_0}=\lim_{x\rightarrow x_0^{\color{red}{+}}}\frac{f(x)-f(x_0)}{x-x_0}$$ You do not have that. You have $$\lim_{x\rightarrow x_0^{\color{red}{-}}}\frac{f(x)-f(x_0)}{x-x_0}=-2\neq 2=\lim_{x\rightarrow x_0^{\color{red}{+}}}\frac{f(x)-f(x_0)}{x-x_0}$$ Therefore your functions is not differentiable at $x=-\frac{1}{2}$",
        "output": {"claims": [
            {"claim_type": "LIMIT",
             "sympy_lhs": "Limit(Abs(2*x+1)/(x + Rational(1,2)), x, -Rational(1,2), '+')",
             "sympy_rhs": "2",
             "relation": "==", "free_symbols": ["x"], "assumptions": [],
             "load_bearing": True,
             "source_quote": r"You have $$\lim_{x\rightarrow x_0^{\color{red}{-}}}\frac{f(x)-f(x_0)}{x-x_0}=-2\neq 2=\lim_{x\rightarrow x_0^{\color{red}{+}}}\frac{f(x)-f(x_0)}{x-x_0}$$",
             "fidelity_note": "Instantiated f(x)=|2x+1| and x_0=-1/2 from the question (f(x_0)=0); '+' encodes the right-hand limit."},
            {"claim_type": "LIMIT",
             "sympy_lhs": "Limit(Abs(2*x+1)/(x + Rational(1,2)), x, -Rational(1,2), '-')",
             "sympy_rhs": "-2",
             "relation": "==", "free_symbols": ["x"], "assumptions": [],
             "load_bearing": True,
             "source_quote": r"You have $$\lim_{x\rightarrow x_0^{\color{red}{-}}}\frac{f(x)-f(x_0)}{x-x_0}=-2\neq 2=\lim_{x\rightarrow x_0^{\color{red}{+}}}\frac{f(x)-f(x_0)}{x-x_0}$$",
             "fidelity_note": "Same instantiation, left-hand limit ('-' direction)."},
        ]},
    },
    # --- pilot row 26: claims 58 + 60 (instantiating a quantified parameter)
    {
        "question": r"I'm trying to find bound on $p \gt 0$ where $\displaystyle \int_0^\infty\frac{\sin x}{x^p + \sin x} dx$ converges.",
        "answer": r"Due to $$\lim_{x\to0^+}\frac{\sin x}{x^p+\sin x}=\lim_{x\to0^+}\frac{\frac{\sin x}x}{x^{p-1}+\frac{\sin x}x}= \begin{cases} 0,& 0<p<1\\ \frac12,& p=1\\ 1,& p>1 \end{cases}, $$ the point $0$ is a continuous point for the function $\frac{\sin x}{x^p+\sin x}(p>0)$ . [...]",
        "output": {"claims": [
            {"claim_type": "LIMIT",
             "sympy_lhs": "Limit(sin(x)/(x + sin(x)), x, 0, '+')",
             "sympy_rhs": "Rational(1,2)",
             "relation": "==", "free_symbols": ["x"], "assumptions": [],
             "load_bearing": False,
             "source_quote": r"$$\lim_{x\to0^+}\frac{\sin x}{x^p+\sin x}=\lim_{x\to0^+}\frac{\frac{\sin x}x}{x^{p-1}+\frac{\sin x}x}= \begin{cases} 0,& 0<p<1\\ \frac12,& p=1\\ 1,& p>1 \end{cases}, $$",
             "fidelity_note": "Quantified case-split over p instantiated at p=1 (the 1/2 case). Spot-check, not a proof of the full claim."},
            {"claim_type": "LIMIT",
             "sympy_lhs": "Limit(sin(x)/(sqrt(x) + sin(x)), x, 0, '+')",
             "sympy_rhs": "0",
             "relation": "==", "free_symbols": ["x"], "assumptions": [],
             "load_bearing": False,
             "source_quote": r"$$\lim_{x\to0^+}\frac{\sin x}{x^p+\sin x}=\lim_{x\to0^+}\frac{\frac{\sin x}x}{x^{p-1}+\frac{\sin x}x}= \begin{cases} 0,& 0<p<1\\ \frac12,& p=1\\ 1,& p>1 \end{cases}, $$",
             "fidelity_note": "Same case-split instantiated in the 0<p<1 branch at p=1/2 (x**p -> sqrt(x))."},
        ]},
    },
    # --- pilot row 1: claims 4 + 6 (SUM with integer bound variables; headline identity)
    {
        "question": r"Combinatorial proof that $\sum_{k=0}^n \binom{n}{k} \frac{(-1)^k}{(k+1)^2} = \frac{H_{n+1}}{n+1}$",
        "answer": r"I found a probabilistic proof with a combinatorial flavor. Select points $(x_1, y_1), \ldots, (x_{n+1}, y_{n+1})$ independently from the bivariate uniform distribution on the unit square. Each side is the probability that, for all $i \in \{2, 3, \ldots, n+1 \}$, either $x_1 < x_i$ or $y_1 < y_i$ . [...] Summing up, we get that the probability we're after is also $$\frac{1}{n+1} \sum_{k=0}^n \frac{1}{k+1} = \frac{H_{n+1}}{n+1}.$$",
        "output": {"claims": [
            {"claim_type": "SUM",
             "sympy_lhs": "Sum(binomial(n,k)*(-1)**k/(k+1)**2, (k, 0, n))",
             "sympy_rhs": "harmonic(n+1)/(n+1)",
             "relation": "==",
             "free_symbols": ["n:integer,positive", "k:integer"],
             "assumptions": [],
             "load_bearing": True,
             "source_quote": r"Each side is the probability that, for all $i \in \{2, 3, \ldots, n+1 \}$, either $x_1 < x_i$ or $y_1 < y_i$ .",
             "fidelity_note": "The identity being proven is in the question title; the answer asserts both sides equal the same probability. H_{n+1} encoded as harmonic(n+1). The probabilistic argument itself is not checkable; only the identity is."},
            {"claim_type": "SUM",
             "sympy_lhs": "Sum(1/(k+1), (k, 0, n))/(n+1)",
             "sympy_rhs": "harmonic(n+1)/(n+1)",
             "relation": "==",
             "free_symbols": ["n:integer,positive", "k:integer"],
             "assumptions": [],
             "load_bearing": False,
             "source_quote": r"$$\frac{1}{n+1} \sum_{k=0}^n \frac{1}{k+1} = \frac{H_{n+1}}{n+1}.$$",
             "fidelity_note": "Faithful transcription of the final summation step."},
        ]},
    },
    # --- pilot row 2: claim 7 (NONE case)
    {
        "question": r"Upper semicontinuous function attains its supremum. Now I want to prove, that if $X$ is a compact space, then $f$ attains its supremum. Any hints how to start?",
        "answer": r"If you stare at the definition of upper semicontinuity long enough, you will find that the following is an equivalent characterization: $f : X \to [-\infty, + \infty )$ is upper semicontinuous iff for each $a \in (-\infty, +\infty)$ the inverse image $f^{-1} ( [ -\infty , a ) )$ is open. To prove that each upper semicontinuous function on a compact space attains a maximum, we take two steps. [...]",
        "output": {"claims": [
            {"claim_type": "NONE",
             "reason": "Pure point-set topology proof (upper semicontinuity, compactness, open covers). No sympy-expressible claim."},
        ]},
    },
]


def _render_few_shots(examples=FEW_SHOT_EXAMPLES):
    chunks = []
    for i, ex in enumerate(examples, 1):
        out = json.dumps(ex["output"], ensure_ascii=False)
        chunks.append(
            f"## Example {i}\n\n"
            f"QUESTION (context):\n{ex['question']}\n\n"
            f"ANSWER (extract from this):\n{ex['answer']}\n\n"
            f"OUTPUT:\n{out}"
        )
    return "\n\n".join(chunks)


SYSTEM_PROMPT = (
    SCHEMA_DOC
    + "\n\nWORKED EXAMPLES\n\n"
    + _render_few_shots()
    + "\n\nNow extract claims from the answer the user provides. "
      "Respond with the JSON object only."
)


def build_user_prompt(question: str, answer: str,
                      max_question_chars: int = 1200) -> str:
    q = question.strip()
    if len(q) > max_question_chars:
        q = q[:max_question_chars] + " [...truncated]"
    return (f"QUESTION (context):\n{q}\n\n"
            f"ANSWER (extract from this):\n{answer.strip()}\n\n"
            f"OUTPUT:")


# ------------------------------------------------------------------ validation
_WS = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _norm_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _parse_json_with_repair(raw: str):
    """Strict JSON parse with one repair pass. Returns (obj, error)."""
    if not raw or not raw.strip():
        return None, "empty output"
    s = _THINK_RE.sub("", raw).strip()
    # strip markdown fences
    if s.startswith("```"):
        s = _FENCE_OPEN.sub("", s, count=1)
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    try:
        return json.loads(s), None
    except Exception:
        pass
    # repair pass: balanced top-level object + trailing-comma removal
    start = s.find("{")
    if start == -1:
        return None, "no JSON object found"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                cand = _TRAILING_COMMA.sub(r"\1", s[start:i + 1])
                try:
                    return json.loads(cand), None
                except Exception as e:
                    return None, f"repair parse failed: {e}"
    return None, "unbalanced JSON object (likely truncated)"


def validate_extraction(raw_llm_text: str, source_text: str = None):
    """Validate one LLM extraction output.

    Returns (claims, errors):
      claims  list of schema-valid claim dicts, each with a `fidelity` dict
              attached ({"quote_found": bool, "match_kind": ...}) when
              source_text is provided;
      errors  list of strings; non-empty errors mean the caller should retry
              this answer (with a different seed) or, on the final attempt,
              accept the valid subset.
    """
    errors = []
    obj, perr = _parse_json_with_repair(raw_llm_text)
    if obj is None:
        return [], [f"json: {perr}"]
    if isinstance(obj, list):
        raw_claims = obj
    elif isinstance(obj, dict):
        raw_claims = obj.get("claims")
        if raw_claims is None:
            return [], ["json: missing 'claims' key"]
        if not isinstance(raw_claims, list):
            return [], ["json: 'claims' is not a list"]
    else:
        return [], ["json: top level is neither object nor list"]

    valid = []
    nsrc = _norm_ws(source_text) if source_text else None
    for i, c in enumerate(raw_claims):
        if not isinstance(c, dict):
            errors.append(f"claim[{i}]: not an object")
            continue
        ct = c.get("claim_type")
        if ct not in CLAIM_TYPES:
            errors.append(f"claim[{i}]: invalid claim_type {ct!r}")
            continue
        if ct == "NONE":
            valid.append({"claim_type": "NONE",
                          "reason": str(c.get("reason", ""))[:500]})
            continue
        # default the relation before schema validation
        if not c.get("relation"):
            c["relation"] = DEFAULT_RELATION.get(ct)
        schema_errs = validate_claim_schema(c)
        if schema_errs:
            errors.append(f"claim[{i}]: " + "; ".join(schema_errs[:4]))
            continue
        # fidelity requirements (the pilot's #1 risk is silent mis-formalization)
        quote = c.get("source_quote")
        if not isinstance(quote, str) or len(quote.strip()) < 8:
            errors.append(f"claim[{i}]: missing/too-short source_quote")
            continue
        note = c.get("fidelity_note")
        if not isinstance(note, str) or not note.strip():
            errors.append(f"claim[{i}]: missing fidelity_note")
            continue
        if "load_bearing" not in c or not isinstance(c["load_bearing"], bool):
            c["load_bearing"] = False
            c.setdefault("warnings", []).append("load_bearing defaulted to false")
        # round-trip quote check against the answer text
        if nsrc is not None:
            if quote in source_text:
                c["fidelity"] = {"quote_found": True, "match_kind": "exact"}
            elif _norm_ws(quote) in nsrc:
                c["fidelity"] = {"quote_found": True, "match_kind": "ws_normalized"}
            else:
                c["fidelity"] = {"quote_found": False, "match_kind": "none"}
                errors.append(f"claim[{i}]: source_quote not found in answer")
                continue
        valid.append(c)

    if not raw_claims:
        errors.append("claims list is empty (use a NONE claim for prose answers)")
    return valid, errors

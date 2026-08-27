#!/usr/bin/env python3
"""mathse_lint.py — deterministic "math text linter" V features for Math.SE.

Pure tools only — NO LLM anywhere: regex + sympy (lark LaTeX backend) +
pylatexenc + pyspellchecker + a cached Wikipedia theorem-name list.
Strategic purpose: deterministic verification coverage that is independent of
the LLM claim-extraction pipeline (verification/run_extraction_sk3.py).

Metric families (one row of features per answer):
  1. step_chain        sympy-checked derivation chains in display math
  2. literal_arith     exact verification of pure-numeric equalities
  3. latex_parse       LaTeX well-formedness (pylatexenc + brace/env balance)
  4. symbol_hygiene    undefined / unused single-letter symbols (conservative)
  5. dangling_refs     "(N)" / \\eqref / "the lemma above" with no referent
  6. typo_density      spellcheck on prose with corpus-derived jargon allowlist
  7. theorem_names     known-theorem mentions + fuzzy-misspelled mentions
  8. near_dup          max 5-gram-shingle jaccard to a sibling answer
  9. form_contract     question speech-act vs answer form (boxed/QED/example)

Sympy work is fork-isolated per row (harness.py pattern): the pool worker
forks a child which streams one JSON line per step verdict; the parent kills
the child at the row budget and keeps whatever verdicts were produced.

Usage (sk3):
  P=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
  $P mathse_lint.py --data math_se_v3_3_propensity_balanced.csv.gz \
      --sample 30 --verbose                       # eyeball validation
  $P mathse_lint.py --data ... --build-allowlist 3000   # jargon candidates
  $P mathse_lint.py --data ... --out mathse_lint_features.csv --workers 24
  $P mathse_lint.py --analyze mathse_lint_features.csv --data ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
STEP_TIMEOUT_S = 2.0          # per derivation step (SIGALRM inside child)
ROW_BUDGET_BASE_S = 4.0       # fork budget = base + STEP_TIMEOUT*n, capped
ROW_BUDGET_CAP_S = 30.0
MAX_STEPS_PER_ROW = 30
MAX_ARITH_PER_ROW = 40
N_EQ_SAMPLES = 8

GREEK = r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Sigma|Phi|Psi|Omega"

# ---------------------------------------------------------------- text prep
QA_SPLIT = "\n\nAnswer:"


def split_qa(text: str):
    q, _, a = text.partition(QA_SPLIT)
    if q.startswith("Question:"):
        q = q[len("Question:"):]
    return q.strip(), a.strip()


ENV_RE = re.compile(
    r"\\begin\{(align\*?|eqnarray\*?|gather\*?|equation\*?|multline\*?|"
    r"alignat\*?|aligned|split|array)\}(.*?)\\end\{\1\}", re.S)
DOLLAR2_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.S)
INLINE_RE = re.compile(r"\$([^$\n]{1,400}?)\$")
CODE_LINE_RE = re.compile(r"^(?:    |\t)", re.M)


def extract_math(text: str):
    """Return (display_blocks, inline_blocks, prose) — prose has math/code
    stripped (replaced with sentinels) but keeps sentence structure."""
    t = text.replace(r"\$", " ")
    display = []

    def grab(m):
        display.append(m.group(m.lastindex))
        return " <DMATH> "

    t = ENV_RE.sub(grab, t)
    t = DOLLAR2_RE.sub(grab, t)
    t = BRACKET_RE.sub(grab, t)
    inline = []

    def grab_i(m):
        inline.append(m.group(1))
        return " <IMATH> "

    t = INLINE_RE.sub(grab_i, t)
    return display, inline, t


# ------------------------------------------------------- chain tokenization
REL_MACROS = {"\\le", "\\leq", "\\leqq", "\\leqslant",
              "\\ge", "\\geq", "\\geqq", "\\geqslant"}
BREAK_MACROS = {"\\quad", "\\qquad", "\\implies", "\\Rightarrow", "\\iff",
                "\\Leftrightarrow", "\\Longrightarrow", "\\longrightarrow",
                "\\to", "\\rightarrow", "\\mapsto", "\\Longleftrightarrow",
                "\\neq", "\\ne", "\\approx", "\\sim", "\\simeq", "\\cong",
                "\\equiv", "\\propto", "\\in", "\\notin", "\\subset",
                "\\subseteq", "\\supset", "\\supseteq", "\\pmod", "\\mid",
                "\\nmid", "\\parallel", "\\perp"}

UNICODE_MATH = {"−": "-", "–": "-", "—": "-", "⋅": " \\cdot ", "·": " \\cdot ",
                "×": " \\times ", "≤": "<=", "≥": ">=", "≠": " \\neq ",
                "∞": " \\infty ", "√": " \\sqrt ", "π": " \\pi ",
                "“": " ", "”": " ", "’": "'"}


def unicode_fix(s: str):
    for k, v in UNICODE_MATH.items():
        if k in s:
            s = s.replace(k, v)
    return s
TEXT_MACROS = {"\\text", "\\mbox", "\\textbf", "\\textit", "\\textrm",
               "\\hbox", "\\mathtext"}
MACRO_RE = re.compile(r"\\[a-zA-Z]+")


def _consume_group(s, i):
    """s[i] should be '{'; return index just past the matching '}'."""
    if i >= len(s) or s[i] != "{":
        return i
    depth = 0
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def split_chain(s: str):
    """Split a math string at top-level relations (outside {} AND ()/[]).
    Returns list of (segments, relations) sub-chains (broken at
    commas/arrows/\\text/set-relations)."""
    segs, rels, cur = [], [], []
    i, n = 0, len(s)
    depth = 0   # brace depth
    pdepth = 0  # paren/bracket depth (single counter tolerates [0, 1) )
    while i < n:
        c = s[i]
        if c == "\\":
            m = MACRO_RE.match(s, i)
            tok = m.group(0) if m else s[i:i + 2]
            j = i + len(tok)
            top = depth == 0 and pdepth == 0
            if top and tok in TEXT_MACROS:
                k = j
                while k < n and s[k].isspace():
                    k += 1
                j = _consume_group(s, k) if k < n and s[k] == "{" else k
                segs.append("".join(cur)); rels.append("BREAK"); cur = []
            elif top and tok in BREAK_MACROS:
                segs.append("".join(cur)); rels.append("BREAK"); cur = []
            elif top and tok in REL_MACROS:
                segs.append("".join(cur)); rels.append("<=" if "l" in tok[1:3] else ">="); cur = []
            else:
                cur.append(tok)
            i = j
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        elif c in "([" and depth == 0:
            pdepth += 1
        elif c in ")]" and depth == 0:
            pdepth = max(0, pdepth - 1)
        elif depth == 0 and pdepth == 0:
            if c == "=":
                if cur and cur[-1] == ":":      # := is a definition: break
                    cur.pop()
                    segs.append("".join(cur)); rels.append("BREAK"); cur = []
                    i += 1; continue
                if cur and cur[-1] in ("<", ">"):
                    cur.append(c); i += 1; continue
                segs.append("".join(cur)); rels.append("="); cur = []
                i += 1; continue
            if c in "<>":
                if cur and cur[-1] == "-":      # "->" pseudo-arrow: break
                    cur.pop()
                    segs.append("".join(cur)); rels.append("BREAK"); cur = []
                    i += 1; continue
                if i + 1 < n and s[i + 1] == "=":
                    segs.append("".join(cur)); rels.append(c + "="); cur = []
                    i += 2; continue
                segs.append("".join(cur)); rels.append(c); cur = []
                i += 1; continue
            if c in ",;":
                segs.append("".join(cur)); rels.append("BREAK"); cur = []
                i += 1; continue
        cur.append(c)
        i += 1
    segs.append("".join(cur))
    # split into sub-chains at BREAKs
    chains, cs, cr = [], [segs[0]], []
    for r, seg in zip(rels, segs[1:]):
        if r == "BREAK":
            if len(cs) >= 2:
                chains.append((cs, cr))
            cs, cr = [seg], []
        else:
            cs.append(seg); cr.append(r)
    if len(cs) >= 2:
        chains.append((cs, cr))
    return chains


def env_to_chain_text(body: str):
    """Join multiline align/eqnarray bodies into one chain string; lines not
    starting with a relation begin a new chain (separated by ';')."""
    out = []
    for line in body.split("\\\\"):
        ls = line.replace("&", " ").strip()
        ls = re.sub(r"\\(?:tag\*?|label)\s*\{[^{}]*\}", " ", ls)
        ls = re.sub(r"\\(?:notag|nonumber)\b", " ", ls)
        if not ls:
            continue
        if out and not re.match(
                r"^\s*(=|<|>|\\le\b|\\leq\b|\\ge\b|\\geq\b|\\leqslant\b|\\geqslant\b)", ls):
            out.append(";")
        out.append(ls)
    return " ".join(out)


# ------------------------------------------------------------- parse prep
DOTS_RE = re.compile(r"\\[lcv]?dots|\.\.\.")
BIG_NUM_RE = re.compile(r"\^\s*\{?\s*-?\d{5,}|\d{5,}\s*!|\^\s*\{?\s*-?\d+\.?\d*[eE]\d")
# Only flag as "unknown applied function" when it actually looks like one:
# common function letters, primed letters, or subscripted symbols before "(".
UNKNOWN_FUNC_RE = re.compile(
    r"(?<![a-zA-Z\\])(?:[fghFGHPQ]|[a-zA-Z](?:'+|_\{[^{}]{0,12}\}|_[a-zA-Z0-9]))\s*\(")


def clean_segment(s: str):
    s = unicode_fix(s)
    s = re.sub(r"\\(?:left|right)\s*\.", " ", s)
    s = re.sub(r"\\(?:left|right|big[lrm]?|Big[lrm]?|bigg[lrm]?|Bigg[lrm]?)\b", " ", s)
    s = re.sub(r"\\(?:displaystyle|limits|nonumber|notag|allowbreak)\b", " ", s)
    s = re.sub(r"\\(?:tag\*?|label)\s*\{[^{}]*\}", " ", s)
    s = re.sub(r"\\[,;!: ]", " ", s)
    s = re.sub(r"\\([dtc])frac\b", r"\\frac", s)
    s = re.sub(r"\\math(?:bb|cal|bf|rm|sf|frak|scr|it)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\math(?:bb|cal|bf|rm|sf|frak|scr|it)\s*([A-Za-z])", r"\1", s)
    s = re.sub(r"\\boldsymbol\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\{", "(").replace("\\}", ")")
    # \pi is missing from the lark grammar; use the numeric constant
    s = re.sub(r"\\pi\b", "(3.141592653589793)", s)
    # lark chokes on implicit multiplication right after an exponent:
    # x^2(x+1), x^2y, (y+h)^3 y^3 — insert an explicit \cdot
    s = re.sub(r"(\^\s*(?:\{[^{}]*\}|\w))(?=\s*[A-Za-z0-9(])", r"\1 \\cdot ", s)
    s = s.strip()
    while s and s[-1] in ".,;:?":
        s = s[:-1].rstrip()
    return s


def segment_parseable_guard(s: str):
    """Cheap pre-checks; return reason string if the segment must be skipped."""
    if not s or not s.strip():
        return "empty"
    if len(s) > 300:
        return "too_long"
    if DOTS_RE.search(s):
        return "dots"
    if BIG_NUM_RE.search(s):
        return "huge_number"
    if UNKNOWN_FUNC_RE.search(s):
        return "unknown_function_application"
    return None


PURE_NUM_MACROS_RE = re.compile(
    r"\\(?:frac|cdot|times|div|binom|sqrt|left|right|,|;|!|quad|qquad|"
    r"d?frac|tfrac|text\{[^{}]*\}|%)")
ARITH_OP_RE = re.compile(r"[-+*/^!]|\\(?:frac|cdot|times|div|binom|sqrt)")
# modular-arithmetic context: "=" inside such a chain is congruence, not
# equality — skip the whole chain (e.g. "2^8 = 23 \pmod{233}")
MOD_CONTEXT_RE = re.compile(r"\\[bp]?mod\b|\(\s*mod\b|\bmod\b|\\equiv")
# Legendre/Jacobi symbols look exactly like a lone parenthesized fraction
LEGENDRE_RE = re.compile(
    r"^\s*\\left\(\s*\\?[dtc]?frac\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\\right\)\s*$")
# mixed numbers ("38\frac{11}{18}" = 38 + 11/18, not 38*(11/18))
MIXED_NUM_RE = re.compile(r"\d\s*\\[dtc]?frac\b")


def is_pure_numeric(seg: str):
    s = PURE_NUM_MACROS_RE.sub(" ", seg)
    s = re.sub(r"\\[a-zA-Z]+", "MACRO", s)
    if "MACRO" in s:
        return False
    return bool(re.search(r"\d", seg)) and bool(
        re.fullmatch(r"[\d+\-*/^!().,{}%\s]*", s))


# ---------------------------------------------------- forked sympy checking
class StepTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise StepTimeout()


def _child_check_tasks(tasks, wfd):
    """Runs in the forked child. Streams one JSON line per task verdict."""
    import sympy as sp
    from sympy.core.function import AppliedUndef
    from sympy.parsing.latex import parse_latex
    signal.signal(signal.SIGALRM, _alarm)

    def parse(seg):
        """Return a list of candidate interpretations (lark may return an
        ambiguity Tree, e.g. n(n+1) = apply-or-multiply)."""
        e = parse_latex(clean_segment(seg), backend="lark")
        if isinstance(e, sp.Basic):
            cands = [e]
        elif hasattr(e, "data") and str(e.data) == "_ambig":
            cands = [c for c in e.children if isinstance(c, sp.Basic)]
            no_undef = [c for c in cands if not c.atoms(AppliedUndef)]
            cands = no_undef or cands
        else:
            cands = []
        if not cands:
            raise ValueError("unparseable")
        # lark leaves e as a plain Symbol; treat it as Euler's number
        return [c.subs(sp.Symbol("e"), sp.E) for c in cands[:3]]

    def emit(i, v, note=""):
        line = json.dumps({"i": i, "v": v, "note": note[:120]}) + "\n"
        os.write(wfd, line.encode())

    def num_pair(a, b, rel, seed):
        """Numeric sampling on positive reals (dodges branch cuts)."""
        frees = sorted(a.free_symbols | b.free_symbols, key=str)
        rng = random.Random(seed)
        match = mis = holds = viol = 0
        for _ in range(N_EQ_SAMPLES):
            pt = {s: sp.Float(rng.uniform(0.15, 2.5)) for s in frees}
            try:
                av = complex(sp.N(a.subs(pt), 18))
                bv = complex(sp.N(b.subs(pt), 18))
            except Exception:
                continue
            if rel == "=":
                scale = max(1.0, abs(av), abs(bv))
                rd = abs(av - bv) / scale
                if rd < 1e-7:
                    match += 1
                elif rd > 1e-4:
                    mis += 1
            else:
                if abs(av.imag) > 1e-9 or abs(bv.imag) > 1e-9:
                    continue
                margin = 1e-6 * max(1.0, abs(av), abs(bv))
                ok = {"<=": av.real <= bv.real + margin,
                      "<": av.real < bv.real + margin,
                      ">=": av.real >= bv.real - margin,
                      ">": av.real > bv.real - margin}[rel]
                holds += ok
                viol += not ok
        if rel == "=":
            if match >= 6 and mis == 0:
                return "VERIFIED"
            if mis >= 6 and match == 0 and a.free_symbols == b.free_symbols:
                return "REFUTED"
            return "INCONCLUSIVE"
        if viol >= 4 and holds == 0 and a.free_symbols == b.free_symbols:
            return "REFUTED"
        if holds >= 6 and viol == 0:
            return "VERIFIED"
        return "INCONCLUSIVE"

    def check_pair(a, b, rel, seed):
        """i may be an index (real) or the imaginary unit: verify if either
        reading verifies; refute only if both readings refute."""
        v = check_pair_real(a, b, rel, seed)
        names = {s.name for s in (a.free_symbols | b.free_symbols)}
        if "i" in names and v in ("REFUTED", "INCONCLUSIVE"):
            try:
                v2 = check_pair_real(a.subs(sp.Symbol("i"), sp.I),
                                     b.subs(sp.Symbol("i"), sp.I), rel, seed)
            except (StepTimeout, Exception):
                v2 = "INCONCLUSIVE"
            if v2 == "VERIFIED":
                return "VERIFIED"
            if v == "REFUTED" and v2 != "REFUTED":
                return "INCONCLUSIVE"
        return v

    def check_pair_real(a, b, rel, seed):
        if rel == "=":
            try:
                d = sp.simplify(sp.expand(a - b))
                if d == 0:
                    return "VERIFIED"
                if d.is_number and not (a.free_symbols | b.free_symbols):
                    try:
                        if abs(complex(sp.N(d, 20))) > 1e-9:
                            return "REFUTED"
                    except Exception:
                        pass
            except StepTimeout:
                # give numeric sampling a fresh (smaller) slice
                signal.setitimer(signal.ITIMER_REAL, 1.0)
            except Exception:
                pass
        return num_pair(a, b, rel, seed)

    def check_arith_pair(a, b):
        has_float = a.atoms(sp.Float) or b.atoms(sp.Float)
        d = abs(complex(sp.N(a - b, 30)))
        scale = max(1.0, abs(complex(sp.N(a, 30))), abs(complex(sp.N(b, 30))))
        tol = 1e-2 if has_float else 1e-25
        return "EQUAL" if d / scale < tol else "WRONG"

    for i, t in enumerate(tasks):
        kind, lhs_s, rhs_s, rel = t
        signal.setitimer(signal.ITIMER_REAL, STEP_TIMEOUT_S)
        try:
            try:
                A = parse(lhs_s)
                B = parse(rhs_s)
            except StepTimeout:
                emit(i, "PARSE_FAIL", "parse timeout"); continue
            except Exception as e:
                emit(i, "PARSE_FAIL", f"{type(e).__name__}"); continue

            # charitable ambiguity resolution: any interpretation pair that
            # verifies counts; refute only if every pair refutes
            verdicts = set()
            try:
                for ca in A[:2]:
                    for cb in B[:2]:
                        if kind == "arith":
                            try:
                                verdicts.add(check_arith_pair(ca, cb))
                            except Exception:
                                verdicts.add("SKIP")
                        else:
                            verdicts.add(check_pair(ca, cb, rel, 1234 + i))
                        if "VERIFIED" in verdicts or "EQUAL" in verdicts:
                            break
                    if "VERIFIED" in verdicts or "EQUAL" in verdicts:
                        break
            except StepTimeout:
                verdicts.add("INCONCLUSIVE")
            if kind == "arith":
                if "EQUAL" in verdicts:
                    v = "EQUAL"
                elif verdicts == {"WRONG"}:
                    v = "WRONG"
                else:
                    v = "SKIP"
                emit(i, v, f"{lhs_s.strip()[:40]} = {rhs_s.strip()[:40]}")
            else:
                if "VERIFIED" in verdicts:
                    v = "VERIFIED"
                elif verdicts == {"REFUTED"}:
                    v = "REFUTED"
                else:
                    v = "INCONCLUSIVE"
                emit(i, v, ",".join(sorted(verdicts))[:60])
        except StepTimeout:
            emit(i, "INCONCLUSIVE", "timeout")
        except Exception as e:
            try:
                emit(i, "INCONCLUSIVE", f"{type(e).__name__}")
            except Exception:
                pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def run_tasks_forked(tasks):
    """Fork-isolated sympy verification (harness.py pattern, streaming)."""
    if not tasks:
        return {}
    budget = min(ROW_BUDGET_CAP_S, ROW_BUDGET_BASE_S + STEP_TIMEOUT_S * len(tasks))
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        try:
            _child_check_tasks(tasks, w)
        finally:
            os._exit(0)
    os.close(w)
    deadline = time.time() + budget
    status = None
    while time.time() < deadline:
        wpid, st = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            status = st
            break
        time.sleep(0.02)
    if status is None:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
    chunks = []
    os.set_blocking(r, False)
    try:
        while True:
            c = os.read(r, 65536)
            if not c:
                break
            chunks.append(c)
    except (BlockingIOError, OSError):
        pass
    os.close(r)
    out = {}
    for line in b"".join(chunks).decode(errors="replace").splitlines():
        try:
            d = json.loads(line)
            out[d["i"]] = (d["v"], d.get("note", ""))
        except Exception:
            continue
    return out


# --------------------------------------------------------- 1+2: chain tasks
SUBSCRIPTED_RE = re.compile(
    r"(\\[a-zA-Z]+|[A-Za-z])\s*_\s*(\{[^{}]{1,12}\}|[A-Za-z0-9])")
FUNC_POOL = "fghFGHPQ"
GEN_POOL = "pqrstuvwabcgjkmnzyxPRSTUVWABJKMNZYX"


def desubscript_chain(segs):
    """lark's LaTeX grammar has no subscripted symbols; map each distinct
    subscripted token (x_1, a_{n+1}, F_X) to a fresh single letter,
    consistently across the whole chain so identity structure survives."""
    joined = " ".join(segs)
    toks = []
    for m in SUBSCRIPTED_RE.finditer(joined):
        if m.group(1).startswith("\\") and not GREEK_RE.fullmatch(m.group(1)):
            continue  # \sum_{...}, \int_{...}, \log_2 — leave big ops alone
        key = re.sub(r"\s+", "", m.group(0))
        if key not in toks:
            toks.append(key)
    if not toks:
        return segs
    used = set(re.findall(r"[A-Za-z]", joined))
    func_pool = [c for c in FUNC_POOL if c not in used]
    gen_pool = [c for c in GEN_POOL if c not in used]
    mapping = {}
    for key in toks:
        funcish = re.search(re.escape(key) + r"\s*\(", joined)
        pool = func_pool if (funcish and func_pool) else gen_pool
        if not pool:
            return segs  # give up; the segment will just fail to parse
        mapping[key] = pool.pop(0)
        if pool is func_pool and mapping[key] in gen_pool:
            gen_pool.remove(mapping[key])

    def repl(m):
        if m.group(1).startswith("\\") and not GREEK_RE.fullmatch(m.group(1)):
            return m.group(0)
        return mapping.get(re.sub(r"\s+", "", m.group(0)), m.group(0))

    return [SUBSCRIPTED_RE.sub(repl, s) for s in segs]


PROSE_ARITH_RE = re.compile(
    r"(?<![\w.$])(\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*\(?\d+(?:\.\d+)?\)?)+)"
    r"\s*=\s*(-?\d+(?:\.\d+)?)(?![\w.])")


def _prose_to_latex(s: str):
    s = s.replace("*", " \\cdot ")
    s = re.sub(r"\^\s*(\d+)", r"^{\1}", s)
    return s


def build_chain_tasks(display, inline, prose=""):
    """Return (step_tasks, arith_tasks, n_unparseable_pre, artifacts)."""
    step_tasks, arith_tasks, pre_unparseable, arts = [], [], 0, []
    chain_texts = [unicode_fix(env_to_chain_text(b)) for b in display]
    for seg in inline:  # inline only mined for arithmetic + a=b=c chains
        if seg.count("=") >= 1 or "≤" in seg or "≥" in seg:
            chain_texts.append(unicode_fix(env_to_chain_text(seg)))
    seen_arith = set()
    # plain-text arithmetic in prose, e.g. "3+4=7"
    for m in PROSE_ARITH_RE.finditer(prose):
        if re.match(r"\s*\(?\s*mod\b", prose[m.end():m.end() + 12], re.I):
            continue  # "2^8 = 23 mod 233" is congruence
        key = (m.group(1).strip(), m.group(2).strip())
        if key not in seen_arith and len(arith_tasks) < MAX_ARITH_PER_ROW:
            seen_arith.add(key)
            arith_tasks.append(("arith", _prose_to_latex(m.group(1)),
                                _prose_to_latex(m.group(2)), "="))
    for ct in chain_texts:
        if MOD_CONTEXT_RE.search(ct):
            continue  # congruence chains: "=" is not equality there
        for segs, rels in split_chain(ct):
            segs = desubscript_chain(segs)
            for k, rel in enumerate(rels):
                lhs, rhs = segs[k], segs[k + 1]
                if rel == "=" and is_pure_numeric(lhs) and is_pure_numeric(rhs):
                    key = (lhs.strip(), rhs.strip())
                    if key in seen_arith:
                        continue
                    seen_arith.add(key)
                    if (LEGENDRE_RE.match(lhs) or LEGENDRE_RE.match(rhs) or
                            MIXED_NUM_RE.search(lhs) or MIXED_NUM_RE.search(rhs)):
                        continue  # Legendre symbols / mixed numbers
                    if len(arith_tasks) < MAX_ARITH_PER_ROW:
                        # require an actual operation somewhere in the pair
                        if ARITH_OP_RE.search(lhs) or ARITH_OP_RE.search(rhs):
                            arith_tasks.append(("arith", lhs, rhs, "="))
                    continue
                g = segment_parseable_guard(clean_segment(lhs)) or \
                    segment_parseable_guard(clean_segment(rhs))
                if g:
                    pre_unparseable += 1
                    arts.append(f"SKIP[{g}]: {lhs.strip()[:60]} {rel} {rhs.strip()[:60]}")
                    continue
                if len(step_tasks) < MAX_STEPS_PER_ROW:
                    step_tasks.append(("step", lhs, rhs, rel))
    return step_tasks, arith_tasks, pre_unparseable, arts


# ----------------------------------------------------------- 3: latex_parse
def latex_parse_features(answer, display, inline):
    from pylatexenc.latexwalker import LatexWalker
    n_blocks = len(display) + len(inline)
    n_err = 0
    blocks_with_err = 0
    for blk in display + inline:
        e = 0
        if blk.count("{") != blk.count("}"):
            e += 1
        nl = len(re.findall(r"\\left\b", blk))
        nr = len(re.findall(r"\\right\b", blk))
        if nl != nr:
            e += 1
        begins = re.findall(r"\\begin\{([^{}]+)\}", blk)
        ends = re.findall(r"\\end\{([^{}]+)\}", blk)
        if sorted(begins) != sorted(ends):
            e += 1
        try:
            LatexWalker("$" + blk + "$", tolerant_parsing=False).get_latex_nodes()
        except Exception:
            e += 1
        n_err += e
        blocks_with_err += e > 0
    # unbalanced $ in the whole answer (after stripping \$ and $$)
    t = answer.replace(r"\$", "").replace("$$", "")
    if t.count("$") % 2 == 1:
        n_err += 1
    return {"n_latex_errors": n_err,
            "frac_blocks_with_errors": blocks_with_err / n_blocks if n_blocks else 0.0,
            "n_math_blocks": n_blocks}


# ------------------------------------------------------- 4: symbol hygiene
SKIP_SYMBOLS = {"e", "i", "d", "o", "O", "C", "I"}
GREEK_SKIP = {"\\pi", "\\infty", "\\epsilon", "\\varepsilon", "\\delta"}
INDEX_LETTERS = {"i", "j", "k", "l", "m", "n"}
LETTER_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])")
GREEK_RE = re.compile(r"\\(?:" + GREEK + r")\b")
BIGOP_SUB_RE = re.compile(
    r"\\(?:lim|liminf|limsup|sum|prod|int|oint|max|min|sup|inf|bigcup|"
    r"bigcap|substack)\s*_\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\w)")

BINDER_RE = re.compile(
    r"\b(let|set|put|define[sd]?|denote[sd]?|write|writing|fix|choose|take|"
    r"suppose|assume|given|consider|if|say|where|for\s+(?:all|every|each|any|some)|"
    r"there\s+(?:exists?|is)|"
    r"lines?|points?|sets?|functions?|numbers?|elements?|vectors?|"
    r"matrix|matrices|groups?|sequences?|primes?|constants?|integers?|"
    r"reals?|variables?|polynomials?|roots?|solutions?|ideals?|subgroups?|"
    r"eigenvalues?|angles?|sides?|triangles?|curves?|spaces?|operators?"
    r")\b[^.\n$]{0,30}\$([^$\n]{1,80})\$", re.I)
POST_BINDER_RE = re.compile(
    r"\$([^$\n]{1,80})\$\s*(?:denotes?|is\s+defined|be\s|is\s+an?\s|is\s+the\s|stands?\s+for)", re.I)
DEF_BINDERS = {"let", "set", "put", "define", "defines", "defined", "denote",
               "denotes", "denoted", "write", "writing"}


def _symbols_in(math_str):
    """Single latin letters + greek macros used as standalone symbols."""
    s = re.sub(r"\\(?:text|mbox|mathrm|operatorname)\s*\{[^{}]*\}", " ", math_str)
    s = re.sub(r"\\math(?:bb|cal|bf|sf|frak|scr)\s*\{[^{}]*\}", " ", s)
    out = []
    for tok in LETTER_TOKEN_RE.findall(s):
        if tok.startswith("\\"):
            if GREEK_RE.fullmatch(tok):
                out.append(tok)
        else:
            out.append(tok)
    return out


def symbol_hygiene_features(q_display, q_inline, a_display, a_inline,
                            answer_raw):
    a_math = a_display + a_inline
    counts = Counter()
    for blk in a_math:
        counts.update(_symbols_in(blk))
    q_bound = set()
    for blk in q_display + q_inline:
        q_bound.update(_symbols_in(blk))

    bound = set(q_bound)
    def_bound = {}  # symbol -> n occurrences inside its binding groups
    for m in BINDER_RE.finditer(answer_raw):
        syms = _symbols_in(m.group(2))
        for s in syms:
            bound.add(s)
            if m.group(1).lower().split()[0] in DEF_BINDERS:
                def_bound[s] = def_bound.get(s, 0) + syms.count(s)
    for m in POST_BINDER_RE.finditer(answer_raw):
        for s in _symbols_in(m.group(1)):
            bound.add(s)
    # defined-by-equation: "X =" or "X(...)=" at any point in answer math
    allmath = " ;; ".join(a_math)
    for m in re.finditer(r"(\\[a-zA-Z]+|[a-zA-Z])\s*(?:\([^()]{0,20}\)|_\{?[^{}=]{0,8}\}?)?\s*=", allmath):
        tok = m.group(1)
        if not tok.startswith("\\") or GREEK_RE.fullmatch(tok):
            bound.add(tok)
    # letters bound by big-operator subscripts: \sum_{n=1}, \lim_{h\to 0}, ...
    for m in BIGOP_SUB_RE.finditer(allmath):
        for s in _symbols_in(m.group(1)):
            bound.add(s)
    # index letters bound when they appear as sub/superscripts anywhere
    for sym in list(counts):
        if sym in INDEX_LETTERS and re.search(
                r"[_^]\s*\{?[^{}]{0,12}" + re.escape(sym), allmath):
            bound.add(sym)

    undefined, unused = [], []
    for sym, c in counts.items():
        if sym in SKIP_SYMBOLS or sym in GREEK_SKIP:
            continue
        if c >= 2 and sym not in bound:
            undefined.append(sym)
    for sym, in_bind in def_bound.items():
        if sym in SKIP_SYMBOLS or sym in GREEK_SKIP:
            continue
        if counts.get(sym, 0) <= in_bind:
            unused.append(sym)
    return ({"n_undefined_symbols": len(undefined),
             "n_unused_definitions": len(unused)},
            {"undefined": undefined, "unused": unused})


# ------------------------------------------------------- 5: dangling refs
TAG_RE = re.compile(r"\\tag\*?\{?\(?\s*([\w.*'-]+)\s*\)?\}?")
LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")
EQREF_RE = re.compile(r"\\(?:eq)?ref\{([^{}]+)\}")
NUMREF_RE = re.compile(
    r"\b(?:equations?|eq|eqn|eqs|identity|inequality|formula|by|from|using|see)"
    r"[.\s]{0,2}\$?\\?\(\s*(\d{1,2}|\*)\s*\\?\)\$?", re.I)
LINESTART_ANCHOR_RE = re.compile(r"^\s*\(?(\d{1,2})[.)]\s", re.M)
LEMMA_REF_RE = re.compile(
    r"\b(?:the|this)\s+(?:above\s+)?(lemma|claim|proposition)\b|"
    r"\b(lemma|claim|proposition)\s+(?:above|below)\b", re.I)


def dangling_refs_features(answer, question=""):
    src = answer + "\n" + question  # question tags are valid referents too
    anchors = set(TAG_RE.findall(src))
    anchors |= {a.lstrip("0") or a for a in anchors}
    labels = set(LABEL_RE.findall(src))
    anchors_num = set(anchors) | set(LINESTART_ANCHOR_RE.findall(src))
    n_refs = n_dangling = 0
    arts = []
    for m in EQREF_RE.finditer(answer):
        n_refs += 1
        if m.group(1) not in labels and m.group(1) not in anchors:
            n_dangling += 1
            arts.append(f"dangling \\ref{{{m.group(1)}}}")
    for m in NUMREF_RE.finditer(answer):
        n_refs += 1
        tgt = m.group(1)
        if tgt not in anchors_num:
            n_dangling += 1
            arts.append(f"dangling ref ({tgt}) in: ...{answer[max(0, m.start()-30):m.end()][:60]}")
    for m in LEMMA_REF_RE.finditer(answer):
        word = (m.group(1) or m.group(2)).lower()
        n_refs += 1
        before = answer[:m.start()].lower() + "\n" + question.lower()
        after = answer[m.end():].lower() + "\n" + question.lower()
        hay = before if "below" not in m.group(0).lower() else after
        if word not in hay:
            n_dangling += 1
            arts.append(f"dangling '{m.group(0).strip()}'")
    return ({"n_refs": n_refs, "n_dangling": n_dangling}, arts)


# ------------------------------------------------------- 6: typo density
WORD_RE = re.compile(r"[A-Za-z']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>\n]{1,80}>")
BACKTICK_RE = re.compile(r"`[^`\n]*`")

_SPELL = None
_ALLOW = None


def _get_spell():
    global _SPELL, _ALLOW
    if _SPELL is None:
        from spellchecker import SpellChecker
        _SPELL = SpellChecker()
        _ALLOW = set()
        p = RESOURCE_DIR / "math_jargon_allowlist.txt"
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    _ALLOW.add(line)
    return _SPELL, _ALLOW


def prose_words(prose: str):
    t = CODE_LINE_RE.sub("", prose)
    t = BACKTICK_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # markdown links
    words = []
    for w in WORD_RE.findall(t):
        w = w.strip("'")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) < 4 or not w.isascii():
            continue
        if not w.islower():
            continue  # skip proper nouns / acronyms (names are not typos)
        words.append(w)
    return words


def typo_features(prose: str, collect=False):
    spell, allow = _get_spell()
    words = prose_words(prose)
    if not words:
        return ({"typos_per_100_words": 0.0, "n_prose_words": 0}, [])
    uniq = set(words)
    unknown = spell.unknown(list(uniq)) - allow
    typos = [w for w in words if w in unknown]
    return ({"typos_per_100_words": 100.0 * len(typos) / len(words),
             "n_prose_words": len(words)},
            sorted(set(typos)) if collect or True else [])


# ----------------------------------------------------- 7: theorem mentions
_THM = None


def _norm_name(s: str):
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'s", "").replace("'", "")
    s = re.sub(r"[-\s]+", " ", s).strip()
    return s


def _get_theorems():
    """Returns (prekey_set, fullname_list). prekey: name before 'theorem|lemma'."""
    global _THM
    if _THM is None:
        pre, full = set(), []
        p = RESOURCE_DIR / "theorem_names.txt"
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n = _norm_name(line)
            m = re.match(r"^(.*?)\s+(theorem|lemma)s?$", n)
            if m and m.group(1):
                pre.add(m.group(1))
            else:
                full.append(n)
        _THM = (pre, full)
    return _THM


def _lev(a, b, cap=2):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for ia, ca in enumerate(a, 1):
        cur = [ia]
        best = ia
        for ib, cb in enumerate(b, 1):
            v = min(prev[ib] + 1, cur[-1] + 1, prev[ib - 1] + (ca != cb))
            cur.append(v)
            best = min(best, v)
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


THM_CAND_RE = re.compile(
    r"((?:[A-Za-z][\w'’]*[-–—\s]+){1,5})(theorem|lemma)s?\b", re.I)
CAND_STOP = {"the", "a", "an", "by", "of", "this", "that", "his", "her", "its",
             "using", "use", "apply", "applying", "applied", "recall", "from",
             "via", "to", "and", "we", "is", "in", "famous", "celebrated",
             "classical", "well", "known", "so", "called", "following",
             "above", "previous", "second", "first", "last", "next", "same",
             "your", "my", "our", "their", "any", "every", "each", "some",
             "there", "then", "with", "for", "on", "or", "as", "be", "but",
             "no", "not", "now", "one", "two", "such", "see", "main", "key",
             "nice", "big", "little", "great", "good", "new", "old"}


def theorem_features(answer):
    pre, full = _get_theorems()
    n_mention = n_misspelled = 0
    arts = []
    seen_spans = set()
    for m in THM_CAND_RE.finditer(answer):
        words_raw = re.split(r"[-–—\s]+", m.group(1).strip())
        words_raw = [w for w in words_raw if w]
        # strip leading stopwords
        while words_raw and words_raw[0].lower() in CAND_STOP:
            words_raw.pop(0)
        if not words_raw:
            continue
        matched = False
        for j in range(len(words_raw)):
            cand = _norm_name(" ".join(words_raw[j:]))
            if cand in pre:
                n_mention += 1
                arts.append(f"mention: {' '.join(words_raw[j:])} {m.group(2)}")
                matched = True
                break
        if matched:
            continue
        # fuzzy: longest stripped candidate, require a capitalized word
        cand_words = words_raw
        while cand_words and cand_words[0].lower() in CAND_STOP:
            cand_words = cand_words[1:]
        if not cand_words or not any(w[0].isupper() for w in cand_words):
            continue
        cand = _norm_name(" ".join(cand_words))
        if len(cand) < 5:
            continue
        cap = 1 if len(cand) < 10 else 2
        for known in pre:
            d = _lev(cand, known, cap)
            if 1 <= d <= cap:
                n_misspelled += 1
                arts.append(f"misspelled: '{' '.join(cand_words)}' ~ '{known}' (d={d})")
                break
    # full-shaped names ("fundamental theorem of calculus"): substring search
    norm_ans = _norm_name(answer)
    for name in full:
        if name in norm_ans:
            n_mention += norm_ans.count(name)
            arts.append(f"mention(full): {name}")
    return ({"n_theorem_mentions": n_mention,
             "n_misspelled_theorem_mentions": n_misspelled}, arts)


# ----------------------------------------------------- 9: form contract
Q_COMPUTE_RE = re.compile(
    r"\b(evaluate|calculate|compute|find the value|how many|"
    r"what is the value|find all (?:solutions|values|roots)|solve for|"
    r"find the (?:sum|limit|integral|derivative|probability|number))\b", re.I)
Q_PROVE_RE = re.compile(
    r"\b(prove|show that|proof (?:of|that)|how (?:to|can i|do i) (?:prove|show)|"
    r"demonstrate that|verify that)\b", re.I)
Q_WHY_RE = re.compile(
    r"\b(why|intuition|intuitively|understand(?:ing)?|what does it mean|"
    r"how (?:do i|to|should i) (?:see|think|interpret))\b", re.I)
Q_REFERENCE_RE = re.compile(
    r"\breference request\b|\brecommend(?:ation)?s?\b|"
    r"\bwhere can i (?:find|read|learn)\b|"
    r"\b(?:good|any|which|what) (?:books?|textbooks?|references?|surveys?)\b|"
    r"\bliterature\b|\blecture notes (?:on|about|for)\b", re.I)

PROOF_MARK_RE = re.compile(
    r"\\square|\\blacksquare|\\Box\b|\bQ\.?E\.?D\b|∎|\bq\.e\.d\b|"
    r"we have (?:shown|proved|proven)|this (?:completes|finishes) the proof|"
    r"as desired|as required|which proves|hence proved|as was to be shown|"
    r"we conclude that|completing the proof", re.I)
EXAMPLE_MARK_RE = re.compile(
    r"\bfor example\b|\bfor instance\b|\be\.g\.|\bas an example\b|"
    r"\bconcretely\b|\bintuitively\b|\bthink of\b|\bconsider the case\b|"
    r"\bto see why\b|\bpicture\b|\bimagine\b", re.I)
ANSWER_IS_RE = re.compile(
    r"\\boxed|the answer is|the value is|final answer", re.I)
REF_ANSWER_RE = re.compile(
    r"https?://|\bbook\b|\btextbook\b|\bchapter\b|\bwikipedia\b|"
    r"\blecture notes\b|\bsee\s+\[", re.I)


def detect_qtype(question: str):
    for name, rx in (("prove", Q_PROVE_RE), ("compute", Q_COMPUTE_RE),
                     ("reference", Q_REFERENCE_RE), ("why", Q_WHY_RE)):
        if rx.search(question):
            return name
    return ""


def final_numeric(answer: str, a_display, a_inline):
    if ANSWER_IS_RE.search(answer):
        return True
    # last nonempty line ends with "= <number>" or is a bare number-ish math
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    tail = " ".join(lines[-2:]) if lines else ""
    return bool(re.search(r"=\s*\$?-?\d[\d,./]*\s*\.?\$?\s*$", tail)) or \
        bool(re.search(r"=\s*-?\d[\d,./]*\s*[.$]*\s*$", tail))


def form_contract_features(question, answer, a_display, a_inline):
    qtype = detect_qtype(question)
    boxed = bool(final_numeric(answer, a_display, a_inline))
    proof = bool(PROOF_MARK_RE.search(answer))
    example = bool(EXAMPLE_MARK_RE.search(answer))
    refok = bool(REF_ANSWER_RE.search(answer))
    met = ""
    if qtype == "compute":
        met = int(boxed)
    elif qtype == "prove":
        met = int(proof)
    elif qtype == "why":
        met = int(example)
    elif qtype == "reference":
        met = int(refok)
    return {"question_type": qtype, "contract_met": met,
            "has_boxed_or_final_numeric": int(boxed),
            "has_proof_markers": int(proof),
            "has_example_marker": int(example)}


# ----------------------------------------------------------- per-row driver
def lint_row(rec, verbose=False):
    """rec: dict with text, answer_id, question_id, judgement, split."""
    question, answer = split_qa(rec["text"])
    a_display, a_inline, a_prose = extract_math(answer)
    q_display, q_inline, _ = extract_math(question)

    feats = {"answer_id": rec["answer_id"], "question_id": rec["question_id"],
             "judgement": rec["judgement"], "split": rec["split"]}
    arts = defaultdict(list)

    # 1 + 2: sympy chains and literal arithmetic (fork-isolated)
    step_tasks, arith_tasks, pre_unparse, chain_arts = \
        build_chain_tasks(a_display, a_inline, a_prose)
    arts["chain_skips"] = chain_arts
    results = run_tasks_forked(step_tasks + arith_tasks)
    sv = Counter()
    for i, t in enumerate(step_tasks):
        v, note = results.get(i, ("MISSING", "killed"))
        sv[v] += 1
        if verbose:
            arts["steps"].append(
                f"[{v}] {t[1].strip()[:70]} {t[3]} {t[2].strip()[:70]} ({note})")
    av = Counter()
    for j, t in enumerate(arith_tasks):
        v, note = results.get(len(step_tasks) + j, ("MISSING", "killed"))
        av[v] += 1
        if verbose or v == "WRONG":
            arts["arith"].append(f"[{v}] {t[1].strip()[:60]} = {t[2].strip()[:60]}")
    n_checked = sv["VERIFIED"] + sv["REFUTED"] + sv["INCONCLUSIVE"]
    feats.update(
        n_steps_total=len(step_tasks) + pre_unparse,
        n_steps_checked=n_checked,
        n_steps_verified=sv["VERIFIED"],
        n_steps_refuted=sv["REFUTED"],
        n_steps_unparseable=pre_unparse + sv["PARSE_FAIL"] + sv["MISSING"],
        frac_steps_verified=(sv["VERIFIED"] / n_checked) if n_checked else "",
        n_arith=av["EQUAL"] + av["WRONG"],
        n_arith_wrong=av["WRONG"],
    )

    # 3
    feats.update(latex_parse_features(answer, a_display, a_inline))
    # 4
    f4, a4 = symbol_hygiene_features(q_display, q_inline, a_display, a_inline, answer)
    feats.update(f4); arts["symbols"] = [str(a4)] if (a4["undefined"] or a4["unused"]) else []
    # 5
    f5, a5 = dangling_refs_features(answer, question)
    feats.update(f5); arts["refs"] = a5
    # 6
    f6, a6 = typo_features(a_prose)
    feats.update(f6); arts["typos"] = a6
    # 7
    f7, a7 = theorem_features(answer)
    feats.update(f7); arts["theorems"] = a7
    # 9
    feats.update(form_contract_features(question, answer, a_display, a_inline))
    return feats, arts


def _worker(rec):
    try:
        feats, _ = lint_row(rec, verbose=False)
        return feats
    except Exception as e:
        return {"answer_id": rec["answer_id"], "question_id": rec["question_id"],
                "judgement": rec["judgement"], "split": rec["split"],
                "lint_error": f"{type(e).__name__}: {e}"[:120]}


def _init_worker():
    # warm the lark latex parser + spellchecker + theorem list so forked
    # children inherit them copy-on-write
    import warnings
    warnings.filterwarnings("ignore")  # lark transformer SymPyDeprecationWarning
    from sympy.parsing.latex import parse_latex
    try:
        parse_latex("1+1", backend="lark")
    except Exception:
        pass
    _get_spell()
    _get_theorems()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ----------------------------------------------------------- 8: near dup
def shingles(s, k=5):
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + k] for i in range(max(0, len(s) - k + 1))}


def near_dup_column(df):
    """max 5-gram char-shingle jaccard to sibling answers (same question)."""
    out = {}
    for qid, grp in df.groupby("question_id"):
        if len(grp) < 2:
            for aid in grp.answer_id:
                out[aid] = 0.0
            continue
        sh = {}
        for aid, text in zip(grp.answer_id, grp.text):
            _, ans = split_qa(text)
            sh[aid] = shingles(ans)
        aids = list(sh)
        best = {a: 0.0 for a in aids}
        for x in range(len(aids)):
            for y in range(x + 1, len(aids)):
                a, b = sh[aids[x]], sh[aids[y]]
                if not a or not b:
                    continue
                j = len(a & b) / len(a | b)
                best[aids[x]] = max(best[aids[x]], j)
                best[aids[y]] = max(best[aids[y]], j)
        out.update(best)
    return out


# ----------------------------------------------------------------- modes
FIELDNAMES = ["answer_id", "question_id", "judgement", "split",
              "n_steps_total", "n_steps_checked", "n_steps_verified",
              "n_steps_refuted", "n_steps_unparseable", "frac_steps_verified",
              "n_arith", "n_arith_wrong",
              "n_latex_errors", "frac_blocks_with_errors", "n_math_blocks",
              "n_undefined_symbols", "n_unused_definitions",
              "n_refs", "n_dangling",
              "typos_per_100_words", "n_prose_words",
              "n_theorem_mentions", "n_misspelled_theorem_mentions",
              "max_jaccard_to_sibling",
              "question_type", "contract_met", "has_boxed_or_final_numeric",
              "has_proof_markers", "has_example_marker", "lint_error"]


def load_df(path):
    import pandas as pd
    return pd.read_csv(path)


def cmd_sample(args):
    df = load_df(args.data)
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(df)), args.sample)
    sub = df.iloc[idx]
    _init_worker()
    nd = {}
    for _, row in sub.iterrows():
        rec = row.to_dict()
        t0 = time.time()
        feats, arts = lint_row(rec, verbose=True)
        print("=" * 100)
        print(f"answer_id={rec['answer_id']} judgement={rec['judgement']} "
              f"({time.time()-t0:.1f}s)")
        _, ans = split_qa(rec["text"])
        print("ANSWER HEAD:", re.sub(r"\s+", " ", ans)[:220])
        print("FEATS:", {k: v for k, v in feats.items()
                         if k not in ("answer_id", "question_id", "judgement", "split")})
        for k, lst in arts.items():
            for item in lst[:12]:
                print(f"  {k}: {item}")


def cmd_allowlist(args):
    df = load_df(args.data)
    rng = random.Random(0)
    idx = rng.sample(range(len(df)), min(args.build_allowlist, len(df)))
    from spellchecker import SpellChecker
    spell = SpellChecker()
    docfreq = Counter()
    for n, i in enumerate(idx):
        _, ans = split_qa(df.iloc[i]["text"])
        _, _, prose = extract_math(ans)
        uniq = set(prose_words(prose))
        unknown = spell.unknown(list(uniq))
        for w in unknown:
            docfreq[w] += 1
        if (n + 1) % 500 == 0:
            print(f"...{n+1}", file=sys.stderr)
    for w, c in docfreq.most_common(400):
        print(f"{c}\t{w}")


def cmd_run(args):
    import csv
    import multiprocessing as mp
    import pandas as pd
    df = load_df(args.data)
    print(f"loaded {len(df):,} rows", flush=True)

    done = set()
    if os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out, usecols=["answer_id"])
            done = set(prev.answer_id.astype(int))
            print(f"resuming: {len(done):,} rows already in {args.out}", flush=True)
        except Exception:
            pass

    print("computing near-dup jaccard...", flush=True)
    nd = near_dup_column(df)

    recs = [r for r in df[["text", "answer_id", "question_id",
                           "judgement", "split"]].to_dict("records")
            if int(r["answer_id"]) not in done]
    print(f"{len(recs):,} rows to lint", flush=True)

    mode = "a" if done else "w"
    fout = open(args.out, mode, newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES, extrasaction="ignore")
    if mode == "w":
        writer.writeheader()
    t0 = time.time()
    n_done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  maxtasksperchild=400) as pool:
        for feats in pool.imap_unordered(_worker, recs, chunksize=16):
            feats["max_jaccard_to_sibling"] = round(
                nd.get(feats["answer_id"], 0.0), 4)
            if isinstance(feats.get("frac_steps_verified"), float):
                feats["frac_steps_verified"] = round(feats["frac_steps_verified"], 4)
            if isinstance(feats.get("typos_per_100_words"), float):
                feats["typos_per_100_words"] = round(feats["typos_per_100_words"], 3)
            if isinstance(feats.get("frac_blocks_with_errors"), float):
                feats["frac_blocks_with_errors"] = round(feats["frac_blocks_with_errors"], 4)
            writer.writerow(feats)
            n_done += 1
            if n_done % args.checkpoint_every == 0:
                fout.flush()
                os.fsync(fout.fileno())
                rate = n_done / (time.time() - t0)
                eta = (len(recs) - n_done) / max(rate, 1e-9) / 60
                print(f"{n_done:,}/{len(recs):,} rows  "
                      f"({rate:.1f}/s, ETA {eta:.0f} min)", flush=True)
    fout.flush()
    fout.close()
    print(f"done: {n_done:,} rows in {(time.time()-t0)/60:.1f} min -> {args.out}")


NUMERIC_FEATS = [
    "n_steps_total", "n_steps_checked", "n_steps_verified", "n_steps_refuted",
    "n_steps_unparseable", "frac_steps_verified", "n_arith", "n_arith_wrong",
    "n_latex_errors", "frac_blocks_with_errors",
    "n_undefined_symbols", "n_unused_definitions", "n_refs", "n_dangling",
    "typos_per_100_words", "n_theorem_mentions",
    "n_misspelled_theorem_mentions", "max_jaccard_to_sibling",
    "contract_met", "has_boxed_or_final_numeric", "has_proof_markers",
    "has_example_marker"]


def cmd_analyze(args):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat = pd.read_csv(args.analyze)
    print(f"feature rows: {len(feat):,}")
    if "lint_error" in feat.columns:
        ne = feat.lint_error.notna().sum()
        print(f"lint errors: {ne}")
        feat = feat[feat.lint_error.isna()]
    feat = feat.drop_duplicates(subset="answer_id", keep="last")

    tr = feat[feat.split == "train"]
    print(f"train: {len(tr):,}  y-balance {tr.judgement.mean():.3f}")
    print("\n## Per-feature AUC (train split) + coverage\n")
    print("| feature | AUC (train) | coverage (defined) | nonzero |")
    print("|---|---|---|---|")
    for c in NUMERIC_FEATS:
        x = pd.to_numeric(tr[c], errors="coerce")
        cov = x.notna().mean()
        m = x.notna()
        nz = (x[m] != 0).mean() if m.any() else 0.0
        if m.sum() > 100 and x[m].std() > 0:
            auc = roc_auc_score(tr.judgement[m], x[m])
        else:
            auc = float("nan")
        print(f"| {c} | {auc:.4f} | {cov:.1%} | {nz:.1%} |")

    # combined LR: train -> eval/test
    use = [c for c in NUMERIC_FEATS]
    X = {}
    for split in ("train", "eval", "test"):
        sub = feat[feat.split == split]
        cols = []
        for c in use:
            x = pd.to_numeric(sub[c], errors="coerce")
            cols.append(x.fillna(0).values)
            cols.append(x.isna().astype(float).values)  # defined-indicator
        X[split] = (np.column_stack(cols), sub.judgement.values)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(*X["train"])
    print("\n## Combined LogisticRegression (fit on train)\n")
    print("| split | AUC | n |")
    print("|---|---|---|")
    for split in ("train", "eval", "test"):
        Xs, ys = X[split]
        auc = roc_auc_score(ys, clf.predict_proba(Xs)[:, 1])
        print(f"| {split} | {auc:.4f} | {len(ys):,} |")
    print("\n(question-only TF-IDF+LR floor on v3.3: 0.461)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--build-allowlist", type=int)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--analyze")
    args = ap.parse_args()
    if args.analyze:
        cmd_analyze(args)
    elif args.build_allowlist:
        cmd_allowlist(args)
    elif args.sample:
        cmd_sample(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()

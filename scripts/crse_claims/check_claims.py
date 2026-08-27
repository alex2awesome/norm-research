"""
check_claims.py — verify CR.SE claims with code-domain tools (the analog of
the Math.SE sympy claim-checker).

Verifiers per claim_type (Python-only for the pilot):

  - complexity   : static AST analysis of inline_code (or the answer's first
                   code block falling back to OP code in the question).
                   We classify nesting depth of for/while + recursion and
                   compare to the claimed O(...).

  - behavior     : run inline_code (or the snippet referenced in the quote)
                   in a sandboxed subprocess. The pilot uses heuristic input
                   synthesis from the claim text (e.g. "empty list" -> []).
                   Sandbox: tempdir, 5 s timeout, no network, no env.

  - api_fact     : assert the smallest assertion in a bare interpreter, e.g.
                   `assert {}.get('x') is None`. We extract the assertion
                   from inline_code if possible; else mark UNCHECKABLE.

  - equivalence /
    improvement  : run BOTH snippets on synthetic inputs and compare outputs.
                   "improvement" 'faster' claims: timeit at small n; if not
                   self-contained, mark UNCHECKABLE.

  - other        : UNCHECKABLE (we don't trust extraction here).

For non-Python languages: UNCHECKABLE with language tag.

Verdict schema (per claim):
  {
    "verdict": "SUPPORTED" | "REFUTED" | "UNCHECKABLE",
    "checker": <name>,
    "evidence": <freeform dict, e.g. stdout, exit code, AST stats>,
    "binding_cue_downgrade": bool   # True if REFUTED was downgraded to
                                    # UNCHECKABLE because the source quote
                                    # starts with a binding cue (if/let/...)
  }

Usage:
  python3 scripts/crse_claims/check_claims.py \
      --claims-flat outputs/crse_claims_pilot/claims_flat.jsonl \
      --pool       datasets/code-review/crse_balanced_v1/crse_v1_propensity_balanced.csv.gz \
      --out-verdicts outputs/crse_claims_pilot/verdicts.jsonl \
      --out-features outputs/crse_claims_pilot/answer_features.jsonl
"""
from __future__ import annotations
import argparse, ast, csv, gzip, json, os, re, shutil, subprocess, sys, tempfile, textwrap
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from code_blocks import (extract_code_blocks, blocks_for_language,
                         guess_language)  # noqa: E402


# -------------------- code extraction ----------------------------------------

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`([^`]{2,200})`")


def extract_first_code_block(text: str) -> str | None:
    if not text:
        return None
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: 4-space indented lines
    lines = text.split("\n")
    block = []
    for L in lines:
        if L.startswith("    "):
            block.append(L[4:])
        elif block and not L.strip():
            block.append("")
        elif block:
            break
    code = "\n".join(block).strip()
    return code or None


# -------------------- complexity checker -------------------------------------

def loop_nesting_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return -1
    max_depth = 0

    def walk(node, depth):
        nonlocal max_depth
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            depth += 1
            max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            walk(child, depth)

    walk(tree, 0)
    return max_depth


def detect_recursion(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    if sub.func.id == fn.name:
                        return True
    return False


BIG_O_RE = re.compile(r"O\s*\(\s*([^)]+?)\s*\)", re.IGNORECASE)


# Word-form complexity phrases ("linear time", "quadratic") — the pilot's
# no_big_o_in_claim failures were dominated by these.
WORD_O_PATTERNS = [
    (re.compile(r"\bn\s*log\s*n\b|\blinearithmic\b", re.I),          "n log n"),
    (re.compile(r"\bconstant[\s-]+(?:time|complexity)\b", re.I),      "1"),
    (re.compile(r"\blog(?:arithmic)?[\s-]+(?:time|complexity)\b", re.I), "log n"),
    (re.compile(r"\blinear\b(?![\s-]*(?:algebra|regression))", re.I), "n"),
    (re.compile(r"\bquadratic\b", re.I),                              "n^2"),
    (re.compile(r"\bcubic\b", re.I),                                  "n^3"),
    (re.compile(r"\bexponential[\s-]*(?:time|complexity)\b", re.I),   "2^n"),
]


def parse_big_o(text: str) -> str | None:
    """Extract canonical Big-O label from a quote/claim. Returns one of
    {'1','log n','n','n log n','n^2','n^3','2^n','other'} or None.
    Handles O(...) notation (incl. unicode superscripts) and word forms
    ("linear time", "quadratic")."""
    text = (text or "").replace("²", "^2").replace("³", "^3")
    m = BIG_O_RE.search(text)
    if not m:
        for pat, label in WORD_O_PATTERNS:
            if pat.search(text):
                return label
        return None
    expr = m.group(1).strip().lower().replace(" ", "")
    expr = expr.replace("·", "*").replace("×", "*")
    if expr in ("1",):                          return "1"
    if expr in ("logn", "log(n)", "log_n"):      return "log n"
    if expr in ("n",):                            return "n"
    if expr in ("nlogn", "n*logn", "nlog(n)"):    return "n log n"
    if expr in ("n^2", "n**2", "n2", "n*n"):      return "n^2"
    if expr in ("n^3", "n**3", "n3"):            return "n^3"
    if expr in ("2^n", "2**n"):                   return "2^n"
    return "other"


# Big-O rank order for the lower-bound refutation rule.
_BIG_O_ORDER = {"1": 0, "log n": 1, "n": 2, "n log n": 3, "n^2": 4,
                "n^3": 5, "2^n": 6}


def _complexity_verdict(claimed_o: str, depth: int, checker: str) -> dict:
    """Shared verdict rule for static complexity checks.

    Visible loop nesting is a LOWER BOUND on work (calls inside the loop can
    only add hidden complexity). So:
      - claimed compatible with visible depth -> SUPPORTED (consistency)
      - claimed BELOW the visible lower bound  -> REFUTED
      - claimed ABOVE the visible depth        -> UNCHECKABLE (the extra
        complexity may hide inside library calls / regex — the pilot-v2
        manual audit showed these refutations are mostly false)
      - claimed unparseable ('other')          -> UNCHECKABLE
    """
    if claimed_o == "other":
        return {"verdict": "UNCHECKABLE", "checker": checker,
                "evidence": {"reason": "unparsed_big_o", "depth": depth}}
    compatible = {
        0: {"1", "n"},
        1: {"n", "n log n"},
        2: {"n^2"},
        3: {"n^3"},
    }.get(depth, set())
    if not compatible:
        return {"verdict": "UNCHECKABLE", "checker": checker,
                "evidence": {"reason": "depth_too_deep", "depth": depth,
                             "claimed": claimed_o}}
    if claimed_o in compatible:
        return {"verdict": "SUPPORTED", "checker": checker,
                "evidence": {"depth": depth, "claimed": claimed_o,
                             "compatible": sorted(compatible)}}
    lower_bound = min(_BIG_O_ORDER[x] for x in compatible)
    if _BIG_O_ORDER.get(claimed_o, 99) < lower_bound:
        return {"verdict": "REFUTED", "checker": checker,
                "evidence": {"depth": depth, "claimed": claimed_o,
                             "compatible": sorted(compatible),
                             "rule": "claimed_below_visible_lower_bound"}}
    return {"verdict": "UNCHECKABLE", "checker": checker,
            "evidence": {"reason": "possible_hidden_complexity", "depth": depth,
                         "claimed": claimed_o, "compatible": sorted(compatible)}}


def check_complexity(claim, code: str | None):
    """Compare claimed Big-O to AST loop-nesting depth.

    Heuristic mapping:
      depth=0, no recursion        -> O(1) or O(n)
      depth=1                      -> O(n) or O(n log n)
      depth=2                      -> O(n^2)
      depth=3                      -> O(n^3)
      recursion                    -> can be O(2^n) or O(n log n) — UNCHECKABLE
    """
    claimed_o = parse_big_o(claim.get("claim_text") or "") or \
                parse_big_o(claim.get("source_quote") or "")
    if claimed_o is None:
        return {"verdict": "UNCHECKABLE", "checker": "complexity_static",
                "evidence": {"reason": "no_big_o_in_claim"}}
    if not code:
        return {"verdict": "UNCHECKABLE", "checker": "complexity_static",
                "evidence": {"reason": "no_code_snippet", "claimed": claimed_o}}
    depth = loop_nesting_depth(code)
    rec = detect_recursion(code)
    if depth < 0:
        return {"verdict": "UNCHECKABLE", "checker": "complexity_static",
                "evidence": {"reason": "ast_parse_failed", "claimed": claimed_o}}
    if rec:
        return {"verdict": "UNCHECKABLE", "checker": "complexity_static",
                "evidence": {"reason": "recursion_present",
                             "claimed": claimed_o, "depth": depth}}
    return _complexity_verdict(claimed_o, depth, "complexity_static")


# -------------------- sandboxed subprocess runner ----------------------------

SANDBOX_RUNNER = textwrap.dedent("""
    import sys, json, traceback
    src = sys.stdin.read()
    glb = {"__name__": "__main__"}
    out = {"ok": False, "stdout": "", "stderr": "", "exception": None}
    import io, contextlib
    sout = io.StringIO(); serr = io.StringIO()
    try:
        with contextlib.redirect_stdout(sout), contextlib.redirect_stderr(serr):
            exec(compile(src, "<sandbox>", "exec"), glb)
        out["ok"] = True
    except BaseException as e:
        out["exception"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    out["stdout"] = sout.getvalue()
    out["stderr"] = serr.getvalue()
    print("___SANDBOX_RESULT___" + json.dumps(out))
""").strip()


def run_in_sandbox(code: str, timeout: float = 5.0) -> dict:
    """Run `code` in a subprocess. Returns {ok, stdout, stderr, exception,
    timeout, returncode}. Sandbox = temp cwd, no network env tweaks, no
    PYTHONPATH inheritance other than the subprocess's default."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    with tempfile.TemporaryDirectory() as td:
        try:
            res = subprocess.run(
                [sys.executable, "-c", SANDBOX_RUNNER],
                input=code, capture_output=True, text=True,
                cwd=td, env=env, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "timeout": True, "stdout": "", "stderr": "",
                    "exception": "TimeoutExpired", "returncode": None}
        out = {"timeout": False, "returncode": res.returncode,
               "raw_stdout": res.stdout, "raw_stderr": res.stderr,
               "ok": False, "stdout": "", "stderr": "", "exception": None}
        marker = "___SANDBOX_RESULT___"
        if marker in res.stdout:
            try:
                payload = json.loads(res.stdout.split(marker, 1)[1].strip())
                out.update(payload)
            except Exception:
                pass
        else:
            out["exception"] = "no_sandbox_marker"
        return out


# -------------------- behavior checker ---------------------------------------

# Very simple input-synthesis heuristics from claim text. The pilot is
# deliberately small and we deliberately mark a lot of things UNCHECKABLE.
BEHAVIOR_PATTERNS = [
    (re.compile(r"empty\s+(?:list|array|sequence|iter|input|collection)", re.I), "[]"),
    (re.compile(r"empty\s+(?:dict|mapping)", re.I),                    "{}"),
    (re.compile(r"empty\s+set\b", re.I),                                "set()"),
    (re.compile(r"empty\s+(?:string|str)", re.I),                       "''"),
    (re.compile(r"none\b|null\b", re.I),                                "None"),
    (re.compile(r"negative\s+(?:number|int|input|value)", re.I),        "-1"),
    (re.compile(r"single[- ]element|one[- ]element", re.I),             "[1]"),
    (re.compile(r"zero\b", re.I),                                       "0"),
]

EXCEPTION_RE = re.compile(
    r"(?P<exc>ValueError|TypeError|KeyError|IndexError|AttributeError|"
    r"NameError|ZeroDivisionError|RecursionError|StopIteration|"
    r"RuntimeError|OverflowError|MemoryError)",
)


def find_top_level_callable(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def check_behavior(claim, code: str | None):
    if not code:
        return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic",
                "evidence": {"reason": "no_code_snippet"}}
    # 1) detect an input pattern in the claim text
    text = (claim.get("claim_text") or "") + " " + (claim.get("source_quote") or "")
    arg_repr = None
    for pat, val in BEHAVIOR_PATTERNS:
        if pat.search(text):
            arg_repr = val
            break
    if arg_repr is None:
        return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic",
                "evidence": {"reason": "no_synth_input_pattern"}}
    # 2) detect a claimed exception (if any)
    exc = None
    em = EXCEPTION_RE.search(text)
    if em:
        exc = em.group("exc")
    # 3) find a top-level callable
    fn = find_top_level_callable(code)
    if not fn:
        return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic",
                "evidence": {"reason": "no_top_level_callable"}}
    test_src = code + f"\n\n_RESULT_ = {fn}({arg_repr})\nprint('RESULT=', _RESULT_)\n"
    res = run_in_sandbox(test_src, timeout=5.0)
    evidence = {"fn": fn, "arg": arg_repr,
                "claimed_exception": exc,
                "ok": res.get("ok"),
                "exception": res.get("exception"),
                "stdout_head": (res.get("stdout") or "")[:200]}
    # Verdict logic
    if exc:
        actual = res.get("exception") or ""
        if exc in actual:
            return {"verdict": "SUPPORTED", "checker": "behavior_dynamic",
                    "evidence": evidence}
        if res.get("ok"):
            return {"verdict": "REFUTED", "checker": "behavior_dynamic",
                    "evidence": {**evidence, "note": "ran cleanly; no exception"}}
        return {"verdict": "REFUTED", "checker": "behavior_dynamic",
                "evidence": {**evidence,
                             "note": f"raised different exception"}}
    # No claimed exception -> "breaks on empty" without naming exc
    if "break" in text.lower() or "fail" in text.lower() or "error" in text.lower():
        if not res.get("ok"):
            return {"verdict": "SUPPORTED", "checker": "behavior_dynamic",
                    "evidence": evidence}
        return {"verdict": "REFUTED", "checker": "behavior_dynamic",
                "evidence": {**evidence,
                             "note": "claimed-to-break but ran cleanly"}}
    return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic",
            "evidence": {**evidence, "reason": "no_clear_verdict_rule"}}


# -------------------- api_fact checker ---------------------------------------

# Hand-curated mini-library of common API facts. The model often emits these
# verbatim from documentation; we run the minimal probe.
API_FACT_PROBES = [
    # (regex on claim_text/source_quote, probe code, expected_ok)
    (re.compile(r"\bdict\.get\b.*default.*None|\.get\([^)]*\).*returns?\s*None",
                re.IGNORECASE),
     "assert {}.get('x') is None", True),
    (re.compile(r"\blist\.sort\b.*in.?place|sorted\(\).*returns?\s+(?:a\s+)?new",
                re.IGNORECASE),
     "x = [3,1,2]; y = x.sort(); assert y is None and x == [1,2,3]", True),
    (re.compile(r"\brange\b.*lazy|range\b.*generator|range\b.*not.*list",
                re.IGNORECASE),
     "assert not isinstance(range(3), list)", True),
    (re.compile(r"\bset\b.*unordered|\bset\b.*no\s+order", re.IGNORECASE),
     "assert hasattr(set(), 'add')", True),  # cheap proxy
    (re.compile(r"\bstr\b.*immutable|strings?\s+are\s+immutable", re.IGNORECASE),
     "try:\n  'x'[0] = 'y'\n  raise AssertionError('mutable')\nexcept TypeError:\n  pass",
     True),
    (re.compile(r"\.append\(.*returns?\s+None|append\(\).*returns?\s+None",
                re.IGNORECASE),
     "assert [].append(1) is None", True),
    (re.compile(r"\.copy\(\).*shallow|shallow\s+copy", re.IGNORECASE),
     "a=[[1]]; b=a.copy(); b[0].append(2); assert a==[[1,2]]", True),
]


def check_api_fact(claim):
    text = (claim.get("claim_text") or "") + " " + (claim.get("source_quote") or "")
    inline = claim.get("inline_code") or ""
    for pat, probe, expected_ok in API_FACT_PROBES:
        if pat.search(text) or (inline and pat.search(inline)):
            res = run_in_sandbox(probe, timeout=2.0)
            v = "SUPPORTED" if res.get("ok") == expected_ok else "REFUTED"
            return {"verdict": v, "checker": "api_fact_probe",
                    "evidence": {"probe": probe, "ok": res.get("ok"),
                                 "exception": res.get("exception")}}
    return {"verdict": "UNCHECKABLE", "checker": "api_fact_probe",
            "evidence": {"reason": "no_matching_probe"}}


# -------------------- multi-language: shared static helpers -------------------

_CSTYLE_STRING_RE = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")
_CSTYLE_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _strip_cstyle_noise(code: str) -> str:
    code = _CSTYLE_COMMENT_RE.sub(" ", code or "")
    return _CSTYLE_STRING_RE.sub('""', code)


def cstyle_loop_nesting_depth(code: str) -> int:
    """Brace-tracked max loop-nesting depth for JS/Java/C#. Heuristic: loops
    whose body is a braced block; single-statement nested loops are missed
    (conservative — undercounts depth, never overcounts)."""
    code = _strip_cstyle_noise(code)
    depth = 0
    loop_bodies: list[int] = []   # brace depth of each open loop body
    max_loops = 0
    pending_loop = False
    for m in re.finditer(r"\b(?:for|while)\b|[{}]", code):
        t = m.group(0)
        if t in ("for", "while"):
            pending_loop = True
        elif t == "{":
            depth += 1
            if pending_loop:
                loop_bodies.append(depth)
                max_loops = max(max_loops, len(loop_bodies))
                pending_loop = False
        else:
            if loop_bodies and loop_bodies[-1] == depth:
                loop_bodies.pop()
            depth = max(0, depth - 1)
    return max_loops


def cstyle_detect_recursion(code: str) -> bool:
    code = _strip_cstyle_noise(code)
    names = re.findall(r"\bfunction\s+(\w+)\s*\(", code)          # JS
    names += re.findall(r"\b(?:void|int|long|double|float|bool|boolean|string|"
                        r"String|var)\s+(\w+)\s*\([^;{]*\)\s*\{", code)  # Java/C#
    for n in set(names):
        body_calls = len(re.findall(rf"\b{re.escape(n)}\s*\(", code))
        if body_calls >= 2:   # definition site + at least one call
            return True
    return False


def check_complexity_cstyle(claim, code: str | None, lang: str):
    claimed_o = parse_big_o(claim.get("claim_text") or "") or \
                parse_big_o(claim.get("source_quote") or "")
    if claimed_o is None:
        return {"verdict": "UNCHECKABLE", "checker": f"complexity_static_{lang}",
                "evidence": {"reason": "no_big_o_in_claim"}}
    if not code:
        return {"verdict": "UNCHECKABLE", "checker": f"complexity_static_{lang}",
                "evidence": {"reason": "no_code_snippet", "claimed": claimed_o}}
    if cstyle_detect_recursion(code):
        return {"verdict": "UNCHECKABLE", "checker": f"complexity_static_{lang}",
                "evidence": {"reason": "recursion_present", "claimed": claimed_o}}
    depth = cstyle_loop_nesting_depth(code)
    return _complexity_verdict(claimed_o, depth, f"complexity_static_{lang}")


# -------------------- JavaScript: Node sandbox ---------------------------------

NODE_BIN = shutil.which("node") or "/usr/bin/node"

JS_SANDBOX_PRELUDE = """'use strict';
process.on('uncaughtException', (e) => {
  console.log('___JS_EXC___' + e.constructor.name + ': ' + e.message);
  process.exit(3);
});
"""


def run_js_in_sandbox(code: str, timeout: float = 5.0) -> dict:
    """Run JS in a node subprocess. Same isolation pattern as the Python
    sandbox: temp cwd, stripped env, hard timeout, per-claim subprocess."""
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
           "NODE_OPTIONS": "--max-old-space-size=256"}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "snippet.js"
        src.write_text(JS_SANDBOX_PRELUDE + code, encoding="utf-8")
        try:
            res = subprocess.run([NODE_BIN, str(src)], capture_output=True,
                                 text=True, cwd=td, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "timeout": True, "exception": "TimeoutExpired",
                    "stdout": "", "stderr": ""}
        exc = None
        for line in (res.stdout or "").splitlines():
            if line.startswith("___JS_EXC___"):
                exc = line[len("___JS_EXC___"):]
        if exc is None and res.returncode != 0:
            m = re.search(r"^(\w*Error): (.*)$", res.stderr or "", re.M)
            exc = f"{m.group(1)}: {m.group(2)}" if m else f"exit_{res.returncode}"
        return {"ok": res.returncode == 0, "timeout": False,
                "returncode": res.returncode, "exception": exc,
                "stdout": (res.stdout or "")[:500], "stderr": (res.stderr or "")[:500]}


def js_syntax_ok(code: str, timeout: float = 5.0) -> tuple[bool, str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "snippet.js"
        src.write_text(code, encoding="utf-8")
        try:
            res = subprocess.run([NODE_BIN, "--check", str(src)],
                                 capture_output=True, text=True, cwd=td,
                                 env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "TimeoutExpired"
        return res.returncode == 0, (res.stderr or "")[:300]


JS_FN_RES = [
    re.compile(r"\bfunction\s+(\w+)\s*\("),
    re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"),
    re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*\w+\s*=>"),
    re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*function\b"),
]

JS_BEHAVIOR_PATTERNS = [
    (re.compile(r"empty\s+(?:array|list)", re.I),   "[]"),
    (re.compile(r"empty\s+(?:object|map)", re.I),   "{}"),
    (re.compile(r"empty\s+(?:string|str)", re.I),    "''"),
    (re.compile(r"\bundefined\b", re.I),             "undefined"),
    (re.compile(r"\bnull\b", re.I),                  "null"),
    (re.compile(r"negative\s+(?:number|int|input|value)", re.I), "-1"),
    (re.compile(r"\bzero\b", re.I),                  "0"),
    (re.compile(r"\bNaN\b"),                          "NaN"),
]

JS_EXCEPTION_RE = re.compile(
    r"(?P<exc>TypeError|RangeError|ReferenceError|SyntaxError|EvalError|URIError)")


def find_js_callable(code: str) -> str | None:
    for pat in JS_FN_RES:
        m = pat.search(_strip_cstyle_noise(code))
        if m:
            return m.group(1)
    return None


def check_behavior_js(claim, code: str | None):
    if not code:
        return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic_js",
                "evidence": {"reason": "no_code_snippet"}}
    text = (claim.get("claim_text") or "") + " " + (claim.get("source_quote") or "")
    arg_repr = None
    for pat, val in JS_BEHAVIOR_PATTERNS:
        if pat.search(text):
            arg_repr = val
            break
    if arg_repr is None:
        return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic_js",
                "evidence": {"reason": "no_synth_input_pattern"}}
    exc = None
    em = JS_EXCEPTION_RE.search(text)
    if em:
        exc = em.group("exc")
    fn = find_js_callable(code)
    if not fn:
        return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic_js",
                "evidence": {"reason": "no_top_level_callable"}}
    ok_syntax, syn_err = js_syntax_ok(code)
    if not ok_syntax:
        return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic_js",
                "evidence": {"reason": "syntax_invalid", "stderr": syn_err}}
    test_src = code + f"\n\nconst _RESULT_ = {fn}({arg_repr});\n" \
                      f"console.log('RESULT=', _RESULT_);\n"
    res = run_js_in_sandbox(test_src, timeout=5.0)
    evidence = {"fn": fn, "arg": arg_repr, "claimed_exception": exc,
                "ok": res.get("ok"), "exception": res.get("exception"),
                "stdout_head": (res.get("stdout") or "")[:200]}
    if exc:
        actual = res.get("exception") or ""
        if exc in actual:
            return {"verdict": "SUPPORTED", "checker": "behavior_dynamic_js",
                    "evidence": evidence}
        if res.get("ok"):
            return {"verdict": "REFUTED", "checker": "behavior_dynamic_js",
                    "evidence": {**evidence, "note": "ran cleanly; no exception"}}
        return {"verdict": "REFUTED", "checker": "behavior_dynamic_js",
                "evidence": {**evidence, "note": "raised different exception"}}
    if "break" in text.lower() or "fail" in text.lower() or "error" in text.lower():
        if not res.get("ok"):
            return {"verdict": "SUPPORTED", "checker": "behavior_dynamic_js",
                    "evidence": evidence}
        return {"verdict": "REFUTED", "checker": "behavior_dynamic_js",
                "evidence": {**evidence, "note": "claimed-to-break but ran cleanly"}}
    return {"verdict": "UNCHECKABLE", "checker": "behavior_dynamic_js",
            "evidence": {**evidence, "reason": "no_clear_verdict_rule"}}


# -------------------- Java / C#: compiler-only validity ------------------------

JAVAC_BIN = shutil.which("javac")
MCS_BIN = (os.environ.get("MCS_BIN")
           or shutil.which("mcs")
           or ("/lfs/skampere3/0/alexspan/envs/mono/bin/mcs"
               if os.path.exists("/lfs/skampere3/0/alexspan/envs/mono/bin/mcs")
               else None))

COMPILE_CLAIM_RE = re.compile(
    r"(?:won'?t|will\s+not|does\s*n[o']t|doesn'?t|cannot|can'?t|fail(?:s)?\s+to)\s+"
    r"(?:compile|build)|compil(?:e|ation)\s+error|syntax\s+error|invalid\s+syntax",
    re.IGNORECASE)
COMPILES_OK_CLAIM_RE = re.compile(
    r"(?:compiles?|builds?)\s+(?:fine|cleanly|ok|correctly|without)", re.IGNORECASE)


def _wrap_java(code: str) -> tuple[str, str]:
    """Return (filename_stem, full_source) — javac needs file == public class."""
    m = re.search(r"\bpublic\s+(?:final\s+|abstract\s+)?class\s+(\w+)", code)
    if m:
        return m.group(1), code
    if re.search(r"\bclass\s+\w+", code):
        return "Snippet", code
    # method/statement fragment -> wrap
    if re.search(r"\b(?:void|int|long|double|float|boolean|String)\s+\w+\s*\(", code):
        return "Wrapper", "public class Wrapper {\n" + code + "\n}"
    return "Wrapper", ("public class Wrapper { public static void m() throws Exception {\n"
                       + code + "\n} }")


def java_compile_ok(code: str, timeout: float = 20.0) -> tuple[bool | None, str]:
    if not JAVAC_BIN:
        return None, "javac_not_available"
    stem, src_text = _wrap_java(code)
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{stem}.java"
        src.write_text(src_text, encoding="utf-8")
        try:
            res = subprocess.run([JAVAC_BIN, "-nowarn", src.name],
                                 capture_output=True, text=True, cwd=td,
                                 env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "TimeoutExpired"
        return res.returncode == 0, (res.stderr or "")[:400]


def _wrap_csharp(code: str) -> str:
    if re.search(r"\bclass\s+\w+", code):
        return code if "using System" in code else "using System;\n" + code
    if re.search(r"\b(?:void|int|long|double|float|bool|string)\s+\w+\s*\(", code):
        return "using System;\npublic class Wrapper {\n" + code + "\n}"
    return ("using System;\npublic class Wrapper { public static void M() {\n"
            + code + "\n} }")


def csharp_compile_ok(code: str, timeout: float = 20.0) -> tuple[bool | None, str]:
    if not MCS_BIN:
        return None, "csharp_compiler_not_available"
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "Snippet.cs"
        src.write_text(_wrap_csharp(code), encoding="utf-8")
        try:
            res = subprocess.run([MCS_BIN, "-t:library", "-nowarn:0219,0168,0414",
                                  src.name], capture_output=True, text=True,
                                 cwd=td, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "TimeoutExpired"
        err = ((res.stdout or "") + (res.stderr or ""))[:400]
        return res.returncode == 0, err


def check_compiled_lang(claim, code: str | None, lang: str):
    """Java/C#: compiler-only validity. Directly decides compile/syntax
    claims; complexity goes to the c-style static checker; everything else is
    UNCHECKABLE but carries compile evidence."""
    ctype = claim.get("claim_type")
    if ctype == "complexity":
        return check_complexity_cstyle(claim, code, lang)
    text = (claim.get("claim_text") or "") + " " + (claim.get("source_quote") or "")
    claims_not_compile = bool(COMPILE_CLAIM_RE.search(text))
    claims_compiles = bool(COMPILES_OK_CLAIM_RE.search(text))
    if not (claims_not_compile or claims_compiles):
        return {"verdict": "UNCHECKABLE", "checker": f"compile_validity_{lang}",
                "evidence": {"reason": "not_a_compile_claim",
                             "note": "execution checks out of scope for "
                                     f"{lang}; compile-only tier"}}
    if not code:
        return {"verdict": "UNCHECKABLE", "checker": f"compile_validity_{lang}",
                "evidence": {"reason": "no_code_snippet"}}
    compile_fn = java_compile_ok if lang == "java" else csharp_compile_ok
    ok, err = compile_fn(code)
    if ok is None:
        return {"verdict": "UNCHECKABLE", "checker": f"compile_validity_{lang}",
                "evidence": {"reason": err}}
    evidence = {"compile_ok": ok, "compiler_err_head": err if not ok else "",
                "claims_not_compile": claims_not_compile}
    if claims_not_compile:
        v = "SUPPORTED" if not ok else "REFUTED"
    else:
        v = "SUPPORTED" if ok else "REFUTED"
    return {"verdict": v, "checker": f"compile_validity_{lang}",
            "evidence": evidence}


# -------------------- equivalence / improvement -------------------------------

def check_equivalence_or_improvement(claim, op_code: str | None):
    # For the pilot, conservative: UNCHECKABLE unless both snippets are
    # self-contained Python with a single top-level function and a clear
    # input pattern.
    return {"verdict": "UNCHECKABLE", "checker": "equivalence_runtime",
            "evidence": {"reason": "needs_two_self_contained_snippets"}}


# -------------------- dispatch -------------------------------------------------

CHECKED_LANGS = {"python", "javascript", "typescript", "java", "c#", "unknown"}


def dispatch_claim(c: dict, lang: str | None, ctype: str | None,
                   bodies: dict[str, str], q_map: dict[str, str]) -> dict:
    """Route a claim to the right per-language checker."""
    if lang and lang not in CHECKED_LANGS:
        return {"verdict": "UNCHECKABLE", "checker": "lang_filter",
                "evidence": {"language": lang}}

    candidates = resolve_code_candidates(c, bodies)
    if not candidates:
        # Fallback to the stripped pool text (pre-fix behavior).
        answer_text = c.get("answer_text", "") or ""
        code = extract_first_code_block(answer_text)
        if not code:
            code = extract_first_code_block(q_map.get(c.get("answer_id"), "") or "")
        candidates = [(code, "answer")] if code else []

    def pick(pred=None):
        """First (code, prov) whose code satisfies pred (or just first)."""
        for k, prov in candidates:
            if pred is None or pred(k):
                return k, prov
        return (candidates[0] if candidates else (None, None))

    code, code_prov = pick()
    side = claim_target_side(c)

    eff_lang = lang
    if not eff_lang or eff_lang == "unknown":
        eff_lang = guess_language(code or "")
        if eff_lang not in CHECKED_LANGS or eff_lang == "unknown":
            eff_lang = "python"   # pilot behavior: try python for unknowns
    if eff_lang == "typescript":
        eff_lang = "javascript"   # node handles most plain-TS-free snippets

    def guard_refuted(verdict: dict, prov: str | None) -> dict:
        """High-precision REFUTED: only stands when we are confident the
        checked snippet IS the claim's target (model-attached inline code, or
        a block from the side the claim names). Side-mismatched refutations
        were the dominant false-REFUTED mode."""
        if verdict.get("verdict") != "REFUTED":
            return verdict
        ok = (prov == "inline") or (side is not None and prov == side)
        if not ok:
            verdict = dict(verdict)
            verdict["verdict"] = "UNCHECKABLE"
            ev = dict(verdict.get("evidence") or {})
            ev["downgrade_reason"] = "snippet_provenance_unclear"
            ev["snippet_prov"] = prov
            ev["claim_side"] = side
            verdict["evidence"] = ev
        return verdict

    if eff_lang == "python":
        if ctype == "complexity":
            best, prov = pick(lambda k: loop_nesting_depth(k) >= 0)
            return guard_refuted(check_complexity(c, best), prov)
        if ctype == "behavior":
            best, prov = pick(find_top_level_callable)
            return guard_refuted(check_behavior(c, best), prov)
        if ctype == "api_fact":
            return check_api_fact(c)
        if ctype in ("equivalence", "improvement"):
            return check_equivalence_or_improvement(c, code)
        return {"verdict": "UNCHECKABLE", "checker": "unsupported_type",
                "evidence": {"claim_type": ctype}}

    if eff_lang == "javascript":
        if ctype == "complexity":
            return guard_refuted(check_complexity_cstyle(c, code, "javascript"),
                                 code_prov)
        if ctype == "behavior":
            best, prov = pick(find_js_callable)
            return guard_refuted(check_behavior_js(c, best), prov)
        if ctype in ("equivalence", "improvement"):
            return check_equivalence_or_improvement(c, code)
        return {"verdict": "UNCHECKABLE", "checker": "js_scope",
                "evidence": {"reason": f"{ctype}_not_supported_for_js"}}

    if eff_lang in ("java", "c#"):
        return guard_refuted(check_compiled_lang(c, code, eff_lang), code_prov)

    return {"verdict": "UNCHECKABLE", "checker": "lang_filter",
            "evidence": {"language": eff_lang}}


# -------------------- driver --------------------------------------------------

def load_pool_questions(pool_path: Path) -> dict[str, str]:
    """Return {answer_id: question_text} for backup OP-code extraction.

    Note: the row's `text` field is "Question: ... \\n\\nAnswer: ...", so the
    OP code is in the question portion. We split on the first '\\n\\nAnswer:'.
    """
    csv.field_size_limit(10_000_000)
    opener = gzip.open if str(pool_path).endswith(".gz") else open
    out = {}
    with opener(pool_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = r.get("text", "")
            m = re.search(r"\n\nAnswer:\s*", t)
            q = t[:m.start()] if m else t
            out[r["answer_id"]] = q
    return out


def load_raw_bodies(parquet_path: str, needed_ids: set[str]) -> dict[str, str]:
    """{post_id: raw HTML Body} from the SE data-dump parquet, restricted to
    the ids we actually need. The Body is RENDERED HTML (<pre><code> blocks),
    unlike the pool's strip_html()'d text — this is the fix for the pilot's
    no_code_snippet failures."""
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=["Id", "Body"])
    df["Id"] = df["Id"].astype(str)
    sub = df[df["Id"].isin(needed_ids)]
    return dict(zip(sub["Id"], sub["Body"].fillna("")))


def claim_target_side(c: dict) -> str | None:
    """Which post the claim is about: 'question' (OP's original code),
    'answer' (the reviewer's proposed code), or None if unclear. Wrong-side
    snippet checking was the dominant false-REFUTED mode in the v2 run."""
    t = ((c.get("target") or "") + " " + (c.get("claim_text") or "")).lower()
    if re.search(r"\bop\b|\boriginal\b|user'?s\s+code|\byour\s+code\b|"
                 r"\bcurrent\s+(?:code|implementation)\b|\bthe\s+question\b", t):
        return "question"
    if re.search(r"\bproposed\b|\bsuggested\b|\brefactor|\bimproved\b|"
                 r"\balternative\b|\bnew\s+(?:code|version|method)\b|"
                 r"\bthe\s+answer'?s\b", t):
        return "answer"
    return None


def resolve_code_candidates(c: dict, bodies: dict[str, str]) -> list[tuple[str, str]]:
    """Ordered (code, provenance) candidates for a claim. provenance is one
    of 'inline' (model-attached snippet), 'answer', 'question'. Substantial
    inline_code first; then blocks from the side the claim TARGETS; then the
    other side."""
    lang = c.get("language")
    aid = str(c.get("answer_id") or "")
    qid = str(c.get("question_id") or "")
    a_blocks = [(b, "answer") for b in
                blocks_for_language(extract_code_blocks(bodies.get(aid, "")), lang)]
    q_blocks = [(b, "question") for b in
                blocks_for_language(extract_code_blocks(bodies.get(qid, "")), lang)]
    inline = c.get("inline_code") or None
    side = claim_target_side(c)
    first, second = (q_blocks, a_blocks) if side == "question" else (a_blocks, q_blocks)
    cands: list[tuple[str, str]] = []
    if inline and ("\n" in inline or re.search(r"\b(?:def |function |class )", inline)):
        cands.append((inline, "inline"))
    cands.extend(first)
    if inline:
        cands.append((inline, "inline"))
    cands.extend(second)
    # de-dup preserving order
    seen, out = set(), []
    for s, prov in cands:
        if s and s not in seen:
            seen.add(s)
            out.append((s, prov))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims-flat", required=True)
    ap.add_argument("--pool", required=True, help="Pool CSV(.gz) for OP code lookup")
    ap.add_argument("--posts-parquet",
                    default="/lfs/skampere3/0/alexspan/norm-research/datasets/"
                            "codereview_se/posts.parquet",
                    help="SE data-dump parquet with raw HTML bodies (code "
                         "blocks intact). If missing, falls back to the "
                         "stripped pool text.")
    ap.add_argument("--out-verdicts", required=True)
    ap.add_argument("--out-features", required=True)
    args = ap.parse_args()

    flat_path = Path(args.claims_flat)
    verdicts_path = Path(args.out_verdicts)
    features_path = Path(args.out_features)
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)

    print("loading question pool for OP code lookup...", flush=True)
    q_map = load_pool_questions(Path(args.pool))
    print(f"  loaded {len(q_map)} questions", flush=True)

    # Collect ids needed for raw-HTML body lookup, then load them.
    needed_ids: set[str] = set()
    with flat_path.open() as f:
        for line in f:
            try:
                c = json.loads(line)
            except Exception:
                continue
            if c.get("answer_id"):
                needed_ids.add(str(c["answer_id"]))
            if c.get("question_id"):
                needed_ids.add(str(c["question_id"]))
    bodies: dict[str, str] = {}
    if os.path.exists(args.posts_parquet):
        print(f"loading raw HTML bodies from {args.posts_parquet} ...", flush=True)
        bodies = load_raw_bodies(args.posts_parquet, needed_ids)
        print(f"  loaded {len(bodies)} / {len(needed_ids)} raw bodies", flush=True)
    else:
        print(f"WARNING: {args.posts_parquet} missing — falling back to "
              "stripped pool text for code extraction", flush=True)

    print(f"compilers: javac={JAVAC_BIN}  mcs={MCS_BIN}  node={NODE_BIN}", flush=True)

    per_answer = defaultdict(lambda: {"n_claims": 0, "n_supported": 0,
                                       "n_refuted": 0, "n_refuted_clean": 0,
                                       "n_uncheckable": 0,
                                       "by_type": Counter(),
                                       "verdicts_by_type": defaultdict(Counter)})

    n_total = 0
    summary = Counter()
    with flat_path.open() as fin, verdicts_path.open("w") as fout:
        for line in fin:
            try:
                c = json.loads(line)
            except Exception:
                continue
            n_total += 1
            ctype = c.get("claim_type")
            lang = c.get("language")
            # Per-row isolation: one bad claim must never kill the batch.
            try:
                verdict = dispatch_claim(c, lang, ctype, bodies, q_map)
            except Exception as e:
                verdict = {"verdict": "UNCHECKABLE", "checker": "checker_crash",
                           "evidence": {"error": repr(e)[:300]}}
            # Binding-cue downgrade: REFUTED claims whose quote starts with a
            # binding cue (if/let/define/suppose/...) are downgraded to
            # UNCHECKABLE — they are usually definitions or conditionals.
            downgraded = False
            if verdict["verdict"] == "REFUTED" and c.get("binding_cue"):
                verdict["verdict"] = "UNCHECKABLE"
                verdict["binding_cue_downgrade"] = True
                downgraded = True
            else:
                verdict["binding_cue_downgrade"] = False

            out_rec = {
                "answer_id":  c["answer_id"],
                "claim_idx":  c["claim_idx"],
                "claim_type": ctype,
                "language":   lang,
                "claim_text": c.get("claim_text"),
                "source_quote": c.get("source_quote"),
                "verdict":    verdict["verdict"],
                "checker":    verdict["checker"],
                "evidence":   verdict["evidence"],
                "binding_cue_downgrade": downgraded,
                "binding_cue": c.get("binding_cue"),
                "conditional": c.get("conditional"),
            }
            fout.write(json.dumps(out_rec) + "\n")

            # aggregate per-answer
            aid = c["answer_id"]
            pa = per_answer[aid]
            pa["judgement"] = c.get("judgement")
            pa["split"] = c.get("split")
            pa["primary_tag"] = c.get("primary_tag")
            pa["n_claims"] += 1
            pa["by_type"][ctype] += 1
            pa["verdicts_by_type"][ctype][verdict["verdict"]] += 1
            summary[verdict["verdict"]] += 1
            if verdict["verdict"] == "SUPPORTED":
                pa["n_supported"] += 1
            elif verdict["verdict"] == "REFUTED":
                pa["n_refuted"] += 1
                if not c.get("binding_cue"):
                    pa["n_refuted_clean"] += 1
            else:
                pa["n_uncheckable"] += 1

    with features_path.open("w") as f:
        for aid, pa in per_answer.items():
            rec = {
                "answer_id": aid,
                "judgement": pa.get("judgement"),
                "split":     pa.get("split"),
                "primary_tag": pa.get("primary_tag"),
                "n_claims":  pa["n_claims"],
                "n_supported": pa["n_supported"],
                "n_refuted":   pa["n_refuted"],
                "n_refuted_clean": pa["n_refuted_clean"],
                "n_uncheckable": pa["n_uncheckable"],
                "by_type":   dict(pa["by_type"]),
                "verdicts_by_type": {k: dict(v) for k, v in pa["verdicts_by_type"].items()},
            }
            f.write(json.dumps(rec) + "\n")

    print(f"\nTOTAL CLAIMS PROCESSED: {n_total}")
    print(f"VERDICT BREAKDOWN: {dict(summary)}")
    print(f"  SUPPORTED:   {summary['SUPPORTED']}  "
          f"({100 * summary['SUPPORTED'] / max(1, n_total):.1f}%)")
    print(f"  REFUTED:     {summary['REFUTED']}  "
          f"({100 * summary['REFUTED'] / max(1, n_total):.1f}%)")
    print(f"  UNCHECKABLE: {summary['UNCHECKABLE']}  "
          f"({100 * summary['UNCHECKABLE'] / max(1, n_total):.1f}%)")
    print(f"\nPer-answer features: {features_path}")
    print(f"Per-claim verdicts:  {verdicts_path}")


if __name__ == "__main__":
    main()

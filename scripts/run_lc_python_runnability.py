"""Compute a480/a481/a482 metrics: LeetCode Python pass/runtime/memory.

Improvements over scripts/runtime_executor_editorial.py (46% -> target 65%):

1. Parse the LC reference completion to extract class + method name.
2. Inspect candidate AST to find what classes/functions are defined.
3. If candidate defines `class Solution` (the typical case) -> use as-is.
4. If candidate defines bare top-level function with method name -> wrap
   it inside `class Solution`.
5. If candidate defines an indented method body without surrounding class
   (e.g. starts with `    def method(self, ...):`) -> dedent and wrap.
6. If candidate defines a *different* class (design problems like
   `class FindSumPairs`) -> use whatever class the test fixture expects.
7. Heal common escape artifacts: `\\'` -> `'`, `\\t` -> tab, etc.
8. 8s SIGALRM in-process timeout + 15s outer subprocess timeout.

Outputs: outputs/v2_analysis/lc_python_runnability_scores.parquet
  columns: candidate_id, question_slug, code_hash,
           a480 (pass 0/1), a481 (runtime pct), a482 (memory pct),
           status, runtime_ms, mem_kb
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

CAND_PATH = "/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/lc_candidate_corpus.parquet"
TESTS_PATH = "/lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_codecontests/leetcode_with_tests.parquet"
SAMPLE_2K = "/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/lc_python_2k_sample.parquet"
OUT_PATH = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/lc_python_runnability_scores.parquet")
TMP_DIR = Path("/lfs/skampere3/0/alexspan/tmp/lc_runnability")
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------- code healing ----------------------

def heal_escapes(code: str) -> str:
    """Heal JSON-style escape artifacts.

    Two regimes:
      (a) Whole-file escaped: `\\n` outnumbers real `\n`. Unescape everything.
      (b) Inline `\\'` / `\\\"` characters that ought to be literal quotes but
          weren't escaped properly when the snippet was extracted from JSON.
          Detect by checking whether code still parses; if not, attempt to
          replace lone `\\'` / `\\\"` (outside of valid Python escapes).
    """
    if not isinstance(code, str):
        return code

    # Regime A: largely-escaped string (one-line with \n literals).
    n_real = code.count("\n")
    n_esc = code.count("\\n")
    if n_esc > n_real and n_esc >= 3:
        c = code.replace("\\'", "'").replace('\\"', '"')
        c = c.replace("\\n", "\n").replace("\\t", "\t")
        return c

    # Regime B: try parsing in BOTH original AND dedented forms (some
    # snippets are bare indented method bodies). If either parses, keep
    # as-is.
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    except Exception:
        return code

    import textwrap as _tw
    try:
        ast.parse(_tw.dedent(code))
        return code  # original parses fine after dedent; healing not needed
    except SyntaxError:
        pass
    except Exception:
        pass

    # Try replacing backslash-quote tokens. Parses are tested in both
    # original and dedented forms.
    candidate = code
    if "\\'" in candidate or '\\"' in candidate:
        c2 = candidate.replace("\\'", "'").replace('\\"', '"')
        for variant in (c2, _tw.dedent(c2)):
            try:
                ast.parse(variant)
                return c2
            except SyntaxError:
                pass

    # Try also unescaping `\t` and `\n`.
    if "\\t" in candidate or "\\n" in candidate:
        c3 = candidate.replace("\\t", "\t").replace("\\n", "\n")
        for variant in (c3, _tw.dedent(c3)):
            try:
                ast.parse(variant)
                return c3
            except SyntaxError:
                pass

    # Combined: try both
    if ("\\'" in candidate or '\\"' in candidate or
            "\\t" in candidate or "\\n" in candidate):
        c4 = (candidate.replace("\\'", "'").replace('\\"', '"')
              .replace("\\t", "\t").replace("\\n", "\n"))
        for variant in (c4, _tw.dedent(c4)):
            try:
                ast.parse(variant)
                return c4
            except SyntaxError:
                pass

    return code


# ---------------------- entry-point parsing ----------------------

_ENTRY_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*\)\s*\.\s*([A-Za-z_]\w*)\s*$")


def parse_entry_point(entry_point: str) -> Tuple[str, str]:
    """`Solution().twoSum` -> ('Solution', 'twoSum'); fall back: ('Solution', entry_point)."""
    if not entry_point:
        return ("Solution", "")
    m = _ENTRY_RE.match(entry_point.strip())
    if m:
        return (m.group(1), m.group(2))
    # Some LC design-problem entries are just the class name -- e.g. 'FindSumPairs'
    s = entry_point.strip()
    if "." not in s and "(" not in s:
        return (s, "")
    return ("Solution", "")


# ---------------------- candidate normalization ----------------------

def parse_candidate(code: str):
    """Return ast.Module or None."""
    try:
        return ast.parse(code)
    except Exception:
        return None


def extract_top_level_class_names(tree) -> list:
    return [n.name for n in tree.body if isinstance(n, ast.ClassDef)]


def extract_top_level_function_names(tree) -> list:
    return [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def is_method_indented(code: str) -> bool:
    """Detect snippets that start with an indented `def method(self, ...)`."""
    lines = code.split("\n")
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if ln.startswith(" ") and (s.startswith("def ") or s.startswith("async def ")):
            return True
        return False
    return False


def normalize_candidate(code: str, want_class: str, want_method: str) -> str:
    """Reshape candidate so that calling `want_class().want_method(...)` works.

    Strategy:
      A. Code parses as-is AND defines `want_class` -> use as-is.
      B. Code parses AND defines a top-level function with name `want_method`
         -> wrap into `class want_class: def want_method(self, *args, **kw):
            return _orig(*args, **kw)`.
      C. Code looks like a bare method body (indented `def method(self,)`) ->
         dedent + wrap in `class want_class:`.
      D. Code parses BUT defines `class Solution` and want_class != 'Solution'
         (design problem mismatch) -> rename the class.
      E. Code defines a different class than expected for a design problem ->
         alias it.
    """
    code = heal_escapes(code)

    tree = parse_candidate(code)
    if tree is None:
        # Try: dedent + wrap if indented method
        if is_method_indented(code):
            dedented = textwrap.dedent(code)
            tree2 = parse_candidate(dedented)
            if tree2 is not None:
                fns = extract_top_level_function_names(tree2)
                if want_method and want_method in fns:
                    body = textwrap.indent(dedented, "    ")
                    return f"class {want_class}:\n{body}\n"
        return code  # let downstream catch the syntax error

    classes = extract_top_level_class_names(tree)
    funcs = extract_top_level_function_names(tree)

    if want_class in classes:
        return code

    # Case B: bare top-level method. The original might or might not have
    # a `self` first arg. If it does, call without self; if not, also fine.
    if want_method and want_method in funcs:
        # Find first arg name to decide
        first_arg = None
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == want_method:
                if n.args.args:
                    first_arg = n.args.args[0].arg
                break
        if first_arg in ("self", "cls"):
            wrap = [
                code, "",
                f"class {want_class}:",
                f"    def {want_method}(self, *args, **kwargs):",
                f"        return {want_method}(self, *args, **kwargs)",
            ]
        else:
            wrap = [
                code, "",
                f"class {want_class}:",
                f"    def {want_method}(self, *args, **kwargs):",
                f"        return {want_method}(*args, **kwargs)",
            ]
        return "\n".join(wrap) + "\n"

    # Case C: indented method body
    if is_method_indented(code):
        dedented = textwrap.dedent(code)
        tree2 = parse_candidate(dedented)
        if tree2 is not None:
            fns2 = extract_top_level_function_names(tree2)
            if want_method and want_method in fns2:
                body = textwrap.indent(dedented, "    ")
                return f"class {want_class}:\n{body}\n"

    # Case D: candidate defines `class Solution` but we want a different class
    # (design problem). Or vice versa.
    if want_class != "Solution" and "Solution" in classes:
        # alias
        return code + f"\n{want_class} = Solution\n"
    if want_class == "Solution" and classes and "Solution" not in classes:
        # take the first class as Solution
        return code + f"\nSolution = {classes[0]}\n"

    # Last resort: leave it alone, may fail.
    return code


# ---------------------- subprocess wrapper ----------------------

WRAPPER_HEADER = r"""
import signal, sys, time as _t, json as _j, resource, tracemalloc
def _h(signum, frame):
    raise TimeoutError("exec timeout")
signal.signal(signal.SIGALRM, _h)
signal.alarm(8)
try:
    from sortedcontainers import SortedList, SortedDict, SortedSet
except Exception:
    pass
"""

WRAPPER_FOOTER_TPL = r"""
_e = {entry_point_repr}
import inspect as _inspect
try:
    _candidate = eval(_e)
except Exception as _ex:
    print("__RESULT__" + _j.dumps({{"status":"eval_err","err":repr(_ex)[:200]}}))
    sys.exit(0)

# Wrap candidate so kwargs are mapped to positional args. Tests sometimes
# call candidate(board=[[..]]) where the candidate's method declares the
# first parameter as `A`.
def _adapt(cand):
    try:
        sig = _inspect.signature(cand)
        params = [p for p in sig.parameters.values()
                  if p.kind in (_inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                _inspect.Parameter.POSITIONAL_ONLY)]
        pnames = [p.name for p in params]
    except (TypeError, ValueError):
        return cand
    def _wrap(*args, **kwargs):
        # If no kwargs, call directly.
        if not kwargs:
            return cand(*args, **kwargs)
        # Map kwargs onto remaining positions after positional args consumed.
        remaining = pnames[len(args):]
        new_args = list(args)
        leftover = {{}}
        # Drain kwargs by remaining position order.
        for nm in remaining:
            if nm in kwargs:
                new_args.append(kwargs.pop(nm))
            else:
                break
        # If kwargs left but the candidate has matching arg names, pass them.
        # Otherwise, pass remaining values positionally in given order.
        if kwargs:
            for v in list(kwargs.values()):
                new_args.append(v)
            kwargs = {{}}
        try:
            return cand(*new_args)
        except TypeError:
            # Fall back to passing as kwargs of candidate's signature names.
            try:
                kw = {{nm: new_args[i] for i, nm in enumerate(pnames[:len(new_args)])}}
                return cand(**kw)
            except Exception:
                return cand(*args, **kwargs)
    return _wrap

_candidate = _adapt(_candidate)

_durs = []
_peak_mem = 0
for _ in range({n_reps}):
    tracemalloc.start()
    _s = _t.perf_counter()
    try:
        check(_candidate)
    except Exception as _ex:
        tracemalloc.stop()
        print("__RESULT__" + _j.dumps({{"status":"test_fail","err":repr(_ex)[:200]}}))
        sys.exit(0)
    _durs.append(_t.perf_counter() - _s)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if peak > _peak_mem:
        _peak_mem = peak
print("__RESULT__" + _j.dumps({{"status":"ok","durs_s":_durs,"peak_bytes":_peak_mem}}))
"""


def build_source(prompt: str, normalized_code: str, test_src: str,
                 entry_point: str, n_reps: int = 1) -> str:
    parts = [
        WRAPPER_HEADER,
        prompt or "",
        "",
        normalized_code or "",
        "",
        test_src or "",
        "",
        WRAPPER_FOOTER_TPL.format(entry_point_repr=repr(entry_point), n_reps=n_reps),
    ]
    return "\n".join(parts)


def _run_one(payload):
    cid, slug, code, prompt, entry_point, test_src, n_reps = payload
    cls, meth = parse_entry_point(entry_point)
    normalized = normalize_candidate(code, cls, meth)
    src = build_source(prompt, normalized, test_src, entry_point, n_reps)
    code_hash = hashlib.sha1(
        code.replace("\r", "").strip().encode("utf-8", errors="replace")
    ).hexdigest()

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(TMP_DIR)) as f:
        f.write(src)
        fp = f.name
    rec = {"candidate_id": int(cid), "question_slug": slug,
           "code_hash": code_hash, "status": "err",
           "runtime_ms": np.nan, "mem_kb": np.nan, "stderr": ""}
    try:
        r = subprocess.run([sys.executable, fp],
                           capture_output=True, text=True, timeout=15)
        out = r.stdout
        if "__RESULT__" in out:
            payload_json = out.split("__RESULT__", 1)[1].strip().splitlines()[0]
            d = json.loads(payload_json)
            rec["status"] = d.get("status", "unk")
            if d.get("status") == "ok":
                durs = d["durs_s"]
                rec["runtime_ms"] = float(np.median(durs) * 1000.0)
                rec["mem_kb"] = float(d.get("peak_bytes", 0)) / 1024.0
            else:
                rec["stderr"] = d.get("err", "")[:200]
        else:
            rec["status"] = "parse_fail"
            rec["stderr"] = (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        rec["status"] = "subproc_timeout"
        rec["stderr"] = "subprocess timeout"
    except Exception as e:
        rec["status"] = "err"
        rec["stderr"] = repr(e)[:300]
    finally:
        try:
            os.unlink(fp)
        except Exception:
            pass
    return rec


# ---------------------- main ----------------------

def build_payloads(candidates_df: pd.DataFrame,
                   tests_df: pd.DataFrame,
                   n_reps: int = 1):
    tests = tests_df[["task_id", "prompt", "completion", "entry_point", "test"]].rename(
        columns={"task_id": "question_slug"}
    )
    merged = candidates_df.merge(tests, on="question_slug", how="inner")
    payloads = []
    for _, r in merged.iterrows():
        payloads.append((
            int(r["candidate_id"]),
            r["question_slug"],
            r["code"] or "",
            r["prompt"] or "",
            r["entry_point"] or "",
            r["test"] or "",
            n_reps,
        ))
    return payloads, merged


def compute_percentiles(scored: pd.DataFrame, min_passing: int = 3) -> pd.DataFrame:
    """Add a481/a482 columns: within-problem percentile of runtime/memory.

    1.0 = fastest / lowest memory. NaN unless candidate passes (a480=1) and
    its problem cell has >= min_passing passing candidates.
    """
    scored = scored.copy()
    scored["a480"] = (scored["status"] == "ok").astype(int)
    scored["a481"] = np.nan
    scored["a482"] = np.nan
    ok = scored[scored["a480"] == 1].copy()
    grp_sizes = ok.groupby("question_slug").size()
    big_slugs = set(grp_sizes[grp_sizes >= min_passing].index)
    ok = ok[ok["question_slug"].isin(big_slugs)]
    # Rank ascending: lower runtime/memory -> larger percentile, after
    # invert.
    ok["a481"] = ok.groupby("question_slug")["runtime_ms"].rank(
        ascending=True, pct=True, method="average"
    ).apply(lambda p: 1.0 - p)
    ok["a482"] = ok.groupby("question_slug")["mem_kb"].rank(
        ascending=True, pct=True, method="average"
    ).apply(lambda p: 1.0 - p)
    # Scatter back
    scored.loc[ok.index, "a481"] = ok["a481"].values
    scored.loc[ok.index, "a482"] = ok["a482"].values
    return scored


def main():
    mode = os.environ.get("MODE", "2k")  # '2k', '5k', or 'all'

    print(f"[mode={mode}] loading inputs...", flush=True)
    cands = pd.read_parquet(CAND_PATH)
    tests = pd.read_parquet(TESTS_PATH)
    py = cands[cands.language_norm.isin(["python", "py", "python3"])].copy()
    print(f"  python candidates: {len(py)}", flush=True)

    if mode == "2k":
        samp = pd.read_parquet(SAMPLE_2K)
        keep = samp.candidate_id.unique().tolist()
        py = py[py.candidate_id.isin(keep)].copy()
        out_path = OUT_PATH
    elif mode == "5k":
        bank = pd.read_parquet("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/lc_python_metric_scores.parquet")
        keep = bank.candidate_id.unique().tolist()
        py = py[py.candidate_id.isin(keep)].copy()
        out_path = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/lc_python_runnability_scores_5k.parquet")
    else:
        out_path = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/lc_python_runnability_scores_all.parquet")

    print(f"  selected: {len(py)}  unique slugs: {py.question_slug.nunique()}", flush=True)
    payloads, _ = build_payloads(py, tests, n_reps=int(os.environ.get("N_REPS", "1")))
    print(f"  payloads: {len(payloads)}", flush=True)

    nproc = int(os.environ.get("NPROC", "24"))
    t0 = time.time()
    results = []
    with mp.Pool(nproc) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_one, payloads, chunksize=4)):
            results.append(res)
            if (i + 1) % 200 == 0:
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-3)
                eta = (len(payloads) - i - 1) / max(rate, 1e-3)
                print(f"  {i+1}/{len(payloads)} elapsed={dt:.1f}s "
                      f"rate={rate:.1f}/s eta={eta:.1f}s", flush=True)

    df = pd.DataFrame(results)
    print("status:", df.status.value_counts().to_dict(), flush=True)
    pass_rate = (df.status == "ok").mean()
    print(f"pass rate: {pass_rate:.3f}", flush=True)

    syn_valid = df  # downstream filtering done by users
    # Compute percentiles
    df = compute_percentiles(df, min_passing=3)
    df.to_parquet(out_path, index=False)
    print(f"saved -> {out_path}", flush=True)
    print("a480 mean:", df.a480.mean(), flush=True)
    print("a481 describe:\n", df.a481.describe(), flush=True)
    print("a482 describe:\n", df.a482.describe(), flush=True)


if __name__ == "__main__":
    main()

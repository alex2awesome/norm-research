"""Sandbox executor for competition-code candidates → execution features.

Supports two test formats found in this repo:
  * LC-style: a `test(check(candidate))` function where candidate=Solution().method,
    plus a fixed prompt header (typing imports + helper data structures).
  * stdin/stdout-style: list of {"input": str, "output": str} pairs; run binary
    or `python3 -` and compare stdout token-wise (float tol 1e-6 when parseable).

For each candidate produces:
  compiled (bool), n_pass, n_fail, n_timeout, n_runtime_err, n_tests,
  wall_ms_median, wall_ms_max, pass_rate, runtime_err_msg_first.

Per-test timeout = 5 s; per-process memory limit = 512 MB (resource.setrlimit).
Networking is not blocked (no container); acceptable for competition code.

Usage (LC):
  python run_candidate_tests.py lc \
      --candidates /tmp/lc_2k_with_code.parquet \
      --tests /lfs/.../leetcode_with_tests.parquet \
      --out /lfs/.../comp_exec_features_lc.parquet \
      [--limit N] [--workers 8] [--per-test-timeout 5]

Usage (stdin/stdout — CC/Phase1b):
  python run_candidate_tests.py stdio \
      --candidates /tmp/cc_449_claude_with_code.parquet \
      --tests /tmp/cc_tests.parquet \
      --out /lfs/.../comp_exec_features_cc.parquet
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------- LC helpers ---------------------------

# A standard prompt header — pulled from leetcode_with_tests.parquet "prompt"
# column row[0]. We embed it once here so we don't have to ship it per candidate.
LC_PROMPT_HEADER = """
import random
import functools
import collections
import string
import math
import datetime
import sys
import heapq
import bisect
import itertools
import re
import operator

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

# re-bind module names overshadowed by star imports above so candidate code that
# does `bisect.bisect(...)` / `itertools.count(...)` still works.
import bisect, itertools, operator, collections, functools, heapq, string, math
xrange = range  # Py2-compat for occasional forum snippets

inf = float('inf')


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def list_node(values):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head


def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def tree_node(values):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = [root]
    while queue:
        node = queue.pop(0)
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    return p.val == q.val and is_same_tree(p.left, p.right) and is_same_tree(p.right, q.right)
"""


def _build_lc_program(candidate_code: str, test_code: str, entry_point: str,
                      assertion_index: int | None = None) -> str:
    """Construct a runnable Python program for one LC test (or all)."""
    # entry_point looks like "Solution().twoSum" — that's the candidate callable.
    # The check function does `candidate(args) == expected` for each assert.
    if assertion_index is None:
        runner = "check(candidate)\nprint('ALL_PASS')"
    else:
        runner = textwrap.dedent(
            f"""
            asserts = [line for line in check.__code__.co_consts]
            # We can't easily split asserts; instead use linecache-style split.
            # Simpler: run the check fn but stop at first failure.
            try:
                check(candidate)
                print('ALL_PASS')
            except AssertionError as e:
                print('FAIL', e)
            """
        )
    return (
        LC_PROMPT_HEADER
        + "\n\n"
        + candidate_code
        + "\n\n"
        + test_code
        + "\n\n"
        + f"candidate = {entry_point}\n"
        + runner
    )


def _split_lc_test_into_asserts(test_code: str) -> list[str]:
    """Split a `def check(candidate):\n    assert ... ` body into per-assert
    standalone programs. Returns a list of assert lines (without 'assert ' prefix
    we keep it; we wrap as `try: <assert>; print('PASS')` per assert)."""
    lines = test_code.splitlines()
    asserts = []
    cur = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("assert "):
            if cur:
                asserts.append("\n".join(cur))
            cur = [ln]
        elif cur:
            # continuation
            if ln.startswith("    ") or ln.startswith("\t"):
                cur.append(ln)
            else:
                asserts.append("\n".join(cur))
                cur = []
    if cur:
        asserts.append("\n".join(cur))
    # Strip leading whitespace and `assert ` prefix
    out = []
    for a in asserts:
        a = textwrap.dedent(a).strip()
        if a.startswith("assert "):
            a = a[len("assert "):]
        out.append(a)
    return out


def _build_lc_per_assert_program(candidate_code: str, entry_point: str,
                                  asserts: list[str]) -> str:
    """Build a single program that runs each assert, prints PASS/FAIL per line."""
    runner_lines = []
    for i, a in enumerate(asserts):
        # Each assert becomes:
        #   try:
        #       _ok = (a)
        #       print(f"R {i} {'PASS' if _ok else 'FAIL'}")
        #   except Exception as e:
        #       print(f"R {i} EXC {type(e).__name__}: {e}")
        runner_lines.append(
            f"try:\n    _ok = bool({a})\n    print('R', {i}, 'PASS' if _ok else 'FAIL')\nexcept BaseException as e:\n    print('R', {i}, 'EXC', type(e).__name__, str(e)[:80])"
        )
    runner = "\n".join(runner_lines)
    return (
        LC_PROMPT_HEADER
        + "\n\n"
        + candidate_code
        + "\n\n"
        + f"candidate = {entry_point}\n"
        + runner
    )


def _exec_python(program: str, timeout_s: float, memory_mb: int = 512) -> dict:
    """Run a Python program in a subprocess. Returns dict with rc, stdout, stderr,
    wall_ms, timed_out."""
    t0 = time.perf_counter()
    try:
        # We rely on the subprocess hitting the per-test timeout via Popen.communicate
        proc = subprocess.Popen(
            [sys.executable, "-c", program],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=lambda: _apply_rlimits(memory_mb),
        )
        try:
            out, err = proc.communicate(timeout=timeout_s)
            rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                out, err = proc.communicate(timeout=2)
            except Exception:
                out, err = b"", b""
            rc = -9
            timed_out = True
    except Exception as e:
        return {
            "rc": -1, "stdout": "", "stderr": f"LAUNCH_FAIL: {e}",
            "wall_ms": (time.perf_counter() - t0) * 1000, "timed_out": False,
        }
    return {
        "rc": rc,
        "stdout": out.decode("utf-8", errors="replace") if isinstance(out, bytes) else out,
        "stderr": err.decode("utf-8", errors="replace") if isinstance(err, bytes) else err,
        "wall_ms": (time.perf_counter() - t0) * 1000,
        "timed_out": timed_out,
    }


def _apply_rlimits(memory_mb: int) -> None:
    """Set address-space and file-size limits in the child (Unix only)."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024,) * 2)
    except Exception:
        pass


# --------------------------- LC runner ---------------------------


def _clean_lc_candidate_code(code: str, entry_point: str) -> str:
    """Heuristic repairs for LC candidate code as stored in our corpus:
      * unescape backslash-escaped single quotes (\\'inf\\' -> 'inf') common in
        forum scrapes.
      * if `class Solution` is missing, wrap the code so `Solution().method`
        works. We detect the method name from entry_point ("Solution().X").
    """
    # Unescape — corpus has some snippets stored with backslash-escapes
    # We only do this if the code looks like raw text (contains `\t` or `\n`
    # at backslash-literal positions but real newlines as well — heuristic:
    # if `\n` appears as literal backslash-n AND there are no actual newlines,
    # OR if `\t` appears as literal backslash-t).
    if "\\t" in code and "\t" not in code:
        code = code.replace("\\t", "\t")
    if "\\n" in code and code.count("\\n") > code.count("\n"):
        # only convert backslash-n if there are more escaped than real
        code = code.replace("\\n", "\n")
    if "\\'" in code:
        code = code.replace("\\'", "'")
    if '\\"' in code:
        code = code.replace('\\"', '"')

    # If `class Solution` is missing, try to wrap
    if "class Solution" not in code:
        # method name
        meth = entry_point.split(".")[-1] if entry_point else ""
        # If code starts with `def <meth>` (top-level fn), wrap it inside Solution.
        stripped = code.lstrip()
        if meth and stripped.startswith(f"def {meth}"):
            indented = textwrap.indent(code, "    ")
            code = f"class Solution:\n{indented}"
        elif stripped.startswith("    def ") or stripped.startswith("\tdef "):
            # body of Solution without the class header
            code = "class Solution:\n" + code
    return code


def run_lc_candidate(candidate_code: str, test_code: str, entry_point: str,
                     per_test_timeout: float = 5.0,
                     max_asserts: int | None = None,
                     overall_timeout: float = 30.0,
                     memory_mb: int = 512) -> dict:
    """Execute one LC candidate against all asserts in a single subprocess
    (faster than per-assert). If overall_timeout is hit we fall back to
    `compiled=False`-style summary.
    """
    asserts = _split_lc_test_into_asserts(test_code)
    if max_asserts:
        asserts = asserts[:max_asserts]
    n = len(asserts)
    if n == 0:
        return {
            "compiled": None, "n_tests": 0, "n_pass": 0, "n_fail": 0,
            "n_timeout": 0, "n_runtime_err": 0,
            "wall_ms_median": None, "wall_ms_max": None, "pass_rate": None,
            "first_error": "no_asserts",
        }

    candidate_code = _clean_lc_candidate_code(candidate_code, entry_point)
    program = _build_lc_per_assert_program(candidate_code, entry_point, asserts)
    # We give the whole batch up to overall_timeout sec (since LC tests are fast)
    res = _exec_python(program, timeout_s=overall_timeout, memory_mb=memory_mb)

    # Parse: stdout lines of form "R <i> PASS/FAIL/EXC ..."
    n_pass = n_fail = n_runtime_err = 0
    reported = set()
    first_error = None
    for line in (res["stdout"] or "").splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 3 or parts[0] != "R":
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        verdict = parts[2]
        reported.add(idx)
        if verdict == "PASS":
            n_pass += 1
        elif verdict == "FAIL":
            n_fail += 1
            if first_error is None:
                first_error = f"FAIL@{idx}"
        elif verdict == "EXC":
            n_runtime_err += 1
            if first_error is None:
                first_error = (parts[3] if len(parts) > 3 else "EXC")[:120]

    # Tests that produced no output (process died early): runtime/compile error
    n_no_output = n - len(reported)
    if res["timed_out"]:
        # All un-reported asserts are TLE
        n_timeout = n_no_output
        n_runtime_err += 0
    else:
        n_timeout = 0
        n_runtime_err += n_no_output

    # Compile detection: SyntaxError, IndentationError → all tests fail w/o output
    compiled = True
    stderr = res["stderr"] or ""
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        compiled = False
        if first_error is None:
            first_error = "SyntaxError"
    elif n_no_output == n and not res["timed_out"]:
        # Possibly NameError / ImportError on module import → not "compiled" in
        # the runtime sense but Python's "compile" passed. We still flag.
        compiled = False
        if first_error is None:
            # Grab first stderr line
            err_first = stderr.strip().splitlines()
            first_error = (err_first[-1] if err_first else "no_output")[:120]

    pass_rate = n_pass / n if n else None
    return {
        "compiled": compiled,
        "n_tests": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_timeout": n_timeout,
        "n_runtime_err": n_runtime_err,
        "wall_ms_median": res["wall_ms"] / max(n, 1),
        "wall_ms_max": res["wall_ms"],
        "pass_rate": pass_rate,
        "first_error": first_error,
    }


# --------------------------- stdio runner (CC etc) ---------------------------


def _normalize_token(t: str) -> str:
    return t.strip()


def _outputs_equal(expected: str, actual: str, float_tol: float = 1e-6) -> bool:
    e_toks = expected.split()
    a_toks = actual.split()
    if len(e_toks) != len(a_toks):
        # try line-then-token after stripping trailing whitespace
        e_lines = [l.rstrip() for l in expected.splitlines()]
        a_lines = [l.rstrip() for l in actual.splitlines()]
        while e_lines and e_lines[-1] == "":
            e_lines.pop()
        while a_lines and a_lines[-1] == "":
            a_lines.pop()
        if len(e_lines) != len(a_lines):
            return False
        for el, al in zip(e_lines, a_lines):
            if el.split() != al.split():
                # try float compare
                if not _toks_float_eq(el.split(), al.split(), float_tol):
                    return False
        return True
    return e_toks == a_toks or _toks_float_eq(e_toks, a_toks, float_tol)


def _toks_float_eq(es, as_, tol):
    if len(es) != len(as_):
        return False
    for x, y in zip(es, as_):
        try:
            fx = float(x); fy = float(y)
            if abs(fx - fy) > tol * max(1.0, abs(fx), abs(fy)):
                return False
        except ValueError:
            if x != y:
                return False
    return True


def run_stdio_python(code: str, tests: list[dict],
                     per_test_timeout: float = 5.0,
                     memory_mb: int = 512) -> dict:
    """Run a Python candidate via subprocess for each test (input piped to stdin)."""
    n_pass = n_fail = n_timeout = n_runtime_err = 0
    wall_ms_list: list[float] = []
    first_error = None
    compiled = None
    # Pre-compile check
    try:
        compile(code, "<cand>", "exec")
        compiled = True
    except SyntaxError as e:
        return {
            "compiled": False, "n_tests": len(tests), "n_pass": 0,
            "n_fail": 0, "n_timeout": 0, "n_runtime_err": len(tests),
            "wall_ms_median": None, "wall_ms_max": None, "pass_rate": 0.0,
            "first_error": f"SyntaxError: {e}",
        }

    for t in tests:
        inp = t.get("input", "")
        exp = t.get("output", "")
        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=lambda: _apply_rlimits(memory_mb),
            )
            try:
                out, err = proc.communicate(input=inp.encode("utf-8"),
                                             timeout=per_test_timeout)
                rc = proc.returncode
                wall = (time.perf_counter() - t0) * 1000
                wall_ms_list.append(wall)
                if rc != 0:
                    n_runtime_err += 1
                    if first_error is None:
                        emsg = (err.decode("utf-8", errors="replace") or "")
                        first_error = emsg.splitlines()[-1][:120] if emsg else f"rc={rc}"
                    continue
                actual = out.decode("utf-8", errors="replace")
                if _outputs_equal(exp, actual):
                    n_pass += 1
                else:
                    n_fail += 1
                    if first_error is None:
                        first_error = f"WA: exp={exp[:30]!r} got={actual[:30]!r}"
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.communicate(timeout=2)
                except Exception:
                    pass
                wall = (time.perf_counter() - t0) * 1000
                wall_ms_list.append(wall)
                n_timeout += 1
                if first_error is None:
                    first_error = "TLE"
        except Exception as e:
            n_runtime_err += 1
            if first_error is None:
                first_error = f"LAUNCH: {e}"

    n = len(tests)
    return {
        "compiled": compiled,
        "n_tests": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_timeout": n_timeout,
        "n_runtime_err": n_runtime_err,
        "wall_ms_median": statistics.median(wall_ms_list) if wall_ms_list else None,
        "wall_ms_max": max(wall_ms_list) if wall_ms_list else None,
        "pass_rate": n_pass / n if n else None,
        "first_error": first_error,
    }


# --------------------------- worker pool ---------------------------


def _lc_worker(args):
    idx, candidate_code, test_code, entry_point, per_test_timeout, max_asserts, overall_timeout = args
    try:
        res = run_lc_candidate(
            candidate_code, test_code, entry_point,
            per_test_timeout=per_test_timeout,
            max_asserts=max_asserts,
            overall_timeout=overall_timeout,
        )
        return idx, res, None
    except Exception as e:
        return idx, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}"


def _stdio_worker(args):
    idx, code, tests, per_test_timeout = args
    try:
        res = run_stdio_python(code, tests, per_test_timeout=per_test_timeout)
        return idx, res, None
    except Exception as e:
        return idx, None, f"{type(e).__name__}: {e}"


# --------------------------- driver ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lc", "stdio"])
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-test-timeout", type=float, default=5.0)
    ap.add_argument("--max-asserts", type=int, default=40,
                    help="LC: cap asserts to avoid runaway test files")
    ap.add_argument("--overall-timeout", type=float, default=60.0)
    args = ap.parse_args()

    cands = pd.read_parquet(args.candidates)
    tests = pd.read_parquet(args.tests)
    print(f"[run_candidate_tests] candidates={len(cands)} tests_table={len(tests)}", flush=True)

    out_rows: list[dict] = []
    if args.mode == "lc":
        # Required cand cols: candidate_id, question_slug, code
        # Required test cols: task_id, test, entry_point
        tmap = tests.set_index("task_id")[["test", "entry_point"]].to_dict("index")
        work = []
        skipped = 0
        for i, r in cands.iterrows():
            slug = r["question_slug"]
            if slug not in tmap:
                skipped += 1
                continue
            tinfo = tmap[slug]
            work.append((
                i,
                r["code"],
                tinfo["test"],
                tinfo["entry_point"],
                args.per_test_timeout,
                args.max_asserts,
                args.overall_timeout,
            ))
        print(f"[run_candidate_tests] runnable={len(work)} skipped_no_tests={skipped}", flush=True)
        if args.limit:
            work = work[:args.limit]

        t_start = time.time()
        with mp.Pool(processes=args.workers) as pool:
            for done_i, (idx, res, err) in enumerate(
                pool.imap_unordered(_lc_worker, work, chunksize=4)
            ):
                row = {"_idx": idx}
                if res is not None:
                    row.update(res)
                else:
                    row["compiled"] = None
                    row["worker_error"] = err
                out_rows.append(row)
                if (done_i + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = (done_i + 1) / elapsed
                    eta = (len(work) - done_i - 1) / max(rate, 1e-6)
                    print(f"  [{done_i+1}/{len(work)}] rate={rate:.1f}/s eta={eta:.0f}s", flush=True)

        # Join back to cands
        out_df = pd.DataFrame(out_rows).set_index("_idx").reindex(cands.index)
        merged = cands.reset_index(drop=True).join(out_df.reset_index(drop=True), rsuffix="_exec")
        # Keep ID + exec features
        keep_cols = [
            "candidate_id", "question_slug", "compiled", "n_tests", "n_pass",
            "n_fail", "n_timeout", "n_runtime_err",
            "wall_ms_median", "wall_ms_max", "pass_rate", "first_error",
        ]
        if "pair_id" in merged.columns:
            keep_cols = ["pair_id"] + keep_cols
        merged = merged[[c for c in keep_cols if c in merged.columns]]
    else:  # stdio
        # tests parquet has columns: canonical_pid (or problem_id) + tests (list of {input,output})
        tmap = tests.set_index("canonical_pid")["tests"].to_dict() if "canonical_pid" in tests.columns else tests.set_index("problem_id")["tests"].to_dict()
        work = []
        skipped = 0
        for i, r in cands.iterrows():
            pid = r.get("canonical_pid") or r.get("problem_id")
            if pid not in tmap:
                skipped += 1
                continue
            tcases = tmap[pid]
            # tcases could be a list of dicts or a json string
            if isinstance(tcases, str):
                tcases = json.loads(tcases)
            work.append((i, r["code"] if "code" in r else r["candidate_code"], tcases,
                         args.per_test_timeout))
        print(f"[run_candidate_tests] runnable={len(work)} skipped_no_tests={skipped}", flush=True)
        if args.limit:
            work = work[:args.limit]
        with mp.Pool(processes=args.workers) as pool:
            for done_i, (idx, res, err) in enumerate(
                pool.imap_unordered(_stdio_worker, work, chunksize=2)
            ):
                row = {"_idx": idx}
                if res is not None:
                    row.update(res)
                else:
                    row["compiled"] = None
                    row["worker_error"] = err
                out_rows.append(row)
                if (done_i + 1) % 50 == 0:
                    print(f"  [{done_i+1}/{len(work)}]", flush=True)
        out_df = pd.DataFrame(out_rows).set_index("_idx").reindex(cands.index)
        merged = cands.reset_index(drop=True).join(out_df.reset_index(drop=True), rsuffix="_exec")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out)
    print(f"[run_candidate_tests] wrote {args.out} shape={merged.shape}", flush=True)


if __name__ == "__main__":
    main()

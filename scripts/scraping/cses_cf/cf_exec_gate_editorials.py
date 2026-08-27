"""Execution-gate the rendered-HTML CF editorial codes against TACO tests.

Inputs:
  datasets/codeforces_delta/editorials_rendered_extracted.parquet
  datasets/codeforces_delta/cf_tests_taco.parquet

For every extracted editorial code whose canonical_pid has tests:
  - python  -> run via run_stdio_python (reused from
               scripts/competition_exec/run_candidate_tests.py)
  - cpp     -> g++ -O2 -std=gnu++17 compile gate, then run binary per test
               (same _outputs_equal comparison, 5 s/test, 512 MB)
  - other   -> recorded as lang_unsupported (no gate)

Output: datasets/codeforces_delta/editorials_rendered_gated.parquet
  original columns + [has_tests, n_tests, compiled, n_pass, n_fail, n_timeout,
  n_runtime_err, pass_rate, first_error, gate_pass]
  gate_pass = compiled AND pass_rate == 1.0

CPU only. APPEND-ONLY (writes its own new output file).
"""
from __future__ import annotations

import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(ROOT / "scripts/competition_exec"))
from run_candidate_tests import run_stdio_python, _outputs_equal  # noqa: E402

ED = ROOT / "datasets/codeforces_delta/editorials_rendered_extracted.parquet"
TESTS = ROOT / "datasets/codeforces_delta/cf_tests_taco.parquet"
OUT = ROOT / "datasets/codeforces_delta/editorials_rendered_gated.parquet"

PER_TEST_TIMEOUT = 5.0
COMPILE_TIMEOUT = 45.0
MEM_MB = 1024
MAX_TESTS = 15


def _rlimits():
    lim = MEM_MB * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:
        pass


def run_stdio_cpp(code: str, tests: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "sol.cpp"
        binp = Path(td) / "sol"
        src.write_text(code)
        try:
            cp = subprocess.run(
                ["g++", "-O2", "-std=gnu++17", "-o", str(binp), str(src)],
                capture_output=True, timeout=COMPILE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return dict(compiled=False, n_tests=len(tests), n_pass=0, n_fail=0,
                        n_timeout=0, n_runtime_err=len(tests), wall_ms_median=None,
                        wall_ms_max=None, pass_rate=0.0, first_error="compile TLE")
        if cp.returncode != 0:
            msg = cp.stderr.decode("utf-8", errors="replace")
            line = next((l for l in msg.splitlines() if "error" in l), msg[:120])
            return dict(compiled=False, n_tests=len(tests), n_pass=0, n_fail=0,
                        n_timeout=0, n_runtime_err=len(tests), wall_ms_median=None,
                        wall_ms_max=None, pass_rate=0.0,
                        first_error=f"CE: {line[:160]}")
        n_pass = n_fail = n_timeout = n_rte = 0
        walls, first_error = [], None
        for t in tests:
            t0 = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    [str(binp)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, preexec_fn=_rlimits,
                )
                try:
                    out, err = proc.communicate(input=t["input"].encode(),
                                                timeout=PER_TEST_TIMEOUT)
                    walls.append((time.perf_counter() - t0) * 1000)
                    if proc.returncode != 0:
                        n_rte += 1
                        if first_error is None:
                            first_error = f"rc={proc.returncode}"
                        continue
                    actual = out.decode("utf-8", errors="replace")
                    if _outputs_equal(t["output"], actual):
                        n_pass += 1
                    else:
                        n_fail += 1
                        if first_error is None:
                            first_error = (f"WA: exp={t['output'][:30]!r} "
                                           f"got={actual[:30]!r}")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.communicate(timeout=2)
                    except Exception:
                        pass
                    walls.append((time.perf_counter() - t0) * 1000)
                    n_timeout += 1
                    if first_error is None:
                        first_error = "TLE"
            except Exception as e:
                n_rte += 1
                if first_error is None:
                    first_error = f"LAUNCH: {e}"
        n = len(tests)
        return dict(compiled=True, n_tests=n, n_pass=n_pass, n_fail=n_fail,
                    n_timeout=n_timeout, n_runtime_err=n_rte,
                    wall_ms_median=statistics.median(walls) if walls else None,
                    wall_ms_max=max(walls) if walls else None,
                    pass_rate=n_pass / n if n else None,
                    first_error=first_error)


def _worker(args):
    idx, lang, code, tests_json = args
    tests = json.loads(tests_json)[:MAX_TESTS]
    try:
        if lang == "python":
            res = run_stdio_python(code, tests, per_test_timeout=PER_TEST_TIMEOUT)
        elif lang == "cpp":
            res = run_stdio_cpp(code, tests)
        else:
            return idx, dict(compiled=None, n_tests=0, n_pass=0, n_fail=0,
                             n_timeout=0, n_runtime_err=0, wall_ms_median=None,
                             wall_ms_max=None, pass_rate=None,
                             first_error="lang_unsupported")
        return idx, res
    except Exception as e:
        return idx, dict(compiled=None, n_tests=0, n_pass=0, n_fail=0,
                         n_timeout=0, n_runtime_err=0, wall_ms_median=None,
                         wall_ms_max=None, pass_rate=None,
                         first_error=f"HARNESS: {type(e).__name__}: {e}")


def main():
    ed = pd.read_parquet(ED)
    tests = pd.read_parquet(TESTS)
    t_map = dict(zip(tests["canonical_pid"], tests["tests_json"]))
    ed = ed.copy()
    ed["has_tests"] = ed["canonical_pid"].map(lambda p: p in t_map if p else False)
    todo = ed[ed["has_code"] & ed["has_tests"]].copy()
    print(f"editorial rows: {len(ed)}; with code: {int(ed['has_code'].sum())}; "
          f"code+tests to gate: {len(todo)}", flush=True)
    work = [
        (i, todo.loc[i, "code_lang"], todo.loc[i, "extracted_code"],
         t_map[todo.loc[i, "canonical_pid"]])
        for i in todo.index
    ]
    results = {}
    n_workers = int(os.environ.get("GATE_WORKERS", "16"))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_worker, w) for w in work]
        for k, f in enumerate(as_completed(futs)):
            idx, res = f.result()
            results[idx] = res
            if (k + 1) % 100 == 0:
                print(f"{k+1}/{len(work)} elapsed={time.time()-t0:.0f}s", flush=True)
    feat_cols = ["compiled", "n_tests", "n_pass", "n_fail", "n_timeout",
                 "n_runtime_err", "wall_ms_median", "wall_ms_max", "pass_rate",
                 "first_error"]
    for c in feat_cols:
        ed[c] = None
    for idx, res in results.items():
        for c in feat_cols:
            ed.at[idx, c] = res.get(c)
    # exec n_tests separate from tests bank n_tests naming: keep as-is
    ed["pass_rate"] = pd.to_numeric(ed["pass_rate"], errors="coerce")
    ed["gate_pass"] = (ed["compiled"] == True) & (ed["pass_rate"] == 1.0)  # noqa: E712
    ed.to_parquet(OUT, index=False)
    gated = ed[ed["has_code"] & ed["has_tests"]]
    print(f"wrote {OUT}")
    print(f"gated rows: {len(gated)}; compiled: {int((gated['compiled']==True).sum())}; "
          f"gate_pass (all tests): {int(gated['gate_pass'].sum())} "
          f"({gated['gate_pass'].mean():.1%})")
    print("pass_rate distribution:", gated["pass_rate"].describe().to_string())


if __name__ == "__main__":
    main()

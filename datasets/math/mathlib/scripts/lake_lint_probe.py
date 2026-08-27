#!/usr/bin/env python3
"""Ground-truth lint/warning probe over already lake-probed mathlib first-push states.

For each PR in lake_probe_results.csv with status in (built, build_failed):
  1. re-checkout first-push state in a throwaway worktree (same idioms as
     lake_build_probe.py), `lake exe cache get <files>`,
  2. DELETE the touched modules' own build artifacts so `lake build` genuinely
     re-elaborates them (deps stay cache-hot), capture FULL build output:
     - count `warning:` lines by kind (deprecated / unused_variable / linter / other)
     - t_rebuild = wall time of the forced rebuild (crude proof-weight proxy)
     - for failures: save full text, classify infra vs lean_error vs lint_as_error,
       keep the first real error line
  3. run the genuine mathlib linter suite per touched module:
     `lake exe runLinter <Module.Name>` (fallback for eras where the exe is
     missing -> lint_available=false). Parse #check-error count and linter names.

Resume-safe: appends to --out, skips PR numbers already present.
Full raw outputs land in lint_probe_logs/pr{N}.txt.

Usage (sk3):
  HOME=/lfs/skampere3/0/alexspan ~/envs/norm-scraper/bin/python \
      lake_lint_probe.py [--numbers 622,4719] [--limit N]
"""
import argparse
import csv
import glob
import os
import re
import shutil
import signal
import subprocess
import time
from collections import Counter

import pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
REPO = f"{BASE}/mathlib4_repo"
HOME = "/lfs/skampere3/0/alexspan"
ENV = dict(os.environ, HOME=HOME,
           PATH=f"{HOME}/.elan/bin:" + os.environ.get("PATH", ""))
LOGDIR = f"{BASE}/lint_probe_logs"
LINT_BUDGET = 600  # max total seconds of runLinter per PR


def run(args, cwd=None, timeout=3600):
    # start_new_session + killpg: `lake exe` spawns child binaries (e.g.
    # runLinter) that would survive a plain subprocess.run timeout kill.
    p = subprocess.Popen(args, cwd=cwd, env=ENV, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args, p.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            out, _ = p.communicate(timeout=30)
        except Exception:
            out = ""
        return subprocess.CompletedProcess(args, -9, out or "", "TIMEOUT")


def git(*a, timeout=600):
    return run(["git", "-C", REPO, *a], timeout=timeout)


# ---------- parsing ----------

def classify_warning(line):
    l = line.lower()
    if "deprecated" in l:
        return "deprecated"
    if "unused variable" in l:
        return "unused_variable"
    if "linter." in l or "this linter can be disabled" in l or "unused" in l:
        return "linter"
    return "other"


# lake style: `error: Mathlib/Foo.lean:268:15: type mismatch`
# lean style: `Mathlib/Foo.lean:268:15: error: unknown identifier`
LEAN_ERR_RES = [
    re.compile(r"^error: \S+\.lean:\d+:\d+:.*$", re.M),
    re.compile(r"^(?:info: )?\S+\.lean:\d+:\d+:\s*error.*$", re.M),
]
INFRA_PATTERNS = [
    "binary package was not provided", "no such file or directory",
    "could not download", "failed to fetch", "invalid manifest",
    "unknown executable", "unknown target", "unknown module",
    "toolchain", "error: stdin", "object file", "ld.lld", "curl",
    "invalid lakefile", "package configuration", "missing module",
]
LINTY_ERR = ("unused variable", "deprecated", "linter", "unnecessary")


def classify_failure(text):
    """-> (failure_class, first_error_line)"""
    hits = [m for rx in LEAN_ERR_RES for m in [rx.search(text)] if m]
    if hits:
        m = min(hits, key=lambda m: m.start())
        line = m.group(0).strip()
        low = line.lower()
        # -DwarningAsError eras surface lint warnings as build errors
        if any(p in low for p in LINTY_ERR):
            return "lint_as_error", line[:400]
        return "lean_error", line[:400]
    for ln in text.split("\n"):
        if "error" in ln.lower():
            low = ln.lower()
            if any(p in low for p in INFRA_PATTERNS):
                return "infra", ln.strip()[:400]
    for ln in text.split("\n"):
        if "error" in ln.lower():
            return "infra", ln.strip()[:400]
    return "infra", ""


FOUND_RE = re.compile(r"-- Found (\d+) error")
LINT_ERRLINE_RE = re.compile(r"\.lean:\d+:\d+:\s*error", re.M)
CHECK_RE = re.compile(r"^#check @", re.M)
LINT_PASSED_RE = re.compile(r"-- Linting passed")
LINT_UNAVAILABLE = ("unknown executable", "unknown script", "unknown target",
                    "no executable", "unknown command")


def parse_lint_output(text):
    """Returns (n_errors, Counter{linter_name: n}, parsed_ok).

    batteries/std runLinter prints `-- Found N errors ...` plus per-linter
    blocks `/- The \\`name\\` linter reports: ... -/` whose bodies are either
    `#check @decl` lines (old format) or `file:line:col: error:` lines
    (useErrorFormat). `-- Linting passed` when clean.
    """
    names = Counter()
    chunks = re.split(r"/- The `", text)
    for chunk in chunks[1:]:
        name = chunk.split("`", 1)[0]
        cnt = (len(CHECK_RE.findall(chunk))
               or len(LINT_ERRLINE_RE.findall(chunk))
               or 1)
        names[name] += cnt
    m = FOUND_RE.search(text)
    if m:
        return int(m.group(1)), names, True
    if names:
        return sum(names.values()), names, True
    if LINT_PASSED_RE.search(text):
        return 0, names, True
    return 0, names, False


# ---------- per-PR work ----------

def touched_files(oid):
    mb = git("merge-base", oid, "origin/master").stdout.strip()
    if not mb:
        return []
    return [f for f in git("diff", "--name-only", "--diff-filter=AM",
                           mb, oid).stdout.split()
            if f.endswith(".lean") and f.startswith("Mathlib/")]


def nuke_artifacts(wt, files):
    """Delete the touched modules' own build products so lake re-elaborates."""
    n = 0
    for f in files:
        rel = f[:-5]  # strip .lean
        for pat in (f"{wt}/build/lib/{rel}.*",
                    f"{wt}/.lake/build/lib/{rel}.*",
                    f"{wt}/.lake/build/lib/lean/{rel}.*",
                    f"{wt}/build/ir/{rel}.*",
                    f"{wt}/.lake/build/ir/{rel}.*"):
            for p in glob.glob(pat):
                try:
                    os.remove(p)
                    n += 1
                except OSError:
                    pass
    return n


FIELDS = ["number", "year", "y", "oid", "n_files", "toolchain", "status_prev",
          "t_cache", "t_rebuild", "build_exit",
          "n_warn_total", "n_warn_deprecated", "n_warn_unused_variable",
          "n_warn_linter", "n_warn_other",
          "lint_available", "n_lint_modules", "n_lint_errors",
          "lint_linters", "n_lint_timeouts", "t_lint",
          "failure_class", "first_error_line", "status"]


def process_pr(row, wt, build_timeout, lint_timeout):
    n = int(row.number)
    rec = {k: "" for k in FIELDS}
    rec.update(number=n, year=int(row.year), y=int(row.y), oid=row.oid,
               status_prev=row.status)
    log_lines = []

    rp = git("rev-parse", f"{row.oid}^{{commit}}")
    if rp.returncode != 0:
        rec["status"] = "rev_parse_failed"
        return rec, ""
    oid = rp.stdout.strip()
    files = touched_files(oid)
    rec["n_files"] = len(files)
    if not files:
        rec["status"] = "no_files"
        return rec, ""

    git("worktree", "remove", "--force", wt)
    shutil.rmtree(wt, ignore_errors=True)
    if git("worktree", "add", "--detach", wt, oid).returncode != 0:
        rec["status"] = "worktree_failed"
        return rec, ""
    try:
        rec["toolchain"] = open(f"{wt}/lean-toolchain").read().strip()

        t0 = time.time()
        c = run(["lake", "exe", "cache", "get", *files], cwd=wt, timeout=3600)
        rec["t_cache"] = round(time.time() - t0, 1)
        log_lines.append(f"===== CACHE (exit {c.returncode}) =====\n"
                         f"{c.stdout}\n{c.stderr}")

        nuked = nuke_artifacts(wt, files)
        log_lines.append(f"===== nuked {nuked} artifacts =====")

        mods = [f[:-5].replace("/", ".") for f in files]
        t0 = time.time()
        b = run(["lake", "build", *mods], cwd=wt, timeout=build_timeout)
        rec["t_rebuild"] = round(time.time() - t0, 1)
        rec["build_exit"] = b.returncode
        btext = (b.stdout or "") + "\n" + (b.stderr or "")
        log_lines.append(f"===== BUILD (exit {b.returncode}, "
                         f"{rec['t_rebuild']}s) =====\n{btext}")

        kinds = Counter()
        for ln in btext.split("\n"):
            if "warning:" in ln:
                kinds[classify_warning(ln)] += 1
        rec["n_warn_total"] = sum(kinds.values())
        for k in ("deprecated", "unused_variable", "linter", "other"):
            rec[f"n_warn_{k}"] = kinds.get(k, 0)

        if b.returncode != 0:
            if b.stderr == "TIMEOUT":
                rec["status"] = "build_timeout"
            else:
                rec["failure_class"], rec["first_error_line"] = \
                    classify_failure(btext)
                rec["status"] = "build_failed"
            return rec, "\n".join(log_lines)

        # ---- linter suite ----
        lint_ok, n_to, t_lint = None, 0, 0.0
        all_names = Counter()
        n_lint_errors = 0
        n_mods_linted = 0
        for mod in mods:
            remaining = LINT_BUDGET - t_lint
            if remaining <= 5:
                n_to += 1  # budget exhausted: count unprocessed as timeout
                continue
            t0 = time.time()
            lr = run(["lake", "exe", "runLinter", mod], cwd=wt,
                     timeout=min(lint_timeout, remaining))
            t_lint += time.time() - t0
            ltext = (lr.stdout or "") + "\n" + (lr.stderr or "")
            log_lines.append(f"===== LINT {mod} (exit {lr.returncode}, "
                             f"{round(time.time()-t0,1)}s) =====\n{ltext}")
            if lr.stderr == "TIMEOUT":
                n_to += 1
                continue
            low = ltext.lower()
            if lr.returncode != 0 and any(p in low for p in LINT_UNAVAILABLE):
                lint_ok = False
                break
            nc, names, parsed = parse_lint_output(ltext)
            if not parsed and lr.returncode != 0:
                # ran but crashed for a non-lint reason
                log_lines.append(f"[lint crash, no parse] exit {lr.returncode}")
                continue
            lint_ok = True
            n_mods_linted += 1
            n_lint_errors += nc
            all_names.update(names)
        rec["lint_available"] = (int(lint_ok) if lint_ok is not None else 0)
        rec["n_lint_modules"] = n_mods_linted
        rec["n_lint_errors"] = n_lint_errors if lint_ok else ""
        rec["lint_linters"] = ";".join(f"{k}:{v}" for k, v in
                                       all_names.most_common(10))
        rec["n_lint_timeouts"] = n_to
        rec["t_lint"] = round(t_lint, 1)
        rec["status"] = "ok"
        return rec, "\n".join(log_lines)
    except Exception as e:
        rec["status"] = f"exception:{type(e).__name__}"
        return rec, "\n".join(log_lines) + f"\nEXCEPTION {e!r}"
    finally:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default=f"{BASE}/lake_probe_results.csv")
    ap.add_argument("--out", default=f"{BASE}/lake_lint_results.csv")
    ap.add_argument("--numbers", default="",
                    help="comma-separated PR numbers (validation mode)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--build-timeout", type=int, default=1800)
    ap.add_argument("--lint-timeout", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)
    df = pd.read_csv(args.in_csv)
    df = df[df.status.isin(["built", "build_failed"])].drop_duplicates(
        subset="number", keep="first")

    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out).number.astype(int))

    if args.numbers:
        want = [int(x) for x in args.numbers.split(",")]
        rows = [df[df.number == n].iloc[0] for n in want
                if n not in done and (df.number == n).any()]
    else:
        # round-robin by year so partial runs stay year-balanced
        by_year = {y: list(g.itertuples()) for y, g in df.groupby("year")}
        rows, i = [], 0
        while any(by_year.values()):
            for y in sorted(by_year):
                if by_year[y]:
                    r = by_year[y].pop(0)
                    if int(r.number) not in done:
                        rows.append(r)
        if args.limit:
            rows = rows[:args.limit]

    print(f"processing {len(rows)} PRs ({len(done)} already done)", flush=True)
    write_header = not os.path.exists(args.out)
    wt = f"{BASE}/probe_wt"
    t_start = time.time()
    with open(args.out, "a", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for i, row in enumerate(rows):
            n = int(row.number)
            print(f"[{time.strftime('%H:%M:%S')}] ({i+1}/{len(rows)}) PR {n} "
                  f"({row.year}, y={row.y}, prev={row.status})", flush=True)
            rec, log = process_pr(row, wt, args.build_timeout,
                                  args.lint_timeout)
            with open(f"{LOGDIR}/pr{n}.txt", "w") as lf:
                lf.write(log)
            w.writerow(rec)
            fout.flush()
            print(f"    -> {rec['status']} warn={rec['n_warn_total']} "
                  f"lint_avail={rec['lint_available']} "
                  f"lint_err={rec['n_lint_errors']} "
                  f"fail={rec['failure_class']} "
                  f"t_rebuild={rec['t_rebuild']} t_lint={rec['t_lint']}",
                  flush=True)
        git("worktree", "remove", "--force", wt)
        shutil.rmtree(wt, ignore_errors=True)
    print(f"DONE in {round((time.time()-t_start)/60,1)} min", flush=True)


if __name__ == "__main__":
    main()

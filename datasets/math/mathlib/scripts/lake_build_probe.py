#!/usr/bin/env python3
"""Stratified lake-build probe: does the FIRST-PUSH state of a mathlib PR compile?

Samples PRs per year from friction_dataset_v2, checks out each first-commit
state in a throwaway worktree, fetches the build cache for the touched files
only (`lake exe cache get <files>` — content-hash keyed, so old states hit),
and builds the touched modules only. Records per-PR cache/build outcomes.

The probe answers two questions:
 1. cache hit-rate by era (the feasibility number for scaling to 500+)
 2. first-push build-failure rate by friction class y (the V signal itself)

Resume-safe: appends to --out, skips PR numbers already present.

Usage (sk3):
  ~/envs/norm-scraper/bin/python lake_build_probe.py --n-per-year 4
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import time

import pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
REPO = f"{BASE}/mathlib4_repo"
HOME = "/lfs/skampere3/0/alexspan"
ENV = dict(os.environ, HOME=HOME,
           PATH=f"{HOME}/.elan/bin:" + os.environ.get("PATH", ""))


def run(args, cwd=None, timeout=3600):
    try:
        return subprocess.run(args, cwd=cwd, env=ENV, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        r = subprocess.CompletedProcess(args, returncode=-9)
        r.stdout = (e.stdout or b"").decode(errors="replace") if isinstance(
            e.stdout, bytes) else (e.stdout or "")
        r.stderr = "TIMEOUT"
        return r


def git(*a, timeout=600):
    return run(["git", "-C", REPO, *a], timeout=timeout)


def pick_targets(df, years, n_per_year, max_files, seed, done):
    targets = []
    for year in years:
        pool = df[(df.year == year) & df.first_commit_oid.notna()].sample(
            frac=1.0, random_state=seed)
        picked = 0
        for _, r in pool.iterrows():
            if picked >= n_per_year:
                break
            if int(r.number) in done:
                picked += 1  # count prior runs toward the stratum
                continue
            rp = git("rev-parse", f"{r.first_commit_oid}^{{commit}}")
            if rp.returncode != 0:
                continue
            oid = rp.stdout.strip()
            mb = git("merge-base", oid, "origin/master").stdout.strip()
            if not mb:
                continue
            files = [f for f in git("diff", "--name-only", "--diff-filter=AM",
                                    mb, oid).stdout.split()
                     if f.endswith(".lean") and f.startswith("Mathlib/")]
            if not (1 <= len(files) <= max_files):
                continue
            targets.append(dict(number=int(r.number), y=int(r.y), year=year,
                                oid=oid, files=files))
            picked += 1
    return targets


FIELDS = ["number", "year", "y", "oid", "n_files", "toolchain",
          "cache_attempted", "cache_warned_missing", "cache_exit",
          "build_exit", "build_errors_head", "t_cache", "t_build", "status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-year", type=int, default=4)
    ap.add_argument("--years", default="2022,2023,2024,2025,2026")
    ap.add_argument("--out", default=f"{BASE}/lake_probe_results.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=5)
    ap.add_argument("--build-timeout", type=int, default=1800)
    args = ap.parse_args()

    df = pd.read_csv(f"{BASE}/friction_dataset_v2.csv.gz")
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out).number.astype(int))

    years = [int(y) for y in args.years.split(",")]
    targets = pick_targets(df, years, args.n_per_year, args.max_files,
                           args.seed, done)
    print(f"probing {len(targets)} PRs "
          f"({[(t['number'], t['year']) for t in targets]})", flush=True)

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        wt = f"{BASE}/probe_wt"
        for t in targets:
            n, oid = t["number"], t["oid"]
            rec = {k: "" for k in FIELDS}
            rec.update(number=n, year=t["year"], y=t["y"], oid=oid[:12],
                       n_files=len(t["files"]))
            print(f"[{time.strftime('%H:%M:%S')}] PR {n} ({t['year']}, "
                  f"y={t['y']}, {len(t['files'])} files)", flush=True)
            git("worktree", "remove", "--force", wt)
            if git("worktree", "add", "--detach", wt, oid).returncode != 0:
                rec["status"] = "worktree_failed"
                w.writerow(rec); fout.flush(); continue
            try:
                rec["toolchain"] = open(f"{wt}/lean-toolchain").read().strip()
                t0 = time.time()
                c = run(["lake", "exe", "cache", "get", *t["files"]],
                        cwd=wt, timeout=3600)
                rec["t_cache"] = round(time.time() - t0, 1)
                rec["cache_exit"] = c.returncode
                out = (c.stdout or "") + (c.stderr or "")
                m = re.search(r"download (\d+) file", out)
                rec["cache_attempted"] = m.group(1) if m else ""
                rec["cache_warned_missing"] = int(
                    "not found in the cache" in out)
                t0 = time.time()
                mods = [f[:-5].replace("/", ".") for f in t["files"]]
                b = run(["lake", "build", *mods], cwd=wt,
                        timeout=args.build_timeout)
                rec["t_build"] = round(time.time() - t0, 1)
                rec["build_exit"] = b.returncode
                if b.returncode == 0:
                    rec["status"] = "built"
                elif b.stderr == "TIMEOUT":
                    rec["status"] = "build_timeout"
                else:
                    errs = [l for l in (b.stdout + "\n" + b.stderr).split("\n")
                            if "error" in l.lower()][:3]
                    rec["build_errors_head"] = " | ".join(errs)[:400]
                    rec["status"] = "build_failed"
            except Exception as e:
                rec["status"] = f"exception:{type(e).__name__}"
            w.writerow(rec); fout.flush()
            print(f"    -> {rec['status']} (cache {rec['t_cache']}s, "
                  f"build {rec['t_build']}s)", flush=True)
        git("worktree", "remove", "--force", wt)
        shutil.rmtree(wt, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

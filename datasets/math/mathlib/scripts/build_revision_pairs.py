#!/usr/bin/env python3
"""Build the mathlib "revision pairs" corpus (label 3b).

Context: datasets/math/mathlib/README.md (§4 data plan). For each review-
thread comment on a mathlib4 PR, pair the reviewer's comment with the code
region it targeted in the FIRST-PUSH state and the same region in the FINAL
(merged-head) state — the math analog of (review comment -> code revision)
supervision.

Inputs (all under BASE on sk3):
- pr_thread_comments.jsonl  — per-PR review-thread comment text (from
  fetch_thread_comments_graphql.py); lines with {"error": ...} are skipped.
- friction_full_v2.csv.gz   — gives first_commit_oid, y, year per PR.
- mathlib4_repo             — full clone with refs/remotes/pr/N for all
  dataset PRs (fetched as +pull/N/head:refs/remotes/pr/N).

State resolution (mirrors firstpush_pilot.py):
- first-push rev = first_commit_oid if `git cat-file -e <oid>^{commit}`
  succeeds ("first_commit"), else refs/remotes/pr/N ("head_fallback"),
  else the PR is skipped ("unresolved", logged).
- final rev = refs/remotes/pr/N (the merged PR head).

Output: revision_pairs.jsonl — one JSON line per review THREAD:
{pr, year, y, path, line, is_resolved, is_outdated, n_comments,
 comments: [{author, body, createdAt}], first_push_excerpt, final_excerpt,
 first_push_state, file_in_first_push, file_in_final}
Excerpts are +/-25 lines around `line` (file head if line is null) from
`git show <rev>:<path>`; null + bool=false if the file is absent at that rev.

Resume-safe: skips (pr, path, line, first_comment_createdAt) keys already in
the output; append-only with periodic flush. Per PR, revs are resolved once
and `git show` is cached per (rev, path).

Not every PR in pr_thread_comments.jsonl has a pr/N ref in the clone
(~7.5K were never fetched); --fetch-missing batch-fetches them first
(+pull/N/head:refs/remotes/pr/N, 50 refspecs per fetch, per-ref retry on
batch failure — same idiom as firstpush_pilot.batch_fetch).

Usage (sk3, CPU-only):
  export HOME=/lfs/skampere3/0/alexspan
  python3 build_revision_pairs.py --fetch-missing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
REPO = os.path.join(BASE, "mathlib4_repo")
GIT_ENV = dict(os.environ, HOME="/lfs/skampere3/0/alexspan")

CONTEXT = 25  # lines of context on each side of the commented line

# ---------------------------------------------------------------- git helpers


def git(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", REPO, *args], env=GIT_ENV,
                          capture_output=True, text=True, errors="replace",
                          timeout=timeout)


def object_exists(oid: str) -> bool:
    return git("cat-file", "-e", f"{oid}^{{commit}}").returncode == 0


def resolve_revs(number: int, first_commit_oid) -> tuple[str | None, str | None, str | None]:
    """Return (first_rev, first_state, final_rev)."""
    pr_ref = f"refs/remotes/pr/{number}"
    have_ref = git("rev-parse", "--verify", "--quiet", pr_ref).returncode == 0
    final_rev = pr_ref if have_ref else None
    if isinstance(first_commit_oid, str) and object_exists(first_commit_oid):
        return first_commit_oid, "first_commit", final_rev
    if have_ref:
        return pr_ref, "head_fallback", final_rev
    return None, None, final_rev


def fetch_missing_refs(numbers: list[int], batch: int = 50) -> None:
    """Fetch refs/pull/N/head -> refs/remotes/pr/N for PRs lacking the ref."""
    have = {int(x) for x in git(
        "for-each-ref", "refs/remotes/pr/", "--format=%(refname:lstrip=3)",
        timeout=300).stdout.split()}
    todo = sorted(set(numbers) - have)
    print(f"fetch-missing: {len(todo)} PRs lack a pr/N ref", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        refspecs = [f"+pull/{n}/head:refs/remotes/pr/{n}" for n in chunk]
        r = git("fetch", "origin", "--no-tags", "--quiet", *refspecs,
                timeout=1200)
        if r.returncode != 0:
            print(f"  batch {i//batch} failed (rc={r.returncode}); "
                  f"retrying per-ref", flush=True)
            for n in chunk:
                r1 = git("fetch", "origin", "--no-tags", "--quiet",
                         f"+pull/{n}/head:refs/remotes/pr/{n}", timeout=300)
                if r1.returncode != 0:
                    print(f"    pr {n}: fetch failed: "
                          f"{r1.stderr.strip()[:200]}", flush=True)
        if (i // batch) % 10 == 0:
            print(f"  fetched batch {i//batch + 1}/"
                  f"{(len(todo)+batch-1)//batch}", flush=True)


def show_file(rev: str, path: str, cache: dict) -> list[str] | None:
    """File content at rev as a list of lines, or None if absent. Cached."""
    key = (rev, path)
    if key not in cache:
        r = git("show", f"{rev}:{path}")
        cache[key] = r.stdout.split("\n") if r.returncode == 0 else None
    return cache[key]


def excerpt(lines: list[str] | None, line: int | None) -> str | None:
    if lines is None:
        return None
    if line is None:
        return "\n".join(lines[: 2 * CONTEXT])
    lo = max(0, int(line) - 1 - CONTEXT)
    hi = int(line) + CONTEXT
    return "\n".join(lines[lo:hi])

# ---------------------------------------------------------------- main


def thread_key(pr: int, path, line, created):
    return f"{pr}\x00{path}\x00{line}\x00{created}"


def load_done(out_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            created = d["comments"][0]["createdAt"] if d.get("comments") else None
            done.add(thread_key(d["pr"], d.get("path"), d.get("line"), created))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads-jsonl",
                    default=os.path.join(BASE, "pr_thread_comments.jsonl"))
    ap.add_argument("--friction-full",
                    default=os.path.join(BASE, "friction_full_v2.csv.gz"))
    ap.add_argument("--out", default=os.path.join(BASE, "revision_pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N PRs (0 = all; for smoke tests)")
    ap.add_argument("--fetch-missing", action="store_true",
                    help="batch-fetch pr/N refs absent from the clone first")
    args = ap.parse_args()

    if args.fetch_missing:
        numbers = []
        with open(args.threads_jsonl) as f:
            for raw in f:
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not rec.get("error") and "reviewThreads" in rec:
                    numbers.append(int(rec["number"]))
        fetch_missing_refs(numbers)

    meta = pd.read_csv(args.friction_full,
                       usecols=["number", "first_commit_oid", "y", "year"],
                       low_memory=False).set_index("number")

    done = load_done(args.out)
    print(f"resume: {len(done)} thread keys already in {args.out}", flush=True)

    n_pr = n_threads = n_skipped = n_unresolved = n_nometa = 0
    out = open(args.out, "a")
    with open(args.threads_jsonl) as f:
        for raw in f:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if rec.get("error") or "reviewThreads" not in rec:
                continue
            pr = int(rec["number"])
            n_pr += 1
            if args.limit and n_pr > args.limit:
                break
            if n_pr % 500 == 0:
                print(f"[{n_pr} PRs] threads written={n_threads} "
                      f"resumed/skipped={n_skipped} unresolved={n_unresolved}",
                      flush=True)
                out.flush()

            threads = rec["reviewThreads"].get("nodes") or []
            if not threads:
                continue
            if pr not in meta.index:
                n_nometa += 1
                continue
            m = meta.loc[pr]

            # cheap resume pre-check: skip PR entirely if all threads done
            keys = []
            for t in threads:
                comments = (t.get("comments") or {}).get("nodes") or []
                created = comments[0]["createdAt"] if comments else None
                keys.append(thread_key(pr, t.get("path"), t.get("line"), created))
            if all(k in done for k in keys):
                n_skipped += len(keys)
                continue

            first_rev, first_state, final_rev = resolve_revs(
                pr, m.first_commit_oid)
            if first_rev is None:
                n_unresolved += 1
                print(f"  PR {pr}: unresolved (no first_commit, no pr ref)",
                      flush=True)
                continue

            cache: dict = {}
            for t, key in zip(threads, keys):
                if key in done:
                    n_skipped += 1
                    continue
                comments = (t.get("comments") or {}).get("nodes") or []
                if not comments:
                    continue
                path, line = t.get("path"), t.get("line")
                first_lines = show_file(first_rev, path, cache)
                final_lines = (show_file(final_rev, path, cache)
                               if final_rev else None)
                row = {
                    "pr": pr,
                    "year": int(m.year),
                    "y": int(m.y),
                    "path": path,
                    "line": line,
                    "is_resolved": t.get("isResolved"),
                    "is_outdated": t.get("isOutdated"),
                    "n_comments": len(comments),
                    "comments": [{"author": (c.get("author") or {}).get("login"),
                                  "body": c.get("body"),
                                  "createdAt": c.get("createdAt")}
                                 for c in comments],
                    "first_push_excerpt": excerpt(first_lines, line),
                    "final_excerpt": excerpt(final_lines, line),
                    "first_push_state": first_state,
                    "file_in_first_push": first_lines is not None,
                    "file_in_final": final_lines is not None,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                done.add(key)
                n_threads += 1

    out.close()
    print(f"DONE: {n_pr} PRs, {n_threads} threads written, "
          f"{n_skipped} skipped (resume), {n_unresolved} unresolved, "
          f"{n_nometa} missing from friction_full", flush=True)


if __name__ == "__main__":
    main()

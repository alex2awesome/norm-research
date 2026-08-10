#!/usr/bin/env python3
"""Wait for a sealed agent's output file to be PARSEABLE, not merely to exist.

Coordinator ruling 2026-08-09, inherited from the jokes_community campaign: an
existence-only wait loop hit a half-written-verdicts race in `audit.py finalize`
-- the file appeared the moment the writing agent opened it, and the reader got
truncated JSON.  Every wait in this campaign therefore polls for a file that
(a) exists, (b) has been byte-stable for one poll interval, and (c) parses as
JSON with the required keys present.

Usage:
    python3 waitfile.py <path> [<path> ...] [--keys k1,k2] [--timeout 3600]
or, in-process:
    from waitfile import wait_parseable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def try_parse(path: Path, keys=()):
    """-> (ok, payload_or_reason)."""
    p = Path(path)
    if not p.exists():
        return False, "missing"
    try:
        raw = p.read_text()
    except OSError as e:
        return False, f"unreadable {e}"
    if not raw.strip():
        return False, "empty"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"unparseable (likely mid-write): {e}"
    for k in keys:
        if k not in obj:
            return False, f"missing key {k!r}"
    return True, obj


def wait_parseable(paths, keys=(), timeout=3600, poll=10, verbose=True):
    """Block until every path parses AND is byte-stable across one poll.
    Returns {path: parsed}.  Raises TimeoutError with the per-file reason."""
    paths = [Path(p) for p in paths]
    sizes, out = {}, {}
    t0 = time.time()
    while True:
        pending = []
        for p in paths:
            if str(p) in out:
                continue
            ok, res = try_parse(p, keys)
            sz = p.stat().st_size if p.exists() else -1
            stable = sizes.get(str(p)) == sz
            sizes[str(p)] = sz
            if ok and stable:
                out[str(p)] = res
            else:
                pending.append((p.name, res if not ok else "awaiting byte-stability"))
        if not pending:
            return out
        if time.time() - t0 > timeout:
            raise TimeoutError(f"still not parseable after {timeout}s: {pending}")
        if verbose:
            print(f"[waitfile] {len(pending)} pending: "
                  + "; ".join(f"{n}: {r}" for n, r in pending[:4]), flush=True)
        time.sleep(poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--keys", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--poll", type=int, default=10)
    a = ap.parse_args()
    keys = tuple(k for k in a.keys.split(",") if k)
    try:
        got = wait_parseable(a.paths, keys, a.timeout, a.poll)
    except TimeoutError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print(f"ALL_PARSEABLE {len(got)}")

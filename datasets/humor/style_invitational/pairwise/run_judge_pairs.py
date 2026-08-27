#!/usr/bin/env python3
"""SI PAIRWISE PROBE -- frontier judge leg (gpt-5.6-sol via `codex exec`).

Claude's subagent budget is exhausted for this session, so the judge family is recorded
rather than chosen: **gpt-5.6-sol**, read-only sandbox in a scratch directory OUTSIDE the
repo, one prompt file in / one JSON object out.  The judge never sees the repository, the
labels, the pair construction, or any other job's output.

Resume-by-output-file: a rerun repeats no finished work.  HOLISTIC jobs run FIRST because
they alone decide whether the probe separates; the criterion jobs are only interpretable
once that is known.

Bounded per-call timeout (the cap_crowd r4 wedge: one call sat 44 min past its own
timeout and blocked a serial leg).  A timed-out job is recorded and skipped, never
retried forever.

Usage:
  python run_judge_pairs.py --workers 4 [--only holistic] [--timeout 900]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
# --root lets Phase 1 reuse this leg without clobbering the probe artifacts
import sys as _sys
_ROOT = HERE
for _i, _a in enumerate(_sys.argv):
    if _a == "--root":
        _ROOT = HERE / _sys.argv[_i + 1]
PROMPTS = _ROOT / "prompts"
OUT = _ROOT / "out"
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/si_pairwise")
MODEL = "gpt-5.6-sol"
JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract(raw):
    body = raw.rsplit("tokens used", 1)[0] if "tokens used" in raw else raw
    for s in sorted(JSON_RE.findall(body), key=len, reverse=True):
        try:
            d = json.loads(s)
            if "answers" in d:
                return d
        except json.JSONDecodeError:
            continue
    raise ValueError("no parseable {'answers': ...} object")


def one(job, timeout, effort, model=MODEL):
    tag = job["tag"]
    out = OUT / f"{tag}.json"
    if out.exists():
        try:
            json.loads(out.read_text())
            return tag, "skip", 0
        except Exception:
            pass
    wd = SCRATCH / f"wd_{tag}"
    wd.mkdir(parents=True, exist_ok=True)
    prompt = (PROMPTS / f"{tag}.txt").read_text()
    cmd = ["codex", "exec", "--model", model, "-c", f"model_reasoning_effort={effort}",
           "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        (OUT / f"{tag}.TIMEOUT.txt").write_text(f"timeout {timeout}s")
        return tag, "TIMEOUT", time.time() - t0
    (OUT / f"{tag}.raw.txt").write_text(p.stdout)
    try:
        d = extract(p.stdout)
    except Exception as e:  # noqa: BLE001
        (OUT / f"{tag}.PARSEFAIL.txt").write_text(f"{type(e).__name__}: {e}")
        return tag, f"PARSEFAIL {type(e).__name__}", time.time() - t0
    d["_meta"] = {"model": model, "effort": effort, "seconds": round(time.time() - t0),
                  "question": job.get("question"), "n_pairs_asked": job["n_pairs"],
                  "n_answers": len(d.get("answers", []))}
    out.write_text(json.dumps(d, indent=1))
    return tag, f"ok {len(d.get('answers', []))}/{job['n_pairs']}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--only", default=None, help="holistic | criteria")
    ap.add_argument("--root", default=None)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    man = json.loads((_ROOT / "si_prompt_manifest.json").read_text())
    jobs = man["jobs"]
    if a.only == "holistic":
        jobs = [j for j in jobs if j["question"] == "holistic"]
    elif a.only == "criteria":
        jobs = [j for j in jobs if j["question"] != "holistic"]
    jobs = sorted(jobs, key=lambda j: (j["question"] != "holistic", j["tag"]))

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for tag, status, secs in ex.map(lambda j: one(j, a.timeout, a.effort, a.model), jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {tag} {status} {secs:.0f}s", flush=True)
    print("JUDGE_LEG_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

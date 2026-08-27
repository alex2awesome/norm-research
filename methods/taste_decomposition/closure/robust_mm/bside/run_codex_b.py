#!/usr/bin/env python3
"""OpenAI leg of the B-side sealed fleet -- gpt-5.6-luna via `codex exec`.

Identical mechanics to ../run_codex.py (sandbox read-only, scratch working dir
outside the repo, resume-by-output-file); repointed at the bside scratch tree.

Usage: python run_codex_b.py --tags bside_rep1,bside_rep2,bside_rep3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm/bside")


def run_one(tag, pid, model, effort="high", timeout=2400):
    d = SCRATCH / tag
    out = d / f"out_{pid}.txt"
    if out.exists() and len(out.read_text()) > 500:
        print(f"[{tag}/{pid}] already done, skip", flush=True)
        return
    prompt = (d / f"prompt_{pid}.txt").read_text()
    wd = d / f"wd_{pid}"
    wd.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--model", model, "-c", f"model_reasoning_effort={effort}",
           "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    raw = p.stdout
    (d / f"raw_{pid}.txt").write_text(raw)
    body = raw
    if "tokens used" in raw:
        body = raw.rsplit("tokens used", 1)[1]
        body = body.split("\n", 2)[-1] if body.strip().split("\n")[0].strip().replace(",", "").isdigit() else body
    out.write_text(body.strip())
    (d / f"meta_{pid}.json").write_text(json.dumps(
        {"model": model, "effort": effort, "seconds": round(dt),
         "returncode": p.returncode, "stderr_tail": p.stderr[-500:]}, indent=1))
    print(f"[{tag}/{pid}] {model} rc={p.returncode} {dt:.0f}s -> {len(body)} chars", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--ids", default="codex_luna_a,codex_luna_b")
    ap.add_argument("--model", default="gpt-5.6-luna")
    a = ap.parse_args()
    for tag in a.tags.split(","):
        for pid in a.ids.split(","):
            run_one(tag.strip(), pid.strip(), a.model)
    print("codex leg complete", flush=True)


if __name__ == "__main__":
    main()

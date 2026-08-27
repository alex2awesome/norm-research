#!/usr/bin/env python3
"""OpenAI leg of the sealed fleet -- gpt-5.6-luna via the Codex CLI (`codex exec`).

Codex is used purely as a PROPOSER here (it cannot run inference-scoring, and M3
needs none: rediscovered concepts are measured with their existing score columns).

Seal: the run happens in a scratch working directory OUTSIDE the repo, with the
sandbox pinned read-only, so the proposer cannot reach the criterion bank, the
labels, or the other proposers' outputs.  The prompt is piped on stdin.

Escalation policy (design Sec 8, user authorization 2026-08-06): luna is the default;
if luna output is degenerate (< k/2 parseable criteria, or duplicate-name collapse)
the same slot is re-run on gpt-5.6-sol and the escalation is recorded.

Usage: python run_codex.py --tags rep1,rep2 --ids codex_luna_a,codex_luna_b
       python run_codex.py --tags rep1 --ids codex_luna_a --model gpt-5.6-sol --suffix _sol
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm")


def run_one(tag, pid, model, suffix, effort="high", timeout=2400):
    d = SCRATCH / tag
    out = d / f"out_{pid}{suffix}.txt"
    if out.exists() and len(out.read_text()) > 500:
        print(f"[{tag}/{pid}{suffix}] already done, skip", flush=True)
        return
    prompt = (d / f"prompt_{pid}.txt").read_text()
    wd = d / f"wd_{pid}{suffix}"
    wd.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--model", model, "-c", f"model_reasoning_effort={effort}",
           "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    raw = p.stdout
    (d / f"raw_{pid}{suffix}.txt").write_text(raw)
    # `codex exec` echoes a banner, the prompt, then the assistant turn, then repeats
    # the final message after a "tokens used" line.  Take the tail after that marker
    # when present; otherwise take everything after the last "codex" turn marker.
    body = raw
    if "tokens used" in raw:
        body = raw.rsplit("tokens used", 1)[1]
        body = body.split("\n", 2)[-1] if body.strip().split("\n")[0].strip().replace(",", "").isdigit() else body
    out.write_text(body.strip())
    (d / f"meta_{pid}{suffix}.json").write_text(json.dumps(
        {"model": model, "effort": effort, "seconds": round(dt),
         "returncode": p.returncode, "stderr_tail": p.stderr[-500:]}, indent=1))
    print(f"[{tag}/{pid}{suffix}] {model} rc={p.returncode} {dt:.0f}s -> {len(body)} chars", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--ids", default="codex_luna_a,codex_luna_b")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    for tag in a.tags.split(","):
        for pid in a.ids.split(","):
            run_one(tag.strip(), pid.strip(), a.model, a.suffix)
    print("codex leg complete", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sealed decomposer leg for the code_v3 Addendum-3 pass.

RECORDED INSTRUMENT CHANGE: the freeze names a "sealed frontier decomposer" and the
worked campaigns used a Claude Opus subagent. This session hit the hard subagent cap
(500/500), so the Claude family is unavailable. The decomposer is therefore
gpt-5.6-luna via `codex exec` at effort high -- frontier class, different family.
Recorded, not silent.

SEAL: codex runs read-only in a scratch working directory OUTSIDE the repo, so the
decomposer cannot reach the bank, the labels, or any other artifact. Its entire input
is prompt_decomposer.txt.
"""
import json, subprocess, sys, time
from pathlib import Path

D = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
         "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/code_v3/code_v3_rd")
prompt = (D / "prompt_decomposer.txt").read_text()
wd = D / "wd_decomposer"; wd.mkdir(parents=True, exist_ok=True)
out = D / "out_decomposer.txt"

cmd = ["codex", "exec", "--model", "gpt-5.6-luna", "-c", "model_reasoning_effort=high",
       "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
t0 = time.time()
p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=3000)
raw = p.stdout
(D / "raw_decomposer.txt").write_text(raw)
body = raw
if "tokens used" in raw:
    body = raw.rsplit("tokens used", 1)[1]
    first = body.strip().split("\n")[0].strip().replace(",", "")
    if first.isdigit():
        body = body.split("\n", 2)[-1]
out.write_text(body.strip())
(D / "meta_decomposer.json").write_text(json.dumps(
    {"model": "gpt-5.6-luna", "effort": "high", "seconds": round(time.time() - t0),
     "returncode": p.returncode, "stderr_tail": p.stderr[-500:],
     "recorded_instrument_change": "Claude subagent cap 500/500 -> frontier decomposer is "
                                   "gpt-5.6-luna via codex exec, not Claude Opus"}, indent=1))
print(f"rc={p.returncode} {time.time()-t0:.0f}s -> {len(body)} chars")

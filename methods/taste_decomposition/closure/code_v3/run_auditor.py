#!/usr/bin/env python3
"""Fresh blind routing auditor for a code_v3 round.

RECORDED INSTRUMENT CHANGE: the freeze asks for a "fresh blind Sonnet-class auditor per
round"; the worked campaigns used a sealed Claude subagent. This session hit the hard
subagent cap (500/500), so the auditor is gpt-5.6-luna via `codex exec` at effort high --
strictly stronger than Sonnet, different family, fresh sealed context per round, run
read-only in a scratch working directory outside the repo. Recorded, not silent.
"""
import json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/code_v3")
rnd = sys.argv[1] if len(sys.argv) > 1 else "d"
tag = f"code_v3_r{rnd}"
prompt = (HERE / f"{tag}_audit_prompt.txt").read_text()
wd = SCRATCH / f"wd_audit_{rnd}"; wd.mkdir(parents=True, exist_ok=True)

cmd = ["codex", "exec", "--model", "gpt-5.6-luna", "-c", "model_reasoning_effort=high",
       "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
t0 = time.time()
p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=3000)
raw = p.stdout
(SCRATCH / f"raw_audit_{rnd}.txt").write_text(raw)
body = raw
if "tokens used" in raw:
    body = raw.rsplit("tokens used", 1)[1]
    first = body.strip().split("\n")[0].strip().replace(",", "")
    if first.isdigit():
        body = body.split("\n", 2)[-1]
m = re.search(r"\{[\s\S]*\}", body)
assert m, f"PARSEABILITY GATE: no JSON in auditor output\n{body[:800]}"
obj = json.loads(m.group(0))
n_items = len(json.loads((HERE / f"{tag}_audit_key.json").read_text()))
assert len(obj["verdicts"]) == n_items, (
    f"PARSEABILITY GATE: {len(obj['verdicts'])} verdicts for {n_items} items")
obj["auditor"] = "gpt-5.6-luna (codex exec, effort high, fresh sealed context)"
(HERE / f"{tag}_audit_verdicts.json").write_text(json.dumps(obj, indent=1))
print(f"rc={p.returncode} {time.time()-t0:.0f}s  verdicts={len(obj['verdicts'])}/{n_items}")

#!/usr/bin/env python3
"""Run a sealed blind JUDGE leg (species-merge judge, or blind routing auditor) through a
non-Claude frontier model.

WHY THIS EXISTS, recorded rather than silent.  Rounds 1-4 of this campaign ran their
merge judges and their blind auditor as sealed Claude subagents (Sonnet + Opus).  The
session's subagent budget was exhausted after round 4, so round 5's judge legs run
through the same two non-Claude endpoints the proposer fleet already uses --
gpt-5.6-luna via `codex exec` (read-only sandbox outside the repo) and glm-5.2 via the
z.ai endpoint.  Both are frontier-class, both are independent of each other and of the
proposer that wrote any given item, and the planted anchor pairs / planted probe pairs
verify each of them in the same batch exactly as before.  The family change is recorded
in every artifact this writes and in the campaign note; it is a DEVIATION from the
round-1..4 judge roster, not a silent substitution.

Usage:
  python run_judge.py --prompt <file> --out <file> --engine codex|glm [--extract-json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import run_fleet as RF

HERE = Path(__file__).resolve().parent
SCRATCH = RF.SCRATCH


def codex_call(prompt, wd, effort="high", timeout=3600):
    wd.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--model", "gpt-5.6-luna", "-c",
           f"model_reasoning_effort={effort}", "-s", "read-only",
           "--skip-git-repo-check", "--cd", str(wd), "-"]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    return p.stdout, {"model": "gpt-5.6-luna", "effort": effort,
                      "returncode": p.returncode, "stderr_tail": p.stderr[-400:]}


def glm_call(prompt, key_id="glm_a"):
    order = [key_id] + [q for q in RF.KEYFILE if q != key_id]
    keys = [(q, RF.KEYFILE[q].read_text().strip()) for q in order if RF.KEYFILE[q].exists()]
    last = None
    for ki, (kname, key) in enumerate(keys):
        for attempt in range(1, 6):
            try:
                d = RF.glm_call(key, prompt, budget=2048, max_tokens=32000)
                blocks = d.get("content", [])
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                return text, {"model": d.get("model"), "key_used": kname,
                              "attempts": attempt, "usage": d.get("usage"),
                              "stop_reason": d.get("stop_reason")}
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                print(f"[glm {kname}] attempt {attempt} {last}", flush=True)
                time.sleep(30 * attempt)
    raise RuntimeError(f"glm judge failed: {last}")


JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(raw):
    body = raw
    if "tokens used" in body:
        body = body.rsplit("tokens used", 1)[0]
    cands = JSON_RE.findall(body)
    for s in sorted(cands, key=len, reverse=True):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    raise ValueError("no parseable JSON object in judge output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["codex", "glm"], required=True)
    ap.add_argument("--key-id", default="glm_a")
    ap.add_argument("--wd", default=None)
    a = ap.parse_args()

    prompt = Path(a.prompt).read_text()
    if a.engine == "codex":
        wd = Path(a.wd) if a.wd else (SCRATCH / ("judge_wd_" + Path(a.out).stem))
        raw, meta = codex_call(prompt, wd)
    else:
        raw, meta = glm_call(prompt, a.key_id)
    Path(a.out + ".raw.txt").write_text(raw)
    d = extract_json(raw)
    d.setdefault("judge", meta.get("model"))
    d["_judge_meta"] = meta
    d["_judge_roster_deviation"] = (
        "Round-5 judge legs run through non-Claude frontier endpoints because this "
        "session's Claude subagent budget was exhausted after round 4. Blindness, the "
        "planted anchors/probes and the strict merge rule are unchanged.")
    Path(a.out).write_text(json.dumps(d, indent=1))
    n = len(d.get("verdicts", d.get("rulings", [])))
    print(f"{a.engine} judge -> {a.out}: {n} items, meta={json.dumps(meta)[:200]}")


if __name__ == "__main__":
    main()

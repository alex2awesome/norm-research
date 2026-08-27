#!/usr/bin/env python3
"""Non-Claude legs of the sealed dual-track fleet: gpt-5.6-luna via `codex exec`
and glm-5.2 via the z.ai Anthropic-compatible endpoint.

Both are lifted from robust_mm/run_codex.py and robust_mm/run_glm.py with one
change: prompt/output files are named `{track}_{proposer}` because every round now
has two sealed tracks.  Resume-by-output-file, so a rerun repeats no finished work.

Seal: codex runs read-only in a scratch working directory OUTSIDE the repo, so the
proposer cannot reach the bank, the labels, or another proposer's output.

Usage:
  python run_fleet.py codex --tags cap_crowd_r1,cap_finalist_r1 --tracks A,B
  python run_fleet.py glm   --tags cap_crowd_r1 --tracks B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/maps_batch1")
URL = "https://api.z.ai/api/anthropic/v1/messages"
KEYFILE = {"glm_a": Path.home() / ".z-ai-api-key.txt",
           "glm_b": Path.home() / ".z-ai-api-key-spangher.txt"}
GAP = 25
MAX_TRIES = 24


# --------------------------------------------------------------- codex leg --
def codex_one(tag, track, pid, model="gpt-5.6-luna", effort="high", timeout=2400):
    d = SCRATCH / tag
    out = d / f"out_{track}_{pid}.txt"
    if out.exists() and len(out.read_text()) > 500:
        print(f"[{tag}/{track}/{pid}] already done, skip", flush=True)
        return
    prompt = (d / f"prompt_{track}_{pid}.txt").read_text()
    wd = d / f"wd_{track}_{pid}"
    wd.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--model", model, "-c", f"model_reasoning_effort={effort}",
           "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{tag}/{track}/{pid}] TIMEOUT after {timeout}s", flush=True)
        return
    raw = p.stdout
    (d / f"raw_{track}_{pid}.txt").write_text(raw)
    body = raw
    if "tokens used" in raw:
        body = raw.rsplit("tokens used", 1)[1]
        first = body.strip().split("\n")[0].strip().replace(",", "")
        if first.isdigit():
            body = body.split("\n", 2)[-1]
    out.write_text(body.strip())
    (d / f"meta_{track}_{pid}.json").write_text(json.dumps(
        {"model": model, "effort": effort, "seconds": round(time.time() - t0),
         "returncode": p.returncode, "stderr_tail": p.stderr[-400:]}, indent=1))
    print(f"[{tag}/{track}/{pid}] {model} rc={p.returncode} "
          f"{time.time()-t0:.0f}s -> {len(body)} chars", flush=True)


# ----------------------------------------------------------------- glm leg --
def glm_call(key, prompt, budget=2048, max_tokens=32000, model="glm-5.2"):
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def glm_one(tag, track, pid):
    d = SCRATCH / tag
    out = d / f"out_{track}_{pid}.txt"
    if out.exists() and len(out.read_text()) > 500:
        print(f"[{tag}/{track}/{pid}] already done, skip", flush=True)
        return
    prompt = (d / f"prompt_{track}_{pid}.txt").read_text()
    order = [pid] + [q for q in KEYFILE if q != pid]
    keys = [(q, KEYFILE[q].read_text().strip()) for q in order if KEYFILE[q].exists()]
    if not keys:
        print(f"[{tag}/{track}/{pid}] no GLM key on disk, skip", flush=True)
        return
    label = f"{tag}/{track}/{pid}"
    ki, wait = 0, 60
    for attempt in range(1, MAX_TRIES + 1):
        if attempt > 5 and ki + 1 < len(keys):
            ki += 1
            print(f"[{label}] rotating key -> {keys[ki][0]}", flush=True)
        kname, key = keys[ki]
        try:
            t0 = time.time()
            dd = glm_call(key, prompt)
            blocks = dd.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            think = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
            print(f"[{label}] OK attempt {attempt} {time.time()-t0:.0f}s "
                  f"usage={dd.get('usage')} think={len(think)} text={len(text)} "
                  f"stop={dd.get('stop_reason')}", flush=True)
            if dd.get("stop_reason") == "max_tokens" and text.count('"name"') < 8:
                time.sleep(GAP)
                dd = glm_call(key, prompt, budget=None)
                blocks = dd.get("content", [])
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                print(f"[{label}] retry-no-think text={len(text)}", flush=True)
            out.write_text(text)
            (d / f"meta_{track}_{pid}.json").write_text(json.dumps(
                {"model": dd.get("model"), "attempts": attempt, "key_used": kname,
                 "usage": dd.get("usage"), "thinking_chars": len(think),
                 "stop_reason": dd.get("stop_reason")}, indent=1))
            time.sleep(GAP)
            return
        except urllib.error.HTTPError as e:
            raw = e.read()[:200].decode(errors="replace")
            print(f"[{label}] attempt {attempt} HTTP {e.code} {raw} -> sleep {wait}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] attempt {attempt} {type(e).__name__} {e} -> sleep {wait}s", flush=True)
        time.sleep(wait)
        wait = min(int(wait * 1.4), 600)
    print(f"[{label}] GAVE UP after {MAX_TRIES} attempts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["codex", "glm"])
    ap.add_argument("--tags", required=True)
    ap.add_argument("--tracks", default="A,B")
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    ids = (a.ids.split(",") if a.ids else
           (["codex_luna_a", "codex_luna_b"] if a.leg == "codex" else ["glm_a", "glm_b"]))
    for tag in a.tags.split(","):
        for track in a.tracks.split(","):
            for pid in ids:
                (codex_one if a.leg == "codex" else glm_one)(tag.strip(), track.strip(), pid.strip())
    print(f"{a.leg} leg complete", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""BBC most-read closure — non-Claude legs of the sealed fleet.

codex gpt-5.6-luna via `codex exec` (read-only, scratch cwd outside the repo) and
glm-5.2 via the z.ai Anthropic-compatible endpoint. Resume-by-output-file, so a rerun
repeats no finished work and a mid-wave failure loses nothing.

MID-WAVE FAILURE POLICY (coordinator, 2026-08-12): if GLM 1302s again mid-wave,
checkpoint whatever completed and STOP that family — never re-issue a sealed prompt to
a different family mid-seal, because the seal is per (track, proposer) and swapping the
model behind a slot would silently mix families inside one proposal set.

  python3 run_fleet_bbc.py codex --round 1
  python3 run_fleet_bbc.py glm   --round 1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

SCRATCH = pathlib.Path(
    "/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
    "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/unified_closure")
URL = "https://api.z.ai/api/anthropic/v1/messages"
KEYS = {"a": "~/.z-ai-api-key.txt", "b": "~/.z-ai-api-key-spangher.txt",
        "c": "~/.z-ai-api-key.txt", "d": "~/.z-ai-api-key-spangher.txt"}
GAP = 8


def done(p):
    return p.exists() and len(p.read_text().strip()) > 300


def run_codex(d, name, track, timeout=1800):
    out = d / f"out_{track}_{name}.txt"
    if done(out):
        print(f"[{track}/{name}] done, skip", flush=True)
        return True
    wd = d / f"wd_{track}_{name}"
    wd.mkdir(parents=True, exist_ok=True)
    prompt = (d / f"prompt_{track}_{name}.txt").read_text()
    cmd = ["codex", "exec", "--model", "gpt-5.6-luna",
           "-c", "model_reasoning_effort=high", "-s", "read-only",
           "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{track}/{name}] TIMEOUT", flush=True)
        return False
    txt = p.stdout or ""
    if len(txt.strip()) < 300:
        print(f"[{track}/{name}] short output ({len(txt)}) rc={p.returncode}", flush=True)
        (d / f"err_{track}_{name}.txt").write_text((p.stderr or "")[:4000])
        return False
    out.write_text(txt)
    print(f"[{track}/{name}] ok {time.time()-t0:.0f}s {len(txt)}ch", flush=True)
    return True


def run_glm(d, name, track, pid, max_tokens=4000, tries=4):
    out = d / f"out_{track}_{name}.txt"
    if done(out):
        print(f"[{track}/{name}] done, skip", flush=True)
        return True
    key = pathlib.Path(KEYS[pid]).expanduser().read_text().strip()
    prompt = (d / f"prompt_{track}_{name}.txt").read_text()
    body = {"model": "glm-5.2", "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(tries):
        req = urllib.request.Request(
            URL, data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            txt = "".join(c.get("text", "") for c in r.get("content", []))
            if len(txt.strip()) < 300:
                print(f"[{track}/{name}] short ({len(txt)})", flush=True)
                return False
            out.write_text(txt)
            print(f"[{track}/{name}] ok {r.get('usage', {}).get('output_tokens')}tok",
                  flush=True)
            return True
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:160]
            if "1302" in msg or e.code == 429:
                print(f"[{track}/{name}] 1302/429 attempt {attempt}", flush=True)
                if attempt == tries - 1:
                    print("GLM_RATE_LIMIT_STOP", flush=True)
                    return "RATELIMIT"
                time.sleep(45 * (attempt + 1))
                continue
            print(f"[{track}/{name}] HTTP {e.code} {msg}", flush=True)
            return False
        except Exception as e:
            print(f"[{track}/{name}] {type(e).__name__} {str(e)[:100]}", flush=True)
            time.sleep(20)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["codex", "glm"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    d = SCRATCH / f"{a.cell}_r{a.round}"
    man = json.loads((d / "fleet_manifest.json").read_text())
    fam = "codex_luna" if a.leg == "codex" else "glm"
    todo = [p for p in man["proposers"] if p["family"] == fam]
    ok = 0
    for p in todo:
        for track in ("A", "B"):
            if a.leg == "codex":
                r = run_codex(d, p["name"], track)
            else:
                r = run_glm(d, p["name"], track, p["name"].split("_")[-1])
                if r == "RATELIMIT":
                    print(f"CHECKPOINT: stopping GLM leg, {ok} slots complete",
                          flush=True)
                    return
                time.sleep(GAP)
            ok += bool(r)
    print(f"{a.leg.upper()}_LEG_DONE {ok}/{2*len(todo)} slots", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GLM-5.2 supplemental leg of the B-side sealed fleet (bonus 3rd family).

Identical mechanics to ../run_glm.py; repointed at the bside scratch tree. Not
required to hit the P>=4/>=2-family floor (Claude + Codex already clears it) --
attempted opportunistically per the standing GLM-live convention, non-blocking.

Usage: python run_glm_b.py --tags bside_rep1,bside_rep2,bside_rep3 --ids glm_a
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm/bside")
URL = "https://api.z.ai/api/anthropic/v1/messages"
KEYFILE = {"glm_a": Path.home() / ".z-ai-api-key.txt",
           "glm_b": Path.home() / ".z-ai-api-key-spangher.txt"}
GAP = 25
MAX_TRIES = 12  # bonus leg: don't burn the whole session on backoff


def call(key, prompt, budget=2048, max_tokens=32000, model="glm-5.2"):
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


def patient(keys, prompt, label):
    ROTATE_AFTER = 3
    ki = 0
    wait = 45
    for attempt in range(1, MAX_TRIES + 1):
        if attempt > ROTATE_AFTER and ki + 1 < len(keys):
            ki += 1
            print(f"[{label}] rotating key -> {keys[ki][0]} after {ROTATE_AFTER} failures", flush=True)
        kname, key = keys[ki]
        try:
            t0 = time.time()
            d = call(key, prompt)
            blocks = d.get("content", [])
            think = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            print(f"[{label}] OK attempt {attempt} {time.time()-t0:.0f}s "
                  f"usage={d.get('usage')} think_chars={len(think)} text_chars={len(text)} "
                  f"stop={d.get('stop_reason')}", flush=True)
            if d.get("stop_reason") == "max_tokens" and '"channels"' in text and text.count('"name"') < 8:
                print(f"[{label}] truncated -> retrying with thinking disabled", flush=True)
                time.sleep(GAP)
                d = call(key, prompt, budget=None, max_tokens=32000)
                blocks = d.get("content", [])
                think = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                print(f"[{label}] retry usage={d.get('usage')} text_chars={len(text)} "
                      f"stop={d.get('stop_reason')}", flush=True)
            return text, {"attempts": attempt, "key_used": kname, "usage": d.get("usage"),
                          "thinking_chars": len(think), "stop_reason": d.get("stop_reason"),
                          "model": d.get("model")}, think
        except urllib.error.HTTPError as e:
            raw = e.read()[:300].decode(errors="replace")
            print(f"[{label}] attempt {attempt} HTTP {e.code} {raw} -> sleep {wait}s", flush=True)
            time.sleep(wait)
            wait = min(int(wait * 1.5), 300)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] attempt {attempt} {type(e).__name__} {e} -> sleep {wait}s", flush=True)
            time.sleep(wait)
            wait = min(int(wait * 1.5), 300)
    print(f"[{label}] GAVE UP after {MAX_TRIES} attempts (bonus leg, non-fatal)", flush=True)
    return None, {"attempts": MAX_TRIES, "gave_up": True}, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--ids", default="glm_a")
    a = ap.parse_args()
    jobs = [(t.strip(), p.strip()) for t in a.tags.split(",") for p in a.ids.split(",")]
    for tag, pid in jobs:
        d = SCRATCH / tag
        out = d / f"out_{pid}.txt"
        if out.exists() and len(out.read_text()) > 500:
            print(f"[{tag}/{pid}] already done, skip", flush=True)
            continue
        pf = d / f"prompt_{pid}.txt"
        if not pf.exists():
            print(f"[{tag}/{pid}] no prompt file ({pf}), skip -- run harness_b.py build --with-glm first", flush=True)
            continue
        prompt = pf.read_text()
        order = [pid] + [q for q in KEYFILE if q != pid]
        keys = [(q, KEYFILE[q].read_text().strip()) for q in order if KEYFILE[q].exists()]
        text, meta, think = patient(keys, prompt, f"{tag}/{pid}")
        if text is not None:
            out.write_text(text)
            (d / f"trace_{pid}.txt").write_text(think)
        (d / f"meta_{pid}.json").write_text(json.dumps(meta, indent=1))
        time.sleep(GAP)
    print("GLM bonus leg complete", flush=True)


if __name__ == "__main__":
    main()

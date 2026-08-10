#!/usr/bin/env python3
"""GLM-5.2 proposer leg of the sealed fleet (z.ai subscription, /api/anthropic).

Endpoint discipline (reference_glm_subscription_api): the LITE subscription covers
`https://api.z.ai/api/anthropic/v1/messages` ONLY -- `/api/paas/v4` and
`/api/coding/paas/v4` return 1302/1113 on these keys.  Thinking mode is enabled
explicitly with a budget (smoke-tested first: trivial question -> 401 output tokens,
1,741 chars of trace, so the trace is real and the budget dial works).

The Lite plan enforces a tight REQUEST-rate limit (1302) that trips within seconds of
a successful call, so this runner is deliberately patient: long sleeps between calls,
exponential backoff with a floor of 60s on 1302, resume-by-output-file so a rerun
never repeats finished work.

Usage: python run_glm.py --tag rep1 --ids glm_a,glm_b
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm")
URL = "https://api.z.ai/api/anthropic/v1/messages"
KEYFILE = {"glm_a": Path.home() / ".z-ai-api-key.txt",
           "glm_b": Path.home() / ".z-ai-api-key-spangher.txt"}
GAP = 25          # polite gap between successful calls
MAX_TRIES = 40


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
    """keys = ordered list of (keyname, keystring); rotate on persistent 1302.

    The two GLM slots are meant to sit on the two subscription accounts, but the
    -spangher key has been returning 1302 (request-rate) on every probe today.  After
    ROTATE_AFTER failed attempts the slot falls back to the next key.  That keeps the
    slot a genuinely independent DRAW (different slice ordering, fresh context) even
    when it is no longer an independent ACCOUNT -- and the substitution is recorded.
    """
    ROTATE_AFTER = 5
    ki = 0
    wait = 60
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
            if d.get("stop_reason") == "max_tokens" and '"criteria"' in text and text.count('"name"') < 12:
                # truncated mid-array: the thinking trace ate the budget.  Retry once
                # with the trace switched off rather than editing the sealed prompt.
                print(f"[{label}] truncated ({text.count(chr(34)+'name'+chr(34))} names) -> "
                      f"retrying with thinking disabled", flush=True)
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
            wait = min(int(wait * 1.5), 900)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] attempt {attempt} {type(e).__name__} {e} -> sleep {wait}s", flush=True)
            time.sleep(wait)
            wait = min(int(wait * 1.5), 900)
    raise RuntimeError(f"{label}: exhausted {MAX_TRIES} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated tags, e.g. rep1,rep2")
    ap.add_argument("--ids", default="glm_a,glm_b")
    a = ap.parse_args()

    jobs = []
    for tag in a.tags.split(","):
        for pid in a.ids.split(","):
            jobs.append((tag.strip(), pid.strip()))

    for tag, pid in jobs:
        d = SCRATCH / tag
        out = d / f"out_{pid}.txt"
        if out.exists() and len(out.read_text()) > 500:
            print(f"[{tag}/{pid}] already done, skip", flush=True)
            continue
        prompt = (d / f"prompt_{pid}.txt").read_text()
        order = [pid] + [q for q in KEYFILE if q != pid]
        keys = [(q, KEYFILE[q].read_text().strip()) for q in order if KEYFILE[q].exists()]
        text, meta, think = patient(keys, prompt, f"{tag}/{pid}")
        out.write_text(text)
        (d / f"meta_{pid}.json").write_text(json.dumps(meta, indent=1))
        (d / f"trace_{pid}.txt").write_text(think)
        time.sleep(GAP)
    print("GLM leg complete", flush=True)


if __name__ == "__main__":
    main()

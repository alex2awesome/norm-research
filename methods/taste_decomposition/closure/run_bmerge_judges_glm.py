#!/usr/bin/env python3
"""GLM-5.2 leg for the b_merge blind concept-identity pass.

WHY THIS EXISTS (2026-08-12): the codex account hit its usage limit mid-wave
(resets Aug 18), stranding three of four planned legs. GLM-5.2 (z.ai Lite,
Anthropic-format endpoint) substitutes as the second judge. This UPGRADES the
independence claim relative to the sol+luna plan: sol+glm is a genuine
CROSS-FAMILY two-judge merge, not two legs of one family — recorded in the
output as judge_family "zai/glm-5.2".

Instrument identity: PREAMBLE, pair rendering and JSON extraction are imported
from run_bmerge_judges.py so the prompt and parser are byte-identical to the
codex legs. Only the transport differs.

Usage:
  python run_bmerge_judges_glm.py --packet <..>_bmerge<T>_packet.json \
      --out <..>_bmerge<T>_judge_glm.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request

from run_bmerge_judges import PREAMBLE, render, extract_json

KEYS = [pathlib.Path(p).expanduser()
        for p in ("~/.z-ai-api-key.txt", "~/.z-ai-api-key-spangher.txt")]
URL = "https://api.z.ai/api/anthropic/v1/messages"


def ask_glm(prompt, max_tokens=16000, timeout=1800, retries=5):
    last = None
    for attempt in range(retries):
        for kp in KEYS:
            if not kp.exists():
                continue
            key = kp.read_text().strip()
            body = {"model": "glm-5.2", "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            req = urllib.request.Request(
                URL, data=json.dumps(body).encode(),
                headers={"content-type": "application/json", "x-api-key": key,
                         "anthropic-version": "2023-06-01"})
            t0 = time.time()
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
                txt = "".join(b.get("text", "") for b in r.get("content", [])
                              if b.get("type") == "text")
                print(f"    [glm-5.2/{kp.name}] {time.time()-t0:.0f}s "
                      f"out={len(txt)}B stop={r.get('stop_reason')}", flush=True)
                return txt
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"    [glm-5.2/{kp.name}] FAIL {time.time()-t0:.0f}s "
                      f"{str(e)[:120]}", flush=True)
        time.sleep(60 * (attempt + 1))
    raise SystemExit(f"GLM exhausted after {retries} rounds over both keys: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    if out.exists():
        print(f"{out.name} already exists, skip")
        return

    pk = json.loads(pathlib.Path(a.packet).read_text())
    items = list(pk["items"]) + list(pk["anchors"])
    want = [it["pair_id"] for it in items]
    by_id = {it["pair_id"]: it for it in items}
    verdicts, passes = {}, []

    todo = list(want)
    for attempt in range(4):
        if not todo:
            break
        batches = ([todo] if a.chunk <= 0
                   else [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)])
        print(f"  pass {attempt + 1}: {len(todo)} pairs in {len(batches)} prompt(s)",
              flush=True)
        for b in batches:
            raw = ask_glm(PREAMBLE + render([by_id[i] for i in b]))
            obj = extract_json(raw)
            if not obj:
                print("    no JSON parsed from this prompt", flush=True)
                continue
            for v in obj.get("verdicts", []):
                pid = v.get("pair_id")
                vd = str(v.get("verdict", "")).strip().upper()
                if pid in by_id and vd in ("SAME", "DIFFERENT"):
                    verdicts[pid] = {"pair_id": pid, "verdict": vd}
        passes.append({"pass": attempt + 1, "asked": len(todo), "have": len(verdicts)})
        todo = [p for p in want if p not in verdicts]

    if todo:
        raise SystemExit(f"COVERAGE FAIL: {len(todo)} unjudged: {todo[:10]}")

    res = {"judge": "glm-5.2", "judge_family": "zai/glm-5.2",
           "transport": "api.z.ai anthropic-format",
           "cross_family_note": ("substituted for the planned codex second leg "
                                 "(codex usage limit, resets 2026-08-18); sol+glm "
                                 "is a CROSS-FAMILY merge — stronger independence "
                                 "than the sol+luna hive-mind pairing"),
           "passes": passes,
           "verdicts": sorted(verdicts.values(), key=lambda v: v["pair_id"])}
    out.write_text(json.dumps(res, indent=1))
    n_same = sum(1 for v in res["verdicts"] if v["verdict"] == "SAME")
    print(f"wrote {out.name}: {len(res['verdicts'])} verdicts, {n_same} SAME")


if __name__ == "__main__":
    main()

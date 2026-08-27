#!/usr/bin/env python3
"""SI PAIRWISE — PHASE 1 cross-family replication slice (approval 2026-08-11).

The probe's .810 came from ONE frontier family (gpt-5.6-sol). The single-family caveat is
bounded here rather than merely carried: **30 of the same 200 MATCHED winner-vs-HM pairs**
are re-judged by **glm-5.2**, a different family and a different vendor, on the identical
prompt text and the identical A/B order assignment. Because the pairs and the presentation
are byte-identical, the two numbers are directly comparable and the per-pair agreement
rate is meaningful.

Pairs are drawn by stable sha256 of the pair id (unseeded, reproducible). ANCHOR_FRAGMENT
pairs are included so the replication carries its own anchor check.

Usage: python xfamily.py [--n 30] [--engine glm|luna]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SALT = "si-xfamily-2026-08-11|"
URL = "https://api.z.ai/api/anthropic/v1/messages"
KEYS = [Path.home() / ".z-ai-api-key.txt", Path.home() / ".z-ai-api-key-spangher.txt"]
JSON_RE = re.compile(r"\{[\s\S]*\}")

HOLISTIC = (
    "For each pair below, two entries were submitted to the SAME humour contest, in "
    "answer to the SAME prompt. Both were good enough to be published. One of them was "
    "chosen by the contest's editor as the week's WINNER; the other was published only as "
    "an honourable mention.\n\nFor each pair, say which entry the editor picked."
)
TAIL = (
    "\n\nOUTPUT. Emit exactly one JSON object and nothing else:\n"
    '{"answers": [{"pair_id": "<id>", "choice": "A" or "B", '
    '"confidence": "high"/"medium"/"low"}, ...]}\n'
    "One entry per pair, using the pair_id shown, covering EVERY pair listed. "
    "You must choose A or B for every pair; there is no tie option.\n\nPAIRS:\n\n"
)


def glm(prompt):
    last = None
    for kf in KEYS:
        if not kf.exists():
            continue
        key = kf.read_text().strip()
        for att in range(1, 5):
            try:
                body = {"model": "glm-5.2", "max_tokens": 32000,
                        "thinking": {"type": "enabled", "budget_tokens": 2048},
                        "messages": [{"role": "user", "content": prompt}]}
                req = urllib.request.Request(
                    URL, data=json.dumps(body).encode(),
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1200) as r:
                    d = json.loads(r.read())
                txt = "".join(b.get("text", "") for b in d.get("content", [])
                              if b.get("type") == "text")
                return txt, {"model": d.get("model"), "key": kf.name, "attempt": att,
                             "stop_reason": d.get("stop_reason")}
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                print(f"  glm attempt {att}: {last}", flush=True)
                time.sleep(20 * att)
    raise RuntimeError(f"glm failed: {last}")


def luna(prompt):
    wd = HERE / "phase1" / "wd_xfam"
    wd.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["codex", "exec", "--model", "gpt-5.6-luna", "-c",
                        "model_reasoning_effort=high", "-s", "read-only",
                        "--skip-git-repo-check", "--cd", str(wd), "-"],
                       input=prompt, capture_output=True, text=True, timeout=1800)
    return p.stdout, {"model": "gpt-5.6-luna", "rc": p.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--engine", default="glm")
    a = ap.parse_args()

    pairs = {p["pair_id"]: p for p in json.loads((HERE / "si_pairs.json").read_text())}
    matched = [p for p in pairs.values() if p["arm"] == "MATCHED"]
    anchors = [p for p in pairs.values() if p["arm"] == "ANCHOR_FRAGMENT"]
    pick = sorted(matched, key=lambda c: int(hashlib.sha256(
        (SALT + c["pair_id"]).encode()).hexdigest(), 16))[:a.n]
    pick += sorted(anchors, key=lambda c: int(hashlib.sha256(
        (SALT + c["pair_id"]).encode()).hexdigest(), 16))[:6]

    body = "\n\n".join(
        f"--- pair_id={p['pair_id']} ---\nCONTEST PROMPT: {p['prompt']}\n"
        f"ENTRY A: {p['entry_A']}\nENTRY B: {p['entry_B']}" for p in pick)
    prompt = HOLISTIC + TAIL + body
    (HERE / "phase1" / f"xfam_{a.engine}_prompt.txt").write_text(prompt)

    raw, meta = (glm if a.engine == "glm" else luna)(prompt)
    (HERE / "phase1" / f"xfam_{a.engine}_raw.txt").write_text(raw)
    b = raw.rsplit("tokens used", 1)[0] if "tokens used" in raw else raw
    d = None
    for s in sorted(JSON_RE.findall(b), key=len, reverse=True):
        try:
            cand = json.loads(s)
            if "answers" in cand:
                d = cand
                break
        except json.JSONDecodeError:
            continue
    assert d, "no parseable answers object"

    # sol's answers on the same pairs
    sol = {}
    for f in (HERE / "out").glob("holistic_*.json"):
        for x in json.load(open(f)).get("answers", []):
            sol[x["pair_id"]] = x["choice"]

    hit = n = ahit = an = agree = both = 0
    detail = []
    for x in d["answers"]:
        pid, ch = str(x.get("pair_id", "")).strip(), str(x.get("choice", "")).strip().upper()
        if pid not in pairs or ch not in ("A", "B"):
            continue
        p = pairs[pid]
        ok = int(ch == p["pos_side"])
        if p["arm"] == "ANCHOR_FRAGMENT":
            an += 1
            ahit += ok
            continue
        n += 1
        hit += ok
        if pid in sol:
            both += 1
            agree += int(sol[pid] == ch)
        detail.append({"pair_id": pid, "xfam": ch, "sol": sol.get(pid),
                       "pos_side": p["pos_side"], "xfam_correct": bool(ok)})

    sol_hits = sum(1 for x in detail if sol.get(x["pair_id"]) == pairs[x["pair_id"]]["pos_side"])
    out = {
        "engine": a.engine, "meta": meta,
        "n_matched_pairs": n, "xfamily_accuracy": (hit / n) if n else None,
        "sol_accuracy_same_pairs": (sol_hits / n) if n else None,
        "n_both_answered": both, "per_pair_agreement": (agree / both) if both else None,
        "anchors": {"n": an, "acc": (ahit / an) if an else None},
        "note": "identical pair text and identical A/B order assignment as the main wave, "
                "so the two accuracies and the agreement rate are directly comparable",
        "detail": detail,
    }
    (HERE / "phase1" / f"xfam_{a.engine}_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=1))


if __name__ == "__main__":
    main()

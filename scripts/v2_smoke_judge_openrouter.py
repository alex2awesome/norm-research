"""Local smoke test: build 1 bundle prompt, send via OpenRouter, parse response.

Verifies the full assembly → call → parse pipeline before kicking off sk3.

Usage:
  python scripts/v2_smoke_judge_openrouter.py --model anthropic/claude-3.5-sonnet --bundle b0 --n-texts 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from openrouter import chat, make_client
from v2_assemble_judge_prompt import build_prompt


def parse_response(raw: str):
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e+1])


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-3.5-sonnet")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--n-texts", type=int, default=5)
    args = ap.parse_args()

    v2 = Path("runs/validity_full/full_v2")
    bundles = json.loads((v2 / "judge_bundles.json").read_text())
    datapoints = json.loads((v2 / "datapoints.json").read_text())
    enr_dir = v2 / "judge_enrichment"

    # Pick bundle
    if args.bundle:
        b = next(b for b in bundles if b["bundle_id"] == args.bundle)
    else:
        # First bundle with all enrichments ready
        for b in bundles:
            ok = all((enr_dir / f"{aid}.json").exists() for aid in b["aspect_ids"])
            if ok:
                break
        else:
            print("no bundle has full enrichments yet")
            return

    enrs = [json.loads((enr_dir / f"{aid}.json").read_text())
            for aid in b["aspect_ids"] if (enr_dir / f"{aid}.json").exists()]
    texts = [(d["datapoint_id"], d["text"]) for d in datapoints[:args.n_texts]]

    print(f"bundle: {b['bundle_id']}  aspects: {len(enrs)}/{len(b['aspect_ids'])}")
    print(f"aspect ids: {[e['aspect_id'] for e in enrs]}")
    print(f"texts: {len(texts)}  total dp: {len(datapoints)}")

    system, user = build_prompt(b, enrs, texts, text_max_chars=4000)
    sys_tok = len(system) // 4
    user_tok = len(user) // 4
    print(f"\nprompt size: ~{sys_tok+user_tok} tokens "
          f"(system {sys_tok} + user {user_tok})")

    print(f"\n=== sending to {args.model} ===")
    client = make_client()
    import time
    t0 = time.time()
    try:
        raw = await chat(client, args.model, system, user,
                          temperature=0.0, max_tokens=4000)
    except Exception as e:
        print(f"FAIL: {e}")
        return
    dt = time.time() - t0
    print(f"got response in {dt:.1f}s ({len(raw)} chars)")
    print(f"\n=== RAW (first 800 chars) ===")
    print(raw[:800])

    print(f"\n=== PARSE ===")
    try:
        obj = parse_response(raw)
    except Exception as e:
        print(f"PARSE FAIL: {e}")
        return
    if "results" not in obj:
        print(f"no 'results' field. keys: {list(obj.keys())}")
        return
    print(f"  n results: {len(obj['results'])}")
    for tr in obj["results"][:2]:
        tid = tr.get("text_id")
        scs = tr.get("scores", [])
        print(f"  text {tid}: {len(scs)} scores")
        for sc in scs[:3]:
            print(f"    {sc.get('aspect_id')}: "
                  f"applicable={sc.get('applicable')} "
                  f"score={sc.get('score')} "
                  f"reason={(sc.get('reason') or '')[:60]!r}")

    # Coverage check
    expected = len(texts) * len(enrs)
    actual = sum(len(tr.get("scores", [])) for tr in obj["results"])
    print(f"\nexpected {expected} score records, got {actual} "
          f"({actual/expected*100:.0f}% coverage)")
    if actual >= expected * 0.95:
        print("✅ smoke test passed — ready for sk3 scale-up")
    else:
        print("⚠️  coverage low, may need prompt tweak")


if __name__ == "__main__":
    asyncio.run(amain())

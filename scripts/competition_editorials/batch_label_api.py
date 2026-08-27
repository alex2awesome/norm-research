#!/usr/bin/env python3
"""Label editorial-similarity shards via the Anthropic Batch API (USC key).

Replicates the Max-plan worker's prompt EXACTLY (same R4 prompt, same
instruction block, same 8000-char truncation) so labels stay comparable —
only the transport changes (claude CLI -> Message Batches + prompt caching).

Usage:
  python3 batch_label_api.py --platforms cf ac            # submit
  python3 batch_label_api.py --platforms cf ac --poll     # poll + harvest

Writes results in the same JSONL format as worker.py into results/, so
build_training_table.py and the four-platform eval consume them unchanged.
"""
import argparse
import json
import re
import time
from pathlib import Path

import anthropic

SHARD_DIR = Path("/Users/spangher/Projects/stanford-research/norm-research/"
                 "outputs/v2_analysis/comp_fourplatform_label_shards")
KEY_PATH = Path.home() / ".anthropic-usc-key.txt"
MODEL = "claude-sonnet-4-6"
CHAR_BUDGET = 8000
BATCH_STATE = SHARD_DIR / "batch_api_state.json"


def truncate(s, n=CHAR_BUDGET):
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "\n...[truncated]"


def build_blocks(r4_prompt, pair):
    system_block = f"""{r4_prompt}

You will be given ONE pair only. Output a SINGLE JSON OBJECT (not an array):
{{"label": 0 or 1, "reason": "<at most 20 words>"}}"""
    user_block = f"""PAIR:
  source: {pair.get('source', '?')}
  language: {pair.get('language', '?')}
  cosine: {pair.get('cosine', 0.0):.4f}

EDITORIAL CODE:
```
{truncate(pair.get('editorial_text', ''))}
```

CANDIDATE CODE:
```
{truncate(pair.get('candidate_text', ''))}
```
"""
    return system_block, user_block


def parse_label(text):
    m = re.search(r'\{[^{}]*"label"\s*:\s*([01])[^{}]*\}', text, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
        return int(obj["label"]), str(obj.get("reason", ""))[:200]
    except Exception:
        return int(m.group(1)), ""


def pending_pairs(platforms):
    done = set()
    for f in (SHARD_DIR / "results").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add(r["pair_id"])
            except json.JSONDecodeError:
                pass
    out = []
    for plat in platforms:
        for f in sorted((SHARD_DIR / "shards").glob(f"shard_{plat}_*.jsonl")):
            for line in f.read_text().splitlines():
                pair = json.loads(line)
                if pair["pair_id"] not in done:
                    out.append(pair)
    return out


def submit(platforms):
    client = anthropic.Anthropic(api_key=KEY_PATH.read_text().strip())
    r4 = (SHARD_DIR / "r4_prompt.txt").read_text()
    pairs = pending_pairs(platforms)
    print(f"submitting {len(pairs)} pending pairs for {platforms}")
    requests = []
    for pair in pairs:
        system_block, user_block = build_blocks(r4, pair)
        requests.append({
            "custom_id": pair["pair_id"],
            "params": {
                "model": MODEL,
                # Sonnet reasons in prose before emitting the JSON; 200 tokens
                # truncated 34% of responses pre-JSON (stop_reason=max_tokens)
                "max_tokens": 2000,
                "system": [{"type": "text", "text": system_block,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user_block}],
            },
        })
    batch = client.messages.batches.create(requests=requests)
    state = json.loads(BATCH_STATE.read_text()) if BATCH_STATE.exists() else {}
    state[batch.id] = {"platforms": platforms, "n": len(pairs),
                       "submitted": time.strftime("%F %T")}
    BATCH_STATE.write_text(json.dumps(state, indent=2))
    print(f"batch {batch.id} submitted ({len(pairs)} requests). "
          f"Poll with --poll.")


def poll_and_harvest():
    client = anthropic.Anthropic(api_key=KEY_PATH.read_text().strip())
    state = json.loads(BATCH_STATE.read_text())
    pair_meta = {}
    for f in sorted((SHARD_DIR / "shards").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            p = json.loads(line)
            pair_meta[p["pair_id"]] = f.stem  # shard name
    for batch_id, info in state.items():
        if info.get("harvested"):
            continue
        b = client.messages.batches.retrieve(batch_id)
        print(f"{batch_id}: {b.processing_status} "
              f"(counts: {b.request_counts})")
        if b.processing_status != "ended":
            continue
        by_shard = {}
        n_ok = n_fail = 0
        for res in client.messages.batches.results(batch_id):
            pid = res.custom_id
            shard = pair_meta.get(pid, "shard_unknown")
            if res.result.type == "succeeded":
                text = "".join(blk.text for blk in res.result.message.content
                               if blk.type == "text")
                label, reason = parse_label(text)
                if label is not None:
                    by_shard.setdefault(shard, []).append({
                        "pair_id": pid, "shard": 0, "claude_label": label,
                        "claude_reason": reason, "wall_seconds": 0,
                        "attempts": 1, "status": "ok", "transport": "batch_api"})
                    n_ok += 1
                    continue
            n_fail += 1
            by_shard.setdefault(shard, []).append({
                "pair_id": pid, "shard": 0, "status": "parse_fail",
                "transport": "batch_api"})
        (SHARD_DIR / "results").mkdir(exist_ok=True)
        for shard, rows in by_shard.items():
            out = SHARD_DIR / "results" / f"{shard}.jsonl"
            with out.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        info["harvested"] = True
        info["n_ok"] = n_ok
        info["n_fail"] = n_fail
        BATCH_STATE.write_text(json.dumps(state, indent=2))
        print(f"harvested {n_ok} ok / {n_fail} fail -> results/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platforms", nargs="+", default=["cf", "ac"])
    ap.add_argument("--poll", action="store_true")
    args = ap.parse_args()
    if args.poll:
        poll_and_harvest()
    else:
        submit(args.platforms)


if __name__ == "__main__":
    main()

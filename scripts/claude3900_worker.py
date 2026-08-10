#!/usr/bin/env python3
"""
Worker for Stage 2 of the 3,900-pair Claude re-labeling.

For each (editorial_code, candidate_code) pair in its shard, calls:
    claude --print --model sonnet  <prompt>

where prompt = R4 system prompt + EDITORIAL CODE block + CANDIDATE CODE block
+ JSON response instruction. One Claude call per pair. No batching,
no heuristics, no shortcuts.

Writes shard parquet incrementally so progress is recoverable.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def build_user_prompt(system_prompt: str, editorial_code: str, candidate_code: str) -> str:
    return (
        f"{system_prompt.strip()}\n\n"
        f"EDITORIAL CODE:\n{editorial_code}\n\n"
        f"CANDIDATE CODE:\n{candidate_code}\n\n"
        'Respond with a single JSON object on its own (no surrounding prose, '
        'no markdown fence):\n'
        '{"verdict": 0 or 1, "reason": "<one or two short sentences>"}'
    )


_JSON_RE = re.compile(r"\{[^{}]*\"verdict\"\s*:\s*([01])[^{}]*\}", re.DOTALL)


def parse_response(text: str):
    """Return (verdict_int, reason_str) or (None, raw) on failure."""
    if text is None:
        return None, ""
    s = text.strip()
    # Strip code fences
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    # Try direct JSON load first
    try:
        obj = json.loads(s)
        v = int(obj.get("verdict"))
        r = str(obj.get("reason", ""))
        if v in (0, 1):
            return v, r
    except Exception:
        pass
    # Try to find a JSON object inside
    m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = int(obj.get("verdict"))
            r = str(obj.get("reason", ""))
            if v in (0, 1):
                return v, r
        except Exception:
            pass
    # Last resort: regex for verdict number
    m2 = _JSON_RE.search(s)
    if m2:
        try:
            v = int(m2.group(1))
            return v, s[:240]
        except Exception:
            pass
    return None, s[:240]


def call_claude(prompt: str, model: str = "sonnet", timeout_s: int = 180) -> tuple[str, int]:
    """Run `claude --print --model <model>` with prompt on stdin. Returns (stdout, returncode)."""
    proc = subprocess.run(
        ["claude", "--print", "--model", model],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return proc.stdout, proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--input", default="/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/comp_unified_claude3900_shards/_input_3900.parquet")
    ap.add_argument("--system-prompt", default="/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/r4_error_targeted.txt")
    ap.add_argument("--outdir", default="/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/comp_unified_claude3900_shards")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--flush-every", type=int, default=10)
    args = ap.parse_args()

    sys_prompt = Path(args.system_prompt).read_text()
    df = pd.read_parquet(args.input)
    shard_df = df[df["shard"] == args.shard].reset_index(drop=True).copy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"comp_unified_claude3900_shard{args.shard:02d}.parquet"
    log_path = outdir / f"shard{args.shard:02d}.log"

    # Resume if previously written
    done_ids = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        done_ids = set(prev["pair_id"].tolist())
        results = prev.to_dict(orient="records")
    else:
        results = []

    def log(msg: str):
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    log(f"shard {args.shard} start; total={len(shard_df)} resumed={len(done_ids)}")

    n_total = len(shard_df)
    n_done = 0
    n_parse_fail = 0
    n_retry_ok = 0
    t_start = time.time()

    for idx, row in shard_df.iterrows():
        pid = row["pair_id"]
        if pid in done_ids:
            continue
        prompt = build_user_prompt(sys_prompt, row["editorial_code"], row["candidate_code"])
        attempts = 0
        verdict = None
        reason = ""
        status = "ok"
        last_raw = ""
        t0 = time.time()
        for attempt in range(args.max_retries + 1):
            attempts += 1
            try:
                out, rc = call_claude(prompt, model=args.model, timeout_s=args.timeout)
            except subprocess.TimeoutExpired:
                out, rc = "", -1
                last_raw = "<timeout>"
                continue
            last_raw = out
            if rc != 0:
                continue
            v, r = parse_response(out)
            if v is not None:
                verdict = v
                reason = r
                if attempt > 0:
                    status = "retry_ok"
                    n_retry_ok += 1
                break
        wall = time.time() - t0
        if verdict is None:
            status = "parse_fail"
            n_parse_fail += 1

        results.append({
            "pair_id": pid,
            "platform": row.get("platform"),
            "language": row.get("language"),
            "max_sim": row.get("max_sim"),
            "qwen_label": row.get("qwen_label"),
            "qwen_reason": row.get("qwen_reason"),
            "claude_label": verdict,
            "claude_reason": reason,
            "status": status,
            "attempts": attempts,
            "wall_seconds": wall,
            "raw_tail": last_raw[-300:] if status == "parse_fail" else "",
        })
        n_done += 1

        if n_done % args.flush_every == 0 or n_done == 1:
            pd.DataFrame(results).to_parquet(out_path, index=False)
            elapsed = time.time() - t_start
            rate = n_done / elapsed * 60 if elapsed > 0 else 0
            log(f"shard {args.shard} progress {n_done}/{n_total - len(done_ids)} parse_fail={n_parse_fail} retry_ok={n_retry_ok} rate={rate:.1f}/min wall_last={wall:.1f}s")

    pd.DataFrame(results).to_parquet(out_path, index=False)
    elapsed = time.time() - t_start
    log(f"shard {args.shard} DONE total={n_done} parse_fail={n_parse_fail} retry_ok={n_retry_ok} elapsed_s={elapsed:.1f}")
    print(f"shard {args.shard} done; results={len(results)} parse_fail={n_parse_fail}")


if __name__ == "__main__":
    main()

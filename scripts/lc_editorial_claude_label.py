"""Dispatch 15 parallel Claude labeling jobs over the 1480-pair sample.

Writes prompt files and runs `claude --print --model sonnet` in parallel.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SAMPLE_JSONL = ROOT / "outputs/v2_analysis/lc_editorial_judging_sample_1500.jsonl"
WORK_DIR = ROOT / "outputs/v2_analysis/claude_labeling_work"
OUT_JSONL = ROOT / "outputs/v2_analysis/lc_editorial_judging_claude_raw.jsonl"

PROMPT_PREAMBLE = """You are labeling code pairs for approach-similarity.

For each pair below, the candidate code and the editorial code both attempt to solve the same LeetCode problem.

Classify each pair as ONE of:
  0 = same approach - same core algorithm + similar structure (allows minor variation in style)
  1 = partial overlap - one is a strict optimization/variant of the other, OR shares the main idea but differs in execution
  2 = different approach - fundamentally different algorithm (e.g. recursive vs iterative DP, hashmap vs sorting, two-pointer vs sliding window)
  3 = not real code - at least one side is markdown/prose/template stub, not a solving attempt

For each pair, output one JSON object per line:
  {"pair_id": "<id>", "label": <0|1|2|3>, "reason": "<5-15 words>"}

Return a JSON array (list) containing one object per input pair, in the same order. Do not invent pairs you didn't see. Use only the labels 0/1/2/3. Output ONLY the JSON array, no prose before or after.

Pairs to label:
"""


def make_shards(records, n_shards):
    shards = [[] for _ in range(n_shards)]
    for i, r in enumerate(records):
        shards[i % n_shards].append(r)
    return shards


def write_prompt(shard, prompt_path):
    # Strip down to only fields the labeler needs
    payload = [
        {
            "pair_id": r["pair_id"],
            "language": r["language"],
            "candidate_code": r["candidate_code"],
            "editorial_code": r["editorial_code"],
        }
        for r in shard
    ]
    body = PROMPT_PREAMBLE + json.dumps(payload, ensure_ascii=False)
    Path(prompt_path).write_text(body)


def run_one(shard_idx, prompt_path, out_path, model):
    t0 = time.time()
    prompt = Path(prompt_path).read_text()
    # Pipe prompt via stdin to keep argv short
    proc = subprocess.run(
        ["claude", "--print", "--model", model],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=2400,  # 40 min cap
    )
    elapsed = time.time() - t0
    Path(out_path).write_text(proc.stdout)
    return {
        "shard_idx": shard_idx,
        "rc": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_len": len(proc.stdout),
        "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=15)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--shard-only", type=int, default=None, help="run only this shard idx (for retry)")
    ap.add_argument("--max-workers", type=int, default=15)
    args = ap.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(SAMPLE_JSONL)]
    print(f"loaded {len(records)} pairs")
    shards = make_shards(records, args.n_shards)
    for i, s in enumerate(shards):
        print(f"  shard {i}: {len(s)} pairs")

    # Write prompts (always, so re-runs are byte-identical)
    prompt_paths = []
    out_paths = []
    for i, s in enumerate(shards):
        pp = WORK_DIR / f"shard_{i:02d}_prompt.txt"
        op = WORK_DIR / f"shard_{i:02d}_response.txt"
        write_prompt(s, pp)
        prompt_paths.append(pp)
        out_paths.append(op)

    to_run = [args.shard_only] if args.shard_only is not None else list(range(args.n_shards))
    print(f"\ndispatching {len(to_run)} shards with model={args.model}, max_workers={args.max_workers}")

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(run_one, i, prompt_paths[i], out_paths[i], args.model): i for i in to_run}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"  shard {r['shard_idx']:02d} done rc={r['rc']} {r['elapsed_s']:.1f}s out_len={r['stdout_len']}")
            if r["rc"] != 0:
                print(f"    STDERR: {r['stderr_tail']}")
    total = time.time() - t_start
    print(f"\nALL DONE in {total:.1f}s")

    # Write summary
    summary_path = WORK_DIR / "dispatch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

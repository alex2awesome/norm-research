"""Multi-dimensional similarity labeling for 2000 LC (candidate, editorial) pairs.

Dispatches N parallel `claude --print --model sonnet` jobs on the laptop.
Each shard receives ~100 pairs and returns a JSON array of dimension scores.

Dimensions (each 0-3):
  A_approach   : same algorithm / data-structure choice?
  A_lexical    : token overlap excluding boilerplate?
  A_structural : control-flow / AST shape similar?
  A_naming     : variable/function naming style similar?
  A_comments   : similar amount + style of comments?
  A_length     : similar code length?
  A_idiom      : similar use of language-specific idioms?
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SAMPLE_JSONL = ROOT / "outputs/v2_analysis/lc_multidim_sample_2000.jsonl"
WORK_DIR = ROOT / "outputs/v2_analysis/claude_multidim_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_PREAMBLE = """You are rating CODE-PAIR SIMILARITY along seven INDEPENDENT axes.

Each pair shows a CANDIDATE solution and an EDITORIAL solution to the same LeetCode problem (the candidate is user-submitted; the editorial is the official walkthrough). The two pieces of code may be in different languages — that's fine, judge each axis on its own terms.

For each pair, output ONE JSON object with these fields:
  pair_id        : the integer pair_id we gave you
  A_approach     : 0..3  (same algorithm / data-structure choice?)
                   0 = totally different algorithm (e.g. BFS vs dynamic programming)
                   3 = same algorithm AND same key data structures (e.g. both two-pointer on prefix sums)
  A_lexical      : 0..3  (word/token overlap excluding language boilerplate?)
                   0 = essentially no shared content tokens
                   3 = nearly the same identifiers, keywords, idioms (huge token overlap)
  A_structural   : 0..3  (control-flow / AST shape similar?)
                   0 = totally different shape (one recursive, other iterative; nesting depth very different)
                   3 = same structure: same loop nesting, same function decomposition, same branching skeleton
  A_naming       : 0..3  (variable/function naming style similar?)
                   0 = very different naming style (e.g. one uses i/j/k, other uses verbose snake_case)
                   3 = matching names OR matching naming conventions (camelCase vs snake_case, abbreviations, etc.)
  A_comments     : 0..3  (similar amount + style of commenting?)
                   0 = one has none, the other has extensive comments / docstrings
                   3 = both have very similar commenting habits (both none, or both heavily annotated in same style)
  A_length       : 0..3  (similar code length?)
                   0 = one is at least 3x the other
                   3 = within 20% of each other
  A_idiom        : 0..3  (similar use of language-specific idioms?)
                   0 = different idioms (e.g. explicit for-loop vs comprehension; iterative vs functional pipeline)
                   3 = same idiomatic style (both use comprehensions, both use STL algorithms, etc.)
  note           : 5-15 word explanation of WHY the scores differ across axes

CRITICAL: rate each axis INDEPENDENTLY. Two code pieces can be lexically very similar but structurally different, or use the same approach but at very different lengths. Spread your scores accordingly. Avoid giving identical numbers across all 7 axes unless the codes truly are nearly-identical (or truly disjoint).

Output a JSON ARRAY containing one object per input pair, in the same order. Use only integers 0/1/2/3 for the seven axes. Output ONLY the JSON array — no prose, no markdown fences.

Pairs to rate:
"""


def make_shards(records, n_shards):
    shards = [[] for _ in range(n_shards)]
    for i, r in enumerate(records):
        shards[i % n_shards].append(r)
    return shards


def write_prompt(shard, prompt_path):
    payload = [
        {
            "pair_id": r["pair_id"],
            "language": r["language"],
            "editorial_approach_hint": r.get("editorial_approach"),
            "candidate_code": r["candidate_code"],
            "editorial_code": r["editorial_code"],
        }
        for r in shard
    ]
    body = PROMPT_PREAMBLE + json.dumps(payload, ensure_ascii=False)
    Path(prompt_path).write_text(body)


def run_one(shard_idx, prompt_path, out_path, model, timeout):
    t0 = time.time()
    prompt = Path(prompt_path).read_text()
    try:
        proc = subprocess.run(
            ["claude", "--print", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
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
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "shard_idx": shard_idx,
            "rc": -1,
            "elapsed_s": elapsed,
            "stdout_len": 0,
            "stderr_tail": "TIMEOUT",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=20)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--shard-only", type=int, default=None, help="run only this shard idx (for retry)")
    ap.add_argument("--max-workers", type=int, default=20)
    ap.add_argument("--smoke", type=int, default=0, help="if N>0, run a single smoke shard of N pairs")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--rerun-failed", action="store_true", help="rerun shards whose response.txt is empty/missing or had a non-zero rc previously")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(SAMPLE_JSONL)]
    print(f"loaded {len(records)} pairs")

    if args.smoke > 0:
        shard = records[: args.smoke]
        pp = WORK_DIR / f"smoke_{args.smoke:04d}_prompt.txt"
        op = WORK_DIR / f"smoke_{args.smoke:04d}_response.txt"
        write_prompt(shard, pp)
        print(f"smoke: {args.smoke} pairs -> {pp}")
        r = run_one(0, pp, op, args.model, args.timeout)
        print(json.dumps(r, indent=2))
        return

    shards = make_shards(records, args.n_shards)
    for i, s in enumerate(shards):
        print(f"  shard {i:02d}: {len(s)} pairs")

    # Write prompts (idempotent — re-runs are byte-identical)
    prompt_paths, out_paths = [], []
    for i, s in enumerate(shards):
        pp = WORK_DIR / f"shard_{i:02d}_prompt.txt"
        op = WORK_DIR / f"shard_{i:02d}_response.txt"
        write_prompt(s, pp)
        prompt_paths.append(pp)
        out_paths.append(op)

    if args.shard_only is not None:
        to_run = [args.shard_only]
    elif args.rerun_failed:
        to_run = []
        for i in range(args.n_shards):
            op = out_paths[i]
            if not op.exists() or op.stat().st_size < 200:
                to_run.append(i)
                continue
            # Try to parse to check
            try:
                txt = op.read_text()
                # naive: needs to look like an array
                if "[" not in txt or "]" not in txt or '"pair_id"' not in txt:
                    to_run.append(i)
            except Exception:
                to_run.append(i)
        print(f"rerun-failed: {to_run}")
    else:
        to_run = list(range(args.n_shards))

    print(f"\ndispatching {len(to_run)} shards with model={args.model}, max_workers={args.max_workers}")

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(run_one, i, prompt_paths[i], out_paths[i], args.model, args.timeout): i for i in to_run}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"  shard {r['shard_idx']:02d} done rc={r['rc']} {r['elapsed_s']:.1f}s out_len={r['stdout_len']}")
            if r["rc"] != 0:
                print(f"    STDERR: {r['stderr_tail']}")
    total = time.time() - t_start
    print(f"\nALL DONE in {total:.1f}s")

    summary_path = WORK_DIR / "dispatch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

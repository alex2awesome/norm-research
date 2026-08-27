"""Label one shard of CR.SE answers via `claude --print --model sonnet`.

Usage: python crse_quality_label_shard.py <shard_path> <out_jsonl>
"""
import sys
import json
import re
import subprocess
from pathlib import Path

PROMPT_HEADER = """You are evaluating the CODE in a Stack Exchange Code Review answer. Each answer typically suggests an improved version of code or critiques the original.

For each answer, classify the code Quality:
  CORRECT = code looks correct, would compile/run, addresses the problem as stated
  SYNTAX_ERR = code has a clear syntax error (typos, missing colons, mismatched braces)
  WRONG_LOGIC = code compiles but has a logic bug (off-by-one, wrong condition, missing edge case)
  SLOW = code is correct but the algorithmic complexity is clearly suboptimal (e.g., O(n^2) where O(n log n) is straightforward)
  PARTIAL = answer contains code snippets but doesn't provide a complete solution (snippets are illustrative fragments)
  NOT_CODE = answer is mostly prose with no real code, or just discussing
  CANT_TELL = code is in a language/framework you can't reliably judge, or context is insufficient

For each, output one JSON line per answer (JSONL):
{"answer_id": <id>, "code_quality": "<one of above>", "evidence": "<3-15 word justification>"}

Be CONSERVATIVE - only call SYNTAX_ERR or WRONG_LOGIC if you can specifically identify the bug. Default to CORRECT or PARTIAL.

Output ONLY the JSONL lines, one per input answer, no other commentary.

Answers to label:
"""


def build_prompt(items):
    payload = json.dumps(items, ensure_ascii=False)
    return PROMPT_HEADER + payload


def call_claude(prompt: str) -> str:
    proc = subprocess.run(
        ["claude", "--print", "--model", "sonnet"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"claude rc={proc.returncode} stderr={proc.stderr[:500]}\n")
    return proc.stdout


def parse_jsonl(text: str):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "answer_id" in obj and "code_quality" in obj:
                rows.append(obj)
        except Exception:
            continue
    return rows


def main():
    shard_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    items = json.loads(shard_path.read_text())
    print(f"[{shard_path.name}] loading {len(items)} answers", flush=True)

    prompt = build_prompt(items)
    print(f"[{shard_path.name}] prompt {len(prompt)} chars; calling claude...", flush=True)
    out_text = call_claude(prompt)
    rows = parse_jsonl(out_text)
    print(f"[{shard_path.name}] got {len(rows)} valid JSON rows", flush=True)

    # If we got fewer than expected, save raw output for debugging.
    if len(rows) < len(items):
        debug_path = out_path.with_suffix(".raw.txt")
        debug_path.write_text(out_text)
        print(f"[{shard_path.name}] WARNING: expected {len(items)}, got {len(rows)}; raw saved {debug_path}", flush=True)

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{shard_path.name}] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

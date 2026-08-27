"""Label approach-matched LeetCode triples via `claude --print --model sonnet`.

For each triple (editorial, candidate_A, candidate_B), we ask Claude:
  1) Do both A and B use the same approach as the editorial?  (yes / no)
  2) If yes, which is MORE STYLISTICALLY similar to the editorial (naming,
     comments, code organization, idioms — ignoring algorithm choice)?
     (A / B / tie)
  3) Brief reason.

Output: outputs/v2_analysis/lc_approach_matched_triples.parquet (with labels)
Checkpointed JSONL is written for resumability.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
INPUT = REPO / "outputs/v2_analysis/lc_approach_matched_triples_with_bank.parquet"
CHKPT = REPO / "outputs/v2_analysis/lc_approach_matched_triples.labels.jsonl"
OUTPUT = REPO / "outputs/v2_analysis/lc_approach_matched_triples.parquet"

CLAUDE_BIN = "/Users/spangher/.local/bin/claude"

PROMPT_TMPL = """You are evaluating LeetCode solutions for stylistic similarity to a reference editorial.

PROBLEM SLUG: {slug}
DIFFICULTY: {difficulty}

EDITORIAL ({ed_lang}):
```{ed_lang}
{editorial_code}
```

CANDIDATE A ({lang_a}):
```{lang_a}
{code_a}
```

CANDIDATE B ({lang_b}):
```{lang_b}
{code_b}
```

Answer TWO questions:

Q1 (same_approach): Do BOTH candidates use the same core algorithmic approach as the editorial? Treat translations across languages as same-approach if the algorithmic strategy matches (e.g., both DFS, both DP with same recurrence, both sliding window). Answer "yes" only if BOTH A and B use the editorial's approach. Otherwise "no".

Q2 (style_winner): IF Q1 = yes, which candidate is MORE STYLISTICALLY similar to the editorial? Consider:
  - identifier naming (variable / function names mirror the editorial's choices)
  - comments (presence, density, style — does the editorial comment? do they?)
  - code organization (helper function structure, control-flow shape, line breaks)
  - idioms (use of stdlib helpers, loop forms, early returns)
  - DELIBERATELY IGNORE the algorithm choice itself (that's Q1)
Answer "A", "B", or "tie".

Output STRICT JSON only, no prose:
{{"same_approach": "yes" or "no", "style_winner": "A" or "B" or "tie" or "na", "reason": "<one sentence>"}}
"""


def build_prompt(row) -> str:
    return PROMPT_TMPL.format(
        slug=row["question_slug"],
        difficulty=row.get("difficulty", "?") or "?",
        ed_lang=str(row.get("editorial_lang", "") or "").lower() or "text",
        editorial_code=row["editorial_code"],
        lang_a=str(row.get("lang_a", "") or "").lower() or "text",
        code_a=row["code_a"],
        lang_b=str(row.get("lang_b", "") or "").lower() or "text",
        code_b=row["code_b"],
    )


def call_claude(prompt: str, model: str = "sonnet", timeout: int = 120) -> str:
    cmd = [CLAUDE_BIN, "--print", "--model", model]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"claude failed (rc={p.returncode}): {p.stderr[:500]}")
    return p.stdout.strip()


def parse_response(raw: str) -> dict:
    # Find first JSON object in response
    s = raw.strip()
    # Try to extract JSON
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"_parse_error": "no_json", "_raw": raw[:300]}
    blob = s[start : end + 1]
    try:
        d = json.loads(blob)
    except Exception as e:
        return {"_parse_error": f"json: {e}", "_raw": blob[:300]}
    # Normalize
    sa = str(d.get("same_approach", "")).strip().lower()
    sw = str(d.get("style_winner", "")).strip().upper()
    rs = str(d.get("reason", "")).strip()
    if sa not in {"yes", "no"}:
        sa = "no"
    if sw not in {"A", "B", "TIE", "NA"}:
        sw = "NA"
    return {"same_approach": sa, "style_winner": sw, "reason": rs}


def process_one(row_dict: dict, model: str) -> dict:
    tid = int(row_dict["triple_id"])
    prompt = build_prompt(row_dict)
    t0 = time.time()
    try:
        raw = call_claude(prompt, model=model)
        parsed = parse_response(raw)
    except Exception as e:
        parsed = {"_parse_error": f"exception: {e!r}", "_raw": ""}
    parsed["triple_id"] = tid
    parsed["elapsed_s"] = round(time.time() - t0, 2)
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", type=str, default="sonnet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(INPUT)
    if args.limit:
        df = df.iloc[args.start : args.start + args.limit].reset_index(drop=True)

    # Resume from checkpoint
    done_ids = set()
    if CHKPT.exists():
        with open(CHKPT) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(int(rec["triple_id"]))
                except Exception:
                    pass
    print(f"total={len(df)} already_done={len(done_ids)}", flush=True)

    todo = df[~df.triple_id.isin(done_ids)].to_dict("records")
    print(f"to process: {len(todo)}", flush=True)

    if not todo:
        print("all done, building final parquet")
    else:
        n_ok = 0
        n_err = 0
        t0 = time.time()
        CHKPT.parent.mkdir(parents=True, exist_ok=True)
        with open(CHKPT, "a") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, r, args.model): r for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"triple_id": int(futs[fut]["triple_id"]), "_parse_error": f"top: {e!r}"}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if "_parse_error" in rec:
                    n_err += 1
                else:
                    n_ok += 1
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(
                        f"  {i}/{len(todo)} ok={n_ok} err={n_err} elapsed={el:.0f}s",
                        flush=True,
                    )

    # Merge checkpoint into final parquet
    labels = []
    with open(CHKPT) as f:
        for line in f:
            try:
                labels.append(json.loads(line))
            except Exception:
                pass
    lab_df = pd.DataFrame(labels)
    print(f"labels loaded: {len(lab_df)}")
    if "_parse_error" in lab_df.columns:
        print("parse errors:", lab_df["_parse_error"].notna().sum())
    full = pd.read_parquet(INPUT).merge(lab_df, on="triple_id", how="left")
    full.to_parquet(OUTPUT)
    print(f"WROTE {OUTPUT} shape={full.shape}")


if __name__ == "__main__":
    main()

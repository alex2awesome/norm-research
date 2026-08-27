"""Empirical V/A/T level-assignment pilot.

For each rubric in a sample, collect scores across multiple measurement
methods on a fixed set of datapoints, then compute pairwise agreement and
variance decomposition to infer the rubric's level (V / A / T) -- without
needing per-feature gold labels (only datapoints).

Phases (each is a separate function so they can be re-run in isolation):
  1. code_gen        rubric -> Python `score(text) -> float` via K coding models
  2. code_exec       run each generated function on each datapoint
  3. llm_judge       rubric + datapoint -> score in [0,1] via M LLM judges
  4. paraphrase      rubric -> P paraphrases via one model
  5. paraphrase_judge each paraphrase + datapoint -> score via one judge
  6. aggregate       per-rubric pairwise ICC/alpha + variance decomposition

This file is the skeleton; prompts and the orchestration are here, the
actual model calls and execution sandbox are TODOs for Sat May 23.

Run: python scripts/metric_test_pilot.py --task code-review --n-rubrics 30 \\
                                        --n-datapoints 50 --output runs/pilot
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------- prompts ----------

CODE_GEN_PROMPT = """You are given an evaluation rubric. Write a single Python function `score(text: str) -> float` that returns a score in [0.0, 1.0] indicating how well `text` satisfies the rubric.

Constraints:
- `text` is a string (a code snippet, a review comment, a press release, etc.).
- Return 1.0 if the rubric is fully satisfied, 0.0 if clearly violated, intermediate if partial, 0.5 if not applicable / uncertain.
- Use ONLY the Python standard library. Do NOT import third-party packages.
- The function must run without network or filesystem access.
- The function must handle empty / malformed `text` gracefully (return 0.5).
- Wrap any risky logic in try/except; never raise.

Rubric: "{rubric}"

Output ONLY the Python code -- no markdown fences, no commentary. Start with `def score` or with the imports you need."""


LLM_JUDGE_PROMPT_SYSTEM = """You score a datapoint on a single evaluation rubric.

Output ONLY a single integer 0..10. No commentary, no explanation.
- 0 = the datapoint clearly violates the rubric
- 5 = neutral / not applicable / cannot tell
- 10 = the datapoint clearly satisfies the rubric"""


LLM_JUDGE_PROMPT_USER = """Rubric: "{rubric}"

Datapoint:
{datapoint}

Score (0..10):"""


PARAPHRASE_PROMPT = """Rewrite this evaluation rubric in 4 different ways. Each rewrite should preserve the EXACT MEANING but use different wording, sentence structure, and word choice. The rewrites should be substantively interchangeable -- a competent reviewer would judge them as the same rubric.

Original rubric: "{rubric}"

Output VALID JSON ONLY, no markdown fences:
{{"paraphrases": ["...", "...", "...", "..."]}}"""


# ---------- code execution sandbox ----------

def run_score(code: str, text: str, timeout: float = 5.0) -> float | None:
    """Execute generated `score(text)` in a subprocess with timeout.

    Returns the float score on success, None on failure (timeout, exception,
    non-numeric return, out-of-range). NEVER raises to the caller.
    """
    runner = (
        f"{code}\n\n"
        "import json, sys\n"
        "text = json.loads(sys.stdin.read())\n"
        "try:\n"
        "    s = float(score(text))\n"
        "    if not (0.0 <= s <= 1.0):\n"
        "        s = max(0.0, min(1.0, s))\n"
        "    print(json.dumps({'score': s}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)[:120]}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            input=json.dumps(text),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        return out.get("score")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None


def extract_code(text: str) -> str:
    """Strip markdown fences if the coder put them in despite instructions."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


# ---------- data loading (per-task) ----------

def load_datapoints_code_review(n: int = 50, seed: int = 0):
    """Sample n code-review datapoints. Format each as a single text blob
    combining PR title + diff hunk + reviewer comment so that any of the
    rubric variants (code-level, comment-level, PR-level) has context."""
    import pandas as pd
    df = pd.read_csv(
        "datasets/code-review/code_review_modeling_dataset.csv.gz",
        usecols=["pr_title", "diff_hunk", "body", "outcome",
                 "pr_additions", "pr_deletions", "pr_changed_files"],
    ).dropna(subset=["diff_hunk", "body"])
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = []
    for i, r in sample.iterrows():
        text = (f"PR title: {r['pr_title']}\n\n"
                f"Code diff being reviewed:\n{r['diff_hunk']}\n\n"
                f"Review comment: {r['body']}")
        rows.append({"id": int(i), "text": text, "outcome": r["outcome"]})
    return rows


# ---------- rubric loading ----------

def load_rubric_sample(task: str, n: int = 30, seed: int = 0):
    """Sample n R1 rules from the locked clustering for `task`.

    TODO Sat May 23: stratify by inferred level if we have per-rule labels;
    for now just random sample multi-member R1 families (so the rubric has
    member text to draw on).
    """
    import random
    fp = (f"outputs/analyses/structural_metrics/r1_v4a/"
          f"r1_families_{task}.json")
    d = json.loads(Path(fp).read_text())
    fams = [f for f in d["families"] if len(f["cluster_ids"]) >= 2]
    fams = random.Random(seed).sample(fams, min(n, len(fams)))
    return [{"family_id": f["family_id"],
             "name": f["name"],
             "description": f["description"]} for f in fams]


# ---------- skeleton orchestration ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="code-review")
    ap.add_argument("--n-rubrics", type=int, default=30)
    ap.add_argument("--n-datapoints", type=int, default=50)
    ap.add_argument("--output", default="runs/metric_test_pilot")
    ap.add_argument("--phase", default="all",
                    choices=["all", "code_gen", "code_exec", "llm_judge",
                             "paraphrase", "paraphrase_judge", "aggregate"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)

    # --- data ---
    rubrics = load_rubric_sample(args.task, args.n_rubrics, args.seed)
    if args.task == "code-review":
        datapoints = load_datapoints_code_review(args.n_datapoints, args.seed)
    else:
        raise NotImplementedError(f"datapoint loader for {args.task}")

    (out / "rubrics.json").write_text(json.dumps(rubrics, indent=1))
    (out / "datapoints.json").write_text(json.dumps(datapoints, indent=1))
    print(f"prepared {len(rubrics)} rubrics x {len(datapoints)} datapoints")
    print(f"-> {out}")
    print("\nTODO Sat: wire up phases:")
    print("  code_gen        rubric x K coders -> code")
    print("  code_exec       code x datapoint -> score")
    print("  llm_judge       rubric x datapoint x M judges -> score")
    print("  paraphrase      rubric -> 4 paraphrases")
    print("  paraphrase_judge paraphrase x datapoint x 1 judge -> score")
    print("  aggregate       per-rubric agreement matrices + variance decomp")


if __name__ == "__main__":
    main()

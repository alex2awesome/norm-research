#!/usr/bin/env python
"""LLM judge for math answer correctness on the score<=0 subset.

Sends question+answer to gpt-5-mini and asks whether the answer is
mathematically correct. Saves results to a JSONL file for downstream
filtering of the binary dataset.

Runs on: 8K random sample of score=0 + all ~2.6K score<0 answers.
"""

import asyncio
import csv
import gzip
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
SPLITS_DIR = DATA_DIR / "splits_150k"
OUTPUT_PATH = DATA_DIR / "correctness_judgments.jsonl"

JUDGE_PROMPT = """\
You are evaluating whether a mathematics answer is correct.

Read the question and answer below. Determine whether the answer is \
MATHEMATICALLY CORRECT — i.e., the reasoning is valid, the final result \
is right, and there are no significant mathematical errors.

Minor issues (notation inconsistencies, missing edge cases that don't \
affect the main argument) should NOT make an answer "incorrect." Only \
flag as incorrect if there is a genuine mathematical error, a wrong \
final answer, or fatally flawed reasoning.

If the answer is just a link, a "me too" comment, or doesn't actually \
attempt to answer the question, judge it as INCORRECT.

Question and Answer:
{text}

Reply with EXACTLY one of these two formats:
CORRECT: <one sentence explaining why>
INCORRECT: <one sentence explaining the error>"""

MODEL = "gpt-5-mini"
CONCURRENCY = 30
N_SAMPLE_SCORE_ZERO = 8000
SEED = 42


async def judge_one(client, row_id: str, text: str, sem: asyncio.Semaphore):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=1.0,
                max_completion_tokens=8000,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(text=text[:3000])}],
            )
            response = (resp.choices[0].message.content or "").strip()
            is_correct = response.upper().startswith("CORRECT")
            return row_id, is_correct, response
        except Exception as exc:
            logger.warning("Judge failed for %s: %s", row_id, exc)
            return row_id, None, str(exc)


async def judge_batch(rows: List[Dict], output_path: Path):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)

    # Load existing results (for resumability)
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done.add(rec["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        logger.info("Resuming: %d already judged", len(done))

    pending = [r for r in rows if r["id"] not in done]
    logger.info("Judging %d answers (%d already done)", len(pending), len(done))

    tasks = [
        asyncio.create_task(judge_one(client, r["id"], r["text"], sem))
        for r in pending
    ]

    completed = 0
    n_correct = 0
    n_incorrect = 0
    n_error = 0

    with open(output_path, "a") as f:
        for coro in asyncio.as_completed(tasks):
            row_id, is_correct, response = await coro
            rec = {"id": row_id, "is_correct": is_correct, "response": response[:500]}
            f.write(json.dumps(rec) + "\n")
            f.flush()

            completed += 1
            if is_correct is True:
                n_correct += 1
            elif is_correct is False:
                n_incorrect += 1
            else:
                n_error += 1

            if completed % 500 == 0 or completed == len(tasks):
                logger.info(
                    "Progress: %d/%d | correct=%d (%.1f%%) incorrect=%d (%.1f%%) error=%d",
                    completed, len(tasks),
                    n_correct, 100 * n_correct / max(completed, 1),
                    n_incorrect, 100 * n_incorrect / max(completed, 1),
                    n_error,
                )


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required")

    rng = random.Random(SEED)

    # Load all label=0 rows from all splits
    all_neg_rows = []
    for split in ["train", "eval", "test"]:
        path = SPLITS_DIR / f"{split}.csv.gz"
        with gzip.open(path, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["label"]) == 0:
                    all_neg_rows.append(row)

    score_neg = [r for r in all_neg_rows if int(r["answer_score"]) < 0]
    score_zero = [r for r in all_neg_rows if int(r["answer_score"]) == 0]

    logger.info("Label=0 answers: %d total (%d score<0, %d score=0)",
                len(all_neg_rows), len(score_neg), len(score_zero))

    # Sample
    sample_zero = rng.sample(score_zero, min(N_SAMPLE_SCORE_ZERO, len(score_zero)))
    to_judge = score_neg + sample_zero
    rng.shuffle(to_judge)

    logger.info("Judging: %d score<0 + %d score=0 sample = %d total",
                len(score_neg), len(sample_zero), len(to_judge))

    asyncio.run(judge_batch(to_judge, OUTPUT_PATH))

    # Summary
    results = {}
    with open(OUTPUT_PATH) as f:
        for line in f:
            rec = json.loads(line.strip())
            results[rec["id"]] = rec["is_correct"]

    for subset_name, subset in [("score<0", score_neg), ("score=0 sample", sample_zero)]:
        judged = [results.get(r["id"]) for r in subset if r["id"] in results]
        n = len(judged)
        n_correct = sum(1 for j in judged if j is True)
        n_incorrect = sum(1 for j in judged if j is False)
        n_unknown = sum(1 for j in judged if j is None)
        logger.info(
            "%s: %d judged — %d correct (%.1f%%), %d incorrect (%.1f%%), %d error",
            subset_name, n,
            n_correct, 100 * n_correct / max(n, 1),
            n_incorrect, 100 * n_incorrect / max(n, 1),
            n_unknown,
        )


if __name__ == "__main__":
    main()

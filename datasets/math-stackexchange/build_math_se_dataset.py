#!/usr/bin/env python
"""Download and build Math.SE preference dataset from the Stack Exchange data dump.

Downloads the 7z archive from Archive.org, extracts Posts.xml and Votes.xml,
and builds preference pairs: for each question with 2+ answers, the
highest-scored answer is 'chosen' and the lowest-scored is 'rejected'.

Output: train/eval/test CSV splits with columns:
  id, question_id, question_text, question_tags,
  text (chosen or rejected answer), label (1=chosen, 0=rejected)
"""

import csv
import gzip
import hashlib
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent
DUMP_URL = "https://archive.org/download/stackexchange/math.stackexchange.com.7z"
ARCHIVE_PATH = DATA_DIR / "math.stackexchange.com.7z"
EXTRACT_DIR = DATA_DIR / "raw_dump"


def download_dump():
    if ARCHIVE_PATH.exists():
        print(f"Archive already exists: {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size / 1e9:.1f} GB)")
        return
    print(f"Downloading {DUMP_URL}...")
    subprocess.run(["wget", "-c", "-O", str(ARCHIVE_PATH), DUMP_URL], check=True)
    print(f"Downloaded: {ARCHIVE_PATH.stat().st_size / 1e9:.1f} GB")


def extract_dump():
    posts_path = EXTRACT_DIR / "Posts.xml"
    if posts_path.exists():
        print(f"Already extracted: {posts_path}")
        return
    EXTRACT_DIR.mkdir(exist_ok=True)
    print(f"Extracting 7z archive to {EXTRACT_DIR}...")
    # Only extract Posts.xml (the big one we need)
    subprocess.run(
        ["7z", "e", str(ARCHIVE_PATH), "-o" + str(EXTRACT_DIR), "Posts.xml", "-y"],
        check=True,
    )
    print("Extracted Posts.xml")


def parse_posts():
    """Parse Posts.xml iteratively (memory-efficient for large files).

    PostTypeId=1 → Question, PostTypeId=2 → Answer.
    """
    posts_path = EXTRACT_DIR / "Posts.xml"
    print(f"Parsing {posts_path}...")

    questions = {}  # id -> {title, body, tags, score, creation_date}
    answers_by_question = defaultdict(list)  # parent_id -> [{id, body, score, creation_date}]

    n_questions = 0
    n_answers = 0

    for event, elem in ET.iterparse(str(posts_path), events=("end",)):
        if elem.tag != "row":
            continue

        post_type = elem.get("PostTypeId")

        if post_type == "1":  # Question
            qid = elem.get("Id")
            questions[qid] = {
                "title": elem.get("Title", ""),
                "body": elem.get("Body", ""),
                "tags": elem.get("Tags", ""),
                "score": int(elem.get("Score", 0)),
                "creation_date": elem.get("CreationDate", ""),
            }
            n_questions += 1
            if n_questions % 100000 == 0:
                print(f"  Questions: {n_questions}, Answers: {n_answers}")

        elif post_type == "2":  # Answer
            parent_id = elem.get("ParentId")
            if parent_id:
                answers_by_question[parent_id].append({
                    "id": elem.get("Id"),
                    "body": elem.get("Body", ""),
                    "score": int(elem.get("Score", 0)),
                    "creation_date": elem.get("CreationDate", ""),
                })
                n_answers += 1

        elem.clear()

    print(f"Parsed: {n_questions} questions, {n_answers} answers")
    return questions, answers_by_question


def strip_html(html_text):
    """Simple HTML tag stripping."""
    import re
    text = re.sub(r'<[^>]+>', ' ', html_text)
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    text = ' '.join(text.split())
    return text.strip()


def parse_tags(tag_str):
    """Parse SE tag format — handles both '|tag1|tag2|' and '<tag1><tag2>'."""
    import re
    if not tag_str:
        return ''
    # Pipe-delimited (data dump format): |tag1|tag2|
    if tag_str.startswith('|') or tag_str.endswith('|'):
        return '|'.join(t for t in tag_str.split('|') if t)
    # Angle-bracket (API format): <tag1><tag2>
    tags = re.findall(r'<([^>]+)>', tag_str)
    return '|'.join(tags)


def build_preference_pairs(questions, answers_by_question, min_score_diff=1):
    """Build preference pairs from questions with 2+ answers.

    For each question with 2+ answers:
    - chosen = highest-scored answer
    - rejected = lowest-scored answer (must have score_diff >= min_score_diff)
    """
    pairs = []
    n_skipped_single = 0
    n_skipped_tie = 0

    for qid, q in questions.items():
        answers = answers_by_question.get(qid, [])
        if len(answers) < 2:
            n_skipped_single += 1
            continue

        answers_sorted = sorted(answers, key=lambda a: a["score"], reverse=True)
        best = answers_sorted[0]
        worst = answers_sorted[-1]

        if best["score"] - worst["score"] < min_score_diff:
            n_skipped_tie += 1
            continue

        question_text = q["title"] + "\n\n" + strip_html(q["body"])
        tags = parse_tags(q["tags"])

        pairs.append({
            "question_id": qid,
            "question_text": question_text.strip(),
            "question_tags": tags,
            "chosen_id": best["id"],
            "chosen_text": strip_html(best["body"]),
            "chosen_score": best["score"],
            "rejected_id": worst["id"],
            "rejected_text": strip_html(worst["body"]),
            "rejected_score": worst["score"],
            "score_diff": best["score"] - worst["score"],
            "n_answers": len(answers),
        })

    print(f"Built {len(pairs)} preference pairs")
    print(f"  Skipped (single answer): {n_skipped_single}")
    print(f"  Skipped (tie/small diff): {n_skipped_tie}")
    return pairs


def build_classification_splits(pairs, seed=42, train_frac=0.8, eval_frac=0.1):
    """Convert preference pairs into classification rows and split.

    Each pair becomes 2 rows:
    - chosen answer with label=1
    - rejected answer with label=0

    Text format: "Question: {q}\n\nAnswer: {a}"
    """
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_frac)
    n_eval = int(n * eval_frac)

    splits = {
        "train": pairs[:n_train],
        "eval": pairs[n_train:n_train + n_eval],
        "test": pairs[n_train + n_eval:],
    }

    out_dir = DATA_DIR / "splits"
    out_dir.mkdir(exist_ok=True)

    for split_name, split_pairs in splits.items():
        rows = []
        for p in split_pairs:
            base = {
                "question_id": p["question_id"],
                "question_tags": p["question_tags"],
                "score_diff": p["score_diff"],
                "n_answers": p["n_answers"],
            }
            # Chosen
            rows.append({
                **base,
                "id": f"chosen_{p['chosen_id']}",
                "text": f"Question: {p['question_text']}\n\nAnswer: {p['chosen_text']}",
                "label": 1,
                "answer_score": p["chosen_score"],
            })
            # Rejected
            rows.append({
                **base,
                "id": f"rejected_{p['rejected_id']}",
                "text": f"Question: {p['question_text']}\n\nAnswer: {p['rejected_text']}",
                "label": 0,
                "answer_score": p["rejected_score"],
            })

        rng.shuffle(rows)

        path = out_dir / f"{split_name}.csv.gz"
        import gzip
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "text", "label", "question_id", "question_tags",
                "score_diff", "n_answers", "answer_score",
            ])
            writer.writeheader()
            writer.writerows(rows)

        print(f"  {split_name}: {len(rows)} rows ({len(split_pairs)} pairs) → {path}")

    # Also save raw pairs for analysis
    import json
    pairs_path = DATA_DIR / "preference_pairs.jsonl.gz"
    with gzip.open(pairs_path, "wt") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  Raw pairs → {pairs_path}")


def print_stats(pairs):
    """Print dataset statistics."""
    import numpy as np

    scores = [p["score_diff"] for p in pairs]
    n_ans = [p["n_answers"] for p in pairs]

    # Tag distribution
    tag_counts = defaultdict(int)
    for p in pairs:
        for tag in p["question_tags"].split("|"):
            if tag:
                tag_counts[tag] += 1

    proof_tags = {"proof-writing", "proof-verification", "proof-explanation",
                  "alternative-proof", "solution-verification"}
    n_proof = sum(1 for p in pairs
                  if any(t in proof_tags for t in p["question_tags"].split("|")))

    print(f"\n{'='*50}")
    print(f"Dataset Statistics:")
    print(f"  Total preference pairs: {len(pairs)}")
    print(f"  Score diff: mean={np.mean(scores):.1f}, median={np.median(scores):.0f}, "
          f"max={max(scores)}")
    print(f"  Answers per question: mean={np.mean(n_ans):.1f}, max={max(n_ans)}")
    print(f"  Proof-related questions: {n_proof} ({100*n_proof/len(pairs):.1f}%)")
    print(f"  Top 15 tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {tag:<35} {count:>8}")
    print(f"{'='*50}")


def main():
    download_dump()
    extract_dump()
    questions, answers = parse_posts()
    pairs = build_preference_pairs(questions, answers, min_score_diff=1)
    print_stats(pairs)
    build_classification_splits(pairs)
    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build a binary classification dataset from Math.SE answers.

Positive (label=1): Accepted answers with score >= 3
Negative (label=0): Answers with score <= 0

Ambiguous middle (score 1-2, or high-score but not accepted) excluded
to create a clean signal gap.

Debiasing:
  - Length-balanced: positive and negative classes matched on answer
    length distribution (binned, sampled proportionally)
  - Question-disjoint: no question appears in both classes (prevents
    the model from learning question-level signal)
  - Minimum length filter: answers < 50 chars excluded (trivial/stub)
"""

import csv
import gzip
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
EXTRACT_DIR = DATA_DIR / "raw_dump"


def strip_html(html_text):
    text = re.sub(r'<[^>]+>', ' ', html_text)
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    text = ' '.join(text.split())
    return text.strip()


def parse_tags(tag_str):
    if not tag_str:
        return ''
    if tag_str.startswith('|') or tag_str.endswith('|'):
        return '|'.join(t for t in tag_str.split('|') if t)
    tags = re.findall(r'<([^>]+)>', tag_str)
    return '|'.join(tags)


def parse_posts():
    """Parse Posts.xml → questions + answers."""
    posts_path = EXTRACT_DIR / "Posts.xml"
    print(f"Parsing {posts_path}...")

    questions = {}
    answers = []
    accepted_ids = set()

    n_q, n_a = 0, 0
    for event, elem in ET.iterparse(str(posts_path), events=("end",)):
        if elem.tag != "row":
            continue

        post_type = elem.get("PostTypeId")
        if post_type == "1":
            qid = elem.get("Id")
            acc = elem.get("AcceptedAnswerId")
            questions[qid] = {
                "title": elem.get("Title", ""),
                "body": elem.get("Body", ""),
                "tags": elem.get("Tags", ""),
                "score": int(elem.get("Score", 0)),
            }
            if acc:
                accepted_ids.add(acc)
            n_q += 1

        elif post_type == "2":
            answers.append({
                "id": elem.get("Id"),
                "parent_id": elem.get("ParentId"),
                "body": elem.get("Body", ""),
                "score": int(elem.get("Score", 0)),
                "creation_date": elem.get("CreationDate", ""),
            })
            n_a += 1

        if (n_q + n_a) % 500000 == 0:
            print(f"  Q: {n_q}, A: {n_a}")
        elem.clear()

    print(f"Parsed: {n_q} questions, {n_a} answers, {len(accepted_ids)} accepted")
    return questions, answers, accepted_ids


def build_labeled_answers(questions, answers, accepted_ids,
                          pos_min_score=3, neg_max_score=0, min_answer_len=50):
    """Label answers:
    Positive: accepted AND score >= pos_min_score
    Negative: score <= neg_max_score
    Skip: everything else (ambiguous)
    """
    positives = []
    negatives = []
    n_skipped_ambiguous = 0
    n_skipped_short = 0
    n_skipped_no_question = 0

    for a in answers:
        q = questions.get(a["parent_id"])
        if not q:
            n_skipped_no_question += 1
            continue

        answer_text = strip_html(a["body"])
        if len(answer_text) < min_answer_len:
            n_skipped_short += 1
            continue

        is_accepted = a["id"] in accepted_ids
        score = a["score"]

        if is_accepted and score >= pos_min_score:
            question_text = q["title"] + "\n\n" + strip_html(q["body"])
            positives.append({
                "answer_id": a["id"],
                "question_id": a["parent_id"],
                "text": f"Question: {question_text.strip()}\n\nAnswer: {answer_text}",
                "answer_text": answer_text,
                "question_tags": parse_tags(q["tags"]),
                "answer_score": score,
                "is_accepted": True,
                "answer_len": len(answer_text),
            })
        elif score <= neg_max_score:
            question_text = q["title"] + "\n\n" + strip_html(q["body"])
            negatives.append({
                "answer_id": a["id"],
                "question_id": a["parent_id"],
                "text": f"Question: {question_text.strip()}\n\nAnswer: {answer_text}",
                "answer_text": answer_text,
                "question_tags": parse_tags(q["tags"]),
                "answer_score": score,
                "is_accepted": False,
                "answer_len": len(answer_text),
            })
        else:
            n_skipped_ambiguous += 1

    print(f"\nLabeled answers:")
    print(f"  Positive (accepted + score >= {pos_min_score}): {len(positives)}")
    print(f"  Negative (score <= {neg_max_score}): {len(negatives)}")
    print(f"  Skipped ambiguous: {n_skipped_ambiguous}")
    print(f"  Skipped short (<{min_answer_len} chars): {n_skipped_short}")
    print(f"  Skipped no question: {n_skipped_no_question}")
    return positives, negatives


def make_question_disjoint(positives, negatives):
    """Ensure no question appears in both classes.
    If a question has both positive and negative answers, drop it entirely."""
    pos_qids = {p["question_id"] for p in positives}
    neg_qids = {n["question_id"] for n in negatives}
    overlap = pos_qids & neg_qids

    if overlap:
        print(f"\nQuestion-disjoint filter: {len(overlap)} questions appear in both classes")
        positives = [p for p in positives if p["question_id"] not in overlap]
        negatives = [n for n in negatives if n["question_id"] not in overlap]
        print(f"  After filter: {len(positives)} positive, {len(negatives)} negative")
    else:
        print("\nNo question overlap between classes")
    return positives, negatives


def balanced_downsample(positives, negatives, n_len_bins=10, seed=42):
    """Downsample to equal 1/0 per primary tag AND matched on both
    question length and answer length.

    Strategy:
    1. Group by primary tag (first tag)
    2. Within each tag, balance pos/neg to min(pos, neg)
    3. Within each tag+class, joint-bin by (question_len_bin, answer_len_bin)
       and sample from the larger class to match the smaller's joint distribution
    """
    rng = random.Random(seed)

    # Add question_len to each item
    for item in positives + negatives:
        # Extract question portion length from the combined text
        text = item["text"]
        ans_start = text.find("\n\nAnswer: ")
        item["question_len"] = ans_start if ans_start > 0 else len(text) // 2

    # Group by primary tag
    def primary_tag(item):
        tags = item["question_tags"].split("|")
        return tags[0] if tags and tags[0] else "_none_"

    pos_by_tag = defaultdict(list)
    neg_by_tag = defaultdict(list)
    for p in positives:
        pos_by_tag[primary_tag(p)].append(p)
    for n in negatives:
        neg_by_tag[primary_tag(n)].append(n)

    all_tags = set(pos_by_tag.keys()) | set(neg_by_tag.keys())

    final_pos = []
    final_neg = []
    n_tags_skipped = 0

    for tag in sorted(all_tags):
        tag_pos = pos_by_tag.get(tag, [])
        tag_neg = neg_by_tag.get(tag, [])
        if not tag_pos or not tag_neg:
            n_tags_skipped += 1
            continue

        # Target count per class = min of the two
        target_per_class = min(len(tag_pos), len(tag_neg))

        # Determine which is smaller/larger within this tag
        if len(tag_pos) <= len(tag_neg):
            smaller, larger = tag_pos, tag_neg
        else:
            smaller, larger = tag_neg, tag_pos

        # Joint-bin by (question_len, answer_len) on the smaller class
        q_lens = [x["question_len"] for x in smaller]
        a_lens = [x["answer_len"] for x in smaller]
        q_edges = np.percentile(q_lens, np.linspace(0, 100, n_len_bins + 1))
        a_edges = np.percentile(a_lens, np.linspace(0, 100, n_len_bins + 1))
        q_edges[-1] += 1
        a_edges[-1] += 1

        def joint_bin(item):
            qb = int(np.searchsorted(q_edges[1:], item["question_len"]))
            ab = int(np.searchsorted(a_edges[1:], item["answer_len"]))
            return (qb, ab)

        small_binned = defaultdict(list)
        for x in smaller:
            small_binned[joint_bin(x)].append(x)

        large_binned = defaultdict(list)
        for x in larger:
            large_binned[joint_bin(x)].append(x)

        sampled_larger = []
        for bkey, small_items in small_binned.items():
            available = large_binned.get(bkey, [])
            if not available:
                continue
            n_sample = min(len(small_items), len(available))
            sampled_larger.extend(rng.sample(available, n_sample))

        # Also subsample smaller to match sampled_larger count (for exact balance)
        if len(sampled_larger) < len(smaller):
            smaller = rng.sample(smaller, len(sampled_larger))

        if len(tag_pos) <= len(tag_neg):
            final_pos.extend(smaller)
            final_neg.extend(sampled_larger)
        else:
            final_neg.extend(smaller)
            final_pos.extend(sampled_larger)

    print(f"\nBalanced downsampling (per-tag + length-matched):")
    print(f"  Tags processed: {len(all_tags) - n_tags_skipped} (skipped {n_tags_skipped} one-class tags)")
    print(f"  Final positive: {len(final_pos)}")
    print(f"  Final negative: {len(final_neg)}")

    pos_qlens = [p["question_len"] for p in final_pos]
    neg_qlens = [n["question_len"] for n in final_neg]
    pos_alens = [p["answer_len"] for p in final_pos]
    neg_alens = [n["answer_len"] for n in final_neg]
    print(f"  Question len — pos: mean={np.mean(pos_qlens):.0f} med={np.median(pos_qlens):.0f}"
          f"  neg: mean={np.mean(neg_qlens):.0f} med={np.median(neg_qlens):.0f}")
    print(f"  Answer len   — pos: mean={np.mean(pos_alens):.0f} med={np.median(pos_alens):.0f}"
          f"  neg: mean={np.mean(neg_alens):.0f} med={np.median(neg_alens):.0f}")

    return final_pos, final_neg


def build_splits(positives, negatives, seed=42, train_frac=0.8, eval_frac=0.1):
    """Build train/eval/test splits."""
    rng = random.Random(seed)

    # Combine and label
    rows = []
    for p in positives:
        rows.append({
            "id": f"pos_{p['answer_id']}",
            "text": p["text"],
            "label": 1,
            "question_id": p["question_id"],
            "question_tags": p["question_tags"],
            "answer_score": p["answer_score"],
            "answer_len": p["answer_len"],
        })
    for n in negatives:
        rows.append({
            "id": f"neg_{n['answer_id']}",
            "text": n["text"],
            "label": 0,
            "question_id": n["question_id"],
            "question_tags": n["question_tags"],
            "answer_score": n["answer_score"],
            "answer_len": n["answer_len"],
        })

    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_frac)
    n_eval = int(n * eval_frac)

    splits = {
        "train": rows[:n_train],
        "eval": rows[n_train:n_train + n_eval],
        "test": rows[n_train + n_eval:],
    }

    out_dir = DATA_DIR / "splits"
    out_dir.mkdir(exist_ok=True)

    for split_name, split_rows in splits.items():
        path = out_dir / f"{split_name}.csv.gz"
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "text", "label", "question_id", "question_tags",
                "answer_score", "answer_len",
            ])
            writer.writeheader()
            writer.writerows(split_rows)

        n_pos = sum(1 for r in split_rows if r["label"] == 1)
        n_neg = sum(1 for r in split_rows if r["label"] == 0)
        print(f"  {split_name}: {len(split_rows)} rows ({n_pos} pos, {n_neg} neg) → {path}")


def print_stats(positives, negatives):
    print(f"\n{'='*60}")
    print(f"Final Dataset Statistics:")
    print(f"  Positive: {len(positives)}")
    print(f"  Negative: {len(negatives)}")
    print(f"  Total: {len(positives) + len(negatives)}")

    # Tag analysis
    all_items = positives + negatives
    tag_counts = defaultdict(lambda: [0, 0])  # tag -> [pos_count, neg_count]
    for item in positives:
        for t in item["question_tags"].split("|"):
            if t:
                tag_counts[t][0] += 1
    for item in negatives:
        for t in item["question_tags"].split("|"):
            if t:
                tag_counts[t][1] += 1

    print(f"\n  Top 10 tags (pos/neg balance):")
    for tag, (pc, nc) in sorted(tag_counts.items(), key=lambda x: -(x[1][0]+x[1][1]))[:10]:
        total = pc + nc
        print(f"    {tag:<35} {total:>6}  (pos: {100*pc/total:.0f}%, neg: {100*nc/total:.0f}%)")

    # Length stats
    pos_lens = [p["answer_len"] for p in positives]
    neg_lens = [n["answer_len"] for n in negatives]
    print(f"\n  Answer length (chars):")
    print(f"    Positive: mean={np.mean(pos_lens):.0f}, median={np.median(pos_lens):.0f}")
    print(f"    Negative: mean={np.mean(neg_lens):.0f}, median={np.median(neg_lens):.0f}")

    # Score stats
    pos_scores = [p["answer_score"] for p in positives]
    neg_scores = [n["answer_score"] for n in negatives]
    print(f"\n  Answer score:")
    print(f"    Positive: mean={np.mean(pos_scores):.1f}, median={np.median(pos_scores):.0f}")
    print(f"    Negative: mean={np.mean(neg_scores):.1f}, median={np.median(neg_scores):.0f}")
    print(f"{'='*60}")


def main():
    questions, answers, accepted_ids = parse_posts()
    positives, negatives = build_labeled_answers(
        questions, answers, accepted_ids,
        pos_min_score=3, neg_max_score=0, min_answer_len=50,
    )
    positives, negatives = make_question_disjoint(positives, negatives)
    positives, negatives = balanced_downsample(positives, negatives)
    print_stats(positives, negatives)
    build_splits(positives, negatives)
    print("\nDone!")


if __name__ == "__main__":
    main()

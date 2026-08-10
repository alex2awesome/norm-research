#!/usr/bin/env python
"""Build the v3 position-matched pointwise Math.SE binary dataset.

Positive (label=1): Accepted answers with score >= --pos-min-score (default 3)
Negative (label=0): Answers with score <= --neg-max-score (default 0)
Ambiguous middle excluded, as in build_binary_dataset.py.

Fixes the three open issues documented in README.md:
  1. Metadata columns carried in the output: question_id, answer_id,
     answer_position (rank by CreationDate among ALL answers to the question,
     1 = first), n_answers_on_question, answer_age_gap_days (this answer's
     CreationDate minus the question's FIRST answer's CreationDate),
     answer_year, score, accepted, primary_tag.
  2. Position matching: the joint matched-downsampling key is
     (question_len_bin x answer_len_bin x position_bin) per primary tag,
     where position_bin in {1, 2, 3+}. This kills the first-answer-advantage
     confound (earlier answer wins 67.3% of within-question pairs).
  3. Group split baked in: a `split` column (train/eval/test, 80/10/10),
     hash-based on question_id so all answers to a question share a split.

Target size ~--target-rows (default 100K, 50/50), with priority-tag
retention as in build_150k_focused.py (keep all proof/intuition/soft-question
rows, downsample the rest to fill quota).

IMPORTANT ordering invariant: answer_position / n_answers_on_question /
answer_age_gap_days are computed over ALL answers to a question in the dump,
BEFORE any labeling, length, or disjointness filtering.

Streams Posts.xml (tens of GB) in two iterparse passes:
  pass 1: questions (id, AcceptedAnswerId, title, tags, stripped body)
  pass 2: answers (timing metadata for ALL answers; full text only for
          labeled candidates)

Outputs (next to this script):
  math_se_v3_position_matched.csv.gz            single file with `split` column
  math_se_v3_position_matched.manifest.json     self-audit build manifest
"""

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
EXTRACT_DIR = DATA_DIR / "raw_dump"

PRIORITY_TAGS = {
    "proof-writing", "proof-verification", "proof-explanation",
    "alternative-proof", "solution-verification",
    "intuition", "soft-question",
}

POSITION_BIN_LABELS = ("1", "2", "3+")


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


def parse_creation_date(date_str):
    """Parse a Stack Exchange CreationDate (e.g. '2010-07-20T19:12:14.353')."""
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def position_bin(position):
    """Bin answer position into 1 / 2 / 3+."""
    return min(position, 3)


def parse_questions(posts_path):
    """Pass 1: questions only (PostTypeId=1).

    Stores stripped body (needed for the output `text` column) — strictly
    lighter than build_binary_dataset.py, which keeps raw HTML for all posts.
    """
    print(f"Pass 1 (questions): parsing {posts_path}...")
    questions = {}
    accepted_ids = set()

    n_q, n_rows = 0, 0
    for event, elem in ET.iterparse(str(posts_path), events=("end",)):
        if elem.tag != "row":
            continue
        n_rows += 1

        if elem.get("PostTypeId") == "1":
            qid = elem.get("Id")
            acc = elem.get("AcceptedAnswerId")
            body = strip_html(elem.get("Body", ""))
            questions[qid] = {
                "title": elem.get("Title", ""),
                "body": body,
                "tags": parse_tags(elem.get("Tags", "")),
                "creation_date": elem.get("CreationDate", ""),
                "question_len": len(elem.get("Title", "")) + 2 + len(body),
            }
            if acc:
                accepted_ids.add(acc)
            n_q += 1

        if n_rows % 1000000 == 0:
            print(f"  rows: {n_rows}, questions: {n_q}")
        elem.clear()

    print(f"Pass 1 done: {n_q} questions, {len(accepted_ids)} accepted-answer ids")
    return questions, accepted_ids


def parse_answers(posts_path, questions, accepted_ids,
                  pos_min_score=3, neg_max_score=0, min_answer_len=50):
    """Pass 2: answers only (PostTypeId=2).

    For EVERY answer whose parent question exists, record
    (creation_date, answer_id) in q_answer_times — position metadata must be
    computed over ALL answers, BEFORE any labeling/length filtering.

    Full stripped text is kept only for labeled candidates:
      Positive: accepted AND score >= pos_min_score
      Negative: score <= neg_max_score
      Skip: everything else (ambiguous), and stubs < min_answer_len chars.
    """
    print(f"\nPass 2 (answers): parsing {posts_path}...")
    q_answer_times = defaultdict(list)  # qid -> [(creation_date, answer_id_int)]
    positives = []
    negatives = []
    n_a = 0
    n_skipped_ambiguous = 0
    n_skipped_short = 0
    n_skipped_no_question = 0

    for event, elem in ET.iterparse(str(posts_path), events=("end",)):
        if elem.tag != "row":
            continue
        if elem.get("PostTypeId") != "2":
            elem.clear()
            continue

        n_a += 1
        if n_a % 1000000 == 0:
            print(f"  answers: {n_a} (pos: {len(positives)}, neg: {len(negatives)})")

        aid = elem.get("Id")
        parent_id = elem.get("ParentId")
        creation_date = elem.get("CreationDate", "")
        q = questions.get(parent_id)
        if not q:
            n_skipped_no_question += 1
            elem.clear()
            continue

        # Position bookkeeping for ALL answers — before any filtering.
        q_answer_times[parent_id].append((creation_date, int(aid)))

        score = int(elem.get("Score", 0))
        is_accepted = aid in accepted_ids
        is_pos = is_accepted and score >= pos_min_score
        is_neg = score <= neg_max_score

        if not is_pos and not is_neg:
            n_skipped_ambiguous += 1
            elem.clear()
            continue

        answer_text = strip_html(elem.get("Body", ""))
        if len(answer_text) < min_answer_len:
            n_skipped_short += 1
            elem.clear()
            continue

        record = {
            "answer_id": aid,
            "question_id": parent_id,
            "answer_text": answer_text,
            "answer_len": len(answer_text),
            "question_len": q["question_len"],
            "question_tags": q["tags"],
            "score": score,
            "accepted": is_accepted,
            "creation_date": creation_date,
        }
        if is_pos:
            positives.append(record)
        else:
            negatives.append(record)
        elem.clear()

    print(f"Pass 2 done: {n_a} answers")
    print(f"  Positive (accepted + score >= {pos_min_score}): {len(positives)}")
    print(f"  Negative (score <= {neg_max_score}): {len(negatives)}")
    print(f"  Skipped ambiguous: {n_skipped_ambiguous}")
    print(f"  Skipped short (<{min_answer_len} chars): {n_skipped_short}")
    print(f"  Skipped no question: {n_skipped_no_question}")

    stage_counts = {
        "answers_total": n_a,
        "labeled_positive": len(positives),
        "labeled_negative": len(negatives),
        "skipped_ambiguous": n_skipped_ambiguous,
        "skipped_short": n_skipped_short,
        "skipped_no_question": n_skipped_no_question,
    }
    return positives, negatives, q_answer_times, stage_counts


def attach_position_metadata(positives, negatives, q_answer_times):
    """Compute answer_position / n_answers_on_question / answer_age_gap_days /
    answer_year for every candidate, from the FULL per-question answer lists
    (i.e. positions are over all answers in the dump, not just candidates)."""
    print("\nAttaching position metadata...")
    needed_qids = ({r["question_id"] for r in positives}
                   | {r["question_id"] for r in negatives})

    lookup = {}  # answer_id_int -> (position, n_answers, gap_days)
    for qid in needed_qids:
        lst = q_answer_times.get(qid)
        if not lst:
            continue
        lst.sort()  # by (creation_date, answer_id) — ISO dates sort lexically
        first_dt = parse_creation_date(lst[0][0])
        for i, (cd, aid_int) in enumerate(lst):
            dt = parse_creation_date(cd)
            if first_dt is not None and dt is not None:
                gap_days = (dt - first_dt).total_seconds() / 86400.0
            else:
                gap_days = 0.0
            lookup[aid_int] = (i + 1, len(lst), gap_days)

    n_missing = 0
    for r in positives + negatives:
        meta = lookup.get(int(r["answer_id"]))
        if meta is None:  # should not happen; defensive
            n_missing += 1
            meta = (1, 1, 0.0)
        r["answer_position"] = meta[0]
        r["n_answers_on_question"] = meta[1]
        r["answer_age_gap_days"] = round(meta[2], 3)
        r["answer_year"] = int(r["creation_date"][:4]) if r["creation_date"][:4].isdigit() else 0

    if n_missing:
        print(f"  WARNING: {n_missing} candidates missing from position lookup")
    print(f"  Positioned answers for {len(needed_qids)} candidate questions")
    return positives, negatives


def audit_earlier_wins(positives, negatives):
    """Self-audit of the time-order confound, computed BEFORE the
    question-disjoint filter (which removes exactly the mixed-class
    questions these pairs live on): over all (positive, negative) answer
    pairs within the same question, P(positive was posted earlier)."""
    pos_by_q = defaultdict(list)
    neg_by_q = defaultdict(list)
    for r in positives:
        pos_by_q[r["question_id"]].append((r["creation_date"], int(r["answer_id"])))
    for r in negatives:
        neg_by_q[r["question_id"]].append((r["creation_date"], int(r["answer_id"])))

    n_questions_with_both = 0
    n_pairs = 0
    n_pos_earlier = 0
    for qid, pos_list in pos_by_q.items():
        neg_list = neg_by_q.get(qid)
        if not neg_list:
            continue
        n_questions_with_both += 1
        for p in pos_list:
            for n in neg_list:
                n_pairs += 1
                if p < n:
                    n_pos_earlier += 1

    p_earlier = n_pos_earlier / n_pairs if n_pairs else float("nan")
    print(f"\nTime-order self-audit (pre-disjoint, within-question pos/neg pairs):")
    print(f"  Questions with both classes: {n_questions_with_both}")
    print(f"  Pairs: {n_pairs}")
    print(f"  P(positive answer posted earlier): {p_earlier:.3f}")
    return {
        "note": ("Computed on multi-answer questions having both a positive and "
                 "a negative labeled answer, BEFORE the question-disjoint filter "
                 "(which drops exactly these questions) and before matching."),
        "n_questions_with_both_classes": n_questions_with_both,
        "n_pos_neg_pairs": n_pairs,
        "p_positive_posted_earlier": p_earlier,
    }


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
    return positives, negatives, len(overlap)


def primary_tag(item):
    tags = item["question_tags"].split("|")
    return tags[0] if tags and tags[0] else "_none_"


def position_distribution(items):
    """Fraction of items at position bin 1 / 2 / 3+."""
    counts = {label: 0 for label in POSITION_BIN_LABELS}
    for x in items:
        counts[POSITION_BIN_LABELS[position_bin(x["answer_position"]) - 1]] += 1
    total = len(items) or 1
    return {label: round(c / total, 4) for label, c in counts.items()}


def balanced_downsample(positives, negatives, n_len_bins=10, seed=42):
    """Downsample to equal 1/0 per primary tag AND matched on question
    length, answer length, AND answer position.

    Strategy:
    1. Group by primary tag (first tag)
    2. Within each tag, joint-bin by
       (question_len_bin, answer_len_bin, position_bin) — position_bin in
       {1, 2, 3+} — using percentile length edges from the smaller class
    3. Sample from the larger class to match the smaller's joint distribution
    """
    rng = random.Random(seed)

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

        # Determine which is smaller/larger within this tag
        if len(tag_pos) <= len(tag_neg):
            smaller, larger = tag_pos, tag_neg
        else:
            smaller, larger = tag_neg, tag_pos

        # Length bin edges from the smaller class; position bins are fixed
        q_lens = [x["question_len"] for x in smaller]
        a_lens = [x["answer_len"] for x in smaller]
        q_edges = np.percentile(q_lens, np.linspace(0, 100, n_len_bins + 1))
        a_edges = np.percentile(a_lens, np.linspace(0, 100, n_len_bins + 1))
        q_edges[-1] += 1
        a_edges[-1] += 1

        def joint_bin(item):
            qb = int(np.searchsorted(q_edges[1:], item["question_len"]))
            ab = int(np.searchsorted(a_edges[1:], item["answer_len"]))
            pb = position_bin(item["answer_position"])
            # v3.1: 3-year buckets in the matching key — the v3 leakage audit
            # found answer_year single-feature AUC 0.40 (recent answers can't
            # have accumulated score>=3 yet, and site activity drifts), the
            # only structural confound that survived v3 matching.
            yb = item["answer_year"] // 3
            return (qb, ab, pb, yb)

        small_binned = defaultdict(list)
        for x in smaller:
            small_binned[joint_bin(x)].append(x)

        large_binned = defaultdict(list)
        for x in larger:
            large_binned[joint_bin(x)].append(x)

        sampled_larger = []
        sampled_smaller = []
        for bkey, small_items in small_binned.items():
            available = large_binned.get(bkey, [])
            if not available:
                continue
            n_sample = min(len(small_items), len(available))
            sampled_larger.extend(rng.sample(available, n_sample))
            # Subsample the smaller class within the SAME bin so the two
            # classes stay matched bin-by-bin (not just in total count)
            sampled_smaller.extend(rng.sample(small_items, n_sample))

        if len(tag_pos) <= len(tag_neg):
            final_pos.extend(sampled_smaller)
            final_neg.extend(sampled_larger)
        else:
            final_neg.extend(sampled_smaller)
            final_pos.extend(sampled_larger)

    print(f"\nBalanced downsampling (per-tag + length-matched + position-matched):")
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
    print(f"  Position dist — pos: {position_distribution(final_pos)}")
    print(f"                  neg: {position_distribution(final_neg)}")

    return final_pos, final_neg


def has_priority_tag(item):
    tags = set(item["question_tags"].split("|"))
    return bool(tags & PRIORITY_TAGS)


def select_target_rows(positives, negatives, target_rows=100000, seed=42):
    """Downsample to ~target_rows (50/50) keeping 100% of priority-tag rows
    (proof/intuition/soft-question, per build_150k_focused.py), then fill the
    quota with a uniform random sample of the rest. Uniform per-class
    sampling preserves the matched distributions in expectation."""
    rng = random.Random(seed)
    target_per_class = target_rows // 2

    priority_pos = [r for r in positives if has_priority_tag(r)]
    priority_neg = [r for r in negatives if has_priority_tag(r)]
    other_pos = [r for r in positives if not has_priority_tag(r)]
    other_neg = [r for r in negatives if not has_priority_tag(r)]

    print(f"\nTarget-size selection ({target_rows} rows, priority-tag retention):")
    print(f"  Priority: {len(priority_pos)} pos, {len(priority_neg)} neg")
    print(f"  Other:    {len(other_pos)} pos, {len(other_neg)} neg")

    # Balance within priority first (keep as many priority rows as possible)
    priority_min = min(len(priority_pos), len(priority_neg), target_per_class)
    priority_pos_keep = (rng.sample(priority_pos, priority_min)
                         if len(priority_pos) > priority_min else priority_pos)
    priority_neg_keep = (rng.sample(priority_neg, priority_min)
                         if len(priority_neg) > priority_min else priority_neg)

    need_pos = target_per_class - len(priority_pos_keep)
    need_neg = target_per_class - len(priority_neg_keep)

    if need_pos > len(other_pos) or need_neg > len(other_neg):
        print(f"  WARNING: not enough other rows, shrinking target")
        need_both = min(need_pos, len(other_pos), need_neg, len(other_neg))
        need_pos = need_neg = need_both

    other_pos_sample = rng.sample(other_pos, need_pos)
    other_neg_sample = rng.sample(other_neg, need_neg)

    final_pos = priority_pos_keep + other_pos_sample
    final_neg = priority_neg_keep + other_neg_sample

    print(f"  Final positive: {len(final_pos)} ({len(priority_pos_keep)} priority + {len(other_pos_sample)} other)")
    print(f"  Final negative: {len(final_neg)} ({len(priority_neg_keep)} priority + {len(other_neg_sample)} other)")
    print(f"  Priority fraction: {100*(len(priority_pos_keep)+len(priority_neg_keep))/max(1, len(final_pos)+len(final_neg)):.1f}%")
    print(f"  Position dist — pos: {position_distribution(final_pos)}")
    print(f"                  neg: {position_distribution(final_neg)}")

    selection_stats = {
        "priority_kept_per_class": priority_min,
        "other_sampled_pos": len(other_pos_sample),
        "other_sampled_neg": len(other_neg_sample),
        "final_positive": len(final_pos),
        "final_negative": len(final_neg),
    }
    return final_pos, final_neg, selection_stats


def split_of_question(question_id, train_frac=0.8, eval_frac=0.1):
    """Deterministic hash-based group split: all answers to a question land
    in the same split."""
    h = int(hashlib.md5(str(question_id).encode("utf-8")).hexdigest()[:8], 16)
    u = h / 0xFFFFFFFF
    if u < train_frac:
        return "train"
    if u < train_frac + eval_frac:
        return "eval"
    return "test"


def write_output(positives, negatives, questions, out_path, seed=42):
    """Write a single csv.gz with text/judgement + metadata + split column."""
    rng = random.Random(seed)

    rows = []
    for label, items in ((1, positives), (0, negatives)):
        for r in items:
            q = questions[r["question_id"]]
            question_text = (q["title"] + "\n\n" + q["body"]).strip()
            rows.append({
                "text": f"Question: {question_text}\n\nAnswer: {r['answer_text']}",
                "judgement": label,
                "split": split_of_question(r["question_id"]),
                "question_id": r["question_id"],
                "answer_id": r["answer_id"],
                "answer_position": r["answer_position"],
                "n_answers_on_question": r["n_answers_on_question"],
                "answer_age_gap_days": r["answer_age_gap_days"],
                "answer_year": r["answer_year"],
                "score": r["score"],
                "accepted": r["accepted"],
                "primary_tag": primary_tag(r),
                "question_tags": r["question_tags"],
            })
    rng.shuffle(rows)

    fieldnames = ["text", "judgement", "split", "question_id", "answer_id",
                  "answer_position", "n_answers_on_question",
                  "answer_age_gap_days", "answer_year", "score", "accepted",
                  "primary_tag", "question_tags"]

    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_stats = {}
    for split_name in ("train", "eval", "test"):
        split_rows = [r for r in rows if r["split"] == split_name]
        n_pos = sum(1 for r in split_rows if r["judgement"] == 1)
        split_stats[split_name] = {
            "rows": len(split_rows),
            "positive": n_pos,
            "negative": len(split_rows) - n_pos,
        }
        print(f"  {split_name}: {len(split_rows)} rows "
              f"({n_pos} pos, {len(split_rows) - n_pos} neg)")

    print(f"\nWrote {len(rows)} rows → {out_path}")
    return split_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--posts-xml", type=Path, default=EXTRACT_DIR / "Posts.xml",
                        help="Path to the Math.SE Posts.xml dump")
    parser.add_argument("--pos-min-score", type=int, default=3,
                        help="Min score for accepted answers to count as positive")
    parser.add_argument("--neg-max-score", type=int, default=0,
                        help="Max score for answers to count as negative")
    parser.add_argument("--min-chars", type=int, default=50,
                        help="Min stripped answer length (stub filter)")
    parser.add_argument("--target-rows", type=int, default=100000,
                        help="Target final balanced row count (50/50)")
    parser.add_argument("--n-len-bins", type=int, default=10,
                        help="Number of percentile length bins for matching")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=DATA_DIR / "math_se_v3_position_matched.csv.gz")
    parser.add_argument("--manifest", type=Path,
                        default=DATA_DIR / "math_se_v3_position_matched.manifest.json")
    args = parser.parse_args()

    manifest = {"args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
                "build_date": datetime.now().isoformat(timespec="seconds")}

    # Pass 1: questions
    questions, accepted_ids = parse_questions(args.posts_xml)
    manifest["n_questions"] = len(questions)
    manifest["n_accepted_answer_ids"] = len(accepted_ids)

    # Pass 2: answers (full timing metadata + labeled candidates)
    positives, negatives, q_answer_times, stage_counts = parse_answers(
        args.posts_xml, questions, accepted_ids,
        pos_min_score=args.pos_min_score,
        neg_max_score=args.neg_max_score,
        min_answer_len=args.min_chars,
    )
    manifest["labeling"] = stage_counts

    # Positions over ALL answers (before any filtering)
    positives, negatives = attach_position_metadata(positives, negatives, q_answer_times)
    del q_answer_times

    # Self-audit of the time-order confound (must run before the disjoint
    # filter, which removes the mixed-class questions the pairs live on)
    manifest["time_order_audit"] = audit_earlier_wins(positives, negatives)
    manifest["position_dist_before_matching"] = {
        "positive": position_distribution(positives),
        "negative": position_distribution(negatives),
    }

    # Question-disjoint filter
    positives, negatives, n_overlap = make_question_disjoint(positives, negatives)
    manifest["question_disjoint"] = {
        "overlap_questions_dropped": n_overlap,
        "positive_after": len(positives),
        "negative_after": len(negatives),
    }

    # Per-tag + (question_len x answer_len x position) matched downsampling
    positives, negatives = balanced_downsample(
        positives, negatives, n_len_bins=args.n_len_bins, seed=args.seed)
    manifest["after_matching"] = {
        "positive": len(positives),
        "negative": len(negatives),
        "position_dist": {
            "positive": position_distribution(positives),
            "negative": position_distribution(negatives),
        },
    }

    # Priority-tag retention down to target size
    positives, negatives, selection_stats = select_target_rows(
        positives, negatives, target_rows=args.target_rows, seed=args.seed)
    manifest["target_selection"] = selection_stats
    manifest["position_dist_final"] = {
        "positive": position_distribution(positives),
        "negative": position_distribution(negatives),
    }

    # Single output file with hash-based group split column
    print(f"\nWriting output (group split by question_id, 80/10/10)...")
    manifest["splits"] = write_output(positives, negatives, questions,
                                      args.out, seed=args.seed)

    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest → {args.manifest}")
    print("\nDone!")


if __name__ == "__main__":
    main()

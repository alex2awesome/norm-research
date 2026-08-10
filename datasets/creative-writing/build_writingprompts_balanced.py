#!/usr/bin/env python3
"""
Build length-balanced WritingPrompts modeling dataset.
Uses the full 993K deduped corpus, balances 1s/0s within length buckets,
targets 100K total rows.
"""
import csv
import gzip
import json
import random
import statistics
from collections import defaultdict

random.seed(42)
csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing"
COMMENTS = BASE + "/writingprompts_comments.jsonl.gz"
SUBMISSIONS = BASE + "/writingprompts_submissions.jsonl.gz"
OUTPUT = BASE + "/writingprompts_modeling.csv.gz"
OUTPUT_TOPICS = BASE + "/writingprompts_modeling_with_topics.csv.gz"

SCORE_THRESHOLD = 10
TARGET_PER_CLASS = 50000
# Length buckets (char boundaries for story body)
BUCKET_EDGES = [0, 500, 1000, 2000, 3000, 5000, 10000, 999999]

def get_bucket(length):
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= length < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_EDGES) - 2

# Step 1: Load prompts
print("Loading submissions...")
prompts = {}
with gzip.open(SUBMISSIONS, "rt") as f:
    for line in f:
        rec = json.loads(line)
        prompts[rec["id"]] = rec["title"]
print("  Loaded %d prompts" % len(prompts))

# Step 2: Load all stories, join with prompts, assign labels and buckets
print("Loading stories...")
bucket_groups = defaultdict(lambda: {0: [], 1: []})
total = 0
no_prompt = 0
seen = set()

with gzip.open(COMMENTS, "rt") as f:
    for line in f:
        rec = json.loads(line)
        prompt_title = prompts.get(rec["link_id"], "")
        if not prompt_title:
            no_prompt += 1
            continue

        body = rec["body"]
        score = rec["score"]
        text = "PROMPT: " + prompt_title + "\n\nSTORY: " + body

        # Dedup
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)

        label = 1 if score >= SCORE_THRESHOLD else 0
        bucket = get_bucket(len(body))
        bucket_groups[bucket][label].append(text)
        total += 1

        if total % 200000 == 0:
            print("  loaded %d..." % total)

print("  Total unique stories: %d" % total)

# Step 3: Compute per-bucket balance
# Strategy: balance within each bucket, then allocate across buckets
# proportional to the minority class count
print("\nPer-bucket distribution:")
bucket_mins = {}
total_available = 0
for b in sorted(bucket_groups.keys()):
    lo = BUCKET_EDGES[b]
    hi = BUCKET_EDGES[b + 1] if b + 1 < len(BUCKET_EDGES) else 999999
    n_pos = len(bucket_groups[b][1])
    n_neg = len(bucket_groups[b][0])
    n_min = min(n_pos, n_neg)
    bucket_mins[b] = n_min
    total_available += n_min * 2
    print("  Bucket %d (%d-%d chars): pos=%d, neg=%d, balanced=%d per class" % (
        b, lo, hi, n_pos, n_neg, n_min))

print("  Total available after per-bucket balance: %d" % total_available)

# Step 4: Allocate proportionally to hit target
# If total_available >= 100K, scale down proportionally
# If < 100K, take everything
if total_available >= TARGET_PER_CLASS * 2:
    scale = TARGET_PER_CLASS / sum(bucket_mins.values())
else:
    scale = 1.0

print("\nAllocating (scale=%.3f):" % scale)
balanced = []
for b in sorted(bucket_groups.keys()):
    lo = BUCKET_EDGES[b]
    hi = BUCKET_EDGES[b + 1] if b + 1 < len(BUCKET_EDGES) else 999999
    n_take = int(bucket_mins[b] * scale)
    if n_take == 0:
        continue

    pos = bucket_groups[b][1]
    neg = bucket_groups[b][0]
    random.shuffle(pos)
    random.shuffle(neg)
    balanced.extend([(t, 1) for t in pos[:n_take]])
    balanced.extend([(t, 0) for t in neg[:n_take]])
    print("  Bucket %d (%d-%d): %d per class" % (b, lo, hi, n_take))

random.shuffle(balanced)
print("\nTotal: %d rows (%d per class)" % (len(balanced), len(balanced) // 2))

# Step 5: Write
print("Writing to %s..." % OUTPUT)
with gzip.open(OUTPUT, "wt", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "judgement"])
    writer.writeheader()
    for text, j in balanced:
        writer.writerow({"text": text, "judgement": j})

# Step 6: Verify length balance
print("\nVerification:")
lens = {0: [], 1: []}
for text, j in balanced:
    parts = text.split("\n\nSTORY: ", 1)
    story = parts[1] if len(parts) > 1 else ""
    lens[j].append(len(story))

for j in [0, 1]:
    print("  Label %d: mean=%d, median=%d, stdev=%d" % (
        j, statistics.mean(lens[j]), statistics.median(lens[j]), statistics.stdev(lens[j])))
ratio = statistics.mean(lens[1]) / statistics.mean(lens[0])
print("  Length ratio (label1/label0): %.3f" % ratio)

print("\nDone!")

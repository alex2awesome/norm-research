#!/usr/bin/env python3
"""Build modeling dataset from downloaded WritingPrompts data."""
import csv
import gzip
import json
import random
from collections import defaultdict

random.seed(42)
csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing"
OUT = BASE + "/writingprompts_modeling.csv.gz"

# Step 1: Load submissions (prompts)
print("Loading submissions...")
prompts = {}
with gzip.open(BASE + "/writingprompts_submissions.jsonl.gz", "rt") as f:
    for line in f:
        rec = json.loads(line)
        prompts[rec["id"]] = rec["title"]
print("  Loaded {:,} prompts".format(len(prompts)))

# Step 2: Load comments, join with prompts
print("Loading and joining stories...")
rows = []
no_prompt = 0
with gzip.open(BASE + "/writingprompts_comments.jsonl.gz", "rt") as f:
    for line in f:
        rec = json.loads(line)
        prompt_title = prompts.get(rec["link_id"], "")
        if not prompt_title:
            no_prompt += 1
            continue

        score = rec["score"]
        body = rec["body"]
        text = "PROMPT: " + prompt_title + "\n\nSTORY: " + body

        rows.append({"text": text, "score": score})
        if len(rows) % 200000 == 0:
            print("  processed {:,}...".format(len(rows)))

print("  Total with prompts: {:,} (no prompt match: {:,})".format(len(rows), no_prompt))

# Step 3: Exact dedup on text
print("Exact dedup...")
seen = set()
deduped = []
for r in rows:
    h = hash(r["text"])
    if h not in seen:
        seen.add(h)
        deduped.append(r)
print("  After dedup: {:,} (removed {:,})".format(len(deduped), len(rows) - len(deduped)))

# Step 4: Label — score >= 10 = good (1), < 10 = bad (0)
pos = [r for r in deduped if r["score"] >= 10]
neg = [r for r in deduped if r["score"] < 10]
print("  Score >= 10 (positive): {:,}".format(len(pos)))
print("  Score < 10 (negative): {:,}".format(len(neg)))

# Step 5: Balance to 50K per class = 100K total
target_per_class = 50000
random.shuffle(pos)
random.shuffle(neg)
pos = pos[:target_per_class]
neg = neg[:target_per_class]
print("  Balanced: {:,} pos, {:,} neg".format(len(pos), len(neg)))

# Combine and shuffle
combined = [(r["text"], 1) for r in pos] + [(r["text"], 0) for r in neg]
random.shuffle(combined)

# Step 6: Write
print("Writing {:,} rows to {}...".format(len(combined), OUT))
with gzip.open(OUT, "wt", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "judgement"])
    writer.writeheader()
    for text, j in combined:
        writer.writerow({"text": text, "judgement": j})

print("Done!")

#!/usr/bin/env python3
"""Inspect new WritingPrompts dataset for biases and quality issues."""
import csv
import gzip
import random
import statistics
from collections import Counter

csv.field_size_limit(2**31 - 1)
random.seed(42)

INPUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/writingprompts_modeling.csv.gz"

lengths = {0: [], 1: []}
story_lens = {0: [], 1: []}
prompt_lens = {0: [], 1: []}
samples = {0: [], 1: []}
short_stories = {0: 0, 1: 0}
has_deleted = {0: 0, 1: 0}
has_edit = {0: 0, 1: 0}
total = {0: 0, 1: 0}

with gzip.open(INPUT, "rt") as f:
    for row in csv.DictReader(f):
        j = int(row["judgement"])
        text = row["text"]
        total[j] += 1
        lengths[j].append(len(text))

        parts = text.split("\n\nSTORY: ", 1)
        prompt = parts[0].replace("PROMPT: ", "") if parts else ""
        story = parts[1] if len(parts) > 1 else ""

        prompt_lens[j].append(len(prompt))
        story_lens[j].append(len(story))

        if len(story) < 300:
            short_stories[j] += 1
        if "[deleted]" in story or "[removed]" in story:
            has_deleted[j] += 1
        if "EDIT:" in story or "Edit:" in story or "edit:" in story:
            has_edit[j] += 1

        if len(samples[j]) < 3 and random.random() < 0.001:
            samples[j].append(text[:500])

print("=== BASIC STATS ===")
print("Label 0 (low score): %d" % total[0])
print("Label 1 (high score): %d" % total[1])

print("\n=== TOTAL TEXT LENGTH ===")
for j in [0, 1]:
    print("  Label %d: mean=%d, median=%d, stdev=%d, min=%d, max=%d" % (
        j, statistics.mean(lengths[j]), statistics.median(lengths[j]),
        statistics.stdev(lengths[j]), min(lengths[j]), max(lengths[j])))

print("\n=== STORY LENGTH (without prompt) ===")
for j in [0, 1]:
    print("  Label %d: mean=%d, median=%d, stdev=%d, min=%d, max=%d" % (
        j, statistics.mean(story_lens[j]), statistics.median(story_lens[j]),
        statistics.stdev(story_lens[j]), min(story_lens[j]), max(story_lens[j])))

ratio = statistics.mean(story_lens[1]) / statistics.mean(story_lens[0])
print("\n  ** Length ratio (label1/label0): %.2fx **" % ratio)

print("\n=== PROMPT LENGTH ===")
for j in [0, 1]:
    print("  Label %d: mean=%d, median=%d" % (
        j, statistics.mean(prompt_lens[j]), statistics.median(prompt_lens[j])))

print("\n=== STORY LENGTH DISTRIBUTION ===")
for j in [0, 1]:
    n = len(story_lens[j])
    buckets = Counter()
    for l in story_lens[j]:
        if l < 500: buckets["<500"] = buckets.get("<500", 0) + 1
        elif l < 1000: buckets["500-1K"] = buckets.get("500-1K", 0) + 1
        elif l < 3000: buckets["1K-3K"] = buckets.get("1K-3K", 0) + 1
        elif l < 5000: buckets["3K-5K"] = buckets.get("3K-5K", 0) + 1
        elif l < 10000: buckets["5K-10K"] = buckets.get("5K-10K", 0) + 1
        else: buckets["10K+"] = buckets.get("10K+", 0) + 1
    print("  Label %d:" % j)
    for k in ["<500", "500-1K", "1K-3K", "3K-5K", "5K-10K", "10K+"]:
        print("    %s: %d (%.1f%%)" % (k, buckets.get(k, 0), buckets.get(k, 0) / n * 100))

print("\n=== QUALITY CHECKS ===")
for j in [0, 1]:
    n = total[j]
    print("  Label %d:" % j)
    print("    Short stories (<300 chars): %d (%.1f%%)" % (short_stories[j], short_stories[j] / n * 100))
    print("    Contains [deleted]/[removed]: %d (%.1f%%)" % (has_deleted[j], has_deleted[j] / n * 100))
    print("    Contains EDIT/Edit: %d (%.1f%%)" % (has_edit[j], has_edit[j] / n * 100))

print("\n=== SAMPLES ===")
for j in [0, 1]:
    for s in samples[j]:
        print("--- label=%d ---" % j)
        print(s)
        print()

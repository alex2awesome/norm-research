#!/usr/bin/env python3
"""
Balance homepage newsworthiness dataset by topic using LDA.
1. Extract headlines from balanced dataset
2. Run TF-IDF + LDA with 50 topics
3. Hard-assign each document to top topic
4. Balance 1s/0s within each topic
5. Write new CSV
"""
import csv
import gzip
import random
from collections import defaultdict

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(2**31 - 1)

INPUT = "homepage_newsworthiness_balanced.csv.gz"
OUTPUT = "homepage_newsworthiness_topic_balanced.csv.gz"
N_TOPICS = 50
SEED = 42

rng = random.Random(SEED)

# Step 1: Load data and extract headlines
print("Loading data...")
rows = []
headlines = []
with gzip.open(INPUT, "rt") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
        # Extract just the headline (before CONTEXT)
        text = row["text"]
        parts = text.split("\n\nCONTEXT: ")
        headline = parts[0].replace("HEADLINE: ", "")
        headlines.append(headline)

print(f"Loaded {len(rows):,} rows")

# Step 2: TF-IDF
print("Running TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000, stop_words="english", max_df=0.95, min_df=5)
X = tfidf.fit_transform(headlines)
print(f"TF-IDF matrix: {X.shape}")

# Step 3: LDA with 50 topics
print(f"Running LDA with {N_TOPICS} topics...")
lda = LatentDirichletAllocation(
    n_components=N_TOPICS, random_state=SEED, max_iter=20, n_jobs=-1
)
topic_dist = lda.fit_transform(X)
print("LDA done.")

# Print top words per topic for sanity check
feature_names = tfidf.get_feature_names_out()
for i in range(min(10, N_TOPICS)):
    top_words = [feature_names[j] for j in lda.components_[i].argsort()[-8:][::-1]]
    print(f"  Topic {i}: {', '.join(top_words)}")
print(f"  ... ({N_TOPICS - 10} more topics)")

# Step 4: Hard-assign each document to top topic
topic_assignments = topic_dist.argmax(axis=1)

# Step 5: Balance within each topic
topic_groups = defaultdict(lambda: {0: [], 1: []})
for i, row in enumerate(rows):
    topic = int(topic_assignments[i])
    label = int(row["judgement"])
    topic_groups[topic][label].append(row)

balanced = []
print(f"\nBalancing per topic:")
for topic in sorted(topic_groups):
    pos = topic_groups[topic][1]
    neg = topic_groups[topic][0]
    n = min(len(pos), len(neg))
    if n == 0:
        print(f"  Topic {topic}: SKIPPED (pos={len(pos)}, neg={len(neg)})")
        continue
    rng.shuffle(pos)
    rng.shuffle(neg)
    balanced.extend(pos[:n])
    balanced.extend(neg[:n])
    ratio_before = len(pos) / (len(pos) + len(neg)) * 100
    print(f"  Topic {topic}: {n} per class (was {len(pos)}/{len(neg)}, {ratio_before:.0f}% pos)")

rng.shuffle(balanced)

# Step 6: Write
print(f"\nWriting {len(balanced):,} rows to {OUTPUT}")
with gzip.open(OUTPUT, "wt", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "judgement"])
    writer.writeheader()
    writer.writerows(balanced)

total_pos = sum(1 for r in balanced if int(r["judgement"]) == 1)
print(f"Label 1: {total_pos:,} ({total_pos/len(balanced)*100:.1f}%)")
print(f"Label 0: {len(balanced)-total_pos:,} ({(len(balanced)-total_pos)/len(balanced)*100:.1f}%)")
print("Done!")

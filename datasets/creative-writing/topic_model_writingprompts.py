#!/usr/bin/env python3
"""Run LDA topic modeling on the new WritingPrompts dataset."""
import csv
import gzip
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(2**31 - 1)
INPUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/writingprompts_modeling.csv.gz"
OUTPUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/writingprompts_modeling_with_topics.csv.gz"

print("Loading data...")
rows = []
texts = []
with gzip.open(INPUT, "rt") as f:
    for row in csv.DictReader(f):
        rows.append(row)
        texts.append(row["text"])
print("  Loaded %d rows" % len(rows))

print("TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000, stop_words="english", max_df=0.95, min_df=5)
X = tfidf.fit_transform(texts)

print("LDA k=100...")
lda = LatentDirichletAllocation(n_components=100, random_state=42, max_iter=20, n_jobs=-1)
topic_dist = lda.fit_transform(X)
topics = topic_dist.argmax(axis=1)

print("  Unique topics used: %d/100" % len(set(topics)))

# Print top words for first 5 topics
feature_names = tfidf.get_feature_names_out()
for i in range(min(5, 100)):
    top_words = [feature_names[j] for j in lda.components_[i].argsort()[-6:][::-1]]
    print("  Topic %d: %s" % (i, ", ".join(top_words)))

print("Writing output...")
with gzip.open(OUTPUT, "wt", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "judgement", "topic"])
    writer.writeheader()
    for i, row in enumerate(rows):
        writer.writerow({"text": row["text"], "judgement": row["judgement"], "topic": int(topics[i])})

print("Done! Saved to %s" % OUTPUT)

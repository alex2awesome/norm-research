#!/usr/bin/env python3
"""Audit creative writing (writingprompts) dataset for leakage:

1. Prompt-level: same prompt in train and val with different stories.
2. Story near-duplicates (MinHash should have caught these; sanity check).
3. Within-prompt label consistency (signal vs noise per prompt).
4. Length / surface features by label (post-balancing residuals).
5. Author leakage proxy: shared n-grams or stylistic fingerprints by label.
"""
import csv
import gzip
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict

csv.field_size_limit(2**31 - 1)

INPUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/writingprompts_modeling.csv.gz"


def split_text(t):
    parts = t.split("\n\nSTORY: ", 1)
    p = parts[0].replace("PROMPT: ", "").strip()
    s = parts[1].strip() if len(parts) > 1 else ""
    return p, s


def main():
    rows = []
    with gzip.open(INPUT, "rt") as f:
        for r in csv.DictReader(f):
            p, s = split_text(r["text"])
            rows.append({"prompt": p, "story": s, "label": int(r["judgement"]), "text": r["text"]})

    n = len(rows)
    print("Total rows:", n)
    print()

    # 1. Prompt-level distribution
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r["label"])

    unique_prompts = len(by_prompt)
    multi = {p: ls for p, ls in by_prompt.items() if len(ls) > 1}
    print("--- 1. Prompt-level structure ---")
    print(f"Unique prompts: {unique_prompts}")
    print(f"Multi-story prompts: {len(multi)}")
    rows_in_multi = sum(len(ls) for ls in multi.values())
    print(f"Rows in multi-story groups: {rows_in_multi} ({rows_in_multi/n*100:.1f}%)")
    # Label entropy within multi-story prompts
    all_one = sum(1 for ls in multi.values() if all(l == 1 for l in ls))
    all_zero = sum(1 for ls in multi.values() if all(l == 0 for l in ls))
    mixed = len(multi) - all_one - all_zero
    print(f"  all-pos: {all_one}, all-neg: {all_zero}, mixed: {mixed}")
    print()

    # 2. Within-prompt label correlation: how predictable is label from prompt?
    # MI = H(L) - H(L|prompt)
    # H(L) for multi-story rows
    multi_rows = [(p, l) for p, ls in multi.items() for l in ls]
    n_multi = len(multi_rows)
    if n_multi > 0:
        pos_rate = sum(l for _, l in multi_rows) / n_multi
        if 0 < pos_rate < 1:
            h_l = -pos_rate * math.log(pos_rate) - (1 - pos_rate) * math.log(1 - pos_rate)
        else:
            h_l = 0
        # H(L|prompt)
        h_l_given_p = 0
        for p, ls in multi.items():
            w = len(ls) / n_multi
            r = sum(ls) / len(ls)
            if 0 < r < 1:
                hp = -r * math.log(r) - (1 - r) * math.log(1 - r)
            else:
                hp = 0
            h_l_given_p += w * hp
        mi = h_l - h_l_given_p
        print("--- 2. Mutual information (multi-prompt rows only) ---")
        print(f"H(label) = {h_l:.4f} nats")
        print(f"H(label|prompt) = {h_l_given_p:.4f} nats")
        print(f"MI(label, prompt) = {mi:.4f} nats")
        print(f"Prompt determines label fraction (MI/H(L)): {mi/h_l*100:.1f}%")
        print()

    # 3. Story near-duplicates (full-text hash and 9-gram)
    print("--- 3. Surface duplicates ---")
    story_hash = Counter()
    for r in rows:
        story_hash[hashlib.md5(r["story"].encode("utf-8")).hexdigest()] += 1
    dupes = sum(c - 1 for c in story_hash.values() if c > 1)
    print(f"Exact story duplicates: {dupes} (out of {n})")
    # 9-gram (char)
    def gram_shingles(s, k=9):
        s = re.sub(r"\s+", " ", s).strip()
        return set(s[i:i + k] for i in range(0, max(1, len(s) - k + 1)))
    # Sample 1K rows; for each, find max-overlap neighbor via inverted index on shingles
    import random
    rng = random.Random(0)
    sample = rng.sample(rows, min(1000, n))
    shingle_lists = [gram_shingles(r["story"][:1500]) for r in sample]
    # Build inverted index: shingle -> list of doc indices that contain it
    inv = defaultdict(list)
    for i, sh in enumerate(shingle_lists):
        for s in sh:
            inv[s].append(i)
    high_overlap = 0
    seen_pairs = set()
    for i, sh in enumerate(shingle_lists):
        if not sh:
            continue
        candidates = Counter()
        for s in sh:
            for j in inv[s]:
                if j > i:
                    candidates[j] += 1
        for j, common in candidates.most_common(3):
            sj = shingle_lists[j]
            if not sj:
                continue
            jac = common / max(1, len(sh | sj))
            if jac > 0.5 and (i, j) not in seen_pairs:
                seen_pairs.add((i, j))
                high_overlap += 1
                if high_overlap <= 3:
                    print(f"  Pair ({i},{j}) Jaccard={jac:.2f}: ", sample[i]["story"][:80], "| vs |", sample[j]["story"][:80])
    print(f"1K-sample pairs with shingle Jaccard>0.5: {high_overlap}")
    print()

    # 4. Length features by label
    print("--- 4. Surface features by label ---")
    for label in (0, 1):
        lens = [len(r["story"]) for r in rows if r["label"] == label]
        words = [len(r["story"].split()) for r in rows if r["label"] == label]
        print(f"  label={label}: n={len(lens)}, char mean={statistics.mean(lens):.0f} med={statistics.median(lens):.0f}, word mean={statistics.mean(words):.0f}")
    print()

    # 5. Distinctive top n-grams by label (post-balancing residual features)
    print("--- 5. Top distinguishing trigrams by label (top 8 by lift) ---")
    c0 = Counter()
    c1 = Counter()
    for r in rows:
        toks = re.findall(r"\b[a-z]+\b", r["story"].lower())
        for k in range(len(toks) - 2):
            tg = " ".join(toks[k:k + 3])
            if r["label"] == 1:
                c1[tg] += 1
            else:
                c0[tg] += 1
    # Lift = (p_pos - p_neg) / (p_pos + p_neg)
    total1 = sum(c1.values())
    total0 = sum(c0.values())
    candidates = [g for g, c in c1.items() if c > 200 and c0.get(g, 0) > 200]
    scored = []
    for g in candidates:
        p1 = c1[g] / total1
        p0 = c0.get(g, 1) / total0
        if p0 == 0:
            continue
        lift = (p1 - p0) / (p1 + p0)
        scored.append((lift, g, c1[g], c0.get(g, 0)))
    scored.sort()
    print("  Top neg-leaning trigrams:")
    for s, g, n1, n0 in scored[:8]:
        print(f"    {s:+.3f} '{g}' pos={n1} neg={n0}")
    print("  Top pos-leaning trigrams:")
    for s, g, n1, n0 in scored[-8:][::-1]:
        print(f"    {s:+.3f} '{g}' pos={n1} neg={n0}")
    print()

    # 6. [deleted]/[removed] artifacts
    print("--- 6. [deleted]/[removed] artifacts (Reddit moderator residue) ---")
    for label in (0, 1):
        deleted = sum(1 for r in rows if r["label"] == label and "[deleted]" in r["story"])
        removed = sum(1 for r in rows if r["label"] == label and "[removed]" in r["story"])
        edit = sum(1 for r in rows if r["label"] == label and re.search(r"\b[Ee][Dd][Ii][Tt]:", r["story"]))
        print(f"  label={label}: [deleted]={deleted} [removed]={removed} EDIT:={edit}")


if __name__ == "__main__":
    main()

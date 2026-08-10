#!/usr/bin/env python3
"""Audit notice_and_comment_len_balanced.csv.gz for leakage.

Checks:
 1. Surface length features by label.
 2. Exact text duplicates and within-label duplicate rate.
 3. Top distinguishing trigrams by label (lift) → look for templated boilerplate.
 4. Specific N&C boilerplate patterns ("the agency should", "EPA should", agency
    name leak, "we agree", "the proposed rule", etc.).
 5. Near-duplicates via 9-char shingle Jaccard on 1K sample.
 6. Cross-reference: are there per-agency files? If so, check whether agency
    itself predicts label (forbidden confound if so).
"""
import csv
import gzip
import hashlib
import math
import os
import re
import statistics
from collections import Counter, defaultdict

csv.field_size_limit(2**31 - 1)

INPUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/notice_and_comment_len_balanced.csv.gz"
AGENCIES_DIR = "/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/agencies"


def main():
    rows = []
    with gzip.open(INPUT, "rt") as f:
        for r in csv.DictReader(f):
            rows.append({"text": r["text"], "label": int(r["judgement"])})
    n = len(rows)
    print(f"Total rows: {n}")
    pos = sum(r["label"] for r in rows)
    print(f"Positives: {pos} ({pos/n*100:.1f}%)")
    print()

    print("--- 1. Length by label ---")
    for label in (0, 1):
        lens = [len(r["text"]) for r in rows if r["label"] == label]
        words = [len(r["text"].split()) for r in rows if r["label"] == label]
        print(f"  label={label}: n={len(lens)}, char mean={statistics.mean(lens):.0f} "
              f"med={statistics.median(lens):.0f}, word mean={statistics.mean(words):.0f}, "
              f"word med={statistics.median(words):.0f}")
    print()

    print("--- 2. Exact text duplicates ---")
    text_hash = Counter()
    for r in rows:
        text_hash[hashlib.md5(r["text"].encode("utf-8")).hexdigest()] += 1
    dupes = sum(c - 1 for c in text_hash.values() if c > 1)
    print(f"Exact dup rows: {dupes} ({dupes/n*100:.2f}%)")
    # Same-text-different-label?
    by_text = defaultdict(list)
    for r in rows:
        by_text[r["text"][:200]].append(r["label"])
    conflicts = sum(1 for ls in by_text.values() if len(set(ls)) > 1 and len(ls) > 1)
    print(f"Same-prefix-different-label: {conflicts}")
    print()

    print("--- 3. Top distinguishing trigrams (lift) ---")
    c0, c1 = Counter(), Counter()
    for r in rows:
        toks = re.findall(r"\b[a-z]+\b", r["text"].lower())
        for k in range(len(toks) - 2):
            tg = " ".join(toks[k:k + 3])
            if r["label"] == 1:
                c1[tg] += 1
            else:
                c0[tg] += 1
    total1, total0 = sum(c1.values()), sum(c0.values())
    candidates = [g for g, c in c1.items() if c > 100 and c0.get(g, 0) > 100]
    scored = []
    for g in candidates:
        p1 = c1[g] / total1
        p0 = c0[g] / total0
        if p0 == 0: continue
        lift = (p1 - p0) / (p1 + p0)
        scored.append((lift, g, c1[g], c0.get(g, 0)))
    scored.sort()
    print("  Top neg-leaning trigrams (label=0 dominant):")
    for s, g, n1, n0 in scored[:10]:
        print(f"    {s:+.3f} '{g}' pos={n1} neg={n0}")
    print("  Top pos-leaning trigrams (label=1 dominant):")
    for s, g, n1, n0 in scored[-10:][::-1]:
        print(f"    {s:+.3f} '{g}' pos={n1} neg={n0}")
    print()

    print("--- 4. Specific N&C boilerplate patterns ---")
    patterns = {
        "agency_should":   re.compile(r"\bthe agency should\b", re.IGNORECASE),
        "epa_should":      re.compile(r"\bEPA should\b"),
        "we_agree":        re.compile(r"\bwe agree\b", re.IGNORECASE),
        "we_disagree":     re.compile(r"\bwe disagree\b", re.IGNORECASE),
        "we_strongly":     re.compile(r"\bwe strongly\b", re.IGNORECASE),
        "the_proposed":    re.compile(r"\bthe proposed rule\b", re.IGNORECASE),
        "the_commenter":   re.compile(r"\bthe commenter\b", re.IGNORECASE),
        "the_final_rule":  re.compile(r"\bthe final rule\b", re.IGNORECASE),
        "agency_response": re.compile(r"\bagency response\b", re.IGNORECASE),
        "thank_you":       re.compile(r"\bthank you\b", re.IGNORECASE),
        "express_support": re.compile(r"\bexpress(ed)? support\b", re.IGNORECASE),
        "express_oppos":   re.compile(r"\boppose[ds]?\b", re.IGNORECASE),
        "should_not":      re.compile(r"\bshould not\b", re.IGNORECASE),
    }
    print(f"  {'pattern':20s} | {'label=1':>10s} | {'label=0':>10s} | {'lift':>6s}")
    for name, pat in patterns.items():
        n1 = sum(1 for r in rows if r["label"] == 1 and pat.search(r["text"]))
        n0 = sum(1 for r in rows if r["label"] == 0 and pat.search(r["text"]))
        n_pos = pos
        n_neg = n - pos
        if n_pos and n_neg:
            p1 = n1 / n_pos
            p0 = n0 / n_neg
            lift = (p1 - p0) / (p1 + p0) if (p1 + p0) else 0
            print(f"  {name:20s} | {n1:>10d} | {n0:>10d} | {lift:+.3f}")
    print()

    print("--- 5. Near-duplicate via 9-char shingle Jaccard (1K sample) ---")
    import random
    rng = random.Random(0)
    sample = rng.sample(rows, min(1000, n))
    def shingles(s, k=9):
        s = re.sub(r"\s+", " ", s).strip()
        return set(s[i:i + k] for i in range(0, max(1, len(s) - k + 1)))
    sl = [shingles(r["text"][:1500]) for r in sample]
    inv = defaultdict(list)
    for i, sh in enumerate(sl):
        for s in sh:
            inv[s].append(i)
    high_overlap = 0
    cross_label_overlap = 0
    for i, sh in enumerate(sl):
        if not sh: continue
        cand = Counter()
        for s in sh:
            for j in inv[s]:
                if j > i:
                    cand[j] += 1
        for j, common in cand.most_common(5):
            sj = sl[j]
            if not sj: continue
            jac = common / max(1, len(sh | sj))
            if jac > 0.5:
                high_overlap += 1
                if sample[i]["label"] != sample[j]["label"]:
                    cross_label_overlap += 1
    print(f"  Pairs with Jaccard>0.5: {high_overlap}, of which cross-label: {cross_label_overlap}")
    print()

    print("--- 6. Agency-level confound check ---")
    if not os.path.isdir(AGENCIES_DIR):
        print(f"  No agencies dir at {AGENCIES_DIR}")
    else:
        agency_stats = []
        for ag in sorted(os.listdir(AGENCIES_DIR)):
            d = os.path.join(AGENCIES_DIR, ag)
            if not os.path.isdir(d):
                continue
            # Look for csv inside
            csvs = [f for f in os.listdir(d) if f.endswith(".csv") or f.endswith(".csv.gz")]
            if not csvs:
                continue
            # Pick the first csv, read text+judgement
            path = os.path.join(d, csvs[0])
            try:
                op = gzip.open if path.endswith(".gz") else open
                with op(path, "rt") as f:
                    rdr = csv.DictReader(f)
                    counts = Counter()
                    for r in rdr:
                        j = r.get("judgement")
                        if j is None: continue
                        counts[int(j)] += 1
                if counts:
                    tot = sum(counts.values())
                    pos_rate = counts.get(1, 0) / tot
                    agency_stats.append((ag, tot, pos_rate))
            except Exception as e:
                print(f"  [warn] couldn't read {path}: {e}")
        agency_stats.sort(key=lambda x: -x[1])
        print(f"  {'agency':10s} | {'n_rows':>8s} | {'pos_rate':>9s}")
        for ag, t, p in agency_stats[:30]:
            print(f"  {ag:10s} | {t:>8d} | {p:>8.1%}")
        # Compute weighted mean and variance
        if agency_stats:
            tot = sum(t for _, t, _ in agency_stats)
            mean = sum(t * p for _, t, p in agency_stats) / tot
            var = sum(t * (p - mean) ** 2 for _, t, p in agency_stats) / tot
            print(f"  weighted pos-rate mean: {mean:.3f}, weighted var: {var:.5f}, std: {math.sqrt(var):.3f}")
            # MI(agency, label) on per-agency counts
            h_l = -mean * math.log(mean) - (1 - mean) * math.log(1 - mean) if 0 < mean < 1 else 0
            h_l_given_ag = 0
            for ag, t, p in agency_stats:
                w = t / tot
                if 0 < p < 1:
                    h_l_given_ag += w * (-p * math.log(p) - (1 - p) * math.log(1 - p))
            mi = h_l - h_l_given_ag
            print(f"  H(L) = {h_l:.4f}, H(L|agency) = {h_l_given_ag:.4f}, MI = {mi:.4f} ({mi/h_l*100:.1f}% of H(L))")


if __name__ == "__main__":
    main()
